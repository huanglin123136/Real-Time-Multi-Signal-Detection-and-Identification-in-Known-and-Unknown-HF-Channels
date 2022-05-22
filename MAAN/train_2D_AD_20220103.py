# -*- coding: utf-8 -*-
import os
import torch
import time
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torch.nn.functional as F

from data import *
from data.SignalDetection_20210924 import SignalDetectionv2

from layers.modules import MultiBoxLoss, MultiBoxLossv2
from Resnet_2D_AD_20220103 import build_SSD,KL_div
from nms_ground import decode_result,match_result,compute_p_r ,decode_result_all_class
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch:
        (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    fft_1 = []
    fft_2 = []
    fft_3 = []
    fft_4 = []
    fft_5 = []
    for sample in batch:
        fft_1.append(sample[0])
        fft_2.append(sample[1])
        fft_3.append(sample[2])
        fft_4.append(sample[3])
        fft_5.append(sample[4])
        targets.append(torch.FloatTensor(sample[5]))
    return torch.stack(fft_1, 0), torch.stack(fft_2, 0),torch.stack(fft_3, 0),\
           torch.stack(fft_4, 0), torch.stack(fft_5, 0),targets

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

dic ={'AM': 1, 'SSB': 2, 'FM': 3, 'ASK': 4, 'PSK': 5,'CW': 6,
                '2FSK':7,'4FSK': 8,'8FSK': 9,'GMSK': 10}

def get_signal_num(path_, num_):
    if os.path.exists(path_):
        data = np.load(path_, allow_pickle=True)['labels']
    # num_ = [0] * 3
    for i in range(len(data)):
        l = len(data[i])
        for j in range(l // 3):
            num_[data[i][j * 3 + 2]] += 1
    return num_

def get_dataloader_num(dataset):
    num_ = [0] * cfg['num_classes']
    length = dataset.__len__()
    for i in range(length):
        _, l = dataset.pull_anno(i)
        for j in range(l.shape[0]):
            label = int(l[j][2])
            num_[label] += 1
    return num_

def targetlabel2mul(targe_data):
    batch_size = len(targe_data)
    multi_lab = np.zeros((batch_size,cfg['num_classes']-1))

    for bs in range(batch_size):
        num_target = len(targe_data[bs])
        for nt in range(num_target):
            target_class = targe_data[bs][nt][2]
            multi_lab[bs][target_class.long()]= 1

    multi_lab = torch.FloatTensor(multi_lab)
    if cfg['using_gpu']:
        multi_lab = multi_lab.cuda()
    return multi_lab

def model_evalute(net,eval_data_loader,num_data):
    net.eval()
    correct = 0
    total_len = len(eval_data_loader.dataset)
    num_signal = cfg['num_classes'] - 1
    confusion_matrix = np.zeros((num_signal+1, num_signal+1))

    # loss counters
    loc_loss = 0
    conf_loss = 0

    for fft1, fft2, fft3, fft4, fft5, targets in eval_data_loader:

        if cfg['using_gpu']:
            fft1 = Variable(fft1.cuda())
            fft2 = Variable(fft2.cuda())
            fft3 = Variable(fft3.cuda())
            fft4 = Variable(fft4.cuda())
            fft5 = Variable(fft5.cuda())
            with torch.no_grad():
                targets = [Variable(ann.cuda()) for ann in targets]
        else:
            fft1 = Variable(fft1)
            fft2 = Variable(fft2)
            fft3 = Variable(fft3)
            fft4 = Variable(fft4)
            fft5 = Variable(fft5)

            with torch.no_grad():
                targets = [Variable(ann) for ann in targets]

        with torch.no_grad():
            out = net.for_eval(fft1, fft2, fft3, fft4, fft5)

        # pred = decode_result_all_class(out, NMS_thre = settings['NMS_thre'],score_thre = settings['score_thre'],
        #                          variance =cfg['variance'] ,top_kN= 100)

        pred = decode_result(out, NMS_thre = settings['NMS_thre'],score_thre = settings['score_thre'],
                                 variance =cfg['variance'] ,top_kN= 100)

        confusion_matrix += match_result(pred,targets,settings)


    det_p,acc_p = compute_p_r(confusion_matrix,num_data)
    net.train()
    return det_p,acc_p

def train(settings, save_dir, resume=None, using_gpu=True, using_vim=True):
    if torch.cuda.is_available():
        if using_gpu:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            print('GPU detected, why not using it? It`s way more faster.')
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists('./fft_weights'):
        os.mkdir('./fft_weights')

    if not os.path.exists('./DA_fft_weights'):
        os.mkdir('./DA_fft_weights')
    #source data
    datadir = settings['data_dir']
    labeldir = settings['label_dir']
    #tgt data
    tgt_datadir = settings['tgt_data_dir']
    tgt_labeldir = settings['tgt_label_dir']
    #tgt data
    tgt_eval_datadir = settings['tgt_eval_datadir']
    tgt_eval_labeldir = settings['tgt_eval_labeldir']

    dataset = SignalDetectionv2(datadir, labeldir, settings['data_augmentation'])
    tgt_dataset = SignalDetectionv2(tgt_datadir, tgt_labeldir, False)
    tgt_eval_dataset = tgt_dataset

    print('source data length :',len(dataset))
    print('eval_dataset data length :',len(dataset))
    print('target data length :',len(tgt_dataset))

    num_train = get_dataloader_num(dataset)
    num_eval = get_dataloader_num(dataset)
    num_tgt = get_dataloader_num(tgt_dataset)

    print('num_train :',num_train)
    print('num_eval :',num_eval)
    print('num_tgt :', num_tgt)

    # 检测网络结构
    ssd_net = build_SSD('train', cfg['min_dim'], cfg['num_classes'], cfg)
    net = ssd_net

    # print('parameters: :', sum(param.numel() for param in net.parameters()))
    if using_gpu:
        # net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True
        net = net.cuda()
    if resume:
        print('resume weights...')
        net.load_state_dict(torch.load(resume))
    else:
        print('Start training, initializing weights...')
        net.apply(weights_init)

    criterion = MultiBoxLossv2(cfg['num_classes'], overlap_threshold=0.5, prior_for_matchingm=True,
                               bkg_label=0, neg_mining=True, neg_pos=4, neg_overlap=0.5,
                               encode_target=False, use_gpu=using_gpu)
    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading dataset.')
    step_index = 0

    # dataloader 数据读取器
    data_loader = data.DataLoader(dataset, settings['batch_size'],
                                  num_workers=0, shuffle=True,
                                  collate_fn=detection_collate, pin_memory=True)

    eval_data_loader = data.DataLoader(dataset, settings['batch_size'],
                                  num_workers=0, shuffle=False,
                                  collate_fn=detection_collate, pin_memory=True)

    tgt_data_loader = data.DataLoader(tgt_dataset, settings['batch_size'],
                                  num_workers=0, shuffle=True,
                                  collate_fn=detection_collate, pin_memory=True)

    tgt_eval_loader = data.DataLoader(tgt_eval_dataset, settings['batch_size'],
                                  num_workers=0, shuffle=False,
                                  collate_fn=detection_collate, pin_memory=True)
    epoch_size = len(data_loader) // settings['batch_size']

    batch_iterator = iter(data_loader)
    eval_batch_iterator = iter(eval_data_loader)
    tgt_batch_iterator = iter(tgt_data_loader)


    DA_lamda = settings['DA_lamda']
    DA_lamda_ins = settings['DA_lamda_ins']
    DA_mu = settings['DA_mu']
    DA_epsilon = settings['DA_epsilon']
    if settings['pre_train']:

        # optimizer = optim.SGD(net.parameters(), lr=settings['lr'],
        #                       momentum=settings['momentum'], weight_decay=settings['weight_decay'])

        optimizer = optim.Adam(net.parameters(), lr=settings['lr'], betas=[0.9, 0.99], eps=1e-8,
                               weight_decay=settings['weight_decay'])

        for iteration in range(settings['start_iter'], cfg['max_iter']):
            if iteration in cfg['lr_steps']:
                step_index += 1
                adjust_learning_rate(optimizer, settings['lr'], settings['gamma'], step_index)

            try:
                fft1, fft2, fft3, fft4, fft5, targets = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(data_loader)
                fft1, fft2, fft3, fft4, fft5, targets = next(batch_iterator)
            # except Exception as e:
            #     print("Loading data Exception:", e)

            if using_gpu:
                fft1 = Variable(fft1.cuda())
                fft2 = Variable(fft2.cuda())
                fft3 = Variable(fft3.cuda())
                fft4 = Variable(fft4.cuda())
                fft5 = Variable(fft5.cuda())
                with torch.no_grad():
                    targets = [Variable(ann.cuda()) for ann in targets]
            else:
                fft1 = Variable(fft1)
                fft2 = Variable(fft2)
                fft3 = Variable(fft3)
                fft4 = Variable(fft4)
                fft5 = Variable(fft5)
                with torch.no_grad():
                    targets = [Variable(ann) for ann in targets]


            try:
                tgt_fft1, tgt_fft2, tgt_fft3, tgt_fft4,tgt_fft5, tgt_targets = next(tgt_batch_iterator)
            except StopIteration:
                tgt_batch_iterator = iter(tgt_data_loader)
                tgt_fft1, tgt_fft2, tgt_fft3, tgt_fft4,tgt_fft5, tgt_targets = next(tgt_batch_iterator)
            # except Exception as e:
            #     print("Loading data Exception:", e)

            if using_gpu:
                tgt_fft1 = Variable(tgt_fft1.cuda())
                tgt_fft2 = Variable(tgt_fft2.cuda())
                tgt_fft3 = Variable(tgt_fft3.cuda())
                tgt_fft4 = Variable(tgt_fft4.cuda())
                tgt_fft5 = Variable(tgt_fft5.cuda())
                with torch.no_grad():
                    targets = [Variable(ann.cuda()) for ann in targets]
            else:
                tgt_fft1 = Variable(tgt_fft1)
                tgt_fft2 = Variable(tgt_fft2)
                tgt_fft3 = Variable(tgt_fft3)
                tgt_fft4 = Variable(tgt_fft4)
                tgt_fft5 = Variable(tgt_fft5)
                with torch.no_grad():
                    targets = [Variable(ann) for ann in targets]

            # forward
            t0 = time.time()

            out, DA_img_loss_cls, tgt_out, tgt_DA_img_loss_cls, DA_ins_loss_cls, \
            tgt_DA_ins_loss_cls, multi_class_out, tgt_multi_class_out \
                = net(fft1, fft2,fft3,fft4,fft5,tgt_fft1, tgt_fft2, tgt_fft3,tgt_fft4,tgt_fft5)

            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets,min=10)
            loss = loss_l +  loss_c
            loss.backward()
            optimizer.step()

            t1 = time.time()

            if iteration % settings['show_loss_iter'] == 0:

                print('timer: %.4f sec.' % (t1 - t0))
                print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')

                result2txt = str(data)  # data是前面运行出的数据，先将其转为字符串才能写入
                # with open('D:/USTC/20200913/20210512/train_25KHz_fft_10_80_with_FPN/fft_weights/fft_loss.txt','a')\
                #         as file_handle:
                with open('./fft_weights/fft_loss.txt', 'a') as file_handle:  # .txt可以不自己新建,代码会自动新建
                    file_handle.write('timer: %.4f sec.' % (t1 - t0))  # 写入
                    file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
                    file_handle.write('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()))  # 写入
                    file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据


            if iteration != 0 and iteration % settings['save_model_iter'] == 0:
                localtime = time.localtime(time.time())
                track = str(localtime[0]) + str(localtime[1]).zfill(2) + str(localtime[2]).zfill(2)
                print('Saving state, iter:', iteration)


                torch.save(net.state_dict(),
                           './fft_weights/ResNet_' + repr(iteration) + '.pth')

                if settings['eval_model']:
                    val_det_p , val_acc_p = model_evalute(net, eval_data_loader,num_eval)
                    print(repr(iteration) + ' ： det_p :' + ' '.join(str(c) for c in val_det_p) +'\n'+
                          'acc_p :' + ' '.join(str(c) for c in val_acc_p) + '\n')
                    with open('./fft_weights/eval_results.txt', 'a') \
                            as file_handle:
                        file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
                        file_handle.writelines('iter '+
                            repr(iteration)  + '  : det_p :'+ ' '.join(str(c) for c in val_det_p) +'\n'+
                            'acc_p :'+' '.join(str(c) for c in val_acc_p) +'\n')

                    tgt_det_p , tgt_acc_p = model_evalute(net, tgt_eval_loader,num_tgt)

                    print(repr(iteration) + ' ： tgt_det_p :' + ' '.join(str(c) for c in tgt_det_p) +'\n'+
                            'tgt_acc_p :' + ' '.join(str(c) for c in tgt_acc_p) + '\n')
                    with open('./fft_weights/tgt_results.txt', 'a') \
                            as file_handle:
                        file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
                        file_handle.writelines('iter '+
                            repr(iteration)  + ' : tgt_det_p :'+ ' '.join(str(c) for c in tgt_det_p) +'\n'+
                            'tgt_acc_p :'+' '.join(str(c) for c in tgt_acc_p) +'\n')
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading dataset.')
    step_index = 0
    #DA process
    if True :
        iteration = 0
        if settings['eval_model']:
            val_det_p, val_acc_p = model_evalute(net, eval_data_loader,num_eval)
            print(repr(iteration) + ' : det_p :' + ' '.join(str(c) for c in val_det_p) + '\n' +
                  'acc_p :' + ' '.join(str(c) for c in val_acc_p) + '\n')
            with open('./DA_fft_weights/eval_results.txt', 'a') \
                    as file_handle:
                file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
                file_handle.writelines(
                    repr(iteration) + ' : det_p :' + ' '.join(str(c) for c in val_det_p) + '\n' +
                    'acc_p :' + ' '.join(str(c) for c in val_acc_p) + '\n')

            tgt_det_p, tgt_acc_p = model_evalute(net, tgt_eval_loader,num_tgt)

            print(repr(iteration) + 'tgt_det_p :' + ' '.join(str(c) for c in tgt_det_p) + '\n' +
                  'tgt_acc_p :' + ' '.join(str(c) for c in tgt_acc_p) + '\n')
            with open('./DA_fft_weights/tgt_results.txt', 'a') \
                    as file_handle:
                file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
                file_handle.writelines(
                    repr(iteration) + 'tgt_det_p :' + ' '.join(str(c) for c in tgt_det_p) + '\n' +
                    'tgt_acc_p :' + ' '.join(str(c) for c in tgt_acc_p) + '\n')

        # optimizer = optim.SGD(net.parameters(), lr=settings['DA_lr'],
        #                       momentum=settings['momentum'], weight_decay=settings['weight_decay'])

        optimizer = optim.Adam(net.parameters(), lr=settings['DA_lr'], betas=[0.9, 0.99], eps=1e-8,
                               weight_decay=settings['weight_decay'])

        for iteration in range(settings['DA_start_iter'], cfg['DA_max_iter']):

            if iteration in cfg['DA_lr_steps']:
                step_index += 1
                adjust_learning_rate(optimizer, settings['DA_lr'], settings['gamma'], step_index)

            try:
                fft1, fft2, fft3, fft4, fft5, targets = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(data_loader)
                fft1, fft2, fft3, fft4, fft5, targets = next(batch_iterator)
            # except Exception as e:
            #     print("Loading data Exception:", e)

            if using_gpu:
                fft1 = Variable(fft1.cuda())
                fft2 = Variable(fft2.cuda())
                fft3 = Variable(fft3.cuda())
                fft4 = Variable(fft4.cuda())
                fft5 = Variable(fft5.cuda())
                with torch.no_grad():
                    targets = [Variable(ann.cuda()) for ann in targets]
            else:
                fft1 = Variable(fft1)
                fft2 = Variable(fft2)
                fft3 = Variable(fft3)
                fft4 = Variable(fft4)
                fft5 = Variable(fft5)
                with torch.no_grad():
                    targets = [Variable(ann) for ann in targets]

            try:
                tgt_fft1, tgt_fft2, tgt_fft3, tgt_fft4,tgt_fft5, tgt_targets = next(tgt_batch_iterator)
            except StopIteration:
                tgt_batch_iterator = iter(tgt_data_loader)
                tgt_fft1, tgt_fft2, tgt_fft3, tgt_fft4,tgt_fft5, tgt_targets = next(tgt_batch_iterator)
            # except Exception as e:
            #     print("Loading data Exception:", e)

            if using_gpu:
                tgt_fft1 = Variable(tgt_fft1.cuda())
                tgt_fft2 = Variable(tgt_fft2.cuda())
                tgt_fft3 = Variable(tgt_fft3.cuda())
                tgt_fft4 = Variable(tgt_fft4.cuda())
                tgt_fft5 = Variable(tgt_fft5.cuda())
                with torch.no_grad():
                    targets = [Variable(ann.cuda()) for ann in targets]
            else:
                tgt_fft1 = Variable(tgt_fft1)
                tgt_fft2 = Variable(tgt_fft2)
                tgt_fft3 = Variable(tgt_fft3)
                tgt_fft4 = Variable(tgt_fft4)
                tgt_fft5 = Variable(tgt_fft5)
                with torch.no_grad():
                    targets = [Variable(ann) for ann in targets]
            # except Exception as e:
            #     print("Loading data Exception:", e)
            #生成multi_label
            multi_lab = targetlabel2mul(targets)


            # forward
            t0 = time.time()
            # print(net._modules['res']._modules['0'].weight)

            # out = net(seq_1,seq_80)
            out, DA_img_loss_cls, tgt_out, tgt_DA_img_loss_cls, DA_ins_loss_cls, \
            tgt_DA_ins_loss_cls, multi_class_out, tgt_multi_class_out \
                = net(fft1, fft2,fft3,fft4,fft5,tgt_fft1, tgt_fft2, tgt_fft3,tgt_fft4,tgt_fft5)

            # backprop

            loss_l, loss_c = criterion(out, targets, min=10)
            loss_det = loss_l + loss_c

            loc_data, conf_data, priors = out

            # batch_conf = conf_data.view(settings['batch_size'], -1, cfg['num_classes'])
            out_class, _ = torch.max(F.softmax(conf_data), dim=1)
            out_class = out_class[:, 1: cfg['num_classes']]

            tgt_loc_data, tgt_conf_data, priors = tgt_out
            tgt_out_class, _ = torch.max(F.softmax(tgt_conf_data), dim=1)
            tgt_out_class = tgt_out_class[:, 1: cfg['num_classes']]

            # 多标签分类损失
            multi_lab_loss = nn.BCELoss()
            object_mul_loss = multi_lab_loss(out_class, multi_lab)
            class_mul_loss = multi_lab_loss(multi_class_out, multi_lab)
            KL_loss = 1/2 *(KL_div(out_class,multi_class_out)+KL_div(multi_class_out,out_class))

            #tgt多标签KL距离

            KL_tgt_loss = 1/2 *(KL_div(tgt_out_class,tgt_multi_class_out)+KL_div(tgt_multi_class_out,tgt_out_class))

            consit_los = F.smooth_l1_loss(out_class, multi_class_out)

            # loss_all = loss_det
            # loss_all = loss_det + DA_lamda *(DA_img_loss_cls+tgt_DA_img_loss_cls) \
            #            + DA_lamda_ins * (DA_ins_loss_cls + tgt_DA_ins_loss_cls) \
            #            + DA_mu* (class_mul_loss+ object_mul_loss) \
            #            + DA_epsilon*KL_loss

            loss_all = loss_det + DA_lamda *(DA_img_loss_cls+tgt_DA_img_loss_cls) \
                       + DA_lamda_ins * (DA_ins_loss_cls + tgt_DA_ins_loss_cls) \
                       + DA_mu* (class_mul_loss+ object_mul_loss) \
                       + DA_epsilon*(KL_loss + KL_tgt_loss)

            # loss_all = DA_ins_loss_cls + tgt_DA_ins_loss_cls
            optimizer.zero_grad()
            loss_all.backward()

            if settings['clip_gradient']:
                clip_gradient(optimizer,2.)

            optimizer.step()

            t1 = time.time()
            if iteration != 0 and iteration % settings['save_model_iter'] == 0:
                localtime = time.localtime(time.time())
                track = str(localtime[0]) + str(localtime[1]).zfill(2) + str(localtime[2]).zfill(2)
                print('Saving state, iter:', iteration)
                torch.save(net.state_dict(),
                           './DA_fft_weights/DA_ResNet_i' + str(
                               cfg['input_channels']) + '_2500_' +
                           track + '_' + repr(iteration) + '.pth')

                if settings['eval_model']:
                    val_det_p , val_acc_p = model_evalute(net, eval_data_loader,num_eval)
                    print(repr(iteration) + ' : det_p :' + ' '.join(str(c) for c in val_det_p) +'\n'+
                          'acc_p :' + ' '.join(str(c) for c in val_acc_p) + '\n')
                    with open('./DA_fft_weights/eval_results.txt', 'a') \
                            as file_handle:
                        file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
                        file_handle.writelines(
                            repr(iteration)  + ' : det_p :'+ ' '.join(str(c) for c in val_det_p) +'\n'+
                            'acc_p :'+' '.join(str(c) for c in val_acc_p) +'\n')

                    tgt_det_p , tgt_acc_p = model_evalute(net, tgt_eval_loader,num_tgt)

                    print(repr(iteration) + 'tgt_det_p :' + ' '.join(str(c) for c in tgt_det_p) +'\n'+
                            'tgt_acc_p :' + ' '.join(str(c) for c in tgt_acc_p) + '\n')
                    with open('./DA_fft_weights/tgt_results.txt', 'a') \
                            as file_handle:
                        file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
                        file_handle.writelines(
                            repr(iteration)  + 'tgt_det_p :'+ ' '.join(str(c) for c in tgt_det_p) +'\n'+
                            'tgt_acc_p :'+' '.join(str(c) for c in tgt_acc_p) +'\n')


            if iteration % settings['show_loss_iter'] == 0:
                print('timer: %.4f sec.' % (t1 - t0))
                print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss_all.item()), end=' ')

                print('loss_c :', loss_c)
                print('loss_l :', loss_l)
                print('DA_img_loss_cls :', DA_img_loss_cls)
                print('tgt_DA_img_loss_cls :', tgt_DA_img_loss_cls)
                print('object_mul_loss :', object_mul_loss)
                print('class_mul_loss :', class_mul_loss)
                print('KL_loss :', KL_loss)
                print('consit_los :', consit_los)
                print('DA_ins_loss_cls :', DA_ins_loss_cls)
                print('tgt_DA_ins_loss_cls :', tgt_DA_ins_loss_cls)

                result2txt = str(data)  # data是前面运行出的数据，先将其转为字符串才能写入
                with open('./DA_fft_weights/fft_loss_all.txt','a')\
                        as file_handle:
                    file_handle.write('timer: %.4f sec.' % (t1 - t0))  # 写入
                    file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
                    file_handle.write('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss_all.item()))  # 写入
                    file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据

                    file_handle.write('loss_c ：%.4f'% (loss_c.item()))  # 写入
                    file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
                    file_handle.write('loss_l ：%.4f'% (loss_l.item()))  # 写入
                    file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
                    file_handle.write('DA_img_loss_cls ：%.4f'% (DA_img_loss_cls.item()))  # 写入
                    file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
                    file_handle.write('tgt_DA_img_loss_cls ：%.4f'% (tgt_DA_img_loss_cls.item()))  # 写入
                    file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
                    file_handle.write('object_mul_loss： %.4f'% (object_mul_loss.item()))  # 写入
                    file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
                    file_handle.write('class_mul_loss ：%.4f'% (class_mul_loss.item()))  # 写入
                    file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
                    file_handle.write('KL_loss： %.4f'% (KL_loss.item()))  # 写入
                    file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
                    file_handle.write('consit_los ：%.4f'% (consit_los.item()))  # 写入
                    file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
                    file_handle.write('DA_ins_loss_cls： %.4f'% (KL_loss.item()))  # 写入
                    file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
                    file_handle.write('tgt_DA_ins_loss_cls： %.4f'% (tgt_DA_ins_loss_cls.item()))  # 写入
                    file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据

def kaiming(param):
    init.kaiming_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv1d):
        kaiming(m.weight.data)


def adjust_learning_rate(optimizer, init_lr, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = init_lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    settings = {
        'batch_size': 16,
        'lr': 1e-03,
        'momentum': 0.9,
        'weight_decay': 5e-05,
        'gamma': 0.5,
        'eval_model': True,
        'show_loss_iter': 20,
        'save_model_iter': 200,
        # 'data_dir': 'D:\\USTC\\G35_recorder_npydata\\traindata_0924',
        # 'label_dir': 'D:\\USTC\\G35_recorder_npydata\\trainlabels_0924',
        'data_dir': 'D:\\USTC\\20211230\\train_simulation_data\\20211226\\Sim_awgndata_snr_-10_4_Npy\\traindata_1229',
        'label_dir': 'D:\\USTC\\20211230\\train_simulation_data\\20211226\\Sim_awgndata_snr_-10_4_Npy\\trainlabels_1229',
        'tgt_data_dir': 'D:\\USTC\\20211230\\train_simulation_data\\20211226\\Sim_Fd_100_snr_-10_4_Npy\\traindata_1229',
        'tgt_label_dir': 'D:\\USTC\\20211230\\train_simulation_data\\20211226\\Sim_Fd_100_snr_-10_4_Npy\\trainlabels_1229',
        'tgt_eval_datadir': 'D:\\USTC\\20211230\\train_simulation_data\\20211226\\Sim_Fd_100_snr_-10_4_Npy\\traindata_1229',
        'tgt_eval_labeldir': 'D:\\USTC\\20211230\\train_simulation_data\\20211226\\Sim_Fd_100_snr_-10_4_Npy\\trainlabels_1229',
        'data_augmentation': True,
        'start_iter': 0,
        #是否存在预训练检测网络
        'pre_train': False,
        'resume': '.\\fft_weights\\ResNet_17000.pth',
        # 'resume': None,
        # DA param
        'DA_start_iter': 0,
        'DA_lr': 1e-04,
        'DA_lamda': 1.,
        'DA_lamda_ins': 1.,
        'DA_mu': 1.,
        'DA_epsilon': 1.,
        'clip_gradient': False,
        # eval_param
        'NMS_thre': 0.3,
        'score_thre': 0.5,
    }
    train_weights =settings['resume']
    train(settings, './checkpoints', train_weights,cfg['using_gpu'] , False)
