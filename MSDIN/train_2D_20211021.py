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

from data import *
from data.SignalDetection_20210924 import SignalDetectionv2

from layers.modules import MultiBoxLoss, MultiBoxLossv2
from ResNet_2D_20211021 import build_SSD
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


dic ={'AM': 1, 'SSB': 2, 'FM': 3, 'ASK': 4, 'PSK': 5,'CW': 6,
                '2FSK':7,'4FSK': 8,'8FSK': 9,'GMSK': 10}

def get_weights(dataset):
    sample = []
    length = dataset.__len__()
    for i in range(length):
        _, l = dataset.pull_anno(i)
        weight = 1
        sample.append(weight)
    return sample


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

    if not os.path.exists(settings['model_path']):
        os.mkdir(settings['model_path'])

    # if not os.path.exists('/home/linhuang/download/train_simulation_data_0830_C20_fftpca_fft_all/fft_weights'):
    #     os.mkdir('/home/linhuang/download/train_simulation_data_0830_C20_fftpca_fft_all/fft_weights')

    datadir = settings['data_dir']
    labeldir = settings['label_dir']
    dataset = SignalDetectionv2(datadir, labeldir, settings['data_augmentation'])
    print(len(dataset))
    # 采样权重
    sample_weights = get_weights(dataset)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(sample_weights)*10)
    ssd_net = build_SSD('train', cfg['min_dim'], cfg['num_classes'], cfg)
    net = ssd_net


    # print('parameters: :', sum(param.numel() for param in net.parameters()))
    if using_gpu:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True
        net = net.cuda()
    if resume:
        print('resume weights...')
        ssd_net.load_weights(resume)
    else:
        print('Start training, initializing weights...')
        ssd_net.res.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    # fine tune
    # for k, v in net.res.named_parameters():
    #     v.requires_grad = False

    if using_vim:
        import visdom
        global viz
        viz = visdom.Visdom()

    # optimizer = optim.SGD(net.parameters(), lr=settings['lr'],
    #                       momentum=settings['momentum'], weight_decay=settings['weight_decay'])

    optimizer = optim.Adam(net.parameters(), lr=settings['lr'], betas=[0.9, 0.99], eps=1e-8,
                           weight_decay=settings['weight_decay'])
    criterion = MultiBoxLossv2(cfg['num_classes'], 0.5, True, 0, True, 4, 0.5, False, using_gpu)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0

    print('Loading dataset.')

    step_index = 0
    if using_vim:
        vis_title = 'Signal Detection'
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    # data_loader = data.DataLoader(dataset, settings['batch_size'],
    #                               num_workers=0, shuffle=True,
    #                               collate_fn=detection_collate, pin_memory=True)

    data_loader = data.DataLoader(dataset, settings['batch_size'],
                                  num_workers=0, shuffle=False, sampler=sampler,
                                  collate_fn=detection_collate, pin_memory=True)
    epoch_size = len(data_loader) // settings['batch_size']
    batch_iterator = iter(data_loader)

    for iteration in range(settings['start_iter'], cfg['max_iter']):
        if using_vim and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1


        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, settings['lr'], settings['gamma'], step_index)



        try:
            fft1,fft2,fft3,fft4,fft5,targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            fft1,fft2,fft3,fft4,fft5,targets = next(batch_iterator)
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

        # forward
        t0 = time.time()
        # print(net._modules['res']._modules['0'].weight)
        out = net(fft1,fft2,fft3,fft4,fft5)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets,min=10)
        loss = loss_l +  loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()

        loc_loss += loss_l.item()
        conf_loss += loss_c.item()

        if iteration % 20 == 0:
            s1 = 'timer: %.4f sec.' % (t1 - t0)
            s2 = 'iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item())
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')

            result2txt = str(data)  # data是前面运行出的数据，先将其转为字符串才能写入
            with open(settings['model_path']+'\\fft_loss.txt','a') as file_handle:
            # with open('/home/linhuang/download/train_simulation_data_0830_C20_fftpca_fft_all/fft_weights/fft_loss.txt', 'a') as file_handle:  # .txt可以不自己新建,代码会自动新建
                file_handle.write('timer: %.4f sec.' % (t1 - t0))  # 写入
                file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
                file_handle.write('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()))  # 写入
                file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据




        if using_vim:
            epoch += 1
            update_vis_plot(iteration, loss_l.data[0], loss_c.data[0],
                            iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 400 == 0:
            localtime = time.localtime(time.time())
            track = str(localtime[0]) + str(localtime[1]).zfill(2) + str(localtime[2]).zfill(2)
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(),
                       settings['model_path']+'\\ResNet_' + repr(iteration) + '.pth')


    torch.save(ssd_net.state_dict(),
               save_dir + '/' + 'ResNet_max.pth')


def kaiming(param):
    init.kaiming_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv1d):
        kaiming(m.weight.data)

        # m.bias.data.zero_()


def adjust_learning_rate(optimizer, init_lr, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = init_lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend,
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    settings = {
        'batch_size': 64,
        'lr': 1e-03,
        'momentum': 0.9,
        'weight_decay': 5e-05,
        'gamma': 0.5,
        # 'data_dir': 'D:\\USTC\\G35_recorder_npydata\\traindata_0924',
        # 'label_dir': 'D:\\USTC\\G35_recorder_npydata\\trainlabels_0924',
        'data_dir': 'D:\\USTC\\20211230\\train_simulation_data\\20211226\\Sim_awgndata_snr_-10_4_Npy\\traindata_1229',
        'label_dir': 'D:\\USTC\\20211230\\train_simulation_data\\20211226\\Sim_awgndata_snr_-10_4_Npy\\trainlabels_1229',
        'data_augmentation': True,
        'start_iter': 0,
        'using_gpu': True,
        'model_path' : '.\\fft_weights'
    }


    train_weights =None
    train(settings, './checkpoints', train_weights,settings['using_gpu'] , False)
