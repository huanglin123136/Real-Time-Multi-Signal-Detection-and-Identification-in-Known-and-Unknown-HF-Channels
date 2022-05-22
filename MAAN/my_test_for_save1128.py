# -*- coding: utf-8 -*-
import os
import torch
import time
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from utils import *
from data import *
from data.SignalDetection_for_save import SignalDetectionv2
from ResNet_1D import build_SSD
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
final_co = []
final_sc = []
final_cl = []

dic = {'AM': 0, 'SSB': 1, 'PSK': 2, 'PSKSound': 2, '2FSK': 3, 'CW': 4,
       '8FSK': 5, 'Saopin': 6, 'Interp_flash_2': 7, 'Interp_flash': 7,
       'Interp': 8, 'Interp_small': 8, 'Unknow': 9,
       'SingleSound': 10,'AM_expand_large': 11,'Saopin_small': 12,
       'CW_fast': 13, 'Saopin_large': 14, 'PSKsmall': 15,
       'Noise': 16, 'Interp_low': 17, 'CFSK': 18,'CCS': 19  }

usefull_dic = range(20)
def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """

    imgs = []
    imgs_80 = []
    imgs_RC20 = []
    imgs_RC40 = []
    for sample in batch:
        imgs.append(sample[0])
        imgs_80.append(sample[1])
        imgs_RC20.append(sample[2])
        imgs_RC40.append(sample[3])
    return torch.stack(imgs, 0), torch.stack(imgs_80, 0),torch.stack(imgs_RC20, 0), torch.stack(imgs_RC40, 0)

def my_test(settings, using_gpu=True):
    global final_co
    global final_sc
    global final_cl
    if torch.cuda.is_available():
        if using_gpu:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            print('GPU detected, why not using it? It`s way more faster.')
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')


    datadir = settings['data_dir']
    dataset = SignalDetectionv2(datadir)
    print(len(dataset))
    # 采样权重
    ssd_net = build_SSD('test', cfg['min_dim'], cfg['num_classes'], cfg)
    net = ssd_net

    # print('parameters: :', sum(param.numel() for param in net.parameters()))
    if using_gpu:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True
        net = net.cuda()

    ssd_net.load_state_dict(torch.load(settings['trained_model']))
    net.eval()

    print('Loading dataset.')


    data_loader = data.DataLoader(dataset,settings['batch_size'],
                                  num_workers=1, shuffle=False,
                                  collate_fn=detection_collate, pin_memory=True)

    epoch_size = len(dataset) // settings['batch_size']
    batch_iterator = iter(data_loader)

    threshold = 0.5
    seq_idx = 1

    filename = 'my_test_nolabel.txt'
    for iteration in tqdm(range(epoch_size)):
        try:
            seq_1,seq_80,seq_2,seq_3 = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            seq_1 ,seq_80,seq_2,seq_3 = next(batch_iterator)


        if using_gpu:
            seq_1 = Variable(seq_1.cuda())
            seq_2 = Variable(seq_2.cuda())
            seq_3 = Variable(seq_3.cuda())
            seq_80 = Variable(seq_80.cuda())

        else:
            seq_1 = Variable(seq_1)
            seq_2 = Variable(seq_2)
            seq_3 = Variable(seq_3)
            seq_80 = Variable(seq_80)
        # forward
        t0 = time.time()
        # print(net._modules['res']._modules['0'].weight)
        out = net(seq_1,seq_2,seq_3,seq_80)

        scale_ = 22

        for j in range(settings['batch_size']):
            coords = []
            scores = []
            co = []
            sc = []
            cl = []
            # for i in range(1,out.size(1)):

            for use_idx in usefull_dic:
                i = use_idx + 1
                coord, score = extract_signal(out[j, i, :, :], threshold=threshold, scale=scale_)
                coords.append(coord)
                co += coord
                scores.append(score)
                sc += score
                cl += [i - 1] * len(coord)

            final_co.append(co)
            final_sc.append(sc)
            final_cl.append(cl)

            # with open(filename, mode='a') as f:
            #     f.write('\n Ground Truth for seq ' + str(seq_idx) + '\n')
            # txt_output('test',coords, scores, threshold, cfg['labelmap'], filename)
            seq_idx += 1




if __name__ == '__main__':
    settings = {
        'batch_size': 64,
        'lr': 1e-03,
        'momentum': 0.9,
        'weight_decay': 5e-05,
        'gamma': 0.5,
        'data_dir': 'D:/USTC/20200913/20201110/test_npy/testdata_i10_1110',
        'data_augmentation': False,
        'start_iter': 0,
        'trained_model': 'D:/USTC/20200913/20201109/train_20KHz_simple_2/RC20_RC40_fft_weights/ResNet_i1_1760_20201110_46800.pth',
    }

    my_test(settings,  using_gpu=True)
    list_co = final_co
    list_sc = final_sc
    list_cl = final_cl
    np.savez( 'result.npz', final_co=list_co, final_sc=list_sc,
             final_cl=list_cl)
