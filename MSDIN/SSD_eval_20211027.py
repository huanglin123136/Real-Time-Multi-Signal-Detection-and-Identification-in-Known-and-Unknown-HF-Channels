# -*- coding: utf-8 -*-
import os
import numpy as np
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
from ResNet_2D_20211021 import build_SSD
# from nms_test import decode_result,match_result,compute_p_r ,decode_result_all_class
from nms_ground import decode_result,match_result,compute_p_r ,decode_result_all_class
import scipy.io as sio
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
signal_check = Signal_label = ['SSB', 'SingleNoise', 'AM', 'SingleSound', 'CFSK',
                               'CCS', 'AM_expand_small', 'AM_expand_large',
                               'Interp_flash', 'Saopin', 'CW', '2FSK', 'PSK',
                               'Interp', 'Noise', 'Interp_flash_2', 'Unknow',
                               'Interp_low', 'PSKsmall', 'Saopin_small', '8FSK',
                               'Saopin_large', 'CW_fast', 'Unknow_DD', 'Interp_small',
                               'PSKSound', 'Saolarge', 'Saosmall']

dic = {'AM': 0, 'SSB': 1, 'PSK': 2, 'PSKSound': 2, '2FSK': 3,
       'CFSK': 3, 'CW': 4, 'CW_fast': 4, '8FSK': 5, 'Saopin': 6,
       'Interp_flash_2': 6, 'Interp_flash': 6, 'Interp': 7,
       'Interp_small': 7, 'Unknow': 8, 'Unknow_DD': 8, 'SingleSound': 9,
       'CCS': 9, 'SingleNoise': 9, 'AM_expand_large': 10, 'AM_expand_small': 10,
       'Saopin_small': 11, 'Saopin_large': 11, 'Saolarge': 11, 'Saosmall': 11,
       'PSKsmall': 12, 'Noise': 13, 'None1': 14, 'None2': 15, 'None3': 16, 'None4': 17,
       'None5': 18, 'None6': 19}




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

def model_evalute(net,eval_data_loader,num_data,settings):
    net.eval()
    correct = 0
    total_len = len(eval_data_loader.dataset)
    num_signal = cfg['num_classes'] - 1
    confusion_matrix = np.zeros((num_signal+1, num_signal+1))

    # loss counters
    loc_loss = 0
    conf_loss = 0

    for fft1,fft2,fft3,fft4,fft5,targets in eval_data_loader:
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
            out = net(fft1,fft2,fft3,fft4,fft5)

        # pred = decode_result_all_class(out, NMS_thre = settings['NMS_thre'],score_thre = settings['score_thre'],
        #                          variance =cfg['variance'] ,top_kN= 100)

        pred = decode_result(out, NMS_thre = settings['NMS_thre'],score_thre = settings['score_thre'],
                                 variance =cfg['variance'] ,top_kN= 100)

        confusion_matrix += match_result(pred,targets,settings,NMS_thre = settings['match_thre'],)


    det_p,acc_p = compute_p_r(confusion_matrix,num_data)
    # print('检测概率',det_p)
    # print('准确率',acc_p)

    return det_p,acc_p,confusion_matrix

def get_dataloader_num(dataset):
    num_ = [0] * cfg['num_classes']
    length = dataset.__len__()
    for i in range(length):
        _, l = dataset.pull_anno(i)
        for j in range(l.shape[0]):
            label = int(l[j][2])
            num_[label] += 1
    return num_



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




if __name__ == '__main__':

    settings = {
        'batch_size': 128,
        'lr': 1e-03,
        'momentum': 0.9,
        'weight_decay': 5e-05,
        'gamma': 0.25,
        'eval_model':True,

        # 'tgt_data_dir': '.\\npydata_snr_refine\\tgtdata_test_0515',
        # 'tgt_label_dir': '.\\npydata_snr_refine\\tgtlabels_test_0515',
        'tgt_data_dir': 'D:\\USTC\\20210915\\20210915\\G35_recorder_npydata\\testdata_0924',
        'tgt_label_dir': 'D:\\USTC\\20210915\\20210915\\G35_recorder_npydata\\testlabels_0924',
        'start_iter': 0,
        # 'resume': None,
        'resume': './fft_weights/ResNet_28000.pth',
        'using_gpu': True,
        #eval_param
        'NMS_thre' :0.5,
        'match_thre' : 0.3,
        'score_thre': 0.3,

    }


    resume= settings['resume']
    using_gpu = settings['using_gpu']
    #读取数据
    if True:
        if torch.cuda.is_available():
            if using_gpu:
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            else:
                print('GPU detected, why not using it? It`s way more faster.')
                torch.set_default_tensor_type('torch.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        # tgt data
        tgt_datadir = settings['tgt_data_dir']
        tgt_labeldir = settings['tgt_label_dir']

        # tgt data
        tgt_dataset = SignalDetectionv2(tgt_datadir, tgt_labeldir, False)

        print('target data length :', len(tgt_dataset))

        num_tgt = get_dataloader_num(tgt_dataset)

        print('num_tgt :', num_tgt)

        # 采样权重

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

        tgt_data_loader = data.DataLoader(tgt_dataset, settings['batch_size'],
                                          num_workers=0, shuffle=False,
                                          collate_fn=detection_collate, pin_memory=True)


    tgt_det_p, tgt_acc_p, confuse_matx = model_evalute(net, tgt_data_loader, num_tgt,settings)
    sio.savemat('result_28000_all.mat', dict(
        [('tgt_det_p', tgt_det_p), ('tgt_acc_p', tgt_acc_p), ('confuse_matx', confuse_matx),
         ('num_sig', num_tgt)]))
    print('1\n')

    settings = {
        'batch_size': 128,
        'lr': 1e-03,
        'momentum': 0.9,
        'weight_decay': 5e-05,
        'gamma': 0.25,
        'eval_model':True,

        # 'tgt_data_dir': '.\\npydata_snr_refine\\tgtdata_test_0515',
        # 'tgt_label_dir': '.\\npydata_snr_refine\\tgtlabels_test_0515',
        'tgt_data_dir': 'D:\\USTC\\20210915\\20210915\\G35_recorder_npydata\\testdata_0924',
        'tgt_label_dir': 'D:\\USTC\\20210915\\20210915\\G35_recorder_npydata\\testlabels_0924',
        'start_iter': 0,
        # 'resume': None,
        'resume': './fft_weights/ResNet_40000.pth',
        'using_gpu': True,
        #eval_param
        'NMS_thre' :0.5,
        'match_thre' : 0.3,
        'score_thre': 0.3,

    }
    resume = settings['resume']
    if resume:
        print('resume weights...')
        net.load_state_dict(torch.load(resume))

    tgt_det_p, tgt_acc_p, confuse_matx = model_evalute(net, tgt_data_loader, num_tgt,settings)
    sio.savemat('result_40000_all.mat', dict(
        [('tgt_det_p', tgt_det_p), ('tgt_acc_p', tgt_acc_p), ('confuse_matx', confuse_matx),
         ('num_sig', num_tgt)]))
    print('2\n')

