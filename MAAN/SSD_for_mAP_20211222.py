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
from Resnet_2D_AD_20220103 import build_SSD
from mAP_test import decode_result, match_result, compute_p_r, decode_nms,match_AP
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
    return torch.stack(fft_1, 0), torch.stack(fft_2, 0), torch.stack(fft_3, 0), \
           torch.stack(fft_4, 0), torch.stack(fft_5, 0), targets


def model_evalute_AP(net, eval_data_loader, num_data):
    net.eval()
    correct = 0
    total_len = len(eval_data_loader.dataset)
    num_signal = cfg['num_classes'] - 1
    all_confu_matrix = np.zeros((num_signal + 1, num_signal + 1))
    all_tp = None
    all_score = None
    # loss counters
    loc_loss = 0
    conf_loss = 0
    batch_idx = 0
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

        pred = decode_nms(out, NMS_thre=settings['NMS_thre'], score_thre=settings['score_thre'],
                             variance=cfg['variance'], top_kN=100)

        confu_matrix,tp,pred_score = match_AP(pred, targets, IOU_thr = 0.5)
        all_confu_matrix += confu_matrix

        if (batch_idx == 0):
            all_tp = tp
            all_score = pred_score
        else:
            all_tp = np.concatenate((all_tp,tp),axis = 0)
            all_score = np.concatenate((all_score, pred_score), axis=0)
        batch_idx += 1

    tp_idx = np.argsort(-all_score)
    new_tp = all_tp[tp_idx]
    new_fp = 1 - new_tp
    fp = np.cumsum(new_fp)
    tp = np.cumsum(new_tp)
    rec = tp / float(sum(num_data))
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    det_p, acc_p = compute_p_r(all_confu_matrix, num_data)
    # print('检测概率',det_p)
    # print('准确率',acc_p)
    mAP = voc_ap(rec, prec, use_07_metric=False)
    return det_p, acc_p, all_confu_matrix,rec, prec,mAP

def voc_ap(rec, prec, use_07_metric=False):
    if use_07_metric:  # 使用07年方法
        # 11 个点
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])  # 插值
            ap = ap + p / 11.
    else:  # 新方式，计算所有点
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision 曲线值（也用了插值）
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def get_dataloader_num(dataset):
    num_ = [0] * cfg['num_classes']
    length = dataset.__len__()
    for i in range(length):
        _, l = dataset.pull_anno(i)
        for j in range(l.shape[0]):
            label = int(l[j][2])
            num_[label] += 1
    return num_


def eval_test(settings, resume=None, using_gpu=True, using_vim=True):
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
    iteration = 0
    if settings['eval_model']:
        det_p, acc_p, all_confu_matrix,rec, prec,mAP = model_evalute_AP(net, tgt_data_loader, num_tgt)

    return det_p, acc_p, all_confu_matrix,rec, prec,mAP,num_tgt

def kaiming(param):
    init.kaiming_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv1d):
        kaiming(m.weight.data)
        m.bias.data.zero_()

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
    snr = [4,2,0,-2,-4,-6,-8,-10]

    for snr_i in range(len(snr)):
        snr_now = snr[snr_i]
        data_dir = 'D:\\USTC\\20211230\\train_simulation_data\\20211226\\Sim_Fd_100_snr_'+str(snr_now)+'_Npy\\traindata_1229'
        label_dir = 'D:\\USTC\\20211230\\train_simulation_data\\20211226\\Sim_Fd_100_snr_'+str(snr_now)+'_Npy\\trainlabels_1229'

        settings = {
            'batch_size': 64,
            'lr': 1e-03,
            'momentum': 0.9,
            'weight_decay': 5e-05,
            'gamma': 0.25,
            'eval_model': True,
            'tgt_data_dir': data_dir,
            'tgt_label_dir': label_dir,
            # 'tgt_data_dir': 'D:\\USTC\\20211230\\train_simulation_data\\20211226\\Sim_Fd_100_snr_4_Npy\\traindata_1229',
            # 'tgt_label_dir': 'D:\\USTC\\20211230\\train_simulation_data\\20211226\\Sim_Fd_100_snr_4_Npy\\trainlabels_1229',
            # 'resume':'.\\fft_snr_4_weights\\ResNet_8000.pth',
            'resume':'.\\DA_fft_weights\\DA_ResNet_i1_2500_20220109_9400.pth',
            # eval_param
            'NMS_thre': 0.5,
            'score_thre': 0.3,
        }

        resume_train_weights = settings['resume']
        det_p, acc_p, all_confu_matrix,rec, prec,mAP,num_tgt = eval_test(settings, resume_train_weights, using_gpu=cfg['using_gpu'], using_vim=False)

        print('det_p :' + ' '.join(str(c) for c in det_p) + '\n' +
              'acc_p :' + ' '.join(str(c) for c in acc_p) + '\n'+
              'mAP:'+str(mAP))
        # sio.savemat('result_14400_Fd_100_snr_4.mat', dict(
        #     [('det_p', det_p), ('acc_p', acc_p), ('all_confu_matrix', all_confu_matrix),('rec', rec),('mAP', mAP),
        #      ('prec', prec),('num_tgt', num_tgt)]))
        savemat = 'DA_awgn2watterson' + str(snr_now) + '.mat'
        sio.savemat(savemat, dict(
            [('det_p', det_p), ('acc_p', acc_p), ('all_confu_matrix', all_confu_matrix),('rec', rec),('mAP', mAP),
             ('prec', prec),('num_tgt', num_tgt)]))
