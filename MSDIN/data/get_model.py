# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from ResNet_1D_fft_with_FPN import build_SSD
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from nms_test import decode_result,match_result,compute_p_r,decode_result_all_class
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from ResNet_2D_20211021 import build_SSD

if __name__ == '__main__':
    cfg = {
            'min_dim': 250,
            # 'lr_steps': (1000, 3000, 5000, 7000, 10000, 15000, 25000, 35000, 45000),
            'lr_steps': (100, 300, 500, 700, 1000, 1500, 2500, 3500, 4500),
            'max_iter': 50001,
            'variance': [0.1, 0.2],
            'num_classes': 21 + 1,
            'feature_maps': [32, 16, 8, 4, 2 ,1],
            'steps': [8, 16, 32, 64, 125 , 250],
            'num_filters': [256, 256, 256, 256, 512, 512],
            'input_channels': 1,
            'num_scales': [9, 9, 12, 12, 12 , 9],
            'min_size': [6, 12, 24, 48, 96 , 200 ],
            'max_size': [12, 24, 48, 96, 200, 265],
            'variance': [0.1, 0.2],
            'clip': True,
            'using_gpu': True,
            'name': 'Signals',
            'FPN_feature_size': 256,
            'labelmap' : ['AM', 'SSB', 'PSK', '2FSK', 'CW', '8FSK',
                          'Saopin','Interp','Unknow',
                          'SingleSound','Amexpandlarge','Saosmall',
                           'PSKsmall', 'Noise', 'None1','None2',
                          'None3','None4','None5','None6']
    }
    settings = {
        # eval_param
        'NMS_thre': 0.5,
        'score_thre': 0.3,
    }
    if torch.cuda.is_available():
        if cfg['using_gpu']:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            print('GPU detected, why not using it? It`s way more faster.')
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    net = build_SSD('train', cfg['min_dim'], cfg['num_classes'], cfg)
    cudnn.benchmark = True
    net = net.cuda()
    path = ''
    batch_size = 32
    fft1 = np.random.rand(batch_size, 9, 4000)
    fft2 = np.random.rand(batch_size, 39, 1000)
    fft3 = np.random.rand(batch_size, 78, 500)
    fft4 = np.random.rand(batch_size, 159, 250)
    fft5 = np.random.rand(batch_size, 318, 125)

    net.eval()
    while(1):
        t1 = time.time()

        x4=torch.Tensor(fft4)

        outputs = net(x4)

        class_keep = decode_result(outputs, NMS_thre=settings['NMS_thre'], score_thre=settings['score_thre'],
                             variance=cfg['variance'], top_kN=50)

        t2 = time.time()
        print(t2 - t1)
