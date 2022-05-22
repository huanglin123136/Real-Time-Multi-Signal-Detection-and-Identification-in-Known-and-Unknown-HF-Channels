import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import numpy as np


def conv1x3(in_planes, out_planes, stride=1):
    """1x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=9, stride=stride,
                     padding=4, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, BatchNorm=True, stride=1, DownSample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.BatchNorm = BatchNorm
        self.DownSample = DownSample
        self.stride = stride

    def forward(self, x):
        residual = x

        y = self.conv1(x)
        if self.BatchNorm:
            y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        if self.BatchNorm:
            y = self.bn2(y)

        if self.DownSample is not None:
            residual = self.DownSample(x)

        out = y + residual
        out = self.relu(out)
        return out


class SSD_ResNet(nn.Module):
    """Single Shot Multibox Architecture
     The network is composed of a homebrew ResNet network .
     Each multibox layer branches into
         1) conv1d for class conf scores
         2) conv1d for localization predictions
         3) associated priorbox layer to produce default bounding
            boxes specific to the layer's feature map size.
     See: https://arxiv.org/pdf/1512.02325.pdf for more details.

     Args:
         phase: (string) Can be "test" or "train"
         size: input image size
         base: ResNets layers for input, size of input is 1x8192
         head: "multibox head" consists of loc and conf conv layers
     """

    def __init__(self, phase, size, cfg, base, head, num_classes):
        super(SSD_ResNet, self).__init__()
        self.phase = phase
        self.size = size
        self.num_calsses = num_classes
        self.cfg = cfg
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())
        # SSD network
        self.res = nn.ModuleList(base[0])
        self.res2 = nn.ModuleList(base[1])
        self.res3 = nn.ModuleList(base[2])
        # # Layer learns to scale the l2 normalized features from conv4_3
        # self.L2Norm = L2Norm(512, 20)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if self.phase is 'test':
            self.softmax = nn.Softmax(dim=-1)
            # num_classes, bkg_label, top_k, conf_thresh, nms_thresh
            self.detect = detect(self.num_calsses, 0, 200, 0.01, 0.1)

    def forward(self, x1, x2,x3):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,1,8192].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*2]
                    3: priorbox layers, Shape: [2,num_priors*2]
        """
        source = list()
        loc = list()
        conf = list()

        for i in range(6):
            x1 = self.res[i](x1)
            x2 = self.res2[i](x2)
            x3 = self.res3[i](x3)

        for i in range(6, len(self.res)):
            x1 = self.res[i](x1)
            x2 = self.res2[i](x2)
            x3 = self.res3[i](x3)
            "x1 ,x2 合并"
            x_concat = torch.cat((x1, x2, x3), 1)
            source.append(x_concat)

        for (x, l, c) in zip(source, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 1).contiguous())
            conf.append(c(x).permute(0, 2, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)

        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)


        out = self.detect(
                loc.view(loc.size(0), -1, 2),  # loc prediction
                self.softmax(conf.view(conf.size(0), -1, self.num_calsses)),  # label conf
                self.priors.type(type(x.data))
            )

        return out

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def backbone(cfg, in_planes, BatchNorm, stride=2):
    Downsample = None
    layers = []
    in_channels = in_planes
    conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=stride,
                      padding=3, bias=False)
    relu = nn.ReLU(inplace=True)
    maxpool = nn.MaxPool1d(kernel_size=7, stride=2, padding=3)

    if BatchNorm is not None:
        bn1 = nn.BatchNorm1d(64)
        layers += [conv1, bn1, relu, maxpool]
    else:
        layers += [conv1, relu, maxpool]
    in_channels = 64

    for out_channels in cfg:
        Downsample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        layers += [BasicBlock(in_channels, out_channels, BatchNorm, stride, Downsample)]
        in_channels = out_channels
    return layers

def backbone_2(cfg, in_planes, BatchNorm, stride=2):
    Downsample = None
    layers = []
    in_channels = in_planes
    conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=stride,
                      padding=3, bias=False)
    relu = nn.ReLU(inplace=True)
    maxpool = nn.MaxPool1d(kernel_size=7, stride=4, padding=3)

    if BatchNorm is not None:
        bn1 = nn.BatchNorm1d(64)
        layers += [conv1, bn1, relu, maxpool]
    else:
        layers += [conv1, relu, maxpool]
    in_channels = 64

    for out_channels in cfg:
        Downsample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        layers += [BasicBlock(in_channels, out_channels, BatchNorm, stride, Downsample)]
        in_channels = out_channels
    return layers

def backbone_3(cfg, in_planes, BatchNorm, stride=2):
    Downsample = None
    layers = []
    in_channels = in_planes
    conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=stride*2 ,
                      padding=3, bias=False)
    relu = nn.ReLU(inplace=True)
    maxpool = nn.MaxPool1d(kernel_size=7, stride=4, padding=3)

    if BatchNorm is not None:
        bn1 = nn.BatchNorm1d(64)
        layers += [conv1, bn1, relu, maxpool]
    else:
        layers += [conv1, relu, maxpool]
    in_channels = 64

    for out_channels in cfg:
        Downsample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        layers += [BasicBlock(in_channels, out_channels, BatchNorm, stride, Downsample)]
        in_channels = out_channels
    return layers


def multibox(backbones,backbones_2, backbones_3,cfg, num_classes):
    loc_layers = []
    conf_layers = []
    res_source = [i for i in range(6, len(backbones))]
    res_source_2 = [i for i in range(6, len(backbones_2))]
    res_source_3 = [i for i in range(6, len(backbones_3))]
    for k, v in enumerate(res_source):
        loc_layers += [nn.Conv1d(backbones[v].conv1.out_channels+backbones_2[v].conv1.out_channels+backbones_3[v].conv1.out_channels, cfg[k] * 2, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv1d(backbones[v].conv1.out_channels+backbones_2[v].conv1.out_channels+backbones_3[v].conv1.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
    return (backbones,backbones_2,backbones_3), (loc_layers, conf_layers)


def build_SSD(phase, size, num_classes, cfg):
    # base = [64, 128, 128, 256, 256, 512, 512]
    # mbox = [5, 5, 5, 15, 15]
    # mbox = [6, 9, 12, 15, 15]
    base_, head_ = multibox(backbone(cfg['num_filters'], cfg['input_channels'], True, 2),backbone_2(cfg['num_filters'], 1, True, 2) ,backbone_3(cfg['num_filters'], 1, True, 2),
                            cfg['num_scales'], num_classes)

    return SSD_ResNet(phase, size, cfg, base_, head_, num_classes)


if __name__ == '__main__':
    cfg = {
        'num_classes': 8+1,
        'min_dim': 800,
        'lr_steps': (8000, 15000, 25000, 32000),
        'max_iter': 40100,
        'feature_maps': [25, 13, 7, 4],
        'steps': [32, 64, 128, 256],
        'num_filters': [128, 128, 256, 256, 512, 512],
        'input_channels': 1,
        'num_scales': [6, 9, 12, 15],
        'min_size': [24, 48, 96, 192],
        'max_size': [48, 96, 192, 384],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'Signals',
    }
    ssd_net = build_SSD('train', 800, 9, cfg)
    torch.manual_seed(42)
    inputs = torch.randn(10, 1, 800)
    inputs2 = torch.randn(10, 1, 8000)
    inputs3 = torch.randn(10, 1, 16000)
    outputs = ssd_net(inputs,inputs2,inputs3)
    print(outputs)
