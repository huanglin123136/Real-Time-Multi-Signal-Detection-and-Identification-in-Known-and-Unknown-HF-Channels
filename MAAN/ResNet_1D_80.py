import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def conv1x3(in_planes, out_planes, stride=1):
    """1x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


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


        self.relu = nn.ReLU(inplace=True)

        self.fft80_down_1 = conv1x3(80, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.fft80_down_2 = conv1x3(128, 256, 1)
        self.bn2 = nn.BatchNorm1d(256)
        self.fft80_down_3 = conv1x3(256, 128, 1)
        self.bn3 = nn.BatchNorm1d(128)

        self.phase = phase
        self.size = size
        self.num_calsses = num_classes
        self.cfg = cfg
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())
        # SSD network
        self.res = nn.ModuleList(base)

        # # Layer learns to scale the l2 normalized features from conv4_3
        # self.L2Norm = L2Norm(512, 20)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if self.phase is 'test' or  'Out_model':
            self.softmax = nn.Softmax(dim=-1)
            # num_classes, bkg_label, top_k, conf_thresh, nms_thresh
            self.detect = detect(self.num_calsses, 0, 200, 0.1, 0.2)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,80,220].

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

        x = self.fft80_down_1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.fft80_down_2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.fft80_down_3(x)
        x = self.relu(x)
        x = self.bn3(x)

        for i in range(4):
            x = self.res[i](x)

        for i in range(4, len(self.res)):
            x = self.res[i](x)
            source.append(x)

        for (x, l, c) in zip(source, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 1).contiguous())
            conf.append(c(x).permute(0, 2, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)

        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == 'test':
            # out = self.detect(
            #     loc.view(loc.size(0), -1, 2),  # loc prediction
            #     self.softmax(conf.view(conf.size(0), -1, self.num_calsses)),  # label conf
            #     self.priors.type(type(x.data))
            #  )
            out = self.detect.forward(
                loc.view(loc.size(0), -1, 2),  # loc prediction
                self.softmax(conf.view(conf.size(0), -1, self.num_calsses)),  # label conf
                self.priors.type(type(x.data))
             )

        elif self.phase == 'Out_model':
            out = (loc.view(loc.size(0), -1, 2),
                   self.softmax(conf.view(conf.size(0), -1, self.num_calsses)),  # label conf
                   self.priors.type(type(x.data)))
        else:
            out = (
                loc.view(loc.size(0), -1, 2),
                conf.view(conf.size(0), -1, self.num_calsses),
                self.priors
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
    first_out = 128
    conv1 = nn.Conv1d(in_channels, first_out, kernel_size=3, stride=stride,
                      padding= 1, bias=False)
    relu = nn.ReLU(inplace=True)
    maxpool = nn.MaxPool1d(kernel_size= 3, stride=2, padding= 1 )

    if BatchNorm is not None:
        bn1 = nn.BatchNorm1d(first_out)
        layers += [conv1, bn1, relu, maxpool]
    else:
        layers += [conv1, relu, maxpool]

    in_channels = first_out

    for out_channels in cfg:
        Downsample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        layers += [BasicBlock(in_channels, out_channels, BatchNorm, stride, Downsample)]

        in_channels = out_channels
    return layers


def multibox(backbones,cfg, num_classes):
    loc_layers = []
    conf_layers = []
    res_source = [i for i in range(4, len(backbones))]
    for k, v in enumerate(res_source):
        loc_layers += [nn.Conv1d(backbones[v].conv1.out_channels, cfg[k] * 2, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv1d(backbones[v].conv1.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
    return backbones, (loc_layers, conf_layers)


def build_SSD(phase, size, num_classes, cfg):

    base_, head_ = multibox(backbone(cfg['num_filters'], 128, True, 2),
                            cfg['num_scales'], num_classes)

    return SSD_ResNet(phase, size, cfg, base_, head_, num_classes)


if __name__ == '__main__':
    cfg = {
        'min_dim': 250,
        'lr_steps': (1000, 3000, 5000, 7000, 10000, 15000, 25000, 35000, 45000),
        'max_iter': 50001,
        'variance': [0.1, 0.2],
        'num_classes': 20 + 1,
        'feature_maps': [32, 16, 8, 4, 2],
        'steps': [8, 16, 32, 64, 128],
        'num_filters': [128, 256, 256, 512, 512],
        'input_channels': 1,
        'num_scales': [6, 6, 6, 6, 6],
        'min_size': [4, 12, 24, 48, 96],
        'max_size': [12, 24, 48, 96, 250],
        'variance': [0.1, 0.2],
        'clip': True,
        'using_gpu': True,
        'name': 'Signals',
    }
    ssd_net = build_SSD('train', 250 , 21, cfg)
    torch.manual_seed(42)
    inputs80 = torch.randn(10, 80, 250)
    outputs = ssd_net(inputs80)
    print(outputs)
