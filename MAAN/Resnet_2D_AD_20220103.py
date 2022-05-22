import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import numpy as np
import warnings
import torch.backends.cudnn as cudnn
warnings.filterwarnings("ignore")
from gru_layer import _ImageDA,_InstanceDA

def KL_div(input1 , target ,epsilon=1e-5):
    KL_div_data = input1* torch.log(torch.div(input1+epsilon,target+epsilon)) \
                  + (1-input1)* torch.log(torch.div(1.-input1+epsilon,1.-target+epsilon))
    return KL_div_data.mean()

def conv7x7(in_planes, out_planes, stride=1 ,padding=3):
    """1x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(7,7), stride=stride,
                     padding=padding, bias=True)
def conv3x3(in_planes, out_planes, stride=1,padding=1):
    """1x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3,3), stride=stride,
                     padding=padding, bias=True)
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, BatchNorm=True, stride=1, DownSample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size,C6_size,feature_channel=256):
        super(FeaturePyramidNetwork, self).__init__()
        # upsample C6 to get P6 from the FPN paper, P6_1 映射层 P6_upsampled 上采样层 P6_2平滑层
        self.P6_1 = nn.Conv1d(C6_size, feature_channel, kernel_size=1, stride=1, padding=0)
        self.P6_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P6_2 = nn.Conv1d(feature_channel, feature_channel, kernel_size=3, stride=1, padding=1)

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv1d(C5_size, feature_channel, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv1d(feature_channel, feature_channel, kernel_size=3, stride=1, padding=1)
        # add P5 elementwise to C4
        self.P4_1 = nn.Conv1d(C4_size, feature_channel, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv1d(feature_channel, feature_channel, kernel_size=3, stride=1, padding=1)
        # add P4 elementwise to C3
        self.P3_1 = nn.Conv1d(C3_size, feature_channel, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv1d(feature_channel, feature_channel, kernel_size=3, stride=1, padding=1)


    def forward(self, inputs):
        C3, C4, C5, C6 ,C7  = inputs

        P6_x = self.P6_1(C6)
        P6_upsampled_x = self.P6_upsampled(P6_x)
        P6_x = self.P6_2(P6_x)

        P5_x = self.P5_1(C5)
        P5_x = P6_upsampled_x + P5_x
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        return [P3_x, P4_x, P5_x,P6_x,C7 ]
class SimpleBlock(nn.Module):
    def __init__(self, in_planes, out_planes, BatchNorm=True, stride=[1,2],padding=1):
        super(SimpleBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride,padding=padding)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.BatchNorm = BatchNorm

    def forward(self, x):

        y = self.conv1(x)
        if self.BatchNorm:
            y = self.bn1(y)
        y = self.relu(y)
        return y

class SimpleBlock_7(nn.Module):
    def __init__(self, in_planes, out_planes, BatchNorm=True, stride=[2,2] ,padding = 3):
        super(SimpleBlock_7, self).__init__()
        self.conv1 = conv7x7(in_planes, out_planes, stride , padding = padding)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.BatchNorm = BatchNorm

    def forward(self, x):

        y = self.conv1(x)
        if self.BatchNorm:
            y = self.bn1(y)
        y = self.relu(y)
        return y

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
        self.res = nn.ModuleList(base)
        self.relu = nn.ReLU(inplace=True)

        self.fft2_down = SimpleBlock_7(1, 64, True,[1,2])
        self.fft2_down2 = SimpleBlock(64, 128, True, [1,2])
        self.fft2_down3 = SimpleBlock(128, 128, True, [1, 2])

        self.fft3_down = SimpleBlock_7(1, 64, True,[1,2])
        self.fft3_down2 = SimpleBlock(64, 128, True, [2,1])
        self.fft3_down3 = SimpleBlock(128, 128, True, [1, 2])

        self.fft4_down = SimpleBlock_7(1, 64, True,[2,1])
        self.fft4_down2 = SimpleBlock(64, 128, True, [2,1])
        self.fft4_down3 = SimpleBlock(128, 128, True, [1, 2])

        self.fft80_mix = SimpleBlock(128 *3, 256, True, [1,1])

        C3_size, C4_size, C5_size, C6_size = self.cfg['num_filters'][0:4]
        self.FNPmodule = FeaturePyramidNetwork(C3_size,C4_size,C5_size,C6_size,self.cfg['FPN_feature_size'])
        self.Pooling_module = [nn.AvgPool2d((3,1),stride =(1,1)),
                               nn.AvgPool2d((3,1),stride =(1,1)),
                               nn.AvgPool2d((3,1),stride =(1,1)),
                               nn.AvgPool2d((3,1),stride =(1,1)),
                               nn.AvgPool2d((3,1),stride =(1,1)),
                               nn.AvgPool2d((3,1),stride =(1,1))]
        self.RCNN_imageDA_1 = _ImageDA(self.cfg['num_filters'][0],self.cfg['using_gpu'])
        self.RCNN_imageDA_2 = _ImageDA(self.cfg['num_filters'][1],self.cfg['using_gpu'])
        self.RCNN_imageDA_3 = _ImageDA(self.cfg['num_filters'][2],self.cfg['using_gpu'])
        self.RCNN_imageDA_4 = _ImageDA(self.cfg['num_filters'][3],self.cfg['using_gpu'])
        self.RCNN_imageDA_5 = _ImageDA(self.cfg['num_filters'][4],self.cfg['using_gpu'])
        self.RCNN_instanceDA = _InstanceDA(self.cfg['using_gpu'])

        # # Layer learns to scale the l2 normalized features from conv4_3
        # self.L2Norm = L2Norm(512, 20)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.adpat_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(256,self.cfg['num_classes']-1)
        self.sigmoid = nn.Sigmoid()
        self.need_backprop = torch.ones(1)
        self.tgt_need_backprop = torch.zeros(1)

        if self.phase is 'test' or  'Out_model':
            self.softmax = nn.Softmax(dim=-1)
            # num_classes, bkg_label, top_k, conf_thresh, nms_thresh
            self.detect = detect(self.num_calsses, 0, 200, 0.1, 0.2)

    def forward(self,x1,x2,x3,x4,x5, xt1,xt2,xt3,xt4,xt5):
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
        pl_source = list()
        loc = list()
        conf = list()
        base_score_list = list()
        base_label_list = list()


        x2 = torch.unsqueeze(x2,dim = 1)
        x2 = self.fft2_down(x2)
        x2 = self.fft2_down2(x2)

        x3 = torch.unsqueeze(x3,dim = 1)
        x3 = self.fft3_down(x3)
        x3 = self.fft3_down2(x3)

        x4 = torch.unsqueeze(x4,dim = 1)
        x4 = self.fft4_down(x4)
        x4 = self.fft4_down2(x4)

        x = torch.cat((x2,x3,x4),dim = 1)

        x = self.fft80_mix(x)

        for i in range(3):
            x = self.res[i](x)


        for i in range(3, len(self.res)):
            x = self.res[i](x)
            source.append(x)

        for (xd,pl) in zip(source,self.Pooling_module):
            pl_x = pl(xd)
            pl_x =  torch.squeeze(pl_x,dim =2 )
            pl_source.append(pl_x)

        source = self.FNPmodule(pl_source)

        for (x, l, c) in zip(source, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 1).contiguous())
            conf.append(c(x).permute(0, 2, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)

        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        DA_img_loss_cls = 0
        for i in range(len(source)):
            x = source[i]
            base_score, base_label = self.RCNN_imageDA_1(x,self.need_backprop)
            base_score_list.append(base_score)
            base_label_list.append(base_label)

        for i in range(len(base_score_list)):
            base_score = base_score_list[i]
            base_label = base_label_list[i]
            base_prob = F.log_softmax(base_score, dim=1)
            DA_img_loss_cls += F.nll_loss(base_prob, base_label)

        out = (
                loc.view(loc.size(0), -1, 2),
                conf.view(conf.size(0), -1, self.num_calsses),
                self.priors
            )
        #TARGET process
        if True:
            tgt_source = list()
            tgt_pl_source = list()
            tgt_loc = list()
            tgt_conf = list()
            tgt_base_score_list = list()
            tgt_base_label_list = list()

            x2 = torch.unsqueeze(xt2, dim=1)
            x2 = self.fft2_down(x2)
            x2 = self.fft2_down2(x2)

            x3 = torch.unsqueeze(xt3, dim=1)
            x3 = self.fft3_down(x3)
            x3 = self.fft3_down2(x3)

            x4 = torch.unsqueeze(xt4, dim=1)
            x4 = self.fft4_down(x4)
            x4 = self.fft4_down2(x4)

            x = torch.cat((x2, x3, x4), dim=1)

            x = self.fft80_mix(x)

            for i in range(3):
                x = self.res[i](x)

            for i in range(3, len(self.res)):
                x = self.res[i](x)
                tgt_source.append(x)

            for (xd, pl) in zip(tgt_source, self.Pooling_module):
                pl_x = pl(xd)
                pl_x = torch.squeeze(pl_x, dim=2)
                tgt_pl_source.append(pl_x)

            tgt_source = self.FNPmodule(tgt_pl_source)

            for (x, l, c) in zip(tgt_source, self.loc, self.conf):
                tgt_loc.append(l(x).permute(0, 2, 1).contiguous())
                tgt_conf.append(c(x).permute(0, 2, 1).contiguous())
            tgt_loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
            tgt_conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

            tgt_DA_img_loss_cls = 0
            for i in range(len(tgt_source)):
                x = tgt_source[i]
                tgt_base_score, tgt_base_label = self.RCNN_imageDA_1(x, self.tgt_need_backprop)
                tgt_base_score_list.append(tgt_base_score)
                tgt_base_label_list.append(tgt_base_label)

            for i in range(len(tgt_base_score_list)):
                tgt_base_score = tgt_base_score_list[i]
                tgt_base_label = tgt_base_label_list[i]
                tgt_base_prob = F.log_softmax(tgt_base_score, dim=1)
                tgt_DA_img_loss_cls += F.nll_loss(tgt_base_prob, tgt_base_label)

            tgt_out = (
                tgt_loc.view(loc.size(0), -1, 2),
                tgt_conf.view(conf.size(0), -1, self.num_calsses),
                self.priors
            )

        #mulit_class process
        if True:
            pool_feat = self.adpat_pool(source[0])
            pool_feat = pool_feat.squeeze()
            multi_class = self.classifier(pool_feat)
            multi_class_out = self.sigmoid(multi_class)


            tgt_pool_feat = self.adpat_pool(tgt_source[0])
            tgt_pool_feat = tgt_pool_feat.squeeze()
            tgt_multi_class = self.classifier(tgt_pool_feat)
            tgt_multi_class_out = self.sigmoid(tgt_multi_class)


            # instance_source
            instance_sigmoid, same_size_label = self.RCNN_instanceDA(pool_feat, self.need_backprop)
            instance_loss = nn.BCELoss()
            DA_ins_loss_cls = instance_loss(instance_sigmoid, same_size_label)

            # instance_focal_loss = FocalLoss (alpha=1, gamma=2, logits=True, reduce=True)
            # DA_ins_focal_loss_cls = instance_focal_loss(instance_sigmoid, same_size_label)

            # instance_target
            tgt_instance_sigmoid, tgt_same_size_label = \
                self.RCNN_instanceDA(tgt_pool_feat, self.tgt_need_backprop)
            tgt_instance_loss = nn.BCELoss()

            tgt_DA_ins_loss_cls = \
                tgt_instance_loss(tgt_instance_sigmoid, tgt_same_size_label)

        return out,DA_img_loss_cls/len(base_score_list),tgt_out,tgt_DA_img_loss_cls/len(tgt_base_score_list),DA_ins_loss_cls,\
               tgt_DA_ins_loss_cls,multi_class_out,tgt_multi_class_out

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
    def for_eval(self,x1,x2,x3,x4,x5):
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
        pl_source = list()
        loc = list()
        conf = list()

        x2 = torch.unsqueeze(x2,dim = 1)
        x2 = self.fft2_down(x2)
        x2 = self.fft2_down2(x2)

        x3 = torch.unsqueeze(x3,dim = 1)
        x3 = self.fft3_down(x3)
        x3 = self.fft3_down2(x3)

        x4 = torch.unsqueeze(x4,dim = 1)
        x4 = self.fft4_down(x4)
        x4 = self.fft4_down2(x4)

        x = torch.cat((x2,x3,x4),dim = 1)

        x = self.fft80_mix(x)

        for i in range(3):
            x = self.res[i](x)


        for i in range(3, len(self.res)):
            x = self.res[i](x)
            source.append(x)

        for (xd,pl) in zip(source,self.Pooling_module):
            pl_x = pl(xd)
            pl_x =  torch.squeeze(pl_x,dim =2 )
            pl_source.append(pl_x)

        source = self.FNPmodule(pl_source)

        for (x, l, c) in zip(source, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 1).contiguous())
            conf.append(c(x).permute(0, 2, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)

        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == 'test':

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


def backbone(cfg, in_planes, BatchNorm):
    Downsample = None
    layers = []
    layers += [SimpleBlock(256, 256, True, [2, 1])]
    layers += [SimpleBlock(256, 256, True, [2, 2])]
    layers += [SimpleBlock(256, 256, True, [2, 1])]
    in_channels = 256
    for out_channels in cfg:
        Downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size= 1, stride=[1,2], bias=True),
            nn.BatchNorm2d(out_channels)
        )
        layers += [BasicBlock(in_channels, out_channels, BatchNorm, [1,2], Downsample)]
        in_channels = out_channels
    return layers


def multibox(backbones,cfg, num_classes):
    loc_layers = []
    conf_layers = []
    res_source = [i for i in range(3, len(backbones))]
    for k, v in enumerate(res_source):
        loc_layers += [nn.Conv1d(backbones[v].conv1.out_channels, cfg[k] * 2, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv1d(backbones[v].conv1.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
    return backbones, (loc_layers, conf_layers)


def build_SSD(phase, size, num_classes, cfg):

    base_, head_ = multibox(backbone(cfg['num_filters'],256, True),
                            cfg['num_scales'], num_classes)

    return SSD_ResNet(phase, size, cfg, base_, head_, num_classes)


if __name__ == '__main__':
    cfg = {
        'min_dim': 256,
        'lr_steps': (1000, 3000, 5000, 7000, 10000, 15000, 20000, 25000, 30000, 35000),
        'max_iter': 40001,
        'num_classes': 21 + 1,
        'feature_maps': [128, 64, 32, 16, 8],
        'steps': [2, 4, 8, 16, 32],
        'num_filters': [256, 256, 256, 256, 256],
        'input_channels': 1,
        'num_scales': [9, 9, 12, 12, 12, 16],
        'min_size': [1, 2, 4, 8, 16],
        'max_size': [4, 8, 16, 32, 64],
        'variance': [0.1, 0.2],
        'clip': True,
        'using_gpu': False,
        'name': 'Signals',
        'FPN_feature_size': 256,
    }
    ssd_net = build_SSD('train', 256, 21, cfg)
    torch.manual_seed(42)
    x1 = torch.randn(10, 5, 8192)
    x2 = torch.randn(10, 20, 2048)
    x3 = torch.randn(10, 40, 1024)
    x4 = torch.randn(10, 80, 512)
    x5 = torch.randn(10, 160, 256)
    outputs = ssd_net(x1,x2,x3,x4,x5)
    # net = torch.nn.DataParallel(ssd_net)
    # cudnn.benchmark = True
    # net = net.cuda()
    print(outputs)
