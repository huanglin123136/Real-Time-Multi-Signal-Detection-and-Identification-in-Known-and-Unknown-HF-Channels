from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np

import torch.nn as nn
from torch.autograd import Function
from model.da_faster_rcnn.LabelResizeLayer import ImageLabelResizeLayer
from model.da_faster_rcnn.LabelResizeLayer import InstanceLabelResizeLayer
import torch.optim as optim


class GRLayer(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.alpha=0.1

        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output=grad_outputs.neg() * ctx.alpha
        # output = grad_outputs * ctx.alpha
        return output

def grad_reverse(x):
    return GRLayer.apply(x)


class _ImageDA(nn.Module):
    def __init__(self,dim ,using_gpu=True):
        super(_ImageDA,self).__init__()
        self.using_gpu = using_gpu
        self.dim=dim  # feat layer          256*H*W for vgg16
        self.Conv1 = nn.Conv1d(self.dim, 512, kernel_size=1, stride=1,bias=False)
        self.Conv2 = nn.Conv1d(512,2,kernel_size=1,stride=1,bias=False)
        self.reLu=nn.ReLU(inplace=False)
        self.LabelResizeLayer=ImageLabelResizeLayer()

    def forward(self,x,need_backprop):
        x=grad_reverse(x)
        x=self.reLu(self.Conv1(x))
        x=self.Conv2(x)
        label=self.LabelResizeLayer(x,need_backprop,using_gpu = self.using_gpu)
        return x,label

class _ImageDA_all(nn.Module):
    def __init__(self,dim,using_gpu=True):
        super(_ImageDA_all,self).__init__()
        self.dim=dim  # feat layer          256*H*W for vgg16
        self.using_gpu = using_gpu
        self.Conv1 = nn.Conv1d(self.dim, 512, kernel_size=1, stride=1,bias=True)
        self.Conv2 = nn.Conv1d(512,256,kernel_size=1,stride=1,bias=True)
        self.adpat_pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(256,2)
        self.reLu=nn.ReLU(inplace=False)
        self.LabelResizeLayer=ImgallLabelResizeLayer()

    def forward(self,x,need_backprop):
        x = grad_reverse(x)
        x = self.reLu(self.Conv1(x))
        x = self.reLu(self.Conv2(x))
        x = self.adpat_pool(x)
        x = self.linear(x.squeeze())
        label=self.LabelResizeLayer(x,need_backprop,using_gpu = self.using_gpu)
        return x,label


class _InstanceDA(nn.Module):
    def __init__(self,using_gpu=True):
        super(_InstanceDA,self).__init__()
        self.using_gpu = using_gpu
        self.dc_ip1 = nn.Linear(256, 1024)
        self.dc_relu1 = nn.ReLU()
        self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(1024, 1024)
        self.dc_relu2 = nn.ReLU()
        self.dc_drop2 = nn.Dropout(p=0.5)

        self.clssifer=nn.Linear(1024,1)
        self.LabelResizeLayer=InstanceLabelResizeLayer()

    def forward(self,x,need_backprop):
        x=grad_reverse(x)
        x=self.dc_drop1(self.dc_relu1(self.dc_ip1(x)))
        x=self.dc_drop2(self.dc_relu2(self.dc_ip2(x)))
        x=F.sigmoid(self.clssifer(x))
        label = self.LabelResizeLayer(x, need_backprop, using_gpu = self.using_gpu)
        return x,label

class test_net(nn.Module):
    def __init__(self):
        super(test_net, self).__init__()
        self.RCNN_imageDA = _ImageDA(256)
        self.RCNN_instanceDA = _InstanceDA()
        self.fc_l = nn.Linear(256,256)
        self.relu = nn.ReLU()
    def forward(self,im_data,instance_data,need_backprop,tgt_im_data,tgt_instance_data,tgt_need_backprop):
        batch_size = im_data.size(0)
        need_backprop = need_backprop.data
        base_score, base_label = self.RCNN_imageDA(im_data, need_backprop)
        base_prob = F.log_softmax(base_score, dim=1)
        DA_img_loss_cls = F.nll_loss(base_prob, base_label)

        instance_data = self.fc_l(instance_data)
        instance_data = self.relu(instance_data)
        tgt_instance_data = self.fc_l(tgt_instance_data)
        tgt_instance_data = self.relu(tgt_instance_data)
        instance_sigmoid, same_size_label = self.RCNN_instanceDA(instance_data, need_backprop)
        instance_loss = nn.BCELoss()
        DA_ins_loss_cls = instance_loss(instance_sigmoid, same_size_label)

        tgt_instance_sigmoid, tgt_same_size_label = \
            self.RCNN_instanceDA(tgt_instance_data,tgt_need_backprop)
        tgt_instance_loss = nn.BCELoss()
        tgt_DA_ins_loss_cls = \
            tgt_instance_loss(tgt_instance_sigmoid, tgt_same_size_label)

        return DA_img_loss_cls,DA_ins_loss_cls,tgt_DA_ins_loss_cls

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_imageDA, 0, 0.01 )
        normal_init(self.RCNN_instanceDA, 0, 0.01 )

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

if __name__ ==  "__main__":
    need_backprop = torch.zeros(1)
    tgt_need_backprop = torch.ones(1)
    im_data = torch.rand(2,256,256) + 0.05
    instance_data = torch.rand(2,256)+0.03
    im_data = Variable(im_data.cuda())
    instance_data = Variable(instance_data.cuda())

    tgt_im_data = torch.rand(2,256,256) + 0.05
    tgt_instance_data = torch.rand(2,256)+0.03
    tgt_im_data = Variable(tgt_im_data.cuda())
    tgt_instance_data = Variable(tgt_instance_data.cuda())


    net = test_net()
    net = net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=[0.9, 0.99], eps=1e-8,
                           weight_decay=1e-5)
    for i in range(100):
        DA_img_loss_cls,DA_ins_loss_cls,tgt_DA_ins_loss_cls = net(im_data,instance_data,need_backprop,tgt_im_data,tgt_instance_data,tgt_need_backprop)
        loss_c = DA_ins_loss_cls  +tgt_DA_ins_loss_cls
        optimizer.zero_grad()
        loss_c.backward()
        optimizer.step()
        print(loss_c)

