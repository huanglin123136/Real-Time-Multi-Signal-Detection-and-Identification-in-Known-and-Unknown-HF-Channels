
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
# import cv2


class ImageLabelResizeLayer(nn.Module):
    """
    Resize label to be the same size with the samples
    """
    def __init__(self):
        super(ImageLabelResizeLayer, self).__init__()


    def forward(self,x,need_backprop,using_gpu = True):

        feats = x.detach().cpu().numpy()
        lbs = need_backprop.detach().cpu().numpy()
        gt_blob = np.zeros((feats.shape[0], feats.shape[2]), dtype=np.float32)
        for i in range(feats.shape[0]):
            gt_blob[i,:] = lbs
        y=Variable(torch.from_numpy(gt_blob))

        if using_gpu :
            y=y.cuda()

        return y.long()


class InstanceLabelResizeLayer(nn.Module):


    def __init__(self):
        super(InstanceLabelResizeLayer, self).__init__()

    def forward(self, x,need_backprop,using_gpu = True):
        feats = x.data.cpu().numpy()
        lbs = need_backprop.data.cpu().numpy()
        resized_lbs = np.ones((feats.shape[0], 1), dtype=np.float32)
        for i in range(lbs.shape[0]):
            resized_lbs[:,0] = lbs
        y=torch.from_numpy(resized_lbs)
        if using_gpu :
            y=y.cuda()

        return y


class ImgallLabelResizeLayer(nn.Module):


    def __init__(self):
        super(ImgallLabelResizeLayer, self).__init__()

    def forward(self, x,need_backprop,using_gpu = True):
        feats = x.data.cpu().numpy()
        lbs = need_backprop.data.cpu().numpy()
        resized_lbs = np.ones((feats.shape[0]), dtype=np.float32)
        for i in range(lbs.shape[0]):
            resized_lbs[:] = lbs
        y=torch.from_numpy(resized_lbs)
        if using_gpu :
            y=y.cuda()

        return y.long()
