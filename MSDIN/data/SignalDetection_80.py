# from .config import HOME
import os
import sys
import torch
import torch.utils.data as data
import scipy.io as sio
import numpy as np
# from data_augmentation_80 import *
from .data_augmentation_80 import *

def concat_data(path_):
    path = path_ + '_'
    # data_ = np.load(path +'1.npy')
    print(path_)
    path = os.path.abspath(path)
    data_80 = np.load(path + '1.npz', allow_pickle=True)['datas_80_fft']
    i = 2
    while os.path.exists(path + str(i) + '.npz'):
        print('loading data '+path + str(i) + '.npz....')
        new_data_80 = np.load(path + str(i) + '.npz', allow_pickle=True)['datas_80_fft']
        data_80 = np.concatenate((data_80, new_data_80), axis=0)
        i += 1
    return data_80



def concat_label(path_):
    path = path_ + '_'
    # data_ = np.load(path +'1.npy')
    print(path_)
    path=os.path.abspath(path)
    data_ = np.load(path+'1.npz',allow_pickle = True)['labels']
    i = 2
    while os.path.exists(path + str(i) + '.npz'):
        print('loading data '+path + str(i) + '.npz....')
        new_data = np.load(path + str(i) + '.npz', allow_pickle=True)['labels']
        data_ = np.concatenate((data_, new_data), axis=0)
        i += 1
    return data_

class SignalDetectionv2(data.Dataset):
    def __init__(self, data_root, label_root, data_aug=False, dataset_name='SignalDetedtion'):
        self.data_root = data_root
        self.label_root = label_root
        self.data_80, self.labels = self.load_json()
        self.data_aug = data_aug
        self.dataset_name = dataset_name

    def __getitem__(self, idx):
        seq_80 = np.array(self.data_80[idx])
        seq_label = np.array(self.labels[idx])
        if self.data_aug:
            # max_value = np.max(seq, axis=(0, 1))
            roll = np.random.rand(1)
            if roll < 0.5 :
                seq_80, seq_label = sample_filplr_80(seq_80, seq_label)
            roll2 = np.random.rand(1)
            if roll2 < 0.5:
                seq_80, seq_label = sample_up_filplr_80( seq_80, seq_label)
            seq_80, seq_label = sample_jitter_80( seq_80, seq_label)
            # seq_80, seq_label = sample_shift_80(seq_80, seq_label)

        import scipy.io as scio
        # scio.savemat('1.mat',
        #              {'new_data': seq_80,  'label': seq_label})

        #转换为torch

        seq_80 = torch.from_numpy(seq_80).type(torch.FloatTensor)

        # n x 3, n is the number of objects for each sequence
        labels = torch.from_numpy(seq_label).type(torch.FloatTensor).view(-1, 3)

        return seq_80, labels

    def __len__(self):
        return len(self.data_80)

    def load_json(self):
        if os.path.exists(self.label_root + '.npz'):
            label_ = np.load(self.label_root + '.npz', allow_pickle=True)['labels']
        else:
            label_ = concat_label(self.label_root)
        if os.path.exists(self.data_root + '.npz'):
            data_80 = np.load(self.data_root + '.npz', allow_pickle=True)['datas_80_fft']
        else:
            data_80 = concat_data(self.data_root)

        return data_80, label_

    def pull_seq(self, idx):
        """
        return m x 8192 np.array
        """
        return self.data_80[idx]

    def pull_anno(self, idx):
        """
        return  n x 3 np.array
         """
        labels = np.reshape(self.labels[idx], [-1, 3])
        return str(idx), labels


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

    imgs_80 = []
    for sample in batch:
        imgs_80.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs_80, 0), targets


if __name__ == '__main__':
    data_set = SignalDetectionv2('D:/USTC/20200913/20201114/G35_recorder_npydata/traindata_i10_1114_1',
                                 'D:/USTC/20200913/20201114/G35_recorder_npydata//trainlabels_i10_1114_1', data_aug=True)
    data_loader = data.DataLoader(data_set, 10,
                                  num_workers=1, shuffle=True,
                                  collate_fn=detection_collate, pin_memory=True)
    batch_iterator = iter(data_loader)
    sample80,target1= next(batch_iterator)
    # sample1_, sample2_,target2 = next(batch_iterator)
    sample80 = sample80.numpy()
    # target1 = target1.numpy()


