# from .config import HOME
import os
import sys
import torch
import torch.utils.data as data
# import scipy.io as sio
import numpy as np
from data.data_augmentation import *


def concat_data(path_, signal):
    path = path_ + '_' + signal + '_'
    data_ = np.load(path +'0.npy')
    i = 1
    while os.path.exists(path + str(i) + '.npy'):
        new_data = np.load(path + str(i) + '.npy', allow_pickle=True)
        data_ = np.concatenate((data_, new_data), axis=0)
        i += 1
    return data_


class SignalDetectionv3(data.Dataset):
    def __init__(self, data_root, label_root, data_aug=False, dataset_name='SignalDetedtion'):
        self.data_root = data_root
        self.label_root = label_root
        self.am_data, self.am_labels, self.ssb_data, self.ssb_labels, self.psk_data, self.psk_labels = self.load_json()
        self.len_am = len(self.am_data)
        self.len_ssb = len(self.ssb_data)
        self.len_psk = len(self.psk_data)
        self.data_aug = data_aug
        self.dataset_name = dataset_name

    def __getitem__(self, idx):
        if idx < self.len_am:
            getdata = self.am_data
            getlabel = self.am_labels
        elif idx < self.len_am + self.len_ssb:
            getdata = self.ssb_data
            getlabel = self.ssb_labels
            idx -= self.len_am
        else:
            getdata = self.psk_data
            getlabel = self.psk_labels
            idx -= (self.len_am + self.len_ssb)
        seq = np.array(getdata[idx])
        seq_label = np.array(getlabel[idx])
        if self.data_aug:
            # max_value = np.max(seq, axis=(0, 1))
            roll = np.random.rand(1)
            if roll < 0.5:
                seq, seq_label = sample_filplr(seq, seq_label)
            seq, seq_label = sample_jitter(seq, seq_label)
            seq, seq_label = sample_shift(seq, seq_label)

        # m x 8192
        seq = torch.from_numpy(seq).type(torch.FloatTensor)
        # n x 3, n is the number of objects for each sequence
        labels = torch.from_numpy(seq_label).type(torch.FloatTensor).view(-1, 3)
        return seq, labels

    def __len__(self):
        return len(self.am_data) + len(self.ssb_data) + len(self.psk_data)

    def load_json(self):
        label_am = np.load(self.label_root + '_am.npy', allow_pickle=True)
        label_ssb = np.load(self.label_root + '_ssb.npy', allow_pickle=True)
        # label_psk = np.load(self.label_root + '_psk.npy', allow_pickle=True)
        if os.path.exists(self.data_root + '_psk.npy'):
            label_psk = np.load(self.label_root + '_psk.npy', allow_pickle=True)
        else:
            label_psk = concat_data(self.label_root, 'psk')
        if os.path.exists(self.data_root + '_am.npy'):
            data_am = np.load(self.data_root + '_am.npy', allow_pickle=True)
            data_ssb = np.load(self.data_root + '_ssb.npy', allow_pickle=True)
            data_psk = np.load(self.data_root + '_psk.npy', allow_pickle=True)
        else:
            data_am = concat_data(self.data_root, 'am')
            data_ssb = concat_data(self.data_root, 'ssb')
            data_psk = concat_data(self.data_root, 'psk')
        return data_am, label_am, data_ssb, label_ssb, data_psk, label_psk

    def pull_seq(self, idx):
        """
        return m x 8192 np.array
        """
        # return self.data[idx]

        if idx < self.len_am:
            getdata = self.am_data
        elif idx < self.len_am + self.len_ssb:
            getdata = self.ssb_data
            idx -= self.len_am
        else:
            getdata = self.psk_data
            idx -= (self.len_am + self.len_ssb)
        return getdata[idx]

    def pull_anno(self, idx):
        """
        return  n x 3 np.array
         """
        no = idx
        # labels = np.reshape(self.labels[idx], [-1, 3])
        # return str(idx), labels
        if idx < self.len_am:
            getlabel = self.am_labels
        elif idx < self.len_am + self.len_ssb:
            getlabel = self.ssb_labels
            idx -= self.len_am
        else:
            getlabel = self.psk_labels
            idx -= (self.len_am + self.len_ssb)
        seq_label = np.array(getlabel[idx])
        labels = np.reshape(seq_label, [-1, 3])
        return str(no), labels


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
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


if __name__ == '__main__':
    data_set = SignalDetectionv3('./testdata_i10_1217', './testlabel_i10_1217', False)
    data_loader = data.DataLoader(data_set, 4,
                                  num_workers=1, shuffle=True,
                                  collate_fn=detection_collate, pin_memory=True)
    batch_iterator = iter(data_loader)
    sample1, sample2 = next(batch_iterator)
    sample1_, sample2_ = next(batch_iterator)

