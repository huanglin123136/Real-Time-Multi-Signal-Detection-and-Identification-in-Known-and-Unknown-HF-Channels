# from .config import HOME
import os
import sys
import torch
import torch.utils.data as data
# import scipy.io as sio
import numpy as np
from data.data_augmentation import *


class SignalDetectionv4(data.Dataset):
    def __init__(self, data_root, label_root, data_aug=False, dataset_name='SignalDetedtion'):
        self.data_root = data_root
        self.label_root = label_root
        self.data_0, self.labels_0, self.data_1, self.labels_1, self.data_2, self.labels_2 = self.load_json()
        self.len_am = len(self.labels_0)
        self.len_ssb = len(self.labels_1)
        self.len_psk = len(self.labels_2)
        self.data_aug = data_aug
        self.dataset_name = dataset_name

    def __getitem__(self, idx):
        if idx < self.len_am:
            getdata = self.data_0
            getlabel = self.labels_0
        elif idx < self.len_am + self.len_ssb:
            getdata = self.data_1
            getlabel = self.labels_1
            idx -= self.len_am
        else:
            getdata = self.data_2
            getlabel = self.labels_2
            idx -= (self.len_am + self.len_ssb)
        seq = np.array(getdata[idx])
        seq_label = np.array(getlabel[idx])
        if self.data_aug:
            # max_value = np.max(seq, axis=(0, 1))
            roll = np.random.rand(1)
            if roll < 0.5:
                seq, seq_label = sample_filplr(seq, seq_label)
            # roll = np.random.rand(1)
            # if roll < 0.6:
            #     add_noise_sqr = np.random.randn(1)
            #     seq = seq + add_noise_sqr * np.random.uniform(-0.02, 0.02)
            seq, seq_label = sample_jitter(seq, seq_label)
            seq, seq_label = sample_shift(seq, seq_label)

        # m x 8192
        seq = torch.from_numpy(seq).type(torch.FloatTensor)
        # n x 3, n is the number of objects for each sequence
        labels = torch.from_numpy(seq_label).type(torch.FloatTensor).view(-1, 3)
        return seq, labels

    def __len__(self):
        return len(self.labels_0) + len(self.labels_1) + len(self.labels_2)

    def load_json(self):
        # label_am = np.load(self.label_root + '_am.npy', allow_pickle=True)
        # label_ssb = np.load(self.label_root + '_ssb.npy', allow_pickle=True)
        # # label_psk = np.load(self.label_root + '_psk.npy', allow_pickle=True)
        # if os.path.exists(self.data_root + '_psk.npy'):
        #     label_psk = np.load(self.label_root + '_psk.npy', allow_pickle=True)
        # else:
        #     label_psk = concat_data(self.label_root, 'psk')
        # if os.path.exists(self.data_root + '_am.npy'):
        #     data_am = np.load(self.data_root + '_am.npy', allow_pickle=True)
        #     data_ssb = np.load(self.data_root + '_ssb.npy', allow_pickle=True)
        #     data_psk = np.load(self.data_root + '_psk.npy', allow_pickle=True)
        # else:
        #     data_am = concat_data(self.data_root, 'am')
        #     data_ssb = concat_data(self.data_root, 'ssb')
        #     data_psk = concat_data(self.data_root, 'psk')

        data_02 = np.load(self.data_root + '_-2.npy', allow_pickle=True)
        labels_02 = np.load(self.label_root + '_-2.npy', allow_pickle=True)
        data_01 = np.load(self.data_root + '_-1.npy', allow_pickle=True)
        labels_01 = np.load(self.label_root + '_-1.npy', allow_pickle=True)

        data_0 = np.load(self.data_root + '_0.npy', allow_pickle=True)
        labels_0 = np.load(self.label_root + '_0.npy', allow_pickle=True)

        labels_02_list = labels_02.tolist()
        labels_0_list = labels_0.tolist()
        data_0 = np.concatenate((data_02, data_0), axis=0)
        labels_list_0 = labels_02_list + labels_0_list
        labels_0 = np.array(labels_list_0)

        data_1 = np.load(self.data_root + '_1.npy', allow_pickle=True)
        labels_1 = np.load(self.label_root + '_1.npy', allow_pickle=True)

        labels_01_list = labels_01.tolist()
        labels_1_list = labels_1.tolist()
        data_1 = np.concatenate((data_01, data_1), axis=0)
        labels_list_1 = labels_01_list + labels_1_list
        labels_1 = np.array(labels_list_1)

        data_2 = np.load(self.data_root + '_2.npy', allow_pickle=True)
        labels_2 = np.load(self.label_root + '_2.npy', allow_pickle=True)
        data_3 = np.load(self.data_root + '_3.npy', allow_pickle=True)
        labels_3 = np.load(self.label_root + '_3.npy', allow_pickle=True)
        labels_2_list = labels_2.tolist()
        labels_3_list = labels_3.tolist()
        data_2 = np.concatenate((data_2, data_3), axis=0)
        labels_list = labels_2_list + labels_3_list[0]
        labels_list = labels_list[:-1]
        labels_2 = np.array(labels_list)
        # 空list无法保存为npy文件，所以加了一个无用值-1，这里去除
        # labels_2 = labels_2[:-1]
        # data_4 = np.load(self.data_root + '_4.npy', allow_pickle=True)
        # labels_4 = np.load(self.label_root + '_4.npy', allow_pickle=True)
        # data_2 = np.concatenate((data_2, data_3, data_4), axis=0)
        # labels_2 = np.concatenate((labels_2, labels_3, labels_4), axis=0)
        return data_0, labels_0, data_1, labels_1, data_2, labels_2

    def pull_seq(self, idx):
        """
        return m x 8192 np.array
        """
        # return self.data[idx]

        if idx < self.len_am:
            getdata = self.data_0
        elif idx < self.len_am + self.len_ssb:
            getdata = self.data_1
            idx -= self.len_am
        else:
            getdata = self.data_2
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
            getlabel = self.labels_0
        elif idx < self.len_am + self.len_ssb:
            getlabel = self.labels_1
            idx -= self.len_am
        else:
            getlabel = self.labels_2
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
    data_set = SignalDetectionv4('../../DataTransform/0408_3/traindata_i10_0408', '../../DataTransform/0408_3/trainlabels_i10_0408', False)
    data_loader = data.DataLoader(data_set, 4,
                                  num_workers=0, shuffle=True,
                                  collate_fn=detection_collate, pin_memory=True)
    batch_iterator = iter(data_loader)
    sample1, sample2 = next(batch_iterator)
    sample1_, sample2_ = next(batch_iterator)

