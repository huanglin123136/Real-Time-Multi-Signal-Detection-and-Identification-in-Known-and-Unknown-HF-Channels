# from .config import HOME
import os
import sys
import torch
import torch.utils.data as data
import scipy.io as sio
import numpy as np
from .data_augmentation import *


def concat_data(path_):
    path = path_ + '_'
    # data_ = np.load(path +'1.npy')
    print(path_)
    path=os.path.abspath(path)
    data_ = np.load(path+'1.npy',allow_pickle = True)
    i = 2
    while os.path.exists(path + str(i) + '.npy'):
        print('loading data '+path + str(i) + '.npy....')
        new_data = np.load(path + str(i) + '.npy', allow_pickle=True)
        data_ = np.concatenate((data_, new_data), axis=0)
        i += 1
    return data_


class SignalDetectionv2(data.Dataset):
    def __init__(self, data_root, label_root, data_aug=False, dataset_name='SignalDetedtion'):
        self.data_root = data_root
        self.label_root = label_root
        self.data, self.labels = self.load_json()
        self.data_aug = data_aug
        self.dataset_name = dataset_name

    def __getitem__(self, idx):
        seq = np.array(self.data[idx])
        seq_label = np.array(self.labels[idx])
        if self.data_aug:
            # max_value = np.max(seq, axis=(0, 1))
            roll = np.random.rand(1)
            if roll < 0.5:
                seq, seq_label = sample_filplr(seq, seq_label)
            seq, seq_label = sample_jitter(seq, seq_label)
            seq, seq_label = sample_shift(seq, seq_label)

        seq_1 = seq[0, : ].reshape((1,8192))
        seq_2 = seq[1, : ].reshape((1,8192))


        # m x 8192
        seq_1 = torch.from_numpy(seq_1).type(torch.FloatTensor)
        seq_2 = torch.from_numpy(seq_2).type(torch.FloatTensor)
        # n x 3, n is the number of objects for each sequence
        labels = torch.from_numpy(seq_label).type(torch.FloatTensor).view(-1, 3)
        return (seq_1 , seq_2), labels

    def __len__(self):
        return len(self.data)

    def load_json(self):
        # label_ = np.load(self.label_root + '.npy', allow_pickle=True)
        if os.path.exists(self.label_root + '.npy'):
            label_ = np.load(self.label_root + '.npy', allow_pickle=True)
        else:
            label_ = concat_data(self.label_root)
        if os.path.exists(self.data_root + '.npy'):
            data_ = np.load(self.data_root + '.npy', allow_pickle=True)
        else:
            data_ = concat_data(self.data_root)

        return data_, label_

    def pull_seq(self, idx):
        """
        return m x 8192 np.array
        """
        return self.data[idx]

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
    imgs_1 = []
    imgs_2 = []
    for sample in batch:
        imgs_1.append(sample[0][0])
        imgs_2.append(sample[0][1])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs_1, 0),torch.stack(imgs_2, 0), targets


if __name__ == '__main__':
    data_set = SignalDetectionv2('J:/npydata_-5_5dB/testdata_i2_0728_10', 'J:/npydata_-5_5dB/testlabels_i2_0728_10', False)
    data_loader = data.DataLoader(data_set, 4,
                                  num_workers=1, shuffle=True,
                                  collate_fn=detection_collate, pin_memory=True)
    batch_iterator = iter(data_loader)
    seq_1 ,seq_2, labels = next(batch_iterator)
    seq_1 ,seq_2, labels= next(batch_iterator)

