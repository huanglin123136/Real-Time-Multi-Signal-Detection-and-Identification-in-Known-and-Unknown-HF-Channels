# from .config import HOME
import os
import sys
import torch
import torch.utils.data as data
import scipy.io as sio
import numpy as np
# from data_augmentation import *
from .data_augmentation_20210924 import *

def concat_data(path_):
    path = path_ + '_'
    # data_ = np.load(path +'1.npy')
    print(path_)
    path = os.path.abspath(path)
    data_1 = np.load(path + '1.npz', allow_pickle=True)['datas_fft1']
    data_2 = np.load(path + '1.npz', allow_pickle=True)['datas_fft2']
    data_3 = np.load(path + '1.npz', allow_pickle=True)['datas_fft3']
    data_4 = np.load(path + '1.npz', allow_pickle=True)['datas_fft4']
    data_5 = np.load(path + '1.npz', allow_pickle=True)['datas_fft5']
    i = 2
    while os.path.exists(path + str(i) + '.npz'):
        print('loading data '+path + str(i) + '.npz....')

        new_data = np.load(path + str(i) + '.npz', allow_pickle=True)['datas_fft1']
        new_data_2 = np.load(path + str(i) + '.npz', allow_pickle=True)['datas_fft2']
        new_data_3 = np.load(path + str(i) + '.npz', allow_pickle=True)['datas_fft3']
        new_data_4 = np.load(path + str(i) + '.npz', allow_pickle=True)['datas_fft4']
        new_data_5 = np.load(path + str(i) + '.npz', allow_pickle=True)['datas_fft5']
        data_1 = np.concatenate((data_1, new_data), axis=0)
        data_2 = np.concatenate((data_2, new_data_2), axis=0)
        data_3 = np.concatenate((data_3, new_data_3), axis=0)
        data_4 = np.concatenate((data_4, new_data_4), axis=0)
        data_5 = np.concatenate((data_5, new_data_5), axis=0)
        i += 1
    return data_1,data_2,data_3,data_4,data_5



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
        self.data,self.data_2, self.data_3, self.data_4,self.data_5,self.labels = self.load_json()
        self.data_aug = data_aug
        self.dataset_name = dataset_name

    def __getitem__(self, idx):
        seq = np.array(self.data[idx])
        seq_2 = np.array(self.data_2[idx])
        seq_3 = np.array(self.data_3[idx])
        seq_4 = np.array(self.data_4[idx])
        seq_5 = np.array(self.data_5[idx])
        seq_label = np.array(self.labels[idx])
        if self.data_aug:
            # max_value = np.max(seq, axis=(0, 1))
            roll = np.random.rand(1)
            if roll < 0.5:
                seq,seq_2,seq_3,seq_4,seq_5, seq_label = sample_filplr(seq,seq_2,seq_3,seq_4,seq_5, seq_label)
            roll = np.random.rand(1)
            if roll < 0.5:
                seq,seq_2,seq_3,seq_4,seq_5, seq_label = sample_up_filplr(seq,seq_2,seq_3,seq_4,seq_5, seq_label)
            roll = np.random.rand(1)
            if roll < 0.5:
                seq,seq_2,seq_3,seq_4,seq_5, seq_label= sample_jitter(seq,seq_2,seq_3,seq_4,seq_5, seq_label)
            roll = np.random.rand(1)
            if roll < 0.5:
                seq, seq_2, seq_3, seq_4, seq_5, seq_label = sample_noise(seq, seq_2, seq_3, seq_4, seq_5, seq_label)
            # seq,seq_80,seq_RC20,seq_RC40, seq_label = sample_shift(seq,seq_80,seq_RC20,seq_RC40, seq_label)

        import scipy.io as scio
        # scio.savemat('1.mat',
        #              {'new_data': seq, 'new_dataRC20': seq_RC20, 'new_dataRC40': seq_RC40, 'label': seq_label})

        #转换为torch
        seq = torch.from_numpy(seq).type(torch.FloatTensor)
        seq_2 = torch.from_numpy(seq_2).type(torch.FloatTensor)
        seq_3 = torch.from_numpy(seq_3).type(torch.FloatTensor)
        seq_4 = torch.from_numpy(seq_4).type(torch.FloatTensor)
        seq_5 = torch.from_numpy(seq_5).type(torch.FloatTensor)

        # n x 3, n is the number of objects for each sequence
        labels = torch.from_numpy(seq_label).type(torch.FloatTensor).view(-1, 3)

        return seq ,seq_2,seq_3 ,seq_4,seq_5, labels

    def __len__(self):
        return len(self.data)

    def load_json(self):
        if os.path.exists(self.label_root + '.npz'):
            label_ = np.load(self.label_root + '.npz', allow_pickle=True)['labels']
        else:
            label_ = concat_label(self.label_root)
        if os.path.exists(self.data_root + '.npz'):
            data_1 = np.load(self.data_root + '.npz', allow_pickle=True)['datas_fft1']
            data_2 = np.load(self.data_root + '.npz', allow_pickle=True)['datas_fft2']
            data_3 = np.load(self.data_root + '.npz', allow_pickle=True)['datas_fft3']
            data_4 = np.load(self.data_root + '.npz', allow_pickle=True)['datas_fft4']
            data_5 = np.load(self.data_root + '.npz', allow_pickle=True)['datas_fft5']
        else:
            data_1,data_2,data_3,data_4,data_5 = concat_data(self.data_root)

        return data_1,data_2,data_3,data_4,data_5, label_

    def pull_seq(self, idx):
        """
        return m x 8192 np.array
        """
        return self.data[idx],self.data_2[idx],self.data_3[idx],self.data_4[idx],self.data_5[idx]

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
    fft_1 = []
    fft_2 = []
    fft_3 = []
    fft_4 = []
    fft_5 = []
    for sample in batch:
        fft_1.append(sample[0])
        fft_2.append(sample[1])
        fft_3.append(sample[2])
        fft_4.append(sample[3])
        fft_5.append(sample[4])
        targets.append(torch.FloatTensor(sample[5]))
    return torch.stack(fft_1, 0), torch.stack(fft_2, 0),torch.stack(fft_3, 0),\
           torch.stack(fft_4, 0), torch.stack(fft_5, 0),targets


if __name__ == '__main__':
    data_set = SignalDetectionv2(
         'D:\\USTC\\G35_recorder_npydata\\traindata_0924_1',
         'D:\\USTC\\G35_recorder_npydata\\trainlabels_0924_1', True)
    data_loader = data.DataLoader(data_set, 10,
                                  num_workers=1, shuffle=True,
                                  collate_fn=detection_collate, pin_memory=True)
    batch_iterator = iter(data_loader)
    fft1, fft2,fft3,fft4,fft5,target1= next(batch_iterator)
    # sample1_, sample2_,target2 = next(batch_iterator)
    fft1= fft1.numpy()
    fft2 = fft2.numpy()
    target1 = target1.numpy()


