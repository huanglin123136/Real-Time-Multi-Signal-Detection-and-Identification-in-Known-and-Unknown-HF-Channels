# from .config import HOME
import os
import sys
import torch
import torch.utils.data as data
import scipy.io as sio
import numpy as np
# from data_augmentation import *
from .data_augmentation import *

def concat_data(path_):
    path = path_ + '_'
    # data_ = np.load(path +'1.npy')
    print(path_)
    path = os.path.abspath(path)
    data_ = np.load(path+'1.npz' , allow_pickle = True)['datas_10_fft']
    data_RC20 = np.load(path + '1.npz', allow_pickle=True)['datas_RC20']
    data_RC40 = np.load(path + '1.npz', allow_pickle=True)['datas_RC40']
    data_80 = np.load(path + '1.npz', allow_pickle=True)['datas_80_fft']
    i = 2
    while os.path.exists(path + str(i) + '.npz'):
        print('loading data '+path + str(i) + '.npz....')
        new_data = np.load(path + str(i) + '.npz', allow_pickle=True)['datas_10_fft']
        new_data_RC20 = np.load(path + str(i) + '.npz', allow_pickle=True)['datas_RC20']
        new_data_RC40 = np.load(path + str(i) + '.npz', allow_pickle=True)['datas_RC40']
        new_data_80 = np.load(path + str(i) + '.npz', allow_pickle=True)['datas_80_fft']

        data_ = np.concatenate((data_, new_data), axis=0)
        data_RC20 = np.concatenate((data_RC20, new_data_RC20), axis=0)
        data_RC40 = np.concatenate((data_RC40, new_data_RC40), axis=0)
        data_80 = np.concatenate((data_80, new_data_80), axis=0)
        i += 1
    return data_,data_80,data_RC20,data_RC40



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
    def __init__(self, data_root, data_aug=False, dataset_name='SignalDetedtion'):
        self.data_root = data_root
        self.data,self.data_80,self.dataRC20,self.dataRC40= self.load_json()
        self.data_aug = data_aug
        self.dataset_name = dataset_name

    def __getitem__(self, idx):
        seq = np.array(self.data[idx])
        seq_80 = np.array(self.data_80[idx])
        seq_RC20 = np.array(self.dataRC20[idx])
        seq_RC40 = np.array(self.dataRC40[idx])
        #转换为torch
        seq = torch.from_numpy(seq).type(torch.FloatTensor)
        seq_80 = torch.from_numpy(seq_80).type(torch.FloatTensor)
        seq_RC20 = torch.from_numpy(seq_RC20).type(torch.FloatTensor)
        seq_RC40 = torch.from_numpy(seq_RC40).type(torch.FloatTensor)

        return seq ,seq_80,seq_RC20, seq_RC40

    def __len__(self):
        return len(self.data)

    def load_json(self):
        if os.path.exists(self.data_root + '.npz'):
            data_ = np.load(self.data_root + '.npz', allow_pickle=True)['datas_10_fft']
            data_RC20 = np.load(self.data_root + '.npz', allow_pickle=True)['datas_RC20']
            data_RC40 = np.load(self.data_root + '.npz', allow_pickle=True)['datas_RC40']
            data_80 = np.load(self.data_root + '.npz', allow_pickle=True)['datas_80']
        else:
            data_,data_80,data_RC20,data_RC40 = concat_data(self.data_root)

        return data_,data_80,data_RC20,data_RC40

    def pull_seq(self, idx):
        """
        return m x 8192 np.array
        """
        return self.data[idx],self.data_80[idx],self.dataRC20[idx],self.dataRC40[idx]




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

    imgs = []
    imgs_80 = []
    imgs_RC20 = []
    imgs_RC40 = []
    for sample in batch:
        imgs.append(sample[0])
        imgs_80.append(sample[1])
        imgs_RC20.append(sample[2])
        imgs_RC40.append(sample[3])
    return torch.stack(imgs, 0), torch.stack(imgs_80, 0),torch.stack(imgs_RC20, 0), torch.stack(imgs_RC40, 0)


if __name__ == '__main__':
    data_set = SignalDetectionv2('D:/USTC/20200913/20201028/test_npy/testdata_i10_1028', False)
    data_loader = data.DataLoader(data_set, 128,
                                  num_workers=1, shuffle=False,
                                  collate_fn=detection_collate, pin_memory=True)
    batch_iterator = iter(data_loader)
    sample1, sample2 ,sample3,sample4 = next(batch_iterator)
    # sample1_, sample2_,target2 = next(batch_iterator)
    sample1= sample1.numpy()
    sample2 = sample2.numpy()
    sample3 = sample3.numpy()
    sample4 = sample4.numpy()
    # target1 = target1.numpy()

    import scipy.io as scio
    scio.savemat('fft_RC20_RC40.mat', {'sample1': sample1, 'sample2': sample2,'sample3': sample3})
    print("wait")

