# from .config import HOME
import os
import sys
import torch
import torch.utils.data as data
import scipy.io as sio
import numpy as np


class SignalDetection(data.Dataset):
    """Signal Detection Dataset Object

    input is sequence, target is annotation

    Arguments:
        root (string): filepath to .mat folder.
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
    """
    def __init__(self, phase, root, dataset_name='SignalDetedtion'):
        self.train_test = phase
        self.root = root
        self.content = self.load_mat()
        self.data, self.labels = self.content
        self.dataset_name = dataset_name

    def __getitem__(self, idx):
        seq = self.data[idx]
#        max_value = np.max(seq, axis=(0, 1))
        seq = torch.from_numpy(seq).type(torch.FloatTensor)  #1 x 8192
        labels = self.labels[idx].astype(np.int)
        # for i in range(labels.shape[1] // 3):                # AM signals only
        #     labels[0][i * 3 + 2] = 0
        labels = torch.from_numpy(labels).type(torch.FloatTensor).view(-1, 3) # n x 3, n is the number of objects for each sequence
        length = torch.from_numpy(np.array([1e+05, 1e+05, 1])).type(torch.FloatTensor)
        return seq , labels / length

    def __len__(self):
        return len(self.data)

    def load_mat(self):
        data = sio.loadmat(self.root)
        if self.train_test == 'train':
            # sequences = data['test_data_split'][0]  # sample length x 8192?
            # labels = data['test_label_split'][0]  # sample length x 3n, n is the number of objects for each sequence?
            sequences = data['train_data_split'][0]                  # sample length x 8192
            labels = data['train_label_split'][0]                    #sample length x 3n, n is the number of objects for each sequence
        else:
            sequences = data['train_data_split'][0]  # sample length x 8192
            labels = data['train_label_split'][0]
            # sequences = data['test_data_split'][0]  # sample length x 8192
            # labels = data['test_label_split'][0]  # sample length x 3n, n is the number of objects for each sequence
        return sequences, labels

    def pull_seq(self, idx):
        """
        return 1 x 8192 np.array
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
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets

if __name__ == '__main__':
    data_set = SignalDetection('train', './targetDomain_train_0924.mat')
    data_loader = data.DataLoader(data_set, 4,
                                  num_workers=1, shuffle=True,
                                  collate_fn=detection_collate, pin_memory=True)
    batch_iterator = iter(data_loader)
    sample1, sample2 = next(batch_iterator)
    sample_ = next(batch_iterator)
    1
