import torch
import torch.utils.data as data
import scipy.io as sio
import numpy as np

class SignalPredictionv2(data.Dataset):
    """Signal Detection Dataset Object

    input is sequence, target is annotation

    Arguments:
        root (string): filepath to .mat folder.
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
    """
    def __init__(self, root, dataset_name='SignalPrediction'):

        self.root = root
        self.data = self.load_json()
        self.dataset_name = dataset_name

    def __getitem__(self, idx):
        seq = self.data[idx]
        # max_value = np.max(seq, axis=(0, 1))
        max_value = 1
        seq = torch.from_numpy(seq).type(torch.FloatTensor)  #1 x 8192
        return seq / max_value, idx

    def __len__(self):
        return len(self.data)

    def load_json(self):
        data = np.load(self.root)
        return data

    def pull_seq(self, idx):
        """
        return 1 x 8192 np.array
        """
        return self.data[idx]

if __name__ == '__main__':

    path = './testdata_startwith0.mat'
    testset = SignalPredictionv2(path)
    data_loader = data.DataLoader(testset, 1,
                                  num_workers=1, shuffle=False,
                                  pin_memory=True)
    batch_iterator = iter(data_loader)
    sample, index = next(batch_iterator)
    sample_ = next(batch_iterator)


