import numpy as np
import torch
# from data_plot import draw_plot


def sample_filplr_80(data_80, label):

    data_80 = data_80[:, ::-1]
    label = label.reshape(-1, 3)
    len = label.shape[0]
    for i in range(len):
        label[:, i * 3: i * 3 + 2] = 1.0 - label[:, i * 3: i * 3 + 2]
        label[:, i * 3: i * 3 + 2] = label[:, i * 3: i * 3 + 2][:, ::-1]

    return data_80.copy(), label

def sample_up_filplr_80(data_80, label):
    data_80 = data_80[::-1, :]
    return data_80.copy(), label

def sample_shift_80(data_80,label):
    label = label.reshape(-1, 3)
    len = label.shape[0]
    sample_len = data_80.shape[1]
    min_value = 1
    for i in range(len):
       min_value = min(min_value, label[i][0])
    shift_ = np.random.uniform(0, 1, 1) * max(min_value, 0)

    sample_len80 = data_80.shape[1]
    shift80 = int(shift_ * sample_len80)
    new_data80 = data_80.copy()
    new_data80[:, :(sample_len80 - shift80)] = data_80[:, shift80:]
    new_data80[:, sample_len80 - shift80:] = data_80[:, :shift80]

    for i in range(len):
        label[:, i * 3: i * 3 + 2] = label[:, i * 3: i * 3 + 2] - shift_

    return new_data80,label


def sample_jitter_80(data_80, label):

    factor80 = np.random.uniform(0, 1, 1) * 0.2 + 0.9
    data_80 = data_80 * factor80

    return data_80, label
