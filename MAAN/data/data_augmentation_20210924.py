import numpy as np
import torch
# from data_plot import draw_plot


def sample_filplr(data,data_2,data_3,data_4,data_5,label):
    data = data[:, ::-1]
    data_2 = data_2[:, ::-1]
    data_3 = data_3[:, ::-1]
    data_4 = data_4[:, ::-1]
    data_5 = data_5[:, ::-1]

    label = label.reshape(-1, 3)
    len = label.shape[0]
    for i in range(len):
        label[:, i * 3: i * 3 + 2] = 1.0 - label[:, i * 3: i * 3 + 2]
        label[:, i * 3: i * 3 + 2] = label[:, i * 3: i * 3 + 2][:, ::-1]

    return data.copy(),data_2.copy(), data_3.copy(),data_4.copy(),data_5.copy(),label


def sample_up_filplr(data,data_2,data_3,data_4,data_5,label):

    data = data[::-1,:]
    data_2 = data_2[::-1,:]
    data_3 = data_3[::-1, :]
    data_4 = data_4[::-1, :]
    data_5 = data_5[::-1, :]
    return data.copy(),data_2.copy(), data_3.copy(),data_4.copy(),data_5.copy(),label


def sample_jitter(data,data_2,data_3,data_4,data_5,label):
    factor = np.random.uniform(0, 1, 1) * 0.1 + 0.95
    add_items = np.random.uniform(0, 1, 1) * 0.05 - 0.025
    data = data * factor + add_items

    factor = np.random.uniform(0, 1, 1) * 0.1 + 0.95
    add_items = np.random.uniform(0, 1, 1) * 0.05 - 0.025
    data_2 = data_2 * factor + add_items

    factor = np.random.uniform(0, 1, 1) * 0.1 + 0.95
    add_items = np.random.uniform(0, 1, 1) * 0.05 - 0.025
    data_3 = data_3 * factor + add_items

    factor = np.random.uniform(0, 1, 1) * 0.1 + 0.95
    add_items = np.random.uniform(0, 1, 1) * 0.05 - 0.025
    data_4 = data_4 * factor + add_items

    factor = np.random.uniform(0, 1, 1) * 0.1 + 0.95
    add_items = np.random.uniform(0, 1, 1) * 0.05 - 0.025
    data_5 = data_5 * factor + add_items
    return data.copy(),data_2.copy(), data_3.copy(),data_4.copy(),data_5.copy(),label
