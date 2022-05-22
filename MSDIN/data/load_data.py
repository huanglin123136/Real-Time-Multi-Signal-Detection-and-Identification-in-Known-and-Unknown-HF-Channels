import os
import numpy as np
import torch
from data_plot import draw_plot, draw_plotv2

def sample_filplr(data, label):
    data = np.array(data)
    label = np.array(label)
    data = data[:,::-1]
    len = label.shape[0]
    for i in range(len // 3):
        label[i * 3: i * 3 + 2] = 1.0 - label[i * 3: i * 3 + 2]
        label[i * 3: i * 3 + 2] = label[i * 3: i * 3 + 2][::-1]
    return data, label

def sample_shift(data, label):
    data = np.array(data)
    label = np.array(label)
    label = label.reshape(-1, 3)
    len = label.shape[0]
    sample_len = data.shape[1]
    min_value = 1
    for i in range(len):
       min_value = min(min_value, label[i][0])
    shift_ = np.random.rand(1) * (min_value)
    shift = int(shift_ * sample_len)
    new_data = data.copy()
    new_data[:, :(sample_len - shift)] = data[:, shift:]
    new_data[:, sample_len - shift:] = data[:, :shift]
    for i in range(len):
        label[i * 3: i * 3 + 2] = label[i * 3: i * 3 + 2] - shift_

    return new_data, label

def sample_jitter(data, sample):
    factor = np.random.rand() * 0.2 + 0.9
    data = data * factor
    return data, sample

data_path = './data_1123.npy'
label_path = './labels_1123.npy'

data = np.load(data_path)
label = np.load(label_path)
# np.save('test_data_1.npy', data[8000:8080])
# np.save('test_label_1.npy', label[8000:8080])
index = 5600
length = 50
for i in range(index, index + length, 1):
    data = np.load(data_path)
    label = np.load(label_path)
    # np.save('test_data.npy', data[:10])
    # np.save('test_label.npy', label[:10])
    new_data = data[i]
    new_label = label[i]
    print(new_label)
    draw_plotv2(new_data, new_label, i)
# print(new_label)
# data1, label1 = sample_filplr(new_data, new_label)
# draw_plot(data1, label1)
# data2, label2 = sample_shift(new_data, new_label)
# draw_plot(data2, label2)
# data3, label3 = sample_jitter(new_data, new_label)
# draw_plot(data3, label3)

