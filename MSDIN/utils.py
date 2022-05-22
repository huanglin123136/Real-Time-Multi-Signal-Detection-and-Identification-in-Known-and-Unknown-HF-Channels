from __future__ import print_function
import os
import imageio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def extract_signal(detection, threshold, scale=102.4):
    mask = torch.gt(detection[:, 0], threshold)
    detection_ = detection[mask].detach().cpu().numpy()
    if len(detection_) == 0:
        coords_ = []
        scores_ = []
    else:
        coords_ = [(x * scale, y * scale) for x, y in detection_[:, 1:]]
        scores_ = [x for x in detection_[:, 0]]
    return coords_, scores_


def path_valid(path):
    if not os.path.exists(path):
        os.mkdir(path)


def png2gif(source, dir_name, gifname, time):
    file_list = os.listdir(source)
    frames = []  # 读入缓冲区
    for png in file_list:
        png_ = os.path.join(source, png)
        frames.append(imageio.imread(png_))
    gifname = os.path.join(gifname, dir_name + '.gif')
    imageio.mimsave(gifname, frames, 'GIF', duration=time)


def gene_gifs(png_dir, gif_dir):
    path_valid(gif_dir)
    for dir_name in os.listdir(png_dir):
        path_ = os.path.join(png_dir, dir_name)
        png2gif(path_, dir_name, gif_dir, 0.1)


def txt_output(phase, coords, scores, threshold, labelmap, file_name, idx=0):
    pre_num = 0

    for i in range(len(coords)):
        if not scores[i]:
            continue
        j = 0
        while j < len(scores[i]) and scores[i][j] >= threshold:
            if phase == 'test' or phase == 'pred':
                if pre_num == 0:
                    with open(file_name, mode='a') as f:
                        f.write('Prediction: \n')
            score = scores[i][j]
            # print(i)
            # print(labelmap[i - 1])
            label_name = labelmap[i - 1]
            pt = coords[i][j]
            coordination = (pt[0] + idx * 90, pt[1] + idx * 90)
            pre_num += 1
            with open(file_name, mode='a') as f:
                # print(score)
                f.writelines(
                    str(pre_num) + ' label: ' + label_name + ' scores: ' + str(score) + ' ' + '||'.join(
                        str(round(c, 3)) for c in coordination) + '\n')
            j += 1


def plot_value(seqs, coords_, gap, value_):
    pt0 = max(0, int(coords_[0] / gap))
    pt1 = int(coords_[1] / gap)
    value = value_ * np.ones(pt1 - pt0 + 1)
    x = np.array([i * gap for i in range(pt0, pt1 + 1)])
    return x, value, pt0


def plot_spec(seqs, coords, scores, idx, labelmap, saved_fig_dir, order, inter=True):
    scale = 22
    seqs = seqs[order, :]
    gap = scale / len(seqs)
    plt.figure()
#    x = np.linspace(scale * idx + start, scale * (idx + 1)+ start, len(seqs + 1))
    if inter:
        begin = (idx % 64) * 1250/64
    else:
        begin = (idx % 64) * scale
    x = np.linspace(0, scale, len(seqs + 1))
    plt.plot(x + begin, seqs, c='blue', linewidth=0.5)

    plot_color = ['brown', 'red', 'green', 'black', 'yellow',
                  'darkred','gray','darkgoldenrod','deeppink','limegreen',
                  'purple','tomato','palegoldenrod','peru','powderblue',
                  'rosybrown','salmon','blue','red','black',
                  'yellow','green','darkred','tomato','gray',
                  'darkgoldenrod','peru']
    plot_place = [-0.5,-0.55,-0.6,-0.65,-0.7,
                  -0.75,-0.8,-0.85,-0.9,-0.95,
                  -0.2,-0.25,-0.3,-0.35,-0.4,
                  -0.45,-0.15,-0.1,-0.5,-0.05,
                  0.55,0.5]
    for i in range(len(coords)):
        coord = coords[i]
        if not coord:
            continue
        for j in range(len(coord)):
            x_, value_, pt0_ = plot_value(seqs, coord[j], gap, plot_place[i])
            if i < len(coords) - 1:
                plt.plot(x_ + begin, value_, label=labelmap[i - 1], color=plot_color[i], linewidth=1) if j == 0 else plt.plot(x_ + begin, value_, color=plot_color[i], linewidth=1)
                plt.annotate(labelmap[i-1]+':'+str('%.2f' % scores[i][j]), xy=(pt0_ * scale / len(seqs) + begin, max(value_)),
                             color='green', fontsize=7)
            else:
                plt.plot(x_ + begin, value_, color=plot_color[i], linewidth=1)
                plt.annotate(labelmap[int(scores[i][j])], xy=(pt0_ * scale / len(seqs) + begin,
                                                              max(value_)), color='green', fontsize=7)

    # plt.legend()
    plt.xlabel('kHz')
    plt.ylabel('Spec')
    plt.title('prediction for seq {}'.format(idx))
    plt.xlim(0 + begin, scale + begin)
    plt.ylim(-1, 1)

    saved_fig_dir_ = saved_fig_dir + '/{}'.format(idx)
    path_valid(saved_fig_dir_)
    plt.savefig(saved_fig_dir_ + '/seq_{}.png'.format(order),
                format='png', transparent=True, dpi=300, pad_inches=0)