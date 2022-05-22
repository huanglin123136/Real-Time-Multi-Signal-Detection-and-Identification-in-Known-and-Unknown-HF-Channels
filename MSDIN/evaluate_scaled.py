import numpy as np
import torch
from torch.autograd import Variable
import os
from tqdm import tqdm
# from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from data import *
from data.SignalDetectionv2 import SignalDetectionv2
from ResNet_1D import build_SSD
from detect_txt.get_confuse_matrix import main

import scipy.io as scio

def get_signal_num(path, num_):
    data = np.load(path, allow_pickle=True)['labels']
    # num_ = [0] * 3
    for i in range(len(data)):
        l = len(data[i])
        for j in range(l // 3):
            num_[data[i][j * 3 + 2]] += 1
    return num_


def my_test_net(save_folder, txtname, net, cuda, testset, labelmap, threshold):
    filename = save_folder + txtname
    num_seqs = len(testset)
    for idx in tqdm(range(num_seqs)):
        # print('Testing seqs: {:d}/{:d} ... '.format(i+1, num_seqs))
        seq = testset.pull_seq(idx)
        seq_id, annotation = testset.pull_anno(idx)

        x = torch.from_numpy(seq).type(torch.FloatTensor)
        x = Variable(x.unsqueeze(0))

        with open(filename, mode='a') as f:
            f.write('\nGround Truth for seq ' + seq_id + '\n')
            for box in annotation:
                box[:2] *= 20
                f.write('label: ' + '||'.join(str(b) for b in box) + '\n')
        if cuda:
            x = x.cuda()
        # forward pass
        y = net(x)
        detection = y.data
        # scale each detection back up to the seq
        scale = torch.Tensor([20, 20])

        pre_num = 0
        for i in range(detection.size(1)):
            j = 0
            while j < detection.shape[2] and detection[0, i, j, 0] >= threshold:
                if pre_num == 0:
                    with open(filename, mode='a') as f:
                        f.write('Prediction: ' + '\n')
                score = detection[0, i, j, 0].cpu().numpy()
                label_name = labelmap[i - 1]
                pt = (detection[0, i, j, 1:] * scale).cpu().numpy()
                coordination = (pt[0], pt[1])
                pre_num += 1
                with open(filename, mode='a') as f:
                    # print(score)
                    f.writelines(str(pre_num) + ' label: ' + label_name + ' scores: ' + str(score) + ' ' + '||'.join(
                        str(c) for c in coordination) + '\n')
                j += 1


def my_test_signals(test_cfg):
    # num_classes = 3
    net = build_SSD('test', cfg['min_dim'], cfg['num_classes'], cfg)
    net.load_state_dict(torch.load(test_cfg['trained_model']))
    net.eval()
    print('model loaded!')
    # load data
    testset = SignalDetectionv2(test_cfg['test_data_root'], test_cfg['test_label_root'], False)
    if cfg['using_gpu']:
        net.cuda()

    # evaluation
    my_test_net(test_cfg['saved_folder'], test_cfg['txt_name'], net, test_cfg['using_gpu'], testset,
             test_cfg['labelmap'], test_cfg['visual_threshold'])


def voc_ap(rec, prec):
    rec = rec[::-1]
    prec = prec[::-1]
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluation_loop( signal_num=(1163, 176, 92), th_iou=0.3):
    # (2971, 520, 448)
    # settings = {'feature_maps': [256, 128, 64, 32, 16], 'min_dim': 8192, 'steps': [32, 64, 128, 256, 512],
    #             'num_filters': [64, 128, 128, 256, 256, 512, 512],'num_scales': [16, 16, 12, 8, 8],
    #             'min_size': [164, 328, 656, 820, 1228], 'num_classes': 4, 'input_channels': 10,
    #             'max_size': [328, 656, 820, 1228, 1638], 'variance': [0.1, 0.2], 'clip': True, 'name': 'Signals',
    #             'trained_model': './weights/ResNet_i1_8192_20191130_30000.pth', 'using_gpu': False,
    #             'test_data_root': './data/test_data_1115.npy', 'test_label_root': './data/test_label_1115.npy',
    #             'saved_folder': './test/', 'txt_name': 'test_0503.txt', 'labelmap': ['AM', 'SSB', 'PSK'],
    #             'visual_threshold': 0.1}
    # test_signals(code, test_settings)
    settings = {
                'min_dim': 800,
                'lr_steps': (1000, 3000, 5000, 7000, 10000, 15000, 25000, 35000, 45000),
                'max_iter': 101,
                'variance': [0.1, 0.2],
                'num_classes': 8 + 1,
                'min_dim': 800,
                'feature_maps': [25, 13, 7, 4],
                'steps': [32, 64, 128, 256],
                'num_filters': [128, 128, 256, 256, 512, 512],
                'input_channels': 10,
                'num_scales': [6, 9, 12, 15],
                'min_size': [24, 48, 96, 192],
                'max_size': [48, 96, 192, 384],
                'variance': [0.1, 0.2],
                'clip': True,
                'name': 'Signals',
                # 'trained_model': 'G:/train_simulation_data_0830_C20/fft_weights/ResNet_i10_8192_20200831_100.pth',
                'trained_model':  '/home/linhuang/download/train_simulation_data_0830_C20_fftpca_fft_all/fft_weights/ResNet_i10_800_20200831_99600.pth',
                'using_gpu': True,

                # 'test_data_root': 'J:/npydata_0830_10KHz/traindata_0830_1',
                # 'test_label_root': 'J:/npydata_0830_10KHz/trainlabels_0830_1',
                # 'saved_folder': "./fft_weights/",
                # 'saved_fig_dir': "./fft_weights/",
                # 'txt_name': 'test_fft.txt',

                'test_data_root': '/home/linhuang/download/npydata_10KHz_simulation/testdata_0830_1',
                'test_label_root': '/home/linhuang/download/npydata_10KHz_simulation/testlabels_0830_1',
                'saved_folder': '/home/linhuang/download/train_simulation_data_0830_C20_fftpca_fft_all/fft_weights/',
                'txt_name': 'test_fft_1.txt',
                'labelmap': ['AM', 'SSB', 'BPSK','QPSK','QAM','2FSK','8FSK','CW'],
                'visual_threshold': 0.1}
    res ,confuse_matrix= main(settings['saved_folder'] + settings['txt_name'], th_iou, signal_num)
    # dataNew= './fft_weights/res_0831.mat'
    dataNew= '/home/linhuang/download/train_simulation_data_0830_C20_fftpca_fft_all/fft_weights/res_0831.mat'
    scio.savemat(dataNew, {'resmat': res,'confuse_matrix':confuse_matrix,'signal_num':signal_num})
    # np.savetxt('/home/linhuang/download/train_simulation_data_0830/fft_weights/res_0831.mat', res[:, 0, :], delimiter=',')
    # np.savetxt('/home/linhuang/download/train_simple/fft_weights/ssb_0820_3.csv', res[:, 1, :], delimiter=',')
    # np.savetxt('/home/linhuang/download/train_simple/fft_weights/psk_0820_3.csv', res[:, 2, :], delimiter=',')

    # np.savetxt('/home/linhuang/download/datatrain_scaled/New_train/csv/am_0805.csv', res[:, 0, :], delimiter=',')
    # np.savetxt('/home/linhuang/download/datatrain_scaled/New_train/csv/ssb_0805.csv', res[:, 1, :], delimiter=',')
    # np.savetxt('/home/linhuang/download/datatrain_scaled/New_train/csv/psk_0805.csv', res[:, 2, :], delimiter=',')
    mAP = 0
    for i in range(3):
        AP = voc_ap(res[:, i, 0], res[:, i, 1])
        mAP += AP
    return mAP


def plot_PR(data, name='AM'):
    x = np.array(data[:, 0])
    y = np.array(data[:, 1])
    # plt.scatter(x, y, s=100)
    plt.plot(x, y, 'b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title('PR curve for ' + name + ' signal')
    plt.show()


if __name__ == '__main__':
    signal_num = [0] * 8
    # path = 'J:/npydata_10KHz_simulation/trainlabels_0830_1.npz'
    path = '/home/linhuang/download/npydata_10KHz_simulation/testlabels_0830_1.npz'

    # path = '/home/linhuang/download/Datatrain/npydata_-5_5dB/testlabels_i2_0728_10.npy'
    # path = '/home/linhuang/download/Datatrain/npydata0810_fft/recorder/testlabels_i10_0810_3.npy'
    get_signal_num(path , signal_num)

    #tl
    # get_signal_num('F:/CSPL/DataTransform/0526/testlabels_i10_0526.npy', signal_num)
    print(signal_num)
    # signal_num = [1383, 101, 49]
    evaluation_loop(signal_num)

