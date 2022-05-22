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


def get_signal_num(path_, num_):
    if os.path.exists(path_):
        data = np.load(path_, allow_pickle=True)['labels']
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
                box[:2] *= 22
                f.write('label: ' + '||'.join(str(b) for b in box) + '\n')
        if cuda:
            x = x.cuda()
        # forward pass
        y = net(x)
        detection = y.data
        # scale each detection back up to the seq
        scale = torch.Tensor([22, 22])

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


def evaluation_loop( signal_num=(1163, 176, 92), th_iou=0.2):
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
            'min_dim': 2000,
            'lr_steps': (1000, 3000, 5000, 7000, 10000, 15000, 25000, 35000, 45000),
            'max_iter': 50001,
            'variance': [0.1, 0.2],
            'num_classes': 30 + 1,
            'feature_maps': [32, 16, 8, 4, 2],
            'steps': [32, 64, 128, 256, 512],
            'num_filters': [128, 128, 128, 256, 256, 256, 512, 512],
            'input_channels': 1,
            'num_scales': [6, 9, 12, 12, 12],
            'min_size': [50, 100, 200, 400, 800],
            'max_size': [100, 200, 400, 800, 1600],
            'variance': [0.1, 0.2],
            'clip': True,
            'using_gpu': False,
            'name': 'Signals',

            'labelmap': ['AM', 'SSB', 'PSK', '2FSK', 'CW', 'Saopin',
                         'Interp', 'SingleSound', 'Amexpandlarge', 'Inflash', 'Unknow',
                         'Saolarge', 'Noise', 'Cfast',
                         'None1', 'None2', 'None3', 'None4', 'None5', 'None6', 'None7'],

            'trained_model': 'D:/USTC/20200913/20201112/train_25KHz_simple/RC20_RC40_fft_weights/ResNet_i1_1760_20201113_27200.pth',
            'test_data_root': 'D:/USTC/20200913/20201113/G35_recorder_npydata/traindata_i10_1112_2',
            'test_label_root': 'D:/USTC/20200913/20201113/G35_recorder_npydata/trainlabels_i10_1112_2',
            'saved_folder': 'D:/USTC/20200913/20201112/train_25KHz_simple/RC20_RC40_fft_weights/',
            'saved_fig_dir': 'D:/USTC/20200913/20201112/train_25KHz_simple/figs/',
            'txt_name': 'test_fft_2.txt',
            # 'labelmap': ['AM', 'SSB', 'BPSK','QPSK','QAM','2FSK','8FSK','CW'],
            'visual_threshold': 0.1,
            'name': 'Signals',
    }
    res ,confuse_matrix= main(settings['saved_folder'] + settings['txt_name'], th_iou, signal_num)
    dataNew= 'D:/USTC/20200913/20201112/train_25KHz_simple/RC20_RC40_fft_weights/res_1113_2.mat'
    # dataNew= '/home/linhuang/download/train_simulation_data_0830_C20_fftpca_fft_all/fft_weights/res_0831.mat'
    scio.savemat(dataNew, {'resmat': res,'confuse_matrix':confuse_matrix,'signal_num':signal_num})


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
    signal_num = [0] * (20)

    path ='D:/USTC/20200913/20201113/G35_recorder_npydata/trainlabels_i10_1112_2.npz'

    get_signal_num(path, signal_num)
    signal_check =  ['AM', 'SSB', 'PSK', '2FSK', 'CW', 'Saopin',
                'Interp','SingleSound','Amexpandlarge','Inflash','Unknow',
                'Saolarge', 'Noise', 'Cfast',
                'None1','None2','None3','None4','None5','None6','None7']
    #tl
    # get_signal_num('F:/CSPL/DataTransform/0526/testlabels_i10_0526.npy', signal_num)
    print(signal_num)
    # signal_num = [1383, 101, 49]
    evaluation_loop(signal_num)

