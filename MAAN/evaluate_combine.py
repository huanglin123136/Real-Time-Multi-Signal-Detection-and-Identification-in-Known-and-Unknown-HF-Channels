import numpy as np
import torch
from torch.autograd import Variable
import os
from tqdm import tqdm
# from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from data import *
from data.SignalDetectionv3 import SignalDetectionv3
from data.SignalDetectionv4 import SignalDetectionv4
from ResNet_1D import build_SSD
from detect_txt.get_confuse_matrix import main


def get_signal_num(path, num_):
    data = np.load(path, allow_pickle=True)
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
                box[:2] *= 1.024e5
                f.write('label: ' + '||'.join(str(b) for b in box) + '\n')
        if cuda:
            x = x.cuda()
        # forward pass
        y = net(x)
        detection = y.data
        # scale each detection back up to the seq
        scale = torch.Tensor([1.024e5, 1.024e5])

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
    testset = SignalDetectionv3(test_cfg['test_data_root'], test_cfg['test_label_root'], False)
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
    settings = {'feature_maps': [256, 128, 64, 32, 16,8],
                'min_dim': 8192,
                'num_classes': 4, 'input_channels': 2,
                'steps': [32, 64, 128, 256, 512,1024],
                'num_filters': [64, 128, 128, 256, 256, 512, 512,512],
                'num_scales': [8, 8, 16, 16, 16,16],
                'min_size': [24, 48, 96, 192, 384, 768],
                'max_size': [48, 96, 192, 384, 768, 1536],
                'variance': [0.1, 0.2], 'clip': True, 'name': 'Signals',
                # 'trained_model': 'G:/height_enhance/DataTraining/weight_combine/ResNet_i1_8192_20200804_54900.pth',
                'trained_model': '/home/linhuang/download/height_enhance/New_train/simulation_weights_combine_new_scale/ResNet_i2_8192_20200730_54800.pth',
                'using_gpu': True,

                'test_data_root': 'J:/npydata_-5_5dB/testdata_i2_0728_11.npy',
                'test_label_root': 'J:/npydata_-5_5dB/testlabels_i2_0728_11.npy',
                'saved_folder': './',
                'txt_name': 'test_simulation_5dB_combine_10.txt',
                # 'test_data_root': '/home/linhuang/download/Datatrain/npydata_-5_5dB/testdata_i2_0728_10.npy',
                # 'test_label_root': '/home/linhuang/download/Datatrain/npydata_-5_5dB/testlabels_i2_0728_10.npy',
                # 'saved_folder': '/home/linhuang/download/height_enhance/Datatrain/test_simulation_5dB_combine/',
                # 'txt_name': 'test_simulation_5dB_combine_10.txt',

                'labelmap': ['AM', 'SSB', 'PSK'],
                'visual_threshold': 0.1}
    res = main(settings['saved_folder'] + settings['txt_name'], th_iou, signal_num)
    np.savetxt('./am_0730_5dB.csv', res[:, 0, :], delimiter=',')
    np.savetxt('./ssb_0730_5dB.csv', res[:, 1, :], delimiter=',')
    np.savetxt('./psk_0730_8192.csv', res[:, 2, :], delimiter=',')
    # np.savetxt('/home/linhuang/download/height_enhance/Datatrain/csv/am_0804_combine.csv', res[:, 0, :], delimiter=',')
    # np.savetxt('/home/linhuang/download/height_enhance/Datatrain/csv/ssb_0804_combine.csv', res[:, 1, :], delimiter=',')
    # np.savetxt('/home/linhuang/download/height_enhance/Datatrain/csv/psk_0804_combine.csv', res[:, 2, :], delimiter=',')
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
    signal_num = [0] * 3
    # path = 'data/testlabels_i10_0403'
    # path = '/home/linhuang/download/Datatrain/npydata_-5_5dB/testlabels_i2_0728_10.npy'
    path = 'J:/npydata_-5_5dB/testlabels_i2_0728_11.npy'

    get_signal_num(path, signal_num)

    #tl
    # get_signal_num('F:/CSPL/DataTransform/0526/testlabels_i10_0526.npy', signal_num)
    print(signal_num)
    # signal_num = [1383, 101, 49]
    evaluation_loop(signal_num)


