from __future__ import print_function
from tqdm import tqdm
import torch
from torch.autograd import Variable
from data.SignalDetectionv2 import SignalDetectionv2
from ResNet_1D import build_SSD
from Sample_match import match_sample
from utils import *
from get_gif import make_gif
import time
import warnings

warnings.filterwarnings("ignore")

# signal_name =  ['AM', 'SSB', 'BPSK','QPSK','QAM','2FSK','8FSK','CW']
signal_check = ['SingleNoise', 'AM', 'CFSK', 'CW_fast',
                'SingleSound', 'AM_expand_large', 'AM_expand_small',
                'SSB', 'Interp_flash', 'CW', 'Saopin', '2FSK',
                'PSKSound', 'Interp', 'Noise', 'Unknow', 'Interp_flash_2',
                'PSK', 'Unknow_DD', 'Interp_small', 'Saopin_large']

def path_valid(path):
    if not os.path.exists(path):
        os.mkdir(path)


def run(phase, cfg, net, data_set, threshold, txt=True, png=False):
    if png:
        matplotlib.use('Agg')
    file_name = cfg['saved_folder'] + cfg['txt_name']
    num_seqs = len(data_set)
    scale_ = 22
    # plot_idx = [num_seqs - 1 - x for x in range(100)]
    # for idx in tqdm(range(0, num_seqs, 10)):
    for idx in tqdm(range(0,num_seqs)):

        seq1,seq_80,seq2,seq3 = data_set.pull_seq(idx)

        if phase == 'test':
            seq_id, annos = data_set.pull_anno(idx)
            annos[:, :-1] *= scale_
            if txt:
                with open(file_name, mode='a') as f:
                    f.write('\n Ground Truth for seq ' + seq_id + '\n')
                    for box in annos:
                        f.write('label: ' + '||'.join(("%.2f" % b) for b in box) + ' :'+ cfg['labelmap'][int(box[2])] + '\n')

        x1 = torch.from_numpy(seq1).type(torch.FloatTensor)
        x1 = Variable(x1.unsqueeze(0),requires_grad=False)
        x2 = torch.from_numpy(seq2).type(torch.FloatTensor)
        x2 = Variable(x2.unsqueeze(0),requires_grad=False)

        x3 = torch.from_numpy(seq3).type(torch.FloatTensor)
        x3 = Variable(x3.unsqueeze(0),requires_grad=False)


        x4 = torch.from_numpy(seq_80).type(torch.FloatTensor)
        x4 = Variable(x4.unsqueeze(0),requires_grad=False)
        # x1 = torch.ones((256,10,1760))
        # x2 = torch.ones((256, 1, 3520))
        # x3 = torch.ones((256, 1, 7040))

        if cfg['using_gpu']:
            x1 = x1.cuda()
            x2 = x2.cuda()
            x3 = x3.cuda()
            x4 = x4.cuda()
        # forward pass

        # t0 = time.time()
        net.requires_grad_= False
        y = net(x1,x2,x3,x4)

        detection = y.data

        # print(time.time() - t0, "seconds wall time")

        coords = []
        scores = []
        co = []
        sc = []
        cl = []
        for i in range(detection.size(1)):
            coord, score = extract_signal(detection[0, i, :, :], threshold, scale=scale_)
            coords.append(coord)
            co += coord
            scores.append(score)
            sc += score
            cl += [i - 1] * len(coord)
        if txt:
            txt_output(phase, coords, scores, threshold, cfg['labelmap'], file_name)
        if phase == 'test':
            c = []
            s = []
            for i in range(annos.shape[0]):
                c.append((annos[i, 0], annos[i, 1]))
                s.append(annos[i, 2])
            coords.append(c)
            scores.append(s)
        if png:
            path_valid(cfg['saved_fig_dir'])
            if not match_sample(annos, co, cl, sc):
                # continue
                path_valid(cfg['saved_fig_dir'] + '/wrong/')
                for i in range(0):
                    plt.clf()
                    plot_spec(seq1, coords, scores, idx, cfg['labelmap'], cfg['saved_fig_dir'] + '/wrong/', i)
            else:
                path_valid(cfg['saved_fig_dir'] + '/right/')
                for i in range(0):
                    plot_spec(seq1, coords, scores, idx, cfg['labelmap'], cfg['saved_fig_dir'] + '/right/', i)


def my_test_signals(cfg):
    # num_classes = 4
    net = build_SSD('test', cfg['min_dim'], cfg['num_classes'], cfg)
    net.load_state_dict(torch.load(cfg['trained_model'], map_location='cpu'))
    net.eval()
    print('model loaded!')
    # load data
    testset = SignalDetectionv2(cfg['test_data_root'], cfg['test_label_root'], False)
    if cfg['using_gpu']:
        net.cuda()
    run('test', cfg, net, testset, cfg['visual_threshold'], True, False)


if __name__ == '__main__':

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
    my_test_signals(settings)
    gif_path = 'D:/USTC/20200913/20201112/train_25KHz_simple/gifs/'
    # path_valid(gif_path)
    # path_valid(gif_path + '/wrong/')
    # path_valid(gif_path + '/right/')
    # make_gif(settings['saved_fig_dir'] + '/wrong/', gif_path + '/wrong/')
    # make_gif(settings['saved_fig_dir'] + '/right/', gif_path + '/right/')
