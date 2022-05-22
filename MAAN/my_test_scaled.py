from __future__ import print_function
from tqdm import tqdm
from torch.autograd import Variable
from data.SignalDetectionv5 import SignalDetectionv5
from data.SignalDetectionv2 import SignalDetectionv2
from ResNet_1D import build_SSD
from Sample_match import match_sample
from utils import *
from get_gif import make_gif
import warnings
warnings.filterwarnings("ignore")
signal_name =  ['AM', 'SSB', 'BPSK','QPSK','QAM','2FSK','8FSK','CW']


def path_valid(path):
    if not os.path.exists(path):
        os.mkdir(path)


def run(phase, cfg, net, data_set, threshold, txt=True, png=False):
    if png:
        matplotlib.use('Agg')
    file_name = cfg['saved_folder'] + cfg['txt_name']
    num_seqs = len(data_set)
    scale_ = 20
    # plot_idx = [num_seqs - 1 - x for x in range(100)]
    # for idx in tqdm(range(0, num_seqs, 10)):
    for idx in tqdm(range(0,num_seqs)):

        seq1,seq2,seq3 = data_set.pull_seq(idx)

        if phase == 'test':
            seq_id, annos = data_set.pull_anno(idx)
            annos[:, :-1] *= scale_
            if txt:
                with open(file_name, mode='a') as f:
                    f.write('\nGround Truth for seq ' + seq_id + '\n')
                    for box in annos:
                        f.write('label: ' + '||'.join(str(b) for b in box) + '\n')

        x1 = torch.from_numpy(seq1).type(torch.FloatTensor)
        x1 = Variable(x1.unsqueeze(0))
        x2 = torch.from_numpy(seq2).type(torch.FloatTensor)
        x2 = Variable(x2.unsqueeze(0))

        x3 = torch.from_numpy(seq3).type(torch.FloatTensor)
        x3 = Variable(x3.unsqueeze(0))
        if cfg['using_gpu']:
            x1 = x1.cuda()
            x2 = x2.cuda()
            x3 = x3.cuda()
        # forward pass
        y = net(x1,x2,x3)
        detection = y.data

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
                for i in range(0,cfg['input_channels'],4):
                    plt.clf()
                    plot_spec(seq1, coords, scores, idx, cfg['labelmap'], cfg['saved_fig_dir'] + '/wrong/', i)
            else:
                path_valid(cfg['saved_fig_dir'] + '/right/')
                for i in range(0,cfg['input_channels'],4):
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
        'min_dim': 800,
        'lr_steps': (1000, 3000, 5000, 7000, 10000, 15000, 25000, 35000, 45000),
        'max_iter': 101,
        'variance': [0.1, 0.2],
        'num_classes': 8 + 1,
        'min_dim': 800,
        'feature_maps': [25, 13, 7, 4],
        'steps': [32, 64, 128, 256],
        'num_filters': [128, 128, 256, 256, 512, 512],
        'input_channels': 1,
        'num_scales': [6, 9, 12, 15],
        'min_size': [24, 48, 96, 192],
        'max_size': [48, 96, 192, 384],
        'variance': [0.1, 0.2],
        'clip': True,

        # 'trained_model': 'G:/train_simulation_data_0830_C20/fft_weights/ResNet_i10_8192_20200831_100.pth',
        'trained_model': '/home/linhuang/download/train_simulation_data_0830_C20_fftpca_fft_all/fft_weights/ResNet_i1_800_20200903_50000.pth',
        'using_gpu': False,

        # 'test_data_root': 'J:/npydata_0830_10KHz/traindata_0830_1',
        # 'test_label_root': 'J:/npydata_0830_10KHz/trainlabels_0830_1',
        # 'saved_folder': "./fft_weights/",
        # 'saved_fig_dir':"./fft_weights/",
        # 'txt_name': 'test_fft.txt',

        'test_data_root': '/home/linhuang/download/npydata_10KHz_simulation/testdata_0830_1',
        'test_label_root': '/home/linhuang/download/npydata_10KHz_simulation/testlabels_0830_1',
        'saved_folder': '/home/linhuang/download/train_simulation_data_0830_C20_fftpca_fft_all/fft_weights/',
        'saved_fig_dir': '/home/linhuang/download/train_simulation_data_0830_C20_fftpca_fft_all/fft_weights/',
        'txt_name': 'test_fft_1.txt',

        'labelmap': ['AM', 'SSB', 'BPSK','QPSK','QAM','2FSK','8FSK','CW'],
        'visual_threshold': 0.1,
        'name': 'Signals',
    }
    my_test_signals(settings)
    # gif_path = 'G:/backup/code/DataTraining/fig/gifs/gifs_test_all_data'
    # path_valid(gif_path)
    # path_valid(gif_path + '/wrong/')
    # path_valid(gif_path + '/right/')
    # make_gif(settings['saved_fig_dir'] + '/wrong/', gif_path + '/wrong/')
    # make_gif(settings['saved_fig_dir'] + '/right/', gif_path + '/right/')
