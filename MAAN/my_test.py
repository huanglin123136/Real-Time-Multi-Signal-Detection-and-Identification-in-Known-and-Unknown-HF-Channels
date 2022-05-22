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
signal_name = ['AM', 'SSB', 'PSK']


def path_valid(path):
    if not os.path.exists(path):
        os.mkdir(path)


def run(phase, cfg, net, data_set, threshold, txt=True, png=False):
    if png:
        matplotlib.use('Agg')
    file_name = cfg['saved_folder'] + cfg['txt_name']
    num_seqs = len(data_set)
    scale_ = 102.4
    # plot_idx = [num_seqs - 1 - x for x in range(100)]
    # for idx in tqdm(range(0, num_seqs, 10)):
    for idx in tqdm(range(512,num_seqs)):
        seq = data_set.pull_seq(idx)
        if phase == 'test':
            seq_id, annos = data_set.pull_anno(idx)
            annos[:, :-1] *= scale_
            if txt:
                with open(file_name, mode='a') as f:
                    f.write('\nGround Truth for seq ' + seq_id + '\n')
                    for box in annos:
                        f.write('label: ' + '||'.join(str(b) for b in box) + '\n')

        x = torch.from_numpy(seq).type(torch.FloatTensor)
        x = Variable(x.unsqueeze(0))
        if cfg['using_gpu']:
            x = x.cuda()
        # forward pass
        y = net(x)
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
                    plot_spec(seq, coords, scores, idx, cfg['labelmap'], cfg['saved_fig_dir'] + '/wrong/', i)
            else:
                path_valid(cfg['saved_fig_dir'] + '/right/')
                for i in range(0,cfg['input_channels'],4):
                    plot_spec(seq, coords, scores, idx, cfg['labelmap'], cfg['saved_fig_dir'] + '/right/', i)


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
    run('test', cfg, net, testset, cfg['visual_threshold'], True, True)


if __name__ == '__main__':

    settings = {
        'num_classes': 4,
        'min_dim': 8192,
        'feature_maps': [256, 128, 64, 32, 16],
        'steps': [32, 64, 128, 256, 512],
        'num_filters': [64, 128, 128, 256, 256, 512, 512],
        'input_channels': 20,
        'num_scales': [16, 16, 16, 12, 8],
        # 'num_scales': [32, 32, 32, 16, 12],
        # 'min_size': [164, 328, 656, 820, 2130],
        # 'max_size': [328, 656, 820, 2130, 3441],
        'min_size': [82, 164, 328, 656, 1228],
        'max_size': [164, 328, 656, 1228, 1638],
        'variance': [0.1, 0.2],
        'clip': True,
        'trained_model': 'G:/backup/code/DataTraining/weights/ResNet_i20_8192_20200711_49600.pth',
        'using_gpu': False,
        'test_data_root': 'J:/npydata0709/testdata_i10_0709_6',
        'test_label_root': 'J:/npydata0709/testlabels_i10_0709_6',
        'saved_folder': 'G:/backup/code/DataTraining/test_0711/',
        'saved_fig_dir': 'G:/backup/code/DataTraining/fig/pics/test_0711_ceshi',
        'txt_name': 'test_0711_train.txt',
        'labelmap': ['AM', 'SSB', 'PSK', 'CW', '2FSK'],
        'visual_threshold': 0.1,
        'name': 'Signals',
    }
    my_test_signals(settings)
    gif_path = 'G:/backup/code/DataTraining/fig/gifs/gifs_0711_ceshi'
    path_valid(gif_path)
    path_valid(gif_path + '/wrong/')
    path_valid(gif_path + '/right/')
    make_gif(settings['saved_fig_dir'] + '/wrong/', gif_path + '/wrong/')
    make_gif(settings['saved_fig_dir'] + '/right/', gif_path + '/right/')
