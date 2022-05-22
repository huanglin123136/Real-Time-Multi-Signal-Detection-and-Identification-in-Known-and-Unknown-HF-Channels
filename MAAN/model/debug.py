
from __future__ import print_function
from tqdm import tqdm
from torch.autograd import Variable
from data.SignalDetectionv2 import SignalDetectionv2
from data.SignalDetectionv5 import SignalDetectionv5
from ResNet_1D import build_SSD
from Sample_match import match_sample
from utils import *
from get_gif import make_gif


signal_name = ['AM', 'SSB', 'PSK']


def path_valid(path):
    if not os.path.exists(path):
        os.mkdir(path)


def run(phase, cfg, net, data_set, threshold, txt=True, png=False):
    scale_ = 102.4
    x = data_set
    if cfg['using_gpu']:
        x = x.cuda()
    # forward pass
    x = torch.ones((1, 10, 8192))
    y = net(x)
    detection = y.data
    coords = []
    scores = []
    co = []
    sc = []
    cl = []
    for i in range(detection.size(1)):
        coord, score = extract_signal(detection[9, i, :, :], threshold, scale=scale_)
        coords.append(coord)
        co += coord
        scores.append(score)
        sc += score
        cl += [i - 1] * len(coord)

    return co, sc, cl


def test_signals(cfg):

    net = build_SSD('test', cfg['min_dim'], cfg['num_classes'], cfg)
    net.load_state_dict(torch.load(cfg['trained_model'], map_location='cpu'))
    # example = torch.zeros([1, 10, 8192])
    # traced_script_module = torch.jit.trace(net, example)
    net.eval()
    # traced_script_module.save("model_alpha.pt")
    print('model loaded!')
    # load data
    testset = txt2npy2tensor(cfg['test_data_path'], 0)
    if cfg['using_gpu']:
        net.cuda()
    run('test', cfg, net, testset, cfg['visual_threshold'], True, False)


def test_func(txt_filename='test_0408_3.txt', model_path='./weights/ResNet_i10_8192_20200408_10000.pth',
              pic_filename='./fig/pics/test_0408_3', gif_filename='./fig/gifs/gifs_0408_3'):

    settings = {
        'num_classes': 4,
        'min_dim': 8192,
        'feature_maps': [256, 128, 64, 32, 16],
        'steps': [32, 64, 128, 256, 512],
        'num_filters': [64, 128, 128, 256, 256, 512, 512],
        'input_channels': 10,
        'num_scales': [16, 16, 16, 12, 8],
        # 'num_scales': [32, 32, 32, 16, 12],
        # 'min_size': [164, 328, 656, 820, 2130],
        # 'max_size': [328, 656, 820, 2130, 3441],
        'min_size': [82, 164, 328, 656, 1228],
        'max_size': [164, 328, 656, 1228, 1638],
        'variance': [0.1, 0.2],
        'clip': True,
        'trained_model': model_path,
        'using_gpu': False,
        # 'test_data_root': './data/testdata_i10_0103',
        # 'test_label_root': './data/testlabel_i10_0103',
        'test_data_root': './data/testdata_i10_0403',
        'test_label_root': './data/testlabels_i10_0403',
        'saved_folder': './test/',
        'saved_fig_dir': pic_filename,
        'txt_name': txt_filename,
        'labelmap': ['AM', 'SSB', 'PSK', 'CW', '2FSK'],
        'visual_threshold': 0.1,
        'name': 'Signals',
    }
    test_signals(settings)
    gif_path = gif_filename
    path_valid(gif_path)

    make_gif(settings['saved_fig_dir'] + '/wrong/', gif_path + '/wrong/')


def txt2npy2tensor(path, idx):
    res = np.zeros((10, 10, 8192))
    base_idx = [8192*idx + i for i in range(8192)]
    for i in range(10):
        for j in range(10):
            txt_prefix = 'Data_' + str(i) + '_' + str(j) + '.txt'
            datas = np.loadtxt(os.path.join(path, txt_prefix))
            data = datas[base_idx]
            res[i, j, :] = data
    result = torch.from_numpy(res).type(torch.FloatTensor)
    return result


if __name__ == '__main__':

    settings = {
        'num_classes': 4,
        'min_dim': 8192,
        'feature_maps': [256, 128, 64, 32, 16],
        'steps': [32, 64, 128, 256, 512],
        'num_filters': [64, 128, 128, 256, 256, 512, 512],
        'input_channels': 10,
        'num_scales': [16, 16, 16, 12, 8],
        'min_size': [82, 164, 328, 656, 1228],
        'max_size': [164, 328, 656, 1228, 1638],
        'variance': [0.1, 0.2],
        'clip': True,
        # 'trained_model': './weights/ResNet_i10_8192_20200408_4100.pth',
        'trained_model': 'F:/CSPL/DataTraining/weights/ResNet_i10_8192_20200408_4100.pth',
        'test_data_path': 'F:/CSPL/debug/build',
        'using_gpu': False,
        'saved_folder': './test/',
        'saved_fig_dir': './fig/pics/test_0510',
        'txt_name': 'test_0510.txt',
        'labelmap': ['AM', 'SSB', 'PSK', 'CW', '2FSK'],
        'visual_threshold': 0.01,
        'name': 'Signals',
    }
    test_signals(settings)
    gif_path = './fig/gifs/gifs_0511'
    path_valid(gif_path)

    make_gif(settings['saved_fig_dir'] + '/wrong/', gif_path + '/wrong/')
    make_gif(settings['saved_fig_dir'] + '/right/', gif_path + '/right/')

