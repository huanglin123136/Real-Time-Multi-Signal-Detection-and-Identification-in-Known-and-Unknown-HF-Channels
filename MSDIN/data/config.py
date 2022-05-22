# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")
signal_check = {'AM', 'SSB', 'FM', 'ASK', 'PSK/QAM', 'CW', '2FSK', '4FSK', '8FSK', 'GMSK'}

dic = {'AM': 1, 'SSB': 2, 'FM': 3, 'ASK': 4, 'PSK': 5, 'CW': 6,
       '2FSK': 7, '4FSK': 8, '8FSK': 9, 'GMSK': 10}

cfg = {
    'min_dim': 256,
    'lr_steps': (1000, 3000, 5000, 7000, 10000, 15000, 20000, 25000, 30000, 35000),
    'max_iter': 40001,
    'num_classes': 21 + 1,
    'feature_maps': [128, 64, 32, 16, 8],
    'steps': [2, 4, 8, 16, 32],
    'num_filters': [256, 256, 256, 256, 256],
    'input_channels': 1,
    'num_scales': [9, 9, 12, 12, 12, 16],
    'min_size': [1, 2, 4, 8, 16],
    'max_size': [4, 8, 16, 32, 64],
    'variance': [0.1, 0.2],
    'clip': True,
    'using_gpu': True,
    'name': 'Signals',
    'FPN_feature_size': 256,
}