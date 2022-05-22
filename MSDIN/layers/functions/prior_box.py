from __future__ import division
import torch
from torch.autograd import Variable
import numpy as np

class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.vector_size = cfg['min_dim']
        self.num_priors = cfg['num_scales']
        self.variance = cfg['variance']
        self.feature_maps = cfg['feature_maps']
        self.min_size = cfg['min_size']
        self.max_size = cfg['max_size']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.version = cfg['name']

    def forward(self):
        boxes = []
        for k, f in enumerate(self.feature_maps):
            for i in range(f):
                f_k = self.vector_size / self.steps[k]
                #unit center x
                cx = (i + 0.5) / f_k

                #length s
                s = np.linspace(self.min_size[k], self.max_size[k], self.num_priors[k])
                s = s / self.vector_size
                for s_k in s:
                    boxes += [cx, s_k]
        #priors size (number of prior boxes, 2) -> (cx, s)
        output = torch.Tensor(boxes).view(-1, 2)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

if __name__ == '__main__':
    cfg = {
        'num_classes': 4,
        'min_dim': 8192,
        'lr_steps': (8000, 15000, 25000, 32000),
        'max_iter': 40100,
        'feature_maps': [256, 128, 64, 32, 16],
        'steps': [32, 64, 128, 256, 512],
        'num_filters': [64, 128, 128, 256, 256, 512, 512],
        'input_channels': 1,
        'num_scales': [6, 9, 12, 15, 15],
        'min_size': [164, 328, 656, 820, 1228],
        'max_size': [328, 656, 820, 1228, 1638],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'Signals',
    }
    priorbox = PriorBox(cfg)
    with torch.no_grad():
        priors = Variable(priorbox.forward())
