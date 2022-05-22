import torch
import numpy as np
import time
from ResNet_1D import build_SSD

cfg = {
            'min_dim': 1760,
            'lr_steps':(1000, 3000, 5000, 7000, 10000,15000,25000, 35000 ,45000),
            'max_iter': 101,
            'variance': [0.1, 0.2],
            'num_classes': 15 + 1,
            'feature_maps': [55, 28, 14, 7],
            'steps': [32, 64, 128, 256],
            'num_filters': [128, 128, 256, 256, 512,512],
            'input_channels': 10,
            'num_scales': [6, 9, 12, 15],
            'min_size': [24, 48, 96, 192],
            'max_size': [48, 96, 192, 384],
            'variance': [0.1, 0.2],
            'clip': True,
            'using_gpu': True,
            'name': 'Signals'
       }


t1 = time.time()
net = build_SSD('Out_model', cfg['min_dim'], cfg['num_classes'], cfg)
path = ''

# net.load_weights('D:/USTC/20200913/20201016/train_20KHz_simple_2/RC20_RC40_fft_weights/ResNet_i1_1760_20201017_50000.pth')
net.eval()
t2 = time.time()
# An instance of your model.

# An example input you would normally provide to your model's forward() method.
# f = np.load('demo.npy')
# example = torch.from_numpy(f).type(torch.FloatTensor)
in1 = torch.rand(16, 10, 1760)
in2 = torch.rand(16, 1, 3520)
in3 = torch.rand(16, 1, 7040)
net.cpu()

y = net(in1,in2,in3)


loc = y[0]
# conf = y[1]
# prior = y[2]

t3 = time.time()
print('load_mode:',t2 - t1)
print('forwar_mode:',t3 - t2)
# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(net, (in1,in2,in3))

in1 = torch.rand(64, 10, 1760)
in2 = torch.rand(64, 1, 3520)
in3 = torch.rand(64, 1, 7040)
traced_script_module.save("D:/USTC/20200913/train_20KHz_simple/RC20_RC40_fft_weights/model_20KHz_simple.pt")
while 1:

        t2 = time.time()
        output = traced_script_module(in1, in2, in3)
        t3 = time.time()
        print('TRANSMODLE:', t3 - t2)



# print(output)
# print(traced_script_module)