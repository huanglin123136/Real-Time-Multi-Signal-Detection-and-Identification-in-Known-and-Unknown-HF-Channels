import torch
import torch.nn as nn
class CAB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CAB, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool1d(output_size=1)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.sigmod = nn.Sigmoid()

    def forward(self, x1, x2):
        # x1, x2 = x  # high, low
        x = torch.cat([x1, x2],1)
        # x = self.global_pooling(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmod(x)
        # x2 = x * x2
        # res = x2 + x1

        return x

CAB_model = CAB(128, 64)
X1 = torch.ones([1, 28, 48])
X2 = torch.rand([1, 100, 48])
Y = CAB_model(X1, X2)
CAB_traced_script_module = torch.jit.trace(CAB_model, (X1, X2))
CAB_traced_script_module.save("D:/USTC/20200913/train_20KHz/RC20_RC40_fft_weights/traced_CAB.pt")

