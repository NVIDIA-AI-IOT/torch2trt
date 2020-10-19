'''
Contains basic model definitions 
'''

import torch 
import torch.nn as nn
import torch.nn.intrinsic.qat as iqat
from torch.quantization import QuantStub , DeQuantStub
from torch.quantization.observer import MinMaxObserver, default_observer
from utilities import conv,ConvBNReLU

class cnn(nn.Module):
    def __init__(self,qat_mode=False):
        super().__init__()
        self.qat = qat_mode
        self.layer1=conv(3,32,qat=qat_mode)
        self.layer2=conv(32,64,qat=qat_mode)
        self.layer3=conv(64,128,qat=qat_mode)
        self.layer4=conv(128,256,qat=qat_mode)
        self.layer5 = nn.MaxPool2d(kernel_size=2,stride=8)
        self.fcs = nn.Sequential(
                nn.Linear(4096,1024),
                nn.ReLU(),
                nn.Linear(1024,512),
                nn.ReLU(),
                nn.Linear(512,10))

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0),-1)
        x = self.fcs(x)
        return x

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)


