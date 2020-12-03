'''
Contains basic model definitions 
'''

import torch 
import torch.nn as nn
from utils.utilities import qrelu,qconv2d

class vanilla_cnn(nn.Module):
    def __init__(self,qat_mode=False,infer=False):
        super().__init__()
        self.qat = qat_mode
        self.layer1=qconv2d(3,32,padding=1,qat=qat_mode,infer=infer)
        self.layer2=qconv2d(32,64,padding=1,qat=qat_mode,infer=infer)
        self.layer3=qconv2d(64,128,padding=1,qat=qat_mode,infer=infer)
        self.layer4=qconv2d(128,256,padding=1,qat=qat_mode,infer=infer)
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



