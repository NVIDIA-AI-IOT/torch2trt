'''
Contains basic model definitions 
'''

import torch 
import torch.nn as nn
from utils.utilities import qrelu,qlinear,qconv2d

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
                qlinear(4096,1024,qat=qat_mode,infer=infer),
                qrelu(qat=qat_mode,infer=infer),
                qlinear(1024,512,qat=qat_mode,infer=infer),
                qrelu(qat=qat_mode,infer=infer),
                qlinear(512,10,qat=qat_mode,infer=infer))

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0),-1)
        x = self.fcs(x)
        return x

class vanilla_cnn2(nn.Module):
    def __init__(self,qat_mode=False,infer=False):
        super().__init__()
        self.qat = qat_mode
        self.layer0 = qrelu(qat=qat_mode,infer=infer)
        self.layer1=qconv2d(3,32,padding=1,act=False,qat=qat_mode,infer=infer)
        self.layer1a = qrelu(qat=qat_mode,infer=infer)
        self.layer2=qconv2d(32,64,padding=1,act=False,qat=qat_mode,infer=infer)
        self.layer2a = qrelu(qat=qat_mode,infer=infer)
        self.layer3=qconv2d(64,128,padding=1,act=False,qat=qat_mode,infer=infer)
        self.layer3a = qrelu(qat=qat_mode,infer=infer)
        self.layer4=qconv2d(128,256,padding=1,act=False,qat=qat_mode,infer=infer)
        self.layer4a = qrelu(qat=qat_mode,infer=infer)
        self.layer5 = nn.MaxPool2d(kernel_size=2,stride=8)
        self.layer6 = qlinear(4096,1024,qat=qat_mode,infer=infer)
        self.layer7 = qrelu(qat=qat_mode,infer=infer)
        self.layer8 = qlinear(1024,512,qat=qat_mode,infer=infer)
        self.layer9 = qrelu(qat=False,infer=infer)
        self.layer10 = qlinear(512,10,qat=qat_mode,infer=infer)

    def forward(self,x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer1a(x)
        x = self.layer2(x)
        x = self.layer2a(x)
        x = self.layer3(x)
        x = self.layer3a(x)
        x = self.layer4(x)
        x = self.layer4a(x)
        x = self.layer5(x)
        x = x.view(x.size(0),-1)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        return x


