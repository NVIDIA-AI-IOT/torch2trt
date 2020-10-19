import torch
import torch.nn as nn
from torch.nn.intrinsic.qat.modules.conv_fused import ConvBnReLU2d as CBR
from torch.nn.intrinsic.qat.modules.conv_fused import ConvReLU2d as CR
import torch.nn as nn
import numpy as np
import collections

#QConfig is accordance with TRT support matrix
qconfig=torch.quantization.QConfig(activation=torch.quantization.FakeQuantize.with_args(observer=torch.quantization.MovingAverageMinMaxObserver,reduce_range=False,quant_min=0,quant_max=255),weight=torch.quantization.FakeQuantize.with_args(observer=torch.quantization.MovingAverageMinMaxObserver,dtype=torch.qint8,qscheme=torch.per_tensor_symmetric,quant_min=-128,quant_max=127))

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            # Replace with ReLU
            nn.ReLU(inplace=False)
        )

#qconfig =  torch.quantization.get_default_qat_qconfig('fbgemm')
class conv(torch.nn.Module):
    """
    common layer for qat and non qat mode
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int=3,
            stride: int=1,
            padding: int=1,
            groups: int=1,
            bias = None,
            padding_mode: str='zeros',
            eps: float=1e-5,
            momentum: float=0.1,
            freeze_bn = False,
            qconfig=qconfig,
            act: bool= True,
            norm: bool=True,
            qat: bool=False):
        super().__init__()
        if qat:
            assert qconfig != None
            if act and norm:
                self.qconv = CBR(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        groups=groups,
                        bias=bias,
                        padding_mode=padding_mode,
                        eps=eps,
                        momentum=momentum,
                        freeze_bn=freeze_bn,
                        qconfig=qconfig)
            elif act and not norm:
                self.qconv = CR(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        groups=groups,
                        bias=bias, # in the absence of BN, bias = True
                        padding_mode = padding_mode,
                        qconfig=qconfig)
            else:
                self.qconv = nn.qat.Conv2d(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        groups=groups,
                        bias=bias, # in the absence of BN, bias=True
                        padding_mode=padding_mode,
                        qconfig=qconfig)

        else:
            self.qconv = ConvBNReLU(in_channels,out_channels,kernel_size,stride,groups)

    def forward(self,inputs):
        return self.qconv(inputs)


def calculate_accuracy(model,data_loader, is_cuda=True):
    correct=0
    total=0
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            images , labels = data
            if is_cuda:
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
    acc = correct * 100 / total
    return acc 


            
