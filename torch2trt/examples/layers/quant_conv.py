"""
These layers represent the inference side for nvidia qat library
"""

import torch
from . import _utils

class QuantConv2d_v2(torch.nn.Conv2d,_utils.QuantMixinWeight):
    '''
    mimicking inference side of things
    no input quantizer , only weight quantizer
    '''
    def __init__(self,in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                padding_mode='zeros'):
        super().__init__(in_channels,out_channels,kernel_size,stride=stride,padding=padding,dilation=dilation,groups=groups,bias=bias,padding_mode=padding_mode)
        self.init_quantizer()


class QuantConv2d(torch.nn.Conv2d,_utils.QuantMixin):
    '''
    mimicking inference side of things
    '''
    def __init__(self,in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                padding_mode='zeros'):
        super().__init__(in_channels,out_channels,kernel_size,stride=stride,padding=padding,dilation=dilation,groups=groups,bias=bias,padding_mode=padding_mode)
        self.init_quantizer()


