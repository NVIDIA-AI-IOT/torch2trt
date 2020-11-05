"""
These layers represent the inference side for nvidia qat library
"""

import torch

class QuantConv2d(torch.nn.Conv2d):
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
        self.register_buffer('_input_quantizer.learned_amax',torch.tensor(1))
        self.register_buffer('_weight_quantizer.learned_amax',torch.tensor(1))


