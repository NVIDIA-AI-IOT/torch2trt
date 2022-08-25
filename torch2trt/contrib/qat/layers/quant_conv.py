"""
Original source code taken from nvidia quantization library. 
Changes made to correctly map quantized pytorch layers to TensorRT layers at INT8

Original source: tools/pytorch_quantization/pytorch_quantization/nn/modules/quant_conv.py under 
https://github.com/NVIDIA/TensorRT.git
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_quantization import tensor_quant
from . import _utils

## Inference class for quantized conv2d
class IQuantConv2d(torch.nn.Conv2d,_utils.QuantMixin):
    '''
    mimicking inference side of Conv2d to map correctly to TRT
    Layer to be used with TRT only.
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

    def __repr__(self):
        s = super().__repr__()
        s = "(" + s + "learned amax for weights {0:.4f})".format(self._weight_quantizer.learned_amax) + "\n" +  "(" + s + "learned amax for input {0:.4f})".format(self._input_quantizer.learned_amax)
        return s

    def forward(self,inputs):
        output = F.conv2d(inputs,self.weight,self.bias,self.stride,self.padding,self.dilation,self.groups)
        return output



