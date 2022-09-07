"""
Original source code taken from nvidia quantization library. 
Changes made to correctly map quantized pytorch layers to TensorRT layers at INT8

Original source: tools/pytorch_quantization/pytorch_quantization/nn/modules/quant_conv.py under 
https://github.com/NVIDIA/TensorRT.git
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
from pytorch_quantization import tensor_quant
from pytorch_quantization.nn.modules.quant_conv import _QuantConvNd
import pytorch_quantization.nn.modules._utils as _utils 
from . import _utils as utils
from absl import logging

'''
Training class to quantize the input and weights of conv2d.
Source code inspired from <insert path>
'''

class QuantConv2d(_QuantConvNd):
    """Quantized 2D conv"""

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 **kwargs):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        quant_desc_input, quant_desc_weight = _utils.pop_quant_desc_in_kwargs(self.__class__, **kwargs)
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False,
                                          _pair(0), groups, bias, padding_mode,
                                          quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight)

    def _extract_info(self,quantizer):
        bound = (1 << (quantizer._num_bits - 1 + int(quantizer._unsigned))) - 1
        amax = quantizer.learned_amax
        quantizer._scale = amax
        if amax.numel() == 1:
            scale=amax.item() / bound
            zero_point = 0
            quant_min = -bound - 1 if not quantizer._unsigned else 0
            quant_max = bound
            axis = None
        else:
            amax_sequeeze = amax.squeeze().detach()
            if len(amax_sequeeze.shape) != 1:
                raise TypeError("Multiple axis is not supported in quantization")
            quant_dim = list(amax.shape).index(list(amax_sequeeze.shape)[0])
            scale = amax_sequeeze / bound
            scale = scale.data
            zero_point = torch.zeros_like(scale, dtype=torch.int32).data
            axis = quant_dim
            quant_min = -bound - 1 if not quantizer._unsigned else 0
            quant_max = bound
        return scale, zero_point, quant_min, quant_max, axis

    def extract_quant_info(self):
        logging.log_first_n(logging.WARNING, "Calculating quantization metrics for {}".format(self.__class__), 1) 
        if self._input_quantizer.learned_amax.numel() == 1:
            logging.log_first_n(logging.WARNING, "per tensor quantization for input quantizer", 1)
        else:
            logging.log_first_n(logging.WARNING, "per channel quantization for input quantizer", 1)
        scale, zero_point,quant_min, quant_max, axis = self._extract_info(self._input_quantizer)
        
        setattr(self._input_quantizer, 'quant_scale', torch.nn.Parameter(torch.as_tensor(scale),requires_grad=False))
        setattr(self._input_quantizer, 'zero_point', torch.nn.Parameter(torch.as_tensor(zero_point),requires_grad=False))
        setattr(self._input_quantizer, 'quant_min', torch.nn.Parameter(torch.as_tensor(quant_min),requires_grad=False))
        setattr(self._input_quantizer, 'quant_max', torch.nn.Parameter(torch.as_tensor(quant_max),requires_grad=False))
        if not axis == None:
            setattr(self._input_quantizer, 'quant_axis', torch.nn.Parameter(torch.as_tensor(axis),requires_grad=False))

        if self._weight_quantizer.learned_amax.numel() == 1:
            logging.log_first_n(logging.WARNING, "per tensor quantization for weight quantizer", 1)
        else:
            logging.log_first_n(logging.WARNING, "per channel quantization for weight quantizer", 1)
        scale, zero_point, quant_min, quant_max, axis = self._extract_info(self._weight_quantizer)

        setattr(self._weight_quantizer, 'quant_scale', torch.nn.Parameter(torch.as_tensor(scale),requires_grad=False))
        setattr(self._weight_quantizer, 'zero_point', torch.nn.Parameter(torch.as_tensor(zero_point),requires_grad=False))
        setattr(self._weight_quantizer, 'quant_min', torch.nn.Parameter(torch.as_tensor(quant_min),requires_grad=False))
        setattr(self._weight_quantizer, 'quant_max', torch.nn.Parameter(torch.as_tensor(quant_max),requires_grad=False))
        if not axis == None:
            setattr(self._weight_quantizer, 'quant_axis', torch.nn.Parameter(torch.as_tensor(axis),requires_grad=False))

    def forward(self, input):
        # the actual quantization happens in the next level of the class hierarchy
        quant_input, quant_weight = self._quant(input)
        if self.eval:
            self.extract_quant_info()

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            output = F.conv2d(F.pad(quant_input, expanded_padding, mode='circular'),
                              quant_weight, self.bias, self.stride,
                              _pair(0), self.dilation, self.groups)
        else:
            output = F.conv2d(quant_input, quant_weight, self.bias, self.stride, self.padding, self.dilation,
                              self.groups)

        return output


## Inference class for quantized conv2d
class IQuantConv2d(torch.nn.Conv2d,utils.QuantMixin):
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



