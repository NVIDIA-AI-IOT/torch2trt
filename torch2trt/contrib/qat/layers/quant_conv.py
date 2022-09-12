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
from absl import logging
from . import utils
'''
Custom class to quantize the input and weights of conv2d.
'''

class QuantConv2d(_QuantConvNd):
    """Quantized 2D conv"""

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL
    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
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
        
        scale = self.correct_tensor_type(scale)
        zero_point = self.correct_tensor_type(zero_point)
        quant_min = self.correct_tensor_type(quant_min)
        quant_max = self.correct_tensor_type(quant_max)
        axis = self.correct_tensor_type(axis)
        return scale, zero_point, quant_min, quant_max, axis

    def correct_tensor_type(self,variable):
        if torch.is_tensor(variable):
            return torch.nn.Parameter(variable,requires_grad=False)
        elif variable is None:
            return variable
        else:
            return torch.nn.Parameter(torch.as_tensor([variable]),requires_grad=False)

    def extract_quant_info(self):
        logging.log_first_n(logging.WARNING, "Calculating quantization metrics for {}".format(self.__class__), 1) 
        if self._input_quantizer.learned_amax.numel() == 1:
            logging.log_first_n(logging.WARNING, "per tensor quantization for input quantizer", 1)
        else:
            logging.log_first_n(logging.WARNING, "per channel quantization for input quantizer", 1)
        scale, zero_point,quant_min, quant_max, axis = self._extract_info(self._input_quantizer)
        
        setattr(self._input_quantizer, 'quant_scale', scale)
        setattr(self._input_quantizer, 'zero_point', zero_point)
        setattr(self._input_quantizer, 'quant_min', quant_min)
        setattr(self._input_quantizer, 'quant_max', quant_max)
        if not axis == None:
            setattr(self._input_quantizer, 'quant_axis',axis )

        if self._weight_quantizer.learned_amax.numel() == 1:
            logging.log_first_n(logging.WARNING, "per tensor quantization for weight quantizer", 1)
        else:
            logging.log_first_n(logging.WARNING, "per channel quantization for weight quantizer", 1)
        scale, zero_point, quant_min, quant_max, axis = self._extract_info(self._weight_quantizer)
        
        setattr(self._weight_quantizer, 'quant_scale', scale)
        setattr(self._weight_quantizer, 'zero_point', zero_point)
        setattr(self._weight_quantizer, 'quant_min', quant_min)
        setattr(self._weight_quantizer, 'quant_max', quant_max)
        if not axis == None:
            setattr(self._weight_quantizer, 'quant_axis', axis)
    
    def quantize_tensor(self,quantizer,input):
        if quantizer.learned_amax.numel() == 1:
            quant_input = torch.fake_quantize_per_tensor_affine(input,
                    quantizer.quant_scale.to(torch.float32).item(),
                    quantizer.zero_point.to(torch.long).item(),
                    quantizer.quant_min.to(torch.long).item(),
                    quantizer.quant_max.to(torch.long).item())
        else:
            quant_input = torch.fake_quantize_per_channel_affine(input,
                    quantizer.quant_scale,
                    quantizer.zero_point.to(torch.long),
                    quantizer.quant_axis.to(torch.long).item(),
                    quantizer.quant_min.to(torch.long).item(),
                    quantizer.quant_max.to(torch.long).item())

        return quant_input


    def forward(self, input):
        if self.training:
            quant_input, quant_weight = self._quant(input)
            self.extract_quant_info()
        else:
            if not utils.HelperFunction.export_trt:
                quant_input = self.quantize_tensor(self._input_quantizer,input)
                quant_weight = self.quantize_tensor(self._weight_quantizer,self.weight)
            else:
                quant_input = input
                quant_weight = self.weight

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


