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
from . import utils
'''
Custom class to quantize the input and weights of conv2d.
'''

class QuantConv2d(_QuantConvNd):
    """
    Custom class to quantize Conv2d
    Arguments are exactly the same as torch.nn.Conv2d
    quant_desc_input : Quant descriptor to quantize input tensor
    quant_dec_weight : Quant descriptor to quantize weights

    For different type of descriptors, refer to 
    https://github.com/NVIDIA/TensorRT/blob/e5f9ead4a4826cc774325720a26dbf4ec47203ea/tools/pytorch-quantization/pytorch_quantization/tensor_quant.py#L222
    """

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
        self.infer_input_quantizer = utils.InferQuantTensor()
        self.infer_weight_quantizer = utils.InferQuantTensor()


    def forward(self, input):
        if not torch.jit.is_scripting() and self.training:
            quant_input, quant_weight = self._quant(input)
            self.infer_input_quantizer.extract_quant_info(
                    self._input_quantizer.learned_amax.detach(),
                    self._input_quantizer._num_bits,
                    self._input_quantizer._unsigned
                    )
            self.infer_weight_quantizer.extract_quant_info(
                    self._weight_quantizer.learned_amax.detach(),
                    self._weight_quantizer._num_bits,
                    self._weight_quantizer._unsigned
                    )
        else:
            if not utils.HelperFunction.export_trt:
                quant_input = self.infer_input_quantizer(input)
                quant_weight = self.infer_weight_quantizer(self.weight)
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


