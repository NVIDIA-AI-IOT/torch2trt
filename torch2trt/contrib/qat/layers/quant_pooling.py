"""
Original source code taken from nvidia quantization library. 
Changes made to correctly map quantized pytorch layers to TensorRT layers at INT8

Original source: tools/pytorch_quantization/pytorch_quantization/nn/modules/quant_pooling.py under 
https://github.com/NVIDIA/TensorRT.git
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
from pytorch_quantization import tensor_quant
import pytorch_quantization.nn.modules._utils as _utils 
from . import utils
'''
Custom class to quantize the input of various pooling layers.
'''

class QuantMaxPool2d(torch.nn.Module,_utils.QuantInputMixin):
    """
    Custom layer to quantize torch.nn.MaxPool2d layer
    Arguments are exactly the same as torch.nn.MaxPool2d
    quant_desc_input : Quant Descriptor to quantize input tensor
    
    For different type of descriptors, refer to 
    https://github.com/NVIDIA/TensorRT/blob/e5f9ead4a4826cc774325720a26dbf4ec47203ea/tools/pytorch-quantization/pytorch_quantization/tensor_quant.py#L222
    """
    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False, **kwargs):
        super().__init__()
        quant_desc_input = _utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        self.init_quantizer(quant_desc_input)
        self.maxpool2d = torch.nn.MaxPool2d(kernel_size,stride=stride,padding=padding,
                dilation=dilation,return_indices=return_indices, ceil_mode=ceil_mode)

        self.infer_input_quantizer = utils.InferQuantTensor()


    def forward(self, input):
        if not torch.jit.is_scripting() and self.training:
            quant_input = self._input_quantizer(input)
            self.infer_input_quantizer.extract_quant_info(
                    self._input_quantizer.learned_amax.detach(),
                    self._input_quantizer._num_bits,
                    self._input_quantizer._unsigned
                    )
        else:
            if not utils.HelperFunction.export_trt:
                quant_input = self.infer_input_quantizer(input)
            else: quant_input = input
 
        output = self.maxpool2d(quant_input)
        return output


class QuantAdaptiveAvgPool2d(torch.nn.Module,_utils.QuantInputMixin):
    """
    Custom layer to quantize torch.nn.AdaptiveAvgPool2d layer
    Arguments are exactly the same as torch.nn.AdaptiveAvgPool2d
    quant_desc_input : Quant Descriptor to quantize input tensor
    
    For different type of descriptors, refer to 
    https://github.com/NVIDIA/TensorRT/blob/e5f9ead4a4826cc774325720a26dbf4ec47203ea/tools/pytorch-quantization/pytorch_quantization/tensor_quant.py#L222
 
    """

    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    def __init__(self, output_size, **kwargs):
        super().__init__()
        quant_desc_input = _utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        self.init_quantizer(quant_desc_input)
        self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d(output_size)
        self.infer_input_quantizer = utils.InferQuantTensor()

    def forward(self, input):
        if not torch.jit.is_scripting() and self.training:
            quant_input = self._input_quantizer(input)
            self.infer_input_quantizer.extract_quant_info(
                    self._input_quantizer.learned_amax.detach(),
                    self._input_quantizer._num_bits,
                    self._input_quantizer._unsigned
                    )
        else:
            # It is expected that model has already been finetuned / trained with qat.
            # For quick inference test, run a single step of train step to invoke and save quantization parameters
            if not utils.HelperFunction.export_trt:
                quant_input = self.infer_input_quantizer(input)
            else: quant_input = input
        
        output = self.adaptive_avg_pool2d(quant_input)
        return output


