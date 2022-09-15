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
from absl import logging
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

    def _extract_info(self,quantizer):
        bound = (1 << (quantizer._num_bits - 1 + int(quantizer._unsigned))) - 1
        amax = quantizer.learned_amax
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
    
    def quantize_input(self,quantizer,input):
        quantizer._scale = quantizer.learned_amax
        if quantizer.learned_amax.numel() == 1:
            quant_input = torch.fake_quantize_per_tensor_affine(input,
                    quantizer.quant_scale.to(torch.float32).item(),
                    quantizer.zero_point.to(torch.long).item(),
                    quantizer.quant_min.to(torch.long).item(),
                    quantizer.quant_max.to(torch.long).item())
        else:
            quant_input = torch.fake_quantize_per_channel_affine(input,
                    quantizer.quant_scale,
                    quantizer.zero_point,
                    quantizer.quant_axis.to(torch.int32).item(),
                    quantizer.quant_min.to(torch.long).item(),
                    quantizer.quant_max.to(torch.long).item())

        return quant_input

    def forward(self, input):
        if self.training:
            quant_input = self._input_quantizer(input)
            self.extract_quant_info()
        else:
            if not utils.HelperFunction.export_trt:
                quant_input = self.quantize_input(self._input_quantizer,input)
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

    def _extract_info(self,quantizer):
        bound = (1 << (quantizer._num_bits - 1 + int(quantizer._unsigned))) - 1
        amax = quantizer.learned_amax
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

    def quantize_input(self,quantizer,input):
        quantizer._scale = quantizer.learned_amax
        if quantizer.learned_amax.numel() == 1:
            quant_input = torch.fake_quantize_per_tensor_affine(input,
                    quantizer.quant_scale.to(torch.float32).item(),
                    quantizer.zero_point.to(torch.long).item(),
                    quantizer.quant_min.to(torch.long).item(),
                    quantizer.quant_max.to(torch.long).item())
        else:
            quant_input = torch.fake_quantize_per_channel_affine(input,
                    quantizer.quant_scale,
                    quantizer.zero_point,
                    quantizer.quant_axis.to(torch.int32).item(),
                    quantizer.quant_min.to(torch.long).item(),
                    quantizer.quant_max.to(torch.long).item())

        return quant_input

    def forward(self, input):
        if self.training:
            quant_input = self._input_quantizer(input)
            self.extract_quant_info()
        else:
            # It is expected that model has already been finetuned / trained with qat.
            # For quick inference test, run a single step of train step to invoke and save quantization parameters
            if not utils.HelperFunction.export_trt:
                quant_input = self.quantize_input(self._input_quantizer,input)
            else: quant_input = input
        
        output = self.adaptive_avg_pool2d(quant_input)
        return output


