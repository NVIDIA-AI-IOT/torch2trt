"""
Original source code taken from nvidia quantization library. 
Changes made to correctly map quantized pytorch layers to TensorRT layers at INT8

Original source: tools/pytorch_quantization/pytorch_quantization/nn/modules/quant_linear.py under 
https://github.com/NVIDIA/TensorRT.git
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
from pytorch_quantization import tensor_quant
from pytorch_quantization.nn.modules.quant_conv import _QuantConvNd
import pytorch_quantization.nn.modules._utils as _utils 

'''
Custom class to quantize the input and weights of nn.Linear layer.
'''

class QuantLinear(nn.Linear, _utils.QuantMixin):
    """Quantized version of nn.Linear
    Apply quantized linear to the incoming data, y = dequant(quant(x)quant(A)^T + b).
    Keep Module name "Linear" instead of "QuantLinear" so that it can be easily dropped into preexisting model and load
    pretrained weights. An alias "QuantLinear" is defined below. The base code is a copy of nn.Linear, see detailed
    comment of original arguments there.
    Quantization descriptors are passed in in kwargs. If not presents, default_quant_desc_input and
    default_quant_desc_weight are used.
    Keyword Arguments:
        quant_desc_input: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of input.
        quant_desc_wegiht: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of weight.
    Raises:
        ValueError: If unsupported arguments are passed in.
        KeyError: If unsupported kwargs are passed in.
    Readonly properties:
        - input_quantizer:
        - weight_quantizer:
    Static methods:
        - set_default_quant_desc_input: Set default_quant_desc_input
        - set_default_quant_desc_weight: Set default_quant_desc_weight
    """

    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW

    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        quant_desc_input, quant_desc_weight = _utils.pop_quant_desc_in_kwargs(self.__class__, **kwargs)

        self.init_quantizer(quant_desc_input, quant_desc_weight)

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
        scale, zero_point,quant_min, quant_max, axis = self._extract_info(self._input_quantizer)
        
        setattr(self._input_quantizer, 'quant_scale', torch.nn.Parameter(torch.as_tensor([scale]),requires_grad=False))
        setattr(self._input_quantizer, 'zero_point', torch.nn.Parameter(torch.as_tensor([zero_point]),requires_grad=False))
        setattr(self._input_quantizer, 'quant_min', torch.nn.Parameter(torch.as_tensor([quant_min]),requires_grad=False))
        setattr(self._input_quantizer, 'quant_max', torch.nn.Parameter(torch.as_tensor([quant_max]),requires_grad=False))
        if not axis == None:
            setattr(self._input_quantizer, 'quant_axis', torch.nn.Parameter(torch.as_tensor([axis]),requires_grad=False))

        scale, zero_point, quant_min, quant_max, axis = self._extract_info(self._weight_quantizer)

        setattr(self._weight_quantizer, 'quant_scale', torch.nn.Parameter(torch.as_tensor([scale]),requires_grad=False))
        setattr(self._weight_quantizer, 'zero_point', torch.nn.Parameter(torch.as_tensor([zero_point]),requires_grad=False))
        setattr(self._weight_quantizer, 'quant_min', torch.nn.Parameter(torch.as_tensor([quant_min]),requires_grad=False))
        setattr(self._weight_quantizer, 'quant_max', torch.nn.Parameter(torch.as_tensor([quant_max]),requires_grad=False))
        if not axis == None:
            setattr(self._weight_quantizer, 'quant_axis', torch.nn.Parameter(torch.as_tensor([axis]),requires_grad=False))

    def forward(self, input):
        quant_input = self._input_quantizer(input)
        quant_weight = self._weight_quantizer(self.weight)
	self.extract_quant_info()
        output = F.linear(quant_input, quant_weight, bias=self.bias)

        return output

