"""
These layers represent the inference side for nvidia qat library
"""

import torch
from . import _utils
from pytorch_quantization import tensor_quant
from torch2trt.converters.QuantLinear import convert_QuantLinear

class QuantLinear(nn.Linear, _utils.QuantWeightMixin):
    """
    Original description from nvidia quantization library below. We dont quantize the input to linear layer. 

    ---------------------------
    Quantized version of nn.Linear

    No longer valid : Apply quantized linear to the incoming data, y = dequant(quant(x)quant(A)^T + b).

    Keep Module name "Linear" instead of "QuantLinear" so that it can be easily dropped into preexisting model and load
    pretrained weights. An alias "QuantLinear" is defined below. The base code is a copy of nn.Linear, see detailed
    comment of original arguments there.

    Quantization descriptors are passed in in kwargs. If not presents, default_quant_desc_input and
    default_quant_desc_weight are used.

    Keyword Arguments:
        quant_desc_wegiht: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of weight.

    Raises:
        ValueError: If unsupported arguments are passed in.
        KeyError: If unsupported kwargs are passed in.

    Readonly properties:
        - weight_quantizer:

    Static methods:
        - set_default_quant_desc_weight: Set default_quant_desc_weight
    """

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR

    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        quant_desc_weight = _utils.pop_quant_desc_in_kwargs(self.__class__,weight_only=True, **kwargs)

        self.init_quantizer(quant_desc_weight)

    def forward(self, input):
        quant_weight = self._weight_quantizer(self.weight)

        output = F.linear(input, quant_weight, bias=self.bias)

        return output



## inference class for quantized nn.Linear
@tensorrt_method(convert_QuantLinear)
class IQuantLinear(torch.nn.Linear,_utils.QuantMixinWeight):
    '''
    mimicking inference side of things
    '''
    def __init__(self,in_features,
                out_features,
                bias=True):
        super().__init__(in_features,out_features,bias)
        self.init_quantizer()


