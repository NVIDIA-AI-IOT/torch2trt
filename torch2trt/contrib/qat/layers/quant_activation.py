import torch
from . import _utils
from pytorch_quantization import tensor_quant
from pytorch_quantization.nn import TensorQuantizer as TQ
from pytorch_quantization.nn.modules import _utils as utils

class QuantReLU(torch.nn.Module):
    """
    Quantized ReLu. However, output of relu needs to be quantized for it to correclty map to a TRT layer
    """
    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR

    def __init__(self,inplace=False,**kwargs):
        super().__init__()
        self.qrelu=torch.nn.ReLU(inplace=inplace)
        quant_desc_input = _utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        self._output_quantizer = TQ(quant_desc_input)

    def forward(self,input):
        output = self.qrelu(input)
        quantized_output = self._output_quantizer(output)
        return quantized_output

class IQuantReLU(torch.nn.Module):
    """
    Quantized Relu layer for inference used in TRT conversion.
    """

    def __init__(self,inplace=False):
        super().__init__()
        self.qrelu=torch.nn.ReLU(inplace=inplace)
        self._output_quantizer = _utils.TensorQuantizer()

    def __repr__(self):
        s = super().__repr__()
        s = "(" + s + "dynamic_range amax {0:.4f})".format(self._output_quantizer.learned_amax)
        return s

    def forward(self,input):
        return self.qrelu(input)


