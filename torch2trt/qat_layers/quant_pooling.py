import torch
from torch.nn.modules import pooling
from . import _utils
from pytorch_quantization import tensor_quant
from pytorch_quantization.nn.modules import _utils as utils

class QuantMaxPool2d(pooling.MaxPool2d,utils.QuantInputMixin):
    """
    Quantized Maxpool2d. Output of maxpool2d needs to be quantized for it to correclty map to a TRT layer
    Originally implementation can be found here:
    https://github.com/NVIDIA/TensorRT/blob/master/tools/pytorch-quantization/pytorch_quantization/nn/modules/quant_pooling.py
    """
    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR

    def __init__(self,kernel_size, stride=None, padding=0, dilation=1,return_indices=False, ceil_mode=False,**kwargs):
        super(QuantMaxPool2d,self).__init__(kernel_size, stride, padding, dilation,return_indices, ceil_mode)
        quant_desc_input = _utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        self.init_quantizer(quant_desc_input)
    
    def forward(self,input):
        output = super(QuantMaxPool2d,self).forward(input)
        ## Although o/p of maxpool2d is being quantized, terminology still says input quantizer, will change later
        output = self._input_quantizer(output)
        return output

## Inference class for quantized maxpool2d
class IQuantMaxPool2d(pooling.MaxPool2d,_utils.QuantMixinInput):
    '''
    Mimicking inference side for maxpool2d followed by a quantized layer
    '''
    def __init__(self,kernel_size, stride=None, padding=0, dilation=1,return_indices=False, ceil_mode=False,**kwargs):
        super().__init__(kernel_size, stride, padding, dilation,return_indices, ceil_mode)
        self.init_quantizer()

    def forward(self,inputs):
        return super(IQuantMaxPool2d,self).forward(inputs)


