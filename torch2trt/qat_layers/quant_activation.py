import torch
from . import _utils
from pytorch_quantization.nn.modules import _utils as utils
from torch2trt.converters.QuantReLU import convert_QuantReLU

class QuantReLU(torch.nn.ReLU,utils.QuantInputMixin):
    """
    Quantized ReLu. However, output of relu needs to be quantized for it to correclty map to a TRT layer
    """
    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR

    def __init__(self,inplace=False):
        super(QuantReLU,self).__init__(inplace)
        quant_desc_input = _utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        self.init_quantizer(quant_desc_input)
    
    def forward(self,input):
        output = super(QuantReLU,self).forward(input)
        ## Although o/p of relu is being quantized, terminology still says input quantizer, will change later
        output = self._input_quantizer(output)
        return output

## Inference class for quantized relu
@tensorrt_method(convert_QuantReLU)
class IQuantReLU(torch.nn.ReLU,_utils.QuantMixinInput):
    '''
    Mimicking inference side for relu followed by a quantized layer
    '''
    def __init__(self,inplace=False):
        super().__init__(inplace)
        self.init_quantizer()

