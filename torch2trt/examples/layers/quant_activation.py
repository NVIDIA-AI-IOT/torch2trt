import torch
from . import _utils

class QuantReLU(torch.nn.ReLU,_utils.QuantMixinInput):
    '''
    Mimicking inference side for relu followed by a quantized layer
    '''
    def __init__(self,inplace=False):
        super().__init__(inplace)
        self.init_quantizer()


