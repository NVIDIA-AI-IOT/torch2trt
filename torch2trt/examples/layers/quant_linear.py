"""
These layers represent the inference side for nvidia qat library
"""

import torch
from . import _utils

class QuantLinear(torch.nn.Linear,_utils.QuantMixin):
    '''
    mimicking inference side of things
    '''
    def __init__(self,in_features,
                out_features,
                bias=True):
        super().__init__(in_features,out_features,bias)
        self.init_quantizer()


