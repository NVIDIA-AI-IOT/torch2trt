"""
These layers represent the inference side for nvidia qat library
"""

import torch

class TensorQuantizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('learned_amax',torch.tensor(1))

class QuantMixin():
    def init_quantizer(self):
        self._input_quantizer = TensorQuantizer()
        self._weight_quantizer = TensorQuantizer()

    @property
    def input_quantizer(self):
        return self._input_quantizer

    @property
    def weight_quantizer(self):
        return self._weight_quantizer

class QuantLinear(torch.nn.Linear,QuantMixin):
    '''
    mimicking inference side of things
    '''
    def __init__(self,in_features,
                out_features,
                bias=True):
        super().__init__(in_features,out_features,bias)
        self.init_quantizer()


