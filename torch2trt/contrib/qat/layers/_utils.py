import torch

'''
Inference Layers: At inference time, we dont need to carry entire qat library. We only need dynamic range so that layers
can be mapped to TRT layers at INT8.
'''

class TensorQuantizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('learned_amax',torch.tensor(1.0))

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

class QuantMixinInput():
    def init_quantizer(self):
        self._input_quantizer = TensorQuantizer()

    @property
    def input_quantizer(self):
        return self._input_quantizer

class QuantMixinWeight():
    def init_quantizer(self):
        self._weight_quantizer = TensorQuantizer()

    @property
    def weight_quantizer(self):
        return self._weight_quantizer


