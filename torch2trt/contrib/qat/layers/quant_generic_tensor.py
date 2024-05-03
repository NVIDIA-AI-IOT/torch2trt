"""
Original source code taken from nvidia quantization library. 
Changes made to correctly map quantized pytorch layers to TensorRT layers at INT8

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
from pytorch_quantization import tensor_quant
import pytorch_quantization.nn.modules._utils as _utils 
from . import utils
'''
Custom class to quantize the activation of any layer in the network
'''

class QuantGenericTensor(torch.nn.Module,_utils.QuantInputMixin):
    """
    Generic Tensor to quantize activations. 
    Parameters:
    default_quant_desc_input: TensorDescriptor on how to quantize the activations
    
    For different type of descriptors, refer to 
    https://github.com/NVIDIA/TensorRT/blob/e5f9ead4a4826cc774325720a26dbf4ec47203ea/tools/pytorch-quantization/pytorch_quantization/tensor_quant.py#L222
 
    """

    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    def __init__(self,**kwargs): 
        super().__init__()
        quant_desc_input = _utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        self.init_quantizer(quant_desc_input)
        self.identity = torch.nn.Identity()
        self.infer_input_quantizer = utils.InferQuantTensor()

    def forward(self, input):
        if not torch.jit.is_scripting() and self.training:
            quant_input = self._input_quantizer(input)
            self.infer_input_quantizer.extract_quant_info(
                    self._input_quantizer.learned_amax.detach(),
                    self._input_quantizer._num_bits,
                    self._input_quantizer._unsigned
                    )
        else:
            if not utils.HelperFunction.export_trt:
                quant_input = self.infer_input_quantizer(input)
            else: quant_input = self.identity(input)
 
        return quant_input



