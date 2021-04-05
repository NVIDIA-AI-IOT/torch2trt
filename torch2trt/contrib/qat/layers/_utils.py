import torch
import copy
import inspect

from absl import logging

from torch import nn

from pytorch_quantization.nn import TensorQuantizer as TQ
from pytorch_quantization.tensor_quant import QuantDescriptor, QUANT_DESC_8BIT_PER_TENSOR

'''
Currently Nvidia quantization library quantizes the input of the conv layer as opposed to output of ReLU.
utilities classes and functions mentioned below are going to help us map int8 layers correctly to TensorRT layers. 
'''

class QuantWeightMixin():
    """Mixin class for adding basic quantization logic to quantized modules"""

    default_quant_desc_weight = QUANT_DESC_8BIT_PER_TENSOR

    @classmethod
    def set_default_quant_desc_input(cls, value):
        """
        Args:
            value: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`
        """
        if not isinstance(value, QuantDescriptor):
            raise ValueError("{} is not an instance of QuantDescriptor!")
        cls.default_quant_desc_weight = copy.deepcopy(value)

    def init_quantizer(self, quant_desc_weight):
        """Helper function for __init__ of simple quantized module

        Create weight quantizer based on quant_desc passed by kwargs, or default of the class.

        Args:
            quant_desc_weight: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`
        """
        if not inspect.stack()[1].function == "__init__":
            raise TypeError("{} should be only called by __init__ of quantized module.".format(__name__))
        self._fake_quant = True
        if not quant_desc_weight.fake_quant:
            raise ValueError("Only fake quantization is supported!")

        logging.info("Input is %squantized to %d bits in %s with axis %s!", ""
                     if not quant_desc_weight.fake_quant else "fake ",
                     quant_desc_weight.num_bits, self.__class__.__name__, quant_desc_weight.axis)

        self._weight_quantizer = TQ(quant_desc_weight)

    # pylint:disable=missing-docstring
    @property
    def weight_quantizer(self):
        return self._weight_quantizer
    # pylint:enable=missing-docstring


def pop_quant_desc_in_kwargs(quant_cls, input_only=False,weight_only=False, **kwargs):
    """Pop quant descriptors in kwargs
    
    If there is no descriptor in kwargs, the default one in quant_cls will be used

    Arguments:
       quant_cls: A class that has default quantization descriptors
       input_only: A boolean. If True, pop quant_desc_input only, not quant_desc_weight. Default false.

    Keyword Arguments:
       quant_desc_input: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
           Quantization descriptor of input.
       quant_desc_weight: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
           Quantization descriptor of weight.

    Note: Original function doesnt pop quant_desc_weight
    """
    if input_only:
        quant_desc_input = kwargs.pop('quant_desc_input', quant_cls.default_quant_desc_input)
    elif weight_only:
        quant_desc_weight = kwargs.pop('quant_desc_weight', quant_cls.default_quant_desc_weight)
    else:
        quant_desc_input = kwargs.pop('quant_desc_input', quant_cls.default_quant_desc_input)
        quant_desc_weight = kwargs.pop('quant_desc_weight', quant_cls.default_quant_desc_weight)


    # Check if anything is left in **kwargs
    if kwargs:
        raise TypeError("Unused keys: {}".format(kwargs.keys()))

    if input_only:
        return quant_desc_input

    if weight_only:
        return quant_desc_weight

    return quant_desc_input, quant_desc_weight



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


