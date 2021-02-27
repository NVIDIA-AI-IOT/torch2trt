"""
Original source code taken from nvidia quantization library. 
Changes made to correctly map quantized pytorch layers to TensorRT layers at INT8

Original source: tools/pytorch_quantization/pytorch_quantization/nn/modules/quant_conv.py under 
https://github.com/NVIDIA/TensorRT.git
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.modules.conv import _ConvTransposeNd
from pytorch_quantization import tensor_quant

from . import _utils

class _QuantConvNd(torch.nn.modules.conv._ConvNd, _utils.QuantWeightMixin):
    """base class of quantized Conv inherited from _ConvNd

    Comments of original arguments can be found in torch.nn.modules.conv

    Arguments:
        quant_desc_weight: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of weight.

    Raises:
        ValueError: If unsupported arguments are passed in.

    Readonly properties:
        - weight_quantizer:

    Static methods:
        - set_default_quant_desc_weight: Set default_quant_desc_weight
    """

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode, quant_desc_weight):
        super(_QuantConvNd, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                           transposed, output_padding, groups, bias, padding_mode)
        self.init_quantizer(quant_desc_weight)

    def _quant(self, input):
        """WARNING: Originally Applying quantization on input and weight
        Currently , quantization is applied to weights only.

        Function called by the classes lower in the hierarchy, which actually performs the quantization before forward
        in the derivate class the particular Function.

        Arguments:
            input: in_features to quantize
        Returns:
            A tuple: quant_weight
        """
        quant_weight = self._weight_quantizer(self.weight)

        return quant_weight


class QuantConv2d(_QuantConvNd):
    """Quantized 2D conv"""

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR


    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 **kwargs):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        quant_desc_weight = _utils.pop_quant_desc_in_kwargs(self.__class__,weight_only=True, **kwargs)
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False,
                                          _pair(0), groups, bias, padding_mode,quant_desc_weight=quant_desc_weight)

    def forward(self, input):
        # the actual quantization happens in the next level of the class hierarchy
        quant_weight = self._weight_quantizer(self.weight)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            output = F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                              quant_weight, self.bias, self.stride,
                              _pair(0), self.dilation, self.groups)
        else:
            output = F.conv2d(input, quant_weight, self.bias, self.stride, self.padding, self.dilation,self.groups)
	
        return output

class QuantConvBN2d(_QuantConvNd):
    """Quantized 2D conv + BN"""

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR


    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 **kwargs):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        quant_desc_weight = _utils.pop_quant_desc_in_kwargs(self.__class__,weight_only=True, **kwargs)
        super(QuantConvBN2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False,
                                          _pair(0), groups, bias, padding_mode,quant_desc_weight=quant_desc_weight)

        eps = kwargs.pop('eps',1e-5)
        momentum = kwargs.pop('momentum',0.1)
        affine=kwargs.pop('affine',True)
        track_running_stats=kwargs.pop('track_running_stats',True)
        self.bn = nn.BatchNorm2d(num_features=out_channels,eps=eps,momentum=momentum,affine=affine,track_running_stats=track_running_stats)
        self.register_buffer('folded_weight',torch.ones_like(self.weight))
        self.register_buffer('folded_bias',torch.ones_like(self.bn.running_mean))

    
    def _fold_BN(self,conv_w,conv_b,bn_rm,bn_rv,eps,bn_w,bn_b):
        '''
        conv_w, conv_b = conv weight and bias
        bn_rm,bn_rv = batch norm running mean and variance
        bn_w , bn_b = batch norm weight and bias
        eps = epsilon
        '''
        if conv_b is None:
            conv_b = torch.zeros_like(bn_rm)
        if bn_w is None:
            bn_w = torch.ones_like(bn_rm)
        if bn_b is None:
            bn_b = torch.zeros_like(bn_rm)
        bn_var_rsqrt = torch.rsqrt(bn_rv + eps)
        scale_factor=  bn_w * bn_var_rsqrt
        self.folded_weight = conv_w * (scale_factor).reshape([-1] + [1] * (len(conv_w.shape) - 1))
        self.folded_bias = (conv_b - bn_rm) * scale_factor + bn_b
        return self.folded_weight,self.folded_bias

    def forward(self, input):
        # the actual quantization happens in the next level of the class hierarchy
        folded_weight , folded_bias = self._fold_BN(self.weight,self.bias,self.bn.running_mean,self.bn.running_var,self.bn.eps,self.bn.weight,self.bn.bias)
        quant_weight = self._weight_quantizer(folded_weight)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            output = F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                              quant_weight, folded_bias, self.stride,
                              _pair(0), self.dilation, self.groups)
        else:
            output = F.conv2d(input, quant_weight, folded_bias, self.stride, self.padding, self.dilation,self.groups)
	
        return output


## Inference class for quantized convbn2d
class IQuantConvBN2d(torch.nn.Conv2d,_utils.QuantMixinWeight):
    '''
    mimicking inference side of things for convbn2d
    no input quantizer , only weight quantizer
    '''
    def __init__(self,in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                padding_mode='zeros',
                **kwargs):
        super().__init__(in_channels,out_channels,kernel_size,stride=stride,padding=padding,dilation=dilation,groups=groups,bias=bias,padding_mode=padding_mode)
        self.init_quantizer()
        eps = kwargs.pop('eps',1e-5)
        momentum = kwargs.pop('momentum',0.1)
        affine=kwargs.pop('affine',True)
        track_running_stats=kwargs.pop('track_running_stats',True)
        self.bn = nn.BatchNorm2d(num_features=out_channels,eps=eps,momentum=momentum,affine=affine,track_running_stats=track_running_stats)

        self.register_buffer('folded_weight',torch.ones_like(self.weight))
        self.register_buffer('folded_bias',torch.ones_like(self.bn.running_mean))

    def __repr__(self):
        s = super().__repr__()
        s = "(" + s + "dynamic_range amax {0:.4f})".format(self._weight_quantizer.learned_amax)
        return s

    def forward(self,inputs):
        output = F.conv2d(inputs,self.folded_weight,self.folded_bias,self.stride,self.padding,self.dilation,self.groups)
        return output


## Inference class for quantized conv2d
class IQuantConv2d(torch.nn.Conv2d,_utils.QuantMixinWeight):
    '''
    mimicking inference side of things
    no input quantizer , only weight quantizer
    '''
    def __init__(self,in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                padding_mode='zeros'):
        super().__init__(in_channels,out_channels,kernel_size,stride=stride,padding=padding,dilation=dilation,groups=groups,bias=bias,padding_mode=padding_mode)
        self.init_quantizer()

    def forward(self,inputs):
        return super(IQuantConv2d, self).forward(inputs)

#class QuantConv2d(torch.nn.Conv2d,_utils.QuantMixin):
#    '''
#    mimicking inference side of things
#    '''
#    def __init__(self,in_channels,
#                out_channels,
#                kernel_size,
#                stride=1,
#                padding=0,
#                dilation=1,
#                groups=1,
#                bias=True,
#                padding_mode='zeros'):
#        super().__init__(in_channels,out_channels,kernel_size,stride=stride,padding=padding,dilation=dilation,groups=groups,bias=bias,padding_mode=padding_mode)
#        self.init_quantizer()
#
#
