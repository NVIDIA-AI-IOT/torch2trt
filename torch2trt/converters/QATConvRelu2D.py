import numpy as np 
from collections import Iterable
from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test

@tensorrt_converter('torch.nn.intrinsic.qat.modules.conv_fused.ConvReLU2d.forward ', enabled=trt_version() >= '7.1')
def convert_ConvBNRelu2D(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return

    input_dim = input.dim() - 2

    kernel_size = module.kernel_size
    if not isinstance(kernel_size, Iterable):
        kernel_size = (kernel_size, ) * input_dim

    stride = module.stride
    if not isinstance(stride, Iterable):
        stride = (stride, ) * input_dim

    padding = module.padding
    if not isinstance(padding, Iterable):
        padding = (padding, ) * input_dim

    dilation = module.dilation
    if not isinstance(dilation, Iterable):
        dilation = (dilation, ) * input_dim

    kernel = module.weight.detach().cpu().numpy()
    ## There is no bias as BN is being used
    layer = ctx.network.add_convolution_nd(
        input=input_trt,
        num_output_maps=module.out_channels,
        kernel_shape=kernel_size,
        kernel=kernel,
        bias=None)
    layer.stride_nd = stride
    layer.padding_nd = padding
    layer.dilation_nd = dilation
    if module.groups is not None:
        layer.num_groups = module.groups
    
    layer.precision = trt.int8
    layer.set_output_type(0,trt.int8)
    conv_out = layer.get_output(0)
    
    quant_min = module.weight_fake_quant.activation_post_process.min_val.detach().cpu().numpy()
    quant_max = module.weight_fake_quant.activation_post_process.max_val.detach().cpu().numpy()
    conv_out.dynamic_range = (quant_min,quant_max)

    act_layer = ctx.network.add_activation(input=conv_out, type=trt.ActivationType.RELU)
    act_layer.precision=trt.int8
    act_layer.set_output_type(0,trt.int8)
    
    act_layer_out = act_layer.get_output(0)

    quant_min = module.activation_post_process.activation_post_process.min_val.detach().cpu().numpy()
    quant_max = module.activation_post_process.activation_post_process.max_val.detach().cpu().numpy()
    act_layer_out.dynamic_range = (quant_min,quant_max)
    output._trt = act_layer_out
















 
