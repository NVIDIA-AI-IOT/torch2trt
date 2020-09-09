from collections import Iterable
from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test

@tensorrt_converter('torch.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU2d.forward ', enabled=trt_version() >= '7.1')
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
    
    ## Creating an int8 weights tensor 
    mode = trt.ScaleMode.Uniform
    q_kernel = ctx.network.add_scale(kernel,mode,scale=module.weight_fake_quant.scale, shift=zeros)
    q_kernel.precision = trt.int8
    q_kernel.set_output_type(0,trt.int8)
    q_kernel_out = q_kernel.get_output(0)
    q_kernel_out.dynamic_range = (module.weight_fake_quant.quant_min,module.weight_fake_quant.quant_max)

    ## There is no bias as BN is being used
    layer = ctx.network.add_convolution_nd(
        input=input_trt,
        num_output_maps=module.out_channels,
        kernel_shape=kernel_size,
        kernel=q_kernel,
        bias=None)
    layer.stride_nd = stride
    layer.padding_nd = padding
    layer.dilation_nd = dilation
    if module.groups is not None:
        layer.num_groups = module.groups
    
    layer.precision = trt.int8
    layer.set_output_type(0,trt.int8)

    scale = module.bn.weight.detach().cpu().numpy() / np.sqrt(
        module.bn.running_var.detach().cpu().numpy() + module.bn.eps
    )
    bias = (
        module.bn.bias.detach().cpu().numpy()
        - module.bn.running_mean.detach().cpu().numpy() * scale
    )
    power = np.ones_like(scale)

    bn_layer = ctx.network.add_scale(layer.get_output(0), trt.ScaleMode.CHANNEL, bias, scale, power)
    bn_layer.precision=trt.int8
    bn_layer.set_output_type(0,trt.int8)

    act_layer = ctx.network.add_activation(input=bn_layer.get_output(0), type=trt.ActivationType.RELU)
    act_layer.precision=trt.int8
    act_layer.set_output_type(0,trt.int8)
    
    act_layer_out = act_layer.get_output(0)
    act_layer_out.dynamic_range = (module.activation_post_process.quant_min,module.activation_post_process.quant_max)
    output._trt = act_layer.get_output(0)















 
