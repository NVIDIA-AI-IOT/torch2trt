from torch2trt.torch2trt import *


@tensorrt_converter('torch.nn.ConvTranspose2d.forward')
def convert_ConvTranspose2d(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return

    kernel_size = module.kernel_size
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, ) * 2

    stride = module.stride
    if not isinstance(stride, tuple):
        stride = (stride, ) * 2

    padding = module.padding
    if not isinstance(padding, tuple):
        padding = (padding, ) * 2
        
    kernel = module.weight.detach().cpu().numpy()
    
    bias = trt.Weights(torch_dtype_to_trt(module.weight.dtype))
    if module.bias is not None:
        bias = module.bias.detach().cpu().numpy()

    layer = ctx.network.add_deconvolution(
        input=input_trt,
        num_output_maps=module.out_channels,
        kernel_shape=kernel_size,
        kernel=kernel,
        bias=bias)
    layer.stride = stride

    # if output_padding in original pytorch layer is not 0, pre_padding and post_padding should be set respectively. Otherwise the output dimension of pytorch and tensorrt may be different.
    output_padding = module.output_padding
    if output_padding[0] + output_padding[1] > 0:
        layer.pre_padding = padding
        layer.post_padding = trt.tensorrt.DimsHW(padding[0] - output_padding[0], padding[1] - output_padding[1])
    else:
    layer.padding = padding
    
    if module.groups is not None:
        layer.num_groups = module.groups

    output._trt = layer.get_output(0)
