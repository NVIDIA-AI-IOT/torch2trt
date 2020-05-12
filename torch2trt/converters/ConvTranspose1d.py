from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.nn.ConvTranspose1d.forward')
def convert_ConvTranspose1d(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return

    kernel_size = (module.kernel_size[0], 1)
    stride = (module.stride[0], 1)
    padding = (module.padding[0], 0)

    kernel = module.weight.detach().cpu().numpy()

    bias = trt.Weights(torch_dtype_to_trt(module.weight.dtype))
    if module.bias is not None:
        bias = module.bias.detach().cpu().numpy()

    # reshape to 2D
    layer = ctx.network.add_shuffle(input_trt)
    layer.reshape_dims = (-1, input.shape[-1], 1)

    layer = ctx.network.add_deconvolution(
        input=layer.get_output(0),
        num_output_maps=module.out_channels,
        kernel_shape=kernel_size,
        kernel=kernel,
        bias=bias)

    layer.stride = stride
    layer.padding = padding

    if module.groups is not None:
        layer.num_groups = module.groups

    # reshape back to 1D
    layer = ctx.network.add_shuffle(layer.get_output(0))
    layer.reshape_dims = (-1, output.shape[-1])

    output._trt = layer.get_output(0)
