from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test

@tensorrt_converter("torch.nn.ConvTranspose2d.forward", enabled=trt_version() < '7.0')
def convert_ConvTranspose2d(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return

    kernel_size = module.kernel_size
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size,) * 2

    stride = module.stride
    if not isinstance(stride, tuple):
        stride = (stride,) * 2

    padding = module.padding
    if not isinstance(padding, tuple):
        padding = (padding,) * 2

    kernel = module.weight.detach().cpu().numpy()

    bias = trt.Weights(torch_dtype_to_trt(module.weight.dtype))
    if module.bias is not None:
        bias = module.bias.detach().cpu().numpy()

    layer = ctx.network.add_deconvolution(
        input=input_trt,
        num_output_maps=module.out_channels,
        kernel_shape=kernel_size,
        kernel=kernel,
        bias=bias,
    )
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

    
@add_module_test(torch.float32, torch.device("cuda"), [(1,3,224,224)], enabled=trt_version() < '7.0')
def test_square_kernel_equal_stride_mode():
    return torch.nn.ConvTranspose2d(3,3,3,stride=2)

@add_module_test(torch.float32, torch.device("cuda"), [(1,3,224,224)], enabled=trt_version() < '7.0')
def test_square_kernel_equal_stride_mode_unequal_op_size():
    return torch.nn.ConvTranspose2d(3,6,3,stride=2)

@add_module_test(torch.float32, torch.device("cuda"), [(1,3,224,224)], enabled=trt_version() < '7.0')
def test_unequal_stride_mode():
    return torch.nn.ConvTranspose2d(3,3,3, stride=(2,1), padding=(4,2))

@add_module_test(torch.float32, torch.device("cuda"), [(1,3,112,112)], enabled=trt_version() < '7.0')
@add_module_test(torch.float32, torch.device("cuda"), [(1,3,7,7)], enabled=trt_version() < '7.0')
def test_kernelsize_4():
    return torch.nn.ConvTranspose2d(3,3,4, stride=2, padding=1)

