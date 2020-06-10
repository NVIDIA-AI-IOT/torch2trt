from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter("torch.nn.Conv2d.forward", enabled=trt_version() < '7.0')
def convert_Conv2d(ctx):
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

    dilation = module.dilation
    if not isinstance(dilation, tuple):
        dilation = (dilation,) * 2

    kernel = module.weight.detach().cpu().numpy()

    bias = trt.Weights(torch_dtype_to_trt(module.weight.dtype))
    if module.bias is not None:
        bias = module.bias.detach().cpu().numpy()

    layer = ctx.network.add_convolution(
        input=input_trt,
        num_output_maps=module.out_channels,
        kernel_shape=kernel_size,
        kernel=kernel,
        bias=bias,
    )
    layer.stride = stride
    layer.padding = padding
    layer.dilation = dilation

    if module.groups is not None:
        layer.num_groups = module.groups

    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device("cuda"), [(1, 10, 224, 224)], enabled=trt_version() < '7.0')
def test_Conv2d_basic():
    return torch.nn.Conv2d(10, 5, kernel_size=1, stride=1, padding=0)


@add_module_test(torch.float32, torch.device("cuda"), [(1, 10, 224, 224)], enabled=trt_version() < '7.0')
def test_Conv2d_stride2():
    return torch.nn.Conv2d(10, 5, kernel_size=1, stride=2, padding=0)


@add_module_test(torch.float32, torch.device("cuda"), [(1, 10, 224, 224)], enabled=trt_version() < '7.0')
def test_Conv2d_kernel3():
    return torch.nn.Conv2d(10, 5, kernel_size=3, stride=2, padding=1)


@add_module_test(torch.float32, torch.device("cuda"), [(1, 10, 224, 224)], enabled=trt_version() < '7.0')
def test_Conv2d_dilation2():
    return torch.nn.Conv2d(10, 5, kernel_size=3, stride=1, padding=1, dilation=2)
