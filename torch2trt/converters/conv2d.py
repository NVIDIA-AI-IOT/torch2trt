from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test
from .Conv2d import *

@tensorrt_converter('torch.nn.functional.conv2d')
def convert_conv2d(ctx):
    in_channels  = ctx.method_args[0].size()[1]
    out_channels = ctx.method_args[1].size()[0]
    kernel_size  = tuple(ctx.method_args[1].size()[2:4])
    stride       = ctx.method_args[3]
    padding      = ctx.method_args[4]
    dilation     = ctx.method_args[5]
    groups       = ctx.method_args[6]
    bias = False if ctx.method_args[2] is None else True

    module = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias)
    module.weight = ctx.method_args[1]
    module.bias = ctx.method_args[2]

    ctx.method_args = (module, ctx.method_args[0])
    convert_Conv2d(ctx)


# test
g_input_size = (1, 10, 224, 224)
g_output_size = (1, 5, 112, 112)

@add_module_test(torch.float32, torch.device('cuda'), [g_input_size])
def test_conv2d_basic():
    kernel_size = (3, 3)
    groups = 1
    dilation = 1
    bias = False
    in_channels = g_input_size[1]
    out_channels = g_output_size[1]
    weight_channels = in_channels // groups
    if in_channels % groups or out_channels % groups:
        raise ValueError("in_channels and out_channels must be divisible by groups")
    if bias:
        bias_t = torch.nn.Parameter(torch.randn(out_channels).cuda())
    else:
        bias_t = None
    stride = (g_input_size[2] // g_output_size[2], g_input_size[3] // g_output_size[3])
    padding = (kernel_size[0] // 2 * dilation, kernel_size[1] // 2 * dilation)
    weight = torch.nn.Parameter(torch.randn((out_channels, weight_channels,) + kernel_size).cuda())

    module = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        padding_mode='zeros')
    module.weight = weight
    module.bias = bias_t
    return module


@add_module_test(torch.float32, torch.device('cuda'), [g_input_size])
def test_conv2d_groups5():
    kernel_size = (3, 3)
    groups = 5
    dilation = 1
    bias = True
    in_channels = g_input_size[1]
    out_channels = g_output_size[1]
    weight_channels = in_channels // groups
    if in_channels % groups or out_channels % groups:
        raise ValueError("in_channels and out_channels must be divisible by groups")
    if bias:
        bias_t = torch.nn.Parameter(torch.randn(out_channels).cuda())
    else:
        bias_t = None
    stride = (g_input_size[2] // g_output_size[2], g_input_size[3] // g_output_size[3])
    padding = (kernel_size[0] // 2 * dilation, kernel_size[1] // 2 * dilation)
    weight = torch.nn.Parameter(torch.randn((out_channels, weight_channels,) + kernel_size).cuda())

    module = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        padding_mode='zeros')
    module.weight = weight
    module.bias = bias_t
    return module


@add_module_test(torch.float32, torch.device('cuda'), [g_input_size])
def test_conv2d_dilation2():
    kernel_size = (3, 3)
    groups = 1
    dilation = 2
    bias = False
    in_channels = g_input_size[1]
    out_channels = g_output_size[1]
    weight_channels = in_channels // groups
    if in_channels % groups or out_channels % groups:
        raise ValueError("in_channels and out_channels must be divisible by groups")
    if bias:
        bias_t = torch.nn.Parameter(torch.randn(out_channels).cuda())
    else:
        bias_t = None
    stride = (g_input_size[2] // g_output_size[2], g_input_size[3] // g_output_size[3])
    padding = (kernel_size[0] // 2 * dilation, kernel_size[1] // 2 * dilation)
    weight = torch.nn.Parameter(torch.randn((out_channels, weight_channels,) + kernel_size).cuda())

    module = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        padding_mode='zeros')
    module.weight = weight
    module.bias = bias_t
    return module


@add_module_test(torch.float32, torch.device('cuda'), [g_input_size])
def test_conv2d_no_bias():
    kernel_size = (3, 3)
    groups = 1
    dilation = 1
    bias = False
    in_channels = g_input_size[1]
    out_channels = g_output_size[1]
    weight_channels = in_channels // groups
    if in_channels % groups or out_channels % groups:
        raise ValueError("in_channels and out_channels must be divisible by groups")
    if bias:
        bias_t = torch.nn.Parameter(torch.randn(out_channels).cuda())
    else:
        bias_t = None
    stride = (g_input_size[2] // g_output_size[2], g_input_size[3] // g_output_size[3])
    padding = (kernel_size[0] // 2 * dilation, kernel_size[1] // 2 * dilation)
    weight = torch.nn.Parameter(torch.randn((out_channels, weight_channels,) + kernel_size).cuda())

    module = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        padding_mode='zeros')
    module.weight = weight
    module.bias = bias_t
    return module

