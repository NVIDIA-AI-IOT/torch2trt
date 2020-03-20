from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test
from .Conv2d import *

@tensorrt_converter('torch.nn.functional.conv2d')
def convert_conv2d(ctx):
    in_channels = ctx.method_args[0].size()[1]
    out_channels = ctx.method_args[1].size()[0]
    kernel_size = tuple(ctx.method_args[1].size()[2:4])
    stride      = ctx.method_args[3]
    padding     = ctx.method_args[4]
    dilation    = ctx.method_args[5]
    groups      = ctx.method_args[6]
    bias = False if ctx.method_args[2] is None else True

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
    module.weight = ctx.method_args[1]
    module.bias = ctx.method_args[2]

    ctx.method_args = (module, ctx.method_args[0])
    convert_Conv2d(ctx)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 384, 128)])
def test_conv2d_basic():
    x_size = (1, 3, 384, 128)
    weight = torch.nn.Parameter(torch.randn((6,1,3,3)).cuda())
    # bias_t = torch.nn.Parameter(torch.randn(4).cuda())
    bias_t = None
    
    in_channels = x_size[1]
    out_channels = weight.size()[0]
    kernel_size = tuple(weight.size()[2:4])
    stride = 1
    padding = weight.size()[2] // 2
    dilation = 1
    groups = 3
    bias = False if bias_t is None else True

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
