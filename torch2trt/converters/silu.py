from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.nn.functional.silu')
def convert_functional_silu(ctx):
    ctx.method_args = (torch.nn.SiLU(),) + ctx.method_args
    convert_silu(ctx)

@tensorrt_converter('torch.nn.SiLU.forward')
def convert_silu(ctx):
    input = ctx.method_args[1]
    output = ctx.method_return
    layer = ctx.network.add_activation(input=input._trt, type=trt.ActivationType.SIGMOID)
    layer = ctx.network.add_elementwise(input._trt, layer.get_output(0), trt.ElementWiseOperation.PROD)
    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
def test_silu_basic():
    return torch.nn.SiLU()


class FunctionalSilu(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.silu(x)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
def test_functional_silu_basic():
    return FunctionalSilu()