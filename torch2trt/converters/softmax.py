from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.nn.functional.softmax')
def convert_softmax(ctx):
    input = ctx.method_args[0]
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return

    # get dims from args or kwargs
    if 'dim' in ctx.method_kwargs:
        dim = ctx.method_kwargs['dim']
    elif len(ctx.method_args) >= 2:
        dim = ctx.method_args[1]

    axes = 1 << (dim - 1)

    layer = ctx.network.add_softmax(input=input_trt)
    layer.axes = axes

    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_softmax_module():
    return torch.nn.Softmax(1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_softmax_module_dim2():
    return torch.nn.Softmax(2)
