from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.Tensor.__add__')
def convert_add(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return
    layer = ctx.network.add_elementwise(input_a._trt, input_b._trt, trt.ElementWiseOperation.SUM)
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.Tensor.__iadd__')
def convert_iadd(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    layer = ctx.network.add_elementwise(input_a._trt, input_b._trt, trt.ElementWiseOperation.SUM)
    ctx.method_args[0]._trt = layer.get_output(0)


# TEST z = x + y

class Add(torch.nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x, y):
        return x + y

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_add_basic():
    return Add()


# TEST x += y

class IAdd(torch.nn.Module):
    def __init__(self):
        super(IAdd, self).__init__()

    def forward(self, x, y):
        x += y
        return x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_iadd_basic():
    return IAdd()


# TEST y = x + 1


class AddScalar(torch.nn.Module):
    def __init__(self):
        super(AddScalar, self).__init__()

    def forward(self, x):
        x = x + 1
        return x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_add_scalar():
    return AddScalar()
