from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test

def __convert_add(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return
    layer = ctx.network.add_elementwise(input_a._trt, input_b._trt, trt.ElementWiseOperation.SUM)
    output._trt = layer.get_output(0)

@tensorrt_converter('torch.add')
def convert_torch_add(ctx):
    __convert_add(ctx)

class TorchAdd(torch.nn.Module):
    def __init__(self):
        super(TorchAdd, self).__init__()

    def forward(self, x, y):
        return torch.add(x, y)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_torch_add_basic():
    return TorchAdd()

@tensorrt_converter('torch.Tensor.__add__')
def convert_tensor_add(ctx):
    __convert_add(ctx)

class TensorAdd(torch.nn.Module):
    def __init__(self):
        super(TensorAdd, self).__init__()

    def forward(self, x, y):
        return x + y

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_tensor_add_basic():
    return TensorAdd()
