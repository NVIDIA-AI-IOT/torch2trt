from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.add')
@tensorrt_converter('torch.Tensor.__iadd__')
@tensorrt_converter('torch.Tensor.__add__')
def convert_add(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return

    if not hasattr(input_b, '_trt') and isinstance(input_b, torch.Tensor):
        shape = tuple(input_a.shape[1:])
        dtype_ones = torch.ones(shape, dtype = input_b.dtype).cpu().numpy()
        values = dtype_ones * input_b.detach().cpu().numpy()
        input_b._trt = ctx.network.add_constant(shape, values).get_output(0)
    
    if not hasattr(input_b, '_trt') and (isinstance(input_b, int) or isinstance(input_b, float)):
        shape = tuple(input_a.shape[1:])
        dtype_ones = torch.ones(shape, dtype = input_a.dtype).cpu().numpy()
        values = dtype_ones * input_b
        input_b = torch.from_numpy(values)
        input_b._trt = ctx.network.add_constant(shape, values).get_output(0)

    layer = ctx.network.add_elementwise(input_a._trt, input_b._trt, trt.ElementWiseOperation.SUM)
    output._trt = layer.get_output(0)


class Add(torch.nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x, y):
        return x + y

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_add_basic():
    return Add()


class IAdd(torch.nn.Module):
    def __init__(self):
        super(IAdd, self).__init__()

    def forward(self, x, y):
        x += y
        return x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_add_iadd():
    return IAdd()


class TorchAdd(torch.nn.Module):
    def __init__(self):
        super(TorchAdd, self).__init__()

    def forward(self, x, y):
        return torch.add(x, y)


class TorchAdd(torch.nn.Module):
    def __init__(self):
        super(TorchAdd, self).__init__()

    def forward(self, x, y):
        return torch.add(x, y)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_add_torchadd():
    return TorchAdd()


class ConstantAdd(torch.nn.Module):
    def __init__(self, v):
        super(ConstantAdd, self).__init__()
        self.const = v

    def forward(self, x):
        return x + self.const


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_add_constantadd():
    return ConstantAdd(3)


class WeightAdd(torch.nn.Module):
    def __init__(self):
        super(WeightAdd, self).__init__()
        self.w = torch.nn.Parameter(torch.Tensor(1))

    def forward(self, x):
        return x + self.w


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_add_weightadd():
    return WeightAdd()
