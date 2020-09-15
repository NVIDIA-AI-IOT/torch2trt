from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.sub')
@tensorrt_converter('torch.Tensor.__isub__')
@tensorrt_converter('torch.Tensor.__sub__')
def convert_sub(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape) - 1)
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.SUB)
    output._trt = layer.get_output(0)

    
@tensorrt_converter('torch.Tensor.__rsub__')
def convert_sub(ctx):
    input_a = ctx.method_args[1]
    input_b = ctx.method_args[0]  # flipped for rsub
    output = ctx.method_return
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape) - 1)
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.SUB)
    output._trt = layer.get_output(0)
    

class Sub(torch.nn.Module):
    def __init__(self):
        super(Sub, self).__init__()

    def forward(self, x, y):
        return x - y

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_sub_basic():
    return Sub()


class ISub(torch.nn.Module):
    def __init__(self):
        super(ISub, self).__init__()

    def forward(self, x, y):
        x -= y
        return x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_sub_isub():
    return ISub()


class TorchSub(torch.nn.Module):
    def __init__(self):
        super(TorchSub, self).__init__()

    def forward(self, x, y):
        return torch.sub(x, y)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_torch_sub():
    return TorchSub()


class RSubInt(torch.nn.Module):
    def __init__(self):
        super(RSubInt, self).__init__()

    def forward(self, x):
        return 1 - x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_rsub_int():
    return RSubInt()


class RSubFloat(torch.nn.Module):
    def __init__(self):
        super(RSubFloat, self).__init__()

    def forward(self, x):
        return 1.0 - x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_rsub_float():
    return RSubFloat()

class SubConstantNoBatch(torch.nn.Module):
    def __init__(self):
        super(SubConstantNoBatch, self).__init__()
        self.register_buffer('y', torch.ones((3, 10, 10)))

    def forward(self, x):
        return x - self.y

    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10, 10)])
def test_sub_constant_nobatch():
    return SubConstantNoBatch()


class SubConstantBatch(torch.nn.Module):
    def __init__(self):
        super(SubConstantBatch, self).__init__()
        self.register_buffer('y', torch.ones((1, 3, 10, 10)))

    def forward(self, x):
        return x - self.y

    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10, 10)])
def test_sub_constant_batch():
    return SubConstantBatch()
