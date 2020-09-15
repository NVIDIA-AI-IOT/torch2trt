from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test

def convert_elementwise(ctx, op):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape) - 1)
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, op)
    output._trt = layer.get_output(0)

@tensorrt_converter('torch.gt', enabled=trt_version() >= '7.0')
@tensorrt_converter('torch.Tensor.__gt__', enabled=trt_version() >= '7.0')
def convert_gt(ctx):
    return convert_elementwise(ctx, trt.ElementWiseOperation.GREATER)

@tensorrt_converter('torch.lt', enabled=trt_version() >= '7.0')
@tensorrt_converter('torch.Tensor.__lt__', enabled=trt_version() >= '7.0')
def convert_gt(ctx):
    return convert_elementwise(ctx, trt.ElementWiseOperation.LESS)

@tensorrt_converter('torch.eq', enabled=trt_version() >= '7.0')
@tensorrt_converter('torch.Tensor.__eq__', enabled=trt_version() >= '7.0')
def convert_gt(ctx):
    return convert_elementwise(ctx, trt.ElementWiseOperation.EQUAL)

class GT(torch.nn.Module):
    def __init__(self):
        super(GT, self).__init__()

    def forward(self, x, y):
        return x > y

class LT(torch.nn.Module):
    def __init__(self):
        super(LT, self).__init__()

    def forward(self, x, y):
        return x < y

class EQ(torch.nn.Module):
    def __init__(self):
        super(EQ, self).__init__()

    def forward(self, x, y):
        return x == y


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 6, 6), (1, 3, 6, 6)], enabled=trt_version() >= '7.0')
def test_gt_basic():
    return GT()

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 6, 6), (1, 3, 6, 6)], enabled=trt_version() >= '7.0')
def test_gt_basic():
    return LT()

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 6, 6), (1, 3, 6, 6)], enabled=trt_version() >= '7.0')
def test_gt_basic():
    return EQ()
