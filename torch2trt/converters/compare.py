from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test

def convert_elementwise(ctx, op):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], max(len(input_a_trt.shape), len(input_b_trt.shape)))
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


class TensorGTScalar(torch.nn.Module):
    def __init__(self, scalar):
        super().__init__()
        self.scalar = scalar

    def forward(self, tensor):
        return tensor > self.scalar


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 6, 6)], enabled=trt_version() >= '7.0')
def test_tensor_gt_scalar():
    return TensorGTScalar(0.1)


class ScalarGTTensor(torch.nn.Module):
    def __init__(self, scalar):
        super().__init__()
        self.scalar = scalar

    def forward(self, tensor):
        return self.scalar > tensor


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 6, 6)], enabled=trt_version() >= '7.0')
def test_scalar_gt_scalar():
    return ScalarGTTensor(0.1)


class TensorLTScalar(torch.nn.Module):
    def __init__(self, scalar):
        super().__init__()
        self.scalar = scalar

    def forward(self, tensor):
        return tensor < self.scalar


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 6, 6)], enabled=trt_version() >= '7.0')
def test_tensor_lt_scalar():
    return TensorLTScalar(0.1)


class ScalarLTTensor(torch.nn.Module):
    def __init__(self, scalar):
        super().__init__()
        self.scalar = scalar

    def forward(self, tensor):
        return self.scalar < tensor


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 6, 6)], enabled=trt_version() >= '7.0')
def test_scalar_lt_tensor():
    return ScalarLTTensor(0.1)


class TensorEQScalar(torch.nn.Module):
    def __init__(self, scalar):
        super().__init__()
        self.scalar = scalar

    def forward(self, tensor):
        return tensor == self.scalar


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 6, 6)], enabled=trt_version() >= '7.0')
def test_tensor_eq_scalar():
    return TensorEQScalar(0.1)


class ScalarEQTensor(torch.nn.Module):
    def __init__(self, scalar):
        super().__init__()
        self.scalar = scalar

    def forward(self, tensor):
        return self.scalar == tensor


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 6, 6)], enabled=trt_version() >= '7.0')
def test_scalar_eq_tensor():
    return ScalarEQTensor(0.1)
