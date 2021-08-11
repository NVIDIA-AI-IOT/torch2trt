from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


def convert_cast(ctx):
    """
    A simple converter for supporting casting operations.

    IMPORTANT: Note that because TensorRT does not support
    64 bit data types, .long() is not included.
    """
    input_tensor = ctx.method_args[0]
    layer = ctx.network.add_identity(input_tensor._trt)
    output = ctx.method_return
    output._trt = layer.get_output(0)


@tensorrt_converter("torch.float")
@tensorrt_converter("torch.Tensor.float")
def convert_float(ctx):
    convert_cast(ctx)


@tensorrt_converter("torch.bool")
@tensorrt_converter("torch.Tensor.bool")
def convert_bool(ctx):
    convert_cast(ctx)


@tensorrt_converter("torch.float")
@tensorrt_converter("torch.Tensor.float")
def convert_bool(ctx):
    convert_cast(ctx)


class ConvertToFloat(torch.nn.Module):
    def __init__(self):
        super(ConvertToFloat, self).__init__()

    def forward(self, x):
        return x.float()


class ConvertToInt(torch.nn.Module):
    def __init__(self):
        super(ConvertToInt, self).__init__()

    def forward(self, x):
        return x.int()


class ConvertToBool(torch.nn.Module):
    def __init__(self):
        super(ConvertToBool, self).__init__()

    def forward(self, x):
        return x.bool()


@add_module_test(torch.bool, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.int32, torch.device('cuda'), [(1, 3, 3)])
def test_float_casting():
    return ConvertToFloat()


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.bool, torch.device('cuda'), [(1, 3, 3)])
def test_int_casting():
    return ConvertToInt()


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.int32, torch.device('cuda'), [(1, 3, 3)])
def test_bool_casting():
    return ConvertToBool()
