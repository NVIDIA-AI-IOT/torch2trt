from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


def convert_cast(ctx):
    """
    A simple converter for supporting casting operations.

    IMPORTANT: Note that because TensorRT does not support
    64 bit data types, .long() will not be supported
    """
    input_tensor = ctx.method_args[0]
    layer = ctx.network.add_identity(input_tensor._trt)
    output = ctx.method_return
    output._trt = layer.get_output(0)


@tensorrt_converter("torch.Tensor.float")
def convert_float(ctx):
    convert_cast(ctx)


@tensorrt_converter("torch.Tensor.int")
def convert_int(ctx):
    convert_cast(ctx)


@tensorrt_converter("torch.Tensor.bool")
def convert_bool(ctx):
    convert_cast(ctx)


# Used for torch.Tensor.<cast> tests
# --------------------------------------------

class DotFloat(torch.nn.Module):
    def __init__(self):
        super(DotFloat, self).__init__()

    def forward(self, x):
        return x.float()


class DotInt(torch.nn.Module):
    def __init__(self):
        super(DotInt, self).__init__()

    def forward(self, x):
        return x.int()


class DotBool(torch.nn.Module):
    def __init__(self):
        super(DotBool, self).__init__()

    def forward(self, x):
        return x.bool()


@add_module_test(torch.bool, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.int32, torch.device('cuda'), [(1, 3, 3)])
def test_torch_float_cast():
    return DotFloat()


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.int32, torch.device('cuda'), [(1, 3, 3)])
def test_torch_int_cast():
    return DotInt()


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.int32, torch.device('cuda'), [(1, 3, 3)])
def test_torch_bool_cast():
    return DotBool()
