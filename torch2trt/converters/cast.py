from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


def _key_sanity_check(mode_key, torch2trt_properties):
    """
    Raise an error if the given key does not exist.
    This error will be raised as a warning in case
    in case "mode-related" keys change in the future.
    Args:
        mode_key: A string key for the quantization mode.
        E.g. ("int8_mode", "fp16_mode")
        torch2trt_properties: A python dictionary containing
        the torch2trt properties such as "int8_mode".
    """
    if mode_key not in torch2trt_properties:
        raise KeyError("{} is not a valid torch2trt property. "
                       "Check the torch2trt API for any changes.".format(mode_key))


def convert_cast(ctx):
    """
    A simple converter for supporting casting operations.

    IMPORTANT: Note that because TensorRT does not support
    64 bit data types, .long() will not be supported
    """
    input_tensor = ctx.method_args[0]
    layer = ctx.network.add_identity(input_tensor._trt)
    trt_kwargs = ctx.torch2trt_kwargs

    # Sanity checks for debugging in case torch2trt property keys change
    int8_mode_key, fp16_mode_key = "int8_mode", "fp16_mode"
    _key_sanity_check(int8_mode_key, trt_kwargs)
    _key_sanity_check(fp16_mode_key, trt_kwargs)

    is_int8_mode = trt_kwargs[int8_mode_key]
    is_fp16_mode = trt_kwargs[fp16_mode_key]
    if is_int8_mode:
        layer.precision = trt.int8
        layer.set_output_type(0, trt.int8)
    elif is_fp16_mode:
        layer.precision = trt.float16
        layer.set_output_type(0, trt.float16)

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
