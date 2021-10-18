from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter("torch.nn.functional.adaptive_max_pool3d")
def convert_adaptive_max_pool3d(ctx):
    input = ctx.method_args[0]
    output = ctx.method_return

    output_size = ctx.method_args[1]
    if isinstance(output_size, int):
        output_size = (output_size,) * 3

    stride = (
        input._trt.shape[-3] // output_size[-3],
        input._trt.shape[-2] // output_size[-2],
        input._trt.shape[-1] // output_size[-1],
    )
    
    kernel_size = stride
    layer = ctx.network.add_pooling_nd(
        input=input._trt, type=trt.PoolingType.MAX, window_size=kernel_size
    )
    layer.stride_nd = stride

    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 16, 224, 224)])
def test_adaptive_max_pool3d_1x1x1():
    return torch.nn.AdaptiveMaxPool3d((1, 1, 1))


@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 16, 224, 224)])
def test_adaptive_max_pool3d_2x2x2():
    return torch.nn.AdaptiveMaxPool3d((2, 2, 2))


@add_module_test(torch.float32, torch.device("cuda"), [(1, 3, 16, 224, 224)])
def test_adaptive_max_pool3d_3x3x3():
    return torch.nn.AdaptiveMaxPool3d((3, 3, 3))
