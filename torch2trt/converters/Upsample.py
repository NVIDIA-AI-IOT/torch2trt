from torch2trt.module_test import add_module_test
from torch2trt.torch2trt import *


@tensorrt_converter("torch.nn.Upsample.forward")
def convert_batch_norm(ctx):
    module = ctx.method_args[0]

    # only support `nearest` or `linear` and align_corners=True
    assert (
        module.mode == "nearest"
        or module.mode.endswith("linear")
        and module.align_corners
    )

    input = ctx.method_args[1]
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return

    layer = ctx.network.add_resize(input_trt)
    layer.shape = list(output.shape[1:])
    layer.resize_mode = (
        trt.ResizeMode.NEAREST if module.mode == "nearest" else trt.ResizeMode.LINEAR
    )
    layer.align_corners = module.align_corners

    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device("cuda"), [(1, 32, 64)])
def test_Upsample_1d_size():
    return torch.nn.Upsample(size=100)


@add_module_test(torch.float32, torch.device("cuda"), [(1, 32, 64)])
def test_Upsample_1d_size_linear():
    return torch.nn.Upsample(size=100, mode="linear", align_corners=True)


@add_module_test(torch.float32, torch.device("cuda"), [(1, 32, 64)])
def test_Upsample_1d_scale():
    return torch.nn.Upsample(scale_factor=1.7)


@add_module_test(torch.float32, torch.device("cuda"), [(1, 32, 64)])
def test_Upsample_1d_scale_linear():
    return torch.nn.Upsample(scale_factor=1.7, mode="linear", align_corners=True)


@add_module_test(torch.float32, torch.device("cuda"), [(1, 32, 64, 128)])
def test_Upsample_2d_size():
    return torch.nn.Upsample(size=(100, 256))


@add_module_test(torch.float32, torch.device("cuda"), [(1, 32, 64, 128)])
def test_Upsample_2d_size_linear():
    return torch.nn.Upsample(size=(100, 256), mode="bilinear", align_corners=True)


@add_module_test(torch.float32, torch.device("cuda"), [(1, 32, 64, 128)])
def test_Upsample_2d_scale():
    return torch.nn.Upsample(scale_factor=(1.7, 2))


@add_module_test(torch.float32, torch.device("cuda"), [(1, 32, 64, 128)])
def test_Upsample_2d_scale_linear():
    return torch.nn.Upsample(scale_factor=(1.7, 2), mode="bilinear", align_corners=True)


@add_module_test(torch.float32, torch.device("cuda"), [(1, 16, 32, 64, 64)])
def test_Upsample_3d_size():
    return torch.nn.Upsample(size=(48, 100, 128))


@add_module_test(torch.float32, torch.device("cuda"), [(1, 16, 32, 64, 64)])
def test_Upsample_3d_size_linear():
    return torch.nn.Upsample(size=(48, 100, 128), mode="trilinear", align_corners=True)


@add_module_test(torch.float32, torch.device("cuda"), [(1, 16, 32, 64, 64)])
def test_Upsample_3d_scale():
    return torch.nn.Upsample(scale_factor=(1.3, 1.7, 2))


@add_module_test(torch.float32, torch.device("cuda"), [(1, 16, 32, 64, 64)])
def test_Upsample_3d_scale_linear():
    return torch.nn.Upsample(
        scale_factor=(1.3, 1.7, 2), mode="trilinear", align_corners=True
    )
