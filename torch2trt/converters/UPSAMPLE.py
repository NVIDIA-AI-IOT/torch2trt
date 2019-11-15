from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test

@tensorrt_converter('torch.nn.Upsample.forward')
def convert_upsample(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return

    shape = module.size
    if not isinstance(shape, tuple):
        shape = (shape, ) * 2

    scales = module.scale_factor
    if not isinstance(scales, tuple):
        scales = (scales, ) * 2

    resize_mode = module.mode
    align_corners = module.align_corners

    layer = ctx.network.add_resize(
            input=input_trt)

    layer.shape = shape
    layer.scales = scales
    layer.resize_mode=resize_mode
    layer.align_corners = align_corners

    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1,1,2,2)])
def test_nearest_mode():
    return torch.nn.Upsample(scale_factor=2, mode="nearest")

@add_module_test(torch.float32, torch.device('cuda'), [(1,1,2,2)])
def test_bilinear_mode():
    return torch.nn.Upsample(scale_factor=2, mode="bilinear")

@add_module_test(torch.float32, torch.device('cuda'), [(1,1,2,2)])
def test_align_corner():
    return torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

@add_module_test(torch.float32, torch.device('cuda'), [(1,1,3,3)])
def test_bilinear_mode_odd_input_shape():
    return torch.nn.Upsample(scale_factor=2,mode="bilinear")

@add_module_test(torch.float32, torch.device('cuda'), [(1,1,2,2)])
def test_size_parameter():
    return torch.nn.Upsample(size=6,mode="bilinear")
   
@add_module_test(torch.float32, torch.device('cuda'), [(1,1,3,3)])
def test_size_parameter_odd_input():
    return torch.nn.Upsample(size=6,mode="bilinear")
