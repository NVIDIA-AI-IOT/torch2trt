from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test

@tensorrt_converter('torch.nn.Upsample.forward')
def convert_upsample(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return

    layer = ctx.network.add_resize(
            input=input_trt)

    shape = module.size
    if shape != None:
        if isinstance(shape, list):
            if len(shape) == 2:
                shape  = (1,shape[0],shape[1])
            if len(shape) == 3:
                shape = tuple(shape)
        else:
            shape = (1,shape,shape)
        layer.shape = shape

    scales = module.scale_factor
    if scales != None:
        if not isinstance(scales, tuple):
            scales = (1,scales,scales )
        layer.scales = scales

    resize_mode = module.mode
    if resize_mode.lower() in ["linear","bilinear","trilinear"]:
        layer.resize_mode = trt.ResizeMode.LINEAR
    else:
        layer.resize_mode=trt.ResizeMode.NEAREST

    align_corners = module.align_corners
    if align_corners != None:
        layer.align_corners = align_corners
    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1,1,2,2)])
def test_nearest_mode():
    return torch.nn.Upsample(scale_factor=2, mode="nearest")

@add_module_test(torch.float32, torch.device('cuda'), [(1,1,2,2)])
def test_bilinear_mode():
    return torch.nn.Upsample(scale_factor=2, mode="bilinear",align_corners=False)

@add_module_test(torch.float32, torch.device('cuda'), [(1,1,2,2)])
def test_align_corner():
    return torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

@add_module_test(torch.float32, torch.device('cuda'), [(1,1,3,3)])
def test_bilinear_mode_odd_input_shape():
    return torch.nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False)

@add_module_test(torch.float32, torch.device('cuda'), [(1,1,2,2)])
def test_size_parameter():
    return torch.nn.Upsample(size=3,mode="nearest")

@add_module_test(torch.float32, torch.device('cuda'), [(1,1,3,3)])
def test_size_parameter_odd_input():
    return torch.nn.Upsample(size=6,mode="bilinear",align_corners=False)

