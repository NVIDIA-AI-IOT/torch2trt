from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test
from torch2trt.qat_layers.quant_pooling import IQuantMaxPool2d

@tensorrt_converter('IQuantMaxPool2d.forward', enabled=trt_version() >= '7.0')
def convert_quant_maxpool2d(ctx):
    # parse args
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    kernel_size = module.kernel_size
    stride = module.stride
    padding =module.padding
    dilation = module.dilation
    ceil_mode = module.ceil_mode
    
    # get input trt tensor (or create constant if it doesn't exist)
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    
    output = ctx.method_return

    # get kernel size
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, ) * 2

    # get stride
    if not isinstance(stride, tuple):
        stride = (stride, ) * 2

    # get padding
    if not isinstance(padding, tuple):
        padding = (padding, ) * 2

    layer = ctx.network.add_pooling(
        input=input_trt, type=trt.PoolingType.MAX, window_size=kernel_size)
    
    layer.stride = stride
    layer.padding = padding
    
    if ceil_mode:
        layer.padding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP
    
    if ctx.qat_mode:
        amax = module._input_quantizer.learned_amax
        layer.precision = trt.int8
        layer.set_output_type(0,trt.int8)
        out = layer.get_output(0)
        out.dynamic_range=(-amax,amax)

    output._trt = layer.get_output(0)
    
    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 6)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 5, 7)])
def test_MaxPool2d_without_ceil_mode():
    return torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 6)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 5, 7)])
def test_MaxPool2d_with_ceil_mode():
    return torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
