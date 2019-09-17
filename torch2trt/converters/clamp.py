from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


def __add_clamp(network, trt_input, val, op):
    
    # create TensorRT constant for minimum value
    val_shape = (1, ) * len(trt_input.shape)  # broadcast all dimensions
    val_tensor = val * torch.ones(val_shape, dtype=torch_dtype_from_trt(trt_input.dtype)).cpu().numpy()
    val_trt = network.add_constant(val_shape, val_tensor)
    layer = network.add_elementwise(trt_input, val_trt.get_output(0), op)
    
    return layer

    
@tensorrt_converter('torch.clamp_min')
def convert_clamp_min(ctx):
    input = ctx.method_args[0]
    val = ctx.method_args[1]
    output = ctx.method_return
    
    layer = __add_clamp(ctx.network, input._trt, val, trt.ElementWiseOperation.MAX)
    
    output._trt = layer.get_output(0)

    
class ClampMin(torch.nn.Module):
    def forward(self, x):
        return torch.clamp_min(x, -0.1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_clamp_min():
    return ClampMin()

    
@tensorrt_converter('torch.clamp_max')
def convert_clamp_max(ctx):
    input = ctx.method_args[0]
    val = ctx.method_args[1]
    output = ctx.method_return
    
    layer = __add_clamp(ctx.network, input._trt, val, trt.ElementWiseOperation.MIN)
    
    output._trt = layer.get_output(0)
    

class ClampMax(torch.nn.Module):
    def forward(self, x):
        return torch.clamp_max(x, 0.1)
  

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_clamp_max():
    return ClampMax()

    
@tensorrt_converter('torch.clamp')
def convert_clamp(ctx):
    input = ctx.method_args[0]
    min_val = ctx.method_args[1]
    max_val = ctx.method_args[2]
    output = ctx.method_return
    
    layer = __add_clamp(ctx.network, input._trt, min_val, trt.ElementWiseOperation.MAX)
    layer = __add_clamp(ctx.network, layer.get_output(0), max_val, trt.ElementWiseOperation.MIN)
    
    output._trt = layer.get_output(0)
    

class Clamp(torch.nn.Module):
    def forward(self, x):
        return torch.clamp(x, -0.1, 0.1)
    
    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_clamp_max():
    return Clamp()