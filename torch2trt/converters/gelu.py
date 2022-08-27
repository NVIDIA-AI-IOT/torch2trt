from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test
import math


@tensorrt_converter('torch.nn.functional.gelu')
def convert_gelu_v1(ctx):
    # approximate equation 1 from paper
    input = get_arg(ctx, 'input', 0, None)
    output = ctx.method_return
    
    x, c05, c1, cs2pi, c044, c3 = add_missing_trt_tensors(
        ctx.network,
        [input, 0.5, 1.0, math.sqrt(2.0 / math.pi), 0.044715, 3.0]
    )
    
    x, c05, c1, cs2pi, c044, c3 = broadcast_trt_tensors(
        ctx.network, 
        [x, c05, c1, cs2pi, c044, c3], 
        len(output.shape)
    )
    
    y = ctx.network.add_elementwise(x, c3, trt.ElementWiseOperation.POW).get_output(0)
    y = ctx.network.add_elementwise(y, c044, trt.ElementWiseOperation.PROD).get_output(0)
    y = ctx.network.add_elementwise(x, y, trt.ElementWiseOperation.SUM).get_output(0)
    y = ctx.network.add_elementwise(y, cs2pi, trt.ElementWiseOperation.PROD).get_output(0)
    y = ctx.network.add_activation(y, trt.ActivationType.TANH).get_output(0)
    y = ctx.network.add_elementwise(y, c1, trt.ElementWiseOperation.SUM).get_output(0)
    y = ctx.network.add_elementwise(x, y, trt.ElementWiseOperation.PROD).get_output(0)
    y = ctx.network.add_elementwise(y, c05, trt.ElementWiseOperation.PROD).get_output(0)
    
    output._trt = y
    
    
# @tensorrt_converter('torch.nn.functional.gelu')
# def convert_gelu_v2(ctx):
#     # approximate equation 1 from paper
#     input = get_arg(ctx, 'input', 0, None)
#     output = ctx.method_return
    
#     x, c1702 = add_missing_trt_tensors(
#         ctx.network,
#         [input, 1.702]
#     )
    
#     x, c1702 = broadcast_trt_tensors(
#         ctx.network, 
#         [x, c1702], 
#         len(output.shape) - 1
#     )
    
#     y = ctx.network.add_elementwise(x, c1702, trt.ElementWiseOperation.PROD).get_output(0)
#     y = ctx.network.add_activation(y, trt.ActivationType.SIGMOID).get_output(0)
#     y = ctx.network.add_elementwise(x, y, trt.ElementWiseOperation.PROD).get_output(0)
    
#     output._trt = y


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3, 3)])
def test_silu():
    return torch.nn.GELU()