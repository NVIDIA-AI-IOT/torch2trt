from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.nn.functional.silu')
def convert_silu(ctx):
    input = get_arg(ctx, 'input', pos=0, default=None)
    output = ctx.method_return
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    
    layer = ctx.network.add_activation(input_trt, trt.ActivationType.SIGMOID)
    layer = ctx.network.add_elementwise(input_trt, layer.get_output(0), trt.ElementWiseOperation.PROD)
    
    output._trt = layer.get_output(0)
    
    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 5)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3, 3)])
def test_silu():
    return torch.nn.SiLU()