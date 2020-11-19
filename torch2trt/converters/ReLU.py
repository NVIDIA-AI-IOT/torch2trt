from torch2trt.torch2trt import *


@tensorrt_converter('torch.nn.ReLU.forward')
def convert_ReLU(ctx):
    input = ctx.method_args[1]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    layer = ctx.network.add_activation(
        input=input_trt, type=trt.ActivationType.RELU)
    
    amax = 5 
    layer.precision = trt.int8
    layer.set_output_type(0,trt.int8)
    out = layer.get_output(0)
    out.dynamic_range=(-amax,amax)
 
    output._trt = out
