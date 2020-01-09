from torch2trt.torch2trt import *


@tensorrt_converter('torch.nn.ReLU.forward')
def convert_ReLU(ctx):
    input = ctx.method_args[1]
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return
    layer = ctx.network.add_activation(
        input=input_trt, type=trt.ActivationType.RELU)
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.relu')
@tensorrt_converter('torch.relu_')
@tensorrt_converter('torch.nn.functional.relu')
@tensorrt_converter('torch.nn.functional.relu_')
def convert_relu(ctx):
    ctx.method_args = (torch.nn.ReLU(),) + ctx.method_args
    convert_ReLU(ctx)
    
