from torch2trt.torch2trt import *
import tensorrt as trt

@tensorrt_converter('torch2trt.qat_layers.quant_activation.IQuantReLU.forward')
def convert_QuantReLU(ctx):
    print("inside relu custom converter")
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    layer = ctx.network.add_activation(
        input=input_trt, type=trt.ActivationType.RELU)
    
    ## int 8 precision
    amax = module._input_quantizer.learned_amax
    layer.precision = trt.int8
    layer.set_output_type(0,trt.int8)
    out = layer.get_output(0)
    out.dynamic_range=(-amax,amax)

    output._trt = out
