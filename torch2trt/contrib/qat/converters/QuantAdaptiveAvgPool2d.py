from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test

@tensorrt_converter('torch2trt.contrib.qat.layers.quant_pooling.QuantAdaptiveAvgPool2d.forward', enabled=trt_version() >= '8.0') 
def convert_QuantAdaptiveAvgPool2d(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    
    output_size = module.adaptive_avg_pool2d.output_size
    if not isinstance(output_size, tuple):
        output_size = (output_size, ) * 2

    stride = (input_trt.shape[-2] // output_size[-2], input_trt.shape[-1] // output_size[-1])
    kernel_size = stride

    #Add quantization and dequantization nodes for input
    scale_trt = ctx.network.add_constant(tuple(module._input_quantizer.quant_scale.shape),module._input_quantizer.quant_scale.detach().cpu().numpy())
    input_quantizer = ctx.network.add_quantize(
            input=input_trt,
            scale=scale_trt.get_output(0))
    
    if hasattr(module._input_quantizer,'quant_axis'):
        input_quantizer.axis = module._input_quantizer.quant_axis.to(torch.long).item()
    else:
        input_quantizer.axis=0

    input_dequantizer = ctx.network.add_dequantize(
            input = input_quantizer.get_output(0),
            scale = scale_trt.get_output(0))

    if hasattr(module._input_quantizer,'quant_axis'):
        input_dequantizer.axis = module._input_quantizer.quant_axis.to(torch.long).item()
    else:
        input_dequantizer.axis=0
    
    layer = ctx.network.add_pooling(
        input=input_dequantizer.get_output(0), type=trt.PoolingType.AVERAGE, window_size=kernel_size)
    
    layer.stride = stride
    layer.name = "QuantAvgPool2d"
    output._trt = layer.get_output(0)

