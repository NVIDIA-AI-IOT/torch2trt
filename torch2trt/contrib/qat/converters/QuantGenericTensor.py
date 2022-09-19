from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test
import tensorrt as trt

@tensorrt_converter('torch2trt.contrib.qat.layers.quant_generic_tensor.QuantGenericTensor.forward', enabled=trt_version() >= '8.0') 
def convert_GenericTensor(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return

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
    
    output._trt = input_dequantizer.get_output(0)

