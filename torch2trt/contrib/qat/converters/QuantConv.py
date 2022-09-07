from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test
import tensorrt as trt

@tensorrt_converter('torch2trt.contrib.qat.layers.quant_conv.QuantConv2d.forward', enabled=trt_version() >= '8.0') 
def convert_QuantConv(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return

    input_dim = input.dim() - 2

    kernel_size = module.kernel_size
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, ) * input_dim

    stride = module.stride
    if not isinstance(stride, tuple):
        stride = (stride, ) * input_dim

    padding = module.padding
    if not isinstance(padding, tuple):
        padding = (padding, ) * input_dim

    dilation = module.dilation
    if not isinstance(dilation, tuple):
        dilation = (dilation, ) * input_dim

    kernel = module.weight.detach().cpu().numpy()
    
    bias = None #trt.Weights(torch_dtype_to_trt(module.weight.dtype))
    if module.bias is not None:
        bias = module.bias.detach().cpu().numpy()
    
    ## Add quantizatin and dequantization nodes for inputs and weights
    # Input Layer quantization
    input_quantizer = ctx.network.add_quantize(
            input=input_trt,
            scale=module._input_quantizer.quant_scale)
    
    input_dequantizer = ctx.network.add_dequantize(
            input = input_quantizer.get_output(0),
            scale = module._input_quantizer.quant_scale)

    if hasattr(module._input_quantizer.quant_axis):
        input_quantizer.axis = module._input_quantizer.quant_axis.to(torch.long).item()
        input_dequantizer.axis = module._input_quantizer.quant_axis.to(torch.long).item()

    # Weight quantization
    kernel_trt = add_missing_trt_tensors(ctx.network, [kernel])[0]

    weight_quantizer = ctx.network.add_quantize(
            input=kernel_trt,
            scale=module._weight_quantizer.quant_scale)
    
    weight_dequantizer = ctx.network.add_dequantize(
            input = weight_quantizer.get_output(0),
            scale = module._weight_quantizer.quant_scale)

    if hasattr(module._weight_quantizer.quant_axis):
        weight_quantizer.axis = module._weight_quantizer.quant_axis.to(torch.long).item()
        weight_dequantizer.axis = module._weight_quantizer.quant_axis.to(torch.long).item()

    layer = ctx.network.add_convolution_nd(
        input=input_dequantizer.get_output(0),
        num_output_maps=module.out_channels,
        kernel_shape=kernel_size,
        kernel=weight_dequantizer.get_output(0),
        bias=bias)
    layer.stride_nd = stride
    layer.padding_nd = padding
    layer.dilation_nd = dilation

    if module.groups is not None:
        layer.num_groups = module.groups
    
    output._trt = layer.get_output(0)


