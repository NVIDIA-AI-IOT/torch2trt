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

    kernel = module.weight.detach()#.cpu().numpy()
    
    bias = None #trt.Weights(torch_dtype_to_trt(module.weight.dtype))
    if module.bias is not None:
        bias = module.bias.detach().cpu().numpy()
    
    ## Add quantizatin and dequantization nodes for inputs and weights
    # Input Layer quantization
    # Adding scale as ITensor
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
    
    # Weight quantization
    ## currently not using weight quantizer, waiting for a resolution on the issue.
    kernel_trt = ctx.network.add_constant(tuple(kernel.shape),kernel.cpu().numpy())
    scale_trt = ctx.network.add_constant(tuple(module._weight_quantizer.quant_scale.shape),module._weight_quantizer.quant_scale.detach().cpu().numpy()) 
    weight_quantizer = ctx.network.add_quantize(
            input=kernel_trt.get_output(0),
            scale=scale_trt.get_output(0))
    
    if hasattr(module._weight_quantizer,'quant_axis'):
        weight_quantizer.axis = module._weight_quantizer.quant_axis.to(torch.long).item()
    else:
        weight_quantizer.axis = 0

    weight_dequantizer = ctx.network.add_dequantize(
            input = weight_quantizer.get_output(0),
            scale = scale_trt.get_output(0))
    
    if hasattr(module._weight_quantizer,'quant_axis'):
        weight_dequantizer.axis = module._weight_quantizer.quant_axis.to(torch.long).item()
    else:
        weight_dequantizer.axis = 0

    # Creating dummy kernel to pass creation of conv layer. will substitute this with an actual kernel later
    dummy_kernel = trt.Weights()

    layer = ctx.network.add_convolution_nd(
        input=input_dequantizer.get_output(0),
        num_output_maps=module.out_channels,
        kernel_shape=kernel_size,
        kernel=dummy_kernel, #weight_dequantizer.get_output(0),
        bias=bias)
    layer.stride_nd = stride
    layer.padding_nd = padding
    layer.dilation_nd = dilation
    layer.set_input(1,weight_dequantizer.get_output(0))

    if module.groups is not None:
        layer.num_groups = module.groups
    
    output._trt = layer.get_output(0)


