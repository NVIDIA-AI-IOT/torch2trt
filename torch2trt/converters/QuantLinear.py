from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test
import tensorrt as trt

@tensorrt_converter('torch.nn.Linear.forward')
def convert_Linear(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    
    # setting dynamic range for input tensor
    if ctx.qat_mode:
        iquant_amax = module._input_quantizer.learned_amax
        input_trt.dynamic_range=(-iquant_amax,iquant_amax)

    # reshape to ...xNx1x1
    layer = ctx.network.add_shuffle(input_trt)
    layer.reshape_dims = tuple(input_trt.shape) + (1, 1) 

    bias = trt.Weights(torch_dtype_to_trt(module.weight.dtype))
    if module.bias is not None:
        bias = module.bias.detach().cpu().numpy()
        
    # add fully connected
    layer = ctx.network.add_fully_connected(
        input=layer.get_output(0),
        num_outputs=module.out_features,
        kernel=module.weight.detach().cpu().numpy(),
        bias=bias)

    # reshape back to N
    layer = ctx.network.add_shuffle(layer.get_output(0))
    layer.reshape_dims = tuple(output.shape[1:])
    
    if ctx.qat_mode:
        layer.precision = trt.int8
        layer.set_output_type(0,trt.int8) 

        linear_out = layer.get_output(0)
        weight_amax = module._weight_quantizer.learned_amax 
        linear_out.dynamic_range = (-weight_amax,weight_amax) 

    output._trt = linear_out


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 10)])
def test_Linear_basic():
    return torch.nn.Linear(10, 5)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 10)])
def test_Linear_no_bias():
    return torch.nn.Linear(10, 5, bias=False)
