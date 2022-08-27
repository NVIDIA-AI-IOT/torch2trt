from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.nn.functional.linear')
def convert_Linear(ctx):
    input = ctx.method_args[0]
    weight = get_arg(ctx, 'weight', 1, None)
    bias = get_arg(ctx, 'bias', 2, None)
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return

    # reshape to ...xNx1x1
    layer = ctx.network.add_shuffle(input_trt)
    layer.reshape_dims = tuple([0]*input.ndim) + (1, 1) 

    bias_trt = trt.Weights(torch_dtype_to_trt(weight.dtype))
    if bias is not None:
        bias_trt = bias.detach().cpu().numpy()
        
    # add fully connected
    layer = ctx.network.add_fully_connected(
        input=layer.get_output(0),
        num_outputs=int(weight.shape[0]),
        kernel=weight.detach().cpu().numpy(),
        bias=bias_trt)

    # reshape back to N
    layer = ctx.network.add_shuffle(layer.get_output(0))
    layer.reshape_dims = tuple([0] * output.ndim)

    output._trt = layer.get_output(0)
    

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 10)])
def test_Linear_basic():
    return torch.nn.Linear(10, 5)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(2, 3, 4, 10)], max_batch_size=2)
def test_Linear_no_bias():
    return torch.nn.Linear(10, 5, bias=False)
