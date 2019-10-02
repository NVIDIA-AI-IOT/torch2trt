from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.nn.BatchNorm1d.forward')
def convert_BatchNorm2d(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return
    
    scale = module.weight.detach().cpu().numpy() / np.sqrt(module.running_var.detach().cpu().numpy() + module.eps)
    bias = module.bias.detach().cpu().numpy() - module.running_mean.detach().cpu().numpy() * scale
    power = np.ones_like(scale)
    
    # reshape to 2D
    layer = ctx.network.add_shuffle(input_trt)
    
    if len(input.shape) == 2:
        layer.reshape_dims = (input.shape[1], 1, 1)
    else:
        layer.reshape_dims = (input.shape[1], input.shape[2], 1)
    
    layer = ctx.network.add_scale(layer.get_output(0), trt.ScaleMode.CHANNEL, bias, scale, power)

    # reshape back to 1D
    layer = ctx.network.add_shuffle(layer.get_output(0))
    layer.reshape_dims = tuple(output.shape[1:])
    
    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3)])
def test_BatchNorm1d_basic():
    return torch.nn.BatchNorm1d(10)