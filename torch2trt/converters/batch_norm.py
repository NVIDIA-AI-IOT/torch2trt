from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test

@tensorrt_converter('torch.nn.functional.batch_norm')
def convert_batch_norm(ctx):

    input = get_arg(ctx, 'input', pos=0, default=None) 
    running_mean = get_arg(ctx, 'running_mean', pos=1, default=None) 
    running_var = get_arg(ctx, 'running_var', pos=2, default=None) 

    weight = get_arg(ctx, 'weight', pos=3, default=None) 
    bias = get_arg(ctx, 'bias', pos=4, default=None) 
    eps = get_arg(ctx, 'eps', pos=7, default=10e-6) 

    input_trt = trt_(ctx.network, input)
    output = ctx.method_return
    
    scale = weight.detach().cpu().numpy() / np.sqrt(running_var.detach().cpu().numpy() + eps)
    bias = bias.detach().cpu().numpy() - running_mean.detach().cpu().numpy() * scale
    power = np.ones_like(scale)
    
    layer = ctx.network.add_scale_nd(input_trt, trt.ScaleMode.CHANNEL, bias, scale, power, 0)
    output._trt = layer.get_output(0)



@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3)])
def test_batch_norm_2d():
    return torch.nn.BatchNorm2d(10)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3, 3)])
def test_batch_norm_3d_2():
    return torch.nn.BatchNorm3d(10)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 32, 2, 36, 47)])
def test_batch_norm_3d():
    return torch.nn.BatchNorm3d(32)
    