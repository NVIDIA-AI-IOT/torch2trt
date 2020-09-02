from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test

@tensorrt_converter('torch.nn.functional.batch_norm', enabled=trt_version() >= '7.0')
def convert_batch_norm_trt7(ctx):

    input = get_arg(ctx, 'input', pos=0, default=None) 
    running_mean = get_arg(ctx, 'running_mean', pos=1, default=None) 
    running_var = get_arg(ctx, 'running_var', pos=2, default=None) 

    weight = get_arg(ctx, 'weight', pos=3, default=None) 
    bias = get_arg(ctx, 'bias', pos=4, default=None) 
    eps = get_arg(ctx, 'eps', pos=7, default=10e-6) 

    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    
    scale = weight.detach().cpu().numpy() / np.sqrt(running_var.detach().cpu().numpy() + eps)
    bias = bias.detach().cpu().numpy() - running_mean.detach().cpu().numpy() * scale
    power = np.ones_like(scale)
    
    layer = ctx.network.add_scale_nd(input_trt, trt.ScaleMode.CHANNEL, bias, scale, power, 0)
    output._trt = layer.get_output(0)



@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3)], enabled=trt_version() >= '7.0')
def test_batch_norm_2d_trt7():
    return torch.nn.BatchNorm2d(10)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3, 3)], enabled=trt_version() >= '7.0')
def test_batch_norm_3d_2_trt7():
    return torch.nn.BatchNorm3d(10)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 32, 2, 36, 47)], enabled=trt_version() >= '7.0')
def test_batch_norm_3d_trt7():
    return torch.nn.BatchNorm3d(32)
    
