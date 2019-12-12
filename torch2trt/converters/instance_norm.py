from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.instance_norm')
@tensorrt_converter('torch.nn.functional.instance_norm')
def convert_instance_norm(ctx):
    input = get_arg(ctx, 'input', pos=0, default=None)
    running_mean = get_arg(ctx, 'running_mean', pos=1, default=None)
    running_var = get_arg(ctx, 'running_var', pos=2, default=None)
    weight = get_arg(ctx, 'weight', pos=3, default=None)
    bias = get_arg(ctx, 'bias', pos=4, default=None)
    use_input_stats = get_arg(ctx, 'use_input_stats', pos=5, default=True)
    momentum = get_arg(ctx, 'momentum', pos=6, default=0.1)
    eps = get_arg(ctx, 'eps', pos=7, default=1e-05)
    output = ctx.method_return
    
    
    # CASE 1 - USING RUNNING STATISTICS
    if not use_input_stats:
        
        # equivalent to batch norm
        scale = 1.0 / np.sqrt(running_var.detach().cpu().numpy() + eps)
        offset = -running_mean.detach().cpu().numpy() * scale
        power = np.ones_like(scale)
        
        if weight is not None:
            scale *= weight.detach().cpu().numpy()
            
        if bias is not None:
            offset += bias.detach().cpu().numpy()
            
        input_trt = input._trt
        
        # force input to be NCHW if it is not
        if input.ndim != 4:
            
            layer = ctx.network.add_shuffle(input_trt)
            
            if input.ndim == 2:
                layer.reshape_dims = (input.shape[1], 1, 1)  # NC -> NCHW
            elif input.ndim == 3:
                layer.reshape_dims = (input.shape[1], input.shape[2], 1)  # NCH -> NCHW
            elif input.ndim == 5:
                layer.reshape_dims = (input.shape[1], input.shape[2], input.shape[3] * input.shape[4])  # NCHWD -> NCHW
                
            input_trt = layer.get_output(0)
        
        layer = ctx.network.add_scale(input_trt, trt.ScaleMode.CHANNEL, offset, scale, power)
        
        if input.ndim != 4:
            
            layer = ctx.network.add_shuffle(layer.get_output(0))
            layer.reshape_dims = tuple(output.shape[1:])
    
        output._trt = layer.get_output(0)
        
    # CASE 2 - USING INPUT STATS
    else:
        
        raise NotImplementedError
        
        
@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3)])
def test_instance_norm_1d_track_stats():
    return torch.nn.InstanceNorm1d(10, track_running_stats=True)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3)])
def test_instance_norm_2d_track_stats():
    return torch.nn.InstanceNorm2d(10, track_running_stats=True)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3, 3)])
def test_instance_norm_3d_track_stats():
    return torch.nn.InstanceNorm3d(10, track_running_stats=True)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3)])
def test_instance_norm_1d_track_stats_affine():
    return torch.nn.InstanceNorm1d(10, affine=True, track_running_stats=True)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3)])
def test_instance_norm_2d_track_stats_affine():
    return torch.nn.InstanceNorm2d(10, affine=True, track_running_stats=True)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3, 3)])
def test_instance_norm_3d_track_stats_affine():
    return torch.nn.InstanceNorm3d(10, affine=True, track_running_stats=True)