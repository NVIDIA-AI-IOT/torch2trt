import torch.nn as nn
from torch2trt.torch2trt import *                                 
from torch2trt.module_test import add_module_test

def has_group_norm_plugin():
    try:
        from torch2trt.plugins import GroupNormPlugin
        return True
    except:
        return False


def get_group_norm_plugin(num_groups, weight, bias, eps):
    from torch2trt.plugins import GroupNormPlugin
    PLUGIN_NAME = 'group_norm'
    registry = trt.get_plugin_registry()
    creator = [c for c in registry.plugin_creator_list if c.name == PLUGIN_NAME and c.plugin_namespace == 'torch2trt'][0]
    torch2trt_plugin = GroupNormPlugin(num_groups=num_groups, weight=weight, bias=bias, eps=eps)
    return creator.deserialize_plugin(PLUGIN_NAME, torch2trt_plugin.serializeToString())

@tensorrt_converter('torch.nn.GroupNorm.forward', has_group_norm_plugin())
def convert_group_norm_trt(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    num_groups = module.num_groups
    weight = module.weight
    bias = module.bias
    eps = module.eps
    input_trt = add_missing_trt_tensors(ctx.network, [input])
    output = ctx.method_return
    plugin = get_group_norm_plugin(num_groups, weight, bias, eps)

    layer = ctx.network.add_plugin_v2(input_trt, plugin)
    
    output._trt = layer.get_output(0)



@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 112, 112)], has_group_norm_plugin())
def test_group_norm_trt_g2_fp32():
    return torch.nn.GroupNorm(2, 10)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 112, 112)], has_group_norm_plugin())
def test_group_norm_trt_g2_eps_fp32():
    return torch.nn.GroupNorm(2, 10, eps=1e-4)


    
