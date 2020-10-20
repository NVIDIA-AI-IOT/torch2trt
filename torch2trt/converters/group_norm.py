import torch.nn as nn
from torch2trt.torch2trt import *                                 
from torch2trt.module_test import add_module_test
import collections

def has_group_norm_plugin():
    try:
        #from torch2trt.plugins import GroupNormPlugin
        from torch2trt.group_norm_plugin import GroupNormPlugin
        return True
    except:
        return False


def get_group_norm_plugin(num_groups): #, num_channels, height_width, eps):
    #from torch2trt.plugins import GroupNormPlugin
    from torch2trt.group_norm_plugin import GroupNormPlugin
    PLUGIN_NAME = 'group_norm'
    registry = trt.get_plugin_registry()
    creator = [c for c in registry.plugin_creator_list if c.name == PLUGIN_NAME and c.plugin_namespace == 'torch2trt'][0]
    torch2trt_plugin = GroupNormPlugin(num_groups=num_groups)
    return creator.deserialize_plugin(PLUGIN_NAME, torch2trt_plugin.serializeToString())

@tensorrt_converter('torch.nn.GroupNorm.forward', enabled=trt_version() >= '7.0' and has_group_norm_plugin())
def convert_group_norm_trt7(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    num_groups = module.num_groups
    #num_channels = input.shape[1] # NCHW
    #height = input.shape[2]
    #width = input.shape[3]
    #eps = module.eps
    input_trt = add_missing_trt_tensors(ctx.network, [input])
    output = ctx.method_return
    plugin = get_group_norm_plugin(num_groups) #, num_channels, height*width, eps)

    layer = ctx.network.add_plugin_v2(input_trt, plugin)
    
    output._trt = layer.get_output(0)



@add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3)], enabled=trt_version() >= '7.0')
def test_group_norm_trt7():
    return torch.nn.GroupNorm(2, 10)

    
