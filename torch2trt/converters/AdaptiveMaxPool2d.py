from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


def has_adaptive_max_pool2d_plugin():
    try:
        from torch2trt.plugins import AdaptiveMaxPool2dPlugin
        return True
    except:
        return False


def get_adaptive_max_pool2d_plugin(output_size):
    from torch2trt.plugins import AdaptiveMaxPool2dPlugin
    PLUGIN_NAME = "adaptive_max_pool2d"
    registry = trt.get_plugin_registry()
    creator = [c for c in registry.plugin_creator_list if c.name == PLUGIN_NAME and c.plugin_namespace == 'torch2trt'][0]
    torch2trt_plugin = AdaptiveMaxPool2dPlugin(output_size=output_size)
    return creator.deserialize_plugin(PLUGIN_NAME, torch2trt_plugin.serializeToString())


@tensorrt_converter('torch.nn.AdaptiveMaxPool2d.forward')
def convert_AdaptiveMaxPool2d(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    output = ctx.method_return

    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output_size = module.output_size
    if isinstance(output_size, int):
        output_size = (output_size, ) * 2

    output_size = list(output_size)
    plugin = get_adaptive_max_pool2d_plugin(output_size=output_size)

    layer = ctx.network.add_plugin_v2([input_trt], plugin)

    output._trt = layer.get_output(0)

    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_adaptive_max_pool2d_1x1():
    return torch.nn.AdaptiveMaxPool2d((1, 1))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_adaptive_max_pool2d_2x2():
    return torch.nn.AdaptiveMaxPool2d((2, 2))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 1, 224, 224)])
def test_adaptive_max_pool2d_3x3():
    return torch.nn.AdaptiveMaxPool2d((3, 3))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 1, 10, 10)])
def test_adaptive_max_pool2d_6x6():
    return torch.nn.AdaptiveMaxPool2d((6, 6))
