from torch2trt.torch2trt import *

@tensorrt_converter('torch.nn.ZeroPad2d.forward')
def convert_pad(ctx):
    module = ctx.method_args[0]
    trt_input = ctx.method_args[1]._trt
    output = ctx.method_return
    split = len(module.padding)//2
    layer = ctx.network.add_padding(input=trt_input,pre_padding=module.padding[:split],post_padding=module.padding[split:])
    output._trt = layer.get_output(0)
    
    