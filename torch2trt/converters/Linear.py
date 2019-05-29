from torch2trt.torch2trt import *


@tensorrt_converter('torch.nn.Linear.forward')
def convert_Linear(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    output = ctx.method_return

    layer = ctx.network.add_fully_connected(
        input=input._trt,
        num_outputs=module.out_features,
        kernel=module.weight.detach().cpu().numpy(),
        bias=module.bias.detach().cpu().numpy())

    output._trt = layer.get_output(0)