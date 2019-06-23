from torch2trt.torch2trt import *


@tensorrt_converter('torch.nn.Linear.forward')
def convert_Linear(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    output = ctx.method_return

    # reshape to Nx1x1
    layer = ctx.network.add_shuffle(input._trt)
    layer.reshape_dims = (-1, 1, 1)

    # add fully connected
    layer = ctx.network.add_fully_connected(
        input=layer.get_output(0),
        num_outputs=module.out_features,
        kernel=module.weight.detach().cpu().numpy(),
        bias=module.bias.detach().cpu().numpy())

    # reshape back to N
    layer = ctx.network.add_shuffle(layer.get_output(0))
    layer.reshape_dims = (-1,)

    output._trt = layer.get_output(0)
