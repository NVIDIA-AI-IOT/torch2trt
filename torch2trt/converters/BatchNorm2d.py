from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter("torch.nn.BatchNorm2d.forward", enabled=trt_version() < '7.0')
def convert_BatchNorm2d(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return

    scale = module.weight.detach().cpu().numpy() / np.sqrt(
        module.running_var.detach().cpu().numpy() + module.eps
    )
    bias = (
        module.bias.detach().cpu().numpy()
        - module.running_mean.detach().cpu().numpy() * scale
    )
    power = np.ones_like(scale)

    layer = ctx.network.add_scale(input_trt, trt.ScaleMode.CHANNEL, bias, scale, power)

    output._trt = layer.get_output(0)
