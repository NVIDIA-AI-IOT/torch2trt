from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter("torch.nn.BatchNorm3d.forward", enabled=trt_version() < "7.0")
def convert_BatchNorm3d(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
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


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 16, 16, 16)])
@add_module_test(torch.float32, torch.device('cuda'), [(2, 3, 16, 16, 16)], max_batch_size=2)
def test_BatchNorm3d_basic():
    return torch.nn.BatchNorm3d(3)