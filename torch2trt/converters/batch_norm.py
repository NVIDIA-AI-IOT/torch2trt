from torch2trt.module_test import add_module_test
from torch2trt.torch2trt import *


@tensorrt_converter("torch.nn.functional.batch_norm")
@tensorrt_converter("torch.batch_norm")
def convert_batch_norm(ctx):
    input = ctx.method_args[0]
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return

    scale = ctx.method_args[3].detach().cpu().numpy() / np.sqrt(
        ctx.method_args[2].detach().cpu().numpy() + ctx.method_args[7]
    )
    bias = (
        ctx.method_args[4].detach().cpu().numpy()
        - ctx.method_args[1].detach().cpu().numpy() * scale
    )
    power = np.ones_like(scale)

    layer = ctx.network.add_scale_nd(
        input_trt, trt.ScaleMode.CHANNEL, bias, scale, power, 0
    )

    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device("cuda"), [(1, 10, 5)])
def test_batch_norm_1d():
    return torch.nn.BatchNorm1d(10)


@add_module_test(torch.float32, torch.device("cuda"), [(1, 10, 5, 5)])
def test_batch_norm_2d():
    return torch.nn.BatchNorm2d(10)


@add_module_test(torch.float32, torch.device("cuda"), [(1, 10, 5, 5, 5)])
def test_batch_norm_3d():
    return torch.nn.BatchNorm3d(10)
