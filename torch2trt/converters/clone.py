from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.clone')
@tensorrt_converter('torch.Tensor.clone')
def convert_clone(ctx):
    input = ctx.method_args[0]
    input_trt = trt_(ctx.network, input)

    # Clone by making identity layer.
    layer = ctx.network.add_identity(input_trt)
    set_layer_precision(ctx, layer)

    output = ctx.method_return
    output._trt = layer.get_output(0)


class Clone(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.clone()


@add_module_test(torch.float32, torch.device('cuda'), [(1, 64, 64)])
def test_clone_basic():
    return Clone()


@add_module_test(torch.float32, torch.device('cuda'), [(1, 64, 64)], fp16_mode=True)
def test_clone_fp16_mode():
    return Clone()


@add_module_test(torch.float32, torch.device('cuda'), [(1, 64, 64)], int8_mode=True)
def test_clone_int8_mode():
    return Clone()


class TorchClone(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clone(x)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 64, 64)])
def test_torch_clone_basic():
    return TorchClone()


@add_module_test(torch.float32, torch.device('cuda'), [(1, 64, 64)], fp16_mode=True)
def test_torch_clone_fp16_mode():
    return TorchClone()


@add_module_test(torch.float32, torch.device('cuda'), [(1, 64, 64)], int8_mode=True)
def test_torch_clone_int8_mode():
    return TorchClone()
