from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.clone')
@tensorrt_converter('torch.Tensor.clone')
def convert_clone(ctx):
    input = ctx.method_args[0]
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return

    # Clone by making identity layer.
    layer = ctx.network.add_identity(input_trt)
    output._trt =  layer.get_output(0)


class Clone(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.clone()


@add_module_test(torch.float32, torch.device('cuda'), [(1, 64, 64)])
def test_clone_basic():
    return Clone()


# This fails with the below error:
# [TensorRT] ERROR: ../builder/cudnnBuilder2.cpp (2072) - Assertion Error in getSupportedFormats: 0 (!formats.empty())
#
#  @add_module_test(torch.float16, torch.device('cuda'), [(1, 64, 64)])
#  def test_clone_float16():
    #  return Clone()


# This fails with the below error:
# [TensorRT] ERROR: ../builder/cudnnBuilder2.cpp (2072) - Assertion Error in getSupportedFormats: 0 (!formats.empty())
#
#  @add_module_test(torch.int8, torch.device('cuda'), [(1, 64, 64)])
#  def test_clone_int8():
    #  return Clone()


class TorchClone(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clone(x)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 64, 64)])
def test_torch_clone_basic():
    return TorchClone()


# This fails with the below error:
# [TensorRT] ERROR: ../builder/cudnnBuilder2.cpp (2072) - Assertion Error in getSupportedFormats: 0 (!formats.empty())
#
#  @add_module_test(torch.float16, torch.device('cuda'), [(1, 64, 64)])
#  def test_torch_clone_float16():
    #  return TorchClone()


# This fails with the below error:
# [TensorRT] ERROR: ../builder/cudnnBuilder2.cpp (2072) - Assertion Error in getSupportedFormats: 0 (!formats.empty())
#
#  @add_module_test(torch.int8, torch.device('cuda'), [(1, 64, 64)])
#  def test_torch_clone_int8():
    #  return TorchClone()
