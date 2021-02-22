from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.relu')
@tensorrt_converter('torch.relu_')
@tensorrt_converter('torch.nn.functional.relu')
@tensorrt_converter('torch.nn.functional.relu_')
@tensorrt_converter('torch.Tensor.relu')
def convert_functional_relu(ctx):
    ctx.method_args = (torch.nn.ReLU(),) + ctx.method_args
    convert_relu(ctx)


@tensorrt_converter('torch.nn.ReLU.forward')
def convert_relu(ctx):
    input = ctx.method_args[1]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    layer = ctx.network.add_activation(
        input=input_trt, type=trt.ActivationType.RELU)
    output._trt = layer.get_output(0)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
def test_relu_basic():
    return torch.nn.ReLU()


class FunctionalRelu(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.relu(x)
    
    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
def test_functional_relu_basic():
    return FunctionalRelu()


class TensorRelu(torch.nn.Module):
    def __init__(self):
        super(TensorRelu, self).__init__()

    def forward(self, x):
        return x.relu()


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 40, 20)])
def test_tensor_relu():
    return TensorRelu()
