from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.nn.functional.sigmoid')
@tensorrt_converter('torch.sigmoid')
@tensorrt_converter('torch.Tensor.sigmoid')
def convert_sigmoid(ctx):
    input = ctx.method_args[0]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    
    layer = ctx.network.add_activation(input_trt, trt.ActivationType.SIGMOID)
    output._trt = layer.get_output(0)
    

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_sigmoid_basic():
    return torch.nn.Sigmoid()


class TensorSigmoid(torch.nn.Module):
    def __init__(self):
        super(TensorSigmoid, self).__init__()

    def forward(self, x):
        return x.sigmoid()


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 40, 20)])
def test_tensor_sigmoid():
    return TensorSigmoid()
