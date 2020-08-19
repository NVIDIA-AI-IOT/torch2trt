from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test

@tensorrt_converter('torch.Tensor.repeat', enabled=trt_version() >= '6.0')
def convert_repeat(ctx):
    inputs = get_arg(ctx, 'input', pos=0, default=None) 
    output = ctx.method_return
    op_shape = output.size().as_list()
    in_shape =inputs.size().as_list()
    print(op_shape,in_shape)
    layer = ctx.network.add_concatenation(inputs=trt_inputs)
    output._trt = layer.get_output(0)

class Repeat(torch.nn.Module):
    def __init__(self, arg):
        super(Repeat, self).__init__()
        self.arg = arg
        print("self.arg",self.arg)

    def forward(self, x):
        return x.repeat(self.arg)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 4, 4)] , enabled=trt_version() >= '6.0')
def test_Stack_repeat():
    return Repeat(3)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 4, 4)], enabled=trt_version() >= '6.0')
def test_basic2_repeat():
    return Repeat(1)
