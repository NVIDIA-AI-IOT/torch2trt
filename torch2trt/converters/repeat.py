import tensorrt as trt 
from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test

@tensorrt_converter('torch.Tensor.repeat', enabled=trt_version() >= '6.0')
@tensorrt_converter('torch.repeat', enabled=trt_version() >= '6.0')

def convert_repeat(ctx):
    inputs = get_arg(ctx, 'input', pos=0, default=None) 
    output = ctx.method_return
    output_list = output.detach().cpu().numpy()
    op_shape = list(output.size())
    layer = ctx.network.add_constant(shape=trt.Dims(op_shape),weights=trt.Weights(output_list))
    output._trt = layer.get_output(0)

class Repeat(torch.nn.Module):
    def __init__(self, arg):
        super(Repeat, self).__init__()
        self.arg = arg

    def forward(self, x):
        return x.repeat(self.arg)

@add_module_test(torch.float32, torch.device('cuda'), [(4, 4)] , enabled=trt_version() >= '6.0')
def test_Stack_repeat():
    return Repeat((3,3))

@add_module_test(torch.float32, torch.device('cuda'), [(4)], enabled=trt_version() >= '6.0')
def test_basic2_repeat():
    return Repeat(10)
