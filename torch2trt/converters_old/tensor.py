from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.tensor')
def convert_mod(ctx):
    output = ctx.method_return
    layer = ctx.network.add_constant(tuple(output.shape), output.detach().cpu().numpy() )
    output._trt = layer.get_output(0)


class TorchTensor(torch.nn.Module):
    def __init__(self):
        super(TorchTensor, self).__init__()

    def forward(self, x):
        return x + torch.tensor([[1., 2., 3.], [4., 5., 6.]], device=torch.device('cuda'))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 2, 3)])
def test_tensor_creation():
    return TorchTensor()
