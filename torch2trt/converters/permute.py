from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.Tensor.permute')
def convert_permute(ctx):
    input = ctx.method_args[0]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    
    # permutation -1 because TRT does not include batch dim
    if isinstance(ctx.method_args[1], int):
        permutation = tuple(ctx.method_args[1:])  # handle permute(a, b, c)
    else:
        permutation = tuple(ctx.method_args[1])   # handle permute([a, b, c])
        
    assert(permutation[0] == 0)  # cannot move batch dim
    
    trt_permutation = tuple([p - 1 for p in permutation])[1:]
    
    layer = ctx.network.add_shuffle(input_trt)
    layer.second_transpose = tuple(trt_permutation)
   
    output._trt = layer.get_output(0)


class Permute(torch.nn.Module):
    def __init__(self, *args):
        super(Permute, self).__init__()
        self.args = args
    def forward(self, x):
        return x.permute(*self.args).contiguous()


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
def test_permute_2d_0123():
    return Permute(0, 1, 2, 3)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
def test_permute_2d_0312():
    return Permute(0, 3, 1, 2)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5, 6)])
def test_permute_3d_01234():
    return Permute(0, 1, 2, 3, 4)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5, 6)])
def test_permute_3d_04132():
    return Permute(0, 4, 1, 3, 2)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5, 6)])
def test_permute_list():
    return Permute([0, 4, 1, 3, 2])

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5, 6)])
def test_permute_tuple():
    return Permute((0, 4, 1, 3, 2))