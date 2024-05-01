import torch.nn as nn
from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test



@tensorrt_converter('torch.einsum')
def convert_einsum(ctx):
    einsum_eq = ctx.method_args[0]
    input_tensors = ctx.method_args[1:]
    output = ctx.method_return
    
    # parts = einsum_eq.split('->')

    # strip batch dimension
    # if len(parts) > 1:
    #     lhs = parts[0]
    #     rhs = parts[1]
    #     lhs = ','.join([part[1:] for part in lhs.split(',')])
    #     rhs = rhs[1:]
    #     einsum_eq = lhs + '->' + rhs
    # else:
    #     einsum_eq = ','.join([part[1:] for part in einsum_eq.split(',')])
        
    layer = ctx.network.add_einsum(
        [t._trt for t in input_tensors],
        einsum_eq
    )

    output._trt = layer.get_output(0)


class Einsum(nn.Module):

    def __init__(self, einsum_eq):
        super().__init__()
        self.einsum_eq = einsum_eq

    def forward(self, *args):
        return torch.einsum(self.einsum_eq, *args)



@add_module_test(torch.float32, torch.device('cuda'), [(2, 2, 5), (2, 5, 4)], max_batch_size=2)
def test_einsum_bmm():
    return Einsum('bij,bjk->bik')