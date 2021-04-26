from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.topk')
def convert_topk(ctx):
    inputs = get_arg(ctx, 'input', pos=0, default=None)
    k = get_arg(ctx, 'k', pos=1, default=25)
    dim = get_arg(ctx, 'dim', pos=2, default=None)

    if not dim:
        dim = len(inputs.shape) - 1
    else:
        dim -= 1 # TRT doesn't take into account the batch dimension therefore we cannot take topK over the batch

    assert dim != -1 # if axes == -1 then user is trying to perform TopK on batch

    output_val = ctx.method_return[0]
    output_idx = ctx.method_return[1]
    trt_inputs = add_missing_trt_tensors(ctx.network, inputs)[0]

    layer = ctx.network.add_topk(input=trt_inputs, op=trt.TopKOperation.MAX, k=k, axes=torch_dim_to_trt_axes(dim))

    output_val._trt = layer.get_output(0)
    output_idx._trt = layer.get_output(1)

class TopK(torch.nn.Module):
    def __init__(self, k):
        super(TopK, self).__init__()
        self.k = k

    def forward(self, x):
        return torch.topk(x, self.k)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 4, 4), (1, 3, 4), (1, 17, 4)])
def test_TopK_basic():
    return TopK(10)
