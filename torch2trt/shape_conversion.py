from torch2trt.torch2trt import *


@tensorrt_converter('torch.Tensor.size')
def convert_size(ctx):
    input = ctx.method_args[0]
    dim = get_arg(ctx, pos=1, name='dim', default=None)
    output = ctx.method_return

    shape_trt = ctx.network.add_shape(input._trt).get_output(0)

    new_output = SizeWrapper(IntWrapper(d) for d in output)

    for i, d in enumerate(new_output):
        d._raw_trt = ctx.network.add_slice(shape_trt, [i], [1], [1]).get_output(0)

    if dim is None:
        ctx.method_return = new_output
    else:
        ctx.method_return = new_output[dim]
