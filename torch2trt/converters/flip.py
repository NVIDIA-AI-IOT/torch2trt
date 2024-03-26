from torch2trt import torch2trt, tensorrt_converter, get_arg, trt, make_size_wrapper


@tensorrt_converter("torch.Tensor.flip")
@tensorrt_converter("torch.flip")
def convert_flip(ctx):

    input = get_arg(ctx, 'input', 0, None)
    dims = get_arg(ctx, 'dims', 1, None)
    output = ctx.method_return

    input_shape_trt = ctx.network.add_shape(input._trt).get_output(0)

    offset = [0 for i in range(input.ndim)]
    stride = [1 for i in range(input.ndim)]
    shape = tuple(input.size())
    for d in dims:
        offset[d] = -1
        stride[d] = -1

    layer = ctx.network.add_slice(
        input._trt,
        offset,
        shape,
        stride
    )
    layer.set_input(2, input_shape_trt)
    layer.mode = trt.SliceMode.WRAP

    output._trt = layer.get_output(0)
