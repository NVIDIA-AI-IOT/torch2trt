from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


def slice_to_trt(ctx, dim_size, dim_slice):

    start = 0 if dim_slice.start is None else dim_slice.start
    stop = dim_size if dim_slice.stop is None else dim_slice.stop
    stride = 1 if dim_slice.step is None else dim_slice.step

    start = make_int_wrapper(start)
    stop = make_int_wrapper(stop)
    stride = make_int_wrapper(stride)

    size = (stop - start - 1) // stride + 1

    return start, size, stride


def num_slice_types(slices):
    num_slice = 0
    for s in slices:
        if isinstance(s, slice) or isinstance(s, int):
            num_slice += 1
    return num_slice


@tensorrt_converter('torch.Tensor.__getitem__')
def convert_tensor_getitem(ctx):
    input = ctx.method_args[0]
    slices = ctx.method_args[1]
    output = ctx.method_return

    if not hasattr(input, '_trt'):
        return

    input_trt = input._trt

    # Convert slices argument into a tuple if it is not already one.
    # This can happen when the provided slices argument is a single element.
    #
    # In the general case, multiple slice arguments will already be wrapped in a tuple.
    #
    # Note the special case where a single tuple is itself the only slice argument (eg. t[(...)]);
    # PyTorch treats the single tuple argument as if its elements were provided as separate slice arguments.
    # We handle this gracefully by default without requiring special handling.
    #
    # A tuple of tuple of elements is only possible if the inner tuple is combined with other arguments.
    # In this case, the inner tuple is now a gather operation and must be handled differently.
    # This is also commonly known as advanced indexing.
    if not isinstance(slices, tuple):
        slices = (slices,)

    # Step 1 - Replace ellipsis with expanded slices
    num_ellipsis = len(input.shape) - num_slice_types(slices)

    new_slices = []
    for s in slices:

        if s == Ellipsis:
            while num_ellipsis > 0:
                new_slices.append(slice(None, None, None))
                num_ellipsis -= 1
        elif isinstance(s, slice):
            new_slices.append(s)
        elif s is None:
            new_slices.append(None)
        elif isinstance(s, int) or isinstance(s, IntWrapper):
            new_slices.append(s)

    # fill missing slices at end
    while num_slice_types(new_slices) < len(input.shape):
        new_slices.append(slice(None, None, None))

    # Step 2 - Remove batch from slices (TRT from this point)

    slices = tuple(new_slices) # remove batch


    # Step 3 - Add slice layer (will currently ignore 'None' slices)

    starts = []
    sizes = []
    strides = []

    input_dim = 0

    input_size = input.size()

    for s in slices:

        if input_dim >= len(input_trt.shape):
            break

        if isinstance(s, slice):
            start, size, stride = slice_to_trt(ctx, input_size[input_dim], s)
            starts.append(start)
            sizes.append(size)
            strides.append(stride)
            input_dim += 1

        elif isinstance(s, int) or isinstance(s, IntWrapper):
            starts.append(make_int_wrapper(s))
            sizes.append(make_int_wrapper(1))
            strides.append(make_int_wrapper(1))
            input_dim += 1

    starts = make_size_wrapper(starts)
    sizes = make_size_wrapper(sizes)
    strides = make_size_wrapper(strides)

    layer = ctx.network.add_slice(input_trt, starts, sizes, strides)
    layer.set_input(1, starts._trt)
    layer.set_input(2, sizes._trt)
    layer.set_input(3, strides._trt)

    output_trt = layer.get_output(0)

    # Step 4 - Add shuffle layer to insert dimensions for 'None' slices and remove dimensions for 'int' slices

    num_non_slice = len([s for s in slices if not isinstance(s, slice)])
    if num_non_slice > 0:

        final_shape = []
        i = 0
        for s in slices:
            if isinstance(s, slice):
                # copy slice dim
                final_shape.append(sizes[i])
                i += 1
            elif isinstance(s, int) or isinstance(s, IntWrapper):
                # remove int dim
                i += 1
            else:
                # insert None dim
                final_shape.append(make_int_wrapper(1))

        final_shape = make_size_wrapper(final_shape)

        layer = ctx.network.add_shuffle(output_trt)
        layer.reshape_dims = tuple(output.shape) # exclude batch
        layer.set_input(1, final_shape._trt)
        output_trt = layer.get_output(0)

    output._trt = output_trt


class LambdaModule(torch.nn.Module):
    def __init__(self, fn):
        super(LambdaModule, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


@add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
def test_tensor_getitem_0d_colon():
    return LambdaModule(lambda x: x[:])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
def test_tensor_getitem_0d_ellipsis():
    return LambdaModule(lambda x: x[...])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
def test_tensor_getitem_0d_int():
    return LambdaModule(lambda x: x[0])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
def test_tensor_getitem_0d_int_colon():
    return LambdaModule(lambda x: x[0, :])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
def test_tensor_getitem_0d_int_ellipsis():
    return LambdaModule(lambda x: x[0, ...])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
def test_tensor_getitem_0d_1tuple():
    return LambdaModule(lambda x: x[(0,)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
def test_tensor_getitem_0d_2tuple():
    return LambdaModule(lambda x: x[(0, 1)])


# There is currently an issue with this test case.
# Need to investigate this more.
#  @add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
#  def test_tensor_getitem_0d_3tuple():
    #  return LambdaModule(lambda x: x[(0, 1, 2)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
def test_tensor_getitem_0d_range_start():
    return LambdaModule(lambda x: x[0:])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
def test_tensor_getitem_0d_range_end():
    return LambdaModule(lambda x: x[:1])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
def test_tensor_getitem_0d_range():
    return LambdaModule(lambda x: x[0:1])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
def test_tensor_getitem_0d_range_start_colon():
    return LambdaModule(lambda x: x[0:, :])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
def test_tensor_getitem_0d_range_end_colon():
    return LambdaModule(lambda x: x[:3, :])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
def test_tensor_getitem_0d_range_colon():
    return LambdaModule(lambda x: x[0:3, :])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
def test_tensor_getitem_0d_range_start_ellipsis():
    return LambdaModule(lambda x: x[0:, ...])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
def test_tensor_getitem_0d_range_end_ellipsis():
    return LambdaModule(lambda x: x[:3, ...])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
def test_tensor_getitem_0d_range_ellipsis():
    return LambdaModule(lambda x: x[0:3, ...])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
def test_tensor_getitem_0d_strided():
    return LambdaModule(lambda x: x[::2])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
def test_tensor_getitem_0d_strided_offset():
    return LambdaModule(lambda x: x[0::2])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
def test_tensor_getitem_0d_strided_range():
    return LambdaModule(lambda x: x[0:1:2])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
def test_tensor_getitem_0d_strided_colon():
    return LambdaModule(lambda x: x[::2, :])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
def test_tensor_getitem_0d_strided_offset_colon():
    return LambdaModule(lambda x: x[0::2, :])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
def test_tensor_getitem_0d_strided_range_colon():
    return LambdaModule(lambda x: x[0:1:2, :])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
def test_tensor_getitem_0d_strided_ellipsis():
    return LambdaModule(lambda x: x[::2, ...])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
def test_tensor_getitem_0d_strided_offset_ellipsis():
    return LambdaModule(lambda x: x[0::2, ...])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
def test_tensor_getitem_0d_strided_range_ellipsis():
    return LambdaModule(lambda x: x[0:1:2, ...])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
def test_tensor_getitem_0d_insert_dim():
    return LambdaModule(lambda x: x[None])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
def test_tensor_getitem_0d_insert_dim_colon():
    return LambdaModule(lambda x: x[None, :])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
def test_tensor_getitem_0d_insert_dim_ellipsis():
    return LambdaModule(lambda x: x[None, ...])

# Tuple arguments combined with `...` or `:` actually is equivalent to using list arguments.
# This use case is special and not yet currently handled; see
# https://github.com/NVIDIA-AI-IOT/torch2trt/issues/755.
#
#  @add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
#  def test_tensor_getitem_0d_1tuple_colon():
    #  return LambdaModule(lambda x: x[(0,), :])


#  @add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
#  def test_tensor_getitem_0d_2tuple_colon():
    #  return LambdaModule(lambda x: x[(0, 1), :])
#
#
#  @add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
#  def test_tensor_getitem_0d_3tuple_colon():
    #  return LambdaModule(lambda x: x[(0, 1, 2), :])
#
#
#  @add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
#  def test_tensor_getitem_0d_1tuple_ellipsis():
    #  return LambdaModule(lambda x: x[(0,), ...])
#
#
#  @add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
#  def test_tensor_getitem_0d_2tuple_ellipsis():
    #  return LambdaModule(lambda x: x[(0, 1), ...])
#
#
#  @add_module_test(torch.float32, torch.device('cuda'), [(3, 2, 4)], max_batch_size=3)
#  def test_tensor_getitem_0d_3tuple_ellipsis():
    #  return LambdaModule(lambda x: x[(0, 1, 2), ...])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_tensor_getitem_1d_int():
    return LambdaModule(lambda x: x[:, 0])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_int():
    return LambdaModule(lambda x: x[:, 0])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_strided():
    return LambdaModule(lambda x: x[:, ::2])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_strided_offset():
    return LambdaModule(lambda x: x[:, 1::2])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_strided_range():
    return LambdaModule(lambda x: x[:, 1:3:2])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_insert_dim():
    return LambdaModule(lambda x: x[:, None])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_insert_dim_ellipsis():
    return LambdaModule(lambda x: x[:, None, ...])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_append_dim():
    return LambdaModule(lambda x: x[:, ..., None])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_append_2dim():
    return LambdaModule(lambda x: x[:, ..., None, None])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_weird_combo():
    return LambdaModule(lambda x: x[:, 0:3:4, None, None, 1, ...])
