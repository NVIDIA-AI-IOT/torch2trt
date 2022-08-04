from collections import namedtuple

from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test

import tensorrt
import torch


# Our conversion of __getitem__ needs to handle basic and advanced indexing (specifically GatherND).
# See the numpy description for more information on different types of indexing, which pytorch follows:
# https://numpy.org/doc/stable/user/basics.indexing.html
#
# We use the following terms to describe our algorithm:
#   t, a pytorch tensor of arbitrary shape and dimensions on which we are calling __getitem__.
#   s, a slice index; eg. the operators :, ..., None, ().
#   g, a gather index; eg. (x,...), [x,...], torch.tensor((x,...)) for any arbitrary scalar x.
#       Note that we currently only handle 1D gather indices, so g is always 1D where described.
#
# Our algorithm works as follows:
#   For an input tensor t, we check the indices argument.
#   This results in the following cases:
#
#   1. If all of the indices are slices, eg. t[s,s,s,...], this is considered basic indexing,
#   and we can trivially convert this to TRT using the slice layer (along with some supporting layers).
#
#   2. If there are any gather indices, regardless of the presence of slice indices,
#   eg. t[...,g,g,g,...], this is now considered advanced indexing
#   and we are no longer just slicing, but also gathering on the input tensor.
#   We convert differently depending on the composition of the indices.
#
#   2a. If all of the indices are gather indices and there are no slice indices, eg. t[g,g,g,...],
#   then we can trivially convert this to TRT using a single gather layer.
#
#   2b. If we have a mix of slice and gather indices, eg. t[s,s,g,g,...], then the TRT conversion gets more complex.
#   First, we split the indices into slice only indices and gather only indices of the same dimensions,
#   using the colon operator for the axes where a gather or slice index was removed from the slice only
#   or gather only indices, respectively; this allows us to process the slice and gather indices separately,
#   where the colon operator allows us to ignore an axis when not processing that particular type of index.
#
#   Consequently, we can now process t as if the indices only have slice operations, eg. t[s,s,:,:,...],
#   using the same basic indexing methodology previously described in case (1) using a slice layer.
#   Afterwards, all slicing operations are complete and we need only perform gather operations henceforth.
#
#   Now using the output of the slice layer, we process all of the gather indices, eg. t[:,:,g,g,...].
#   As the TRT gather layer does not handle slice indices (ie. colon operators),
#   we cannot pass in all gather indices to the gather layer as in case (2a).
#   This is especially problematic when the colon operator sits between two gather operations, eg. t[g,:,g].
#
#   As a result, to account for these axes in which we have a colon operator,
#   we need to continually transpose (permute) t such that each axis that we are gathering on is adjacent,
#   until all axes on which we are gathering are adjacent; in other words, t[g,:,g] == transposed(t)[g,g,:]
#   is a valid equivalency (we call this coalescing gather indices for brevity).
#   This moves any dimensions with the colon operator out from between any two dimensions with gather operations
#   and allows us to use the TRT gather layer to perform the needed gatherND operation,
#   as now only gather indices are present in the indexing operation.
#
#   The following examples using a 4D tensor of shape (3,3,3,3) shows the equivalent transpose operations needed
#   so that all gather indices can be coalesced when indexing:
#
#   t[:,g,:,:] == t.transpose(1,0)[g].transpose(0,1)
#   t[:,:,g,:] == t.transpose(2,1).transpose(1,0)[g].transpose(0,1).transpose(1,2)
#   t[:,:,:,g] == t.transpose(3,2).transpose(2,1).transpose(1,0)[g].transpose(0,1).transpose(1,2).transpose(2,3)
#   t[g,:,g,:] == t.transpose(2,1)[g,g]
#   t[g,:,:,g] == t.transpose(3,2).transpose(2,1)[g,g]
#   t[:,g,g,:] == t.transpose(1,0).transpose(2,1)[g,g].transpose(0,1)
#   t[:,g,:,g] == t.transpose(1,0).transpose(3,2).transpose(2,1)[g,g]
#   t[:,:,g,g] == t.transpose(2,1).transpose(1,0).transpose(3,2).transpose(2,1)[g,g].transpose(0,1).transpose(1,2)
#   t[g,g,:,g] == t.transpose(3,2)[g,g,g]
#   t[g,:,g,g] == t.transpose(2,1).transpose(3,2)[g,g,g]
#   t[:,g,g,g] == t.transpose(1,0).transpose(2,1).transpose(3,2)[g,g,g].transpose(0,1)
#
#   Note the following from the above examples:
#   - The first gather operation always transposes to dimension 0, if it is not already there.
#   - Final transposes are needed after the gather operation iff gather indices are already coalesced together.
#
#   This is the algorithm implemented below.


# A container to hold the data and metadata needed for a single gather operation.
GatherIndex = namedtuple(
    "GatherIndex",
    {
        "trt",  # The indices used for gathering, as a TRT tensor.
        "axis", # The dimension in which to gather, as given in the original indices argument.
    }
)


def _is_int(obj):
    return isinstance(obj, int)


def _is_list_or_tuple(obj):
    return isinstance(obj, (list, tuple))


def _is_torch_tensor(obj):
    return isinstance(obj, torch.Tensor)


def _is_trt_tensor(obj):
    return isinstance(obj, trt.ITensor)


def _trt(ctx, tensor):
    if hasattr(tensor, "_trt"):
        return tensor._trt

    # Currently, we silently convert int64 to int32 since:
    # 1. int64 is not supported in TRT; and
    # 2. most usages of int64 are still valid as int32.
    if tensor.dtype is torch.int64:
        tensor = tensor.to(torch.int32)

    shape = tuple(tensor.shape)
    data = tensor.detach().cpu().numpy()
    layer = ctx.network.add_constant(shape, data)
    return layer.get_output(0)


def _tensor_trt(ctx, data, dtype=torch.float32):
    t = torch.tensor(data, dtype=dtype)
    t._trt = _trt(ctx, t)
    return t


def _arange_trt(ctx, max_range_trt):
    zero_trt = ctx.network.add_reduce(make_int_wrapper(0)._trt, trt.ReduceOperation.MAX, 1, False).get_output(0)
    one_trt = make_int_wrapper(1)._trt

    fill_layer = ctx.network.add_fill((0,), trt.FillOperation.LINSPACE)
    fill_layer.set_input(0, max_range_trt)
    fill_layer.set_input(1, zero_trt)
    fill_layer.set_input(2, one_trt)
    return fill_layer.get_output(0)


def _cat_trt(ctx, trts, axis=0):
    cat_layer = ctx.network.add_concatenation(trts)
    cat_layer.axis=axis
    return cat_layer.get_output(0)


def _gathernd_trt(ctx, input_trt, indices_trt):
    gather_layer = ctx.network.add_gather_v2(input_trt, indices_trt, tensorrt.GatherMode.ND)
    return gather_layer.get_output(0)


def _transpose_1d_trt(ctx, trt):
    # Transpose a 1D TRT tensor by simply adding a second dimension to the shape end.
    # This turns an array of columns into an array of rows, which is needed for a column vector.

    one_trt = make_int_wrapper(1)._trt
    shape_trt = _shape_trt(ctx, trt)
    shape_trt = _cat_trt(ctx, [shape_trt, one_trt])

    shuffle_layer = ctx.network.add_shuffle(trt)
    shuffle_layer.set_input(1, shape_trt)

    return shuffle_layer.get_output(0)


def _permute_trt(ctx, trt, start_axis, end_axis):
    permutation = list(range(len(trt.shape)))
    permutation.pop(start_axis)
    permutation.insert(end_axis, start_axis)

    shuffle_layer = ctx.network.add_shuffle(trt)
    shuffle_layer.first_transpose = permutation
    return shuffle_layer.get_output(0)


def _elementwise_trt(ctx, trt_a, trt_b, op):
    return ctx.network.add_elementwise(trt_a, trt_b, op).get_output(0)


def _elementwise_gt_trt(ctx, trt_a, trt_b):
    return _elementwise_trt(ctx, trt_a, trt_b, trt.ElementWiseOperation.GREATER)


def _select_trt(ctx, condition_trt, then_trt, else_trt):
    return ctx.network.add_select(condition_trt, then_trt, else_trt).get_output(0)


def _shape_trt(ctx, trt):
    return ctx.network.add_shape(trt).get_output(0)


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


def _replace_ellipses(input_, indices):
    # Remove ellipses from indices and replace them with the appropriate number of slices.
    # This simplifies downstream processing, especially for advanced indexing.
    num_replacements = len(input_.shape) - sum(index != Ellipsis for index in indices)
    ellipsis_replacement = [slice(None, None, None)] * num_replacements

    new_indices = []
    for index in indices:
        new_indices += (ellipsis_replacement if index == Ellipsis else [index])

    return new_indices


def _requires_advanced_indexing(indices):
    # We use advanced indexing for gather operations.
    # This is only necessary when the indices argument contains either list, tensor, or tuple index elements;
    # if there are tensor elements, then at least one must either have greater than 0 dimensions or
    # there must be other list or tuple elements present.
    return any(((_is_list_or_tuple(index) and len(index) > 0) or (_is_torch_tensor(index) and index.dim() > 0))
            for index in indices)


def _indices_to_trt(ctx, indices):
    # Convert any list, tuple, tensor indices to TRT tensors.
    # Note that this still leaves ints, slices, and Nones in the list of returned indices,
    # as these cannot be converted to TRT tensors.
    def _to_trt(index):
        if _is_list_or_tuple(index):
            index = _tensor_trt(ctx, index, dtype=torch.int32)._trt
        elif _is_torch_tensor(index):
            index = _trt(ctx, index)

        return index

    return [_to_trt(index) for index in indices]


def _max_length_on_dim0(ctx, indices_trt):
    # Get the longest length of any index TRT tensor.
    # This is the length all indices will be broadcast to.
    #
    # Note that we currently only broadcast single element 1D TRT tensors;
    # consequently, we only need to examine the length on axis 0.

    max_length_trt = make_int_wrapper(0)._trt
    for index_trt in indices_trt:
        if not _is_trt_tensor(index_trt):
            continue

        assert len(index_trt.shape) == 1, f"Encountered tensor index with shape {index_trt.shape} but only indices of rank 1 are currently supported."

        shape_trt = _shape_trt(ctx, index_trt)
        length_trt = ctx.network.add_slice(shape_trt, [0], [1], [1]).get_output(0) # Length on axis 0.

        gt_trt = _elementwise_gt_trt(ctx, length_trt, max_length_trt)
        max_length_trt = _select_trt(ctx, gt_trt, length_trt, max_length_trt)

    return max_length_trt


def _broadcast_index(ctx, index_trt, broadcast_length_trt):
    # Broadcast the index to the given length, if it's a tensor index and of length 1.
    # Note that we currently only broadcast single element 1D TRT tensors.

    if not _is_trt_tensor(index_trt):
        return index_trt

    # TODO(@chaoz): This implementation broadcasts an index even if it's unnecessary!
    # ie. For an index already at max shape, we just end up slicing into an equivalent tensor!
    # We should find a way to shortcut this processing; I'd like to use an if_conditional here,
    # but because output shapes may be different, it is impossible to do so.
    slice_layer = ctx.network.add_slice(index_trt, [0], [0], [1])
    slice_layer.mode = trt.SliceMode.CLAMP
    slice_layer.set_input(2, broadcast_length_trt)
    slice_trt = slice_layer.get_output(0)

    return slice_trt


def _broadcast_indices(ctx, indices_trt):
    # All tensor indices must have the same length for a gather operation.
    # Since we iterate through each dimension individually to gather one axis at a time,
    # we perform our own broadcasting and reshape all tensor indices to the same max length for later use.
    max_length_trt = _max_length_on_dim0(ctx, indices_trt)
    return [_broadcast_index(ctx, index_trt, max_length_trt) for index_trt in indices_trt]


def _split_indices(indices_trt):
    # Split indices into those used for slicing and those used for gathering.
    # The colon operator (:) fills "blank" indices left by the removal of a slice or gather index,
    # as we select every element in that axis if it is not being gathered or sliced, respectively.
    colon = slice(None, None, None)
    slices = []
    gathers = []

    for axis, index_trt in enumerate(indices_trt):
        if _is_trt_tensor(index_trt):
            gathers.append(index_trt)
            slices.append(colon)
        else:
            slices.append(index_trt)
            # TODO(@chaoz): Add support for 0D tensor.
            # If the index is an int or 0D tensor, then slicing will remove a dimension.
            # Subsequent gathering operations must account for the reduction in dimensions
            # by skipping adding a colon for the index that is causing the dimension reduction.
            if isinstance(index_trt, int):
                continue
            gathers.append(colon)

    return slices, gathers


def _all_gathers_contiguous(gather_indices):
    # Check if all gather indices are contiguous;
    # ie. are all gather indices are immediately preceeded by another gather index (except the first).
    # For our algorithm, we consider a single gather index to be contiguous.
    if len(gather_indices) == 1:
        return True

    return all(gather_indices[i].axis - gather_indices[i-1].axis == 1 for i in range(1, len(gather_indices)))


def _advanced_gathernd_trt(ctx, input_trt, indices_trt):
    # We only need the gather indices going forward so we parse these out along with some metadata.
    # Note that we need to transpose the 1D indices to a column vector as needed eventually by the TRT gather layer.
    gather_indices = [GatherIndex(trt=_transpose_1d_trt(ctx, index_trt), axis=axis)
        for axis, index_trt in enumerate(indices_trt) if _is_trt_tensor(index_trt)]

    # Transpose the input so that each gather index is immediately adjacent to one another at the leftmost dimensions.
    # ie. If we are trying to gather from a tensor t like so:
    #   t[s,g,s,g]
    # Then we can transpose the tensor t so that we coalesce the gather indices for an equivalent operation:
    #   t[g,g,s,s]
    next_axis = 0
    for gather_index in gather_indices:
        gather_axis = gather_index.axis
        if gather_axis != next_axis:
            input_trt = _permute_trt(ctx, input_trt, gather_axis, next_axis)
        next_axis += 1

    # All axes to gather from are immediately adjacent to one another.
    # We can now perform the gatherND operation using a single TRT gather layer.
    gather_indices_trt = _cat_trt(ctx, [gather_index.trt for gather_index in gather_indices], axis=1)
    gather_trt = _gathernd_trt(ctx, input_trt, gather_indices_trt)

    # If all axes gathered from were already contiguous prior to any transposing,
    # then we need an extra transpose after gathering to fix the incorrect shape caused by transposing (see example above).
    # This is done by reversing the transpose applied to the tensor t from the first gather index.
    # Specifically, this handles the following cases:
    #   t[s,s,...,g,g,...,s,s,...]
    #   t[...,s,s,g,g]
    # Note that we do not need any additional processing to handle this case:
    #   t[g,g,s,s,...]
    first_gather_index = gather_indices[0]
    if _all_gathers_contiguous(gather_indices) and first_gather_index.axis != 0:
        gather_trt = _permute_trt(ctx, gather_trt, 0, first_gather_index.axis)

    return gather_trt


def _basic_indexing(ctx, input_, slices):
    input_trt = input_._trt

    # Step 1 - Replace ellipsis with expanded slices
    num_ellipsis = len(input_.shape) - num_slice_types(slices)

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
    while num_slice_types(new_slices) < len(input_.shape):
        new_slices.append(slice(None, None, None))

    slices = tuple(new_slices)

    # Step 2 - Add slice layer (will currently ignore 'None' slices)

    starts = []
    sizes = []
    strides = []

    input_dim = 0

    input_size = input_.size()

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

    # Step 3 - Add shuffle layer to insert dimensions for 'None' slices and remove dimensions for 'int' slices

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
        layer.set_input(1, final_shape._trt)
        output_trt = layer.get_output(0)

    return output_trt


def _advanced_indexing(ctx, input_, indices):
    # Preprocess indices so that all following operations are on TRT tensors.
    indices_trt = _indices_to_trt(ctx, indices)
    indices_trt = _broadcast_indices(ctx, indices_trt)
    slices, gathers = _split_indices(indices_trt)

    # All indices are gather operations, so we trivially solve advanced indexing using a single gather layer.
    if len(gathers) == len(indices) and all([_is_trt_tensor(gather) for gather in gathers]):
        gathers = [_transpose_1d_trt(ctx, gather) for gather in gathers]
        gathers = _cat_trt(ctx, gathers, axis=1)
        return _gathernd_trt(ctx, input_._trt, gathers)

    # Indices are a mixture of slices and gathers;
    # therefore, we can solve this by first applying a slice layer for all slice operations,
    # then successively apply gather layers for each gather operation
    # and transposing out any slice operations between two gather operations.
    output_trt = _basic_indexing(ctx, input_, slices)
    output_trt = _advanced_gathernd_trt(ctx, output_trt, gathers)

    return output_trt


@tensorrt_converter('torch.Tensor.__getitem__')
def convert_tensor_getitem(ctx):
    input_ = ctx.method_args[0]
    indices = ctx.method_args[1]
    output = ctx.method_return

    if not hasattr(input_, '_trt'):
        return

    # Convert indices argument into a tuple if it is not already one.
    # This can happen when the provided indices argument is a single element (ie. a single index).
    #
    # In the general case, multiple index arguments will already be wrapped in a tuple.
    #
    # Note the special case where a single tuple is itself the only index argument (eg. t[(...)]);
    # PyTorch treats the single tuple argument as if its elements were provided as separate indices.
    # We handle this gracefully by default without requiring special handling.
    #
    # A tuple of tuple of elements is only possible if the inner tuple is combined with other arguments.
    # In this case, the inner tuple is now a gather operation and must be handled differently.
    # This is also commonly known as advanced indexing.
    #
    # Note that we don't actually need indices to be a tuple for processing, just an iterable.
    if not isinstance(indices, tuple):
        indices = (indices,)

    # Replace any ellipses (yes, plural) so that downstream processing is more straightforward.
    indices = _replace_ellipses(input_, indices)

    # We use basic indexing when only slicing.
    # Advanced indexing is only necessary when we perform gather operations.
    convert_getitem = _basic_indexing if not _requires_advanced_indexing(indices) else _advanced_indexing
    output._trt = convert_getitem(ctx, input_, indices)


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


@add_module_test(torch.float32, torch.device('cuda'), [(2, 5, 4, 3)])
def test_tensor_getitem_int_tuple():
    return LambdaModule(lambda x: x[0, (0, 1)])


@add_module_test(torch.float32, torch.device('cuda'), [(2, 5, 4, 3)])
def test_tensor_getitem_int_tensor():
    tensor = torch.tensor((0, 1))
    return LambdaModule(lambda x: x[0, tensor])


@add_module_test(torch.float32, torch.device('cuda'), [(2, 5, 4, 3)])
def test_tensor_getitem_tuple_int():
    return LambdaModule(lambda x: x[(0, 1), 0])


@add_module_test(torch.float32, torch.device('cuda'), [(2, 5, 4, 3)])
def test_tensor_getitem_tensor_int():
    tensor = torch.tensor((0, 1))
    return LambdaModule(lambda x: x[tensor, 0])


@add_module_test(torch.float32, torch.device('cuda'), [(2, 5, 4, 3)])
def test_tensor_getitem_int_tuple_int():
    return LambdaModule(lambda x: x[0, (0, 1), 0])


@add_module_test(torch.float32, torch.device('cuda'), [(2, 5, 4, 3)])
def test_tensor_getitem_int_tensor_int():
    tensor = torch.tensor((0, 1))
    return LambdaModule(lambda x: x[0, tensor, 0])


@add_module_test(torch.float32, torch.device('cuda'), [(2, 5, 4, 3)])
def test_tensor_getitem_tuple_int_tuple():
    return LambdaModule(lambda x: x[(0, 1), 0, (0, 1)])


@add_module_test(torch.float32, torch.device('cuda'), [(2, 5, 4, 3)])
def test_tensor_getitem_tensor_int_tensor():
    tensor = torch.tensor((0, 1))
    return LambdaModule(lambda x: x[tensor, 0, tensor])


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


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3)], max_batch_size=3)
def test_tensor_getitem_2d_gather():
    return LambdaModule(lambda x: x[(2,0,1),])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3)], max_batch_size=3)
def test_tensor_getitem_2d_gather_gather():
    return LambdaModule(lambda x: x[(2,0,1), (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3)], max_batch_size=3)
def test_tensor_getitem_2d_gather_broadcast():
    return LambdaModule(lambda x: x[(2,0,1), (2,)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3)], max_batch_size=3)
def test_tensor_getitem_2d_broadcast_gather():
    return LambdaModule(lambda x: x[(2,), (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3)], max_batch_size=3)
def test_tensor_getitem_2d_list_broadcast_gather():
    return LambdaModule(lambda x: x[[2,], [2,0,1]])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3)], max_batch_size=3)
def test_tensor_getitem_2d_tensor_broadcast_gather():
    t0 = torch.tensor((2,))
    t1 = torch.tensor((2,0,1))
    return LambdaModule(lambda x: x[t0, t1])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3)], max_batch_size=3)
def test_tensor_getitem_2d_gather_colon():
    return LambdaModule(lambda x: x[(2,0,1), :])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3)], max_batch_size=3)
def test_tensor_getitem_2d_colon_gather():
    return LambdaModule(lambda x: x[:, (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3)], max_batch_size=3)
def test_tensor_getitem_2d_gather_ellipsis():
    return LambdaModule(lambda x: x[(2,0,1), ...])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3)], max_batch_size=3)
def test_tensor_getitem_2d_ellipsis_gather():
    return LambdaModule(lambda x: x[..., (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3)], max_batch_size=3)
def test_tensor_getitem_2d_gather_none():
    return LambdaModule(lambda x: x[(2,0,1), None])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3)], max_batch_size=3)
def test_tensor_getitem_2d_none_gather():
    return LambdaModule(lambda x: x[None, (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_3d_gather():
    return LambdaModule(lambda x: x[(2,0,1),])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_3d_gather_gather():
    return LambdaModule(lambda x: x[(2,0,1), (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_3d_gather_gather_gather():
    return LambdaModule(lambda x: x[(2,0,1), (2,0,1), (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_3d_gather_broadcast_broadcast():
    return LambdaModule(lambda x: x[(2,0,1), (2,), (1,)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_3d_broadcast_broadcast_gather():
    return LambdaModule(lambda x: x[(2,), (1,), (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_3d_colon_gather_gather():
    return LambdaModule(lambda x: x[:, (2,0,1), (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_3d_gather_colon_gather():
    return LambdaModule(lambda x: x[(2,0,1), :, (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_3d_colon_colon_gather():
    return LambdaModule(lambda x: x[:, :, (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_3d_gather_none_gather():
    return LambdaModule(lambda x: x[(2,0,1), None, (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_3d_colon_none_gather():
    return LambdaModule(lambda x: x[:, None, (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_3d_gather_colon_none():
    return LambdaModule(lambda x: x[(2,0,1), :, None])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_3d_gather_ellipsis():
    return LambdaModule(lambda x: x[(2,0,1), ...])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_3d_ellipsis_gather():
    return LambdaModule(lambda x: x[..., (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_3d_gather_ellipsis_gather():
    return LambdaModule(lambda x: x[(2,0,1), ..., (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_4d_gather():
    return LambdaModule(lambda x: x[(2,0,1),])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_4d_gather_gather():
    return LambdaModule(lambda x: x[(2,0,1), (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_4d_gather_gather_gather():
    return LambdaModule(lambda x: x[(2,0,1), (2,0,1), (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_4d_gather_gather_gather_gather():
    return LambdaModule(lambda x: x[(2,0,1), (2,0,1), (2,0,1), (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_4d_gather_broadcast_broadcast_gather():
    return LambdaModule(lambda x: x[(2,0,1), (2,), (1,), (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_4d_colon_gather_gather_gather():
    return LambdaModule(lambda x: x[:, (2,0,1), (2,0,1), (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_4d_gather_colon_gather_gather():
    return LambdaModule(lambda x: x[(2,0,1), :, (2,0,1), (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_4d_gather_gather_colon_gather():
    return LambdaModule(lambda x: x[(2,0,1), (2,0,1), :, (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_4d_colon_colon_gather_gather():
    return LambdaModule(lambda x: x[:, :, (2,0,1), (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_4d_colon_gather_colon_gather():
    return LambdaModule(lambda x: x[:, (2,0,1), :, (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_4d_gather_colon_colon_gather():
    return LambdaModule(lambda x: x[(2,0,1), :, :, (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_4d_colon_colon_colon_gather():
    return LambdaModule(lambda x: x[:, :, :, (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_4d_ellipsis_gather():
    return LambdaModule(lambda x: x[..., (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_4d_gather_ellipsis():
    return LambdaModule(lambda x: x[(2,0,1), ...])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_4d_gather_ellipsis_gather():
    return LambdaModule(lambda x: x[(2,0,1), ..., (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_4d_none_gather_gather_gather():
    return LambdaModule(lambda x: x[None, (2,0,1), (2,0,1), (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_4d_gather_gather_none_gather():
    return LambdaModule(lambda x: x[(2,0,1), (2,0,1), None, (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_4d_gather_none_none_gather():
    return LambdaModule(lambda x: x[(2,0,1), None, None, (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_4d_none_none_none_gather():
    return LambdaModule(lambda x: x[None, None, None, (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_4d_gather_none_slice_gather():
    return LambdaModule(lambda x: x[(2,0,1), None, :, (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_4d_gather_slice_none_gather():
    return LambdaModule(lambda x: x[(2,0,1), :, None, (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_4d_broadcast_slice_none_gather():
    return LambdaModule(lambda x: x[(2,), :, None, (2,0,1)])


@add_module_test(torch.float32, torch.device('cuda'), [(3, 3, 3, 3)], max_batch_size=3)
def test_tensor_getitem_4d_mixed_indices_type():
    t0 = torch.tensor((2,))
    t1 = torch.tensor((2,0,1))
    return LambdaModule(lambda x: x[[2,], t0, t1, (2,0,1)])
