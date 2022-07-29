from torch2trt.torch2trt import (
    torch2trt, 
    trt,
    tensorrt_converter,
    get_conversion_context,
    get_arg
)


# SHAPE WRAPPING
_int = int
_int_mul = int.__mul__
_int_add = int.__add__
_int_sub = int.__sub__
_int_floordiv = int.__floordiv__


class IntWrapper(int):
    
    @property
    def _trt(self):
        if not hasattr(self, '_raw_trt'):
            ctx = get_conversion_context()
            self._raw_trt = ctx.network._network.add_constant([1], np.array([_int(self)], dtype=np.int32)).get_output(0)
        return self._raw_trt

    def __mul__(self, x):
        ctx = get_conversion_context()
        result = IntWrapper(_int_mul(self, x))
        result._raw_trt = ctx.network._network.add_elementwise(self._trt, x._trt, trt.ElementWiseOperation.PROD).get_output(0)
        return result

    def __add__(self, x):
        ctx = get_conversion_context()
        result = IntWrapper(_int_add(self, x))
        result._raw_trt = ctx.network._network.add_elementwise(self._trt, x._trt, trt.ElementWiseOperation.SUM).get_output(0)
        return result

    def __sub__(self, x):
        ctx = get_conversion_context()
        result = IntWrapper(_int_sub(self, x))
        result._raw_trt = ctx.network._network.add_elementwise(self._trt, x._trt, trt.ElementWiseOperation.SUB).get_output(0)
        return result

    def __floordiv__(self, x):
        ctx = get_conversion_context()
        result = IntWrapper(_int_floordiv(self, x))
        result._raw_trt = ctx.network._network.add_elementwise(self._trt, x._trt, trt.ElementWiseOperation.FLOOR_DIV).get_output(0)
        return result


class SizeWrapper(tuple):
    
    @property
    def _trt(self):
        if not hasattr(self, '__trt'):
            ctx = get_conversion_context()
            self._raw_trt = ctx.network._network.add_concatenation([d._trt for d in self]).get_output(0)
        return self._raw_trt


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
