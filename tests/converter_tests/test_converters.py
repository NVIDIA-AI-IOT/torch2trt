import pytest
import torch
import torch2trt
import torch.nn as nn
from torch2trt.flattener import Flattener


def cross_validate(
        module, 
        inputs,
        fp16_mode: bool,
        tol: float
    ):

    module = module
    

    module_trt = torch2trt.torch2trt(
        module,
        inputs,
        fp16_mode=fp16_mode
    )
    
    output = module(*inputs)
    output_trt = module_trt(*inputs)

    flattener = Flattener.from_value(output)

    output = flattener.flatten(output)
    output_trt = flattener.flatten(output_trt)

    for output_tensor, output_tensor_trt in zip(output, output_trt):
        assert torch.allclose(
            output_tensor, output_tensor_trt, 
        atol=tol, rtol=tol)



# MODULES
    

class UnaryModule(torch.nn.Module):
    def __init__(self, fn):
        super(UnaryModule, self).__init__()
        self.fn = fn
        
    def forward(self, x):
        return self.fn(x)


class BinaryModule(torch.nn.Module):
    def __init__(self, fn):
        super(BinaryModule, self).__init__()
        self.fn = fn
        
    def forward(self, a, b):
        return self.fn(a, b)
# TESTS



@pytest.mark.parametrize("fp16_mode,tol", [(False, 1e-1), (True, 1e-1)])
def test_leaky_relu(fp16_mode, tol):
    module = UnaryModule(lambda x: torch.nn.functional.leaky_relu(x)).cuda().eval()
    inputs = [torch.randn(1, 3, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=fp16_mode, tol=tol)


@pytest.mark.parametrize("fp16_mode,tol", [(False, 1e-1), (True, 1e-1)])
def test_elu(fp16_mode, tol):
    module = UnaryModule(lambda x: torch.nn.functional.elu(x)).cuda().eval()
    inputs = [torch.randn(1, 3, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=fp16_mode, tol=tol)


@pytest.mark.parametrize("fp16_mode,tol", [(False, 1e-1), (True, 1e-1)])
def test_selu(fp16_mode, tol):
    module = UnaryModule(lambda x: torch.nn.functional.selu(x)).cuda().eval()
    inputs = [torch.randn(1, 3, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=fp16_mode, tol=tol)


@pytest.mark.parametrize("fp16_mode,tol", [(False, 1e-1), (True, 1e-1)])
def test_softsign(fp16_mode, tol):
    module = UnaryModule(lambda x: torch.nn.functional.selu(x)).cuda().eval()
    inputs = [torch.randn(1, 3, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=fp16_mode, tol=tol)


@pytest.mark.parametrize("fp16_mode,tol", [(False, 1e-1), (True, 1e-1)])
def test_softplus(fp16_mode, tol):
    module = UnaryModule(lambda x: torch.nn.functional.softplus(x)).cuda().eval()
    inputs = [torch.randn(1, 3, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=fp16_mode, tol=tol)


@pytest.mark.parametrize("output_size,fp16_mode,tol", [
    ((1, 1), False, 1e-1), 
    ((2, 2), False, 1e-1),
    ((1, 1), True, 1e-1)
])
def test_adaptive_avg_pool2d(output_size, fp16_mode, tol):
    module = UnaryModule(lambda x: torch.nn.functional.adaptive_avg_pool2d(x, output_size)).cuda().eval()
    inputs = [torch.randn(1, 3, 4, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=fp16_mode, tol=tol)


@pytest.mark.parametrize("output_size,fp16_mode,tol", [
    ((1, 1, 1), False, 1e-1), 
    ((2, 2, 2), False, 1e-1), 
    ((1, 1, 1), True, 1e-1)
])
def test_adaptive_avg_pool3d(output_size, fp16_mode, tol):
    module = UnaryModule(lambda x: torch.nn.functional.adaptive_avg_pool3d(x, output_size)).cuda().eval()
    inputs = [torch.randn(1, 3, 4, 4, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=fp16_mode, tol=tol)


@pytest.mark.parametrize("output_size,fp16_mode,tol", [
    ((1, 1), False, 1e-1), 
    ((2, 2), False, 1e-1),
    ((1, 1), True, 1e-1)
])
def test_adaptive_max_pool2d(output_size, fp16_mode, tol):
    module = UnaryModule(lambda x: torch.nn.functional.adaptive_max_pool2d(x, output_size)).cuda().eval()
    inputs = [torch.randn(1, 3, 4, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=fp16_mode, tol=tol)


@pytest.mark.parametrize("output_size,fp16_mode,tol", [
    ((1, 1, 1), False, 1e-1), 
    ((2, 2, 2), False, 1e-1), 
    ((1, 1, 1), True, 1e-1)
])
def test_adaptive_max_pool3d(output_size, fp16_mode, tol):
    module = UnaryModule(lambda x: torch.nn.functional.adaptive_max_pool3d(x, output_size)).cuda().eval()
    inputs = [torch.randn(1, 3, 4, 4, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=fp16_mode, tol=tol)


def test_add():
    module = BinaryModule(lambda a, b: a + b).cuda().eval()
    inputs = [torch.randn(1, 3, 4).cuda(), torch.randn(1, 3, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-2)


def test_torch_add():
    module = BinaryModule(lambda a, b: torch.add(a, b)).cuda().eval()
    inputs = [torch.randn(1, 3, 4).cuda(), torch.randn(1, 3, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-2)


def test_iadd():
    class IAdd(torch.nn.Module):
        def __init__(self):
            super(IAdd, self).__init__()

        def forward(self, x, y):
            x += y
            return x

    module = IAdd().cuda().eval()
    inputs = [torch.ones(1, 3, 4).cuda(), torch.ones(1, 3, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-2)


def test_radd_int():
    module = UnaryModule(lambda x: 1 + x).cuda().eval()
    inputs = [torch.randn(1, 3, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-2)


def test_radd_float():
    module = UnaryModule(lambda x: 1.0 + x).cuda().eval()
    inputs = [torch.randn(1, 3, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-2)


# TODO: radd, add, iadd
    

@pytest.mark.parametrize("with_conv", [True, False])
@pytest.mark.parametrize("nd", [1,2,3])
def test_batch_norm_nd(nd, with_conv):
    modules = []
    if nd == 1:
        if with_conv:
            modules.append(nn.Conv1d(3, 3, 1)) # with conv, because scale layer not implemented sometimes.
        modules.append(nn.BatchNorm1d(3))
    if nd == 2:
        if with_conv:
            modules.append(nn.Conv2d(3, 3, 1))
        modules.append(nn.BatchNorm2d(3))
    if nd == 3:
        if with_conv:
            modules.append(nn.Conv3d(3, 3, 1))
        modules.append(nn.BatchNorm3d(3))

    module = nn.Sequential(*modules).cuda().eval()

    input_size = [2, 3] + [4] * nd

    inputs = [torch.randn(*input_size).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)


@pytest.mark.parametrize("dim", [1, -1])
def test_cat(dim):
    module = UnaryModule(lambda x: torch.cat([x, x], dim=dim)).cuda().eval()
    inputs = [torch.randn(1, 3, 3).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)


@pytest.mark.parametrize("chunks,dim", [
    (1, 1),
    (3, 1)
])
def test_chunk(chunks, dim):
    module = UnaryModule(lambda x: torch.chunk(x, chunks, dim)).cuda().eval()
    inputs = [torch.randn(1, 3, 3).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)
    

@pytest.mark.parametrize("split_sections_or_size,dim", [
    (1, 1),
    ([1, 2], 1),
    (1, -1)
])
def test_split_sections(split_sections_or_size, dim):
    module = UnaryModule(lambda x: torch.split(x, split_sections_or_size, dim)).cuda().eval()
    inputs = [torch.randn(1, 3, 3).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)


@pytest.mark.parametrize("min,max", [
    (None, 0.5),
    (-0.5, None),
    (-0.5, 0.5)
])
def test_clamp(min, max):
    module = UnaryModule(lambda x: torch.clamp(x, min, max)).cuda().eval()
    inputs = [torch.randn(1, 8, 8).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)


def test_clone():
    module = UnaryModule(lambda x: x.clone()).cuda().eval()
    inputs = [torch.randn(1, 8, 8).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)
    

def test_gt():
    module = BinaryModule(lambda x, y: x > y).cuda().eval()
    inputs = [torch.randn(1, 4, 4).cuda(), torch.randn(1, 4, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)


@pytest.mark.parametrize("val", [0, 0.0])
def test_gt_scalar(val):
    module = UnaryModule(lambda x: x > 0).cuda().eval()
    inputs = [torch.randn(1, 4, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)


def test_lt():
    module = BinaryModule(lambda x, y: x < y).cuda().eval()
    inputs = [torch.randn(1, 4, 4).cuda(), torch.randn(1, 4, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)


@pytest.mark.parametrize("val", [0, 0.0])
def test_lt_scalar(val):
    module = UnaryModule(lambda x: x < 0).cuda().eval()
    inputs = [torch.randn(1, 4, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)

def test_eq():
    module = BinaryModule(lambda x, y: x == y).cuda().eval()
    inputs = [torch.zeros(1, 4, 4).cuda(), torch.zeros(1, 4, 4).cuda()]

    inputs[0][0, 1, 1] = 1
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)


@pytest.mark.parametrize("val", [0, 0.0])
def test_eq_scalar(val):
    module = UnaryModule(lambda x: x == 0).cuda().eval()
    inputs = [torch.zeros(1, 4, 4).cuda()]
    inputs[0][0, 1, 1] = 1
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)


@pytest.mark.parametrize(
    "kernel_size,stride,padding,dilation,groups,bias,fp16_mode", [
    (3, 1, 1, 1, 1, True, False),
    (3, 2, 1, 1, 1, True, False),
    (3, 1, 0, 1, 1, True, False),
    (3, 1, 1, 1, 1, True, False),
    (3, 1, 1, 2, 1, True, False),
    (3, 1, 1, 1, 3, True, False),
    (3, 1, 1, 1, 1, False, False),
    (3, 1, 1, 1, 1, True, True),
])
@pytest.mark.parametrize("nd", [1,2,3])
def test_conv(
        nd,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        fp16_mode
):
    if nd == 1:
        cls = nn.Conv1d
    elif nd == 2:
        cls = nn.Conv2d
    elif nd == 3:
        cls = nn.Conv3d

    module = cls(3, 3,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias
    ).cuda().eval()
    shape = [1, 3] + [16] * nd
    inputs = [torch.randn(*shape).cuda()]
    cross_validate(module, inputs, fp16_mode=fp16_mode, tol=1e-1)


@pytest.mark.parametrize(
    "kernel_size,stride,padding,dilation,groups,bias,fp16_mode", [
    (3, 1, 1, 1, 1, True, False),
    (3, 2, 1, 1, 1, True, False),
    (3, 1, 0, 1, 1, True, False),
    (3, 1, 1, 1, 1, True, False),
    (3, 1, 1, 2, 1, True, False),
    (3, 1, 1, 1, 3, True, False),
    (3, 1, 1, 1, 1, False, False),
    (3, 1, 1, 1, 1, True, True),
])
@pytest.mark.parametrize("nd", [1,2,3])
def test_conv_transpose(
        nd,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        fp16_mode
):
    if nd == 1:
        cls = nn.ConvTranspose1d
    elif nd == 2:
        cls = nn.ConvTranspose2d
    elif nd == 3:
        cls = nn.ConvTranspose3d

    module = cls(3, 3,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias
    ).cuda().eval()
    shape = [1, 3] + [16] * nd
    inputs = [torch.randn(*shape).cuda()]
    cross_validate(module, inputs, fp16_mode=fp16_mode, tol=1e-1)


def test_div():
    module = BinaryModule(lambda x, y: x / y).cuda().eval()
    inputs = [torch.randn(1, 4, 4).cuda(), torch.ones(1, 4, 4).cuda()*2]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)

@pytest.mark.parametrize(
    "val", [2, 2.0]
)
def test_div_scalar(val):
    module = UnaryModule(lambda x: x / val).cuda().eval()
    inputs = [torch.randn(1, 4, 4).cuda()]
    
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)

@pytest.mark.parametrize(
    "val", [2, 2.0]
)
def test_idiv_scalar(val):
    def fn(x):
        x /= val
        return x
    module = UnaryModule(fn).cuda().eval()
    inputs = [torch.ones(1, 4, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)


@pytest.mark.parametrize(
    "einsum_expr", [
        "bij,bjk->bik"
    ]
)
def test_einsum_binary(einsum_expr):
    module = BinaryModule(lambda x, y: torch.einsum(einsum_expr, x, y)).cuda().eval()
    inputs = [torch.randn(1, 3, 4).cuda(), torch.randn(1, 4, 5).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)


@pytest.mark.parametrize(
    "sizes", 
    [
        (3, 4),
        (-1, 4)
    ]
)
def test_expand(sizes):
    module = UnaryModule(lambda x: x.expand(*sizes)).cuda().eval()
    inputs = [torch.randn(3, 1).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)


@pytest.mark.parametrize(
    "start_dim,end_dim",
    [
        (0, -1),
        (1, -1),
        (1, 3)
    ]
)
def test_flatten(start_dim, end_dim):
    module = UnaryModule(lambda x: torch.flatten(x, start_dim, end_dim)).cuda().eval()
    inputs = [torch.randn(1, 2, 3, 4, 5).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)


@pytest.mark.parametrize("denom", [1., 2.])
def test_floordiv(denom):
    module = BinaryModule(lambda x, y: x // y).cuda().eval()
    inputs = [torch.ones(1, 2, 3, 4, 5).cuda()]
    inputs.append(torch.ones_like(inputs[0]) * denom)
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)


@pytest.mark.parametrize(
    "scalar", [
        2, 2.0
    ]
)
def test_floordiv_scalar(scalar):
    module = UnaryModule(lambda x: x // scalar).cuda().eval()
    inputs = [torch.randn(1, 2, 3).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)


def test_gelu():
    module = nn.GELU().cuda().eval()
    inputs = [torch.randn(1, 2, 3).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)


@pytest.mark.parametrize("expr", [
    lambda x: x[:, 0],
    lambda x: x[..., 0],
    lambda x: x[:, :, 0:2],
    lambda x: x[..., None],
    lambda x: x[:, :, -1]
])
def test_getitem(expr):
    module = UnaryModule(lambda x: expr(x)).cuda().eval()
    inputs = [torch.randn(1, 2, 3, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)


@pytest.mark.parametrize("num_groups,num_channels,affine", [
    (1, 6, False),
    (3, 6, False),
    (3, 6, True)
]
)
def test_group_norm(num_groups, num_channels, affine):
    module = nn.GroupNorm(num_groups, num_channels, affine=affine).cuda().eval()
    inputs = [torch.randn(1, num_channels, 3, 4).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)

@pytest.mark.parametrize("nd", [1, 2, 3])
@pytest.mark.parametrize("num_channels", [4])
def test_instance_norm(nd, num_channels):
    cls_map = {1: nn.InstanceNorm1d, 2: nn.InstanceNorm2d, 3: nn.InstanceNorm3d}
    module = cls_map[nd](num_channels)

    shape = [1, num_channels] + [4] * nd
    inputs = [torch.randn(*shape).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)
    

@pytest.mark.parametrize(
    "input_size,output_size,scale_factor,mode,align_corners",
    [
        ((4,), (8,), None, "nearest", None),
        ((4, 4), (8, 8), None, "nearest", None),
        ((4, 4, 4), (8, 8, 8), None, "nearest", None),
        ((4,), None, 2, "nearest", None),
        ((4, 4), None, 2, "nearest", None),
        ((4, 4, 4), None, 2, "nearest", None),
        ((4,), None, 2, "linear", None),
        ((4, 4), None, 2, "bilinear", None),
        ((4, 4, 4), None, 2, "trilinear", None)
])
def test_interpolate_size(input_size, output_size, scale_factor, mode, align_corners):
    

    module = UnaryModule(lambda x: torch.nn.functional.interpolate(
        x, size=output_size, mode=mode,
        align_corners=align_corners,
        scale_factor=scale_factor
    )).cuda().eval()
    
    input_size = [1, 3] + list(input_size)
    inputs = [torch.randn(*input_size).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)


def test_layer_norm():

    module = nn.LayerNorm(8).cuda().eval()

    inputs = [torch.randn(1, 4, 8).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)

@pytest.mark.parametrize("input_shapes", [
    (1, 4)
])
def test_linear(input_shapes):
    module = nn.Linear(4, 8).cuda().eval()
    inputs = [torch.randn(*input_shapes).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)


@pytest.mark.parametrize("dim", [1])
def test_log_softmax(dim: int):
    module = nn.LogSoftmax(dim).cuda().eval()
    inputs = [torch.randn(1, 2).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)


@pytest.mark.parametrize(
    "shape_a,shape_b", [
        ((1, 2, 3), (1, 3, 4)),
        ((2, 2, 3), (2, 3, 4))
    ]
)
def test_matmul(shape_a, shape_b):
    module = BinaryModule(lambda x, y: torch.matmul(x, y)).cuda().eval()

    inputs = [torch.randn(*shape_a).cuda(), torch.randn(*shape_b).cuda()]
    
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)


@pytest.mark.parametrize(
    "kernel_size,stride,padding,dilation,ceil_mode", [
        (3, 2, 1, 1, False),
    ]
)
@pytest.mark.parametrize("nd", [1,2,3])
def test_max_pool_nd(nd, kernel_size, stride, padding, dilation, ceil_mode):
    if nd == 1:
        cls = nn.MaxPool1d
    elif nd == 2:
        cls = nn.MaxPool2d
    elif nd == 3:
        cls = nn.MaxPool3d
    module = cls(kernel_size,stride,padding,dilation,ceil_mode=ceil_mode).cuda().eval()
    input_size = [1, 3] + [4]*nd
    inputs = [torch.randn(*input_size).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)


@pytest.mark.parametrize(
    "kernel_size,stride,padding,ceil_mode,count_include_pad", [
        (3, 2, 1, False, False),
    ]
)
@pytest.mark.parametrize("nd", [1,2,3])
def test_avg_pool_nd(nd, kernel_size, stride, padding, ceil_mode, count_include_pad):
    if nd == 1:
        cls = nn.AvgPool1d
    elif nd == 2:
        cls = nn.AvgPool2d
    elif nd == 3:
        cls = nn.AvgPool3d
    module = cls(kernel_size,stride,padding,ceil_mode=ceil_mode, count_include_pad=count_include_pad).cuda().eval()
    input_size = [1, 3] + [4]*nd
    inputs = [torch.randn(*input_size).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)


@pytest.mark.parametrize("op", ["min","max", "fmod"])
def test_binary_op_elementwise(op):
    if op == "max":
        fn = lambda x, y: torch.max(x, y)
    elif op == "min":
        fn = lambda x, y: torch.min(x, y)
    elif op == "fmod":
        fn = lambda x, y: torch.fmod(x, y)


    module = BinaryModule(fn).cuda().eval()
    inputs = [torch.randn(1, 3, 3).cuda(), torch.randn(1, 3, 3).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)

unary_1d_randn_ops = {
    "torch.max": lambda x: torch.max(x),
    "torch.Tensor.max": lambda x: x.max(),
    "torch.min": lambda x: torch.min(x),
    "torch.Tensor.min": lambda x: x.min(),
    "torch.mean": lambda x: torch.mean(x),
    "torch.Tensor.mean": lambda x: x.mean(),
    "torch.sum": lambda x: torch.sum(x),
    "torch.Tensor.sum": lambda x: x.sum(),
    "torch.prod": lambda x: torch.prod(x),
    "torch.Tensor.prod": lambda x: x.prod(),
    "torch.relu": lambda x: torch.relu(x),
    "torch.nn.functional.relu": lambda x: torch.nn.functional.relu(x),
    "torch.Tensor.relu": lambda x: x.relu(),
    "torch.nn.functional.relu6": lambda x: torch.nn.functional.relu6(x),
    "torch.sigmoid": lambda x: torch.sigmoid(x),
    "torch.nn.functional.sigmoid": lambda x: torch.nn.functional.sigmoid(x),
    "torch.Tensor.sigmoid": lambda x: x.sigmoid(),
    "torch.nn.functional.silu": lambda x: torch.nn.functional.silu(x),
    "torch.Tensor.softmax": lambda x: x.softmax(1),
    "torch.nn.functional.softmax": lambda x: torch.nn.functional.softmax(x, 1),
    "torch.Tensor.squeeze": lambda x: x.squeeze(),
    "torch.squeeze": lambda x: torch.squeeze(x),
    "torch.stack": lambda x: torch.stack([x, x], dim=1),
    "torch.sub": lambda x: torch.sub(x, x),
    "torch.Tensor.__sub__": lambda x: x - x,
    "torch.Tensor.__rsub__[int]": lambda x: 1 - x,
    "torch.Tensor.__rsub__[float]": lambda x: 1.0 - x,
    "torch.tanh": lambda x: torch.tanh(x),
    "torch.nn.functional.tanh": lambda x: torch.nn.functional.tanh(x),
    "torch.tensor": lambda x: torch.tensor(x),
    "torch.Tensor.transpose": lambda x: x.transpose(1, 2),
    "torch.transpose": lambda x: torch.transpose(x, 1, 2),
    "torch.exp": lambda x: torch.exp(x),
    "torch.Tensor.exp": lambda x: x.exp(),
    "torch.abs": lambda x: torch.abs(x),
    "torch.Tensor.abs": lambda x: x.abs(),
    "torch.neg": lambda x: torch.neg(x),
    "torch.Tensor.neg": lambda x: -x,
    "torch.sin": lambda x: torch.sin(x),
    "torch.Tensor.sin": lambda x: x.sin(),
    "torch.cos": lambda x: torch.cos(x),
    "torch.Tensor.cos": lambda x: x.cos(),
    "torch.sinh": lambda x: torch.sinh(x),
    "torch.Tensor.sinh": lambda x: x.sinh(),
    "torch.cosh": lambda x: torch.cosh(x),
    "torch.Tensor.cosh": lambda x: x.cosh(),
    "torch.atan": lambda x: torch.atan(x),
    "torch.Tensor.atan": lambda x: x.atan(),
    "torch.ceil": lambda x: torch.ceil(x),
    "torch.Tensor.ceil": lambda x: x.ceil(),
    "torch.floor": lambda x: torch.floor(x),
    "torch.Tensor.floor": lambda x: x.floor()
}

@pytest.mark.parametrize("op", unary_1d_randn_ops.keys())
def test_unary_1d_randn(op):
    module = UnaryModule(unary_1d_randn_ops[op]).cuda().eval()
    inputs = [torch.randn(1, 3, 3).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)


unary_1d_positive_ops = {
    "torch.log": lambda x: torch.log(x),
    "torch.Tensor.log": lambda x: x.log(),
    "torch.sqrt": lambda x: torch.sqrt(x),
    "torch.Tensor.sqrt": lambda x: x.sqrt(),
    "torch.reciprocal": lambda x: torch.reciprocal(x),
    "torch.Tensor.reciprocal": lambda x: x.reciprocal(),
    "torch.tan": lambda x: torch.tan(x),
    "torch.Tensor.tan": lambda x: x.tan(),
    "torch.asin": lambda x: torch.asin(x),
    "torch.Tensor.asin": lambda x: x.asin(),
    "torch.acos": lambda x: torch.acos(x),
    "torch.Tensor.acos": lambda x: x.acos(),
}

@pytest.mark.parametrize("op", unary_1d_positive_ops.keys())
def test_unary_1d_ones(op):
    module = UnaryModule(unary_1d_positive_ops[op]).cuda().eval()
    inputs = [0.5 * torch.ones(1, 3, 3).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)


@pytest.mark.parametrize("op", ["mul", "__mul__", "__rmul__"])
@pytest.mark.parametrize("scalar", [2, 2.])
def test_mul_scalar(op, scalar):

    op_map = {
        "mul": lambda x: torch.mul(x, scalar),
        "__mul__": lambda x: x * scalar,
        "__rmul__": lambda x: scalar * x
    }

    module = UnaryModule(op_map[op]).cuda().eval()

    inputs = [torch.randn(1, 3, 3).cuda()]

    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)

@pytest.mark.parametrize("dim,start,length", [
    (0, 0, 2),
    (1, 1, 2),
    (-1, -1, 1)
])
def test_narrow(dim, start, length):
    module = UnaryModule(lambda x: torch.narrow(x, dim, start, length)).cuda().eval()

    inputs = [torch.randn(3, 3).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)


def test_ne_binary():
    module = BinaryModule(lambda x, y: x != y).cuda().eval()
    inputs = [torch.zeros(1, 3, 3).cuda()]
    inputs.append(inputs[0].clone())
    inputs[0][0] = 1

    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("dim", [1, 2])
def test_normalize(p, dim):
    module = UnaryModule(lambda x: torch.nn.functional.normalize(x, p, dim)).cuda().eval()
    inputs = [torch.zeros(1, 3, 3).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)
    

@pytest.mark.parametrize("pad,mode,value", [
    ((1, 1), "constant", 0.),
    ((1, 1, 2, 2), "constant", 0.),
    ((0, 1, 2, 1, 3, 3), "constant", 0.),
])
def test_pad(pad, mode, value):
    module = UnaryModule(
        lambda x: torch.nn.functional.pad(
            x, pad, mode, value
        )
    ).cuda().eval()
    inputs = [torch.randn(3, 3, 4, 2).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)
    

@pytest.mark.parametrize("permutation", [
    (0, 2, 1),
    (0, 2, 1, 3)
])
def test_permute(permutation):

    module = UnaryModule(
        lambda x: x.permute(*permutation)
    ).cuda().eval()
    sizes = [i + 1 for i in range(len(permutation))]

    inputs = [torch.randn(*sizes).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)

@pytest.mark.parametrize("op", [
    "torch.pow",
    "torch.Tensor.__ipow__",
    "torch.Tensor.__pow__",
    "torch.Tensor.__rpow__"
])
@pytest.mark.parametrize("scalar", [2, 2.])
def test_scalar_op(op, scalar):
    if op == "torch.pow":
        fn = lambda x: torch.pow(x, scalar)
    elif op == "torch.Tensor.__ipow__":
        def ipow(x):
            x **= scalar
            return x
        fn = ipow
    elif op == "torch.Tensor.__pow__":
        fn = lambda x: x ** scalar
    elif op == "torch.Tensor.__rpow__":
        fn = lambda x: scalar ** x


    module = UnaryModule(fn).cuda().eval()
    inputs = [torch.randn(1, 2).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)


def test_prelu():
    module = nn.PReLU(4).cuda().eval()
    inputs = [torch.randn(1, 4, 3, 3).cuda()]
    cross_validate(module, inputs, fp16_mode=False, tol=1e-1)

