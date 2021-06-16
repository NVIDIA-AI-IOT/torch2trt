from .torch2trt import *
import numpy as np
    

def tensordot(a, b, axes=2):
    try:
        iter(axes)
    except Exception:
        axes_a = list(range(-axes, 0))
        axes_b = list(range(0, axes))
    else:
        axes_a, axes_b = axes
    try:
        na = len(axes_a)
        axes_a = list(axes_a)
    except TypeError:
        axes_a = [axes_a]
        na = 1
    try:
        nb = len(axes_b)
        axes_b = list(axes_b)
    except TypeError:
        axes_b = [axes_b]
        nb = 1

#     a, b = np.asarray(a), np.asarray(b)
    as_ = a.shape
    nda = len(as_) #a.ndim
    bs = b.shape
    ndb = len(bs) #b.ndim
    equal = True
    if na != nb:
        equal = False
    else:
        for k in range(na):
            if as_[axes_a[k]] != bs[axes_b[k]]:
                equal = False
                break
            if axes_a[k] < 0:
                axes_a[k] += nda
            if axes_b[k] < 0:
                axes_b[k] += ndb
    if not equal:
        raise ValueError("shape-mismatch for sum")

    # Move the axes to sum over to the end of "a"
    # and to the front of "b"
    notin = [k for k in range(nda) if k not in axes_a]
    newaxes_a = notin + axes_a
    N2 = 1
    for axis in axes_a:
        N2 *= as_[axis]
    newshape_a = (int(np.multiply.reduce([as_[ax] for ax in notin])), N2)
    olda = [as_[axis] for axis in notin]

    notin = [k for k in range(ndb) if k not in axes_b]
    newaxes_b = axes_b + notin
    N2 = 1
    for axis in axes_b:
        N2 *= bs[axis]
    newshape_b = (N2, int(np.multiply.reduce([bs[ax] for ax in notin])))
    oldb = [bs[axis] for axis in notin]
    
    network = active_context().network
    
    at = network.add_shuffle(a)
    at.first_transpose = newaxes_a
    at.reshape_dims = newshape_a
    at = at.get_output(0)
    
    bt = network.add_shuffle(b)
    bt.first_transpose = newaxes_b
    bt.reshape_dims = newshape_b
    bt = bt.get_output(0)
    
#     at = a.transpose(newaxes_a).reshape(newshape_a)
#     bt = b.transpose(newaxes_b).reshape(newshape_b)
    res = network.add_matrix_multiply(at, trt.MatrixOperation.NONE, bt, trt.MatrixOperation.NONE).get_output(0)
    res = network.add_shuffle(res)
    res.reshape_dims = olda + oldb
    return res.get_output(0)


def transpose(x, axes=None):
    network = active_context().network
    xt = network.add_shuffle(x)
    xt.first_transpose = axes
    return xt.get_output(0)
