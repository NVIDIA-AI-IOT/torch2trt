# Custom Converter

This page details how to extend or modify the behavior of torch2trt by implementing and registering
custom converters.

## Background

torch2trt works by attaching conversion functions (like ``convert_ReLU``) to the original 
PyTorch functional calls (like ``torch.nn.ReLU.forward``).  The sample input data is passed
through the network, just as before, except now whenever a registered function (``torch.nn.ReLU.forward``)
is encountered, the corresponding converter (``convert_ReLU``) is also called afterwards.  The converter
is passed the arguments and return statement of the original PyTorch function, as well as the TensorRT
network that is being constructed.  The input tensors to the original PyTorch function are modified to
have an attribute ``_trt``, which is the TensorRT counterpart to the PyTorch tensor.  The conversion function
uses this ``_trt`` to add layers to the TensorRT network, and then sets the ``_trt`` attribute for
relevant output tensors.  Once the model is fully executed, the final tensors returns are marked as outputs
of the TensorRT network, and the optimized TensorRT engine is built.

## Add a custom converter

Here we show how to add a converter for the ``ReLU`` module using the TensorRT
python API.

```python
import tensorrt as trt
from torch2trt import tensorrt_converter

@tensorrt_converter('torch.nn.ReLU.forward')
def convert_ReLU(ctx):
    input = ctx.method_args[1]
    output = ctx.method_return
    layer = ctx.network.add_activation(input=input._trt, type=trt.ActivationType.RELU)  
    output._trt = layer.get_output(0)
```

The converter takes one argument, a ``ConversionContext``, which will contain
the following

* ``ctx.network`` - The TensorRT network that is being constructed.

* ``ctx.method_args`` - Positional arguments that were passed to the specified PyTorch function.  The ``_trt`` attribute is set for relevant input tensors.
* ``ctx.method_kwargs`` - Keyword arguments that were passed to the specified PyTorch function.
* ``ctx.method_return`` - The value returned by the specified PyTorch function.  The converter must set the ``_trt`` attribute where relevant.

Please see the [converters](../converters.md) page for a list of implemented converters and links to their source code.  These may help
in learning how to write converters.
