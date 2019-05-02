# torch2trt

torch2trt is a PyTorch to TensorRT converter which utilizes the 
TensorRT Python API.  The converter is

* Easy to use - Convert models with a single function call ``torch2trt``

* Easy to extend - Write your own layer converter in Python and register it with ``@tensorrt_converter``

If you find an issue, please [let us know](../..//issues)!  We'd also love to hear if you create your own ``@tensorrt_converter``. It may be helpful to others.

### Setup

```bash
python setup.py install --user
```

### Usage

```python
from torch2trt import torch2trt
from torchvision.models.alexnet import alexnet

# create some regular pytorch model...
model = alexnet(pretrained=True).eval().cuda()

# create example data
x = torch.ones((1, 3, 224, 224)).cuda()

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x])
```

We can then test the output of the regular and TensorRT optimized models

```
y = model(x)
y_trt = model_trt(x)

print(torch.max(torch.abs(y - y_trt)))
```

### Tested models

Below are models that we benchmarked on NVIDIA Jetson Nano.  Timing just includes model execution (not data copy).

| Model | PyTorch FP16 (Jetson Nano) | TensorRT FP16 (Jetson Nano) |
|-------|--------------|-----------------|
| alexnet | 18ms | 13ms |
| squeezenet1_0 | 21ms | 8.4ms |
| squeezenet1_1 | 13ms | 4.7ms |
| resnet18 | 32ms | 11ms |
| resnet34 | 58ms | 21ms |
| resnet50 | 77ms | 38ms |
| resnet101 | 135ms | 62ms |
| resnet152 | 200ms | 93ms |
| densenet121 | 83ms | 46ms |
| densenet169 | 116ms | 58ms |
| densenet201 | 139ms | 75ms |
| densenet161 | 209ms | 97ms |
| vgg11 | 61ms | 17ms |
| vgg13 | 96ms | 33ms |
| vgg16 | 137ms | 44ms |
| vgg19 |  |  |
| vgg11_bn |  |  |
| vgg13_bn |  |  |
| vgg16_bn |  |  |
| vgg19_bn |  |  |
| [mobilenet_v2](https://github.com/tonylins/pytorch-mobilenet-v2) | 27ms | 16ms |


### How does it work?

This converter works by attaching conversion functions (like ``convert_ReLU``) to the original 
PyTorch functional calls (like ``torch.nn.ReLU.forward``).  The sample input data is passed
through the network, just as before, except now whenever a registered function (``torch.nn.ReLU.forward``)
is encountered, the corresponding converter (``convert_ReLU``) is also called afterwards.  The converter
is passed the arguments and return statement of the original PyTorch function, as well as the TensorRT
network that is being constructed.  The input tensors to the original PyTorch function are modified to
have an attribute ``_trt``, which is the TensorRT counterpart to the PyTorch tensor.  The conversion function
uses this ``_trt`` to add layers to the TensorRT network, and then sets the ``_trt`` attribute for
relevant output tensors.  Once the model is fully executed, the final tensors returns are marked as outputs
of the TensorRT network, and the optimized TensorRT engine is built.

### How to add (or override) a converter

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

Please see the ``torch2trt.py`` module for more examples.

### A comment on variable size tensors

TensorRT currently does not support variable size Tensors, so whatever input shape you use when converting, you must use
when executing.  While this may seem
limiting, it can actually be a good constraint when designing your model for use in embedded systems.  By 
restricting to a fixed input size, we can expect similar memory usage and runtime.  Ultimately, even if 
TensorRT didn't have this constraint, you'd probably want to have it anyways :)
