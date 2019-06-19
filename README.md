# torch2trt

torch2trt is a PyTorch to TensorRT converter which utilizes the 
TensorRT Python API.  The converter is

* Easy to use - Convert models with a single function call ``torch2trt``

* Easy to extend - Write your own layer converter in Python and register it with ``@tensorrt_converter``

If you find an issue, please [let us know](../..//issues)!

> Please note, this converter has limited coverage of TensorRT / PyTorch.  We created it primarily
> to easily optimize the models used in the [JetBot](https://github.com/NVIDIA-AI-IOT/jetbot) project.  If you find the converter helpful with other models, please [let us know](../..//issues).

### Setup

```bash
python setup.py install --user
```

### Usage

Below are some usage examples, for more check out the [notebooks](notebooks).

#### Convert

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

#### Execute

We can execute returned ``TRTModule`` just like the original PyTorch model

```python
y = model(x)
y_trt = model_trt(x)

# check the output against 
print(torch.max(torch.abs(y - y_trt)))
```

#### Save and load

We can save the model as a ``state_dict``.

```python
torch.save(model_trt.state_dict(), 'alexnet_trt.pth')
```

We can load the saved model into a ``TRTModule``

```python
from torch2trt import TRTModule

model_trt = TRTModule()

model_trt.load_state_dict(torch.load('alexnet_trt.pth'))
```

### Models

We tested the converter against these models using the [test.sh](test.sh) script.  You can generate the results by calling

```bash
bash test.sh TEST_OUTPUT.md
```

Below shows the execution time in FPS of each model.  You can find the raw output in the [benchmarks](benchmarks) folder.

> Even though we report the results below in FPS, they are actually a measure of the model's *latency*.  We use batch size 1 and perform [synchronization](https://github.com/NVIDIA-AI-IOT-private/torch2trt/blob/master/torch2trt/test.py#L61) after every model execution call.  Higher *throughput* may be possible by asynchronous execution and increased batch size.


| Model | Nano (PyTorch) | Nano (TensorRT) | Xavier (PyTorch) | Xavier (TensorRT) |
|-------|:--------------:|:---------------:|:----------------:|:-----------------:|
| alexnet |  |  | 250 | 580 |
| squeezenet1_0 |  |  | 130 | 890 |
| squeezenet1_1 |  |  | 132 | 1390 |
| resnet18 |  |  | 140 | 712 |
| resnet34 |  |  | 79.2 | 393 |
| resnet50 |  |  | 55.5 | 312 |
| resnet101 |  |  | 28.5 | 170 |
| resnet152 |  |  | 18.9 | 121 |
| densenet121 |  |  | 23.0 | 168 |
| densenet169 |  |  | 16.3 | 118 |
| densenet201 |  |  | 13.3 | 90.9 |
| densenet161 |  |  | 17.2 | 82.4 |
| vgg11 |  |  | 85.2 | 201 |
| vgg13 |  |  | 71.9 | 166 |
| vgg16 |  |  | 61.7 | 139 |
| vgg19 |  |  | 54.1 | 121 |
| vgg11_bn |  |  | 81.8 | 201 |
| vgg13_bn |  |  | 68.0 | 166 |
| vgg16_bn |  |  | 58.5 | 140 |
| vgg19_bn |  |  | 51.4 | 121 |


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
restricting to a fixed input size, we can expect similar memory usage and runtime. 
