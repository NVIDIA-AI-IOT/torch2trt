# torch2trt - A PyTorch -> TensorRT Converter

This is PyTorch to TensorRT converter which utilizes the 
TensorRT Python API.  The goals of the converter are

* Easy to use - Convert models with a single function call ``torch2trt``
* Easy to extend - Write your own layer converter in Python and register it with ``@tensorrt_converter``

If you find an issue or write your own layer converter, please [let us know](../..//issues)!

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
| squeezenet1_1 | 16ms | 5.5ms |
| resnet18 |  | 13ms |
| resnet34 |  | 23ms |
| resnet50 | 77ms | 38ms |
| resnet101 | 135ms | 62ms |
| resnet152 | 200ms | 93ms |
| densenet121 | 83ms | 46ms |
| densenet169 |  |  |
| densenet201 |  |  |
| densenet161 |  |  |
| vgg11 |  |  |
| vgg13 |  |  |
| vgg16 |  |  |
| vgg19 |  |  |
| vgg11_bn |  |  |
| vgg13_bn |  |  |
| vgg16_bn |  |  |
| vgg19_bn |  |  |


### Add (or override) a converter

Here we show how to add an example converter using the TensorRT
python API.

```python
import tensorrt as trt
from torch2trt import tensorrt_converter

@tensorrt_converter('torch.nn.ReLU.forward')
def convert_ReLU(ctx):
    input_tensor = ctx.method_args[1]
    output_tensor = ctx.method_return
    trt_input = ctx.trt_tensors[input_tensor.__hash__()]
    layer = ctx.network.add_activation(input=trt_input, type=trt.ActivationType.RELU)  
    ctx.trt_tensors[output_tensor.__hash__()] = layer.get_output(0)
```

The converter takes one argument, a ``ConversionContext``, which will contain
the following

* network: The TensorRT network that is being constructed.
* method_args: Positional arguments that were passed to the specified Torch function.
* method_kwargs: Keyword arguments that were passed to the specified Torch function.
* method_return: The value returned by the specified Torch function.
* trt_tensors: A dictionary mapping Torch tensors (by hash value) to TensorRT tensors.  The
  converter must the set values for any output Tensors.  Otherwise, if a later function uses
  the Torch tensor, and there is not an associated TensorRT tensor in the map, results 
  may be unexpected.

Please see the ``torch2trt.py`` module for more examples.
