This module includes code related to PyTorch -> TensorRT conversion.
Most notably it includes the ``TRTModule`` class, which executes a TensorRT
engine like a PyTorch module.  It also includes the ``torch2trt``
function, which converts a PyTorch module to TensorRT using the TensorRT
python API.  The converter works by attaching PyTorch to TensorRT 
conversion functions pytorch functions that are supported by TensorRT. The
list of conversion function mappings are contained in the ``CONVERTERS`` 
global variable.  Pleast note, the list of converters is NOT comprehensive.
Converters and plugins will be added as needed by various models.

### Convert a PyTorch module to TensorRT

```python
from torch2trt import torch2trt
from torchvision.models.alexnet import alexnet

model = alexnet(pretrained=True).eval().cuda()

x = torch.ones((1, 3, 224, 224)).cuda()

# help(create_tensorrt_module) for more information
model_trt = torch2trt(model, [x])

y = model(x)
y_trt = model_trt(x)

print(torch.max(torch.abs(y - y_trt)))
```

### Register or override a converter

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
