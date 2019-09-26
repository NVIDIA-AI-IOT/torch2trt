# torch2trt.plugin

This package defines the ``Plugin`` class, which is used to easily define a specialized TensorRT plugin that can use the Torch C++ library. 
The goal is to make it as easy as possible to define new TensorRT plugins for developers who may be experienced with defining PyTorch C++/CUDA extensions.

## Usage

## Step 1 - Define plugin

First, you define a plugin in Python, passing the following parameters

* name (str) - The unique plugin name
* forward (str) -  The C++ execution code.  Called on TensorRT's ``enqeue`` method
* setup (str, optional) - The C++ initialization code.  Called after the plugin's protobuf is deserialized.
* members (str, optional) - The C++ code used to define custom class members
* proto (str, optional) - Fields to be added to the plugin's protobuf definition
* extra_src (str, optional) - The C++/CUDA code.  This is injected outside of the Plugin class definition. Useful to define for example a custom CUDA kernel.
* extra_include_dirs (list of str, optional) - Extra include directories passed to compiler
* extra_library_dirs (list of str, optional) - Extra library dirs passed to compiler
* extra_libraries (list of str, optional) - Extra libraries to link against
* cflags (str, optional) - Extra C++ flags passed to compiler
* directory (str, optional) - The output directory where plugin files and binaries will be generated

For example, for the interpolate plugin, we define it like this.

```python
from torch2trt.plugin import Plugin


interpolate_plugin = Plugin(
    'interpolate',
    forward=
    """
    auto input = input_tensors[0];
    auto output = output_tensors[0];
    if (msg.mode() == "bilinear") {
      at::upsample_bilinear2d_out(output, input, {msg.size(0), msg.size(1)}, msg.align_corners());
    } else if (msg.mode() == "nearest") {
      at::upsample_nearest2d_out(output, input, {msg.size(0), msg.size(1)});
    } else if (msg.mode() == "area") {
      at::adaptive_avg_pool2d_out(output, input, {msg.size(0), msg.size(1)});
    } else if (msg.mode() == "bicubic") {
      at::upsample_bicubic2d_out(output, input, {msg.size(0), msg.size(1)}, msg.align_corners());
    }
    """,
    proto=
    """
    repeated int64 size = 100;
    string mode = 101;
    bool align_corners = 102;
    """
)
```

We're using torch's underlying ``ATen`` library, so we don't need to define our own CUDA kernels.  If we did, we would do so by adding the ``extra_src`` parameter.


## Step 2 - Define converter

Second we call ``interpolate_plugin.add_to_network`` from the converter to add the plugin to the TensorRT network.  This should be done from within a converter,
We pass the following arguments

* network (tensorrt.INetworkDefinition) - The TensorRT network
* inputs (list of torch.Tensors) - The input torch Tensors (we expect ``input._trt`` is set where necessary)
* outputs (list of torch.Tensors) - The output torch Tensors (this method will set ``output._trt`` where relevant)
* protobuf kwargs (keyword arguments) - These are arguments passed to the protobuf constructor.  Should correspond those defined in ``proto`` passed to Plugin constructor

For example, for the ``interpolate`` plugin, we do

```python
from torch2trt.torch2trt import tensorrt_converter

@tensorrt_converter('torch.nn.functional.interpolate')
def convert_interpolate(ctx):
    input = ctx.method_args[0]
    output = ctx.method_return

    try:
        mode = ctx.method_kwargs['mode']
    except KeyError:
        mode = 'nearest'

    try:
        align_corners = ctx.method_kwargs['align_corners']
    except KeyError:
        align_corners = False

    # currently only works for NCHW
    size = list(output.shape[2:])

    interpolate_plugin.add_to_network(ctx.network, [input], [output], size=size, mode=mode, align_corners=align_corners)
```

## Step 3 - Convert!

We define a module that performs just interpolation

```python3
class Interpolate(torch.nn.Module):
    def __init__(self, size, mode, align_corners):
        super(Interpolate, self).__init__()
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, self.size, mode=self.mode, align_corners=self.align_corners)  
```

Next, we instantiate the model

```python
model = Interpolate((112, 112), 'nearest', False)
```

And call ``torch2trt``

```python
data = torch.zeros((1, 3, 56, 56)).cuda()

model_trt = torch2trt(model, [data])
```

# Extra information

## Pre-build / rebuild plugin

The ``Plugin`` class defines two methods ``build`` and ``rebuild`` which may be used to generate the plugin code / binaries.  
The only difference is that ``build`` will load the plugin if it already exists, while ``rebuild`` will force a new build of the plugin.  By default,
``build`` is called JIT when an attempt to add the plugin to the network is made.  If you make changes to the plugin, you should manually call ``rebuild``.

To force compilation we would call

```python
interpolate_plugin.rebuild()
```

This will perform the following steps

1. Generate three source files 
  1. ``~/.torch2trt/interpolate.cu`` - The CUDA source file defining the plugin C++ / CUDA code.  This is generated from a template with the parameters passed to ``Plugin`` filled in.
  2. ``~/.torch2trt/interpolate.proto`` - The protobuf source file defining the plugin's standalone serialization protobuf format.  
  3. ``~/.torch2trt/build.ninja`` - The ninja build file used to create the plugin.  This defines rules to
    1. Generate Python protobuf source
    2. Generate C++ protobuf source
    3. Compile C++/CUDA source code

2. Call ``ninja`` from ``~/.torch2trt`` (or the directory passed to ``Plugin`` constructor)

## Register plugin

TODO: Registering plugins is necessary to know which shared objects must be loaded when reading a serialized network.

## Dynamic shapes

To make it easier to define plugins, we assume the input / output shape tensors are constant, and we infer them from the PyTorch tensors provided to ``Plugin.add_to_network``.  For this reason, dynamic shapes are not supported.

## Data types

For simplicity, we assume all plugins implement support for ``half`` and ``float`` data types.  Perhaps this can be parameterized in future.  But for now, any custom code / kernels must handle both of these data types.  This is done implicitly for most Torch C++ API calls.

## Naming plugins

All plugins are defined under the ``torch2trt`` namespace.  Because of this, plugins must implement a globally unique name.  
