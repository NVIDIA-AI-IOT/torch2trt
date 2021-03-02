# DLA Usage

Some NVIDIA devices, like Jetson Xavier NX and Jetson Xavier AGX, come equipped with
one or more Deep Learning Accelerators (DLA).  These are dedicated hardware units which are capable of processing a variety of common neural network functions.  Interfacing with these devices is accomplished through TensorRT.  This page details how you
can utilize the DLA(s) on your device with torch2trt.

!!! note

    We've benchmarked resnet18 running on Jetson Xavier NX under varying configurations.  You can find the results [here](../benchmarks/dla_benchmarks.html).  This may help you understand if DLA is applicable for your use case.

## Enable DLA Globally

Enabling the DLA globally is done by passing ``default_device_type=trt.DeviceType.DLA`` to torch2trt.  If a layer is not supported on DLA, it will default to GPU.  Please note,
DLA only supports FP16 and INT8 precision, so you must also set either ``fp16_mode=True``
or ``int8_mode=True``.  To use DLA with FP16 precision you would call the following.

```python
import tensorrt as trt  # used to access trt.DeviceType enumeration
from torch2trt import torch2trt

model_trt = torch2trt(model, [data], fp16_mode=True, default_device_type=trt.DeviceType.DLA)
```

Similarily, you could easily use ``int8_mode`` instead by calling

```python
model_trt = torch2trt(model, [data], int8_mode=True, default_device_type=trt.DeviceType.DLA)
```

INT8 will require calibration to compute the dynamic ranges needed for quantization.  For more details on this, see the page on [reduced precision](/usage/reduced_precision.html)

## Specify DLA submodules

torch2trt allows you to explicitly control the granularity of which submodules should run
on DLA and which should run on GPU.  This is accomplished through the ``device_types`` argument.  This argument takes a dictionary mapping pytorch modules to TensorRT device types. 

For example, we can convert the resnet18 blocks layer1 and layer2 to run on DLA.

```python
model = resnet18(pretrained=True).cuda().eval()

device_types = {
    model.layer1: trt.DeviceType.DLA, 
    model.layer2: trt.DeviceType.DLA
}

model_trt = torch2trt(model, [data], fp16_mode=True, device_types=device_types)
```

If we wanted, we could override the first block in layer2 back to GPU, by specifying the device type for this child element.  This would look like the following,

```python
device_types = {
    model.layer1: trt.DeviceType.DLA, 
    model.layer2: trt.DeviceType.DLA,
    model.layer2[0]: trt.DeviceType.GPU
}

...
```

Device types will default to ``default_device_type`` if a parent module is not specified in ``device_types``.  By default, ``default_device_type`` is set to ``trt.DeviceType.GPU``.  If you wanted to enable DLA globally, and selectively set layers to run on GPU, you could do the following.

```python
model = resnet18(pretrained=True).cuda().eval()

device_types = {
    model.layer1: trt.DeviceType.GPU, 
    model.layer2: trt.DeviceType.GPU
}

model_trt = torch2trt(model, [data], default_device_typee=trt.DeviceType.DLA, fp16_mode=True, device_types=device_types)
```

## Specify DLA core

For devices with multiple DLA cores, you may set the ``dla_core`` attribute to control
which DLA core the TensorRT engine should run on.  This will apply globally to all 
layers using DLA inside the converted module.  For example, to use DLA core 1 we would call,

```python
model_trt = torch2trt(model, [data], dla_core=1, ...)
```

!!! info

    Currently, the TensorRT Python API doesn't support setting the DLA core at runtime.
    For this reason, models which are serialized / deserialized will default to DLA core
    0.  However, the model returned immediately from torch2trt will use the DLA set by the dla_core argument.  
