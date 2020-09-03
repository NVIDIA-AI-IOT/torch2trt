# Reduced Precision

For certain platforms, reduced precision can result in substantial improvements in throughput,
often with little impact on model accuracy.

# Support Matrix

Below is a table of layer precision support for various NVIDIA platforms.

| Platform | FP16 | INT8 |
|----------|------|------|
| Jetson Nano | ![X](../images/check.svg) |  |
| Jetson TX2 | ![X](../images/check.svg)  | ![X](../images/check.svg) |
| Jetson Xavier NX | ![X](../images/check.svg) | ![X](../images/check.svg) |
| Jetson AGX Xavier | ![X](../images/check.svg)  | ![X](../images/check.svg)  |

!!! note

    If the platform you're using is missing from this table or you spot anything incorrect
    please [let us know](https://github.com/NVIDIA-AI-IOT/torch2trt).
    
## FP16 Precision

To enable support for fp16 precision with TensorRT, torch2trt exposes the ``fp16_mode`` parameter.
Converting a model with ``fp16_mode=True`` allows the TensorRT optimizer to select layers with fp16
precision.


```python
model_trt = torch2trt(model, [data], fp16_mode=True)
```

!!! note

    When ``fp16_mode=True``, this does not necessarily mean that TensorRT will select FP16 layers.
    The optimizer attempts to automatically select tactics which result in the best performance.
    
## INT8 Precision

torch2trt also supports int8 precision with TensorRT with the ``int8_mode`` parameter.  Unlike fp16 and fp32 precision, switching
to in8 precision often requires calibration to avoid a significant drop in accuracy.  

### Input Data Calibration

By default
torch2trt will calibrate using the input data provided.  For example, if you wanted
to calibrate on a set of 64 random normal images you could do.

```python
data = torch.randn(64, 3, 224, 224).cuda().eval()

model_trt = torch2trt(model, [data], int8_mode=True)
```

### Dataset Calibration

In many instances, you may want to calibrate on more data than fits in memory.  For this reason,
torch2trt exposes the ``int8_calibration_dataset`` parameter.  This parameter takes an input
dataset that is used for calibration.  If this parameter is specified, the input data is 
ignored during calibration.  You create an input dataset by defining
a class which implements the ``__len__`` and ``__getitem__`` methods.  

* The ``__len__`` method should return the number of calibration samples
* The ``__getitem__`` method must return a single calibration sample.  This is a list of input tensors to the model.  Each tensor should match the shape
you provide to the ``inputs`` parameter when calling ``torch2trt``.

For example, say you trained an image classification network using the PyTorch [``ImageFolder``](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder) dataset.
You could wrap this dataset for calibration, by defining a new dataset which returns only the images without labels in list format.

```python
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Normalize, Resize


class ImageFolderCalibDataset():
    
    def __init__(self, root):
        self.dataset = ImageFolder(
            root=root, 
            transform=Compose([
                Resize((224, 224)),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        )
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        image = image[None, ...]  # add batch dimension
        return [image]
```

You would then provide this calibration dataset to torch2trt as follows

```python
dataset = ImageFolderCalibDataset('images')

model_trt = torch2trt(model, [data], int8_calib_dataset=dataset)
```

### Calibration Algorithm

To override the default calibration algorithm that torch2trt uses, you can set the ``int8_calib_algoirthm``
to the [``tensorrt.CalibrationAlgoType``](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Int8/Calibrator.html#iint8calibrator)
that you wish to use.  For example, to use the minmax calibration algorithm you would do

```python
import tensorrt as trt

model_trt = torch2trt(model, [data], int8_mode=True, int8_calib_algorithm=trt.CalibrationAlgoType.MINMAX_CALIBRATION)
```

### Calibration Batch Size

During calibration, torch2trt pulls data in batches for the TensorRT calibrator.  In some instances
[developers have found](https://github.com/NVIDIA-AI-IOT/torch2trt/pull/398) that the calibration batch size can impact the calibrated model accuracy.  To set the calibration batch size, you can set the ``int8_calib_batch_size``
parameter.  For example, to use a calibration batch size of 32 you could do

```python
model_trt = torch2trt(model, [data], int8_mode=True, int8_calib_batch_size=32)
```

## Binding Data Types

The data type of input and output bindings in TensorRT are determined by the original
PyTorch module input and output data types.
This does not directly impact whether the TensorRT optimizer will internally use fp16 or int8 precision.

For example, to create a model with fp32 precision bindings, you would do the following

```python
model = model.float()
data = data.float()

model_trt = torch2trt(model, [data], fp16_mode=True)
```

In this instance, the optimizer may choose to use fp16 precision layers internally, but the
input and output data types are fp32.  To use fp16 precision input and output bindings you would do

```python
model = model.half()
data = data.half()

model_trt = torch2trt(model, [data], fp16_mode=True)
```

Now, the input and output bindings of the model are half precision, and internally the optimizer may
choose to select fp16 layers as well.
