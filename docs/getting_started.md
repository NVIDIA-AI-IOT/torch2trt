# Getting Started

Follow these steps to get started using torch2trt.

<<<<<<< HEAD
=======
## Installation

>>>>>>> 4933faf05cd9e5d9aef2b23e7fe692bfda5fbc06
!!! note

    torch2trt depends on the TensorRT Python API.  On Jetson, this is included with the latest JetPack.  For desktop, please follow the [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html).  You may also try installing torch2trt inside one of the NGC PyTorch docker containers for [Desktop](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) or [Jetson](https://ngc.nvidia.com/catalog/containers/nvidia:l4t-pytorch).

<<<<<<< HEAD
### Install Without plugins
=======
### Option 1 - Without plugins
>>>>>>> 4933faf05cd9e5d9aef2b23e7fe692bfda5fbc06

To install without compiling plugins, call the following

```bash
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python setup.py install
```

<<<<<<< HEAD
### Install With plugins
=======
### Option 2 - With plugins (experimental)
>>>>>>> 4933faf05cd9e5d9aef2b23e7fe692bfda5fbc06

To install with plugins to support some operations in PyTorch that are not natviely supported with TensorRT, call the following

!!! note
    
    Please note, this currently only includes the interpolate plugin.  This plugin requires PyTorch 1.3+ for serialization.  

```bash
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
sudo python setup.py install --plugins
```

<<<<<<< HEAD
=======
## Basic Usage

Below are some usage examples, for more check out the [usage](usage) guide.

### Convert

```python
import torch
from torch2trt import torch2trt
from torchvision.models.alexnet import alexnet

# create some regular pytorch model...
model = alexnet(pretrained=True).eval().cuda()

# create example data
x = torch.ones((1, 3, 224, 224)).cuda()

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x])
```

### Execute

We can execute the returned ``TRTModule`` just like the original PyTorch model

```python
y = model(x)
y_trt = model_trt(x)

# check the output against PyTorch
print(torch.max(torch.abs(y - y_trt)))
```

### Save and load

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
>>>>>>> 4933faf05cd9e5d9aef2b23e7fe692bfda5fbc06
