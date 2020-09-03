# Basic Usage

This page demonstrates basic torch2trt usage.

## Conversion

You can easily convert a PyTorch module by calling ``torch2trt`` passing example data as input, for example to convert ``alexnet`` we call

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

!!! note

    Currently with torch2trt, once the model is converted, you must use the same input shapes during
    execution.  The exception is
    the batch size, which can vary up to the value specified by the ``max_batch_size`` parameter.
    
## Executution

We can execute the returned ``TRTModule`` just like the original PyTorch model.  Here we
execute the model and print the maximum absolute error.

```python
y = model(x)
y_trt = model_trt(x)

# check the output against PyTorch
print(torch.max(torch.abs(y - y_trt)))
```

## Saving and loading

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
