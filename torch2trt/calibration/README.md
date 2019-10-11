# Int8 inference

This code will support Int8 inference with TensorRT, but still experimental.  


## Usage

You have to implement data loader for Int8 calibration, which needs dataset.  
It is recommended to load at least 100 images.

### In practice

```python
import torch.utils.data as data

class dataLoader(data.Dataset):
    def __init__(self, data_path, batch_size=1, preprocess=100, n=100):
        # you need to set data_path which include images
        # you must implement reset and next_batch function
        self.batch_size = batch_size
        self.calibration_data = np.zeros((batch_size, 3, 224, 224), dtype=np.float32)
        # ...

    def reset(self):
        self.batch = 0

    def next_batch(self):
        # you must fill self.calibration_data as much batch size you set.
        # ...
        return np.ascontiguousarray(self.calibration_data, dtype=np.float32)

# create some regular pytorch model
model = alexnet(pretrained=True).eval().cuda()

# create example data
x = torch.zeros((1, 3, 224, 224)).cuda()

# convert to TensorRT feeding dummy data as input, with Int8 inference mode
model_trt = torch2trt(model, [x], int8_mode=True, int8_stream=dataLoader("data/path", batch_size=5, preprocess=augmentation))
# Note: batch size in here is different from inference, it is for calibration
```

For more details, you can refer ImageBatchStream in calibrator.py or dataLoader in ../module_test.py

### Exucute

```python
y = model(x)
y_trt = model_trt(x)

# check the output against Pytorch
print(torch.max(torch.abs(y - y_trt)))

# In Int8 inference, results can be varying depending on input distribution.
# If you are doing classification, softmax will make output smooth so that results will be similar.
```

The other things are same with ../../README.md


### Samples

<img src="https://github.com/jtlee90/res-model-torch2trt/blob/master/torch2trt/calibration/samples/Float_inference", width=290>
<img src="https://github.com/jtlee90/res-model-torch2trt/blob/master/torch2trt/calibration/samples/Int8_inference", width=290>
