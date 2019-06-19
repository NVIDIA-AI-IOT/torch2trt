| Name | Data Type | Input Shapes | torch2trt kwargs | Max Error | Throughput (PyTorch) | Throughput (TensorRT) | Latency (PyTorch) | Latency (TensorRT) |
|------|-----------|--------------|------------------|-----------|----------------------|-----------------------|-------------------|--------------------|
| torchvision.models.alexnet.alexnet | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 2.29E-05 | 46.4 | 69.9 | 22.1 | 14.7 |
| torchvision.models.squeezenet.squeezenet1_0 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 1.20E-02 | 44 | 137 | 24.2 | 7.6 |
| torchvision.models.squeezenet.squeezenet1_1 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 9.77E-04 | 76.6 | 248 | 14 | 4.34 |
| torchvision.models.resnet.resnet18 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 5.86E-03 | 29.4 | 90.2 | 34.7 | 11.4 |
| torchvision.models.resnet.resnet34 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 1.56E-01 | 15.5 | 50.7 | 64.8 | 20.2 |
| torchvision.models.resnet.resnet50 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 6.45E-02 | 12.4 | 34.2 | 81.7 | 29.8 |
| torchvision.models.resnet.resnet101 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 1.01E+03 | 7.18 | 19.9 | 141 | 51.1 |
| torchvision.models.resnet.resnet152 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 0.00E+00 | 4.96 | 14.1 | 204 | 72.3 |
| torchvision.models.densenet.densenet121 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 3.42E-03 | 11.5 | 41.9 | 84.5 | 24.8 |
| torchvision.models.densenet.densenet169 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 5.86E-03 | 8.25 | 33.2 | 118 | 31.2 |
| torchvision.models.densenet.densenet201 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 3.42E-03 | 6.84 | 25.4 | 141 | 40.8 |
| torchvision.models.densenet.densenet161 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 4.15E-03 | 4.71 | 15.6 | 247 | 65.8 |
| torchvision.models.vgg.vgg11 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 3.51E-04 | 8.9 | 18.3 | 114 | 55.1 |
| torchvision.models.vgg.vgg13 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 3.07E-04 | 6.53 | 14.7 | 156 | 68.7 |
| torchvision.models.vgg.vgg16 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 4.58E-04 | 5.09 | 11.9 | 201 | 85.1 |
| torchvision.models.vgg.vgg11_bn | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 3.81E-04 | 8.74 | 18.4 | 117 | 54.8 |
| torchvision.models.vgg.vgg13_bn | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 5.19E-04 | 6.31 | 14.8 | 162 | 68.5 |
| torchvision.models.vgg.vgg16_bn | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 9.77E-04 | 4.96 | 12 | 207 | 84.3 |
