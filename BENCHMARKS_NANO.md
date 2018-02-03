| Name | Data Type | Input Shapes | torch2trt kwargs | Max Error | FPS (PyTorch) | FPS (TensorRT) |
|------|-----------|--------------|------------------|-----------|---------------|----------------|
| torchvision.models.alexnet.alexnet | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 1.91E-05 | 45.3 | 67.5 |
| torchvision.models.squeezenet.squeezenet1_0 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 1.24E-02 | 40.5 | 130 |
| torchvision.models.squeezenet.squeezenet1_1 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 1.46E-03 | 69.1 | 229 |
| torchvision.models.resnet.resnet18 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 2.93E-03 | 28.6 | 87.6 |
| torchvision.models.resnet.resnet34 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 1.56E-01 | 15.5 | 49.6 |
| torchvision.models.resnet.resnet50 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 3.91E-02 | 11.3 | 33.4 |
| torchvision.models.resnet.resnet101 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 0.00E+00 | 7.05 | 19.7 |
| torchvision.models.resnet.resnet152 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 0.00E+00 | 4.74 | 13.9 |
| torchvision.models.densenet.densenet121 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 2.38E-03 | 11.1 | 40.3 |
| torchvision.models.densenet.densenet169 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 2.93E-03 | 8.13 | 31.9 |
| torchvision.models.densenet.densenet201 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 2.44E-03 | 6.84 | 24.5 |
| torchvision.models.densenet.densenet161 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 4.64E-03 | 4.01 | 15.2 |
| torchvision.models.vgg.vgg11 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 3.81E-04 | 8.79 | 18 |
| torchvision.models.vgg.vgg13 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 3.36E-04 | 6.4 | 14.4 |
| torchvision.models.vgg.vgg16 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 3.83E-04 | 4.96 | 11.7 |
| torchvision.models.vgg.vgg11_bn | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 3.66E-04 | 8.46 | 18.2 |
| torchvision.models.vgg.vgg13_bn | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 4.43E-04 | 6.16 | 14.5 |
| torchvision.models.vgg.vgg16_bn | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 2.37E-04 | 4.83 | 11.8 |
