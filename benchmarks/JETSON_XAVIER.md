| Name | Data Type | Input Shapes | torch2trt kwargs | Max Error | Throughput (PyTorch) | Throughput (TensorRT) | Latency (PyTorch) | Latency (TensorRT) |
|------|-----------|--------------|------------------|-----------|----------------------|-----------------------|-------------------|--------------------|
| Name | Data Type | Input Shapes | torch2trt kwargs | Max Error | Throughput (PyTorch) | Throughput (TensorRT) | Latency (PyTorch) | Latency (TensorRT) |
|------|-----------|--------------|------------------|-----------|----------------------|-----------------------|-------------------|--------------------|
| Name | Data Type | Input Shapes | torch2trt kwargs | Max Error | Throughput (PyTorch) | Throughput (TensorRT) | Latency (PyTorch) | Latency (TensorRT) |
|------|-----------|--------------|------------------|-----------|----------------------|-----------------------|-------------------|--------------------|
| torchvision.models.alexnet.alexnet | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 2.29E-05 | 250 | 580 | 4.75 | 1.93 |
| torchvision.models.squeezenet.squeezenet1_0 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 3.03E-02 | 130 | 890 | 7.31 | 1.37 |
| torchvision.models.squeezenet.squeezenet1_1 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 1.95E-03 | 132 | 1.39e+03 | 7.41 | 0.951 |
| torchvision.models.resnet.resnet18 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 5.37E-03 | 140 | 712 | 7.1 | 1.64 |
| torchvision.models.resnet.resnet34 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 1.09E-01 | 79.2 | 393 | 12.6 | 2.79 |
| torchvision.models.resnet.resnet50 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 9.57E-02 | 55.5 | 312 | 17.6 | 3.48 |
| torchvision.models.resnet.resnet101 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 0.00E+00 | 28.5 | 170 | 34.8 | 6.22 |
| torchvision.models.resnet.resnet152 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 0.00E+00 | 18.9 | 121 | 52.1 | 8.58 |
| torchvision.models.densenet.densenet121 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 1.95E-03 | 23 | 168 | 43.3 | 6.37 |
| torchvision.models.densenet.densenet169 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 4.39E-03 | 16.3 | 118 | 60.2 | 8.83 |
| torchvision.models.densenet.densenet201 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 4.03E-03 | 13.3 | 90.9 | 72.7 | 11.4 |
| torchvision.models.densenet.densenet161 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 3.91E-03 | 17.2 | 82.4 | 56.3 | 12.6 |
| torchvision.models.vgg.vgg11 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 7.32E-04 | 85.2 | 201 | 12 | 5.16 |
| torchvision.models.vgg.vgg13 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 8.24E-04 | 71.9 | 166 | 14.2 | 6.27 |
| torchvision.models.vgg.vgg16 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 2.01E-03 | 61.7 | 139 | 16.6 | 7.46 |
| torchvision.models.vgg.vgg19 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 1.80E-03 | 54.1 | 121 | 18.8 | 8.52 |
| torchvision.models.vgg.vgg11_bn | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 5.80E-04 | 81.8 | 201 | 12.5 | 5.16 |
| torchvision.models.vgg.vgg13_bn | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 6.03E-04 | 68 | 166 | 15 | 6.27 |
| torchvision.models.vgg.vgg16_bn | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 1.45E-03 | 58.5 | 140 | 17.4 | 7.41 |
| torchvision.models.vgg.vgg19_bn | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 6.64E-04 | 51.4 | 121 | 19.8 | 8.52 |
