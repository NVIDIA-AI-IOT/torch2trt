# Jetson Xavier

| Name | Data Type | Input Shapes | torch2trt kwargs | Max Error | Throughput (PyTorch) | Throughput (TensorRT) | Latency (PyTorch) | Latency (TensorRT) |
|------|-----------|--------------|------------------|-----------|----------------------|-----------------------|-------------------|--------------------|
| torch2trt.tests.torchvision.classification.alexnet | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 7.63E-05 | 251 | 565 | 4.96 | 2.02 |
| torch2trt.tests.torchvision.classification.squeezenet1_0 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 9.77E-04 | 121 | 834 | 8.04 | 1.49 |
| torch2trt.tests.torchvision.classification.squeezenet1_1 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 9.77E-04 | 125 | 1.29e+03 | 8.01 | 1.02 |
| torch2trt.tests.torchvision.classification.resnet18 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 9.77E-03 | 136 | 722 | 7.33 | 1.64 |
| torch2trt.tests.torchvision.classification.resnet34 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 2.50E-01 | 77.8 | 396 | 12.9 | 2.79 |
| torch2trt.tests.torchvision.classification.resnet50 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 1.09E-01 | 55.8 | 326 | 17.9 | 3.37 |
| torch2trt.tests.torchvision.classification.resnet101 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 0.00E+00 | 28.3 | 175 | 35.1 | 6.04 |
| torch2trt.tests.torchvision.classification.resnet152 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 0.00E+00 | 18.8 | 122 | 53.2 | 8.57 |
| torch2trt.tests.torchvision.classification.densenet121 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 7.81E-03 | 20.9 | 76.6 | 47.5 | 13 |
| torch2trt.tests.torchvision.classification.densenet169 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 3.91E-03 | 14.8 | 41.7 | 66.7 | 23.7 |
| torch2trt.tests.torchvision.classification.densenet201 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 4.88E-03 | 12.6 | 30.2 | 79.1 | 33 |
| torch2trt.tests.torchvision.classification.densenet161 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 4.88E-03 | 16.1 | 43.7 | 62.1 | 23 |
| torch2trt.tests.torchvision.classification.vgg11 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 2.56E-03 | 84.8 | 201 | 12.1 | 5.24 |
| torch2trt.tests.torchvision.classification.vgg13 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 2.24E-03 | 71.1 | 165 | 14.3 | 6.34 |
| torch2trt.tests.torchvision.classification.vgg16 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 3.78E-03 | 61.5 | 139 | 16.5 | 7.46 |
| torch2trt.tests.torchvision.classification.vgg19 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 2.81E-03 | 54.1 | 120 | 18.7 | 8.61 |
| torch2trt.tests.torchvision.classification.vgg11_bn | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 2.20E-03 | 81.5 | 200 | 12.5 | 5.27 |
| torch2trt.tests.torchvision.classification.vgg13_bn | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 1.71E-03 | 67.5 | 165 | 15.1 | 6.33 |
| torch2trt.tests.torchvision.classification.vgg16_bn | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 2.87E-03 | 58.3 | 139 | 17.4 | 7.48 |
| torch2trt.tests.torchvision.classification.vgg19_bn | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 2.44E-03 | 51.4 | 120 | 19.7 | 8.61 |
| torch2trt.tests.torchvision.classification.mobilenet_v2 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 0.00E+00 | 64.8 | 723 | 15.4 | 1.67 |
| torch2trt.tests.torchvision.classification.shufflenet_v2_x0_5 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 1.53E-05 | 51.2 | 463 | 19.4 | 2.17 |
| torch2trt.tests.torchvision.classification.shufflenet_v2_x1_0 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 1.53E-05 | 49.4 | 419 | 20.4 | 2.43 |
| torch2trt.tests.torchvision.classification.shufflenet_v2_x1_5 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 1.53E-05 | 51.4 | 426 | 19.6 | 2.37 |
| torch2trt.tests.torchvision.classification.shufflenet_v2_x2_0 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 1.53E-05 | 48.2 | 419 | 20.8 | 2.48 |
| torch2trt.tests.torchvision.classification.mnasnet0_5 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 2.03E-06 | 67.8 | 883 | 14.9 | 1.4 |
| torch2trt.tests.torchvision.classification.mnasnet0_75 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 0.00E+00 | 67.6 | 751 | 14.8 | 1.6 |
| torch2trt.tests.torchvision.classification.mnasnet1_0 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 0.00E+00 | 65.7 | 667 | 15.2 | 1.77 |
| torch2trt.tests.torchvision.classification.mnasnet1_3 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 0.00E+00 | 67.4 | 573 | 15 | 2.02 |
