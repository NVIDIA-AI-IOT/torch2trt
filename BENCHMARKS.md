# Benchmarks

This page contains various benchmark results on different platforms using [this script](torch2trt/test.sh).  Currently, all benchmarks target batch size 1.

## Jetson Nano

## Jetson Xavier

| Name | Data Type | Input Shapes | torch2trt kwargs | Max Error | FPS (PyTorch) | FPS (TensorRT) |
|------|-----------|--------------|------------------|-----------|---------------|----------------|
| torchvision.models.alexnet.alexnet | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 3.05E-05 | 354 | 560 |
| torchvision.models.squeezenet.squeezenet1_0 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 1.56E-02 | 98.5 | 1.19e+03 |
| torchvision.models.squeezenet.squeezenet1_1 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 7.32E-04 | 103 | 1.5e+03 |
| torchvision.models.resnet.resnet18 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 2.93E-03 | 104 | 1.17e+03 |
| torchvision.models.resnet.resnet34 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 1.25E-01 | 57.6 | 516 |
| torchvision.models.resnet.resnet50 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 9.38E-02 | 42.1 | 358 |
| torchvision.models.resnet.resnet101 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 0.00E+00 | 24 | 185 |
| torchvision.models.resnet.resnet152 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 0.00E+00 | 17.7 | 127 |
| torchvision.models.densenet.densenet121 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 3.66E-03 | 20.3 | 132 |
| torchvision.models.densenet.densenet169 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 2.93E-03 | 15.1 | 120 |
| torchvision.models.densenet.densenet201 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 5.37E-03 | 12.8 | 93.4 |
| torchvision.models.densenet.densenet161 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 5.13E-03 | 16.2 | 85.3 |
| torchvision.models.vgg.vgg11 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 6.10E-04 | 118 | 183 |
| torchvision.models.vgg.vgg13 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 7.78E-04 | 93.8 | 161 |
| torchvision.models.vgg.vgg16 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 8.77E-04 | 76.2 | 138 |
| torchvision.models.vgg.vgg19 | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 9.92E-04 | 63.9 | 123 |
| torchvision.models.vgg.vgg11_bn | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 6.41E-04 | 109 | 190 |
| torchvision.models.vgg.vgg13_bn | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 6.62E-04 | 86.3 | 163 |
| torchvision.models.vgg.vgg16_bn | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 1.46E-03 | 70.3 | 142 |
| torchvision.models.vgg.vgg19_bn | float16 | [(1, 3, 224, 224)] | {'fp16_mode': True} | 8.16E-04 | 59.4 | 128 |
