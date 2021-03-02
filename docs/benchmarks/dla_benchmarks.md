# DLA Benchmarks

This page presents the results of profiling resnet18 on Jetson Xavier NX under varying DLA configurations with torch2trt.  Please see ``torch2trt/scripts/benchmark_dla.py`` 
for implementation details.

## Concurrent Model Execution

Here we profile the throughput of resnet18 running concurrently in separate processes.  Each section contains the FPS measured from each separate process running concurrently.

### INT8

#### 2GPU

| config | torch2trt_kwargs | tensorrt FPS |
|--------|------------------|--------------|
| resnet18_int8_gpu | {'int8_mode': True, 'default_device_type': DeviceType.GPU, 'max_batch_size': 1, 'dla_core': 0} | 331.0196110748139 |
| resnet18_int8_gpu | {'int8_mode': True, 'default_device_type': DeviceType.GPU, 'max_batch_size': 1, 'dla_core': 0} | 331.8150824541489 |

#### 1GPU - 1DLA

| config | torch2trt_kwargs | tensorrt FPS |
|--------|------------------|--------------|
| resnet18_int8_gpu | {'int8_mode': True, 'default_device_type': DeviceType.GPU, 'max_batch_size': 1, 'dla_core': 0} | 634.021612401935 |
| resnet18_int8_dla | {'int8_mode': True, 'default_device_type': DeviceType.DLA, 'max_batch_size': 1, 'dla_core': 0} | 358.6787695930769 |

#### 2DLA (separate cores)

| config | torch2trt_kwargs | tensorrt FPS |
|--------|------------------|--------------|
| resnet18_int8_dla | {'int8_mode': True, 'default_device_type': DeviceType.DLA, 'max_batch_size': 1, 'dla_core': 1} | 422.87780423155755 |
| resnet18_int8_dla | {'int8_mode': True, 'default_device_type': DeviceType.DLA, 'max_batch_size': 1, 'dla_core': 0} | 426.7473439918015 |

### FP16

#### 2GPU

| config | torch2trt_kwargs | tensorrt FPS |
|--------|------------------|--------------|
| resnet18_fp16_gpu | {'fp16_mode': True, 'default_device_type': DeviceType.GPU, 'max_batch_size': 1, 'dla_core': 0} | 187.75452731265406 |
| resnet18_fp16_gpu | {'fp16_mode': True, 'default_device_type': DeviceType.GPU, 'max_batch_size': 1, 'dla_core': 0} | 186.93638306636586 |

#### 1GPU - 1DLA

| config | torch2trt_kwargs | tensorrt FPS |
|--------|------------------|--------------|
| resnet18_fp16_gpu | {'fp16_mode': True, 'default_device_type': DeviceType.GPU, 'max_batch_size': 1, 'dla_core': 0} | 355.8585787831001 |
| resnet18_fp16_dla | {'fp16_mode': True, 'default_device_type': DeviceType.DLA, 'max_batch_size': 1, 'dla_core': 0} | 209.5668385056025 |

#### 2DLA (separate cores)

| config | torch2trt_kwargs | tensorrt FPS |
|--------|------------------|--------------|
| resnet18_fp16_dla | {'fp16_mode': True, 'default_device_type': DeviceType.DLA, 'max_batch_size': 1, 'dla_core': 1} | 252.11907214674176 |
| resnet18_fp16_dla | {'fp16_mode': True, 'default_device_type': DeviceType.DLA, 'max_batch_size': 1, 'dla_core': 0} | 251.6072665899547 |

## Heterogeneous Model

Here we vary the precision as well as which submodules of resnet18 run on DLA vs. GPU.

| config | torch2trt kwargs | torch FPS | tensorrt FPS |
|---|---|---|---|
| resnet18_fp16_gpu | {'fp16_mode': True, 'default_device_type': DeviceType.GPU, 'max_batch_size': 1} | 41.48863547383661 | 456.3341766145895 |
| resnet18_fp16_dla | {'fp16_mode': True, 'default_device_type': DeviceType.DLA, 'max_batch_size': 1} | 41.69413020799957 | 274.6236549157457 |
| resnet18_int8_gpu | {'int8_mode': True, 'default_device_type': DeviceType.GPU, 'max_batch_size': 1} | 42.16469766415027 | 793.2889028321634 |
| resnet18_int8_dla | {'int8_mode': True, 'default_device_type': DeviceType.DLA, 'max_batch_size': 1} | 42.02777756149902 | 514.262372651439 |
| resnet18_fp16_gpu_dla1 | {'fp16_mode': True, 'default_device_type': DeviceType.GPU, 'device_types': "{'layer1': 'DLA'}", 'max_batch_size': 1} | 42.7403441810802 | 394.57422247913195 |
| resnet18_fp16_gpu_dla12 | {'fp16_mode': True, 'default_device_type': DeviceType.GPU, 'device_types': "{'layer1': 'DLA', 'layer2': 'DLA'}", 'max_batch_size': 1} | 42.40382590738402 | 398.805262354432 |
| resnet18_fp16_gpu_dla123 | {'fp16_mode': True, 'default_device_type': DeviceType.GPU, 'device_types': "{'layer1': 'DLA', 'layer2': 'DLA', 'layer3': 'DLA'}", 'max_batch_size': 1} | 42.2997163615733 | 389.61254308033745 |
| resnet18_int8_gpu_dla1 | {'int8_mode': True, 'default_device_type': DeviceType.GPU, 'device_types': "{'layer1': 'DLA'}", 'max_batch_size': 1} | 42.623790888121285 | 476.278066006966 |
| resnet18_int8_gpu_dla12 | {'int8_mode': True, 'default_device_type': DeviceType.GPU, 'device_types': "{'layer1': 'DLA', 'layer2': 'DLA'}", 'max_batch_size': 1} | 42.320715371630605 | 496.7642812355515 |
| resnet18_int8_gpu_dla123 | {'int8_mode': True, 'default_device_type': DeviceType.GPU, 'device_types': "{'layer1': 'DLA', 'layer2': 'DLA', 'layer3': 'DLA'}", 'max_batch_size': 1} | 41.859373689253914 | 500.02851665716946 |