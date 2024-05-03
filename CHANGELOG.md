# Changes

## [master](https://github.com/NVIDIA-AI-IOT/torch2trt/tree/master)

- Added inference and conversion support for TensorRT 10
- Removed redundant converters, and merged converters for ND convolutions, pooling, etc.
- Migrated test cases to use PyTest
- Added unique axis names when using ONNX to support mis-matched dynamic axes (needed for whisper)

## [v0.5.0](https://github.com/NVIDIA-AI-IOT/torch2trt/tree/v0.5.0) - 05/3/2024

- Added tensor shape tracking to support dynamic shapes for flatten, squeeze, unsqueeze, view, reshape, interpolate, and getitem methods
- Added EasyOCR example
- Added the ``DatasetRecorder`` context manager, allowing to easily capture of module inputs in large pipeline for calibration and shape inference
- Added support for legacy max_batch_size using optimization profiles
- Added support for nested tuple, dict and list module inputs and outputs via. the ``Flattener`` class
- Added ability to accept dataset as ``inputs`` argument, and infer optimization profiles from the data
- Added Dataset, TensorBatchDataset, ListDataset, and FolderDatset classes
- Added support for dynamic shapes
  - Known limitation: Currently some converters (ie: View) may have unexpected behavior if their arguments are defined with dynamic Tensor shapes.

## [0.4.0](https://github.com/NVIDIA-AI-IOT/torch2trt/tree/v0.4.0) - 07/22/2022

- Added converter for ``torch.nn.functional.group_norm`` using native TensorRT layers
- Added converter for ``torch.nn.ReflectionPad2d`` using plugin layer
- Added torch2trt_plugins library
- Added support for Deep Learning Accelerator (DLA)
- Added support for explicit batch
- Added support for TensorRT 8

## [0.3.0](https://github.com/NVIDIA-AI-IOT/torch2trt/tree/v0.3.0) - 07/15/2021

- Added converter for ``torch.nn.functional.adaptive_avg_pool3d``
- Added converter for ``torch.nn.functional.adaptive_max_pool3d``
- Added converter for ``torch.maxpool3d`` and ``torch.nn.functional.max_pool3d``
- Added Quantization Aware Training (QAT) workflow to contrib
- Added converter for ``torch.roll``
- Added converter for ``torch.nn.functional.layer_norm``
- Added converter for ``torch.nn.functional.gelu``
- Added converter for ``torch.nn.functional.linear``
- Added converter for ``torch.nn.functional.silu``

## [0.2.0](https://github.com/NVIDIA-AI-IOT/torch2trt/tree/v0.2.0) - 03/02/2021

- Added converter for ``torch.Tensor.flatten``
- Added converter for ``torch.nn.functional.conv2d`` and ``torch.nn.functional.conv3d``
- Added converter for ``torch.Tensor.expand``
- Added support for custom converters for methods defined outside of ``torch`` module
- Added names for TensorRT layers
- Added GroupNorm plugin which internally uses PyTorch aten::group_norm
- Replaced Tensor.ndim references with len(tensor.shape) to support older pytorch versions
- Added reduced precision documentation page
- Added converters for ``floordiv``, ``mod``, ``ne``, and ``torch.tensor`` operations
- Extended ``relu`` converter to support ``Tensor.relu`` operation
- Extended ``sigmoid`` converter to support ``Tensor.sigmoid`` operation
