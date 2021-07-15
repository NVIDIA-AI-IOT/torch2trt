# Changes

## [Master]

## [0.3.0] - 07/15/2021

- Added converter for ``torch.nn.functional.adaptive_avg_pool3d``
- Added converter for ``torch.nn.functional.adaptive_max_pool3d``
- Added converter for ``torch.maxpool3d`` and ``torch.nn.functional.max_pool3d``
- Added Quantization Aware Training (QAT) workflow to contrib
- Added converter for ``torch.roll``
- Added converter for ``torch.nn.functional.layer_norm``
- Added converter for ``torch.nn.functional.gelu``
- Added converter for ``torch.nn.functional.linear``
- Added converter for ``torch.nn.functional.silu``

## [0.2.0] - 03/02/2021

- Added converter for ``torch.Tensor.flatten``
- Added converter for ``torch.nn.functional.conv2d`` and ``torch.nn.functional.conv3d``

### Added 

- Added converter for ``torch.Tensor.expand``
- Added support for custom converters for methods defined outside of ``torch`` module
- Added names for TensorRT layers
- Added GroupNorm plugin which internally uses PyTorch aten::group_norm
- Replaced Tensor.ndim references with len(tensor.shape) to support older pytorch versions
- Added reduced precision documentation page
- Added converters for ``floordiv``, ``mod``, ``ne``, and ``torch.tensor`` operations
- Extended ``relu`` converter to support ``Tensor.relu`` operation
- Extended ``sigmoid`` converter to support ``Tensor.sigmoid`` operation
