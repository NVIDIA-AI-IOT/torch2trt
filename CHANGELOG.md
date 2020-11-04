# Changes

## [Master]

### Added 

- Added GroupNorm plugin which internally uses PyTorch aten::group_norm
- Replaced Tensor.ndim references with len(tensor.shape) to support older pytorch versions
- Added reduced precision documentation page
