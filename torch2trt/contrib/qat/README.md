## Quantization Aware Training

This contrib folder provides layers and converters for Quantization Aware Training to convert layers into INT8. 

### Supported Layers

- Conv2d
- Conv2d + fused BN
- ReLU
 
### Future Support for Layers

 -Pooling layers    
 -Linear layer    

### Supported Quantization Techniques

- per tensor quantization
- symmetric quantization

### Future Support for Quantization Techniques

- per channel quantization
- asymmetric quantization

### Working example

Please see `examples/quantization_aware_training`
