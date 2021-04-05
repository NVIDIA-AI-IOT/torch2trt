## Layers

- Every layer has two implementations (Training and Inference). This is required as the quantized aware layers quantize the weights / activation in the forward pass. 
- If we try to convert the layers into TRT engine (w quantization happening in the forward pass), then a lot of unwanted ops will be presented in the final TRT engine as Torch2TRT will convert all the ops into their TRT equivalent layers. .   
- Therefore, an inference version of the layer is created so that only the learned parameters (zero point / scale) are carried with the layer for convertng the layer into a TRT engine. 

## Quantization Type

Currently. TRT7 only supports per tensor symmetric quantization. Support for other techniques of quantization (such as per channel , asymmetric etc) will be supported once the newer versions of TensorRT support them.

## Working example

Please refer to `examples/quantization_aware_training/` for a working example. 
