## IQuant Layers (Inference)

Once the training is done using nvidia qat library, we just need `amax` stats from the training run as trt only supports scale quantization with zero point = 0

Therefore, these layers are created so that `amax` can be loaded from the checkpoint successfully at inference time. 
