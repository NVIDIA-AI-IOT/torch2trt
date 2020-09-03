# See Also

!!! note

    The state of these converters may change over time.  We provide this information here with the hope that it will help shed light on the landscape of tools available for optimizing PyTorch models with TensorRT.
    If you find this information helpful or outdated / misleading, please let us know.
    
In addition to torch2trt, there are other workflows for optimizing your PyTorch model with TensorRT.

The other converters we are aware of are

* [ONNX to TensorRT](https://github.com/onnx/onnx-tensorrt)

!!! tip
    
    Since the ONNX parser ships with TensorRT, we have included a convenience method for using this
    workflow with torch2trt.  If you want to quickly try the ONNX method using the torch2trt interface, just call ``torch2trt(..., use_onnx=True)``.
    This will perform conversion on the module by exporting the model using PyTorch's JIT tracer,
    and parsing with TensorRT's ONNX parser.
    
* [TRTorch](https://github.com/NVIDIA/TRTorch)

Which one you use depends largely on your use case. The differences often come down to

## Layer support

Modern deep learning frameworks are large, and there often arise
caveats converting between frameworks using a given workflow.  These could include
limitations in serialization or parsing formats.  Or in some instances, it may be possible
the layer could be supported, but it has just not been done yet.   TRTorch is strong 
in the sense that it will default to the original PyTorch method for layers 
which are not converted to TensorRT.  The best way to know 
which conversion method works for you is to try converting your model. 

## Feature support

TensorRT is evolving and the conversion workflows may have varying level 
of feature support.  In some instances, you may wish to use a latest feature of TensorRT, like dynamic shapes,
but it is not supported in torch2trt or the interface has not yet been exposed.  In this
instance, we recommend checking to see if it is supported by one of the other workflows.  The ONNX
converter is typically strong in this regards, since the parser is distributed with TensorRT.  

!!! note

    If there is a TensorRT feature you wished to see in torch2trt, please let us know.  We can not gaurantee this will be done, but it helps us gauge interest.

## Extensibility / Ease of Use

In case none of the converters satisfy for your use case, you may find it necessary to adapt
the converter to fit your needs.  This is very intuitive with torch2trt,
since it is done inline with Python, and there are many [examples](converters.md) to reference.  If you know 
how the original PyTorch method works, and have the TensorRT Python API on hand, it is relatively straight forward to adapt torch2trt to your needs.
The extensibility is often helpful when you want to implement a converter that is specific to the 
context the layer appears in.  

