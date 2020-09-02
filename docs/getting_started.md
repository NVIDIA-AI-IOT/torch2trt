# Getting Started

Follow these steps to get started using torch2trt.

!!! note

    torch2trt depends on the TensorRT Python API.  On Jetson, this is included with the latest JetPack.  For desktop, please follow the [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html).  You may also try installing torch2trt inside one of the NGC PyTorch docker containers for [Desktop](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) or [Jetson](https://ngc.nvidia.com/catalog/containers/nvidia:l4t-pytorch).

### Install Without plugins

To install without compiling plugins, call the following

```bash
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python setup.py install
```

### Install With plugins

To install with plugins to support some operations in PyTorch that are not natviely supported with TensorRT, call the following

!!! note
    
    Please note, this currently only includes the interpolate plugin.  This plugin requires PyTorch 1.3+ for serialization.  

```bash
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
sudo python setup.py install --plugins
```

