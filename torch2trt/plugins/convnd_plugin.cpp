#include <iostream>
#include <NvInfer.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <torch/torch.h>
#include <cuda_runtime_api.h>
#include "torch_plugin.h"
#include "torch_plugin.pb.h"
#include "convnd_plugin.pb.h"


using namespace nvinfer1;


namespace torch2trt
{


// assumes all tensors CUDA, all dtypes (inputs and outputs) are same
class ConvNdPlugin : public TorchPlugin {
protected:
  ConvNdPluginMsg method;
    
public:
  ConvNdPlugin(TorchPluginMsg msg) : TorchPlugin(msg) {
      msg.method().UnpackTo(&method);
  }
    
  const char* getPluginType() const override {
    return "convnd";
  };

  IPluginV2* clone() const override {
    return new ConvNdPlugin(msg);
  }

  // torch plugins must implement this forward method
  virtual void forward(std::vector<torch::Tensor> outputs, std::vector<torch::Tensor> inputs) override {


  };

};

class ConvNdPluginCreator : public TorchPluginCreator {
public:
  ConvNdPluginCreator() {}

  const char *getPluginName() const override {
    return "convnd";
  }

  IPluginV2 *deserializePlugin(const char *name, const void *data, size_t length) override {
    TorchPluginMsg msg;
    msg.ParseFromArray(data, length);
    return new ConvNdPlugin(msg);
  }

};

REGISTER_TENSORRT_PLUGIN(ConvNdPluginCreator);

}
