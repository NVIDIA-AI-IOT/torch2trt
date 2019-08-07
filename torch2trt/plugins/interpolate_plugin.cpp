#include <iostream>
#include <NvInfer.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <torch/torch.h>
#include <cuda_runtime_api.h>
#include "torch_plugin.h"
#include "torch_plugin.pb.h"
#include "interpolate_plugin.pb.h"


using namespace nvinfer1;


namespace torch2trt
{


// assumes all tensors CUDA, all dtypes (inputs and outputs) are same
class InterpolatePlugin : public TorchPlugin {
protected:
  InterpolatePluginMsg method;
    
public:
  InterpolatePlugin(TorchPluginMsg msg) : TorchPlugin(msg) {
      msg.method().UnpackTo(&method);
  }
    
  const char* getPluginType() const override {
    return "interpolate";
  };

  IPluginV2* clone() const override {
    return new InterpolatePlugin(msg);
  }

  // torch plugins must implement this forward method
  virtual void forward(std::vector<torch::Tensor> outputs, std::vector<torch::Tensor> inputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    if (method.mode() == "bilinear") {
      at::upsample_bilinear2d_out(output, input, {method.size(0), method.size(1)}, method.align_corners());
    } else if (method.mode() == "nearest") {
      at::upsample_nearest2d_out(output, input, {method.size(0), method.size(1)});
    } else if (method.mode() == "area") {
      at::adaptive_avg_pool2d_out(output, input, {method.size(0), method.size(1)});
    } else if (method.mode() == "bicubic") {
      at::upsample_bicubic2d_out(output, input, {method.size(0), method.size(1)}, method.align_corners());
    }
  };

};

class InterpolatePluginCreator : public TorchPluginCreator {
public:
  InterpolatePluginCreator() {}

  const char *getPluginName() const override {
    return "interpolate";
  }

  IPluginV2 *deserializePlugin(const char *name, const void *data, size_t length) override {
    TorchPluginMsg msg;
    msg.ParseFromArray(data, length);
    return new InterpolatePlugin(msg);
  }

};

REGISTER_TENSORRT_PLUGIN(InterpolatePluginCreator);

}
