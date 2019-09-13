PLUGIN_SRC_TEMPLATE = \
"""
#include <iostream>
#include <NvInfer.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <torch/torch.h>
#include <cuda_runtime_api.h>
#include "${PLUGIN_NAME}.pb.h"


using namespace nvinfer1;


namespace torch2trt
{
namespace ${PLUGIN_NAME}
{

${EXTRA_SRC}

// assumes all tensors CUDA, all dtypes (inputs and outputs) are same
class ${PLUGIN_NAME}_Plugin : public IPluginV2 {
protected:
  ${PLUGIN_NAME}_Msg msg;
  at::TensorOptions tensor_options;
  std::vector<std::vector<long>> input_sizes; // includes batch dim
  std::vector<std::vector<long>> output_sizes; // included batch dim
  
  ${PLUGIN_MEMBERS}

public:
  ${PLUGIN_NAME}_Plugin(${PLUGIN_NAME}_Msg msg) : msg(msg) {}

  const char* getPluginType() const override {
    return "${PLUGIN_NAME}";
  };

  const char* getPluginVersion() const override {
    return "1";
  }

  int getNbOutputs() const override {
    return msg.output_shapes_size();
  } 

  Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override {
    Dims dims;
    dims.nbDims = msg.output_shapes(index).size_size();

    for (int i = 0; i < msg.output_shapes(index).size_size(); i++)
    {
        dims.d[i] = msg.output_shapes(index).size(i);
    }

    return dims;
  }

  bool supportsFormat(DataType type, PluginFormat format) const override {
    if (format != PluginFormat::kNCHW) {
      return false;
    }
    if (type == DataType::kINT32 || type == DataType::kINT8) {
      return false;
    }
    return true;
  }

  void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims,
      int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override {
    // set data type
    if (type == DataType::kFLOAT) {
      msg.set_dtype(DataTypeMsg::kFloat);
    } else if (type == DataType::kHALF) {
      msg.set_dtype(DataTypeMsg::kHalf);
    }
  }

  int initialize() override {
    tensor_options = tensor_options.device(c10::kCUDA).layout(c10::kStrided).requires_grad(false);
      
    // set data type
    if (msg.dtype() == DataTypeMsg::kFloat) {
        tensor_options = tensor_options.dtype(c10::kFloat);
    } else if (msg.dtype() == DataTypeMsg::kHalf) {
        tensor_options = tensor_options.dtype(c10::kHalf);
    }
      
    for (int i = 0; i < msg.input_shapes_size(); i++)
    {
        auto shape = msg.input_shapes(i);
        std::vector<long> size;
        size.push_back(0); // add batch dim (needs to be filled out when enqueue)
        for (int j = 0; j < shape.size_size(); j++)
        {
            size.push_back(shape.size(j));
        }
        input_sizes.push_back(size);
    }
      
    for (int i = 0; i < msg.output_shapes_size(); i++)
    {
        auto shape = msg.output_shapes(i);
        std::vector<long> size;
        size.push_back(0); // add batch dim (needs to be filled out when enqueue)
        for (int j = 0; j < shape.size_size(); j++)
        {
            size.push_back(shape.size(j));
        }
        output_sizes.push_back(size);
    }
    
    ${PLUGIN_SETUP}
    
    return 0;
  }

  void terminate() override {}

  size_t getWorkspaceSize(int maxBatchSize) const override { return 0; }

  int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override {
      
    // Create torch tensors for inputs
    std::vector<at::Tensor> input_tensors(input_sizes.size());
    std::vector<at::Tensor> output_tensors(output_sizes.size());
    
    for (int i = 0; i < input_tensors.size(); i++) {
        auto size = input_sizes[i];
        size[0] = batchSize;
        input_tensors[i] = at::from_blob((void*) inputs[i], size, [](void*){}, tensor_options);
    }
      
    for (int i = 0; i < output_tensors.size(); i++) {
        auto size = output_sizes[i];
        size[0] = batchSize;
        output_tensors[i] = at::from_blob((void*) outputs[i], size, [](void*){}, tensor_options);
    }
      
    // create new torch cuda stream
    at::cuda::CUDAStream torch_stream = at::cuda::getStreamFromPool();
    at::cuda::CUDAStreamGuard torch_guard(torch_stream);

    // capture current work on tensorrt cuda stream
    cudaEvent_t event;
    cudaEventCreate(&event);
    cudaEventRecord(event, stream);

    // make torch cuda stream wait on tensorrt work
    cudaStreamWaitEvent(torch_stream.stream(), event, 0);
      
    // BEGIN TORCH METHOD

    // enqueue work
    ${PLUGIN_FORWARD}

    // END TORCH METHOD
      
    // capture event on enqueued stream
    cudaEvent_t torch_event;
    cudaEventCreate(&torch_event);
    cudaEventRecord(torch_event, torch_stream.stream());

    cudaStreamWaitEvent(stream, torch_event, 0);

    cudaEventDestroy(event);
    cudaEventDestroy(torch_event);

    return 0;
  }

  size_t getSerializationSize() const override {
    return msg.SerializeAsString().size();
  }

  void serialize(void* buffer) const override {
    msg.SerializeToArray(buffer, getSerializationSize());
  }

  void destroy() override {}

  IPluginV2* clone() const override {
    return new ${PLUGIN_NAME}_Plugin(msg);
  }

  void setPluginNamespace(const char* pluginNamespace) override {}

  const char *getPluginNamespace() const override {
    return "torch2trt";
  }

};

class ${PLUGIN_NAME}_PluginCreator : public IPluginCreator {
public:
  ${PLUGIN_NAME}_PluginCreator() {}

  const char *getPluginNamespace() const override {
    return "torch2trt";
  }

  const char *getPluginName() const override {
    return "${PLUGIN_NAME}";
  }

  const char *getPluginVersion() const override {
    return "1";
  }

  IPluginV2 *deserializePlugin(const char *name, const void *data, size_t length) override {
    ${PLUGIN_NAME}_Msg msg;
    msg.ParseFromArray(data, length);
    return new ${PLUGIN_NAME}_Plugin(msg);
  }

  void setPluginNamespace(const char *N) override {}
  const PluginFieldCollection *getFieldNames() override { return nullptr; }

  IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) override { return nullptr; }

};


REGISTER_TENSORRT_PLUGIN(${PLUGIN_NAME}_PluginCreator);

} //namespace ${PLUGIN_NAME}
} //namespace torch2trt
"""