#include <iostream>
#include <NvInfer.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <torch/torch.h>
#include <cuda_runtime_api.h>
#include "interpolate.pb.h"


using namespace nvinfer1;


namespace torch2trt
{

class interpolate_Plugin : public IPluginV2 {
private:
  interpolate_Message message;
  at::TensorOptions tensor_options;
  std::vector<long> input_sizes;
  std::vector<long> output_sizes;

public:
  interpolate_Plugin(interpolate_Message message) : message(message) {}

  const char* getPluginType() const override {
    return "interpolate";
  };

  const char* getPluginVersion() const override {
    return "1";
  }

  int getNbOutputs() const override {
    return 1;
  } 

  Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override {
    Dims dims;
    dims.nbDims = inputs->nbDims;

    dims.d[0] = inputs->d[0];
    for (int i = 0; i < message.size_size(); i++) {
      dims.d[i + 1] = message.size(i);
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
      message.set_dtype(DataTypeMessage::kFloat);
    } else if (type == DataType::kHALF) {
      tensor_options = tensor_options.dtype(c10::kHalf);
      message.set_dtype(DataTypeMessage::kHalf);
    }
      
    // set input sizes
    for (int i = 0; i < inputDims[0].nbDims; i++) {
      message.add_input_size(inputDims[0].d[i]);
    }

    // set output sizes
    for (int i = 0; i < outputDims[0].nbDims; i++) {
      message.add_output_size(outputDims[0].d[i]);
    }
  }

  int initialize() override {
    // set device
    tensor_options = tensor_options.device(c10::kCUDA);
      
    // set data type
    if (message.dtype() == DataTypeMessage::kFloat) {
        tensor_options = tensor_options.dtype(c10::kFloat);
    } else if (message.dtype() == DataTypeMessage::kHalf) {
        tensor_options = tensor_options.dtype(c10::kHalf);
    }
      
    input_sizes.resize(message.input_size_size());
    output_sizes.resize(message.output_size_size());
    
    for (int i = 0; i < message.input_size_size(); i++) {
        input_sizes[i] = message.input_size(i);
    }
    for (int i = 0; i < message.output_size_size(); i++) {
        output_sizes[i] = message.output_size(i);
    }
      
    return 0;
  }

  void terminate() override {}

  size_t getWorkspaceSize(int maxBatchSize) const override { return 0; }

  int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override {
    // get input / output dimensions
    std::vector<long> batch_input_sizes = input_sizes;
    std::vector<long> batch_output_sizes = output_sizes;
    batch_input_sizes.insert(batch_input_sizes.begin(), batchSize);
    batch_output_sizes.insert(batch_output_sizes.begin(), batchSize);

    // create tensor wrappers
    at::Tensor input = at::from_blob((void*) inputs[0], batch_input_sizes, [](void*){}, tensor_options);
    at::Tensor output = at::from_blob(outputs[0], batch_output_sizes, [](void*){}, tensor_options);

    // create new torch cuda stream
    at::cuda::CUDAStream torch_stream = at::cuda::getStreamFromPool();
    at::cuda::CUDAStreamGuard torch_guard(torch_stream);

    // capture current work on tensorrt cuda stream
    cudaEvent_t event;
    cudaEventCreate(&event);
    cudaEventRecord(event, stream);

    // make torch cuda stream wait on tensorrt work
    cudaStreamWaitEvent(torch_stream.stream(), event, 0);

    // enqueue work
    if (message.mode() == "bilinear") {
      at::upsample_bilinear2d_out(output, input, {message.size(0), message.size(1)}, message.align_corners());
    } else if (message.mode() == "nearest") {
      at::upsample_nearest2d_out(output, input, {message.size(0), message.size(1)});
    } else if (message.mode() == "area") {
      at::adaptive_avg_pool2d_out(output, input, {message.size(0), message.size(1)});
    } else if (message.mode() == "bicubic") {
      at::upsample_bicubic2d_out(output, input, {message.size(0), message.size(1)}, message.align_corners());
    }

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
    return message.SerializeAsString().size();
  }

  void serialize(void* buffer) const override {
    message.SerializeToArray(buffer, getSerializationSize());
  }

  void destroy() override {}

  IPluginV2* clone() const override {
    return new interpolate_Plugin(message);
  }

  void setPluginNamespace(const char* pluginNamespace) override {}

  const char *getPluginNamespace() const override {
    return "torch2trt";
  }

};

class interpolate_PluginCreator : public IPluginCreator {
public:
  interpolate_PluginCreator() {}

  const char *getPluginNamespace() const override {
    return "torch2trt";
  }

  const char *getPluginName() const override {
    return "interpolate";
  }

  const char *getPluginVersion() const override {
    return "1";
  }

  IPluginV2 *deserializePlugin(const char *name, const void *data, size_t length) override {
    interpolate_Message message;
    message.ParseFromArray(data, length);
    return new interpolate_Plugin(message);
  }

  void setPluginNamespace(const char *N) override {}
  const PluginFieldCollection *getFieldNames() override { return nullptr; }

  IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) override { return nullptr; }

};

REGISTER_TENSORRT_PLUGIN(interpolate_PluginCreator);

}
