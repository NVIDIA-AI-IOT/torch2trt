#include <torch/extension.h>
#include <torch/script.h>
#include <iostream>
#include <string>
#include <sstream>
#include <NvInfer.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <torch/torch.h>
#include <cuda_runtime_api.h>

#ifdef PRE_TRT8
#define TRT_NOEXCEPT
#else
#define TRT_NOEXCEPT noexcept
#endif

using namespace nvinfer1;

namespace torch2trt {

class GroupNormPlugin : public IPluginV2 {
private:
    // configured by class
    at::TensorOptions tensor_options;
    std::vector<int64_t> input_sizes;
    std::vector<int64_t> output_sizes;
    DataType dtype;

    // group norm parameters, configured by user
    int64_t num_groups;
    at::Tensor weight;
    at::Tensor bias;
    double eps;


public:

    // create from arguments
    GroupNormPlugin(int64_t num_groups, at::Tensor weight, at::Tensor bias, double eps) :
            num_groups{num_groups}, weight{weight}, bias{bias}, eps{eps}
    {}

    GroupNormPlugin(const char *data, size_t length) : GroupNormPlugin(std::string(data, length)) {}

    GroupNormPlugin(const std::string &data){
            deserializeFromString(data);
    }

    void deserializeFromString(const std::string &data) {
        std::istringstream data_stream(data);
        torch::serialize::InputArchive input_archive;
        input_archive.load_from(data_stream);
        {
            torch::IValue value;
            input_archive.read("num_groups", value);
#ifdef USE_DEPRECATED_INTLIST
            num_groups = value.toIntListRef().vec();
#else
            num_groups = value.toInt();
#endif
        }
	{
            torch::IValue value;
            input_archive.read("weight", value);
            weight = value.toTensor();
        }
	{
            torch::IValue value;
            input_archive.read("bias", value);
            bias = value.toTensor();
        }

        {
            torch::IValue value;
            input_archive.read("eps", value);
#ifdef USE_DEPRECATED_INTLIST
            eps = value.toDoubleListRef().vec();
#else
            eps = value.toDouble();
#endif
        }
	{
            torch::IValue value;
            input_archive.read("dtype", value);
            dtype = (DataType) value.toInt();
        }
        {
            torch::IValue value;
            input_archive.read("input_sizes", value);
#ifdef USE_DEPRECATED_INTLIST
            input_sizes = value.toIntListRef().vec();
#else
            input_sizes = value.toIntVector();
#endif
        }
        {
            torch::IValue value;
            input_archive.read("output_sizes", value);
#ifdef USE_DEPRECATED_INTLIST
            output_sizes = value.toIntListRef().vec();
#else
            output_sizes = value.toIntVector();
#endif
        }
    }
    std::string serializeToString() const {
        torch::serialize::OutputArchive output_archive;
        output_archive.write("num_groups", torch::IValue(num_groups));
        output_archive.write("weight", torch::IValue(weight));
        output_archive.write("bias", torch::IValue(bias));
        output_archive.write("eps", torch::IValue(eps));
        output_archive.write("dtype", torch::IValue((int) dtype));
        output_archive.write("input_sizes", torch::IValue(input_sizes));
        output_archive.write("output_sizes", torch::IValue(output_sizes));
        std::ostringstream data_str;
        output_archive.save_to(data_str);
        return data_str.str();
    }

    const char* getPluginType() const TRT_NOEXCEPT override {
      return "group_norm";
    };

    const char* getPluginVersion() const TRT_NOEXCEPT override {
      return "1";
    }

    int getNbOutputs() const TRT_NOEXCEPT override {
      return 1;
    }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) TRT_NOEXCEPT override {
      Dims dims;
      dims.nbDims = inputs->nbDims;

      for (int i = 0; i < inputs->nbDims; i++) {
        dims.d[i] = inputs->d[i];
      }

      return dims;
    }

    bool supportsFormat(DataType type, PluginFormat format) const TRT_NOEXCEPT override {
      if (format != PluginFormat::kLINEAR) {
        return false;
      }
      if (type == DataType::kINT32 || type == DataType::kINT8) {
        return false;
      }
      return true;
    }

  void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims,
      int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) TRT_NOEXCEPT override {

    // set data type
    if (type == DataType::kFLOAT) {
      tensor_options = tensor_options.dtype(c10::kFloat);
      dtype = type;
    } else if (type == DataType::kHALF) {
      tensor_options = tensor_options.dtype(c10::kHalf);
      dtype = type;
    }

    // set input sizes
    input_sizes.resize(inputDims[0].nbDims);
    for (int i = 0; i < inputDims[0].nbDims; i++) {
      input_sizes[i] = inputDims[0].d[i];
    }

    // set output sizes
    output_sizes.resize(outputDims[0].nbDims);
    for (int i = 0; i < outputDims[0].nbDims; i++) {
      output_sizes[i] = outputDims[0].d[i];
    }
  }

  int initialize() TRT_NOEXCEPT override {
    // set device
    tensor_options = tensor_options.device(c10::kCUDA);

    // set data type
    if (dtype == DataType::kFLOAT) {
        tensor_options = tensor_options.dtype(c10::kFloat);
    } else if (dtype == DataType::kHALF) {
        tensor_options = tensor_options.dtype(c10::kHalf);
    }


    weight = weight.to(tensor_options);
    bias = bias.to(tensor_options);

    return 0;
  }

  void terminate() TRT_NOEXCEPT override {}

  size_t getWorkspaceSize(int maxBatchSize) const TRT_NOEXCEPT override { return 0; }

#ifdef PRE_TRT8
int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override {
#else
int32_t enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override {
#endif
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
    // Group_norm function from PyTorch: https://pytorch.org/cppdocs/api/function_namespaceat_1a6bc1e9504ea440c6c96ff8a8b94333f2.html#exhale-function-namespaceat-1a6bc1e9504ea440c6c96ff8a8b94333f2
    at::Tensor output_tmp = at::group_norm(input, num_groups, weight, bias, eps=eps);
    output.copy_(output_tmp);

    // capture event on enqueued stream
    cudaEvent_t torch_event;
    cudaEventCreate(&torch_event);
    cudaEventRecord(torch_event, torch_stream.stream());
    cudaStreamWaitEvent(stream, torch_event, 0);

    cudaEventDestroy(event);
    cudaEventDestroy(torch_event);

    return 0;
  }


  size_t getSerializationSize() const TRT_NOEXCEPT override {
    return serializeToString().size();
  }

  void serialize(void* buffer) const TRT_NOEXCEPT override {
      std::string data = serializeToString();
      size_t size = getSerializationSize();
      data.copy((char *) buffer, size);
  }

  void destroy() TRT_NOEXCEPT override {}

  IPluginV2* clone() const TRT_NOEXCEPT override {
    return new GroupNormPlugin(num_groups, weight, bias, eps);
  }

  void setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT override {}

  const char *getPluginNamespace() const TRT_NOEXCEPT override {
    return "torch2trt";
  }

};

class GroupNormPluginCreator : public IPluginCreator {
public:
  GroupNormPluginCreator() {}

  const char *getPluginNamespace() const TRT_NOEXCEPT override {
    return "torch2trt";
  }

  const char *getPluginName() const TRT_NOEXCEPT override {
    return "group_norm";
  }

  const char *getPluginVersion() const TRT_NOEXCEPT override {
    return "1";
  }

  IPluginV2 *deserializePlugin(const char *name, const void *data, size_t length) TRT_NOEXCEPT override {
    return new GroupNormPlugin((const char*) data, length);
  }

  void setPluginNamespace(const char *N) TRT_NOEXCEPT override {}
  const PluginFieldCollection *getFieldNames() TRT_NOEXCEPT override { return nullptr; }

  IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) TRT_NOEXCEPT override { return nullptr; }

};


REGISTER_TENSORRT_PLUGIN(GroupNormPluginCreator);

} // namespace torch2trt




