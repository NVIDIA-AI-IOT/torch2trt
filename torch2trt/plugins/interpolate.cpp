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

using namespace nvinfer1;

namespace torch2trt {


class InterpolatePlugin : public IPluginV2 {
private:
    
  // configured by class
  at::TensorOptions tensor_options;
  std::vector<int64_t> input_sizes;
  std::vector<int64_t> output_sizes;
  DataType dtype;
    
  // configured by user
  std::vector<int64_t> size;
  std::string mode;
  bool align_corners;

public:
    
  // create from arguments
  InterpolatePlugin(std::vector<int64_t> size, std::string mode, bool align_corners) :
    size(size), mode(mode), align_corners(align_corners)
  {}
    
  InterpolatePlugin(const char *data, size_t length) : InterpolatePlugin(std::string(data, length)) {}
    
  // create from serialized data
  InterpolatePlugin(const std::string &data) {
      deserializeFromString(data);
  }
    
  void testPackContainer() {
      packContainer();
  }
    
  // packs data into container for serialization
  torch::jit::script::Module packContainer() const {
      
      torch::jit::script::Module container;
      
      auto i_size = torch::IValue(size);
      auto i_mode = torch::IValue(mode);
      auto i_align_corners = torch::IValue(align_corners);
      auto i_dtype = torch::IValue((int) dtype);
      auto i_input_sizes = torch::IValue(input_sizes);
      auto i_output_sizes = torch::IValue(output_sizes);
      
      // set by user
//       container.register_attribute("size", i_size.type(), i_size);
//       container.register_attribute("mode", i_mode.type(), i_mode);
//       container.register_attribute("align_corners", i_align_corners.type(), i_align_corners);
      
//       // configured by TRT
//       container.register_attribute("dtype", i_dtype.type(), i_dtype);
//       container.register_attribute("input_sizes", i_input_sizes.type(), i_input_sizes);
//       container.register_attribute("output_sizes", i_output_sizes.type(), i_output_sizes);
  }
    
  // unpacks data from container into class attributes
  void unpackContainer(const torch::jit::script::Module& container) {
      for (auto a : container.named_attributes()) {
          if (a.name == "size") {
              // TODO(jwelsh) toIntListRef().vec() is removed 1.5+, use toIntVector() instead... need to handle with preproc
              size = a.value.toIntListRef().vec();
          } else if (a.name == "mode") {
              mode = a.value.toStringRef();
          } else if (a.name == "align_corners") {
              align_corners = a.value.toBool();
          } else if (a.name == "dtype") {
              dtype = (DataType) a.value.toInt();
          } else if (a.name == "input_sizes") {
              input_sizes = a.value.toIntListRef().vec();
          } else if (a.name == "output_sizes") {
              output_sizes = a.value.toIntListRef().vec();
          }
      }
  }
   
  void deserializeFromString(const std::string &data) {
//       char *data_ptr = data.c_str();
//       size_t size = data.size();
      torch::jit::script::Module container;
      std::istringstream data_stream(data);
      container = torch::jit::load(data_stream);
      unpackContainer(container);
  }
    
  std::string serializeToString() const {
//       torch::jit::script::Module container = packContainer();
//       std::ostringstream data_str;
//       container.save(data_str);
//       return data_str.str();
      return "hello";
  }

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
    for (int i = 0; i < size.size(); i++) {
      dims.d[i + 1] = size[i];
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

  int initialize() override {
    // set device
    tensor_options = tensor_options.device(c10::kCUDA);
      
    // set data type
    if (dtype == DataType::kFLOAT) {
        tensor_options = tensor_options.dtype(c10::kFloat);
    } else if (dtype == DataType::kHALF) {
        tensor_options = tensor_options.dtype(c10::kHalf);
    }
      
//     input_sizes.resize(message.input_size_size());
//     output_sizes.resize(message.output_size_size());
    
//     for (int i = 0; i < message.input_size_size(); i++) {
//         input_sizes[i] = message.input_size(i);
//     }
//     for (int i = 0; i < message.output_size_size(); i++) {
//         output_sizes[i] = message.output_size(i);
//     }
      
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
    if (mode == "bilinear") {
      at::upsample_bilinear2d_out(output, input, {size[0], size[1]}, align_corners);
    } else if (mode == "nearest") {
      at::upsample_nearest2d_out(output, input, {size[0], size[1]});
    } else if (mode == "area") {
      at::adaptive_avg_pool2d_out(output, input, {size[0], size[1]});
    } else if (mode == "bicubic") {
      at::upsample_bicubic2d_out(output, input, {size[0], size[1]}, align_corners);
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
    return serializeToString().size();
  }
    
  void serialize(void* buffer) const override {
      std::string data = serializeToString();
      size_t size = getSerializationSize();
      data.copy((char *) buffer, size);
  }

  void destroy() override {}

  IPluginV2* clone() const override {
    return new InterpolatePlugin(size, mode, align_corners);
  }

  void setPluginNamespace(const char* pluginNamespace) override {}

  const char *getPluginNamespace() const override {
    return "torch2trt";
  }

};

class InterpolatePluginCreator : public IPluginCreator {
public:
  InterpolatePluginCreator() {}

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
    return new InterpolatePlugin((const char*) data, length);
  }

  void setPluginNamespace(const char *N) override {}
  const PluginFieldCollection *getFieldNames() override { return nullptr; }

  IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) override { return nullptr; }

};

REGISTER_TENSORRT_PLUGIN(InterpolatePluginCreator);
    

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<InterpolatePlugin>(m, "InterpolatePlugin")
        .def(py::init<std::vector<int64_t>, std::string, bool>(), py::arg("size"), py::arg("mode"), py::arg("align_corners"))
        .def("testPackContainer", &InterpolatePlugin::testPackContainer)
        .def("getSerializationSize", &InterpolatePlugin::getSerializationSize)
        .def("serializeToString", &InterpolatePlugin::serializeToString);
}
    
} // namespace torch2trt