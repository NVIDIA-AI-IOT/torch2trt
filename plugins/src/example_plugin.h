#ifndef TORCH2TRT_PLUGIN_EXAMPLE
#define TORCH2TRT_PLUGIN_EXAMPLE


#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <cuda_fp16.h>
#include <vector>
#include <string>

#define EXAMPLE_PLUGIN_NAME "ExamplePlugin"
#define EXAMPLE_PLUGIN_VERSION "1"


using namespace nvinfer1;


namespace torch2trt_plugins {


template<typename T>
void exampleFuncton(T *x, T *y, float scale, int size, cudaStream_t stream=0);
void exampleFunctonHalf(__half *x, __half *y, float scale, int size, cudaStream_t stream=0);


class ExamplePlugin : public IPluginV2Ext {
public:
    int32_t inputSize;
    DataType dataType;
    float scale;
    std::string pluginNamespace;

    ExamplePlugin(float scale=2.0);
    ExamplePlugin(float scale, int32_t inputSize, DataType dataType);
    ExamplePlugin(void const* serialData, size_t serialLength);
    ~ExamplePlugin();

    /* IPluginV2 methods */

    AsciiChar const* getPluginType() const noexcept override;

    AsciiChar const* getPluginVersion() const noexcept override;

    int32_t getNbOutputs() const noexcept override;

    Dims getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept override;

    bool supportsFormat(DataType type, PluginFormat format) const noexcept;

    int32_t initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(int32_t maxBatchSize) const noexcept override;

    int32_t enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept
        override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    void destroy() noexcept override;


    void setPluginNamespace(AsciiChar const* pluginNamespace) noexcept override;

    AsciiChar const* getPluginNamespace() const noexcept override;

    // IPluginV2Ext methods
    IPluginV2Ext* clone() const noexcept override;
    DataType getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept override;
    bool isOutputBroadcastAcrossBatch(int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept override;
    bool canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept override;
    void configurePlugin(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
        DataType const* inputTypes, DataType const* outputTypes, bool const* inputIsBroadcast,
        bool const* outputIsBroadcast, PluginFormat floatFormat, int32_t maxBatchSize) noexcept override;
};

class ExamplePluginCreator : public IPluginCreator {
private:
    PluginFieldCollection fieldCollection;
    std::vector<PluginField> fields;
    std::string pluginNamespace;

public:
    ExamplePluginCreator();
    
    /* IPluginCreator methods */
    AsciiChar const* getPluginName() const noexcept override;

    AsciiChar const* getPluginVersion() const noexcept override;

    PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV2* createPlugin(AsciiChar const* name, PluginFieldCollection const* fc) noexcept override;

    IPluginV2* deserializePlugin(AsciiChar const* name, void const* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(AsciiChar const* pluginNamespace) noexcept override;


    AsciiChar const* getPluginNamespace() const noexcept override;
};

}

#endif