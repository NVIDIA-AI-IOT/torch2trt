#include "example_plugin.h"
#include "cuda_runtime.h"
#include <cuda_fp16.h>
#include <iostream>


namespace torch2trt_plugins {

// KERNELS

template<typename T>
__global__ void exampleKernel(T *x, T *y, float scale, int size) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < size) {
        y[index] = x[index] * scale;
    }
}

template __global__ void exampleKernel<float>(float *, float *, float, int);
template __global__ void exampleKernel<int>(int *, int *, float, int);
template __global__ void exampleKernel<int8_t>(int8_t *, int8_t *, float, int);


__global__ void exampleKernelHalf(__half *x, __half *y, float scale, int size) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < size) {
        y[index] = __hmul(x[index], __float2half_rn(scale));
    }
}


// FUNCTIONS


template<typename T>
void exampleFuncton(T *x, T *y, float scale, int size, cudaStream_t stream) {
    int nThreads = 32;
    int nBlocks = (size / 32) + 1;
    exampleKernel<<<nBlocks, nThreads, 0, stream>>>(x, y, scale, size);
}

template void exampleFuncton<float>(float *, float *, float, int, cudaStream_t);
template void exampleFuncton<int>(int *, int *, float, int, cudaStream_t);
template void exampleFuncton<int8_t>(int8_t *, int8_t *, float, int, cudaStream_t);

void exampleFunctonHalf(__half *x, __half *y, float scale, int size, cudaStream_t stream) {
    int nThreads = 32;
    int nBlocks = (size / 32) + 1;
    exampleKernelHalf<<<nBlocks, nThreads, 0, stream>>>(x, y, scale, size);
}


// PLUGIN


ExamplePlugin::ExamplePlugin(float scale) : scale(scale) {

}

ExamplePlugin::ExamplePlugin(float scale, int32_t inputSize, DataType dataType) : scale(scale), inputSize(inputSize), dataType(dataType) {

}

ExamplePlugin::ExamplePlugin(void const* serialData, size_t serialLength) {
    this->scale = *reinterpret_cast<float const*>((char const*)serialData);
    this->inputSize = *reinterpret_cast<int32_t const*>((char const*)serialData + sizeof(this->scale));
    this->dataType = *reinterpret_cast<DataType const*>((char const*)serialData + sizeof(this->scale) + sizeof(this->dataType));
}

ExamplePlugin::~ExamplePlugin() {

}

AsciiChar const * ExamplePlugin::getPluginType() const noexcept {
    return EXAMPLE_PLUGIN_NAME;
}

AsciiChar const * ExamplePlugin::getPluginVersion() const noexcept {
    return EXAMPLE_PLUGIN_VERSION;
}

int32_t ExamplePlugin::getNbOutputs() const noexcept {
    return 1;
}

Dims ExamplePlugin::getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept {
    return inputs[0];
}

bool ExamplePlugin::supportsFormat(DataType type, PluginFormat format) const noexcept {
    return (type == DataType::kFLOAT) || (type == DataType::kINT32)
        || (type == DataType::kINT8) || (type == DataType::kHALF);
}

int32_t ExamplePlugin::initialize() noexcept {
    return 0;
};

void ExamplePlugin::terminate() noexcept {
    
};

size_t ExamplePlugin::getWorkspaceSize(int32_t maxBatchSize) const noexcept {
    return 0;
};

int32_t ExamplePlugin::enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept {
    const int totalSize = batchSize * this->inputSize;
    switch (this->dataType) {
        case DataType::kFLOAT: {
            exampleFuncton<float>((float*) inputs[0], (float*) outputs[0], this->scale, totalSize, stream);
            break;
        }
        case DataType::kINT32: {
            exampleFuncton<int>((int*) inputs[0], (int*) outputs[0], this->scale, totalSize, stream);
            break;
        }
        case DataType::kHALF: {
            exampleFunctonHalf((__half*) inputs[0], (__half*) outputs[0], this->scale, totalSize, stream);
            break;
        }
        case DataType::kINT8: {
            exampleFuncton<int8_t>((int8_t*) inputs[0], (int8_t*) outputs[0], this->scale, totalSize, stream);
            break;
        }
        default: {
            return 1;
        }
    }
    return 0;
};

size_t ExamplePlugin::getSerializationSize() const noexcept {
    return sizeof(this->scale) + sizeof(this->inputSize) + sizeof(this->dataType);
};

void ExamplePlugin::serialize(void* buffer) const noexcept {
    *(reinterpret_cast<float*>((char*) buffer)) = this->scale;
    *(reinterpret_cast<int32_t*>((char*)buffer + sizeof(this->scale))) = this->inputSize;
    *(reinterpret_cast<DataType*>((char*)buffer + sizeof(this->scale) + sizeof(this->inputSize))) = this->dataType;
};

void ExamplePlugin::destroy() noexcept {
    delete this;
};


void ExamplePlugin::setPluginNamespace(AsciiChar const* pluginNamespace) noexcept {
    this->pluginNamespace = pluginNamespace;
};

AsciiChar const* ExamplePlugin::getPluginNamespace() const noexcept {
    return this->pluginNamespace.c_str();
};

// IPluginV2Ext methods

IPluginV2Ext* ExamplePlugin::clone() const noexcept { 
    auto *plugin = new ExamplePlugin(this->scale, this->inputSize, this->dataType);
    plugin->setPluginNamespace(this->getPluginNamespace());
    return plugin;
};

DataType ExamplePlugin::getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept {
    if (!nbInputs) {
        return DataType::kFLOAT;
    } else {
        return inputTypes[0];
    }
};

bool ExamplePlugin::isOutputBroadcastAcrossBatch(int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept {
    return false; // TODO: optimize by enabling broadcast
}

bool ExamplePlugin::canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept {
    return false;
}

void ExamplePlugin::configurePlugin(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
        DataType const* inputTypes, DataType const* outputTypes, bool const* inputIsBroadcast,
        bool const* outputIsBroadcast, PluginFormat floatFormat, int32_t maxBatchSize) noexcept {
    Dims d = inputDims[0];
    this->inputSize = 1;
    for (int i = 0; i < d.nbDims; i++) {
        this->inputSize *= d.d[i];
    }
    this->dataType = inputTypes[0];
};

// PLUGIN CREATOR

ExamplePluginCreator::ExamplePluginCreator() {
    this->fields = {
        PluginField("scale")
    };
    this->fieldCollection.fields = this->fields.data();
    this->fieldCollection.nbFields = this->fields.size();
}

AsciiChar const* ExamplePluginCreator::getPluginName() const noexcept {
    return EXAMPLE_PLUGIN_NAME;
}

AsciiChar const* ExamplePluginCreator::getPluginVersion() const noexcept {
    return EXAMPLE_PLUGIN_VERSION;
};

PluginFieldCollection const* ExamplePluginCreator::getFieldNames() noexcept {
    return &this->fieldCollection;
};

IPluginV2* ExamplePluginCreator::createPlugin(AsciiChar const* name, PluginFieldCollection const* fc) noexcept {
    if (fc == nullptr) {
        return new ExamplePlugin();
    } else {
        float scale = *((float*) fc->fields[0].data);
        return new ExamplePlugin(scale);
    }
}

IPluginV2* ExamplePluginCreator::deserializePlugin(AsciiChar const* name, void const* serialData, size_t serialLength) noexcept {
    auto *plugin = new ExamplePlugin(serialData, serialLength);
    plugin->setPluginNamespace(this->getPluginNamespace());
    return plugin;
}

void ExamplePluginCreator::setPluginNamespace(AsciiChar const* pluginNamespace) noexcept {
    this->pluginNamespace = pluginNamespace;
};

AsciiChar const* ExamplePluginCreator::getPluginNamespace() const noexcept {
    return this->pluginNamespace.c_str();
};

REGISTER_TENSORRT_PLUGIN(ExamplePluginCreator);

}