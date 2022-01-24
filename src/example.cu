#include "example.h"
#include "cuda_runtime.h"
#include <cuda_fp16.h>



namespace torch2trt_plugins {

// KERNELS

template<typename T>
__global__ void exampleKernel(T *x, T *y, int size) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < size) {
        y[index] = x[index] * 2;
    }
}

template __global__ void exampleKernel<float>(float *, float *, int);
template __global__ void exampleKernel<int>(int *, int *, int);
template __global__ void exampleKernel<int8_t>(int8_t *, int8_t *, int);


__global__ void exampleKernelHalf(__half *x, __half *y, int size) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < size) {
        y[index] = __hmul(x[index], __float2half_rn(2.0f));
    }
}


// FUNCTIONS


template<typename T>
void exampleFuncton(T *x, T *y, int size, cudaStream_t stream) {
    int nThreads = 32;
    int nBlocks = (size / 32) + 1;
    exampleKernel<<<nBlocks, nThreads, 0, stream>>>(x, y, size);
}

template void exampleFuncton<float>(float *, float *, int, cudaStream_t);
template void exampleFuncton<int>(int *, int *, int, cudaStream_t);
template void exampleFuncton<int8_t>(int8_t *, int8_t *, int, cudaStream_t);

void exampleFunctonHalf(__half *x, __half *y, int size, cudaStream_t stream) {
    int nThreads = 32;
    int nBlocks = (size / 32) + 1;
    exampleKernelHalf<<<nBlocks, nThreads, 0, stream>>>(x, y, size);
}


// PLUGIN


ExamplePlugin::ExamplePlugin() {

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

void ExamplePlugin::configureWithFormat(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
    DataType type, PluginFormat format, int32_t maxBatchSize) noexcept {
    Dims d = inputDims[0];
    this->inputSize = 1;
    for (int i = 0; i < d.nbDims; i++) {
        this->inputSize *= d.d[i];
    }
    this->dataType = type;
};

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
            exampleFuncton<float>((float*) inputs[0], (float*) outputs[0], totalSize, stream);
            break;
        }
        case DataType::kINT32: {
            exampleFuncton<int>((int*) inputs[0], (int*) outputs[0], totalSize, stream);
            break;
        }
        case DataType::kHALF: {
            exampleFunctonHalf((__half*) inputs[0], (__half*) outputs[0], totalSize, stream);
            break;
        }
        case DataType::kINT8: {
            exampleFuncton<int8_t>((int8_t*) inputs[0], (int8_t*) outputs[0], totalSize, stream);
            break;
        }
        default: {
            return 1;
        }
    }
    return 0;
};

size_t ExamplePlugin::getSerializationSize() const noexcept {
    return 0;
};

void ExamplePlugin::serialize(void* buffer) const noexcept {

};

void ExamplePlugin::destroy() noexcept {

};

IPluginV2* ExamplePlugin::clone() const noexcept { 
    return new ExamplePlugin();
};

void ExamplePlugin::setPluginNamespace(AsciiChar const* pluginNamespace) noexcept {
};

AsciiChar const* ExamplePlugin::getPluginNamespace() const noexcept {
    return EXAMPLE_PLUGIN_NAMESPACE;
};


// PLUGIN CREATOR

ExamplePluginCreator::ExamplePluginCreator() {
    memset(&this->fieldCollection, 0, sizeof(this->fieldCollection));
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
    return new ExamplePlugin();
}

IPluginV2* ExamplePluginCreator::deserializePlugin(AsciiChar const* name, void const* serialData, size_t serialLength) noexcept {
    return new ExamplePlugin();
}

void ExamplePluginCreator::setPluginNamespace(AsciiChar const* pluginNamespace) noexcept {};

AsciiChar const* ExamplePluginCreator::getPluginNamespace() const noexcept {
    return "torch2trt_plugins";
};

}