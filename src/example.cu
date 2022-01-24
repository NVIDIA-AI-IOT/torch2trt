#include "example.h"
#include "cuda_runtime.h"



namespace torch2trt_plugins {


template<typename T>
__global__ void exampleKernel(T *x, int size) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < size) {
        x[index] *= 2;
    }
}

template __global__ void exampleKernel<float>(float *, int);
template __global__ void exampleKernel<int>(int *, int);


template<typename T>
void exampleFuncton(T *x, int size) {
    int nThreads = 32;
    int nBlocks = (size / 32) + 1;
    exampleKernel<<<nBlocks, nThreads>>>(x, size);
}

template void exampleFuncton<float>(float *, int);
template void exampleFuncton<int>(int *, int);


ExamplePlugin::ExamplePlugin() {

}

ExamplePlugin::~ExamplePlugin() {

}

AsciiChar const * ExamplePlugin::getPluginType() const noexcept {
    return "ExamplePlugin";
}

AsciiChar const * ExamplePlugin::getPluginVersion() const noexcept {
    return "1";
}

int32_t ExamplePlugin::getNbOutputs() const noexcept {
    return 1;
}

Dims ExamplePlugin::getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept {
    return inputs[0];
}

bool ExamplePlugin::supportsFormat(DataType type, PluginFormat format) const noexcept {
    return (type == DataType::kFLOAT) || (type == DataType::kINT32);
}

void ExamplePlugin::configureWithFormat(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
    DataType type, PluginFormat format, int32_t maxBatchSize) noexcept {
    Dims d = inputDims[0];
    this->inputSize = 1;
    for (int i = 0; i < d.nbDims; i++) {
        this->inputSize *= d.d[i];
    }
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
    return nullptr; 
};

void ExamplePlugin::setPluginNamespace(AsciiChar const* pluginNamespace) noexcept {

};

AsciiChar const* ExamplePlugin::getPluginNamespace() const noexcept {
    return "";
};

}