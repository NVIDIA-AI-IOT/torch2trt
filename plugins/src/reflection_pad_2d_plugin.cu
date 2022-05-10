#include "reflection_pad_2d_plugin.h"
#include "cuda_runtime.h"
#include <cuda_fp16.h>
#include <iostream>


namespace torch2trt_plugins {

// KERNELS

template<typename T>
__global__ void reflectionPad2dKernel(
        T *x, T *y, 
        int N, int C, int H, int W,
        int paddingLeft, int paddingRight, int paddingTop, int paddingBottom) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N*C*H*W) {
        return;
    }
    int IW = W - (paddingLeft + paddingRight);
    int IH = H - (paddingTop + paddingBottom);
    int n = (idx / (C*H*W)) % N;
    int c = (idx / (H*W)) % C;
    int h = (idx / W) % H;
    int w = idx % W;
    int iw;
    int ih; 
    int maxW = IW + paddingLeft;
    int maxH = IH + paddingTop;

    if (w < paddingLeft) {
        iw = paddingLeft - w;
    } else if (w >= maxW) {
        iw = IW - ((w - maxW) + 1) - 1;
    } else {
        iw = w - paddingLeft;
    }

    if (h < paddingTop) {
        ih = paddingTop - h;
    } else if (h >= maxH) {
        ih = IH - ((h - maxH) + 1) - 1;
    } else {
        ih = h - paddingTop;
    }

    y[n*C*H*W + c*H*W + h*W + w] = x[n*C*IH*IW + c*IH*IW + ih*IW + iw];
}

template __global__ void reflectionPad2dKernel<float>(float *, float *, int, int, int, int, int, int, int, int);
template __global__ void reflectionPad2dKernel<int>(int *, int *, int, int, int, int, int, int, int, int);
template __global__ void reflectionPad2dKernel<int8_t>(int8_t *, int8_t *, int, int, int, int, int, int, int, int);




// // FUNCTIONS


template<typename T>
void reflectionPad2dFunction(
    T *x, T *y, 
    int N, int C, int H, int W, 
    int paddingLeft, int paddingRight, int paddingTop, int paddingBottom, 
    cudaStream_t stream) {
    int size = N * C * H * W;
    int nThreads = 32;
    int nBlocks = (size / nThreads) + 1;
    reflectionPad2dKernel<<<nBlocks, nThreads, 0, stream>>>(x, y, N, C, H, W, paddingLeft, paddingRight, paddingTop, paddingBottom);
}

template void reflectionPad2dFunction<float>(
    float *, float *, 
    int, int, int, int, 
    int, int, int, int, 
    cudaStream_t);


// PLUGIN


ReflectionPad2dPlugin::ReflectionPad2dPlugin(int32_t paddingLeft, int32_t paddingRight, int32_t paddingTop, int32_t paddingBottom) : 
paddingLeft(paddingLeft), paddingRight(paddingRight), paddingTop(paddingTop), paddingBottom(paddingBottom) {
    initialize();
}

ReflectionPad2dPlugin::~ReflectionPad2dPlugin() {

}

AsciiChar const * ReflectionPad2dPlugin::getPluginType() const noexcept {
    return REFLECTION_PAD_2D_PLUGIN_NAME;
}

AsciiChar const * ReflectionPad2dPlugin::getPluginVersion() const noexcept {
    return REFLECTION_PAD_2D_PLUGIN_VERSION;
}

int32_t ReflectionPad2dPlugin::getNbOutputs() const noexcept {
    return 1;
}

Dims ReflectionPad2dPlugin::getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept {
    return Dims3(
        inputs->d[0],
        inputs->d[1] + paddingTop + paddingBottom,
        inputs->d[2] + paddingLeft + paddingRight
    );
}

bool ReflectionPad2dPlugin::supportsFormat(DataType type, PluginFormat format) const noexcept {
    if (format != PluginFormat::kLINEAR) {
        return false;
    }
    return (type == DataType::kFLOAT) || (type == DataType::kINT32)
        || (type == DataType::kINT8); // TODO: add half precision
}


int32_t ReflectionPad2dPlugin::initialize() noexcept {
    return 0;
};

void ReflectionPad2dPlugin::terminate() noexcept {
    
};

size_t ReflectionPad2dPlugin::getWorkspaceSize(int32_t maxBatchSize) const noexcept {
    return 0;
};

int32_t ReflectionPad2dPlugin::enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept {
    int N = batchSize;
    int C = outputDims.d[0];
    int H = outputDims.d[1];
    int W = outputDims.d[2];
    switch (this->dataType) {
        case DataType::kFLOAT: {
            reflectionPad2dFunction<float>(
                (float*) inputs[0], (float*) outputs[0],
                N, C, H, W,
                paddingLeft, paddingRight, paddingTop, paddingBottom
            );
            break;
        }
        case DataType::kINT32: {
            reflectionPad2dFunction<int>(
                (int*) inputs[0], (int*) outputs[0],
                N, C, H, W,
                paddingLeft, paddingRight, paddingTop, paddingBottom
            );
            break;
        }
        case DataType::kHALF: {
            // TODO
            break;
        }
        case DataType::kINT8: {
            reflectionPad2dFunction<int8_t>(
                (int8_t*) inputs[0], (int8_t*) outputs[0],
                N, C, H, W,
                paddingLeft, paddingRight, paddingTop, paddingBottom
            );
            break;
        }
        default: {
            return 1;
        }
    }
    return 0;
};

size_t ReflectionPad2dPlugin::getSerializationSize() const noexcept {
    return 4 * sizeof(int32_t) + sizeof(DataType) + sizeof(Dims3);
};

void ReflectionPad2dPlugin::serialize(void* buffer) const noexcept {
    auto bufferInt = reinterpret_cast<int32_t*>(buffer);
    *bufferInt = this->paddingLeft;
    *(bufferInt + 1) = this->paddingRight;
    *(bufferInt + 2) = this->paddingTop;
    *(bufferInt + 3) = this->paddingBottom;
    auto bufferDtype = reinterpret_cast<DataType*>((char*) buffer + sizeof(int32_t) * 4);
    *(bufferDtype) = this->dataType;
    auto bufferDims = reinterpret_cast<Dims3*>((char*) buffer + sizeof(int32_t) * 4 + sizeof(DataType));
    *(bufferDims) = this->outputDims;
};

void ReflectionPad2dPlugin::destroy() noexcept {

};

IPluginV2Ext* ReflectionPad2dPlugin::clone() const noexcept { 
    auto plugin = new ReflectionPad2dPlugin(this->paddingLeft, this->paddingRight, this->paddingTop, this->paddingBottom);
    plugin->dataType = this->dataType;
    plugin->outputDims = this->outputDims;
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
};

void ReflectionPad2dPlugin::setPluginNamespace(AsciiChar const* pluginNamespace) noexcept {
    this->pluginNamespace = pluginNamespace;
};

AsciiChar const* ReflectionPad2dPlugin::getPluginNamespace() const noexcept {
    return this->pluginNamespace.c_str();
};

// IPluginV2Ext methods

DataType ReflectionPad2dPlugin::getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept {
    if (!nbInputs) {
        return DataType::kFLOAT;
    } else {
        return inputTypes[0];
    }
};

bool ReflectionPad2dPlugin::isOutputBroadcastAcrossBatch(int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept {
    return false; // TODO: optimize by enabling broadcast
}

bool ReflectionPad2dPlugin::canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept {
    return false;
}

void ReflectionPad2dPlugin::configurePlugin(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
        DataType const* inputTypes, DataType const* outputTypes, bool const* inputIsBroadcast,
        bool const* outputIsBroadcast, PluginFormat floatFormat, int32_t maxBatchSize) noexcept {
    this->dataType = inputTypes[0];
    this->outputDims = Dims3(
        outputDims->d[0],
        outputDims->d[1],
        outputDims->d[2]
    );
};

// PLUGIN CREATOR

ReflectionPad2dPluginCreator::ReflectionPad2dPluginCreator() {
    this->fields = {
        PluginField("paddingLeft"),
        PluginField("paddingRight"),
        PluginField("paddingTop"),
        PluginField("paddingBottom")
    };
    this->fieldCollection.fields = this->fields.data();
    this->fieldCollection.nbFields = this->fields.size();
}

AsciiChar const* ReflectionPad2dPluginCreator::getPluginName() const noexcept {
    return REFLECTION_PAD_2D_PLUGIN_NAME;
}

AsciiChar const* ReflectionPad2dPluginCreator::getPluginVersion() const noexcept {
    return REFLECTION_PAD_2D_PLUGIN_VERSION;
};

PluginFieldCollection const* ReflectionPad2dPluginCreator::getFieldNames() noexcept {
    return &this->fieldCollection;
};

IPluginV2* ReflectionPad2dPluginCreator::createPlugin(AsciiChar const* name, PluginFieldCollection const* fc) noexcept {
    auto plugin = new ReflectionPad2dPlugin(
        *((int32_t*) fc->fields[0].data),
        *((int32_t*) fc->fields[1].data),
        *((int32_t*) fc->fields[2].data),
        *((int32_t*) fc->fields[3].data)
    );
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

IPluginV2* ReflectionPad2dPluginCreator::deserializePlugin(AsciiChar const* name, void const* serialData, size_t serialLength) noexcept {
    int32_t const* buffer = reinterpret_cast<int32_t const*>(serialData);
    DataType const *bufferDtype = reinterpret_cast<DataType const*>((char*) serialData + 4 * sizeof(int32_t));
    Dims3 const * bufferDims = reinterpret_cast<Dims3 const *>((char*) serialData + 4 * sizeof(int32_t) + sizeof(DataType));
    auto plugin = new ReflectionPad2dPlugin(
        *(buffer),
        *(buffer + 1),
        *(buffer + 2),
        *(buffer + 3)
    );
    plugin->setPluginNamespace(getPluginNamespace());
    plugin->outputDims = *bufferDims;
    plugin->dataType = *bufferDtype;
    return plugin;
}

void ReflectionPad2dPluginCreator::setPluginNamespace(AsciiChar const* pluginNamespace) noexcept {
    this->pluginNamespace = pluginNamespace;
};

AsciiChar const* ReflectionPad2dPluginCreator::getPluginNamespace() const noexcept {
    return this->pluginNamespace.c_str();
};

REGISTER_TENSORRT_PLUGIN(ReflectionPad2dPluginCreator);

}