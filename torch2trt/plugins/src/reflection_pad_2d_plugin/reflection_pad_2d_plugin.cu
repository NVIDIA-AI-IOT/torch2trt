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
    int n = idx / (C*H*W);
    int c = idx / (H*W);
    int h = idx / (W);
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
    // iw = w - paddingLeft;
    // ih = h - paddingTop;

    y[n*C*H*W + c*H*W + h*W + w] = x[n*C*IH*IW + c*IH*IW + ih*IW + iw];
}

template __global__ void reflectionPad2dKernel<float>(float *, float *, int, int, int, int, int, int, int, int);
template __global__ void reflectionPad2dKernel<int>(int *, int *, int, int, int, int, int, int, int, int);
template __global__ void reflectionPad2dKernel<int8_t>(int8_t *, int8_t *, int, int, int, int, int, int, int, int);


// __global__ void exampleKernelHalf(__half *x, __half *y, float scale, int size) {
//     int index = blockDim.x * blockIdx.x + threadIdx.x;
//     if (index < size) {
//         y[index] = __hmul(x[index], __float2half_rn(scale));
//     }
// }


// // FUNCTIONS


template<typename T>
void reflectionPad2dFunction(
    T *x, T *y, 
    int N, int C, int H, int W, 
    int paddingLeft, int paddingRight, int paddingTop, int paddingBottom, 
    cudaStream_t stream) {
    int size = N * C * H * W;
    int nThreads = 32;
    int nBlocks = (size / 32) + 1;
    reflectionPad2dKernel<<<nBlocks, nThreads, 0, stream>>>(x, y, N, C, H, W, paddingLeft, paddingRight, paddingTop, paddingBottom);
}

template void reflectionPad2dFunction<float>(
    float *, float *, 
    int, int, int, int, 
    int, int, int, int, 
    cudaStream_t);

// template void exampleFuncton<float>(float *, float *, float, int, cudaStream_t);
// template void exampleFuncton<int>(int *, int *, float, int, cudaStream_t);
// template void exampleFuncton<int8_t>(int8_t *, int8_t *, float, int, cudaStream_t);

// void exampleFunctonHalf(__half *x, __half *y, float scale, int size, cudaStream_t stream) {
//     int nThreads = 32;
//     int nBlocks = (size / 32) + 1;
//     exampleKernelHalf<<<nBlocks, nThreads, 0, stream>>>(x, y, scale, size);
// }


// PLUGIN


ReflectionPad2dPlugin::ReflectionPad2dPlugin(int32_t paddingLeft, int32_t paddingRight, int32_t paddingTop, int32_t paddingBottom) : 
paddingLeft(paddingLeft), paddingRight(paddingRight), paddingTop(paddingTop), paddingBottom(paddingBottom) {

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

void ReflectionPad2dPlugin::configureWithFormat(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
    DataType type, PluginFormat format, int32_t maxBatchSize) noexcept {
    Dims d = outputDims[0];
    this->outputSize = 1;
    for (int i = 0; i < d.nbDims; i++) {
        this->outputSize *= d.d[i];
    }
    this->dataType = type;
    this->outputDims = Dims3(
        outputDims->d[0],
        outputDims->d[1],
        outputDims->d[2]
    );
};

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
    const int totalSize = batchSize * this->outputSize;
    int N = batchSize;
    int C = outputDims.d[0];
    int H = outputDims.d[1];
    int W = outputDims.d[2];
    int nThreads = 32;
    int nBlocks = (totalSize / 32) + 1;
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
            // exampleFunctonHalf((__half*) inputs[0], (__half*) outputs[0], this->scale, totalSize, stream);
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
    return 4 * sizeof(int32_t);
};

void ReflectionPad2dPlugin::serialize(void* buffer) const noexcept {
    auto bufferInt = reinterpret_cast<int32_t*>(buffer);
    *bufferInt = this->paddingLeft;
    *(bufferInt + 1) = this->paddingRight;
    *(bufferInt + 2) = this->paddingTop;
    *(bufferInt + 3) = this->paddingBottom;
};

void ReflectionPad2dPlugin::destroy() noexcept {

};

IPluginV2* ReflectionPad2dPlugin::clone() const noexcept { 
    return new ReflectionPad2dPlugin(this->paddingLeft, this->paddingRight, this->paddingTop, this->paddingBottom);
};

void ReflectionPad2dPlugin::setPluginNamespace(AsciiChar const* pluginNamespace) noexcept {
    this->pluginNamespace = pluginNamespace;
};

AsciiChar const* ReflectionPad2dPlugin::getPluginNamespace() const noexcept {
    return this->pluginNamespace.c_str();
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
    return new ReflectionPad2dPlugin(
        *((int32_t*) fc->fields[0].data),
        *((int32_t*) fc->fields[1].data),
        *((int32_t*) fc->fields[2].data),
        *((int32_t*) fc->fields[3].data)
    );
}

IPluginV2* ReflectionPad2dPluginCreator::deserializePlugin(AsciiChar const* name, void const* serialData, size_t serialLength) noexcept {
    int32_t const* buffer = reinterpret_cast<int32_t const*>(serialData);
    return new ReflectionPad2dPlugin(
        *(buffer),
        *(buffer + 1),
        *(buffer + 2),
        *(buffer + 3)
    );
}

void ReflectionPad2dPluginCreator::setPluginNamespace(AsciiChar const* pluginNamespace) noexcept {
    this->pluginNamespace = pluginNamespace;
};

AsciiChar const* ReflectionPad2dPluginCreator::getPluginNamespace() const noexcept {
    return this->pluginNamespace.c_str();
};

REGISTER_TENSORRT_PLUGIN(ReflectionPad2dPluginCreator);

}