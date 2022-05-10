#include <catch2/catch_all.hpp>
#include "reflection_pad_2d_plugin.h"
#include <cuda_fp16.h>
#include "NvInfer.h"
#include <iostream>


using namespace torch2trt_plugins;


TEMPLATE_TEST_CASE("Test reflection pad kernel", "[ReflectionPad2d][template]" , float) {
    TestType x_cpu[9] = {
        0, 1, 2,
        3, 4, 5,
        6, 7, 8
    };
    TestType y_cpu[25];
    TestType *x_gpu;
    TestType *y_gpu;
    TestType y_cpu_gt[25] = {
        4, 3, 4, 5, 4,
        1, 0, 1, 2, 1,
        4, 3, 4, 5, 4,
        7, 6, 7, 8, 7,
        4, 3, 4, 5, 4
    };
    
    // y_cpu = (TestType*) malloc(16 * sizeof(TestType));
    cudaMalloc((void**)&x_gpu, 9 * sizeof(TestType));
    cudaMalloc((void**)&y_gpu, 25 * sizeof(TestType));
    cudaMemcpy(x_gpu, x_cpu, 9 * sizeof(TestType), cudaMemcpyHostToDevice);

    reflectionPad2dFunction<TestType>(x_gpu, y_gpu, 
        1, 1, 5, 5,
        1, 1, 1, 1);

    cudaMemcpy(y_cpu, y_gpu, 25 * sizeof(TestType), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 25; i++) {
        REQUIRE(y_cpu[i] == y_cpu_gt[i]);
    }
    cudaFree(x_gpu);
    cudaFree(y_gpu);
}

TEMPLATE_TEST_CASE("Test reflection pad plugin enqueue", "[ReflectionPad2d][template]" , float) {
    TestType x_cpu[9] = {
        0, 1, 2,
        3, 4, 5,
        6, 7, 8
    };
    TestType y_cpu[25];
    TestType *x_gpu;
    TestType *y_gpu;
    TestType y_cpu_gt[25] = {
        4, 3, 4, 5, 4,
        1, 0, 1, 2, 1,
        4, 3, 4, 5, 4,
        7, 6, 7, 8, 7,
        4, 3, 4, 5, 4
    };
    
    // y_cpu = (TestType*) malloc(16 * sizeof(TestType));
    cudaMalloc((void**)&x_gpu, 9 * sizeof(TestType));
    cudaMalloc((void**)&y_gpu, 25 * sizeof(TestType));
    cudaMemcpy(x_gpu, x_cpu, 9 * sizeof(TestType), cudaMemcpyHostToDevice);

    auto plugin = ReflectionPad2dPlugin(1, 1, 1, 1);
    Dims3 inputDims(1, 3, 3);
    Dims3 outputDims(1, 5, 5);
    DataType inputTypes = DataType::kFLOAT;
    DataType outputTypes = DataType::kFLOAT;
    bool inputIsBroadcast = false;
    bool outputIsBroadcast = false;
    plugin.configurePlugin(
        &inputDims, 1, 
        &outputDims, 1, 
        &inputTypes, 
        &outputTypes, 
        &inputIsBroadcast,
        &outputIsBroadcast,
        PluginFormat::kLINEAR, 
        1
    );
    
    void *inputs[] = {(void*)x_gpu};
    void *outputs[] = {(void*)y_gpu};
    plugin.enqueue(1, inputs, outputs, nullptr, 0);
    
    cudaMemcpy(y_cpu, y_gpu, 25 * sizeof(TestType), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 25; i++) {
        REQUIRE(y_cpu[i] == y_cpu_gt[i]);
    }
    cudaFree(x_gpu);
    cudaFree(y_gpu);
}

TEMPLATE_TEST_CASE("Test reflection pad plugin enqueue 2 channels", "[ReflectionPad2d][template]" , float) {
    TestType x_cpu[9*2] = {
        0, 1, 2,
        3, 4, 5,
        6, 7, 8,
        0, 1, 2,
        3, 4, 5,
        6, 7, 8
    };
    TestType y_cpu[25*2];
    TestType *x_gpu;
    TestType *y_gpu;
    TestType y_cpu_gt[25*2] = {
        4, 3, 4, 5, 4,
        1, 0, 1, 2, 1,
        4, 3, 4, 5, 4,
        7, 6, 7, 8, 7,
        4, 3, 4, 5, 4,
        4, 3, 4, 5, 4,
        1, 0, 1, 2, 1,
        4, 3, 4, 5, 4,
        7, 6, 7, 8, 7,
        4, 3, 4, 5, 4
    };
    
    // y_cpu = (TestType*) malloc(16 * sizeof(TestType));
    cudaMalloc((void**)&x_gpu, 2*9 * sizeof(TestType));
    cudaMalloc((void**)&y_gpu, 2*25 * sizeof(TestType));
    cudaMemcpy(x_gpu, x_cpu, 2*9 * sizeof(TestType), cudaMemcpyHostToDevice);

    auto plugin = ReflectionPad2dPlugin(1, 1, 1, 1);
    Dims3 inputDims(2, 3, 3);
    Dims3 outputDims(2, 5, 5);
    DataType inputTypes = DataType::kFLOAT;
    DataType outputTypes = DataType::kFLOAT;
    bool inputIsBroadcast = false;
    bool outputIsBroadcast = false;
    plugin.configurePlugin(
        &inputDims, 1, 
        &outputDims, 1, 
        &inputTypes, 
        &outputTypes, 
        &inputIsBroadcast,
        &outputIsBroadcast,
        PluginFormat::kLINEAR, 
        1
    );

    void *inputs[] = {(void*)x_gpu};
    void *outputs[] = {(void*)y_gpu};
    plugin.enqueue(1, inputs, outputs, nullptr, 0);
    
    cudaMemcpy(y_cpu, y_gpu, 2*25 * sizeof(TestType), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 2*25; i++) {
        REQUIRE(y_cpu[i] == y_cpu_gt[i]);
    }
    cudaFree(x_gpu);
    cudaFree(y_gpu);
}
