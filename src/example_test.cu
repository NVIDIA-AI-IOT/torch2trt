#include <catch2/catch_all.hpp>
#include "example.h"
#include <cuda_fp16.h>



using namespace torch2trt_plugins;


TEMPLATE_TEST_CASE("Example cuda test", "[example][template]" , int, float) {
    TestType *x_cpu;
    TestType *x_gpu;
    
    x_cpu = (TestType *) malloc(sizeof(TestType));
    cudaMalloc(&x_gpu, sizeof(TestType));
    *x_cpu = 2;
    cudaMemcpy(x_gpu, x_cpu, sizeof(TestType), cudaMemcpyHostToDevice);
    exampleFuncton<TestType>(x_gpu, x_gpu, 1);
    cudaMemcpy(x_cpu, x_gpu, sizeof(TestType), cudaMemcpyDeviceToHost);
    REQUIRE(*x_cpu == 4);
    cudaFree(x_gpu);
    free(x_cpu);
}


TEST_CASE("Example cuda test half", "[example]") {
    __half *x_cpu;
    __half *x_gpu;
    
    x_cpu = (__half *) malloc(sizeof(__half));
    cudaMalloc(&x_gpu, sizeof(__half));
    *x_cpu = __float2half_rn(2.0);
    cudaMemcpy(x_gpu, x_cpu, sizeof(__half), cudaMemcpyHostToDevice);
    exampleFunctonHalf(x_gpu, x_gpu, 1);
    cudaMemcpy(x_cpu, x_gpu, sizeof(__half), cudaMemcpyDeviceToHost);
    REQUIRE(__half2float(*x_cpu) == 4.0);
    cudaFree(x_gpu);
    free(x_cpu);
}


TEST_CASE("Example plugin test", "[example]") {
    auto plugin = ExamplePlugin();
    Dims3 inputDims(3, 4, 5);
    plugin.configureWithFormat(
        &inputDims,
        1,
        &inputDims,
        1,
        DataType::kINT32,
        PluginFormat::kCHW4,
        1
    );
    REQUIRE(plugin.inputSize == 3 * 4 * 5);
}


TEST_CASE("Example plugin enqueue test float", "[example]") {
    
    float *x_cpu;
    float *x_gpu;

    // create and configure plugin
    auto plugin = ExamplePlugin();
    Dims3 inputDims(3, 4, 5);
    plugin.configureWithFormat(
        &inputDims,
        1,
        &inputDims,
        1,
        DataType::kFLOAT,
        PluginFormat::kCHW4,
        1
    );

    // get sizes
    int count = 3 * 4 * 5;
    int size = count * sizeof(float);
    
    // allocate buffers
    x_cpu = (float *) malloc(size);
    cudaMalloc(&x_gpu, size);

    // populate host
    for (int i = 0; i < count; i++) {
        x_cpu[i] = 2;
    }
    
    // copy to device
    cudaMemcpy(x_gpu, x_cpu, size, cudaMemcpyHostToDevice);

    // execute plugin
    plugin.enqueue(1, (void**) &x_gpu, (void**) &x_gpu, (void*) nullptr, 0);

    // copy to host
    cudaMemcpy(x_cpu, x_gpu, size, cudaMemcpyDeviceToHost);
    REQUIRE(x_cpu[0] == 4);
    for (int i = 0; i < count; i++) {
        REQUIRE(x_cpu[i] == 4);
    }

    free(x_cpu);
    cudaFree(x_gpu);
}


TEST_CASE("Example plugin enqueue test int32", "[example]") {
    
    int *x_cpu;
    int *x_gpu;

    // create and configure plugin
    auto plugin = ExamplePlugin();
    Dims3 inputDims(3, 4, 5);
    plugin.configureWithFormat(
        &inputDims,
        1,
        &inputDims,
        1,
        DataType::kINT32,
        PluginFormat::kCHW4,
        1
    );

    // get sizes
    int count = 3 * 4 * 5;
    int size = count * sizeof(int);
    
    // allocate buffers
    x_cpu = (int *) malloc(size);
    cudaMalloc(&x_gpu, size);

    // populate host
    for (int i = 0; i < count; i++) {
        x_cpu[i] = 2;
    }
    
    // copy to device
    cudaMemcpy(x_gpu, x_cpu, size, cudaMemcpyHostToDevice);

    // execute plugin
    plugin.enqueue(1, (void**) &x_gpu, (void**) &x_gpu, (void*) nullptr, 0);

    // copy to host
    cudaMemcpy(x_cpu, x_gpu, size, cudaMemcpyDeviceToHost);
    REQUIRE(x_cpu[0] == 4);
    for (int i = 0; i < count; i++) {
        REQUIRE(x_cpu[i] == 4);
    }
    free(x_cpu);
    cudaFree(x_gpu);
}


TEST_CASE("Example plugin enqueue test int8", "[example]") {
    
    int8_t *x_cpu;
    int8_t *x_gpu;

    // create and configure plugin
    auto plugin = ExamplePlugin();
    Dims3 inputDims(3, 4, 5);
    plugin.configureWithFormat(
        &inputDims,
        1,
        &inputDims,
        1,
        DataType::kINT8,
        PluginFormat::kCHW4,
        1
    );

    // get sizes
    int count = 3 * 4 * 5;
    int size = count * sizeof(int8_t);
    
    // allocate buffers
    x_cpu = (int8_t *) malloc(size);
    cudaMalloc(&x_gpu, size);

    // populate host
    for (int i = 0; i < count; i++) {
        x_cpu[i] = 2;
    }
    
    // copy to device
    cudaMemcpy(x_gpu, x_cpu, size, cudaMemcpyHostToDevice);

    // execute plugin
    plugin.enqueue(1, (void**) &x_gpu, (void**) &x_gpu, (void*) nullptr, 0);

    // copy to host
    cudaMemcpy(x_cpu, x_gpu, size, cudaMemcpyDeviceToHost);
    REQUIRE(x_cpu[0] == 4);
    for (int i = 0; i < count; i++) {
        REQUIRE(x_cpu[i] == 4);
    }
    free(x_cpu);
    cudaFree(x_gpu);
}

TEST_CASE("Example plugin enqueue test half", "[example]") {
    __half *x_cpu;
    __half *x_gpu;

    // create and configure plugin
    auto plugin = ExamplePlugin();
    Dims3 inputDims(3, 4, 5);
    plugin.configureWithFormat(
        &inputDims,
        1,
        &inputDims,
        1,
        DataType::kHALF,
        PluginFormat::kCHW4,
        1
    );

    // get sizes
    int count = 3 * 4 * 5;
    int size = count * sizeof(__half);
    
    // allocate buffers
    x_cpu = (__half *) malloc(size);
    cudaMalloc(&x_gpu, size);

    // populate host
    for (int i = 0; i < count; i++) {
        x_cpu[i] = __float2half_rn(2.0);
    }
    
    // copy to device
    cudaMemcpy(x_gpu, x_cpu, size, cudaMemcpyHostToDevice);

    // execute plugin
    plugin.enqueue(1, (void**) &x_gpu, (void**) &x_gpu, (void*) nullptr, 0);

    // copy to host
    cudaMemcpy(x_cpu, x_gpu, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < count; i++) {
        REQUIRE(__half2float(x_cpu[i]) == 4);
    }
    free(x_cpu);
    cudaFree(x_gpu);
}


TEST_CASE("Example plugin creation", "[example]") {
    
    int8_t *x_cpu;
    int8_t *x_gpu;

    // create and configure plugin
    ExamplePluginCreator plugin_creator = ExamplePluginCreator();
    IPluginV2 *plugin = plugin_creator.createPlugin(nullptr, nullptr);
    Dims3 inputDims(3, 4, 5);
    plugin->configureWithFormat(
        &inputDims,
        1,
        &inputDims,
        1,
        DataType::kINT8,
        PluginFormat::kCHW4,
        1
    );

    // get sizes
    int count = 3 * 4 * 5;
    int size = count * sizeof(int8_t);
    
    // allocate buffers
    x_cpu = (int8_t *) malloc(size);
    cudaMalloc(&x_gpu, size);

    // populate host
    for (int i = 0; i < count; i++) {
        x_cpu[i] = 2;
    }
    
    // copy to device
    cudaMemcpy(x_gpu, x_cpu, size, cudaMemcpyHostToDevice);

    // execute plugin
    plugin->enqueue(1, (void**) &x_gpu, (void**) &x_gpu, (void*) nullptr, 0);

    // copy to host
    cudaMemcpy(x_cpu, x_gpu, size, cudaMemcpyDeviceToHost);
    REQUIRE(x_cpu[0] == 4);
    for (int i = 0; i < count; i++) {
        REQUIRE(x_cpu[i] == 4);
    }
    free(x_cpu);
    cudaFree(x_gpu);
}