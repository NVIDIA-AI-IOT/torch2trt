#include <catch2/catch_all.hpp>
#include "example.h"


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

TEMPLATE_TEST_CASE("Example plugin enqueue test", "[example][template]" , int, float) {
    TestType *x_cpu;
    TestType *x_gpu;

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
    int size = count * sizeof(TestType);
    
    // allocate buffers
    x_cpu = (TestType *) malloc(size);
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

    for (int i = 0; i < count; i++) {
        x_cpu[i] = 4;
    }
}
