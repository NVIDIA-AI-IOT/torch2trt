#include <catch2/catch_all.hpp>
#include "example.h"


using namespace torch2trt_plugins;


TEMPLATE_TEST_CASE("Example cuda test", "[example][template]" , int) {
    TestType x_cpu;
    TestType *x_gpu;
    
    cudaMalloc(&x_gpu, sizeof(int));
    x_cpu = 2;
    cudaMemcpy(x_gpu, &x_cpu, sizeof(int), cudaMemcpyHostToDevice);
    exampleFuncton<TestType>(x_gpu, 1);
    cudaMemcpy(&x_cpu, x_gpu, sizeof(int), cudaMemcpyDeviceToHost);
    REQUIRE(x_cpu == 4);
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