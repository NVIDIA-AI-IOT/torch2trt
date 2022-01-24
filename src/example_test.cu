#include <catch2/catch_all.hpp>
#include "cuda_runtime.h"
#include "example.h"


TEMPLATE_TEST_CASE("Example cuda test", "[example][template]" , int) {
    TestType x_cpu;
    TestType *x_gpu;
    
    cudaMalloc(&x_gpu, sizeof(int));
    x_cpu = 2;
    cudaMemcpy(x_gpu, &x_cpu, sizeof(int), cudaMemcpyHostToDevice);
    cuda_double<TestType>(x_gpu);
    cudaMemcpy(&x_cpu, x_gpu, sizeof(int), cudaMemcpyDeviceToHost);
    REQUIRE(x_cpu == 4);
}