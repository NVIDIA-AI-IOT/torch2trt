#include "example.h"


template<typename T>
__global__ void cuda_double_kernel(T *x) {
    *x = (*x) * 2;
}

template __global__ void cuda_double_kernel<float>(float *);
template __global__ void cuda_double_kernel<int>(int *);


template<typename T>
void cuda_double(T *x) {
    cuda_double_kernel<<<1, 1>>>(x);
}

template void cuda_double<float>(float *);
template void cuda_double<int>(int *);