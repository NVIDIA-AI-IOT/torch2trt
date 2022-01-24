#include <catch2/catch_all.hpp>
#include "cuda_runtime.h"
#include "reflection_pad_2d_plugin.h"


TEMPLATE_TEST_CASE("Reflection pad 2d kernel", "[reflection_pad_2d_plugin][template]" , float) {
    float *x;
    float *y;
    
    REQUIRE(true);
}