cmake_minimum_required(VERSION 3.0.0)
project(torch2trt_plugins VERSION 0.1.0)

# VARIABLES
set(CUDA_ARCHITECTURES 53 62 72 87)

# BUILD PLUGINS LIBRARY
find_package(CUDA  REQUIRED)

enable_language(CUDA)


include_directories("${CUDA_INCLUDE_DIRS}")

add_library(torch2trt_plugins SHARED
    torch2trt/plugins/src/example_plugin/example.cu
    torch2trt/plugins/src/reflection_pad_2d_plugin/reflection_pad_2d_plugin.cu
)
set_property(TARGET torch2trt_plugins PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})

target_link_libraries(
    torch2trt_plugins
    nvinfer
)

install (TARGETS torch2trt_plugins
         LIBRARY DESTINATION lib)

# BUILD TESTS
find_package(Catch2 QUIET)

if(Catch2_FOUND)
    include(CTest)
    enable_testing()

    add_executable(torch2trt_plugins_tests 
        torch2trt/plugins/src/tests.cpp
        torch2trt/plugins/src/example_plugin/example_test.cu
        torch2trt/plugins/src/reflection_pad_2d_plugin/reflection_pad_2d_plugin_test.cu
    )

    set_property(TARGET torch2trt_plugins_tests PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})

    target_link_libraries(torch2trt_plugins_tests 
        PRIVATE 
        Catch2::Catch2WithMain 
        torch2trt_plugins
        nvinfer
    )

    set(CPACK_PROJECT_NAME ${PROJECT_NAME})
    set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
    include(CPack)
    include(Catch)
    catch_discover_tests(torch2trt_plugins_tests)
endif()