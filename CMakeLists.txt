cmake_minimum_required(VERSION 3.0.0)
project(torch2trt_plugins VERSION 0.1.0)

# VARIABLES
set(CUDA_ARCHITECTURES 53 62 72 87)

# BUILD PLUGINS LIBRARY
find_package(CUDA REQUIRED)

enable_language(CUDA)

include_directories("${CUDA_INCLUDE_DIRS}")

add_library(torch2trt_plugins SHARED
    plugins/src/example_plugin.cu
    plugins/src/reflection_pad_2d_plugin.cu
)
set_property(TARGET torch2trt_plugins PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})

target_link_libraries(
    torch2trt_plugins
    nvinfer
    ${CUDA_LIBRARIES}
)

install (TARGETS torch2trt_plugins
         LIBRARY DESTINATION lib)

# BUILD TESTS
find_package(Catch2 QUIET)

if(Catch2_FOUND)
    include(CTest)
    include(CPack)
    include(Catch)
    enable_testing()

    add_executable(torch2trt_plugins_test
        plugins/src/tests.cpp
        plugins/src/example_plugin_test.cpp
        plugins/src/reflection_pad_2d_plugin_test.cpp
    )

    set_property(TARGET torch2trt_plugins_test PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})

    target_link_libraries(torch2trt_plugins_test 
        PRIVATE 
        Catch2::Catch2WithMain 
        torch2trt_plugins
        nvinfer
        ${CUDA_LIBRARIES}
    )

    set(CPACK_PROJECT_NAME ${PROJECT_NAME})
    set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
    catch_discover_tests(torch2trt_plugins_test)
endif()