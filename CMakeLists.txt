cmake_minimum_required(VERSION 3.0.0)
project(torch2trt_plugins VERSION 0.1.0)


find_package(Catch2 REQUIRED)
find_package(CUDA  REQUIRED)

include(CTest)
enable_testing()
enable_language(CUDA)

include_directories("${CUDA_INCLUDE_DIRS}")

add_library(torch2trt_plugins 
    src/example.cu
)

add_executable(torch2trt_plugins_tests 
    src/tests.cpp
    src/example_test.cu
)

target_link_libraries(torch2trt_plugins_tests 
    PRIVATE 
    Catch2::Catch2WithMain 
    torch2trt_plugins
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
include(Catch)
catch_discover_tests(torch2trt_plugins_tests)