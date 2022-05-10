#include <catch2/catch_all.hpp>
#include "example_plugin.h"
#include <cuda_fp16.h>
#include "NvInfer.h"
#include <iostream>


using namespace torch2trt_plugins;


TEMPLATE_TEST_CASE("Example cuda test", "[example][template]" , int, float) {
    TestType *x_cpu;
    TestType *x_gpu;
    
    x_cpu = (TestType *) malloc(sizeof(TestType));
    cudaMalloc((void**)&x_gpu, sizeof(TestType));
    *x_cpu = 2;
    cudaMemcpy(x_gpu, x_cpu, sizeof(TestType), cudaMemcpyHostToDevice);
    exampleFuncton<TestType>(x_gpu, x_gpu, 2.0, 1);
    cudaMemcpy(x_cpu, x_gpu, sizeof(TestType), cudaMemcpyDeviceToHost);
    REQUIRE(*x_cpu == 4);
    cudaFree(x_gpu);
    free(x_cpu);
}


TEST_CASE("Example cuda test half", "[example]") {
    __half *x_cpu;
    __half *x_gpu;
    
    x_cpu = (__half *) malloc(sizeof(__half));
    cudaMalloc((void**)&x_gpu, sizeof(__half));
    *x_cpu = __float2half_rn(2.0);
    cudaMemcpy(x_gpu, x_cpu, sizeof(__half), cudaMemcpyHostToDevice);
    exampleFunctonHalf(x_gpu, x_gpu, 2.0, 1);
    cudaMemcpy(x_cpu, x_gpu, sizeof(__half), cudaMemcpyDeviceToHost);
    REQUIRE(__half2float(*x_cpu) == 4.0);
    cudaFree(x_gpu);
    free(x_cpu);
}


TEST_CASE("Example plugin test", "[example]") {
    auto plugin = ExamplePlugin();
    Dims3 inputDims(3, 4, 5);
    Dims3 outputDims(3, 4, 5);
    DataType inputTypes = DataType::kINT32;
    DataType outputTypes = DataType::kINT32;
    bool inputIsBroadcast = false;
    bool outputIsBroadcast = false;
    plugin.configurePlugin(
        &inputDims, 1, 
        &outputDims, 1, 
        &inputTypes, 
        &outputTypes, 
        &inputIsBroadcast,
        &outputIsBroadcast,
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
    Dims3 outputDims(3, 4, 5);
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
        PluginFormat::kCHW4, 
        1
    );

    // get sizes
    int count = 3 * 4 * 5;
    int size = count * sizeof(float);
    
    // allocate buffers
    x_cpu = (float *) malloc(size);
    cudaMalloc((void**)&x_gpu, size);

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
    Dims3 outputDims(3, 4, 5);
    DataType inputTypes = DataType::kINT32;
    DataType outputTypes = DataType::kINT32;
    bool inputIsBroadcast = false;
    bool outputIsBroadcast = false;
    plugin.configurePlugin(
        &inputDims, 1, 
        &outputDims, 1, 
        &inputTypes, 
        &outputTypes, 
        &inputIsBroadcast,
        &outputIsBroadcast,
        PluginFormat::kCHW4, 
        1
    );

    // get sizes
    int count = 3 * 4 * 5;
    int size = count * sizeof(int);
    
    // allocate buffers
    x_cpu = (int *) malloc(size);
    cudaMalloc((void**)&x_gpu, size);

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
    Dims3 outputDims(3, 4, 5);
    DataType inputTypes = DataType::kINT8;
    DataType outputTypes = DataType::kINT8;
    bool inputIsBroadcast = false;
    bool outputIsBroadcast = false;
    plugin.configurePlugin(
        &inputDims, 1, 
        &outputDims, 1, 
        &inputTypes, 
        &outputTypes, 
        &inputIsBroadcast,
        &outputIsBroadcast,
        PluginFormat::kCHW4, 
        1
    );

    // get sizes
    int count = 3 * 4 * 5;
    int size = count * sizeof(int8_t);
    
    // allocate buffers
    x_cpu = (int8_t *) malloc(size);
    cudaMalloc((void**)&x_gpu, size);

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
    Dims3 outputDims(3, 4, 5);
    DataType inputTypes = DataType::kHALF;
    DataType outputTypes = DataType::kHALF;
    bool inputIsBroadcast = false;
    bool outputIsBroadcast = false;
    plugin.configurePlugin(
        &inputDims, 1, 
        &outputDims, 1, 
        &inputTypes, 
        &outputTypes, 
        &inputIsBroadcast,
        &outputIsBroadcast,
        PluginFormat::kCHW4, 
        1
    );

    // get sizes
    int count = 3 * 4 * 5;
    int size = count * sizeof(__half);
    
    // allocate buffers
    x_cpu = (__half *) malloc(size);
    cudaMalloc((void**)&x_gpu, size);

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
    IPluginV2Ext *plugin = (IPluginV2Ext*) plugin_creator.createPlugin(nullptr, nullptr);
    Dims3 inputDims(3, 4, 5);
    Dims3 outputDims(3, 4, 5);
    DataType inputTypes = DataType::kINT8;
    DataType outputTypes = DataType::kINT8;
    bool inputIsBroadcast = false;
    bool outputIsBroadcast = false;
    plugin->configurePlugin(
        &inputDims, 1, 
        &outputDims, 1, 
        &inputTypes, 
        &outputTypes, 
        &inputIsBroadcast,
        &outputIsBroadcast,
        PluginFormat::kCHW4, 
        1
    );

    // get sizes
    int count = 3 * 4 * 5;
    int size = count * sizeof(int8_t);
    
    // allocate buffers
    x_cpu = (int8_t *) malloc(size);
    cudaMalloc((void**)&x_gpu, size);

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

class Logger : public ILogger
{
  void log(Severity severity, AsciiChar const* msg) noexcept
  {
      std::cout << msg << std::endl;
  }
} gLogger;


TEST_CASE("Example engine creation", "[example]") {
    
    float *x_cpu;
    float *x_gpu;
    Dims3 inputDims(3, 4, 5);

    // create and configure plugin
    auto builder = createInferBuilder(gLogger);
    auto network_flags = NetworkDefinitionCreationFlags();
    auto network = builder->createNetworkV2(network_flags);
    auto input = network->addInput("input", DataType::kFLOAT, inputDims);
    auto plugin_creator = ExamplePluginCreator();
    auto plugin = plugin_creator.createPlugin(nullptr, nullptr);
    // auto plugin = ExamplePlugin();
    auto layer = network->addPluginV2(
        &input,
        1,
        *plugin
    );
    network->markOutput(*layer->getOutput(0));
    layer->getOutput(0)->setName("output");

    // auto builder_config = Builder
    auto builder_config = builder->createBuilderConfig();
    auto engine = builder->buildEngineWithConfig(
        *network,
        *builder_config
    );
    
    auto context = engine->createExecutionContext();

    // get sizes
    int count = 3 * 4 * 5;
    int size = count * sizeof(float);
    
    // allocate buffers
    x_cpu = (float *) malloc(size);
    cudaMalloc((void**)&x_gpu, size);

    // populate host
    for (int i = 0; i < count; i++) {
        x_cpu[i] = 2;
    }
    
    // copy to device
    cudaMemcpy(x_gpu, x_cpu, size, cudaMemcpyHostToDevice);

    // execute plugin
    void *bindings[2];
    bindings[engine->getBindingIndex("input")] = x_gpu;
    bindings[engine->getBindingIndex("output")] = x_gpu;

    context->enqueue(1, bindings, 0, nullptr);

    // plugin->enqueue(1, (void**) &x_gpu, (void**) &x_gpu, (void*) nullptr, 0);

    // copy to host
    cudaMemcpy(x_cpu, x_gpu, size, cudaMemcpyDeviceToHost);
    REQUIRE(x_cpu[0] == 4);
    for (int i = 0; i < count; i++) {
        REQUIRE(x_cpu[i] == 4);
    }

    free(x_cpu);
    cudaFree(x_gpu);
}

TEST_CASE("Example plugin creator field names includes scale", "[example]") {
    auto pluginCreator = ExamplePluginCreator();
    REQUIRE(pluginCreator.getFieldNames()->nbFields == 1);
    REQUIRE(strcmp(pluginCreator.getFieldNames()->fields[0].name, "scale") == 0);
}


TEST_CASE("Create example plugin from fields", "[example]") {
    auto fieldCollection = PluginFieldCollection();
    float scale = 3;
    std::vector<PluginField> fields = {
        PluginField("scale", (void*) &scale, PluginFieldType::kFLOAT32, 1)
    };
    fieldCollection.nbFields = fields.size();
    fieldCollection.fields = fields.data();
    auto pluginCreator = ExamplePluginCreator();
    auto plugin = pluginCreator.createPlugin("ExamplePlugin", &fieldCollection);

    REQUIRE(((ExamplePlugin*)plugin)->scale == scale);
}

TEST_CASE("Create example plugin serialize/deserialize", "[example]") {
    auto fieldCollection = PluginFieldCollection();
    float scale = 3;
    std::vector<PluginField> fields = {
        PluginField("scale", (void*) &scale, PluginFieldType::kFLOAT32, 1)
    };
    fieldCollection.nbFields = fields.size();
    fieldCollection.fields = fields.data();
    auto pluginCreator = ExamplePluginCreator();
    auto plugin = pluginCreator.createPlugin("ExamplePlugin", &fieldCollection);

    void *buffer = malloc(sizeof(float));
    plugin->serialize(buffer);
    auto plugin2 = pluginCreator.deserializePlugin("ExamplePlugin", buffer, sizeof(float));

    REQUIRE(((ExamplePlugin*)plugin2)->scale == scale);
    free(buffer);
}