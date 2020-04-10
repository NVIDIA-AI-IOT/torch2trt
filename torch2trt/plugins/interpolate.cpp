#include <torch/extension.h>
#include <torch/script.h>
#include <iostream>
#include <string>
#include <sstream>


class DummyPlugin {
private:
    torch::jit::script::Module params_container;
public:
    InterpolatePlugin() {}
    
    void put_serialized(const char *data, size_t length) {
        std::string data_str(data, length);
        std::istringstream data_stream(data_str);
        params_container = torch::jit::load(data_stream);
        
        for (auto a : params_container.named_attributes()) {
            if (a.name == "d") {
                std::cout << a.name << ': ' << a.value << std::endl;
                a.value = 99;
                std::cout << a.name << ': ' << a.value << std::endl;
            }
        }
    }
    
    size_t get_serialized_size() {
        auto str = get_serialized();
        return str.size();
    }
    
    std::string get_serialized() {
        std::ostringstream data_str;
        params_container.save(data_str);
        return data_str.str();
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<InterpolatePlugin>(m, "InterpolatePlugin")
        .def(py::init<>())
        .def("put_serialized", &InterpolatePlugin::put_serialized)
        .def("get_serialized_size", &InterpolatePlugin::get_serialized_size);
}