#include <torch/extension.h>
#include "interpolate.cpp"
#include "group_norm.cpp"


using namespace nvinfer1;

namespace torch2trt {
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        py::class_<InterpolatePlugin>(m, "InterpolatePlugin")
            .def(py::init<std::vector<int64_t>, std::string, bool>(), py::arg("size"), py::arg("mode"), py::arg("align_corners"))
            .def(py::init<const std::string &>(), py::arg("data"))
            .def("getSerializationSize", &InterpolatePlugin::getSerializationSize)
            .def("deserializeFromString", &InterpolatePlugin::deserializeFromString)
            .def("serializeToString", [](const InterpolatePlugin& plugin) {
                    std::string data = plugin.serializeToString();
                    return py::bytes(data);
                    });
        py::class_<GroupNormPlugin>(m, "GroupNormPlugin")
            .def(py::init<int64_t, at::Tensor, at::Tensor, double>(), py::arg("num_groups"), py::arg("weight"), py::arg("bias"), py::arg("eps"))
            .def(py::init<const std::string &>(), py::arg("data"))
            .def("getSerializationSize", &GroupNormPlugin::getSerializationSize)
            .def("deserializeFromString", &GroupNormPlugin::deserializeFromString)
            .def("serializeToString", [](const GroupNormPlugin& plugin) {
                    std::string data = plugin.serializeToString();
                    return py::bytes(data);
                    });

    }
} // namespace torch2trt
