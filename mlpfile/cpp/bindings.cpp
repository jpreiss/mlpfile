#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "mlpfile.h"

namespace py = pybind11;

PYBIND11_MODULE(_mlpfile_bindings, m) {

    py::enum_<mlpfile::LayerType>(m, "LayerType")
        .value("Input", mlpfile::LayerType::Input)
        .value("Linear", mlpfile::LayerType::Linear)
        .value("ReLU", mlpfile::LayerType::ReLU)
        .export_values()
    ;

    py::class_<mlpfile::Layer> (m, "Layer")
        .def(py::init<>())
        .def_readonly("type", &mlpfile::Layer::type, "Layer type enum.")
        .def_readonly("W", &mlpfile::Layer::W, "Linear layer weight.")
        .def_readonly("b", &mlpfile::Layer::b, "Linear layer bias.")
        .def("__str__", &mlpfile::Layer::describe)
        .doc() = "A single layer in an MLP. Limited functionality."
    ;

    // TODO: It would be great if we can figure out how to make pybind11
    // generate individual-argument descriptions like one would get from
    // writing the `Args:` section in a Google-style docstring. As of now, it
    // seems like one can only get this for `pybind11:arg_v` where one must
    // also specify a default value.
    py::class_<mlpfile::Model> (m, "Model")
        .def(py::init<>())
        .def_static("load", &mlpfile::Model::load,
            "Load the model from a path.",
            py::arg("path"))
        .def_static("random", &mlpfile::Model::random,
            "Generate a randomly initialized model.")
        .def_readonly("layers", &mlpfile::Model::layers,
            "List of the Layer objects.")
        .def("input_dim", &mlpfile::Model::input_dim,
            "Input dimensionality.")
        .def("output_dim", &mlpfile::Model::output_dim,
            "Output dimensionality.")
        .def("forward", &mlpfile::Model::forward,
            "Computes the MLP's forward pass.",
            py::arg("input"))
        .def("jacobian", &mlpfile::Model::jacobian,
            "Computes the MLP's Jacobian.",
            py::arg("input"))
        .def("__str__", &mlpfile::Model::describe)
        .doc() = "Main class for loaded MLP models.";
}
