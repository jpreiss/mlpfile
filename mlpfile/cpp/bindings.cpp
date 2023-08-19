#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "mlpfile.h"

namespace py = pybind11;

PYBIND11_MODULE(_mlpfile_bindings, m) {

    py::class_<mlpfile::Layer> (m, "Layer")
        .def(py::init<>())
        .def_readonly("type", &mlpfile::Layer::type, "Layer type enum.")
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
        .def_readonly("layers", &mlpfile::Model::layers,
            "List of the Layer objects.")
        .def("forward", &mlpfile::Model::forward,
            "Computes the MLP's forward pass.",
            py::arg("input"))
        .def("jacobian", &mlpfile::Model::jacobian,
            "Computes the MLP's Jacobian.",
            py::arg("input"))
        .def("__str__", &mlpfile::Model::describe)
        .doc() = "Main class for loaded MLP models.";
}
