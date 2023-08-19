#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "mlpfile.h"

namespace py = pybind11;

PYBIND11_MODULE(_mlpfile_bindings, m) {

    py::class_<mlpfile::Layer> (m, "Layer")
        .def(py::init<>())
        .def_readonly("type", &mlpfile::Layer::type)
        .def("__str__", &mlpfile::Layer::describe)
    ;

    py::class_<mlpfile::Model> (m, "Model")
        .def(py::init<>())
        .def_static("load", &mlpfile::Model::load)
        .def_readonly("layers", &mlpfile::Model::layers)
        .def("forward", &mlpfile::Model::forward)
        .def("jacobian", &mlpfile::Model::jacobian)
        .def("__str__", &mlpfile::Model::describe)
    ;
}
