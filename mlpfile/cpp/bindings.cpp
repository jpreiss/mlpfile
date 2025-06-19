#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "mlpfile.h"

namespace py = pybind11;

PYBIND11_MODULE(_mlpfile, m) {

    py::enum_<mlpfile::LayerType>(m, "LayerType")
        .value("Linear", mlpfile::LayerType::Linear)
        .value("ReLU", mlpfile::LayerType::ReLU)
        .export_values()
    ;

    py::class_<mlpfile::Layer> (m, "Layer")
        .def(py::init<>())
        .def_readwrite("type", &mlpfile::Layer::type, "Layer type enum.")
        .def_readwrite("W", &mlpfile::Layer::W, "Linear layer weight.")
        .def_readwrite("b", &mlpfile::Layer::b, "Linear layer bias.")
        .def("__str__", &mlpfile::Layer::describe)
        .doc() = "A single layer in an MLP. Limited functionality."
    ;

    py::class_<mlpfile::LayerJacobian> (m, "LayerJacobian")
        .def(py::init<>())
        .def_readonly("dW", &mlpfile::LayerJacobian::dW,
            "Jacobian w.r.t. weights. Dimension is output_dim() x (W.rows * W.cols). Each row is stored row-major, so reshaping a row into a W-shaped ndarray is compatible with W.")
        .def_readonly("db", &mlpfile::LayerJacobian::db, "Jacobian w.r.t. biases.")
        .doc() = "Jacobian of network output w.r.t. a layer's parameters."
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
        .def_readwrite("layers", &mlpfile::Model::layers,
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
        .def("jacobian_params", &mlpfile::Model::jacobian_params,
            "Computes the MLP's Jacobian w.r.t. parameters.",
            py::arg("input"))
        .def("grad_update", &mlpfile::Model::grad_update,
            "Performs one step of gradient descent for one data point.",
            py::arg("x"), py::arg("y"), py::arg("loss"), py::arg("rate"))
        .def("__str__", &mlpfile::Model::describe)
        .def("__copy__",  [](mlpfile::Model const &self) {
            return mlpfile::Model(self);
        })
        .def("__deepcopy__", [](mlpfile::Model const &self, py::dict) {
            return mlpfile::Model(self);
        }, "memo")
        .doc() = "Main class for loaded MLP models.";

    m.def("squared_error", &mlpfile::squared_error);
    m.def("softmax_cross_entropy", &mlpfile::softmax_cross_entropy);
}
