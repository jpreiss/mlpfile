A simple file format and associated tools to save/load multilayer perceptrons
(aka fully-connected neural networks).

Features:
- Create the files in Python from a `torch.nn.Sequential`.
- Load the files in C++, or in Python via bindings.
- Evaluate the network and/or its Jacobian on an input.
- Perform a step of gradient descent (in place, for one datapoint, no momentum).
- C++ interface uses Eigen types.
- Generate fast allocation-free C or C++/Eigen code (faster) for the forward pass.
- Binary file I/O (no C++ dependency on Protobuf, etc.)

API docs: [https://jpreiss.github.io/mlpfile/api.html](https://jpreiss.github.io/mlpfile/api.html)

Installation
------------

To use the Python export and/or bindings, install the
[pip package](https://pypi.org/project/mlpfile/):

```pip install mlpfile```

If you only need to load and evaluate networks in C++, the easiest way is to
either 1) copy the files from `mlpfile/cpp` into your project, or 2) include
this repo as a submodule.


Example code
------------

**Python:**

```
model_torch = <train a torch.nn.Sequential somehow>
mlpfile.torch.write(model_torch, "net.mlp")

model_ours = mlpfile.Model.load("net.mlp")
x = <appropriate input>
y = model.forward(x)
```

**C++:**

```
mlpfile::Model model = mlpfile::Model::load("net.mlp");
Eigen::VectorXf x = <appropriate input>;
Eigen::VectorXf y = model.forward(x);
```

Performance
-----------

`mlpfile` is faster than popular alternatives for small networks on the CPU.
This is a very small example, but such small networks can appear in
time-sensitive realtime applications.

Test hardware is a 2021 MacBook Pro with Apple M1 Pro CPU.

`mlpfile` is over 3x faster than ONNX on both forward pass and Jacobian in this
test. TorchScript is surprisingly fast for the manually-computed Jacobian, but
is still slow for the forward pass. You can test on your own hardware by
running `benchmark.py`.

```
$ python benchmark.py

┌─────────────────┐
│ Model structure │
└─────────────────┘
mlpfile::Model with 5 Layers, 40 -> 10
Linear: 40 -> 100
ReLU
Linear: 100 -> 100
ReLU
Linear: 100 -> 10

┌─────────┐
│ Forward │
└─────────┘
        torch:   15.72 usec
  torchscript:    6.97 usec
         onnx:    5.98 usec
         ours:    1.91 usec
    codegen_c:   10.10 usec
codegen_eigen:    1.11 usec

┌──────────┐
│ Jacobian │
└──────────┘
    torch-autodiff:   88.19 usec
      torch-manual:   40.82 usec
torchscript-manual:   16.81 usec
              onnx:   42.00 usec
              ours:   11.97 usec

┌────────────┐
│ OGD-update │
└────────────┘
torch:  129.38 usec
 ours:   10.17 usec
```

Motivation
----------

The performance shown above is a major motivation, but besides that:

The typical choices for NN deployment from PyTorch to C++ (of which I am aware)
are TorchScript and the ONNX format. Both are heavyweight and complicated
because they are designed to handle general computation graphs like ResNets,
Transformers, etc. Their Python packages are easy to use via `pip`, but their
C++ packages aren't a part of standard package managers. Compiling from source
is very slow for ONNX-runtime; I have not tried TorchScript yet.

Intel and NVidia's ONNX loaders might be better, but they are not cross-platform.

ONNX-runtime also doesn't make it easy to extract the model weights from the
file. This means we can't (easily) use their file format and loader but compute
the neural network function ourselves for maximum speed.

Also, we want to evaluate the NN's Jacobian in our research application. It
turns out that PyTorch's `torch.func.jacrev` generates a computational graph
that can't be serialized with TorchScript or PyTorch's own ONNX exporter.
Therefore, we must write the symbolically-derived Jacobian by hand in PyTorch.
So that unwanted complexity must exist *somewhere*, whether it is C++ or
Python.


File format
-----------

It is a binary file format. All numerical types are little-endian, but the code
currently assumes it's running on a little-endian machine.

The file format is not stable!

```text
layer types enum:
    2 - linear
    3 - relu

header:
    number of layers (uint32)
    input dimension (uint32)

    for each layer:
        enum layer type (uint32)
        if linear:
            output dim (uint32)
        if relu:
            nothing

data:
    for each layer:
        if linear:
            weight (float32[], row-major)
            bias (float32[])
        otherwise:
            nothing
```
