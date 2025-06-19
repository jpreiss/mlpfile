Simple and fast library for working with small multilayer perceptrons (aka fully-connected neural networks) on the CPU. Python/NumPy bindings and C++/Eigen core.

Features:
- Evaluate the forward pass faster than alternatives.
- Compute derivatives of the network's output with respect to its input or parameters.
- Take a step of online gradient descent in place.
- Generate fast allocation-free C or C++/Eigen code for the forward pass.
- Convert from a `torch.nn.Sequential` to our file format.
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
Small networks can appear in time-sensitive realtime applications.

On a 2021 MacBook Pro M1, `mlpfile` is over 3x faster than ONNX and TorchScript
on the forward pass. You can test on your own hardware by running
`benchmark.py`.

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

The performance shown above is the #1 motivation. Aside from that:

Popular tools for NN deployment from PyTorch to C++ are TorchScript and ONNX.
Both are heavyweight because they handle general computation graphs like
ResNets and Transformers. Their C++ packages are hard to compile, and some are
platform-specific. Their file formats are complicated to parse manually.

To take the NN's Jacobian with respect to its input, PyTorch's
`torch.func.jacrev` generates a computation graph that can't be serialized with
TorchScript or PyTorch's own ONNX exporter (circa 2023).


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
