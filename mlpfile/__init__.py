"""mlpfile: Multilayer perceptron file format and evaluation."""

import tempfile
import ctypes
import subprocess

import numpy as np

from _mlpfile import Model, Layer, LayerType


__all__ = ["Model", "ModelCodegen", "Layer", "LayerType", "codegen", "cpp_dir"]


def cpp_dir():
    """Prints the directory containing the C++ header and source file.

    If you install the package via ``pip``, you can use the files in your build
    system. For example, a Makefile might contain::

        CXXFLAGS := -I$(shell python -c "__import__('mlpfile').cpp_dir()")
    """
    import os.path
    print(os.path.dirname(__file__) + "/cpp", end="")  # no newline


class ModelCodegen:
    """Does the codgen, compiles to DLL, then loads the DLL."""
    def __init__(self, model, eigen=True):
        with tempfile.TemporaryDirectory() as d:
            if eigen:
                src = d + "/src.cpp"
            else:
                src = d + "/src.c"
            obj = d + "/obj.o"
            lib = d + "/lib.so"
            with open(src, "w") as f:
                if eigen:
                    codegen_eigen(model, f)
                else:
                    codegen_c(model, f)
            if eigen:
                import eigenpip
                incl = "-I" + eigenpip.get_include()
                result_compile = subprocess.run(["c++", "-c", "-fPIC", "-O3", incl, "--std=c++11", "-o", obj, src])
            else:
                result_compile = subprocess.run(["cc", "-c", "-fPIC", "-O3", "-o", obj, src])
            assert result_compile.returncode == 0
            # TODO: Compile Eigen code without exceptions. We want to be able
            # to link into a C project.
            compiler = "c++" if eigen else "cc"
            result_shared = subprocess.run([compiler, "-shared", "-o", lib, obj])
            assert result_shared.returncode == 0
            self.library = ctypes.cdll.LoadLibrary(lib)

    def forward(self, xptr, yptr):
        self.library.forward(xptr, yptr)


def codegen_c(model, f):
    def array1d(x):
        return "{" + ", ".join([str(xi) for xi in x]) + "}"

    def array2d(X):
        return "{\n" + ",\n".join([array1d(row) for row in X]) + "\n}"

    # static data
    for i, layer in enumerate(model.layers):
        if layer.type == LayerType.Linear:
            rows, cols = layer.W.shape
            f.write(f"float W_{i}[{rows}][{cols}] = {array2d(layer.W)};\n")
            f.write(f"float b_{i}[{rows}] = {array1d(layer.b)};\n")

    # workspace
    linears = [layer for layer in model.layers if layer.type == LayerType.Linear]
    workdim = max(
        [lay.W.shape[0] for lay in linears]
        + [lay.W.shape[1] for lay in linears]
    )
    f.write(f"float work[2][{workdim}];\n")

    # code
    size = model.input_dim()
    f.write(f"void forward(float const *x, float *y) {{")
    f.write(
f"""
for (int i = 0; i < {size}; ++i) {{
    work[0][i] = x[i];
}}
""")
    src = 0
    dst = 1
    for ilayer, layer in enumerate(model.layers):
        if layer.type == LayerType.Input:
            continue
        elif layer.type == LayerType.Linear:
            newsize = layer.W.shape[0]
            f.write(
f"""
for (int i = 0; i < {newsize}; ++i) {{
    work[{dst}][i] = b_{ilayer}[i];
    for (int j = 0; j < {size}; ++j) {{
        work[{dst}][i] += W_{ilayer}[i][j] * work[{src}][j];
    }}
}}
""")
            size = newsize
        elif layer.type == LayerType.ReLU:
            # Avoid importing math.h just for fmaxf.
            f.write(
f"""
for (int i = 0; i < {size}; i++) {{
    work[{dst}][i] = (work[{src}][i] < 0.0f) ? 0.0f : work[{src}][i];
}}
""")
        else:
            raise ValueError("layer type:", layer.type)
        src, dst = dst, src
    f.write(
f"""
for (int i = 0; i < {size}; ++i) {{
    y[i] = work[{src}][i];
}}
}}
""")


def codegen_eigen(model, f):
    def array1d(x):
        return "{" + ", ".join([str(xi) for xi in x]) + "}"

    def array2d(X):
        return "{\n" + ",\n".join([array1d(row) for row in X]) + "\n}"

    f.write("#include <Eigen/Dense>\n")
    f.write('static_assert(EIGEN_HAS_CXX11, "we need C++11");\n')

    # static data
    for i, layer in enumerate(model.layers):
        if layer.type == LayerType.Linear:
            rows, cols = layer.W.shape
            f.write(f"float arr_W_{i}[{rows}][{cols}] = {array2d(layer.W)};\n")
            f.write(f"float arr_b_{i}[{rows}] = {array1d(layer.b)};\n")
            f.write(f"Eigen::Map<Eigen::Matrix<float, {rows}, {cols}, Eigen::RowMajor>> W_{i} (&arr_W_{i}[0][0]);\n")
            f.write(f"Eigen::Map<Eigen::Matrix<float, {rows}, 1>> b_{i} (arr_b_{i});\n")

    # workspace
    linears = [layer for layer in model.layers if layer.type == LayerType.Linear]
    workdim = max(
        [lay.W.shape[0] for lay in linears]
        + [lay.W.shape[1] for lay in linears]
    )
    f.write(f"Eigen::Matrix<float, {workdim}, 1> work[2];\n")

    # code
    size = model.input_dim()
    f.write('extern "C" { void forward(float const *xptr, float *yptr); }\n')
    f.write(f'void forward(float const *xptr, float *yptr) {{\n')
    f.write(f"Eigen::Map<Eigen::Matrix<float, {size}, 1> const> x(xptr);\n")
    f.write(f"work[0].head({size}) = x;\n")
    src = 0
    dst = 1
    for ilayer, layer in enumerate(model.layers):
        if layer.type == LayerType.Input:
            continue
        elif layer.type == LayerType.Linear:
            newsize = layer.W.shape[0]
            f.write(f"work[{dst}].head({newsize}) = W_{ilayer} * work[{src}].head({size}) + b_{ilayer};\n")
            size = newsize
        elif layer.type == LayerType.ReLU:
            f.write(f"work[{dst}].head({size}) = work[{src}].head({size}).array().max(0);\n")
        else:
            raise ValueError("layer type:", layer.type)
        src, dst = dst, src
    f.write(f"Eigen::Map<Eigen::Matrix<float, {model.output_dim()}, 1>> y(yptr);\n")
    f.write(f"y = work[{src}].head({size});\n")
    f.write("}")
