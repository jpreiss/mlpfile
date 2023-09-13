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
    def __init__(self, model):
        with tempfile.TemporaryDirectory() as d:
            src = d + "/src.c"
            obj = d + "/obj.o"
            lib = d + "/lib.so"
            with open(src, "w") as f:
                codegen(model, f)
            result_compile = subprocess.run(["cc", "-c", "-fPIC", "-O3", "-o", obj, src])
            assert result_compile.returncode == 0
            result_shared = subprocess.run(["cc", "-shared", "-o", lib, obj])
            assert result_shared.returncode == 0
            self.library = ctypes.cdll.LoadLibrary(lib)

    def forward(self, x, ydst):
        # check that attr exists
        if x.dtype != np.float32 or ydst.dtype != np.float32:
            raise ValueError("Must call with float32 for max performance.")
        xptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        yptr = ydst.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self.library.forward(xptr, yptr)


def codegen(model, f):
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
