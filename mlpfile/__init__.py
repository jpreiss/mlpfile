"""mlpfile: Multilayer perceptron file format and evaluation."""

import ctypes
import os
import subprocess
import tempfile
import warnings

from _mlpfile import Model, Layer, LayerType
from ._codegen import codegen_c, codegen_eigen


__all__ = ["Model", "Layer", "LayerType", "codegen", "cpp_dir"]


def cpp_dir():
    """Prints the directory containing the C++ header and source file.

    If you install the package via ``pip``, you can use the files in your build
    system. For example, a Makefile might contain::

        CXXFLAGS := -I$(shell python -c "__import__('mlpfile').cpp_dir()")
    """
    import os.path
    print(os.path.dirname(__file__) + "/cpp", end="")  # no newline


def codegen(model, outdir=None, eigen=True, compile=False):
    """Generates and optionally compiles C/C++ code to evaluate the model's
    forward pass.

    The generated code has the function::

        void forward(float const *x, float *y)

    Note that size arguments are not needed because they are known at compile time.

    If ``compile=True``, also compiles the code into a shared library and loads
    that library with ``ctypes``. Calling from Python+NumPy looks like::

        lib = mlpfile.codegen(model, compile=True)
        x = <my input of length model.input_dim()>.astype(np.float32)
        y = np.zeros(model.output_dim(), dtype=np.float32)
        xptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        yptr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        lib.forward(xptr, yptr)
        # Now y has the output.

    Even if you don't want to use the compiled code from Python,
    ``compile=True`` can be helpful because it knows the correct compiler
    flags. This is especially important when ``eigen=True``.

    To compile the Eigen code yourself, these flags ensure the code works in a
    malloc-free (embedded/realtime) environment and can be linked as C::

        -fno-exceptions -DEIGEN_NO_MALLOC

    Args:
        model (Model): Model.
        outdir (str, optional): If specified, writes header and code to this
            directory. If ``compile=True``, also stores object file and shared
            library. If ``outdir=None``, uses a temporary directory deleted
            immediately after call. Note: ``outdir=None`` only makes sense when
            ``compile=True``.
        eigen (bool): Use Eigen internally. If true, we generate code that uses
            the Eigen C++ library. Eigen's matrix-vector multiply is much
            faster than our hand-written naive one when ``eigen=False``.
            Generates a function with ``extern "C"`` linkage, no dynamic
            memory, and no exceptions. Therefore, the compiled object/library
            can be used in C projects, with a C linker, even though it was
            compiled by a C++ compiler.
        compile (bool): Compiles the code into a shared library and loads it
            with ``ctypes``. If ``outdir=None``, the returned ``ctypes.CDLL``
            will be loaded before the temporary dir is deleted, so you can
            still use the library for the remainder of the Python process.

    Returns:
        If ``compile=True``, returns a ``ctypes.CDLL`` library with function
        ``forward`` described in overview. Otherwise, returns nothing.
    """
    if not compile and outdir is None:
        warnings.warn(
            "Will have no observable effect (besides errors)"
            " with compile=False and outdir=None."
        )

    if outdir is None:
        with tempfile.TemporaryDirectory() as d:
            return codegen(model, outdir=d, eigen=eigen, compile=compile)
    else:
        os.makedirs(outdir, exist_ok=True)
        print("writing to", outdir)

    header = outdir + "/mlp.h"
    src = outdir + "/mlp." + ("cpp" if eigen else "c")
    obj = outdir + "/mlp.o"
    lib = outdir + "/mlp.so"

    with open(header, "w") as f:
        f.write(f"#pragma once\n\n")
        f.write(f"#define MLP_INPUT_DIM {model.input_dim()}\n")
        f.write(f"#define MLP_OUTPUT_DIM {model.output_dim()}\n\n")
        f.write("void forward(float const *x, float *y);\n")

    with open(src, "w") as f:
        if eigen:
            codegen_eigen(model, f)
        else:
            codegen_c(model, f)

    if not compile:
        return

    flags_both = ["-c", "-fPIC", "-O3", "-o", obj]
    if eigen:
        # TODO: Useful to allow getting Eigen somewhere else besides eigenpip?
        import eigenpip
        incl = "-I" + eigenpip.get_include()
        flags_eig = [incl, "-fno-exceptions", "-DEIGEN_NO_MALLOC"]
        result_compile = subprocess.run(["c++"] + flags_both + flags_eig + [src])
    else:
        result_compile = subprocess.run(["cc"] + flags_both + [src])
    assert result_compile.returncode == 0
    result_shared = subprocess.run(["cc", "-shared", "-o", lib, obj])
    assert result_shared.returncode == 0
    return ctypes.cdll.LoadLibrary(lib)
