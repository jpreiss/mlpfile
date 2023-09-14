"""mlpfile: Multilayer perceptron file format and evaluation."""

from _mlpfile import *


__all__ = [
    "Model", "Layer", "LayerType", # Bound class/enum.
    "squared_error", "softmax_cross_entropy", # Bound function.
    "cpp_dir", # Pure Python.
]


def cpp_dir():
    """Prints the directory containing the C++ header and source file.

    If you install the package via ``pip``, you can use the files in your build
    system. For example, a Makefile might contain::

        CXXFLAGS := -I$(shell python -c "__import__('mlpfile').cpp_dir()")
    """
    import os.path
    print(os.path.dirname(__file__) + "/cpp", end="")  # no newline
