"""mlpfile: Multilayer perceptron file format and evaluation."""

from _mlpfile_bindings import Model, Layer, LayerType


__all__ = ["Model", "Layer", "LayerType", "cpp_dir"]


def cpp_dir():
    """Returns the directory containing the C++ header and source file.

    If you install the package via ``pip``, you can use the files in your build
    system. For example, a Makefile might contain::

        CXXFLAGS := -I$(shell python -c "import mlpfile; print(mlpfile.cpp_dir())")
    """
    import os.path
    return os.path.dirname(__file__) + "/cpp"
