# Available at setup time due to pyproject.toml
import eigenpip
from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

__version__ = "0.0.1"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

ext_modules = [
    Pybind11Extension("_mlpfile_bindings",
        ["src/bindings.cpp", "src/mlpfile.cpp"],
        include_dirs = [eigenpip.get_include()],
        cxx_std=11,
        ),
]

setup(
    name="mlpfile",
    version=__version__,
    author="James A. Preiss",
    author_email="jamesalanpreiss@gmail.com",
    url="https://github.com/jpreiss/mlpfile",
    description="Multilayer perceptron file format and evaluation.",
    long_description="",
    ext_modules=ext_modules,
    py_modules=["mlpfile", "_mlpfile_python"],
    extras_require={"test": "pytest"},
    zip_safe=False,  # TODO: Understand.
    python_requires=">=3.7",
)
