# Available at setup time due to pyproject.toml
import eigenpip
from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

__version__ = "0.2.1"

ext_modules = [
    Pybind11Extension("_mlpfile",
        ["mlpfile/cpp/bindings.cpp", "mlpfile/cpp/mlpfile.cpp"],
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
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    packages=["mlpfile", "mlpfile.torch"],
    package_data={
        "mlpfile": ["cpp/mlpfile.h", "cpp/mlpfile.cpp"]
    },
    requires=["numpy"],
    extras_require={
        "test": ["pytest", "torch"],
        "torch": "torch",
    },
    zip_safe=False,  # TODO: Understand.
    python_requires=">=3.7",
)
