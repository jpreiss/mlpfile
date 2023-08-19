"""Submodule of ``mlpfile`` for functions that must import PyTorch.

Note that this submodule is written in Python. None of this functionality is available from the ``mlpfile`` C++ API.
"""

from .torch import mlp, write

__all__ = ["mlp", "write"]
