import tempfile

import numpy as np
import pytest
import torch

import mlpfile


INDIM = 40
OUTDIM = 10
HIDDEN = [100] * 2
NET = mlpfile.mlp(INDIM, OUTDIM, HIDDEN)
NET.eval()
JAC = torch.func.jacrev(NET)


# The point of this is to make sure we write the file exactly once and then
# clean up after all tests are complete. See the pytest "How to use fixtures"
# docs for more info.
@pytest.fixture(scope="module", autouse=True)
def model():
    with tempfile.NamedTemporaryFile() as f:
        mlpfile.write_torch(NET, f.name)
        model = mlpfile.Model.load(f.name)
        yield model


def test_forward(model):
    for _ in range(100):
        x = torch.randn(INDIM, requires_grad=False)
        y = model.forward(x)
        ytorch = NET.forward(x).detach().numpy()
        # TODO: Why do we need to loosen the tolerance?
        assert np.all(np.isclose(ytorch, y, atol=1e-6, rtol=1e-4))

def test_jacobian(model):
    for i in range(100):
        x = torch.randn(INDIM, requires_grad=False)
        J = model.jacobian(x)
        Jtorch = JAC(x).detach().numpy()
        # TODO: Why do we need to loosen the tolerance?
        assert np.all(np.isclose(J.flat, Jtorch.flat, atol=1e-6, rtol=1e-4))
