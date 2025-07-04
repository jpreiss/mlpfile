from copy import deepcopy
import ctypes
import tempfile

import numpy as np
import pytest
import torch

import mlpfile
import mlpfile.torch


INDIM = 40
OUTDIM = 10
HIDDEN = [100] * 2
NET = mlpfile.torch.mlp(INDIM, OUTDIM, HIDDEN)
NET_NOTEVAL = deepcopy(NET)
NET.eval()
JAC = torch.func.jacrev(NET)


def norm2(x):
    return np.sum(x ** 2)


def softmax(x):
    e = np.exp(x)
    return e / np.sum(e)


# The point of this is to make sure we write the file exactly once and then
# clean up after all tests are complete. See the pytest "How to use fixtures"
# docs for more info.
@pytest.fixture(scope="module", autouse=True)
def model():
    with tempfile.NamedTemporaryFile() as f:
        mlpfile.torch.write(NET, f.name)
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


def test_jacobian_params(model):
    assert len(model.layers) == len(NET_NOTEVAL)
    for i in range(100):
        x = torch.randn(INDIM, requires_grad=False)
        J = model.jacobian_params(x)
        for j in range(OUTDIM):
            NET_NOTEVAL.zero_grad()
            y = NET_NOTEVAL.forward(x)
            y[j].backward()
            for layer, Jlayer, tparam in zip(model.layers, J, NET_NOTEVAL):
                if layer.type != mlpfile.LayerType.Linear:
                    continue
                d_yj_W = Jlayer.dW[j].reshape(layer.W.shape)
                # TODO: Why do we need to loosen the tolerance?
                assert np.allclose(d_yj_W, tparam.weight.grad, atol=1e-6, rtol=1e-4)
                assert np.allclose(Jlayer.db[j], tparam.bias.grad, atol=1e-6, rtol=1e-4)


class GradCaseSqErr:
    def __init__(self):
        self.randomize()
        self.lossgrad = mlpfile.squared_error
        self.rate = 1e-2

    def randomize(self):
        self.ytrue = np.random.normal(size=OUTDIM)

    def loss(self, y):
        # Accounting for the 1/2 is important!
        return 0.5 * norm2(self.ytrue - y)

    def opt_loss(self):
        return 0.0


# In earlier prototype we set ytrue to a uniform (wrt Borel measure) sample
# from the simplex. However, it needed unreasonably many iterations to reach
# low error in test_ogd_onepoint. Choosing these sharp distributions (like we
# actually use in supervised classification) and a high learning rate gives
# fast convergence. TODO: Build more intuition for softmax cross-entropy to
# understand this. Why would fitting a high-entropy distribution require more
# steps?
class GradCaseXEnt:
    def __init__(self):
        self.randomize()
        self.lossgrad = mlpfile.softmax_cross_entropy
        self.rate = 1e-1

    def randomize(self):
        self.ytrue = (0.1 / (OUTDIM - 1)) * np.ones(OUTDIM)
        hot = np.random.choice(OUTDIM)
        self.ytrue[hot] = 0.9

    def loss(self, y):
        s = np.exp(y)
        s /= np.sum(s)
        return -np.sum(self.ytrue * np.log(s))

    def opt_loss(self):
        return -np.sum(np.log(self.ytrue) * self.ytrue)


GRAD_CASES = [GradCaseSqErr(), GradCaseXEnt()]
GRAD_IDS = ["squared-error", "cross-entropy"]


@pytest.mark.parametrize("gradcase", GRAD_CASES, ids=GRAD_IDS)
def test_ogd_onepoint(model, gradcase):
    x = np.random.normal(size=INDIM)
    # Full log makes debugging easier - plots, etc.
    trials = 100
    iters = 100
    gamma = 1.0 ** (1.0 / iters)
    rate = gradcase.rate
    for trial in range(trials):
        model2 = deepcopy(model)
        gradcase.randomize()
        print("initial:", gradcase.loss(model2.forward(x)))
        print(softmax(model2.forward(x)))
        print("opt:", gradcase.opt_loss())
        print(gradcase.ytrue)
        errs = []
        for i in range(iters):
            y = model2.forward(x);
            errs.append(gradcase.loss(y))
            model2.grad_update(x, gradcase.ytrue, gradcase.lossgrad, rate)
            rate *= gamma
        errs.append(gradcase.loss(model2.forward(x)))
        print("final:", softmax(model2.forward(x)))
        print()
        # We should be able to fit perfectly using last layer's bias.
        assert errs[-1] < gradcase.opt_loss() + 5e-5


@pytest.mark.parametrize("gradcase", GRAD_CASES, ids=GRAD_IDS)
def test_ogd_finitediff(model, gradcase):
    x = np.random.normal(size=INDIM)
    loss_original = gradcase.loss(model.forward(x))
    rate = gradcase.rate / 10

    trials = 100
    for trial in range(trials):
        model2 = deepcopy(model)
        model2.grad_update(x, gradcase.ytrue, gradcase.lossgrad, rate)

        # Reverse engineer what the gradient was.
        layers = model.layers
        grad_dot_step = 0.0
        for i in range(len(layers)):
            if layers[i].type == mlpfile.LayerType.Linear:
                gradW = (model2.layers[i].W - model.layers[i].W) / rate
                grad_dot_step -= rate * norm2(gradW.flatten())
                gradb = (model2.layers[i].b - model.layers[i].b) / rate
                grad_dot_step -= rate * norm2(gradb)

        # Predict what the new loss should be using first-order approximation.
        loss_predicted = loss_original + grad_dot_step
        assert loss_predicted < loss_original
        loss_actual = gradcase.loss(model2.forward(x))
        print(
            f"loss {gradcase.lossgrad.__name__}:\n"
            f"original: {loss_original}, "
            f"predicted: {loss_predicted}, "
            f"actual: {loss_actual}"
        )
        assert np.abs(loss_predicted - loss_actual) / loss_original < 5e-3


def test_ogd_multipoint(model):
    model = deepcopy(model)
    rate = 1e-2
    # Overfit a few random data points.
    # See "Understanding Deep Learning Requires Rethinking Generalization"
    N = 20
    xs = np.random.normal(size=(N, INDIM))
    ys = np.random.normal(size=(N, OUTDIM))
    # Full log makes debugging easier - plots, etc.
    errs = []
    for epoch in range(50):
        for i in range(N):
            y = model.forward(xs[i]);
            model.grad_update(xs[i], ys[i], mlpfile.squared_error, rate)
        errs.append(np.mean([
            norm2(model.forward(x) - y)
            for x, y in zip(xs, ys)
        ]))
    assert errs[0] > 1
    assert errs[-1] < 1e-2


def test_random():
    r = mlpfile.Model.random(2, [3, 4], 5)
    assert len(r.layers) == 5
    types = [lay.type for lay in r.layers]
    linear = mlpfile.LayerType.Linear
    relu = mlpfile.LayerType.ReLU
    assert types == [linear, relu, linear, relu, linear]
    y = r.forward([0, 0])
    assert y.size == 5
    r.jacobian([0, 0])


# Basic test of dimensions.
def test_jacobian_params_dim():
    model = mlpfile.Model.random(INDIM, HIDDEN, OUTDIM)
    x = np.random.normal(size=INDIM)

    jac_params = model.jacobian_params(x)
    assert len(jac_params) == len(model.layers)

    outdim = model.output_dim()

    for i, layer_jac in enumerate(jac_params):
        layer = model.layers[i]
        if layer.type == mlpfile.LayerType.Linear:
            assert layer_jac.dW.shape == (outdim, np.prod(layer.W.shape))
            assert layer_jac.db.shape == (outdim, layer.b.size)
        else:
            # ReLU layers should have empty Jacobians.
            assert layer_jac.dW.shape == (0, 0)
            assert layer_jac.db.shape == (0, 0)


def test_jacobian_params_analytic():

    model = mlpfile.Model()

    layer0 = mlpfile.Layer()
    layer0.type = mlpfile.LayerType.Linear
    layer0.W = np.eye(2)
    layer0.b = np.zeros(2)

    layer1 = mlpfile.Layer()
    layer1.type = mlpfile.LayerType.ReLU

    layer2 = mlpfile.Layer()
    layer2.type = mlpfile.LayerType.Linear
    layer2.W = np.ones((1, 2))
    layer2.b = np.zeros(1)

    model.layers = [layer0, layer1, layer2]

    # no ReLU
    x = np.array([1, 2])

    j0, _, j2 = model.jacobian_params(x)
    assert j0.dW.shape == (1, 4)
    assert np.all(j0.dW.reshape((2, 2)) == [x, x])
    assert np.all(j0.db == [[1, 1]])

    assert np.all(j2.dW == x[None, :])
    assert np.all(j2.db == [[1]])

    # ReLU
    x = np.array([-1, 1])
    j0, _, j2 = model.jacobian_params(x)
    assert j0.dW.shape == (1, 4)
    assert np.all(j0.dW.reshape((2, 2)) == [0 * x, x])
    assert np.all(j0.db == [[0, 1]])

    assert np.all(j2.dW == [[0, 1]])
    assert np.all(j2.db == [[1]])


def test_jacobian_params_finitediff():

    trials = 10
    for trial in range(trials):
        x = np.random.normal(size=2)
        # use prime numbers to bring out anything related to transpose
        model = mlpfile.Model.random(2, [3], 5)
        for layer in model.layers:
            layer.W = np.abs(layer.W)
        y = model.forward(x)
        J = model.jacobian_params(x)

        EPS = 1e-4
        for _ in range(10):
            model2 = deepcopy(model)
            dy_pred = np.zeros(5)
            for layer, Jlayer in zip(model2.layers, J):
                if layer.type == mlpfile.LayerType.Linear:
                    dW = EPS * np.random.normal(size=layer.W.shape)
                    db = EPS * np.random.normal(size=layer.b.shape)
                    # pybind11 doesn't give us +=. Not sure if worth fixing.
                    layer.W = layer.W + dW
                    layer.b = layer.b + db
                    dy_pred += Jlayer.dW @ dW.flat / EPS
                    dy_pred += Jlayer.db @ db / EPS
            dy_actual = (model2.forward(x) - y) / EPS
            # atol has to be pretty loose because ReLU is not smooth.
            assert np.allclose(dy_pred, dy_actual, atol=1e-2)


def test_cpp_dir(capfd):
    mlpfile.cpp_dir()
    out, _ = capfd.readouterr()
    assert out.endswith("cpp")


@pytest.mark.parametrize("eigen", [True, False])
def test_codegen(model, eigen):
    lib = mlpfile.codegen(model, eigen=eigen, compile=True)
    x = np.random.normal(size=INDIM).astype(np.float32)
    y = np.zeros(OUTDIM, dtype=np.float32)
    xptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    yptr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    lib.forward(xptr, yptr)
    y2 = model.forward(x)
    assert np.allclose(y, y2, atol=1e-6)
