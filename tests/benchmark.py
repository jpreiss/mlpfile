import contextlib
import os
import tempfile
import time

import mlpfile
import mlpfile.torch
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import torch


# Implements the manual Jacobian calculated with backward-mode autodiff, same
# as in mlpfile.cpp. The purpose is to see how much of the speed gap between
# mlpfile and pytorch is explained by the algorithm.
class MLPJacobian(torch.nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, x):
        f = x
        J = None
        stack = [f]
        # Forward pass
        for m in self.mlp:
            if isinstance(m, torch.nn.Linear):
                stack.append(m.weight @ stack[-1] + m.bias)
            elif isinstance(m, torch.nn.ReLU):
                stack.append(torch.nn.functional.relu(stack[-1]))
            else:
                raise ValueError
        # Backward pass
        for m in self.mlp[::-1]:
            if isinstance(m, torch.nn.Linear):
                stack.pop()
                if J is None:
                    J = m.weight
                else:
                    J = J @ m.weight
            elif isinstance(m, torch.nn.ReLU):
                f = stack.pop()
                mask = f > 0
                # Note: Ensuring performance optimization that multiplication
                # with diagonal is O(n^2), not O(n^3):
                #     diag(x) @ A == x[:, None] * A
                # and
                #     A @ diag(x) = A * x[None, :].
                # Also, PyTorch was complaining about diag() with a Boolean
                # argument in some past version of the code.
                J = J * mask[None, :]
            else:
                raise ValueError
        assert len(stack) == 1
        return J


INDIM = 40
OUTDIM = 10
HIDDEN = [100] * 2
NET = mlpfile.torch.mlp(INDIM, OUTDIM, HIDDEN)
NET.eval()


def _check_onnx_size(path):
    """Check size of ONNX's output file.

    Note that the Jacobian file should not be much bigger, because the weights
    data is the same -- only the computation graph is larger.
    """
    floats = INDIM * HIDDEN[0] + HIDDEN[0]
    for i in range(len(HIDDEN) - 1):
        floats += HIDDEN[i] * HIDDEN[i + 1] + HIDDEN[i + 1]
    floats += HIDDEN[-1] * OUTDIM + OUTDIM
    minimal = floats * 4
    # Hopefully we don't use more than a kb per layer for metadata.
    allowed = 1000 * (len(HIDDEN) + 1) + floats * 4
    actual = os.path.getsize(path)
    error_msg = f"expected {floats * 4} < size <= {allowed}, actually {actual}"
    if (actual > allowed):
        print("ONNX file is bigger than we expected:\n" + error_msg)
        return False
    if (actual < minimal):
        print("ONNX file is impossibly small:\n" + error_msg)
        return False
    return True


def compare_forward(net_ours):
    path = "forward.onnx"
    # ONNX for forward pass
    input_name = "x"
    output_name = "y"
    x = torch.randn(INDIM, requires_grad=True)
    with open ("/dev/null", "w") as dn:
        with contextlib.redirect_stdout(dn):
            torch.onnx.export(
                NET, x, path , input_names=[input_name], output_names=[output_name])

    _check_onnx_size(path)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Disable CPU multithreading - not worth the overhead for our small NN.
    sess_options.intra_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(path, sess_options)

    def fwd_onnx(x):
        return session.run(None, {input_name: x})[0]

    # Evaluate with a different input to make sure the ReLU activations change.
    x2 = torch.randn(INDIM)
    x2n = x2.numpy()

    # Compare running time.
    TRIALS = 10000
    for f, x, name in [
        (NET.forward, x2, "torch"),
        (fwd_onnx, x2n, " onnx"),
        (net_ours.forward, x2n, " ours"),
    ]:
        t0 = time.time()
        with torch.inference_mode():
            for _ in range(TRIALS):
                _ = f(x)
        us_per = 1000 * 1000 * (time.time() - t0) / TRIALS
        print(f"{name}: {us_per:7.2f} usec")


def compare_jacobian(net_ours):
    # Option 1: PyTorch's functional autodiff.
    with torch.no_grad():
        jac_autodiff = torch.func.jacrev(NET)

    # Option 2: Construct explicit Jacobian in PyTorch.
    jac = MLPJacobian(NET)
    jac.eval()

    # Option 3: Export and reload explicit Jacobian using ONNX.
    path = "jacobian.onnx"
    input_name = "x"
    jac_name = "J"
    x = torch.randn(INDIM, requires_grad=True)
    with open ("/dev/null", "w") as dn:
        with contextlib.redirect_stdout(dn):
            torch.onnx.export(
                jac, x, path, input_names=[input_name], output_names=[jac_name])

    _check_onnx_size(path)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Disable CPU multithreading - not worth the overhead for our small NN.
    sess_options.intra_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(path, sess_options)
    def jac_onnx(x):
        return session.run(None, {input_name: x})[0]

    # (Option 4: ours, but we already loaded it at global scope.)

    # Evaluate with a different input to make sure the ReLU activations change.
    x2 = torch.randn(INDIM)
    x2n = x2.numpy()

    # Compare running time.
    TRIALS = 1000
    for f, x, name in [
        (jac_autodiff,       x2, "torch-autodiff"),
        (jac.forward,        x2, "  torch-manual"),
        (jac_onnx,          x2n, "          onnx"),
        (net_ours.jacobian, x2n, "          ours"),
    ]:
        t0 = time.time()
        with torch.inference_mode():
            for _ in range(TRIALS):
                _ = f(x)
        us_per = 1000 * 1000 * (time.time() - t0) / TRIALS
        print(f"{name}: {us_per:7.2f} usec")


def _printbox(s):
    n = len(s) + 4
    print()
    print("*" * n)
    print("*", s, "*")
    print("*" * n)


def main():
    with tempfile.TemporaryDirectory() as testdir:
        os.chdir(testdir)
        # Our format
        mlpfile.torch.write(NET, "Phi.mlp")
        net_ours = mlpfile.Model.load("Phi.mlp")
        print(net_ours)
        for layer in net_ours.layers:
            print(layer)

        _printbox("Forward")
        compare_forward(net_ours)
        _printbox("Jacobian")
        compare_jacobian(net_ours)


if __name__ == "__main__":
    main()
