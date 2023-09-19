from mlpfile import *


def array1d(x):
    return "{" + ", ".join([str(xi) for xi in x]) + "}"


def array2d(X):
    return "{\n" + ",\n".join([array1d(row) for row in X]) + "\n}"


def codegen_c(model, f):
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


def codegen_eigen(model, f):
    """Generates C++ code with C-only API to evaluate the model's forward pass.

    Takes ``mlpfile.Model`` and open file-like object. See the docstring of
    ``codegen_compile`` for details.
    """

    f.write("#include <Eigen/Dense>\n")

    # static data
    for i, layer in enumerate(model.layers):
        if layer.type == LayerType.Linear:
            rows, cols = layer.W.shape
            # I would have preferred to store data directly in Eigen::Matrices
            # instead of the extra complexity of these Eigen::Maps. But for
            # some reason the initializer list constructor of Eigen::Matrix
            # wasn't working even though I was using C++11.
            f.write(f"float arr_W_{i}[{rows}][{cols}] = {array2d(layer.W)};\n")
            f.write(f"float arr_b_{i}[{rows}] = {array1d(layer.b)};\n")
            f.write(f"Eigen::Map<Eigen::Matrix<float, {rows}, {cols}, Eigen::RowMajor> > W_{i} (arr_W_{i}[0]);\n")
            f.write(f"Eigen::Map<Eigen::Matrix<float, {rows}, 1> > b_{i} (arr_b_{i});\n")

    # workspace
    linears = [layer for layer in model.layers if layer.type == LayerType.Linear]
    workdim = max(
        [lay.W.shape[0] for lay in linears]
        + [lay.W.shape[1] for lay in linears]
    )
    f.write(f"Eigen::Matrix<float, {workdim}, 1> work[2];\n")

    # code
    size = model.input_dim()
    f.write(f'extern "C" void forward(float const *xptr, float *yptr) {{\n')
    f.write(f"Eigen::Map<Eigen::Matrix<float, {size}, 1> const> x(xptr);\n")
    f.write(f"work[0].head<{size}>() = x;\n")
    src = 0
    dst = 1
    for ilayer, layer in enumerate(model.layers):
        if layer.type == LayerType.Input:
            continue
        elif layer.type == LayerType.Linear:
            newsize = layer.W.shape[0]
            f.write(f"work[{dst}].head<{newsize}>() = W_{ilayer} * work[{src}].head<{size}>() + b_{ilayer};\n")
            size = newsize
        elif layer.type == LayerType.ReLU:
            f.write(f"work[{dst}].head<{size}>() = work[{src}].head<{size}>().array().max(0);\n")
        else:
            raise ValueError("layer type:", layer.type)
        src, dst = dst, src
    f.write(f"Eigen::Map<Eigen::Matrix<float, {model.output_dim()}, 1> > y(yptr);\n")
    f.write(f"y = work[{src}].head<{size}>();\n")
    f.write("}")
