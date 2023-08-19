import torch


def mlp(indim, outdim, hidden):
    """Construct a multilayer perceptron that can be represented by mlpfile.

    Will use ReLU activation automatically.

    PyTorch contains some tools that modify the training of an MLP but do not
    change its architecture, such as spectral normalization. To use these
    tools, please construct the ``torch.nn.Sequential`` yourself instead of using
    this function.

    Args:
        indim (int): Input dimension.
        outdim (int): Output dimension.
        hidden (List[int]): List of hidden layer dimensions.

    Returns:
        a ``torch.nn.Sequential`` module.
    """
    dims = [indim] + hidden
    layers = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(dims[-1], outdim))
    return torch.nn.Sequential(*layers)


def write(model: torch.nn.Sequential, path):
    """Writes a fully connected ReLU network to our file format.

    See top-level module documentation for binary format information.

    Args:
        model (torch.nn.Sequential): Trained model, such as one with the
            structure returned by :meth:`mlp()`.
        path (path-like): A `path-like object
            <https://docs.python.org/3/glossary.html#term-path-like-object>`_
            where the file should be written.
    """
    INPUT = 1
    LINEAR = 2
    RELU = 3
    with open(path, "wb") as f:
        def u32(i):
            f.write(i.to_bytes(4, byteorder="little", signed=False))
        u32(len(model) + 1)  # extra one for the input layer
        size = 0
        # First pass - metadata.
        for m in model:
            if isinstance(m, torch.nn.Linear):
                osize, isize = m.weight.shape
                assert osize != 0
                if size == 0:
                    u32(INPUT)
                    u32(isize)
                    size = isize
                else:
                    assert isize == size
                assert size != 0
                u32(LINEAR)
                u32(osize)
                size = osize
            elif isinstance(m, torch.nn.ReLU):
                u32(RELU)
            else:
                raise ValueError
        # Second pass - weight data.
        # TODO: We should enforce little-endian explicitly, here + the loader.
        for m in model:
            if isinstance(m, torch.nn.Linear):
                W = m.weight.detach().to(torch.float32).numpy()
                W = W.reshape(*W.shape, order="C")
                bW = W.tobytes()
                assert len(bW) == W.size * 4
                f.write(bW)
                b = m.bias.detach().to(torch.float32).numpy()
                bb = b.tobytes()
                assert len(bb) == b.size * 4
                f.write(bb)
