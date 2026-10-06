
"""Fashion-MNIST dataset loader."""

import os
import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np


_CACHE_DIR = Path.home() / ".cache" / "dptiny"


_URLS = {
    "train-images-idx3-ubyte.gz":
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-images-idx3-ubyte.gz",

    "train-labels-idx1-ubyte.gz":
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-labels-idx1-ubyte.gz",

    "t10k-images-idx3-ubyte.gz":
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-images-idx3-ubyte.gz",

    "t10k-labels-idx1-ubyte.gz":
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-labels-idx1-ubyte.gz",
}


CLASSES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


def _download_file(url, destination):
    """Download a file without leaving a partial cache file."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary = destination.with_suffix(destination.suffix + ".tmp")

    try:
        urllib.request.urlretrieve(url, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _ensure_cached(data_home=None):
    """Download Fashion-MNIST files if they are not already cached."""

    cache_dir = Path(data_home) if data_home is not None else _CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached_files = {}

    for filename, url in _URLS.items():
        destination = cache_dir / filename

        if destination.exists():
            print(f"Using cached file: {filename}")
        else:
            _download_file(url, destination)

        cached_files[filename] = destination

    return cached_files


def _read_idx(path):
    """Read a uint8 IDX file and return it as a NumPy array."""

    with gzip.open(path, "rb") as f:
        magic = f.read(4)

        if len(magic) != 4:
            raise ValueError("Invalid IDX file: header is too short")

        if magic[0] != 0 or magic[1] != 0:
            raise ValueError("Invalid IDX file: bad magic number")

        dtype_code = magic[2]

        if dtype_code != 0x08:
            raise ValueError("Invalid IDX file: expected uint8 data")

        ndim = magic[3]

        if ndim == 0:
            raise ValueError("Invalid IDX file: zero dimensions")

        dimension_bytes = f.read(4 * ndim)

        if len(dimension_bytes) != 4 * ndim:
            raise ValueError("Invalid IDX file: incomplete dimension header")

        dimensions = struct.unpack(
            ">" + "I" * ndim,
            dimension_bytes
        )

        raw_data = f.read()

    expected_size = int(np.prod(dimensions))

    if len(raw_data) != expected_size:
        raise ValueError(
            f"Invalid IDX file: expected {expected_size} bytes, "
            f"found {len(raw_data)} bytes"
        )

    array = np.frombuffer(raw_data, dtype=np.uint8)

    return array.reshape(dimensions)


def get_fashion_mnist(
    normalize=True,
    flatten=True,
    data_home=None
):
    """
    Load the Fashion-MNIST dataset.

    Returns:
        X_train, X_test, y_train, y_test
    """

    paths = _ensure_cached(data_home)

    X_train = _read_idx(
        paths["train-images-idx3-ubyte.gz"]
    )

    y_train = _read_idx(
        paths["train-labels-idx1-ubyte.gz"]
    )

    X_test = _read_idx(
        paths["t10k-images-idx3-ubyte.gz"]
    )

    y_test = _read_idx(
        paths["t10k-labels-idx1-ubyte.gz"]
    )

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    if normalize:
        X_train /= 255.0
        X_test /= 255.0

    if flatten:
        X_train = X_train.reshape(X_train.shape[0], 784)
        X_test = X_test.reshape(X_test.shape[0], 784)
    else:
        X_train = X_train[:, np.newaxis, :, :]
        X_test = X_test[:, np.newaxis, :, :]

    return X_train, X_test, y_train, y_test
