"""Data loading, hash verification and splits (PROTOKOL_v5 §4).

The test sets (MNIST/F-MNIST test, QMNIST-50k) are reachable only after
``enable_test_access()``, which is called exclusively by ``evaluate_test``
(§4.2, test 9.1.5).
"""
import gzip
import hashlib
import math
import os
import struct
from dataclasses import dataclass

import numpy as np
import torch

from . import streams

DATA_DIR = os.environ.get(
    "FFEXP_DATA", os.path.join(os.path.dirname(__file__), "..", "..", "dane", "raw")
)
SUMS_FILE = os.path.join(os.path.dirname(DATA_DIR.rstrip("/")), "SHA256SUMS_DANE")

FILES = {
    "mnist": {
        "train": ("mnist/train-images-idx3-ubyte.gz", "mnist/train-labels-idx1-ubyte.gz"),
        "test": ("mnist/t10k-images-idx3-ubyte.gz", "mnist/t10k-labels-idx1-ubyte.gz"),
    },
    "fmnist": {
        "train": ("fmnist/train-images-idx3-ubyte.gz", "fmnist/train-labels-idx1-ubyte.gz"),
        "test": ("fmnist/t10k-images-idx3-ubyte.gz", "fmnist/t10k-labels-idx1-ubyte.gz"),
    },
    "qmnist": {
        "test50k": ("qmnist/qmnist-test-images-idx3-ubyte.gz", "qmnist/qmnist-test-labels-idx2-int.gz"),
    },
}
TEST_PARTS = {("mnist", "test"), ("fmnist", "test"), ("qmnist", "test50k")}

_TEST_ACCESS = False


class TestAccessError(RuntimeError):
    pass


def enable_test_access() -> None:
    global _TEST_ACCESS
    _TEST_ACCESS = True


def test_access_enabled() -> bool:
    return _TEST_ACCESS


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sums() -> dict:
    sums = {}
    with open(SUMS_FILE) as f:
        for line in f:
            if line.strip():
                digest, rel = line.split()
                sums[rel] = digest
    return sums


def verify_file(rel: str) -> str:
    path = os.path.join(DATA_DIR, rel)
    expected = load_sums().get(rel)
    if expected is None:
        raise RuntimeError(f"no hash for {rel} in {SUMS_FILE}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"hash mismatch for {rel}: {actual} != {expected}")
    return path


def _read_idx(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    zero, dtype_code, ndim = struct.unpack(">HBB", raw[:4])
    assert zero == 0
    dims = struct.unpack(">" + "I" * ndim, raw[4 : 4 + 4 * ndim])
    dtype = {0x08: np.dtype(">u1"), 0x0C: np.dtype(">i4")}[dtype_code]
    data = np.frombuffer(raw, dtype=dtype, offset=4 + 4 * ndim)
    return data.reshape(dims)


def load(dataset: str, part: str):
    """Return (X float32 [n, 784] in [0, 1], y int64 [n]) on CPU."""
    if (dataset, part) in TEST_PARTS and not _TEST_ACCESS:
        raise TestAccessError(f"test data {dataset}/{part} is not accessible in training mode")
    img_rel, lab_rel = FILES[dataset][part]
    images = _read_idx(verify_file(img_rel))
    labels = _read_idx(verify_file(lab_rel))
    if dataset == "qmnist":
        images, labels = images[10000:], labels[10000:, 0]
    x = torch.from_numpy(images.reshape(len(images), -1).astype(np.float32) / 255.0)
    y = torch.from_numpy(labels.astype(np.int64))
    return x, y


@dataclass
class Split:
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    es_idx: torch.Tensor  # 5k subset of val used for early stopping


def make_split(seed: int, n: int = 60000, n_val: int = 10000, n_es: int = 5000) -> Split:
    g = streams.split_gen(seed)
    perm = torch.randperm(n, generator=g)
    val_idx = perm[:n_val]
    es_idx = val_idx[torch.randperm(n_val, generator=g)[:n_es]]
    return Split(train_idx=perm[n_val:], val_idx=val_idx, es_idx=es_idx)


class ImageTask:
    """Label embedding for images: pixels 0-9 zeroed, pixel y set to 1 (§5.1)."""

    n_classes = 10
    in_dim = 784

    @staticmethod
    def embed(x: torch.Tensor, y) -> torch.Tensor:
        out = x.clone()
        out[:, :10] = 0.0
        if isinstance(y, int):
            out[:, y] = 1.0
        else:
            out[torch.arange(len(out), device=out.device), y] = 1.0
        return out


class SpiralTask:
    """2D spiral: label one-hot concatenated to (x1, x2) (§4.4)."""

    n_classes = 2
    in_dim = 4

    @staticmethod
    def embed(x: torch.Tensor, y) -> torch.Tensor:
        if isinstance(y, int):
            y = torch.full((len(x),), y, dtype=torch.long, device=x.device)
        onehot = torch.nn.functional.one_hot(y, 2).to(x.dtype)
        return torch.cat([x, onehot], dim=1)


def make_spiral(n_per_class: int = 500, noise: float = 0.04, seed: int = 0):
    """Two interleaved 1.5-turn spiral arms (§4.4). Returns dict of splits."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for cls in (0, 1):
        t = rng.uniform(0.1, 1.0, n_per_class)
        angle = 3 * math.pi * t + cls * math.pi
        pts = np.stack([t * np.cos(angle), t * np.sin(angle)], axis=1)
        pts += rng.normal(0.0, noise, pts.shape)
        xs.append(pts)
        ys.append(np.full(n_per_class, cls))
    x = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.int64)
    perm = rng.permutation(len(x))
    x, y = torch.from_numpy(x[perm]), torch.from_numpy(y[perm])
    return {
        "train": (x[:600], y[:600]),
        "val": (x[600:800], y[600:800]),
        "test": (x[800:], y[800:]),
    }
