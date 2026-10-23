"""Random streams (PROTOKOL_v5 §4.3).

Every stream is a CPU ``torch.Generator`` seeded with the first 63 bits of
SHA-256 of the textual representation of a tuple. Indices drawn on CPU are
moved to the target device by the caller.
"""
import hashlib

import torch

SPLIT, ORDER, NEG, INIT = 1, 2, 3, 4


def seed_from(*parts) -> int:
    digest = hashlib.sha256(repr(tuple(parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def generator(*parts) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed_from(*parts))
    return g


def split_gen(seed: int) -> torch.Generator:
    return generator(seed, SPLIT)


def order_gen(seed: int, layer: int, epoch: int) -> torch.Generator:
    return generator(seed, ORDER, layer, epoch)


def neg_gen(seed: int, layer: int, epoch: int) -> torch.Generator:
    return generator(seed, NEG, layer, epoch)


def init_seed(seed: int) -> int:
    return seed_from(seed, INIT)
