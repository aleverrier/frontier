from __future__ import annotations

from typing import Sequence

import numpy as np

from .gf2 import dense_mod2

SUPPORTED_ORDERINGS = ("natural", "reverse", "bit_reversed")


def _check_power_of_two(length: int) -> None:
    if length <= 0 or length & (length - 1):
        raise ValueError(f"length must be a positive power of two, got {length}")


def bit_reversed_order(length: int) -> np.ndarray:
    _check_power_of_two(length)
    bits = int(np.log2(length))
    values = np.arange(length, dtype=np.uint32)
    out = np.zeros(length, dtype=np.int64)
    for bit in range(bits):
        out |= ((values >> bit) & 1) << (bits - 1 - bit)
    return out.astype(np.int64, copy=False)


def ordering_permutation(length: int, ordering: str, *, rng: np.random.Generator | None = None) -> np.ndarray:
    _check_power_of_two(length)
    key = str(ordering).strip()
    if key == "natural":
        return np.arange(length, dtype=np.int64)
    if key == "reverse":
        return np.arange(length - 1, -1, -1, dtype=np.int64)
    if key in {"bit_reversed", "bit-reversed", "bitreverse"}:
        return bit_reversed_order(length)
    if key.startswith("random"):
        local_rng = np.random.default_rng() if rng is None else rng
        return np.asarray(local_rng.permutation(length), dtype=np.int64)
    raise ValueError(f"unsupported ordering: {ordering!r}")


def arikan_matrix(length: int) -> np.ndarray:
    _check_power_of_two(length)
    mat = np.array([[1]], dtype=np.uint8)
    kernel = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    while int(mat.shape[0]) < length:
        mat = np.kron(mat, kernel).astype(np.uint8, copy=False)
        mat &= 1
    return mat


def apply_arikan_transform(vector: np.ndarray | Sequence[int]) -> np.ndarray:
    work = dense_mod2(vector).reshape(-1).copy()
    _check_power_of_two(int(work.shape[0]))
    span = 1
    length = int(work.shape[0])
    while span < length:
        block = span * 2
        for start in range(0, length, block):
            stop = start + span
            work[stop : start + block] ^= work[start:stop]
        span = block
    return work.astype(np.uint8, copy=False)
