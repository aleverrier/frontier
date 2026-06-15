from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def dense_mod2(values: np.ndarray | Sequence[Sequence[int]] | Sequence[int]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.uint8)
    return np.bitwise_and(arr, 1).astype(np.uint8, copy=False)


def matmul_mod2(
    lhs: np.ndarray | Sequence[Sequence[int]],
    rhs: np.ndarray | Sequence[Sequence[int]] | Sequence[int],
) -> np.ndarray:
    a = dense_mod2(lhs)
    b = dense_mod2(rhs)
    prod = np.asarray(a @ b, dtype=np.uint64)
    return np.bitwise_and(prod, 1).astype(np.uint8, copy=False)


def matvec_mod2(matrix: np.ndarray | Sequence[Sequence[int]], vector: np.ndarray | Sequence[int]) -> np.ndarray:
    return matmul_mod2(matrix, np.asarray(vector, dtype=np.uint8).reshape(-1, 1)).reshape(-1)


def rank_mod2(matrix: np.ndarray | Sequence[Sequence[int]]) -> int:
    reduced = right_reduced_row_echelon_mod2(matrix)
    return int(len(reduced.pivot_columns))


@dataclass(frozen=True)
class RowEchelonForm:
    matrix: np.ndarray
    row_transform: np.ndarray
    pivot_columns: tuple[int, ...]


def right_reduced_row_echelon_mod2(matrix: np.ndarray | Sequence[Sequence[int]]) -> RowEchelonForm:
    work = dense_mod2(matrix).copy()
    if work.ndim != 2:
        raise ValueError("matrix must be rank 2")
    n_rows, n_cols = map(int, work.shape)
    row_transform = np.eye(n_rows, dtype=np.uint8)
    pivot_row = 0
    pivots: list[int] = []
    for col in range(n_cols - 1, -1, -1):
        if pivot_row >= n_rows:
            break
        candidates = np.flatnonzero(work[pivot_row:, col])
        if candidates.size == 0:
            continue
        src = pivot_row + int(candidates[0])
        if src != pivot_row:
            work[[pivot_row, src]] = work[[src, pivot_row]]
            row_transform[[pivot_row, src]] = row_transform[[src, pivot_row]]
        ones = np.flatnonzero(work[:, col])
        for row in ones.tolist():
            if row == pivot_row:
                continue
            work[row] ^= work[pivot_row]
            row_transform[row] ^= row_transform[pivot_row]
        pivots.append(col)
        pivot_row += 1
    return RowEchelonForm(
        matrix=work.astype(np.uint8, copy=False),
        row_transform=row_transform.astype(np.uint8, copy=False),
        pivot_columns=tuple(int(col) for col in pivots),
    )


def enumerate_binary_vectors(width: int) -> np.ndarray:
    if width < 0:
        raise ValueError("width must be non-negative")
    if width == 0:
        return np.zeros((1, 0), dtype=np.uint8)
    values = np.arange(1 << width, dtype=np.uint64)
    bits = ((values[:, None] >> np.arange(width, dtype=np.uint64)) & 1).astype(np.uint8)
    return bits


def bits_to_int(bits: Iterable[int]) -> int:
    value = 0
    for index, bit in enumerate(bits):
        value |= (int(bit) & 1) << index
    return int(value)
