from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.sparse as sp


def dense_mod2(values: np.ndarray | Sequence[Sequence[int]] | Sequence[int]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.uint8)
    return np.bitwise_and(arr, 1).astype(np.uint8, copy=False)


def binary_csr_mod2(matrix: sp.spmatrix) -> sp.csr_matrix:
    if not sp.isspmatrix(matrix):
        raise TypeError("expected a scipy sparse matrix")
    coo = matrix.tocoo(copy=True)
    coo.sum_duplicates()
    data = np.mod(np.asarray(coo.data, dtype=np.int64), 2).astype(np.uint8)
    keep = data != 0
    out = sp.coo_matrix((data[keep], (coo.row[keep], coo.col[keep])), shape=coo.shape, dtype=np.uint8)
    out = out.tocsr()
    out.sort_indices()
    return out


def csr_matvec_mod2(matrix: sp.spmatrix, vector: np.ndarray | Sequence[int]) -> np.ndarray:
    vec = dense_mod2(vector).reshape(-1)
    out = np.asarray(matrix.dot(vec), dtype=np.uint64).reshape(-1)
    return np.mod(out, 2).astype(np.uint8)


def rref_dense_mod2(matrix: np.ndarray | Sequence[Sequence[int]]) -> tuple[np.ndarray, tuple[int, ...]]:
    work = dense_mod2(matrix).copy()
    if work.ndim != 2:
        raise ValueError("matrix must be rank 2")
    n_rows, n_cols = work.shape
    pivot_row = 0
    pivots: list[int] = []
    for col in range(n_cols):
        candidates = np.flatnonzero(work[pivot_row:, col])
        if candidates.size == 0:
            continue
        src = pivot_row + int(candidates[0])
        if src != pivot_row:
            work[[pivot_row, src]] = work[[src, pivot_row]]
        ones = np.flatnonzero(work[:, col])
        for row in ones.tolist():
            if row == pivot_row:
                continue
            work[row] ^= work[pivot_row]
        pivots.append(col)
        pivot_row += 1
        if pivot_row == n_rows:
            break
    return work[:pivot_row].copy(), tuple(pivots)


def rank_dense_mod2(matrix: np.ndarray | Sequence[Sequence[int]]) -> int:
    _, pivots = rref_dense_mod2(matrix)
    return int(len(pivots))


def nullspace_basis_dense(matrix: np.ndarray | Sequence[Sequence[int]]) -> np.ndarray:
    rows = dense_mod2(matrix)
    if rows.ndim != 2:
        raise ValueError("matrix must be rank 2")
    _, n_cols = rows.shape
    rref, pivots = rref_dense_mod2(rows)
    pivot_set = set(int(p) for p in pivots)
    free_cols = [col for col in range(n_cols) if col not in pivot_set]
    if not free_cols:
        return np.zeros((0, n_cols), dtype=np.uint8)

    basis: list[np.ndarray] = []
    for free_col in free_cols:
        vec = np.zeros(n_cols, dtype=np.uint8)
        vec[free_col] = 1
        for row_index in range(len(pivots) - 1, -1, -1):
            pivot = int(pivots[row_index])
            row = rref[row_index]
            parity = int(np.bitwise_and(row, vec).sum() & 1)
            if parity:
                vec[pivot] ^= 1
        basis.append(vec)
    return np.vstack(basis).astype(np.uint8, copy=False)


def select_independent_rows_mod2(
    candidates: np.ndarray | Sequence[Sequence[int]],
    *,
    seed_rows: np.ndarray | Sequence[Sequence[int]] | None = None,
    max_rows: int | None = None,
) -> np.ndarray:
    cand = dense_mod2(candidates)
    if cand.ndim == 1:
        cand = cand.reshape(1, -1)
    if cand.ndim != 2:
        raise ValueError("candidates must be rank 1 or 2")
    n_cols = int(cand.shape[1])
    seed = dense_mod2(seed_rows) if seed_rows is not None else np.zeros((0, n_cols), dtype=np.uint8)
    if seed.ndim == 1:
        seed = seed.reshape(1, -1)
    if seed.size == 0:
        seed = np.zeros((0, n_cols), dtype=np.uint8)
    current = seed.copy()
    current_rank = rank_dense_mod2(current)
    chosen: list[np.ndarray] = []
    for row in cand:
        trial = np.vstack([current, row.reshape(1, -1)])
        new_rank = rank_dense_mod2(trial)
        if new_rank > current_rank:
            chosen.append(row.copy())
            current = trial
            current_rank = new_rank
            if max_rows is not None and len(chosen) >= int(max_rows):
                break
    if not chosen:
        return np.zeros((0, n_cols), dtype=np.uint8)
    return np.vstack(chosen).astype(np.uint8, copy=False)


def invert_dense_mod2(matrix: np.ndarray | Sequence[Sequence[int]]) -> np.ndarray:
    work = dense_mod2(matrix).copy()
    if work.ndim != 2 or work.shape[0] != work.shape[1]:
        raise ValueError("matrix must be square")
    n = int(work.shape[0])
    aug = np.concatenate([work, np.eye(n, dtype=np.uint8)], axis=1)
    pivot_row = 0
    for col in range(n):
        candidates = np.flatnonzero(aug[pivot_row:, col])
        if candidates.size == 0:
            raise ValueError("matrix is singular over GF(2)")
        src = pivot_row + int(candidates[0])
        if src != pivot_row:
            aug[[pivot_row, src]] = aug[[src, pivot_row]]
        ones = np.flatnonzero(aug[:, col])
        for row in ones.tolist():
            if row == pivot_row:
                continue
            aug[row] ^= aug[pivot_row]
        pivot_row += 1
    return aug[:, n:].astype(np.uint8, copy=False)

