from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from grosscode.utils.gf2 import binary_csr_mod2, csr_matvec_mod2, dense_mod2, invert_dense_mod2, rref_dense_mod2


@dataclass(frozen=True)
class GF2AffineSolveSystem:
    row_basis_indices: np.ndarray
    pivot_cols: np.ndarray
    pivot_inverse: np.ndarray
    n_cols: int


def independent_row_indices_mod2(matrix: np.ndarray) -> np.ndarray:
    work = dense_mod2(matrix).copy()
    if work.ndim != 2:
        raise ValueError("matrix must be rank 2")
    row_order = np.arange(int(work.shape[0]), dtype=np.int32)
    n_rows, n_cols = map(int, work.shape)
    pivot_row = 0
    chosen: list[int] = []
    for col in range(n_cols):
        candidates = np.flatnonzero(work[pivot_row:, col])
        if candidates.size == 0:
            continue
        src = pivot_row + int(candidates[0])
        if src != pivot_row:
            work[[pivot_row, src]] = work[[src, pivot_row]]
            row_order[[pivot_row, src]] = row_order[[src, pivot_row]]
        ones = np.flatnonzero(work[:, col])
        for row in ones.tolist():
            if int(row) == int(pivot_row):
                continue
            work[int(row)] ^= work[int(pivot_row)]
        chosen.append(int(row_order[pivot_row]))
        pivot_row += 1
        if pivot_row == n_rows:
            break
    return np.asarray(chosen, dtype=np.int32)


def build_gf2_affine_solver(augmented: sp.csr_matrix) -> GF2AffineSolveSystem:
    dense = np.asarray(binary_csr_mod2(augmented).toarray(), dtype=np.uint8)
    row_basis = independent_row_indices_mod2(dense)
    basis_rows = dense[np.asarray(row_basis, dtype=np.int32)]
    _, pivot_cols = rref_dense_mod2(basis_rows)
    pivot_arr = np.asarray(pivot_cols, dtype=np.int32)
    if int(pivot_arr.size) != int(basis_rows.shape[0]):
        raise ValueError("row-basis pivot count mismatch while building GF(2) affine solver")
    pivot_block = np.asarray(basis_rows[:, pivot_arr], dtype=np.uint8)
    pivot_inverse = invert_dense_mod2(pivot_block)
    return GF2AffineSolveSystem(
        row_basis_indices=np.asarray(row_basis, dtype=np.int32),
        pivot_cols=pivot_arr,
        pivot_inverse=np.asarray(pivot_inverse, dtype=np.uint8),
        n_cols=int(dense.shape[1]),
    )


def solve_affine_target_mod2(
    *,
    solver: GF2AffineSolveSystem,
    augmented: sp.csr_matrix,
    target: np.ndarray,
) -> np.ndarray:
    rhs = dense_mod2(target).reshape(-1)
    rhs_basis = rhs[np.asarray(solver.row_basis_indices, dtype=np.int32)]
    pivot_vals = np.mod(
        np.asarray(solver.pivot_inverse, dtype=np.uint64).dot(np.asarray(rhs_basis, dtype=np.uint64)),
        2,
    ).astype(np.uint8, copy=False)
    out = np.zeros(int(solver.n_cols), dtype=np.uint8)
    out[np.asarray(solver.pivot_cols, dtype=np.int32)] = pivot_vals
    predicted = csr_matvec_mod2(augmented, out)
    if not np.array_equal(predicted, rhs):
        raise ValueError("GF(2) affine solve returned an inconsistent solution")
    return out


@dataclass(frozen=True)
class AugmentedReferenceRecoverySolver:
    detector_matrix: sp.csr_matrix
    logical_matrix: sp.csr_matrix
    augmented_matrix: sp.csr_matrix
    affine_solver: GF2AffineSolveSystem

    @classmethod
    def build(cls, *, detector_matrix: sp.csr_matrix, logical_matrix: sp.csr_matrix) -> "AugmentedReferenceRecoverySolver":
        det = binary_csr_mod2(detector_matrix).tocsr()
        log = binary_csr_mod2(logical_matrix).tocsr()
        augmented = sp.vstack([det, log], format="csr")
        solver = build_gf2_affine_solver(augmented)
        return cls(
            detector_matrix=det,
            logical_matrix=log,
            augmented_matrix=augmented,
            affine_solver=solver,
        )

    @property
    def n(self) -> int:
        return int(self.detector_matrix.shape[1])

    @property
    def detector_rows(self) -> int:
        return int(self.detector_matrix.shape[0])

    @property
    def logical_rows(self) -> int:
        return int(self.logical_matrix.shape[0])

    def solve_augmented(self, target: np.ndarray) -> np.ndarray:
        rhs = dense_mod2(target).reshape(-1)
        if int(rhs.size) != int(self.detector_rows + self.logical_rows):
            raise ValueError("augmented target length mismatch")
        return solve_affine_target_mod2(
            solver=self.affine_solver,
            augmented=self.augmented_matrix,
            target=rhs,
        )

    def solve_reference(self, syndrome: np.ndarray, logical_sector: np.ndarray) -> np.ndarray:
        syndrome_bits = dense_mod2(syndrome).reshape(-1)
        logical_bits = dense_mod2(logical_sector).reshape(-1)
        if int(syndrome_bits.size) != int(self.detector_rows):
            raise ValueError("syndrome length mismatch")
        if int(logical_bits.size) != int(self.logical_rows):
            raise ValueError("logical-sector length mismatch")
        return self.solve_augmented(np.concatenate([syndrome_bits, logical_bits]))

    def validate_reference(self, *, syndrome: np.ndarray, logical_sector: np.ndarray, reference: np.ndarray) -> bool:
        ref = dense_mod2(reference).reshape(-1)
        return bool(
            np.array_equal(csr_matvec_mod2(self.detector_matrix, ref), dense_mod2(syndrome).reshape(-1))
            and np.array_equal(csr_matvec_mod2(self.logical_matrix, ref), dense_mod2(logical_sector).reshape(-1))
        )
