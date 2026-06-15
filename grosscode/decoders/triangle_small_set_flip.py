from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from grosscode.dem.builder import SplitSectorMetadata
from grosscode.decoders.logical_coset_triangle_descent import _catalog_all_exact_augmented_triangles
from grosscode.utils.gf2 import binary_csr_mod2, csr_matvec_mod2


_CANONICAL_TARGET_MASKS = np.asarray([0, 1, 2, 4], dtype=np.uint8)
_MASK_BITS = np.asarray(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class TriangleSmallSetFlipConfig:
    max_sweeps: int = 64


@dataclass(frozen=True)
class TriangleSmallSetFlipDecodeResult:
    e_hat: np.ndarray
    post_llr: np.ndarray
    mean_llr: np.ndarray
    residual: np.ndarray
    converged: bool
    decode_iters: int
    sweep_count: int
    accepted_move_count: int
    relation_count: int
    pattern_eval_count: int


def _error_cost_from_llr(llr: np.ndarray) -> np.ndarray:
    llr_vec = np.asarray(llr, dtype=np.float64).reshape(-1)
    return np.logaddexp(0.0, llr_vec)


def _rows_by_column(matrix: sp.csc_matrix) -> tuple[np.ndarray, ...]:
    out: list[np.ndarray] = []
    for col in range(int(matrix.shape[1])):
        start = int(matrix.indptr[col])
        stop = int(matrix.indptr[col + 1])
        out.append(np.asarray(matrix.indices[start:stop], dtype=np.int32))
    return tuple(out)


def _xor_support_rows(*supports: np.ndarray) -> np.ndarray:
    parity: dict[int, int] = {}
    for rows in supports:
        for row in np.asarray(rows, dtype=np.int32).tolist():
            key = int(row)
            parity[key] = 1 - parity.get(key, 0)
    active = sorted(int(row) for row, bit in parity.items() if int(bit) & 1)
    return np.asarray(active, dtype=np.int32)


def _mask_row_supports(
    *,
    relation_supports: np.ndarray,
    column_rows: tuple[np.ndarray, ...],
) -> tuple[tuple[np.ndarray, ...], ...]:
    out: list[tuple[np.ndarray, ...]] = []
    for cols in np.asarray(relation_supports, dtype=np.int32):
        row_sets: list[np.ndarray] = [np.zeros(0, dtype=np.int32)]
        col0 = np.asarray(column_rows[int(cols[0])], dtype=np.int32)
        col1 = np.asarray(column_rows[int(cols[1])], dtype=np.int32)
        col2 = np.asarray(column_rows[int(cols[2])], dtype=np.int32)
        row_sets.append(col0)
        row_sets.append(col1)
        row_sets.append(_xor_support_rows(col0, col1))
        row_sets.append(col2)
        row_sets.append(_xor_support_rows(col0, col2))
        row_sets.append(_xor_support_rows(col1, col2))
        row_sets.append(_xor_support_rows(col0, col1, col2))
        out.append(tuple(row_sets))
    return tuple(out)


class TriangleSmallSetFlipDecoder:
    def __init__(
        self,
        *,
        matrix: sp.csr_matrix,
        observables: sp.csr_matrix,
        priors: np.ndarray,
        metadata: SplitSectorMetadata,
        sector: str,
        config: TriangleSmallSetFlipConfig | None = None,
    ) -> None:
        cfg = TriangleSmallSetFlipConfig() if config is None else config
        if int(cfg.max_sweeps) <= 0:
            raise ValueError("max_sweeps must be positive")
        self.config = cfg
        self.matrix = binary_csr_mod2(matrix).tocsr()
        self.observables = binary_csr_mod2(observables).tocsr()
        clipped_priors = np.clip(np.asarray(priors, dtype=np.float64).reshape(-1), 1e-15, 1.0 - 1e-15)
        self.default_prior_llr = np.log((1.0 - clipped_priors) / clipped_priors)
        self.relations, relation_supports = _catalog_all_exact_augmented_triangles(
            matrix=self.matrix,
            observables=self.observables,
            metadata=metadata,
            sector=str(sector),
        )
        self.relation_supports = np.asarray(relation_supports, dtype=np.int32)
        self.relation_count = int(len(self.relations))
        column_rows = _rows_by_column(self.matrix.tocsc())
        self.mask_row_supports = _mask_row_supports(
            relation_supports=self.relation_supports,
            column_rows=column_rows,
        )

    @property
    def n(self) -> int:
        return int(self.matrix.shape[1])

    def decode(
        self,
        *,
        syndrome: np.ndarray,
        prior_llr: np.ndarray | None = None,
        init_e_hat: np.ndarray | None = None,
    ) -> TriangleSmallSetFlipDecodeResult:
        syndrome_bits = np.asarray(syndrome, dtype=np.uint8).reshape(-1) & 1
        if int(syndrome_bits.size) != int(self.matrix.shape[0]):
            raise ValueError("syndrome size mismatch")
        prior = self.default_prior_llr if prior_llr is None else np.asarray(prior_llr, dtype=np.float64).reshape(-1)
        if int(prior.size) != int(self.n):
            raise ValueError("prior_llr size mismatch")
        error_cost = _error_cost_from_llr(prior)

        if init_e_hat is None:
            e_hat = np.zeros(self.n, dtype=np.uint8)
            residual = syndrome_bits.copy()
        else:
            e_hat = np.asarray(init_e_hat, dtype=np.uint8).reshape(-1) & 1
            if int(e_hat.size) != int(self.n):
                raise ValueError("init_e_hat size mismatch")
            residual = syndrome_bits ^ csr_matvec_mod2(self.matrix, e_hat)
        residual_weight = int(np.count_nonzero(residual))
        if int(residual_weight) == 0:
            signed_llr = np.asarray(prior, dtype=np.float64).reshape(-1).copy()
            return TriangleSmallSetFlipDecodeResult(
                e_hat=e_hat,
                post_llr=signed_llr.copy(),
                mean_llr=signed_llr.copy(),
                residual=residual,
                converged=True,
                decode_iters=0,
                sweep_count=0,
                accepted_move_count=0,
                relation_count=int(self.relation_count),
                pattern_eval_count=0,
            )

        pattern_eval_count = 0
        accepted_move_count = 0
        sweep_count = 0
        for sweep_index in range(int(self.config.max_sweeps)):
            improved = False
            for rel_idx, cols in enumerate(self.relation_supports):
                col_arr = np.asarray(cols, dtype=np.int32)
                current_mask = (
                    int(e_hat[int(col_arr[0])])
                    | (int(e_hat[int(col_arr[1])]) << 1)
                    | (int(e_hat[int(col_arr[2])]) << 2)
                )
                best_target_mask: int | None = None
                best_delta_rows: np.ndarray | None = None
                best_weight = int(residual_weight)
                best_score = -np.inf
                best_tie: tuple[float, int, float, int, int, int] | None = None
                for target_mask in _CANONICAL_TARGET_MASKS.tolist():
                    delta_mask = int(current_mask) ^ int(target_mask)
                    delta_rows = np.asarray(self.mask_row_supports[int(rel_idx)][int(delta_mask)], dtype=np.int32)
                    pattern_eval_count += 1
                    if delta_rows.size == 0:
                        new_weight = int(residual_weight)
                    else:
                        new_weight = int(residual_weight + int(delta_rows.size) - 2 * int(np.sum(residual[delta_rows])))
                    syndrome_drop = int(residual_weight) - int(new_weight)
                    if int(syndrome_drop) <= 0:
                        continue
                    delta_bits = np.asarray(_MASK_BITS[int(delta_mask)], dtype=np.uint8)
                    move_cost = float(np.sum(error_cost[col_arr] * delta_bits.astype(np.float64)))
                    score = float(syndrome_drop) / max(float(move_cost), 1e-12)
                    tie_key = (
                        -float(score),
                        -int(syndrome_drop),
                        float(move_cost),
                        int(np.count_nonzero(_MASK_BITS[int(target_mask)])),
                        int(np.count_nonzero(_MASK_BITS[int(delta_mask)])),
                        int(target_mask),
                    )
                    if float(score) > float(best_score):
                        best_target_mask = int(target_mask)
                        best_delta_rows = delta_rows
                        best_weight = int(new_weight)
                        best_score = float(score)
                        best_tie = tie_key
                    elif np.isfinite(float(best_score)) and best_target_mask is not None and tie_key < tuple(best_tie or tie_key):
                        best_target_mask = int(target_mask)
                        best_delta_rows = delta_rows
                        best_score = float(score)
                        best_tie = tie_key
                if best_target_mask is None or int(best_weight) >= int(residual_weight):
                    continue
                e_hat[col_arr] = _MASK_BITS[int(best_target_mask)]
                if best_delta_rows is not None and int(best_delta_rows.size) > 0:
                    residual[best_delta_rows] ^= 1
                residual_weight = int(best_weight)
                accepted_move_count += 1
                improved = True
                if int(residual_weight) == 0:
                    sweep_count = int(sweep_index) + 1
                    signed_llr = np.asarray(prior, dtype=np.float64).reshape(-1) * (1.0 - 2.0 * e_hat.astype(np.float64))
                    return TriangleSmallSetFlipDecodeResult(
                        e_hat=e_hat,
                        post_llr=signed_llr.copy(),
                        mean_llr=signed_llr.copy(),
                        residual=residual.copy(),
                        converged=True,
                        decode_iters=int(pattern_eval_count),
                        sweep_count=int(sweep_count),
                        accepted_move_count=int(accepted_move_count),
                        relation_count=int(self.relation_count),
                        pattern_eval_count=int(pattern_eval_count),
                    )
            sweep_count = int(sweep_index) + 1
            if not improved:
                break

        signed_llr = np.asarray(prior, dtype=np.float64).reshape(-1) * (1.0 - 2.0 * e_hat.astype(np.float64))
        return TriangleSmallSetFlipDecodeResult(
            e_hat=e_hat,
            post_llr=signed_llr.copy(),
            mean_llr=signed_llr.copy(),
            residual=residual.copy(),
            converged=bool(np.count_nonzero(residual) == 0),
            decode_iters=int(pattern_eval_count),
            sweep_count=int(sweep_count),
            accepted_move_count=int(accepted_move_count),
            relation_count=int(self.relation_count),
            pattern_eval_count=int(pattern_eval_count),
        )
