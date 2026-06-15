from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse as sp

from grosscode.dem.builder import SplitSectorMetadata
from grosscode.dem.triangles import (
    ExactTriangleRelation,
    TriangleColumnMetadata,
    _column_metadata,
    _combined_signature_by_column,
)
from grosscode.utils.gf2 import (
    binary_csr_mod2,
    csr_matvec_mod2,
    dense_mod2,
    invert_dense_mod2,
    rref_dense_mod2,
)


LogicalClassOrder = Literal["gray", "lex"]

_FULL_RELATION_CACHE: dict[tuple[object, ...], tuple[tuple[ExactTriangleRelation, ...], np.ndarray]] = {}


@dataclass(frozen=True)
class LogicalCosetTriangleDescentConfig:
    max_descent_iterations: int = 64
    max_classes: int = 4096
    class_order: LogicalClassOrder = "gray"


@dataclass(frozen=True)
class LogicalCosetTriangleDescentResult:
    e_hat: np.ndarray
    post_llr: np.ndarray
    mean_llr: np.ndarray
    residual: np.ndarray
    converged: bool
    decode_iters: int
    class_count: int
    relation_count: int
    best_logical_class_word: int
    best_logical_bits: np.ndarray
    best_objective_cost: float
    best_descent_steps: int
    total_relation_scans: int


@dataclass(frozen=True)
class _AffineRepresentativeSystem:
    row_basis_indices: np.ndarray
    pivot_cols: np.ndarray
    pivot_inverse: np.ndarray


def _triangle_kind_priority(kind: str) -> int:
    order = {
        "adjacent_round_mixed": 0,
        "same_round_split": 1,
        "adjacent_round_other": 2,
        "same_round_other": 3,
        "nonlocal_other": 4,
    }
    return int(order.get(str(kind), 99))


def _classify_triangle(column_meta: tuple[TriangleColumnMetadata, TriangleColumnMetadata, TriangleColumnMetadata]) -> tuple[str, int, int]:
    metas = tuple(column_meta)
    round_lo = min(int(meta.round_start) for meta in metas)
    round_hi = max(int(meta.round_stop) for meta in metas)
    bridge_count = sum(1 for meta in metas if int(meta.round_stop) > int(meta.round_start))
    within_count = 3 - int(bridge_count)
    fault_classes = tuple(sorted(str(meta.fault_class) for meta in metas))
    detector_weights = tuple(sorted(int(meta.detector_weight) for meta in metas))

    if (
        int(round_lo) == int(round_hi)
        and fault_classes == ("within_round", "within_round", "within_round")
        and detector_weights == (3, 3, 6)
    ):
        return ("same_round_split", int(round_lo), int(round_hi))
    if int(round_hi) == int(round_lo) + 1 and int(bridge_count) == 2 and int(within_count) == 1:
        return ("adjacent_round_mixed", int(round_lo), int(round_hi))
    if int(round_hi) == int(round_lo):
        return ("same_round_other", int(round_lo), int(round_hi))
    if int(round_hi) == int(round_lo) + 1:
        return ("adjacent_round_other", int(round_lo), int(round_hi))
    return ("nonlocal_other", int(round_lo), int(round_hi))


def _cache_key(
    *,
    matrix: sp.csr_matrix,
    observables: sp.csr_matrix,
    metadata: SplitSectorMetadata,
    sector: str,
) -> tuple[object, ...]:
    return (
        str(sector),
        str(metadata.stim_path),
        int(metadata.total_rounds),
        int(matrix.shape[0]),
        int(matrix.shape[1]),
        int(observables.shape[0]),
        int(observables.shape[1]),
        tuple(sorted((str(k), int(v)) for k, v in metadata.local_fault_class_counts.items())),
    )


def _catalog_all_exact_augmented_triangles(
    *,
    matrix: sp.csr_matrix,
    observables: sp.csr_matrix,
    metadata: SplitSectorMetadata,
    sector: str,
) -> tuple[tuple[ExactTriangleRelation, ...], np.ndarray]:
    key = _cache_key(matrix=matrix, observables=observables, metadata=metadata, sector=str(sector))
    cached = _FULL_RELATION_CACHE.get(key)
    if cached is not None:
        return cached

    matrix = binary_csr_mod2(matrix).tocsr()
    observables = binary_csr_mod2(observables).tocsr()
    column_meta = _column_metadata(matrix=matrix, observables=observables, metadata=metadata, sector=str(sector))
    signatures = [int(value) for value in _combined_signature_by_column(matrix, observables)]
    signature_to_indices: dict[int, list[int]] = {}
    for col, signature in enumerate(signatures):
        signature_to_indices.setdefault(int(signature), []).append(int(col))

    relations: list[ExactTriangleRelation] = []
    for i in range(int(matrix.shape[1])):
        sig_i = int(signatures[i])
        for j in range(i + 1, int(matrix.shape[1])):
            target_signature = int(sig_i ^ int(signatures[j]))
            for k in signature_to_indices.get(int(target_signature), ()):
                if int(k) <= int(j):
                    continue
                ordered = (int(i), int(j), int(k))
                metas = (column_meta[ordered[0]], column_meta[ordered[1]], column_meta[ordered[2]])
                kind, round_lo, round_hi = _classify_triangle(metas)
                relations.append(
                    ExactTriangleRelation(
                        sector=str(sector),
                        round_lo=int(round_lo),
                        round_hi=int(round_hi),
                        relation_kind=str(kind),  # type: ignore[arg-type]
                        columns=ordered,
                        column_metadata=metas,
                    )
                )
    relations.sort(
        key=lambda item: (
            int(item.round_lo),
            int(item.round_hi),
            _triangle_kind_priority(str(item.relation_kind)),
            tuple(int(col) for col in item.columns),
        )
    )
    supports = np.asarray([tuple(int(col) for col in relation.columns) for relation in relations], dtype=np.int32)
    result = (tuple(relations), supports)
    _FULL_RELATION_CACHE[key] = result
    return result


def _independent_row_indices_mod2(matrix: np.ndarray) -> np.ndarray:
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


def _build_affine_representative_system(augmented: sp.csr_matrix) -> tuple[_AffineRepresentativeSystem, int]:
    dense = np.asarray(binary_csr_mod2(augmented).toarray(), dtype=np.uint8)
    row_basis = _independent_row_indices_mod2(dense)
    basis_rows = dense[np.asarray(row_basis, dtype=np.int32)]
    _, pivot_cols = rref_dense_mod2(basis_rows)
    pivot_arr = np.asarray(pivot_cols, dtype=np.int32)
    if int(pivot_arr.size) != int(basis_rows.shape[0]):
        raise ValueError("row-basis pivot count mismatch while building affine representative system")
    pivot_block = np.asarray(basis_rows[:, pivot_arr], dtype=np.uint8)
    pivot_inverse = invert_dense_mod2(pivot_block)
    return (
        _AffineRepresentativeSystem(
            row_basis_indices=np.asarray(row_basis, dtype=np.int32),
            pivot_cols=pivot_arr,
            pivot_inverse=np.asarray(pivot_inverse, dtype=np.uint8),
        ),
        int(dense.shape[1]),
    )


def _solve_affine_target(
    *,
    system: _AffineRepresentativeSystem,
    augmented: sp.csr_matrix,
    target: np.ndarray,
    n_cols: int,
) -> np.ndarray:
    rhs = dense_mod2(target).reshape(-1)
    rhs_basis = rhs[np.asarray(system.row_basis_indices, dtype=np.int32)]
    pivot_vals = np.mod(
        np.asarray(system.pivot_inverse, dtype=np.uint64).dot(np.asarray(rhs_basis, dtype=np.uint64)),
        2,
    ).astype(np.uint8, copy=False)
    out = np.zeros(int(n_cols), dtype=np.uint8)
    out[np.asarray(system.pivot_cols, dtype=np.int32)] = pivot_vals
    predicted = csr_matvec_mod2(augmented, out)
    if not np.array_equal(predicted, rhs):
        raise ValueError("affine representative solve returned an inconsistent solution")
    return out


def _logical_word_bits(word: int, width: int) -> np.ndarray:
    return np.asarray([(int(word) >> idx) & 1 for idx in range(int(width))], dtype=np.uint8)


def _descent_work_and_estimate(
    *,
    seed: np.ndarray,
    llr: np.ndarray,
    relation_supports: np.ndarray,
    max_iterations: int,
) -> tuple[np.ndarray, int, int, bool]:
    current = (np.asarray(seed, dtype=np.uint8).reshape(-1) & 1).copy()
    llr_vec = np.asarray(llr, dtype=np.float64).reshape(-1)
    if int(relation_supports.size) == 0:
        return current, 1, 0, True
    signed_cost = (1.0 - 2.0 * current.astype(np.float64)) * llr_vec
    scans = 0
    accepted_steps = 0
    converged = True
    supports = np.asarray(relation_supports, dtype=np.int32)
    for _ in range(max(1, int(max_iterations))):
        scans += 1
        deltas = (
            signed_cost[supports[:, 0]]
            + signed_cost[supports[:, 1]]
            + signed_cost[supports[:, 2]]
        )
        best_idx = int(np.argmin(deltas))
        best_delta = float(deltas[best_idx])
        if not np.isfinite(best_delta) or best_delta >= -1e-12:
            return current, int(scans), int(accepted_steps), True
        cols = np.asarray(supports[best_idx], dtype=np.int32)
        current[cols] ^= 1
        signed_cost[cols] *= -1.0
        accepted_steps += 1
    converged = False
    return current, int(scans), int(accepted_steps), bool(converged)


class LogicalCosetTriangleDescentDecoder:
    def __init__(
        self,
        *,
        matrix: sp.csr_matrix,
        observables: sp.csr_matrix,
        priors: np.ndarray,
        metadata: SplitSectorMetadata,
        sector: str,
        config: LogicalCosetTriangleDescentConfig | None = None,
    ) -> None:
        cfg = LogicalCosetTriangleDescentConfig() if config is None else config
        if int(cfg.max_descent_iterations) <= 0:
            raise ValueError("max_descent_iterations must be positive")
        if int(cfg.max_classes) <= 0:
            raise ValueError("max_classes must be positive")
        if str(cfg.class_order) not in {"gray", "lex"}:
            raise ValueError("class_order must be one of {'gray','lex'}")
        self.config = cfg
        self.matrix = binary_csr_mod2(matrix).tocsr()
        self.observables = binary_csr_mod2(observables).tocsr()
        self.augmented = sp.vstack([self.matrix, self.observables], format="csr")
        clipped_priors = np.clip(np.asarray(priors, dtype=np.float64).reshape(-1), 1e-15, 1.0 - 1e-15)
        self.default_prior_llr = np.log((1.0 - clipped_priors) / clipped_priors)
        self.metadata = metadata
        self.sector = str(sector)
        self.relations, self.relation_supports = _catalog_all_exact_augmented_triangles(
            matrix=self.matrix,
            observables=self.observables,
            metadata=self.metadata,
            sector=self.sector,
        )
        self.relation_count = int(len(self.relations))
        self.affine_system, self.n = _build_affine_representative_system(self.augmented)
        self.logical_dim = int(self.observables.shape[0])
        self.full_class_count = int(1 << int(self.logical_dim))
        self.logical_toggle_basis = tuple(
            _solve_affine_target(
                system=self.affine_system,
                augmented=self.augmented,
                target=np.concatenate(
                    [
                        np.zeros(int(self.matrix.shape[0]), dtype=np.uint8),
                        np.eye(int(self.logical_dim), dtype=np.uint8)[int(logical_idx)],
                    ]
                ),
                n_cols=int(self.n),
            )
            for logical_idx in range(int(self.logical_dim))
        )

    def decode(
        self,
        *,
        syndrome: np.ndarray,
        prior_llr: np.ndarray | None = None,
    ) -> LogicalCosetTriangleDescentResult:
        syndrome_bits = np.asarray(syndrome, dtype=np.uint8).reshape(-1) & 1
        if int(syndrome_bits.size) != int(self.matrix.shape[0]):
            raise ValueError("syndrome size mismatch")
        prior = self.default_prior_llr if prior_llr is None else np.asarray(prior_llr, dtype=np.float64).reshape(-1)
        if int(prior.size) != int(self.n):
            raise ValueError("prior_llr size mismatch")
        base_seed = _solve_affine_target(
            system=self.affine_system,
            augmented=self.augmented,
            target=np.concatenate([syndrome_bits, np.zeros(int(self.logical_dim), dtype=np.uint8)]),
            n_cols=int(self.n),
        )

        class_count = min(int(self.config.max_classes), int(self.full_class_count))
        best_estimate: np.ndarray | None = None
        best_cost = float("inf")
        best_word = 0
        best_steps = 0
        best_converged = True
        total_scans = 0

        current_word = 0
        current_seed = np.asarray(base_seed, dtype=np.uint8).copy()
        for class_rank in range(int(class_count)):
            if int(class_rank) > 0:
                next_word = (
                    int(class_rank) ^ int(class_rank >> 1)
                    if str(self.config.class_order) == "gray"
                    else int(class_rank)
                )
                changed = int(current_word ^ next_word)
                while int(changed) > 0:
                    changed_bit = int(changed.bit_length() - 1)
                    current_seed ^= np.asarray(self.logical_toggle_basis[int(changed_bit)], dtype=np.uint8)
                    changed ^= 1 << int(changed_bit)
                current_word = int(next_word)
            candidate, scans, steps, converged = _descent_work_and_estimate(
                seed=current_seed,
                llr=prior,
                relation_supports=self.relation_supports,
                max_iterations=int(self.config.max_descent_iterations),
            )
            total_scans += int(scans)
            cost = float(np.dot(np.asarray(candidate, dtype=np.float64), np.asarray(prior, dtype=np.float64)))
            if (
                best_estimate is None
                or cost < best_cost - 1e-12
                or (abs(cost - best_cost) <= 1e-12 and int(current_word) < int(best_word))
            ):
                best_estimate = np.asarray(candidate, dtype=np.uint8).copy()
                best_cost = float(cost)
                best_word = int(current_word)
                best_steps = int(steps)
                best_converged = bool(converged)

        if best_estimate is None:
            best_estimate = np.zeros(int(self.n), dtype=np.uint8)
            best_cost = 0.0
            best_word = 0
            best_steps = 0
            best_converged = True

        residual = syndrome_bits ^ csr_matvec_mod2(self.matrix, best_estimate)
        signed_post = np.where(np.asarray(best_estimate, dtype=np.uint8) > 0, -np.abs(prior), np.abs(prior))
        return LogicalCosetTriangleDescentResult(
            e_hat=np.asarray(best_estimate, dtype=np.uint8).copy(),
            post_llr=np.asarray(signed_post, dtype=np.float64).copy(),
            mean_llr=np.asarray(signed_post, dtype=np.float64).copy(),
            residual=np.asarray(residual, dtype=np.uint8).copy(),
            converged=bool(best_converged and int(np.count_nonzero(residual)) == 0),
            decode_iters=int(total_scans),
            class_count=int(class_count),
            relation_count=int(self.relation_count),
            best_logical_class_word=int(best_word),
            best_logical_bits=_logical_word_bits(int(best_word), int(self.logical_dim)),
            best_objective_cost=float(best_cost),
            best_descent_steps=int(best_steps),
            total_relation_scans=int(total_scans),
        )
