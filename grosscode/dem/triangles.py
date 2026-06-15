from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse as sp

from grosscode.core import llr_from_priors
from grosscode.dem.builder import SplitSectorMetadata
from grosscode.utils.gf2 import binary_csr_mod2, csr_matvec_mod2


TriangleRelationKind = Literal[
    "same_round_split",
    "adjacent_round_mixed",
    "same_round_other",
    "adjacent_round_other",
    "nonlocal_other",
]


@dataclass(frozen=True)
class TriangleColumnMetadata:
    index: int
    round_start: int
    round_stop: int
    fault_class: str
    detector_weight: int
    logical_weight: int
    combined_signature: int


@dataclass(frozen=True)
class ExactTriangleRelation:
    sector: str
    round_lo: int
    round_hi: int
    relation_kind: TriangleRelationKind
    columns: tuple[int, int, int]
    column_metadata: tuple[TriangleColumnMetadata, TriangleColumnMetadata, TriangleColumnMetadata]

    @property
    def active_round_window(self) -> tuple[int, int]:
        return (int(self.round_lo), int(self.round_hi))

    @property
    def selection_key(self) -> tuple[int, int, int, int, int]:
        kind_priority = 0 if str(self.relation_kind) == "adjacent_round_mixed" else 1
        a, b, c = self.columns
        return (int(self.round_lo), int(self.round_hi), int(kind_priority), int(a), int(b), int(c))


@dataclass(frozen=True)
class ExactTriangleCatalog:
    sector: str
    relations: tuple[ExactTriangleRelation, ...]
    counts_by_kind: dict[str, int]


@dataclass(frozen=True)
class NonOverlappingTriangleSelection:
    sector: str
    selected_relations: tuple[ExactTriangleRelation, ...]
    residual_relations: tuple[ExactTriangleRelation, ...]
    counts_by_kind: dict[str, int]
    selected_counts_by_kind: dict[str, int]
    overlapping_counts_by_kind: dict[str, int]
    used_columns: frozenset[int]
    relation_index_by_column: dict[int, int]


@dataclass(frozen=True)
class ExactTriangleFactor:
    relation: ExactTriangleRelation
    state_probabilities: np.ndarray
    state_costs: np.ndarray
    representative_bits: tuple[tuple[int, int, int], ...]
    assignment_neglog: np.ndarray


@dataclass(frozen=True)
class TriangleGaugeStep:
    relation: ExactTriangleRelation
    delta_cost: float
    support: tuple[int, int, int]


@dataclass(frozen=True)
class TriangleGaugeDescentResult:
    estimate: np.ndarray
    accepted_steps: tuple[TriangleGaugeStep, ...]
    iterations: int
    converged: bool


def _combined_signature_by_column(matrix: sp.csr_matrix, observables: sp.csr_matrix) -> tuple[int, ...]:
    h_csc = binary_csr_mod2(matrix).tocsc()
    o_csc = binary_csr_mod2(observables).tocsc()
    n_cols = int(h_csc.shape[1])
    row_offset = int(h_csc.shape[0])
    out: list[int] = []
    for col in range(n_cols):
        signature = 0
        begin = int(h_csc.indptr[col])
        end = int(h_csc.indptr[col + 1])
        for row in np.asarray(h_csc.indices[begin:end], dtype=np.int32).tolist():
            signature ^= 1 << int(row)
        begin = int(o_csc.indptr[col])
        end = int(o_csc.indptr[col + 1])
        for row in np.asarray(o_csc.indices[begin:end], dtype=np.int32).tolist():
            signature ^= 1 << int(row_offset + int(row))
        out.append(int(signature))
    return tuple(int(value) for value in out)


def _column_metadata(
    *,
    matrix: sp.csr_matrix,
    observables: sp.csr_matrix,
    metadata: SplitSectorMetadata,
    sector: str,
) -> tuple[TriangleColumnMetadata, ...]:
    h_csc = binary_csr_mod2(matrix).tocsc()
    o_csc = binary_csr_mod2(observables).tocsc()
    signatures = _combined_signature_by_column(matrix, observables)
    out: list[TriangleColumnMetadata] = []
    for col in range(int(matrix.shape[1])):
        det_w = int(h_csc.indptr[col + 1] - h_csc.indptr[col])
        log_w = int(o_csc.indptr[col + 1] - o_csc.indptr[col])
        fault_class = "unknown"
        for group in metadata.variable_groups:
            if int(group.round_start) == int(metadata.column_round_start[col]) and int(group.round_stop) == int(metadata.column_round_stop[col]):
                fault_class = str(group.fault_class)
                break
        out.append(
            TriangleColumnMetadata(
                index=int(col),
                round_start=int(metadata.column_round_start[col]),
                round_stop=int(metadata.column_round_stop[col]),
                fault_class=str(fault_class),
                detector_weight=int(det_w),
                logical_weight=int(log_w),
                combined_signature=int(signatures[col]),
            )
        )
    return tuple(out)


def catalog_exact_local_triangles(
    *,
    matrix: sp.csr_matrix,
    observables: sp.csr_matrix,
    metadata: SplitSectorMetadata,
    sector: str,
) -> ExactTriangleCatalog:
    matrix = binary_csr_mod2(matrix).tocsr()
    observables = binary_csr_mod2(observables).tocsr()
    column_meta = _column_metadata(matrix=matrix, observables=observables, metadata=metadata, sector=str(sector))
    by_round_and_kind: dict[tuple[int, str], list[int]] = {}
    by_window_and_kind: dict[tuple[int, int, str], list[int]] = {}
    for item in column_meta:
        by_round_and_kind.setdefault((int(item.round_start), str(item.fault_class)), []).append(int(item.index))
        by_window_and_kind.setdefault((int(item.round_start), int(item.round_stop), str(item.fault_class)), []).append(int(item.index))

    relations: list[ExactTriangleRelation] = []
    seen_keys: set[tuple[str, tuple[int, int, int]]] = set()

    for round_index in range(int(metadata.total_rounds)):
        within_cols = sorted(int(col) for col in by_round_and_kind.get((int(round_index), "within_round"), ()))
        within_w3 = [col for col in within_cols if int(column_meta[col].detector_weight) == 3]
        within_w6 = [col for col in within_cols if int(column_meta[col].detector_weight) == 6]
        sig_to_w3: dict[int, list[int]] = {}
        for col in within_w3:
            sig_to_w3.setdefault(int(column_meta[col].combined_signature), []).append(int(col))
        for col_c in within_w6:
            sig_c = int(column_meta[col_c].combined_signature)
            for col_a in within_w3:
                sig_need = int(sig_c ^ int(column_meta[col_a].combined_signature))
                for col_b in sig_to_w3.get(int(sig_need), ()):
                    if int(col_b) <= int(col_a):
                        continue
                    ordered = (int(col_a), int(col_b), int(col_c))
                    key = ("same_round_split", ordered)
                    if key in seen_keys:
                        continue
                    relation = ExactTriangleRelation(
                        sector=str(sector),
                        round_lo=int(round_index),
                        round_hi=int(round_index),
                        relation_kind="same_round_split",
                        columns=ordered,
                        column_metadata=(column_meta[col_a], column_meta[col_b], column_meta[col_c]),
                    )
                    if relation_holds_exactly(relation, matrix=matrix, observables=observables):
                        seen_keys.add(key)
                        relations.append(relation)

    for round_index in range(int(metadata.total_rounds) - 1):
        bridge_cols = sorted(
            int(col)
            for col in by_window_and_kind.get((int(round_index), int(round_index + 1), "bridge_consecutive_rounds"), ())
        )
        if int(round_index + 1) == int(metadata.total_rounds - 1):
            bridge_cols.extend(
                int(col)
                for col in by_window_and_kind.get((int(round_index), int(round_index + 1), "bridge_to_final_round"), ())
            )
            bridge_cols = sorted(set(int(col) for col in bridge_cols))
        within_cols = sorted(
            set(
                int(col)
                for col in by_round_and_kind.get((int(round_index), "within_round"), ())
            )
            | set(
                int(col)
                for col in by_round_and_kind.get((int(round_index + 1), "within_round"), ())
            )
        )
        sig_to_within: dict[int, list[int]] = {}
        for col in within_cols:
            sig_to_within.setdefault(int(column_meta[col].combined_signature), []).append(int(col))
        for idx_a, col_a in enumerate(bridge_cols):
            sig_a = int(column_meta[col_a].combined_signature)
            for col_b in bridge_cols[idx_a + 1 :]:
                sig_need = int(sig_a ^ int(column_meta[col_b].combined_signature))
                for col_c in sig_to_within.get(int(sig_need), ()):
                    ordered_bridges = tuple(sorted((int(col_a), int(col_b))))
                    ordered = (int(ordered_bridges[0]), int(ordered_bridges[1]), int(col_c))
                    key = ("adjacent_round_mixed", ordered)
                    if key in seen_keys:
                        continue
                    relation = ExactTriangleRelation(
                        sector=str(sector),
                        round_lo=int(round_index),
                        round_hi=int(round_index + 1),
                        relation_kind="adjacent_round_mixed",
                        columns=ordered,
                        column_metadata=(column_meta[ordered[0]], column_meta[ordered[1]], column_meta[ordered[2]]),
                    )
                    if relation_holds_exactly(relation, matrix=matrix, observables=observables):
                        seen_keys.add(key)
                        relations.append(relation)

    relations.sort(key=lambda item: item.selection_key)
    counts: dict[str, int] = {}
    for relation in relations:
        counts[str(relation.relation_kind)] = counts.get(str(relation.relation_kind), 0) + 1
    return ExactTriangleCatalog(
        sector=str(sector),
        relations=tuple(relations),
        counts_by_kind=counts,
    )


def relation_holds_exactly(
    relation: ExactTriangleRelation,
    *,
    matrix: sp.csr_matrix,
    observables: sp.csr_matrix,
) -> bool:
    vec = np.zeros(int(matrix.shape[1]), dtype=np.uint8)
    for col in relation.columns:
        vec[int(col)] ^= 1
    return bool(
        np.count_nonzero(csr_matvec_mod2(matrix, vec)) == 0
        and np.count_nonzero(csr_matvec_mod2(observables, vec)) == 0
    )


def select_nonoverlapping_triangle_relations(catalog: ExactTriangleCatalog) -> NonOverlappingTriangleSelection:
    used: set[int] = set()
    selected: list[ExactTriangleRelation] = []
    residual: list[ExactTriangleRelation] = []
    selected_counts: dict[str, int] = {}
    overlap_counts: dict[str, int] = {}
    relation_index_by_column: dict[int, int] = {}
    for relation in catalog.relations:
        cols = tuple(int(col) for col in relation.columns)
        if any(int(col) in used for col in cols):
            residual.append(relation)
            overlap_counts[str(relation.relation_kind)] = overlap_counts.get(str(relation.relation_kind), 0) + 1
            continue
        relation_idx = int(len(selected))
        selected.append(relation)
        selected_counts[str(relation.relation_kind)] = selected_counts.get(str(relation.relation_kind), 0) + 1
        for col in cols:
            used.add(int(col))
            relation_index_by_column[int(col)] = int(relation_idx)
    return NonOverlappingTriangleSelection(
        sector=str(catalog.sector),
        selected_relations=tuple(selected),
        residual_relations=tuple(residual),
        counts_by_kind=dict(catalog.counts_by_kind),
        selected_counts_by_kind=selected_counts,
        overlapping_counts_by_kind=overlap_counts,
        used_columns=frozenset(int(col) for col in used),
        relation_index_by_column=relation_index_by_column,
    )


def exact_triangle_state_probabilities(
    p_a: float,
    p_b: float,
    p_c: float,
    *,
    eps: float = 1e-15,
) -> np.ndarray:
    pa = float(np.clip(float(p_a), float(eps), 1.0 - float(eps)))
    pb = float(np.clip(float(p_b), float(eps), 1.0 - float(eps)))
    pc = float(np.clip(float(p_c), float(eps), 1.0 - float(eps)))
    mu_00 = (1.0 - pa) * (1.0 - pb) * (1.0 - pc) + pa * pb * pc
    mu_10 = pa * (1.0 - pb) * (1.0 - pc) + (1.0 - pa) * pb * pc
    mu_01 = (1.0 - pa) * pb * (1.0 - pc) + pa * (1.0 - pb) * pc
    mu_11 = pa * pb * (1.0 - pc) + (1.0 - pa) * (1.0 - pb) * pc
    return np.asarray([mu_00, mu_10, mu_01, mu_11], dtype=np.float64)


def build_exact_triangle_factor(
    relation: ExactTriangleRelation,
    priors: np.ndarray,
    *,
    eps: float = 1e-15,
) -> ExactTriangleFactor:
    priors_vec = np.asarray(priors, dtype=np.float64).reshape(-1)
    col_a, col_b, col_c = (int(value) for value in relation.columns)
    probs = exact_triangle_state_probabilities(
        float(priors_vec[col_a]),
        float(priors_vec[col_b]),
        float(priors_vec[col_c]),
        eps=float(eps),
    )
    clipped = np.clip(probs, float(eps), 1.0)
    costs = -np.log(clipped)

    raw_priors = np.asarray(
        [
            float(np.clip(float(priors_vec[col_a]), float(eps), 1.0 - float(eps))),
            float(np.clip(float(priors_vec[col_b]), float(eps), 1.0 - float(eps))),
            float(np.clip(float(priors_vec[col_c]), float(eps), 1.0 - float(eps))),
        ],
        dtype=np.float64,
    )
    assignment_neglog = np.zeros((4, 2), dtype=np.float64)
    representative_bits: list[tuple[int, int, int]] = []
    state_uv = ((0, 0), (1, 0), (0, 1), (1, 1))
    for state_idx, (u_val, v_val) in enumerate(state_uv):
        candidate_bits = ((int(u_val), int(v_val), 0), (1 - int(u_val), 1 - int(v_val), 1))
        best_cost = float("inf")
        best_bits = candidate_bits[0]
        for t_idx, bits in enumerate(candidate_bits):
            q = np.where(np.asarray(bits, dtype=np.uint8) > 0, raw_priors, 1.0 - raw_priors)
            neglog = float(-np.sum(np.log(q)))
            assignment_neglog[state_idx, t_idx] = float(neglog)
            if neglog < best_cost - 1e-15:
                best_cost = float(neglog)
                best_bits = tuple(int(bit) for bit in bits)
        representative_bits.append(tuple(int(bit) for bit in best_bits))
    return ExactTriangleFactor(
        relation=relation,
        state_probabilities=np.asarray(probs, dtype=np.float64),
        state_costs=np.asarray(costs, dtype=np.float64),
        representative_bits=tuple(representative_bits),
        assignment_neglog=assignment_neglog,
    )


def toggle_support_delta_nll(
    estimate: np.ndarray,
    llr: np.ndarray,
    support: tuple[int, ...] | list[int] | np.ndarray,
) -> float:
    est = (np.asarray(estimate, dtype=np.uint8).reshape(-1) & 1).astype(np.float64, copy=False)
    llr_vec = np.asarray(llr, dtype=np.float64).reshape(-1)
    idx = np.asarray(sorted({int(col) for col in support}), dtype=np.int32)
    if idx.size == 0:
        return 0.0
    return float(np.sum((1.0 - 2.0 * est[idx]) * llr_vec[idx], dtype=np.float64))


def apply_triangle_gauge_descent(
    *,
    estimate: np.ndarray,
    llr: np.ndarray,
    relations: tuple[ExactTriangleRelation, ...] | list[ExactTriangleRelation],
    matrix: sp.csr_matrix,
    observables: sp.csr_matrix,
    max_iterations: int = 128,
) -> TriangleGaugeDescentResult:
    current = (np.asarray(estimate, dtype=np.uint8).reshape(-1) & 1).copy()
    llr_vec = np.asarray(llr, dtype=np.float64).reshape(-1)
    target_syndrome = csr_matvec_mod2(matrix, current)
    target_logical = csr_matvec_mod2(observables, current)
    accepted: list[TriangleGaugeStep] = []
    converged = True
    for iteration in range(max(1, int(max_iterations))):
        best: tuple[float, tuple[int, int, int], ExactTriangleRelation] | None = None
        for relation in relations:
            support = tuple(int(col) for col in relation.columns)
            delta = float(toggle_support_delta_nll(current, llr_vec, support))
            if not np.isfinite(delta) or delta >= -1e-12:
                continue
            candidate_key = (float(delta), support, relation)
            if best is None or candidate_key[0] < best[0] - 1e-12 or (
                abs(candidate_key[0] - best[0]) <= 1e-12 and candidate_key[1] < best[1]
            ):
                best = candidate_key
        if best is None:
            return TriangleGaugeDescentResult(
                estimate=current,
                accepted_steps=tuple(accepted),
                iterations=int(iteration),
                converged=True,
            )
        support = best[1]
        relation = best[2]
        for col in support:
            current[int(col)] ^= 1
        if not np.array_equal(csr_matvec_mod2(matrix, current), target_syndrome):
            raise ValueError("accepted gauge move changed detector syndrome")
        if not np.array_equal(csr_matvec_mod2(observables, current), target_logical):
            raise ValueError("accepted gauge move changed logical label")
        accepted.append(
            TriangleGaugeStep(
                relation=relation,
                delta_cost=float(best[0]),
                support=tuple(int(col) for col in support),
            )
        )
    converged = False
    return TriangleGaugeDescentResult(
        estimate=current,
        accepted_steps=tuple(accepted),
        iterations=int(max(1, int(max_iterations))),
        converged=bool(converged),
    )


def build_cached_triangle_structures(
    *,
    matrix: sp.csr_matrix,
    observables: sp.csr_matrix,
    metadata: SplitSectorMetadata,
    sector: str,
) -> tuple[ExactTriangleCatalog, NonOverlappingTriangleSelection]:
    catalog = catalog_exact_local_triangles(
        matrix=matrix,
        observables=observables,
        metadata=metadata,
        sector=str(sector),
    )
    return catalog, select_nonoverlapping_triangle_relations(catalog)


def llr_to_priors(llr: np.ndarray, *, eps: float = 1e-15) -> np.ndarray:
    llr_vec = np.asarray(llr, dtype=np.float64).reshape(-1)
    probs = 1.0 / (1.0 + np.exp(np.clip(llr_vec, -60.0, 60.0)))
    return np.clip(probs, float(eps), 1.0 - float(eps))


def priors_to_llr(priors: np.ndarray) -> np.ndarray:
    return llr_from_priors(np.asarray(priors, dtype=np.float64))
