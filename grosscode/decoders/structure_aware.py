from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse as sp

from grosscode.dem.builder import SplitSectorMetadata
from grosscode.dem.triangles import (
    ExactTriangleFactor,
    ExactTriangleRelation,
    NonOverlappingTriangleSelection,
    TriangleGaugeDescentResult,
    apply_triangle_gauge_descent,
    build_cached_triangle_structures,
    exact_triangle_state_probabilities,
    llr_to_priors,
)
from grosscode.utils.gf2 import csr_matvec_mod2


LayeredScheduleMode = Literal["off", "auto", "forward", "backward"]
TriangleFactorizationMode = Literal["off", "nonoverlap"]


@dataclass(frozen=True)
class SectorStructureModel:
    catalog_size: int
    counts_by_kind: dict[str, int]
    selection: NonOverlappingTriangleSelection


@dataclass(frozen=True)
class ReducedVariable:
    kind: str
    global_columns: tuple[int, ...]
    local_columns: tuple[int, ...]
    check_rows: np.ndarray
    parity_by_state: np.ndarray
    state_costs: np.ndarray
    factor: ExactTriangleFactor | None = None


@dataclass(frozen=True)
class ReducedLocalModel:
    variables: tuple[ReducedVariable, ...]
    check_to_edges: tuple[np.ndarray, ...]
    edge_var: np.ndarray
    edge_check: np.ndarray
    edge_slot: np.ndarray
    local_cols: np.ndarray
    reduced_triangle_count: int
    residual_triangle_count: int
    used_triangle_relations: tuple[ExactTriangleRelation, ...]


@dataclass(frozen=True)
class ReducedSolveResult:
    bits_local: np.ndarray
    llr_local: np.ndarray
    mean_llr_local: np.ndarray
    converged: bool
    iterations: int
    residual: np.ndarray
    reduced_triangle_count: int
    residual_triangle_count: int
    reduced_variable_count: int


@dataclass(frozen=True)
class ScheduleDirectionDecision:
    requested: str
    resolved: str
    boundary_row_weight_first: float
    boundary_row_weight_last: float


def build_sector_structure_model(
    *,
    matrix: sp.csr_matrix,
    observables: sp.csr_matrix,
    metadata: SplitSectorMetadata,
    sector: str,
) -> SectorStructureModel:
    catalog, selection = build_cached_triangle_structures(
        matrix=matrix,
        observables=observables,
        metadata=metadata,
        sector=str(sector),
    )
    return SectorStructureModel(
        catalog_size=int(len(catalog.relations)),
        counts_by_kind=dict(catalog.counts_by_kind),
        selection=selection,
    )


def resolve_schedule_direction(
    *,
    matrix: sp.csr_matrix,
    metadata: SplitSectorMetadata,
    requested: LayeredScheduleMode,
) -> ScheduleDirectionDecision:
    row_weights = np.diff(matrix.indptr).astype(np.float64, copy=False)
    first_start = int(metadata.detector_round_slices[0][1])
    first_stop = int(metadata.detector_round_slices[0][2])
    last_start = int(metadata.detector_round_slices[-1][1])
    last_stop = int(metadata.detector_round_slices[-1][2])
    first_weight = float(np.mean(row_weights[first_start:first_stop])) if first_stop > first_start else 0.0
    last_weight = float(np.mean(row_weights[last_start:last_stop])) if last_stop > last_start else 0.0
    if str(requested) == "forward":
        resolved = "forward"
    elif str(requested) == "backward":
        resolved = "backward"
    elif str(requested) == "auto":
        resolved = "forward" if first_weight <= last_weight else "backward"
    else:
        resolved = "forward"
    return ScheduleDirectionDecision(
        requested=str(requested),
        resolved=str(resolved),
        boundary_row_weight_first=float(first_weight),
        boundary_row_weight_last=float(last_weight),
    )


def apply_sector_gauge_descent(
    *,
    estimate: np.ndarray,
    priors: np.ndarray,
    matrix: sp.csr_matrix,
    observables: sp.csr_matrix,
    selection: NonOverlappingTriangleSelection,
    max_iterations: int = 128,
) -> TriangleGaugeDescentResult:
    llr = np.log((1.0 - np.clip(np.asarray(priors, dtype=np.float64), 1e-15, 1.0 - 1e-15)) / np.clip(np.asarray(priors, dtype=np.float64), 1e-15, 1.0 - 1e-15))
    return apply_triangle_gauge_descent(
        estimate=np.asarray(estimate, dtype=np.uint8),
        llr=llr,
        relations=selection.selected_relations,
        matrix=matrix,
        observables=observables,
        max_iterations=int(max_iterations),
    )


def build_reduced_local_model(
    *,
    local_sub_matrix: sp.csr_matrix,
    local_cols: np.ndarray,
    local_prior_llr: np.ndarray,
    selection: NonOverlappingTriangleSelection,
    clamp_assignments: dict[int, int] | None = None,
) -> ReducedLocalModel:
    csc = local_sub_matrix.tocsc()
    local_cols_arr = np.asarray(local_cols, dtype=np.int32)
    local_prior = np.asarray(local_prior_llr, dtype=np.float64).reshape(-1)
    global_to_local = {int(col): int(idx) for idx, col in enumerate(local_cols_arr.tolist())}
    clamped_local = set(int(key) for key in (clamp_assignments or {}).keys())
    reduced_relations: list[tuple[ExactTriangleRelation, tuple[int, int, int], ExactTriangleFactor]] = []
    covered_local: set[int] = set()
    residual_triangle_count = 0
    local_priors = llr_to_priors(local_prior)

    for relation in selection.selected_relations:
        cols_global = tuple(int(col) for col in relation.columns)
        if any(int(col) not in global_to_local for col in cols_global):
            continue
        cols_local = tuple(int(global_to_local[int(col)]) for col in cols_global)
        if any(int(local_idx) in clamped_local for local_idx in cols_local):
            residual_triangle_count += 1
            continue
        if any(int(local_idx) in covered_local for local_idx in cols_local):
            residual_triangle_count += 1
            continue
        probs = local_priors[np.asarray(cols_local, dtype=np.int32)]
        state_probabilities = exact_triangle_state_probabilities(float(probs[0]), float(probs[1]), float(probs[2]))
        state_costs = -np.log(np.clip(state_probabilities, 1e-15, 1.0))
        assignment_neglog = np.zeros((4, 2), dtype=np.float64)
        representative_bits: list[tuple[int, int, int]] = []
        for state_idx, (u_val, v_val) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
            candidates = ((int(u_val), int(v_val), 0), (1 - int(u_val), 1 - int(v_val), 1))
            best_cost = float("inf")
            best_bits = candidates[0]
            for t_idx, bits in enumerate(candidates):
                q = np.where(np.asarray(bits, dtype=np.uint8) > 0, probs, 1.0 - probs)
                neglog = float(-np.sum(np.log(np.clip(q, 1e-15, 1.0))))
                assignment_neglog[state_idx, t_idx] = float(neglog)
                if neglog < best_cost - 1e-15:
                    best_cost = float(neglog)
                    best_bits = tuple(int(bit) for bit in bits)
            representative_bits.append(tuple(int(bit) for bit in best_bits))
        factor = ExactTriangleFactor(
            relation=relation,
            state_probabilities=np.asarray(state_probabilities, dtype=np.float64),
            state_costs=np.asarray(state_costs, dtype=np.float64),
            representative_bits=tuple(representative_bits),
            assignment_neglog=assignment_neglog,
        )
        reduced_relations.append((relation, cols_local, factor))
        covered_local.update(int(local_idx) for local_idx in cols_local)

    variables: list[ReducedVariable] = []
    relation_set = {tuple(int(local_idx) for local_idx in cols_local): (relation, factor) for relation, cols_local, factor in reduced_relations}
    ordinary_locals = [int(idx) for idx in range(int(local_cols_arr.size)) if int(idx) not in covered_local]
    for local_idx in ordinary_locals:
        begin = int(csc.indptr[int(local_idx)])
        end = int(csc.indptr[int(local_idx) + 1])
        rows = np.asarray(csc.indices[begin:end], dtype=np.int32)
        parity = np.asarray([[0, 1]] * int(rows.size), dtype=np.uint8)
        p1 = float(np.clip(local_priors[int(local_idx)], 1e-15, 1.0 - 1e-15))
        state_costs = np.asarray([-np.log(1.0 - p1), -np.log(p1)], dtype=np.float64)
        variables.append(
            ReducedVariable(
                kind="ordinary",
                global_columns=(int(local_cols_arr[int(local_idx)]),),
                local_columns=(int(local_idx),),
                check_rows=rows,
                parity_by_state=parity,
                state_costs=state_costs,
                factor=None,
            )
        )
    for relation, cols_local, factor in reduced_relations:
        col_a, col_b, _col_c = (int(value) for value in cols_local)
        begin_a = int(csc.indptr[int(col_a)])
        end_a = int(csc.indptr[int(col_a) + 1])
        begin_b = int(csc.indptr[int(col_b)])
        end_b = int(csc.indptr[int(col_b) + 1])
        rows_a = set(int(row) for row in np.asarray(csc.indices[begin_a:end_a], dtype=np.int32).tolist())
        rows_b = set(int(row) for row in np.asarray(csc.indices[begin_b:end_b], dtype=np.int32).tolist())
        rows = np.asarray(sorted(rows_a | rows_b), dtype=np.int32)
        parity = np.zeros((int(rows.size), 4), dtype=np.uint8)
        for row_slot, row in enumerate(rows.tolist()):
            a_touch = 1 if int(row) in rows_a else 0
            b_touch = 1 if int(row) in rows_b else 0
            parity[row_slot, 0] = 0
            parity[row_slot, 1] = np.uint8(a_touch)
            parity[row_slot, 2] = np.uint8(b_touch)
            parity[row_slot, 3] = np.uint8(a_touch ^ b_touch)
        variables.append(
            ReducedVariable(
                kind="triangle",
                global_columns=tuple(int(local_cols_arr[int(local_idx)]) for local_idx in cols_local),
                local_columns=tuple(int(local_idx) for local_idx in cols_local),
                check_rows=rows,
                parity_by_state=parity,
                state_costs=np.asarray(factor.state_costs, dtype=np.float64),
                factor=factor,
            )
        )

    check_to_edges: list[list[int]] = [[] for _ in range(int(local_sub_matrix.shape[0]))]
    edge_var: list[int] = []
    edge_check: list[int] = []
    edge_slot: list[int] = []
    for var_idx, variable in enumerate(variables):
        for slot, row in enumerate(np.asarray(variable.check_rows, dtype=np.int32).tolist()):
            edge_id = int(len(edge_var))
            edge_var.append(int(var_idx))
            edge_check.append(int(row))
            edge_slot.append(int(slot))
            check_to_edges[int(row)].append(int(edge_id))
    return ReducedLocalModel(
        variables=tuple(variables),
        check_to_edges=tuple(np.asarray(edges, dtype=np.int32) for edges in check_to_edges),
        edge_var=np.asarray(edge_var, dtype=np.int32),
        edge_check=np.asarray(edge_check, dtype=np.int32),
        edge_slot=np.asarray(edge_slot, dtype=np.int32),
        local_cols=local_cols_arr.copy(),
        reduced_triangle_count=int(len(reduced_relations)),
        residual_triangle_count=int(residual_triangle_count),
        used_triangle_relations=tuple(relation for relation, _cols, _factor in reduced_relations),
    )


def _edge_parity(variable: ReducedVariable, edge_slot: int) -> np.ndarray:
    return np.asarray(variable.parity_by_state[int(edge_slot)], dtype=np.uint8)


def solve_reduced_local_model(
    *,
    model: ReducedLocalModel,
    syndrome: np.ndarray,
    max_iter: int,
) -> ReducedSolveResult:
    syndrome_bits = np.asarray(syndrome, dtype=np.uint8).reshape(-1) & 1
    if int(model.local_cols.size) == 0:
        return ReducedSolveResult(
            bits_local=np.zeros(0, dtype=np.uint8),
            llr_local=np.zeros(0, dtype=np.float64),
            mean_llr_local=np.zeros(0, dtype=np.float64),
            converged=True,
            iterations=0,
            residual=syndrome_bits.copy(),
            reduced_triangle_count=int(model.reduced_triangle_count),
            residual_triangle_count=int(model.residual_triangle_count),
            reduced_variable_count=0,
        )

    m_vc = [np.asarray(variable.state_costs, dtype=np.float64).copy() for variable in (model.variables[int(var)] for var in model.edge_var)]
    for payload in m_vc:
        payload -= float(np.min(payload))
    m_cv = [np.zeros_like(payload) for payload in m_vc]
    belief = [np.asarray(variable.state_costs, dtype=np.float64).copy() for variable in model.variables]
    decoded_state = [0 for _ in model.variables]
    residual = np.ones(int(syndrome_bits.size), dtype=np.uint8)
    iterations = 0

    var_to_edges: list[list[int]] = [[] for _ in model.variables]
    for edge_idx, var_idx in enumerate(model.edge_var.tolist()):
        var_to_edges[int(var_idx)].append(int(edge_idx))

    for it in range(1, max(1, int(max_iter)) + 1):
        iterations = int(it)
        for check_idx, edges in enumerate(model.check_to_edges):
            if int(edges.size) == 0:
                continue
            parity_costs: list[tuple[float, float]] = []
            for edge_idx in edges.tolist():
                var_idx = int(model.edge_var[int(edge_idx)])
                variable = model.variables[var_idx]
                parity = _edge_parity(variable, int(model.edge_slot[int(edge_idx)]))
                cost = np.asarray(m_vc[int(edge_idx)], dtype=np.float64)
                even_mask = parity == 0
                odd_mask = parity == 1
                best_even = float(np.min(cost[even_mask])) if np.any(even_mask) else float("inf")
                best_odd = float(np.min(cost[odd_mask])) if np.any(odd_mask) else float("inf")
                parity_costs.append((best_even, best_odd))
            for local_edge_pos, edge_idx in enumerate(edges.tolist()):
                dp0 = 0.0
                dp1 = float("inf")
                for other_pos, _other_edge in enumerate(edges.tolist()):
                    if int(other_pos) == int(local_edge_pos):
                        continue
                    best_even, best_odd = parity_costs[int(other_pos)]
                    next0 = min(dp0 + best_even, dp1 + best_odd)
                    next1 = min(dp0 + best_odd, dp1 + best_even)
                    dp0 = float(next0)
                    dp1 = float(next1)
                var_idx = int(model.edge_var[int(edge_idx)])
                variable = model.variables[var_idx]
                parity = _edge_parity(variable, int(model.edge_slot[int(edge_idx)]))
                msg = np.empty_like(np.asarray(m_cv[int(edge_idx)], dtype=np.float64))
                target = int(syndrome_bits[int(check_idx)])
                for state_idx in range(int(msg.size)):
                    need = int(target ^ int(parity[int(state_idx)]))
                    msg[state_idx] = float(dp0 if need == 0 else dp1)
                msg -= float(np.min(msg))
                m_cv[int(edge_idx)] = msg
        for var_idx, variable in enumerate(model.variables):
            cur = np.asarray(variable.state_costs, dtype=np.float64).copy()
            for edge_idx in var_to_edges[int(var_idx)]:
                cur += np.asarray(m_cv[int(edge_idx)], dtype=np.float64)
            belief[int(var_idx)] = cur
            decoded_state[int(var_idx)] = int(np.argmin(cur))
            for edge_idx in var_to_edges[int(var_idx)]:
                msg = cur - np.asarray(m_cv[int(edge_idx)], dtype=np.float64)
                msg -= float(np.min(msg))
                m_vc[int(edge_idx)] = msg
        residual = np.zeros(int(syndrome_bits.size), dtype=np.uint8)
        for check_idx, edges in enumerate(model.check_to_edges):
            parity_total = 0
            for edge_idx in edges.tolist():
                var_idx = int(model.edge_var[int(edge_idx)])
                variable = model.variables[var_idx]
                parity = _edge_parity(variable, int(model.edge_slot[int(edge_idx)]))
                parity_total ^= int(parity[int(decoded_state[int(var_idx)])])
            residual[int(check_idx)] = np.uint8(int(parity_total) ^ int(syndrome_bits[int(check_idx)]))
        if int(np.count_nonzero(residual)) == 0:
            break

    bits_local = np.zeros(int(model.local_cols.size), dtype=np.uint8)
    llr_local = np.zeros(int(model.local_cols.size), dtype=np.float64)
    for var_idx, variable in enumerate(model.variables):
        cur_belief = np.asarray(belief[int(var_idx)], dtype=np.float64)
        if str(variable.kind) == "ordinary":
            local_idx = int(variable.local_columns[0])
            bits_local[local_idx] = np.uint8(int(decoded_state[int(var_idx)]))
            llr_local[local_idx] = float(cur_belief[1] - cur_belief[0])
            continue
        factor = variable.factor
        if factor is None:
            continue
        chosen_state = int(decoded_state[int(var_idx)])
        chosen_bits = factor.representative_bits[chosen_state]
        external_cost = cur_belief - np.asarray(factor.state_costs, dtype=np.float64)
        candidate_bits = (
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
        )
        for local_slot, local_idx in enumerate(variable.local_columns):
            bits_local[int(local_idx)] = np.uint8(int(chosen_bits[int(local_slot)]))
            cost0 = float("inf")
            cost1 = float("inf")
            for state_idx, (u_val, v_val, _unused) in enumerate(candidate_bits):
                for t_idx, bits in enumerate(((u_val, v_val, 0), (1 - u_val, 1 - v_val, 1))):
                    cost = float(external_cost[int(state_idx)] + factor.assignment_neglog[int(state_idx), int(t_idx)])
                    if int(bits[int(local_slot)]) == 0:
                        cost0 = min(cost0, cost)
                    else:
                        cost1 = min(cost1, cost)
            llr_local[int(local_idx)] = float(cost1 - cost0)
    return ReducedSolveResult(
        bits_local=bits_local,
        llr_local=llr_local,
        mean_llr_local=llr_local.copy(),
        converged=bool(np.count_nonzero(residual) == 0),
        iterations=int(iterations),
        residual=residual,
        reduced_triangle_count=int(model.reduced_triangle_count),
        residual_triangle_count=int(model.residual_triangle_count),
        reduced_variable_count=int(len(model.variables)),
    )


def virtual_column_span_arrays(
    metadata: SplitSectorMetadata,
    *,
    direction: str,
) -> tuple[np.ndarray, np.ndarray]:
    if str(direction) != "backward":
        return (
            np.asarray(metadata.column_round_start, dtype=np.int16).copy(),
            np.asarray(metadata.column_round_stop, dtype=np.int16).copy(),
        )
    total_rounds = int(metadata.total_rounds)
    return (
        (int(total_rounds - 1) - np.asarray(metadata.column_round_stop, dtype=np.int16)).astype(np.int16, copy=False),
        (int(total_rounds - 1) - np.asarray(metadata.column_round_start, dtype=np.int16)).astype(np.int16, copy=False),
    )


def actual_interval_from_virtual(
    *,
    total_rounds: int,
    start_round: int,
    end_round: int,
    direction: str,
) -> tuple[int, int]:
    if str(direction) != "backward":
        return (int(start_round), int(end_round))
    actual_start = int(total_rounds - 1 - int(end_round))
    actual_end = int(total_rounds - 1 - int(start_round))
    return (int(actual_start), int(actual_end))


def virtual_commit_mask(
    metadata: SplitSectorMetadata,
    active_cols: np.ndarray,
    *,
    commit_end_round: int,
    direction: str,
) -> np.ndarray:
    span_start, span_stop = virtual_column_span_arrays(metadata, direction=str(direction))
    active = np.asarray(active_cols, dtype=np.int32)
    return np.asarray(span_stop[active] <= int(commit_end_round), dtype=bool)


def virtual_separator_local_columns(
    metadata: SplitSectorMetadata,
    active_cols: np.ndarray,
    *,
    commit_end_round: int,
    direction: str,
) -> np.ndarray:
    span_start, span_stop = virtual_column_span_arrays(metadata, direction=str(direction))
    active = np.asarray(active_cols, dtype=np.int32)
    crossing = np.flatnonzero(
        (span_start[active] <= int(commit_end_round)) & (span_stop[active] > int(commit_end_round))
    ).astype(np.int32, copy=False)
    if int(crossing.size) > 0:
        return crossing
    future = np.flatnonzero(span_stop[active] > int(commit_end_round)).astype(np.int32, copy=False)
    if int(future.size) > 0:
        return future
    return np.zeros(0, dtype=np.int32)


def validate_gauge_invariance(
    *,
    before: np.ndarray,
    after: np.ndarray,
    matrix: sp.csr_matrix,
    observables: sp.csr_matrix,
) -> bool:
    before_bits = np.asarray(before, dtype=np.uint8).reshape(-1) & 1
    after_bits = np.asarray(after, dtype=np.uint8).reshape(-1) & 1
    return bool(
        np.array_equal(csr_matvec_mod2(matrix, before_bits), csr_matvec_mod2(matrix, after_bits))
        and np.array_equal(csr_matvec_mod2(observables, before_bits), csr_matvec_mod2(observables, after_bits))
    )
