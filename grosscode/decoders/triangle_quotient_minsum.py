from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from grosscode.dem.reference_recovery import AugmentedReferenceRecoverySolver
from grosscode.dem.triangle_basis import TriangleBasisArtifact
from grosscode.utils.gf2 import binary_csr_mod2, csr_matvec_mod2, dense_mod2


@dataclass(frozen=True)
class TriangleQuotientMinSumConfig:
    max_iter: int = 60
    damping: float = 0.0
    factor_scale: float = 1.0
    convergence_tol: float = 1e-9
    stable_rounds_required: int = 2
    local_repair_max_vars: int = 0
    local_repair_margin: float = 0.25


@dataclass(frozen=True)
class TriangleQuotientFactorGraph:
    basis_supports: np.ndarray
    factor_to_vars: tuple[np.ndarray, ...]
    factor_to_slots: tuple[np.ndarray, ...]
    edge_count: int
    max_factor_degree: int


@dataclass(frozen=True)
class TriangleQuotientDecodeResult:
    e_hat: np.ndarray
    u_hat: np.ndarray
    reference: np.ndarray
    target_syndrome: np.ndarray
    target_logical: np.ndarray
    exact_target_ok: bool
    converged: bool
    iterations: int
    stable_rounds: int
    objective_cost: float
    edge_update_proxy: int
    max_message_delta: float
    uncertain_var_count: int
    belief_delta: np.ndarray
    factor_degree_histogram: dict[int, int]
    local_repair_used: bool
    local_repair_examined: int


def build_triangle_quotient_factor_graph(basis_supports: np.ndarray, n_factors: int) -> TriangleQuotientFactorGraph:
    supports = np.asarray(basis_supports, dtype=np.int32)
    incidence_lists: list[list[tuple[int, int]]] = [[] for _ in range(int(n_factors))]
    for var_idx in range(int(supports.shape[0])):
        for slot, factor_idx in enumerate(np.asarray(supports[var_idx], dtype=np.int32).tolist()):
            incidence_lists[int(factor_idx)].append((int(var_idx), int(slot)))
    factor_to_vars = tuple(
        np.asarray([item[0] for item in items], dtype=np.int32) if items else np.zeros(0, dtype=np.int32)
        for items in incidence_lists
    )
    factor_to_slots = tuple(
        np.asarray([item[1] for item in items], dtype=np.int8) if items else np.zeros(0, dtype=np.int8)
        for items in incidence_lists
    )
    max_degree = max((int(arr.size) for arr in factor_to_vars), default=0)
    return TriangleQuotientFactorGraph(
        basis_supports=supports,
        factor_to_vars=factor_to_vars,
        factor_to_slots=factor_to_slots,
        edge_count=int(supports.size),
        max_factor_degree=int(max_degree),
    )


def _normalize_message(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    out -= float(np.min(out))
    return out


def _compute_error_from_u(reference: np.ndarray, basis_supports: np.ndarray, u_hat: np.ndarray) -> np.ndarray:
    out = dense_mod2(reference).reshape(-1).copy()
    active = np.flatnonzero(dense_mod2(u_hat).reshape(-1))
    supports = np.asarray(basis_supports, dtype=np.int32)
    for var_idx in active.tolist():
        out[np.asarray(supports[int(var_idx)], dtype=np.int32)] ^= 1
    return out


def _objective_cost_from_error(cost_zero: np.ndarray, cost_one: np.ndarray, error_bits: np.ndarray) -> float:
    bits = dense_mod2(error_bits).reshape(-1)
    return float(
        np.sum(
            np.where(bits > 0, np.asarray(cost_one, dtype=np.float64), np.asarray(cost_zero, dtype=np.float64)),
            dtype=np.float64,
        )
    )


def _factor_degree_histogram(graph: TriangleQuotientFactorGraph) -> dict[int, int]:
    degrees = [int(items.size) for items in graph.factor_to_vars]
    if not degrees:
        return {}
    values, counts = np.unique(np.asarray(degrees, dtype=np.int32), return_counts=True)
    return {int(v): int(c) for v, c in zip(values.tolist(), counts.tolist())}


class TriangleQuotientMinSumDecoder:
    def __init__(
        self,
        *,
        detector_matrix: sp.csr_matrix,
        logical_matrix: sp.csr_matrix,
        priors: np.ndarray,
        basis_artifact: TriangleBasisArtifact,
        reference_solver: AugmentedReferenceRecoverySolver | None = None,
        config: TriangleQuotientMinSumConfig | None = None,
    ) -> None:
        cfg = TriangleQuotientMinSumConfig() if config is None else config
        if int(cfg.max_iter) <= 0:
            raise ValueError("max_iter must be positive")
        if not (0.0 <= float(cfg.damping) < 1.0):
            raise ValueError("damping must lie in [0,1)")
        if float(cfg.factor_scale) <= 0.0:
            raise ValueError("factor_scale must be positive")
        self.config = cfg
        self.detector_matrix = binary_csr_mod2(detector_matrix).tocsr()
        self.logical_matrix = binary_csr_mod2(logical_matrix).tocsr()
        self.reference_solver = (
            AugmentedReferenceRecoverySolver.build(detector_matrix=self.detector_matrix, logical_matrix=self.logical_matrix)
            if reference_solver is None
            else reference_solver
        )
        if int(self.detector_matrix.shape[1]) != int(basis_artifact.detector_shape[1]):
            raise ValueError("basis artifact width does not match detector matrix")
        self.basis_artifact = basis_artifact
        self.graph = build_triangle_quotient_factor_graph(
            np.asarray(basis_artifact.basis_supports, dtype=np.int32),
            int(self.detector_matrix.shape[1]),
        )
        clipped = np.clip(np.asarray(priors, dtype=np.float64).reshape(-1), 1e-15, 1.0 - 1e-15)
        self.cost_zero = -np.log(1.0 - clipped)
        self.cost_one = -np.log(clipped)

    @property
    def n(self) -> int:
        return int(self.detector_matrix.shape[1])

    @property
    def n_triangle_vars(self) -> int:
        return int(self.basis_artifact.basis_rank)

    def _factor_costs(self, reference: np.ndarray) -> np.ndarray:
        ref = dense_mod2(reference).reshape(-1)
        return np.stack(
            [
                np.where(ref > 0, self.cost_one, self.cost_zero),
                np.where(ref > 0, self.cost_zero, self.cost_one),
            ],
            axis=1,
        ).astype(np.float64, copy=False)

    def _local_repair(
        self,
        *,
        reference: np.ndarray,
        hard_u: np.ndarray,
        belief_delta: np.ndarray,
    ) -> tuple[np.ndarray, bool, int]:
        max_vars = int(self.config.local_repair_max_vars)
        if max_vars <= 0:
            return np.asarray(hard_u, dtype=np.uint8).copy(), False, 0
        margins = np.abs(np.asarray(belief_delta, dtype=np.float64))
        uncertain = np.flatnonzero(margins <= float(self.config.local_repair_margin)).astype(np.int32, copy=False)
        if uncertain.size == 0:
            order = np.argsort(margins, kind="mergesort").astype(np.int32, copy=False)
            uncertain = order[:max_vars]
        else:
            uncertain = uncertain[np.argsort(margins[uncertain], kind="mergesort")][:max_vars]
        if uncertain.size == 0 or int(uncertain.size) > 12:
            return np.asarray(hard_u, dtype=np.uint8).copy(), False, 0

        best_u = np.asarray(hard_u, dtype=np.uint8).copy()
        best_cost = _objective_cost_from_error(
            self.cost_zero,
            self.cost_one,
            _compute_error_from_u(reference, self.graph.basis_supports, best_u),
        )
        examined = 0
        for subset_mask in range(1 << int(uncertain.size)):
            candidate_u = np.asarray(hard_u, dtype=np.uint8).copy()
            for local_idx, var_idx in enumerate(uncertain.tolist()):
                if ((int(subset_mask) >> int(local_idx)) & 1) != 0:
                    candidate_u[int(var_idx)] ^= 1
            cost = _objective_cost_from_error(
                self.cost_zero,
                self.cost_one,
                _compute_error_from_u(reference, self.graph.basis_supports, candidate_u),
            )
            examined += 1
            if float(cost) < float(best_cost) - 1e-12:
                best_cost = float(cost)
                best_u = candidate_u
        return best_u, bool(examined > 0), int(examined)

    def decode(
        self,
        *,
        syndrome: np.ndarray,
        target_logical: np.ndarray,
    ) -> TriangleQuotientDecodeResult:
        syndrome_bits = dense_mod2(syndrome).reshape(-1)
        target_bits = dense_mod2(target_logical).reshape(-1)
        if int(syndrome_bits.size) != int(self.detector_matrix.shape[0]):
            raise ValueError("syndrome length mismatch")
        if int(target_bits.size) != int(self.logical_matrix.shape[0]):
            raise ValueError("target logical length mismatch")

        reference = self.reference_solver.solve_reference(syndrome_bits, target_bits)
        factor_costs = self._factor_costs(reference)
        n_vars = int(self.n_triangle_vars)
        v_to_f = np.zeros((n_vars, 3, 2), dtype=np.float64)
        f_to_v = np.zeros((n_vars, 3, 2), dtype=np.float64)
        belief_delta = np.zeros(n_vars, dtype=np.float64)
        previous_u = np.zeros(n_vars, dtype=np.uint8)
        max_message_delta = 0.0
        stable_rounds = 0
        iterations_used = 0
        converged = False

        for iteration in range(1, int(self.config.max_iter) + 1):
            max_message_delta = 0.0
            for factor_idx in range(self.n):
                vars_j = self.graph.factor_to_vars[int(factor_idx)]
                if vars_j.size == 0:
                    continue
                slots_j = self.graph.factor_to_slots[int(factor_idx)]
                degree = int(vars_j.size)
                incoming = np.asarray(
                    [v_to_f[int(var_idx), int(slot_idx)] for var_idx, slot_idx in zip(vars_j.tolist(), slots_j.tolist())],
                    dtype=np.float64,
                )
                prefix = np.full((degree + 1, 2), np.inf, dtype=np.float64)
                suffix = np.full((degree + 1, 2), np.inf, dtype=np.float64)
                prefix[0, 0] = 0.0
                suffix[degree, 0] = 0.0
                for pos in range(degree):
                    msg = incoming[int(pos)]
                    prefix[int(pos) + 1, 0] = min(prefix[int(pos), 0] + msg[0], prefix[int(pos), 1] + msg[1])
                    prefix[int(pos) + 1, 1] = min(prefix[int(pos), 0] + msg[1], prefix[int(pos), 1] + msg[0])
                for pos in range(degree - 1, -1, -1):
                    msg = incoming[int(pos)]
                    suffix[int(pos), 0] = min(suffix[int(pos) + 1, 0] + msg[0], suffix[int(pos) + 1, 1] + msg[1])
                    suffix[int(pos), 1] = min(suffix[int(pos) + 1, 0] + msg[1], suffix[int(pos) + 1, 1] + msg[0])
                costs = np.asarray(factor_costs[int(factor_idx)], dtype=np.float64)
                for pos, (var_idx, slot_idx) in enumerate(zip(vars_j.tolist(), slots_j.tolist())):
                    excl_even = min(prefix[int(pos), 0] + suffix[int(pos) + 1, 0], prefix[int(pos), 1] + suffix[int(pos) + 1, 1])
                    excl_odd = min(prefix[int(pos), 0] + suffix[int(pos) + 1, 1], prefix[int(pos), 1] + suffix[int(pos) + 1, 0])
                    updated = np.asarray(
                        [
                            min(costs[0] + excl_even, costs[1] + excl_odd),
                            min(costs[1] + excl_even, costs[0] + excl_odd),
                        ],
                        dtype=np.float64,
                    )
                    updated = _normalize_message(updated) * float(self.config.factor_scale)
                    if float(self.config.damping) > 0.0:
                        updated = float(self.config.damping) * f_to_v[int(var_idx), int(slot_idx)] + (
                            1.0 - float(self.config.damping)
                        ) * updated
                        updated = _normalize_message(updated)
                    delta = float(np.max(np.abs(updated - f_to_v[int(var_idx), int(slot_idx)])))
                    max_message_delta = max(float(max_message_delta), delta)
                    f_to_v[int(var_idx), int(slot_idx)] = updated

            beliefs = np.sum(f_to_v, axis=1, dtype=np.float64)
            for var_idx in range(n_vars):
                belief = _normalize_message(beliefs[int(var_idx)])
                belief_delta[int(var_idx)] = float(belief[1] - belief[0])
                for slot_idx in range(3):
                    v_to_f[int(var_idx), int(slot_idx)] = _normalize_message(belief - f_to_v[int(var_idx), int(slot_idx)])
            current_u = (belief_delta < 0.0).astype(np.uint8)
            iterations_used = int(iteration)
            if np.array_equal(current_u, previous_u) and float(max_message_delta) <= float(self.config.convergence_tol):
                stable_rounds += 1
                if stable_rounds >= int(self.config.stable_rounds_required):
                    converged = True
                    previous_u = current_u
                    break
            else:
                stable_rounds = 0
            previous_u = current_u

        hard_u = np.asarray(previous_u, dtype=np.uint8).copy()
        repaired_u, repair_used, repair_examined = self._local_repair(
            reference=reference,
            hard_u=hard_u,
            belief_delta=belief_delta,
        )
        hard_u = repaired_u
        e_hat = _compute_error_from_u(reference, self.graph.basis_supports, hard_u)
        objective_cost = _objective_cost_from_error(self.cost_zero, self.cost_one, e_hat)
        exact_target_ok = bool(
            np.array_equal(csr_matvec_mod2(self.detector_matrix, e_hat), syndrome_bits)
            and np.array_equal(csr_matvec_mod2(self.logical_matrix, e_hat), target_bits)
        )
        return TriangleQuotientDecodeResult(
            e_hat=e_hat,
            u_hat=np.asarray(hard_u, dtype=np.uint8),
            reference=np.asarray(reference, dtype=np.uint8),
            target_syndrome=np.asarray(syndrome_bits, dtype=np.uint8),
            target_logical=np.asarray(target_bits, dtype=np.uint8),
            exact_target_ok=bool(exact_target_ok),
            converged=bool(converged),
            iterations=int(iterations_used),
            stable_rounds=int(stable_rounds),
            objective_cost=float(objective_cost),
            edge_update_proxy=int(iterations_used * self.graph.edge_count),
            max_message_delta=float(max_message_delta),
            uncertain_var_count=int(np.count_nonzero(np.abs(belief_delta) <= float(self.config.local_repair_margin))),
            belief_delta=np.asarray(belief_delta, dtype=np.float64),
            factor_degree_histogram=_factor_degree_histogram(self.graph),
            local_repair_used=bool(repair_used),
            local_repair_examined=int(repair_examined),
        )
