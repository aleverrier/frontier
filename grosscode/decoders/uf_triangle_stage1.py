from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from grosscode.decoders.structure_aware import (
    build_reduced_local_model,
    build_sector_structure_model,
    solve_reduced_local_model,
)
from grosscode.dem.builder import SplitSectorMetadata
from grosscode.utils.gf2 import binary_csr_mod2, csr_matvec_mod2


@dataclass(frozen=True)
class UFTriangleStage1Config:
    max_growth_rounds: int = 3
    local_max_iter: int = 24
    max_total_candidate_vars: int = 768
    max_component_vars: int = 192
    max_component_checks: int = 192


@dataclass(frozen=True)
class UFTriangleStage1DecodeResult:
    e_hat: np.ndarray
    post_llr: np.ndarray
    mean_llr: np.ndarray
    residual: np.ndarray
    converged: bool
    growth_rounds_used: int
    decode_iters: int
    component_count: int
    solved_component_count: int
    covered_residual_check_count: int
    candidate_var_count: int
    max_component_vars_seen: int
    max_component_checks_seen: int
    reduced_triangle_count: int
    residual_triangle_count: int


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(int(size)))
        self.rank = [0] * int(size)

    def find(self, node: int) -> int:
        cur = int(node)
        while self.parent[cur] != cur:
            self.parent[cur] = self.parent[self.parent[cur]]
            cur = self.parent[cur]
        return cur

    def union(self, a: int, b: int) -> None:
        root_a = self.find(int(a))
        root_b = self.find(int(b))
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def _build_var_to_checks(h_csc: sp.csc_matrix) -> tuple[np.ndarray, ...]:
    out: list[np.ndarray] = []
    for col in range(int(h_csc.shape[1])):
        start = int(h_csc.indptr[col])
        stop = int(h_csc.indptr[col + 1])
        out.append(np.asarray(h_csc.indices[start:stop], dtype=np.int32))
    return tuple(out)


def _variables_near_rows(
    *,
    h_csr: sp.csr_matrix,
    h_csc: sp.csc_matrix,
    seed_rows: np.ndarray,
    radius: int,
) -> np.ndarray:
    rows = np.asarray(seed_rows, dtype=np.int32)
    if rows.size == 0 or int(radius) <= 0:
        return np.zeros(0, dtype=np.int32)
    seen_checks = {int(v) for v in rows.tolist()}
    frontier_checks = set(seen_checks)
    seen_vars: set[int] = set()
    for _ in range(int(radius)):
        if not frontier_checks:
            break
        frontier_vars: set[int] = set()
        for check in frontier_checks:
            start = int(h_csr.indptr[int(check)])
            stop = int(h_csr.indptr[int(check) + 1])
            frontier_vars.update(int(col) for col in h_csr.indices[start:stop].tolist())
        new_vars = frontier_vars - seen_vars
        seen_vars.update(new_vars)
        next_checks: set[int] = set()
        for var in new_vars:
            start = int(h_csc.indptr[int(var)])
            stop = int(h_csc.indptr[int(var) + 1])
            next_checks.update(int(row) for row in h_csc.indices[start:stop].tolist())
        frontier_checks = next_checks - seen_checks
        seen_checks.update(frontier_checks)
    if not seen_vars:
        return np.zeros(0, dtype=np.int32)
    return np.asarray(sorted(seen_vars), dtype=np.int32)


def _extract_components_union_find(
    *,
    residual_rows: np.ndarray,
    candidate_vars: np.ndarray,
    var_to_checks: tuple[np.ndarray, ...],
) -> list[dict[str, np.ndarray]]:
    residual_arr = np.asarray(residual_rows, dtype=np.int32)
    candidate_arr = np.asarray(candidate_vars, dtype=np.int32)
    if residual_arr.size == 0 or candidate_arr.size == 0:
        return []
    residual_set = {int(row) for row in residual_arr.tolist()}
    touched_checks: set[int] = set()
    for bit in candidate_arr.tolist():
        touched_checks.update(int(row) for row in np.asarray(var_to_checks[int(bit)], dtype=np.int32).tolist())
    if not touched_checks:
        return []
    touched_rows = np.asarray(sorted(touched_checks), dtype=np.int32)
    row_slot = {int(row): int(slot) for slot, row in enumerate(touched_rows.tolist())}
    base = int(touched_rows.size)
    uf = _UnionFind(int(base + candidate_arr.size))
    for bit_slot, bit in enumerate(candidate_arr.tolist()):
        node = int(base + bit_slot)
        for row in np.asarray(var_to_checks[int(bit)], dtype=np.int32).tolist():
            slot = row_slot.get(int(row))
            if slot is None:
                continue
            uf.union(int(node), int(slot))

    groups_vars: dict[int, list[int]] = {}
    groups_rows: dict[int, list[int]] = {}
    for row_slot_idx, row in enumerate(touched_rows.tolist()):
        root = uf.find(int(row_slot_idx))
        groups_rows.setdefault(int(root), []).append(int(row))
    for bit_slot, bit in enumerate(candidate_arr.tolist()):
        root = uf.find(int(base + bit_slot))
        groups_vars.setdefault(int(root), []).append(int(bit))

    components: list[dict[str, np.ndarray]] = []
    for root in sorted(set(groups_rows) | set(groups_vars)):
        vars_global = sorted(set(int(v) for v in groups_vars.get(int(root), [])))
        checks_global = sorted(set(int(v) for v in groups_rows.get(int(root), [])))
        residual_global = [int(row) for row in checks_global if int(row) in residual_set]
        if not vars_global or not residual_global:
            continue
        components.append(
            {
                "vars": np.asarray(vars_global, dtype=np.int32),
                "all_checks": np.asarray(checks_global, dtype=np.int32),
                "residual_checks": np.asarray(residual_global, dtype=np.int32),
            }
        )
    components.sort(
        key=lambda comp: (
            int(comp["residual_checks"][0]),
            int(comp["vars"].size),
            int(comp["all_checks"].size),
        )
    )
    return components


class UFTriangleStage1Decoder:
    def __init__(
        self,
        *,
        matrix: sp.csr_matrix,
        observables: sp.csr_matrix,
        priors: np.ndarray,
        metadata: SplitSectorMetadata,
        sector: str,
        config: UFTriangleStage1Config | None = None,
    ) -> None:
        cfg = UFTriangleStage1Config() if config is None else config
        if int(cfg.max_growth_rounds) <= 0:
            raise ValueError("max_growth_rounds must be positive")
        if int(cfg.local_max_iter) <= 0:
            raise ValueError("local_max_iter must be positive")
        if int(cfg.max_total_candidate_vars) <= 0:
            raise ValueError("max_total_candidate_vars must be positive")
        if int(cfg.max_component_vars) <= 0:
            raise ValueError("max_component_vars must be positive")
        if int(cfg.max_component_checks) <= 0:
            raise ValueError("max_component_checks must be positive")
        self.config = cfg
        self.matrix = binary_csr_mod2(matrix).tocsr()
        self.observables = binary_csr_mod2(observables).tocsr()
        self.metadata = metadata
        self.sector = str(sector)
        self.h_csc = self.matrix.tocsc()
        clipped_priors = np.clip(np.asarray(priors, dtype=np.float64).reshape(-1), 1e-15, 1.0 - 1e-15)
        self.default_prior_llr = np.log((1.0 - clipped_priors) / clipped_priors)
        self.var_to_checks = _build_var_to_checks(self.h_csc)
        self.structure_model = build_sector_structure_model(
            matrix=self.matrix,
            observables=self.observables,
            metadata=self.metadata,
            sector=self.sector,
        )

    @property
    def n(self) -> int:
        return int(self.matrix.shape[1])

    def _empty_result(
        self,
        *,
        syndrome_bits: np.ndarray,
        prior_llr: np.ndarray,
    ) -> UFTriangleStage1DecodeResult:
        residual = np.asarray(syndrome_bits, dtype=np.uint8).reshape(-1).copy()
        return UFTriangleStage1DecodeResult(
            e_hat=np.zeros(self.n, dtype=np.uint8),
            post_llr=np.asarray(prior_llr, dtype=np.float64).reshape(-1).copy(),
            mean_llr=np.asarray(prior_llr, dtype=np.float64).reshape(-1).copy(),
            residual=residual,
            converged=bool(np.count_nonzero(residual) == 0),
            growth_rounds_used=0,
            decode_iters=0,
            component_count=0,
            solved_component_count=0,
            covered_residual_check_count=0,
            candidate_var_count=0,
            max_component_vars_seen=0,
            max_component_checks_seen=0,
            reduced_triangle_count=0,
            residual_triangle_count=0,
        )

    def decode(
        self,
        *,
        syndrome: np.ndarray,
        prior_llr: np.ndarray | None = None,
    ) -> UFTriangleStage1DecodeResult:
        syndrome_bits = np.asarray(syndrome, dtype=np.uint8).reshape(-1) & 1
        if int(syndrome_bits.size) != int(self.matrix.shape[0]):
            raise ValueError("syndrome size mismatch")
        prior = self.default_prior_llr if prior_llr is None else np.asarray(prior_llr, dtype=np.float64).reshape(-1)
        if int(prior.size) != int(self.n):
            raise ValueError("prior_llr size mismatch")
        residual_rows = np.flatnonzero(syndrome_bits).astype(np.int32, copy=False)
        if residual_rows.size == 0:
            return self._empty_result(syndrome_bits=syndrome_bits, prior_llr=prior)

        best = self._empty_result(syndrome_bits=syndrome_bits, prior_llr=prior)
        for growth_round in range(1, int(self.config.max_growth_rounds) + 1):
            candidate_vars = _variables_near_rows(
                h_csr=self.matrix,
                h_csc=self.h_csc,
                seed_rows=residual_rows,
                radius=int(growth_round),
            )
            if candidate_vars.size == 0:
                continue
            if int(candidate_vars.size) > int(self.config.max_total_candidate_vars):
                break
            components = _extract_components_union_find(
                residual_rows=residual_rows,
                candidate_vars=candidate_vars,
                var_to_checks=self.var_to_checks,
            )
            if not components:
                continue

            trial_e_hat = np.zeros(self.n, dtype=np.uint8)
            trial_post_llr = np.asarray(prior, dtype=np.float64).reshape(-1).copy()
            total_iters = 0
            solved_components = 0
            covered_residual_rows: set[int] = set()
            reduced_triangle_total = 0
            residual_triangle_total = 0
            max_component_vars_seen = 0
            max_component_checks_seen = 0

            for component in components:
                vars_global = np.asarray(component["vars"], dtype=np.int32)
                checks_global = np.asarray(component["all_checks"], dtype=np.int32)
                residual_global = np.asarray(component["residual_checks"], dtype=np.int32)
                max_component_vars_seen = max(int(max_component_vars_seen), int(vars_global.size))
                max_component_checks_seen = max(int(max_component_checks_seen), int(checks_global.size))
                if (
                    int(vars_global.size) > int(self.config.max_component_vars)
                    or int(checks_global.size) > int(self.config.max_component_checks)
                ):
                    continue
                local_sub_matrix = self.matrix[checks_global][:, vars_global]
                local_syndrome = syndrome_bits[checks_global]
                local_model = build_reduced_local_model(
                    local_sub_matrix=local_sub_matrix,
                    local_cols=vars_global,
                    local_prior_llr=np.asarray(prior[vars_global], dtype=np.float64),
                    selection=self.structure_model.selection,
                )
                local_result = solve_reduced_local_model(
                    model=local_model,
                    syndrome=local_syndrome,
                    max_iter=int(self.config.local_max_iter),
                )
                total_iters += int(local_result.iterations)
                trial_e_hat[vars_global] = np.asarray(local_result.bits_local, dtype=np.uint8).reshape(-1)
                finite_mask = np.isfinite(np.asarray(local_result.llr_local, dtype=np.float64))
                if bool(np.any(finite_mask)):
                    trial_post_llr[vars_global[finite_mask]] = np.asarray(local_result.llr_local, dtype=np.float64)[finite_mask]
                covered_residual_rows.update(int(row) for row in residual_global.tolist())
                reduced_triangle_total += int(local_result.reduced_triangle_count)
                residual_triangle_total += int(local_result.residual_triangle_count)
                if bool(local_result.converged) and int(np.count_nonzero(local_result.residual)) == 0:
                    solved_components += 1

            trial_residual = syndrome_bits ^ csr_matvec_mod2(self.matrix, trial_e_hat)
            trial = UFTriangleStage1DecodeResult(
                e_hat=trial_e_hat,
                post_llr=trial_post_llr,
                mean_llr=trial_post_llr.copy(),
                residual=np.asarray(trial_residual, dtype=np.uint8).reshape(-1),
                converged=bool(np.count_nonzero(trial_residual) == 0),
                growth_rounds_used=int(growth_round),
                decode_iters=int(total_iters + growth_round),
                component_count=int(len(components)),
                solved_component_count=int(solved_components),
                covered_residual_check_count=int(len(covered_residual_rows)),
                candidate_var_count=int(candidate_vars.size),
                max_component_vars_seen=int(max_component_vars_seen),
                max_component_checks_seen=int(max_component_checks_seen),
                reduced_triangle_count=int(reduced_triangle_total),
                residual_triangle_count=int(residual_triangle_total),
            )
            best_residual_weight = int(np.count_nonzero(best.residual))
            trial_residual_weight = int(np.count_nonzero(trial.residual))
            if (
                best_residual_weight < 0
                or trial_residual_weight < best_residual_weight
                or (
                    trial_residual_weight == best_residual_weight
                    and int(trial.solved_component_count) > int(best.solved_component_count)
                )
            ):
                best = trial
            if bool(trial.converged):
                return trial
        return best
