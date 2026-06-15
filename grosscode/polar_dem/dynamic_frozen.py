from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .gf2 import RowEchelonForm, dense_mod2, matvec_mod2, right_reduced_row_echelon_mod2


@dataclass(frozen=True)
class FrozenRule:
    index: int
    row_index: int
    alpha: np.ndarray


@dataclass(frozen=True)
class GapProfile:
    info_set_size: int
    reliability_order: tuple[int, ...]
    ideal_info_set: tuple[int, ...]
    free_prefix_count: np.ndarray
    ideal_prefix_count: np.ndarray
    gap: np.ndarray
    g_max: int
    predicted_list_size: int


@dataclass(frozen=True)
class DynamicFrozenSystem:
    q_matrix: np.ndarray
    echelon_form: RowEchelonForm
    rules: tuple[FrozenRule, ...]
    frozen_indices: tuple[int, ...]
    info_indices: tuple[int, ...]
    consistency_rows: tuple[int, ...]

    @property
    def right_pivot_matrix(self) -> np.ndarray:
        return self.echelon_form.matrix

    @property
    def row_transform(self) -> np.ndarray:
        return self.echelon_form.row_transform

    def transformed_syndrome(self, syndrome: np.ndarray | Sequence[int]) -> np.ndarray:
        return matvec_mod2(self.row_transform, syndrome)

    def forced_bit(self, index: int, prefix: np.ndarray | Sequence[int], transformed_syndrome: np.ndarray) -> int:
        rule = next((entry for entry in self.rules if entry.index == int(index)), None)
        if rule is None:
            raise KeyError(f"index {index} is not frozen")
        prefix_arr = dense_mod2(prefix).reshape(-1)
        if int(prefix_arr.shape[0]) != int(index):
            raise ValueError(f"prefix for frozen bit {index} must have length {index}")
        rhs = int(transformed_syndrome[rule.row_index])
        lhs = int(np.bitwise_and(rule.alpha, prefix_arr).sum() & 1)
        return int(rhs ^ lhs)

    def construct_u_from_info_bits(
        self,
        info_bits: np.ndarray | Sequence[int],
        syndrome: np.ndarray | Sequence[int],
    ) -> np.ndarray:
        info = dense_mod2(info_bits).reshape(-1)
        if int(info.shape[0]) != len(self.info_indices):
            raise ValueError("info_bits length does not match number of free positions")
        rhs = self.transformed_syndrome(syndrome)
        u = np.zeros(self.q_matrix.shape[1], dtype=np.uint8)
        cursor = 0
        frozen = set(self.frozen_indices)
        for index in range(self.q_matrix.shape[1]):
            if index in frozen:
                u[index] = self.forced_bit(index, u[:index], rhs)
            else:
                u[index] = info[cursor]
                cursor += 1
        for row in self.consistency_rows:
            if int(rhs[row]) != 0:
                raise ValueError("syndrome is inconsistent with the dynamic frozen constraints")
        return u

    def satisfies(self, u: np.ndarray | Sequence[int], syndrome: np.ndarray | Sequence[int]) -> bool:
        vec = dense_mod2(u).reshape(-1)
        rhs = self.transformed_syndrome(syndrome)
        lhs = matvec_mod2(self.right_pivot_matrix, vec)
        return bool(np.array_equal(lhs, rhs))


def derive_dynamic_frozen_system(q_matrix: np.ndarray | Sequence[Sequence[int]]) -> DynamicFrozenSystem:
    q = dense_mod2(q_matrix)
    if q.ndim != 2:
        raise ValueError("q_matrix must be rank 2")
    echelon = right_reduced_row_echelon_mod2(q)
    rank = len(echelon.pivot_columns)
    rules: list[FrozenRule] = []
    for row_index, pivot in enumerate(echelon.pivot_columns):
        row = echelon.matrix[row_index, :pivot].copy().astype(np.uint8, copy=False)
        rules.append(FrozenRule(index=int(pivot), row_index=int(row_index), alpha=row))
    frozen = tuple(int(pivot) for pivot in echelon.pivot_columns)
    frozen_set = set(frozen)
    info = tuple(index for index in range(q.shape[1]) if index not in frozen_set)
    consistency = tuple(
        row for row in range(rank, q.shape[0]) if int(np.any(echelon.matrix[row]))
    )
    return DynamicFrozenSystem(
        q_matrix=q,
        echelon_form=echelon,
        rules=tuple(rules),
        frozen_indices=frozen,
        info_indices=info,
        consistency_rows=tuple(range(rank, q.shape[0])),
    )


def compute_gap_profile(
    *,
    length: int,
    frozen_indices: Sequence[int],
    reliability_scores: Sequence[float],
) -> GapProfile:
    if len(reliability_scores) != int(length):
        raise ValueError("reliability_scores length must equal block length")
    frozen_set = {int(index) for index in frozen_indices}
    free_count = int(length) - len(frozen_set)
    score_array = np.asarray(reliability_scores, dtype=np.float64).reshape(-1)
    reliability_order = tuple(
        int(index)
        for index in np.lexsort((np.arange(length, dtype=np.int64), -score_array.astype(np.float64)))
    )
    ideal_set = tuple(sorted(reliability_order[:free_count]))
    ideal_lookup = set(ideal_set)
    free_prefix = np.zeros(length, dtype=np.int64)
    ideal_prefix = np.zeros(length, dtype=np.int64)
    free_seen = 0
    ideal_seen = 0
    for index in range(length):
        if index not in frozen_set:
            free_seen += 1
        if index in ideal_lookup:
            ideal_seen += 1
        free_prefix[index] = free_seen
        ideal_prefix[index] = ideal_seen
    gap = (free_prefix - ideal_prefix).astype(np.int64, copy=False)
    g_max = int(gap.max(initial=0))
    return GapProfile(
        info_set_size=free_count,
        reliability_order=reliability_order,
        ideal_info_set=ideal_set,
        free_prefix_count=free_prefix,
        ideal_prefix_count=ideal_prefix,
        gap=gap,
        g_max=g_max,
        predicted_list_size=1 << g_max,
    )
