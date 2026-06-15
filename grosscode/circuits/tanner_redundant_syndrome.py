from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import scipy.io as spio
import scipy.sparse as sp

from grosscode.utils.gf2 import binary_csr_mod2, dense_mod2, nullspace_basis_dense, rank_dense_mod2


DEFAULT_TANNER_HX = Path("data/lrz_paper_mtx/633x633/HX_C2C2_144_12_11.mtx")
DEFAULT_TANNER_HZ = Path("data/lrz_paper_mtx/633x633/HZ_C2C2_144_12_11.mtx")
DetectorMode = Literal["raw_only", "raw_plus_relations"]


@dataclass(frozen=True, slots=True)
class RedundantCheckMatrix:
    H_base: sp.csr_matrix
    H_ext: sp.csr_matrix
    B_ext: sp.csr_matrix
    redundant_parent_sets: tuple[tuple[int, ...], ...]
    relation_matrix: sp.csr_matrix
    stats: dict[str, object]


@dataclass(frozen=True, slots=True)
class _CandidateRow:
    parent_set: tuple[int, ...]
    support: tuple[int, ...]
    parent_mask: int
    tie_break: int


def load_tanner144_css_matrices(
    *,
    hx_path: str | Path = DEFAULT_TANNER_HX,
    hz_path: str | Path = DEFAULT_TANNER_HZ,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Load Tanner [[144,12,11]] CSS check matrices as binary CSR matrices.

    The returned order is `(HX, HZ)`. These are the base-code parity-check
    matrices, not a Gross split-sector DEM benchmark.
    """

    hx_source = Path(hx_path)
    hz_source = Path(hz_path)
    if not hx_source.exists() or not hz_source.exists():
        raise FileNotFoundError(
            f"Tanner144 MTX files missing: {hx_source} and/or {hz_source}. "
            "Pass hx_path/hz_path for the public Tanner144 matrix files."
        )
    hx = binary_csr_mod2(spio.mmread(str(hx_source))).tocsr()
    hz = binary_csr_mod2(spio.mmread(str(hz_source))).tocsr()
    if hx.shape != hz.shape:
        raise ValueError(f"HX/HZ shape mismatch: {hx.shape} vs {hz.shape}")
    if hx.shape[1] != 144:
        raise ValueError(f"expected Tanner144 matrices with 144 columns, got {hx.shape}")
    return hx, hz


def _row_supports(matrix: sp.spmatrix) -> tuple[tuple[int, ...], ...]:
    csr = binary_csr_mod2(matrix).tocsr()
    supports: list[tuple[int, ...]] = []
    for row in range(int(csr.shape[0])):
        start = int(csr.indptr[row])
        stop = int(csr.indptr[row + 1])
        supports.append(tuple(int(value) for value in csr.indices[start:stop]))
    return tuple(supports)


def _stable_tie(seed: int, parent_set: Sequence[int]) -> int:
    payload = f"{int(seed)}:" + ",".join(str(int(value)) for value in parent_set)
    digest = hashlib.blake2b(payload.encode("ascii"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _candidate_parent_mask(parent_set: Sequence[int]) -> int:
    mask = 0
    for row in parent_set:
        mask |= 1 << int(row)
    return int(mask)


def _reduces_in_basis(mask: int, basis: dict[int, int]) -> int:
    value = int(mask)
    while value:
        pivot = int(value.bit_length() - 1)
        base = basis.get(pivot)
        if base is None:
            break
        value ^= int(base)
    return int(value)


def _basis_would_increase(mask: int, basis: dict[int, int]) -> bool:
    return _reduces_in_basis(int(mask), basis) != 0


def _basis_insert(mask: int, basis: dict[int, int]) -> None:
    value = _reduces_in_basis(int(mask), basis)
    if value == 0:
        return
    pivot = int(value.bit_length() - 1)
    for other_pivot, other in list(basis.items()):
        if (int(other) >> pivot) & 1:
            basis[int(other_pivot)] = int(other) ^ int(value)
    basis[pivot] = int(value)


def _enumerate_candidates(
    dense_h: np.ndarray,
    *,
    max_parent_size: int,
    max_redundant_row_weight: int,
    seed: int,
) -> list[_CandidateRow]:
    base_supports = set(_row_supports(sp.csr_matrix(dense_h, dtype=np.uint8)))
    seen = set(base_supports)
    candidates: list[_CandidateRow] = []
    m = int(dense_h.shape[0])
    for parent_size in range(2, int(max_parent_size) + 1):
        for parent_set in itertools.combinations(range(m), int(parent_size)):
            row = np.bitwise_xor.reduce(dense_h[list(parent_set), :], axis=0)
            support = tuple(int(value) for value in np.flatnonzero(row))
            if not support:
                continue
            if len(support) > int(max_redundant_row_weight):
                continue
            if support in seen:
                continue
            seen.add(support)
            candidates.append(
                _CandidateRow(
                    parent_set=tuple(int(value) for value in parent_set),
                    support=support,
                    parent_mask=_candidate_parent_mask(parent_set),
                    tie_break=_stable_tie(seed, parent_set),
                )
            )
    return candidates


def _matrix_stats(
    *,
    h_base: sp.csr_matrix,
    h_ext: sp.csr_matrix,
    relation_matrix: sp.csr_matrix,
    redundant_parent_sets: Sequence[Sequence[int]],
    max_column_degree_factor: float,
    max_column_degree_limit: int,
    selected_parent_masks: Sequence[int],
    target_extra_rows: int,
    warnings: Sequence[str],
) -> dict[str, object]:
    base_dense = dense_mod2(h_base.toarray())
    ext_dense = dense_mod2(h_ext.toarray())
    base_row_weights = np.asarray(h_base.getnnz(axis=1), dtype=np.int64)
    ext_row_weights = np.asarray(h_ext.getnnz(axis=1), dtype=np.int64)
    base_col_degrees = np.asarray(h_base.getnnz(axis=0), dtype=np.int64)
    ext_col_degrees = np.asarray(h_ext.getnnz(axis=0), dtype=np.int64)
    relation_row_weights = np.asarray(relation_matrix.getnnz(axis=1), dtype=np.int64)
    intrinsic_basis = nullspace_basis_dense(ext_dense.T)
    intrinsic_weights = np.asarray(intrinsic_basis.sum(axis=1), dtype=np.int64)
    selected_parent_rank = (
        rank_dense_mod2(
            np.asarray(
                [[(int(mask) >> row) & 1 for row in range(int(h_base.shape[0]))] for mask in selected_parent_masks],
                dtype=np.uint8,
            )
        )
        if selected_parent_masks
        else 0
    )
    return {
        "base_rows": int(h_base.shape[0]),
        "cols": int(h_base.shape[1]),
        "target_extra_rows": int(target_extra_rows),
        "selected_extra_rows": int(len(redundant_parent_sets)),
        "ext_rows": int(h_ext.shape[0]),
        "rank_base": int(rank_dense_mod2(base_dense)),
        "rank_ext": int(rank_dense_mod2(ext_dense)),
        "base_nnz": int(h_base.nnz),
        "ext_nnz": int(h_ext.nnz),
        "base_row_weight_mean": float(base_row_weights.mean()) if base_row_weights.size else 0.0,
        "base_row_weight_max": int(base_row_weights.max()) if base_row_weights.size else 0,
        "ext_row_weight_mean": float(ext_row_weights.mean()) if ext_row_weights.size else 0.0,
        "ext_row_weight_max": int(ext_row_weights.max()) if ext_row_weights.size else 0,
        "redundant_row_weight_mean": (
            float(ext_row_weights[int(h_base.shape[0]) :].mean())
            if int(h_ext.shape[0]) > int(h_base.shape[0])
            else 0.0
        ),
        "redundant_row_weight_max": (
            int(ext_row_weights[int(h_base.shape[0]) :].max())
            if int(h_ext.shape[0]) > int(h_base.shape[0])
            else 0
        ),
        "base_column_degree_mean": float(base_col_degrees.mean()) if base_col_degrees.size else 0.0,
        "base_column_degree_max": int(base_col_degrees.max()) if base_col_degrees.size else 0,
        "ext_column_degree_mean": float(ext_col_degrees.mean()) if ext_col_degrees.size else 0.0,
        "ext_column_degree_max": int(ext_col_degrees.max()) if ext_col_degrees.size else 0,
        "max_column_degree_factor": float(max_column_degree_factor),
        "max_column_degree_limit": int(max_column_degree_limit),
        "max_column_degree_limit_convention": "ceil(original max column degree * max_column_degree_factor)",
        "relation_rows": int(relation_matrix.shape[0]),
        "relation_cols": int(relation_matrix.shape[1]),
        "relation_row_weight_mean": (
            float(relation_row_weights.mean()) if relation_row_weights.size else 0.0
        ),
        "relation_row_weight_max": int(relation_row_weights.max()) if relation_row_weights.size else 0,
        "selected_parent_incidence_rank": int(selected_parent_rank),
        "intrinsic_dependency_rows": int(intrinsic_basis.shape[0]),
        "intrinsic_dependency_row_weight_mean": (
            float(intrinsic_weights.mean()) if intrinsic_weights.size else 0.0
        ),
        "intrinsic_dependency_row_weight_max": int(intrinsic_weights.max()) if intrinsic_weights.size else 0,
        "parent_set_sizes": [int(len(parent_set)) for parent_set in redundant_parent_sets],
        "warnings": list(str(value) for value in warnings),
    }


def build_redundant_check_matrix(
    H: sp.spmatrix,
    *,
    target_extra_rows: int,
    max_parent_size: int = 3,
    max_redundant_row_weight: int = 12,
    max_column_degree_factor: float = 2.5,
    seed: int = 0,
) -> RedundantCheckMatrix:
    """Build a default-off redundant Tanner-check extension.

    Candidate rows are GF(2) sums of original check rows. The column-degree
    guard uses `ceil(original max column degree * max_column_degree_factor)`.
    Parent relation rows are kept sparse and separate: each selected redundant
    measurement bit XOR its parent measurement bits equals zero.
    """

    if int(target_extra_rows) < 0:
        raise ValueError("target_extra_rows must be >= 0")
    if int(max_parent_size) < 2:
        raise ValueError("max_parent_size must be >= 2")
    if int(max_redundant_row_weight) <= 0:
        raise ValueError("max_redundant_row_weight must be > 0")
    if float(max_column_degree_factor) < 1.0:
        raise ValueError("max_column_degree_factor must be >= 1")

    h_base = binary_csr_mod2(H).tocsr()
    m, n = (int(h_base.shape[0]), int(h_base.shape[1]))
    dense_h = dense_mod2(h_base.toarray())
    base_col_degrees = np.asarray(h_base.getnnz(axis=0), dtype=np.int64)
    max_column_degree_limit = int(math.ceil(float(base_col_degrees.max(initial=0)) * float(max_column_degree_factor)))
    if max_column_degree_limit <= 0:
        max_column_degree_limit = 1

    if int(target_extra_rows) == 0:
        h_ext = h_base.copy()
        b_ext = sp.eye(m, dtype=np.uint8, format="csr")
        relation = sp.csr_matrix((0, m), dtype=np.uint8)
        stats = _matrix_stats(
            h_base=h_base,
            h_ext=h_ext,
            relation_matrix=relation,
            redundant_parent_sets=(),
            max_column_degree_factor=float(max_column_degree_factor),
            max_column_degree_limit=int(max_column_degree_limit),
            selected_parent_masks=(),
            target_extra_rows=0,
            warnings=(),
        )
        return RedundantCheckMatrix(
            H_base=h_base,
            H_ext=h_ext,
            B_ext=b_ext,
            redundant_parent_sets=(),
            relation_matrix=relation,
            stats=stats,
        )

    candidates = _enumerate_candidates(
        dense_h,
        max_parent_size=int(max_parent_size),
        max_redundant_row_weight=int(max_redundant_row_weight),
        seed=int(seed),
    )

    selected: list[_CandidateRow] = []
    selected_supports: set[tuple[int, ...]] = set(_row_supports(h_base))
    col_degrees = base_col_degrees.astype(np.int64, copy=True)
    parent_relation_degrees = np.zeros(m, dtype=np.int64)
    parent_basis: dict[int, int] = {}
    used_candidate_indices: set[int] = set()

    for _ in range(int(target_extra_rows)):
        best_index: int | None = None
        best_key: tuple[float, int, int, int, int] | None = None
        for candidate_index, candidate in enumerate(candidates):
            if candidate_index in used_candidate_indices:
                continue
            if candidate.support in selected_supports:
                continue
            support = np.asarray(candidate.support, dtype=np.int64)
            if support.size and np.any(col_degrees[support] + 1 > int(max_column_degree_limit)):
                continue
            parent_reward = sum(1.0 / float(1 + int(parent_relation_degrees[int(row)])) for row in candidate.parent_set)
            column_reward = sum(1.0 / float(1 + int(col_degrees[int(col)] - base_col_degrees[int(col)])) for col in candidate.support)
            independent = _basis_would_increase(int(candidate.parent_mask), parent_basis)
            future_degrees = col_degrees[support] + 1 if support.size else np.zeros(0, dtype=np.int64)
            imbalance_penalty = float(future_degrees.max(initial=0) - future_degrees.min(initial=0))
            score = (
                4.0 * float(parent_reward)
                + 1.5 * float(column_reward)
                + (3.0 if independent else 0.0)
                - 0.35 * float(len(candidate.support))
                - 0.45 * float(len(candidate.parent_set))
                - 0.15 * float(imbalance_penalty)
            )
            key = (
                float(score),
                -int(len(candidate.parent_set)),
                -int(len(candidate.support)),
                -int(candidate.tie_break >> 32),
                -int(candidate.tie_break & 0xFFFFFFFF),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_index = int(candidate_index)
        if best_index is None:
            break
        chosen = candidates[int(best_index)]
        used_candidate_indices.add(int(best_index))
        selected.append(chosen)
        selected_supports.add(chosen.support)
        col_degrees[np.asarray(chosen.support, dtype=np.int64)] += 1
        for parent in chosen.parent_set:
            parent_relation_degrees[int(parent)] += 1
        _basis_insert(int(chosen.parent_mask), parent_basis)

    warnings: list[str] = []
    if len(selected) < int(target_extra_rows):
        warnings.append(
            f"selected {len(selected)} redundant rows, below requested target_extra_rows={int(target_extra_rows)}"
        )

    redundant_rows = []
    for candidate in selected:
        row = np.zeros(n, dtype=np.uint8)
        if candidate.support:
            row[np.asarray(candidate.support, dtype=np.int64)] = 1
        redundant_rows.append(row)
    if redundant_rows:
        h_extra = sp.csr_matrix(np.vstack(redundant_rows), dtype=np.uint8)
        h_ext = binary_csr_mod2(sp.vstack([h_base, h_extra], format="csr")).tocsr()
    else:
        h_ext = h_base.copy()

    b_rows: list[int] = []
    b_cols: list[int] = []
    for row in range(m):
        b_rows.append(int(row))
        b_cols.append(int(row))
    for extra_index, candidate in enumerate(selected):
        out_row = m + int(extra_index)
        for parent in candidate.parent_set:
            b_rows.append(int(out_row))
            b_cols.append(int(parent))
    b_ext = sp.coo_matrix(
        (np.ones(len(b_rows), dtype=np.uint8), (b_rows, b_cols)),
        shape=(m + len(selected), m),
        dtype=np.uint8,
    ).tocsr()

    r_rows: list[int] = []
    r_cols: list[int] = []
    for rel_index, candidate in enumerate(selected):
        for parent in candidate.parent_set:
            r_rows.append(int(rel_index))
            r_cols.append(int(parent))
        r_rows.append(int(rel_index))
        r_cols.append(int(m + rel_index))
    relation = sp.coo_matrix(
        (np.ones(len(r_rows), dtype=np.uint8), (r_rows, r_cols)),
        shape=(len(selected), m + len(selected)),
        dtype=np.uint8,
    ).tocsr()

    h_from_b = binary_csr_mod2(b_ext @ h_base)
    if (h_from_b != h_ext).nnz:
        raise AssertionError("internal error: H_ext != B_ext @ H_base over GF(2)")

    relation_product = binary_csr_mod2(relation @ h_ext)
    if relation_product.nnz:
        raise AssertionError("internal error: relation_matrix @ H_ext is nonzero over GF(2)")

    parent_sets = tuple(tuple(int(value) for value in candidate.parent_set) for candidate in selected)
    stats = _matrix_stats(
        h_base=h_base,
        h_ext=h_ext,
        relation_matrix=relation,
        redundant_parent_sets=parent_sets,
        max_column_degree_factor=float(max_column_degree_factor),
        max_column_degree_limit=int(max_column_degree_limit),
        selected_parent_masks=tuple(int(candidate.parent_mask) for candidate in selected),
        target_extra_rows=int(target_extra_rows),
        warnings=tuple(warnings),
    )
    return RedundantCheckMatrix(
        H_base=h_base,
        H_ext=h_ext,
        B_ext=b_ext,
        redundant_parent_sets=parent_sets,
        relation_matrix=relation,
        stats=stats,
    )


def build_one_shot_detector_matrices(
    H_ext: sp.spmatrix,
    relation_matrix: sp.spmatrix,
    *,
    detector_mode: DetectorMode = "raw_plus_relations",
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Build one-shot data and measurement detector-response matrices.

    `D_init` columns are high-`p_init` data-error columns. `D_meas` columns are
    low-`p` measurement-error columns. For `raw_plus_relations`, detector rows
    are stacked as raw check detectors followed by sparse relation detectors.
    """

    mode = str(detector_mode)
    if mode not in {"raw_only", "raw_plus_relations"}:
        raise ValueError("detector_mode must be 'raw_only' or 'raw_plus_relations'")
    h = binary_csr_mod2(H_ext).tocsr()
    relation = binary_csr_mod2(relation_matrix).tocsr()
    if int(relation.shape[1]) != int(h.shape[0]):
        raise ValueError(f"relation_matrix columns {relation.shape[1]} do not match H_ext rows {h.shape[0]}")
    raw_count = int(h.shape[0])
    if mode == "raw_only":
        return h.copy(), sp.eye(raw_count, dtype=np.uint8, format="csr")
    zero_data = sp.csr_matrix((int(relation.shape[0]), int(h.shape[1])), dtype=np.uint8)
    d_init = sp.vstack([h, zero_data], format="csr", dtype=np.uint8)
    d_meas = sp.vstack([sp.eye(raw_count, dtype=np.uint8, format="csr"), relation], format="csr", dtype=np.uint8)
    return binary_csr_mod2(d_init).tocsr(), binary_csr_mod2(d_meas).tocsr()


def build_few_shot_detector_matrices(
    H_ext: sp.spmatrix,
    relation_matrix: sp.spmatrix,
    *,
    rounds: int,
    detector_mode: DetectorMode = "raw_plus_relations",
) -> tuple[sp.csr_matrix, sp.csr_matrix, dict[str, int]]:
    """Build matrix-level one/few-shot detector responses.

    Detector rows are first-shot raw syndrome rows, optional per-round relation
    rows, and same-check time-difference rows for rounds after the first. Data
    errors only support the first raw block; persistent data syndromes cancel in
    relation and time-difference detector blocks.
    """

    if int(rounds) <= 0:
        raise ValueError("rounds must be >= 1")
    mode = str(detector_mode)
    if mode not in {"raw_only", "raw_plus_relations"}:
        raise ValueError("detector_mode must be 'raw_only' or 'raw_plus_relations'")
    h = binary_csr_mod2(H_ext).tocsr()
    relation = binary_csr_mod2(relation_matrix).tocsr()
    if int(relation.shape[1]) != int(h.shape[0]):
        raise ValueError(f"relation_matrix columns {relation.shape[1]} do not match H_ext rows {h.shape[0]}")
    raw_rows = int(h.shape[0])
    data_cols = int(h.shape[1])
    meas_cols = raw_rows * int(rounds)
    d_init_blocks: list[sp.csr_matrix] = [h]
    meas_blocks: list[sp.csr_matrix] = []
    first_meas = sp.lil_matrix((raw_rows, meas_cols), dtype=np.uint8)
    first_meas[:, 0:raw_rows] = sp.eye(raw_rows, dtype=np.uint8, format="csr")
    meas_blocks.append(first_meas.tocsr())

    relation_rows = 0
    if mode == "raw_plus_relations":
        for round_index in range(int(rounds)):
            rel_block = sp.lil_matrix((int(relation.shape[0]), meas_cols), dtype=np.uint8)
            start = int(round_index) * raw_rows
            rel_block[:, start : start + raw_rows] = relation
            meas_blocks.append(rel_block.tocsr())
            d_init_blocks.append(sp.csr_matrix((int(relation.shape[0]), data_cols), dtype=np.uint8))
            relation_rows += int(relation.shape[0])

    time_diff_rows = 0
    for round_index in range(1, int(rounds)):
        td_block = sp.lil_matrix((raw_rows, meas_cols), dtype=np.uint8)
        prev_start = int(round_index - 1) * raw_rows
        cur_start = int(round_index) * raw_rows
        eye = sp.eye(raw_rows, dtype=np.uint8, format="csr")
        td_block[:, prev_start : prev_start + raw_rows] = eye
        td_block[:, cur_start : cur_start + raw_rows] = eye
        meas_blocks.append(td_block.tocsr())
        d_init_blocks.append(sp.csr_matrix((raw_rows, data_cols), dtype=np.uint8))
        time_diff_rows += raw_rows

    d_init = binary_csr_mod2(sp.vstack(d_init_blocks, format="csr")).tocsr()
    d_meas = binary_csr_mod2(sp.vstack(meas_blocks, format="csr")).tocsr()
    metadata = {
        "raw_detector_rows": raw_rows,
        "relation_detector_rows": int(relation_rows),
        "time_diff_detector_rows": int(time_diff_rows),
        "detector_rows": int(d_init.shape[0]),
        "data_columns": int(data_cols),
        "measurement_columns": int(meas_cols),
    }
    return d_init, d_meas, metadata
