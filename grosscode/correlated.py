from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import scipy.sparse as sp

from bravyi_bbc_baseline import BravyiGrossConfig, default_export_dir
from grosscode.utils.gf2 import binary_csr_mod2


DEFAULT_PICKLE_CONFIG = BravyiGrossConfig(error_rate=0.003, num_cycles=12)
DEFAULT_UPSTREAM_PICKLE = default_export_dir(DEFAULT_PICKLE_CONFIG) / "upstream_decoder_setup.pkl"


@dataclass(frozen=True)
class CorrelatedFaultChoice:
    pauli: str
    probability: float
    top_syndrome_mask: int
    bottom_syndrome_mask: int
    top_logical_mask: int
    bottom_logical_mask: int


@dataclass(frozen=True)
class CorrelatedFaultGroup:
    cycle_index: int
    gate_index: int
    gate: tuple[object, ...]
    none_probability: float
    choices: tuple[CorrelatedFaultChoice, ...]


@dataclass(frozen=True)
class CorrelatedVariant:
    name: str
    check_matrix: sp.csr_matrix
    observables_top: sp.csr_matrix
    observables_bottom: sp.csr_matrix
    priors: np.ndarray
    selection_priors: np.ndarray | None
    hybrid_aux_row_count: int
    summary: Mapping[str, object]


@dataclass(frozen=True)
class CorrelatedGrossProblem:
    error_rate: float
    num_cycles: int
    total_rounds: int
    top_row_count: int
    bottom_row_count: int
    logical_qubits: int
    top_class_count: int
    bottom_class_count: int
    y_class_count: int
    invisible_fault_probability: float
    top_component_matrix: sp.csr_matrix
    bottom_component_matrix: sp.csr_matrix
    original_correlated: CorrelatedVariant
    paper_gari: CorrelatedVariant
    fault_groups: tuple[CorrelatedFaultGroup, ...]
    metadata: Mapping[str, object]


def _mask_from_bit(bit: int) -> int:
    return 1 << int(bit)


def _bit_indices(mask: int) -> Iterable[int]:
    work = int(mask)
    while work:
        lsb = int(work & -work)
        yield int(lsb.bit_length() - 1)
        work ^= lsb


def _xor_apply(mask: int, cols: Sequence[int]) -> int:
    out = 0
    for idx in _bit_indices(mask):
        out ^= int(cols[int(idx)])
    return int(out)


def _matrix_from_column_masks(n_rows: int, masks: Sequence[int]) -> sp.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []
    for col, mask in enumerate(masks):
        for row in _bit_indices(int(mask)):
            rows.append(int(row))
            cols.append(int(col))
            data.append(1)
    return binary_csr_mod2(
        sp.coo_matrix((np.asarray(data, dtype=np.uint8), (rows, cols)), shape=(int(n_rows), int(len(masks))), dtype=np.uint8)
    ).tocsr()


def _build_logical_row_masks(
    logical_matrix: sp.csr_matrix,
    *,
    data_qubit_order: Sequence[tuple[object, ...]],
    lin_order: Mapping[tuple[object, ...], int],
) -> tuple[int, ...]:
    logical_matrix = logical_matrix.tocsr()
    out: list[int] = []
    for row in range(int(logical_matrix.shape[0])):
        a = int(logical_matrix.indptr[row])
        b = int(logical_matrix.indptr[row + 1])
        mask = 0
        for data_col in logical_matrix.indices[a:b]:
            q = tuple(data_qubit_order[int(data_col)])
            mask |= _mask_from_bit(int(lin_order[q]))
        out.append(int(mask))
    return tuple(out)


def _logical_mask_from_state(state_mask: int, row_masks: Sequence[int]) -> int:
    out = 0
    for row, row_mask in enumerate(row_masks):
        if int((int(state_mask) & int(row_mask)).bit_count()) & 1:
            out |= _mask_from_bit(int(row))
    return int(out)


def _mask_to_bits(mask: int, n_bits: int) -> np.ndarray:
    out = np.zeros(int(n_bits), dtype=np.uint8)
    for idx in _bit_indices(mask):
        if int(idx) < int(n_bits):
            out[int(idx)] = 1
    return out


def _measurement_indices_for_cycle(
    cycle: Sequence[tuple[object, ...]],
    *,
    sector: str,
) -> list[int]:
    target = "MeasX" if str(sector) == "top" else "MeasZ"
    indices = [-1] * int(len(cycle))
    cursor = 0
    for idx, gate in enumerate(cycle):
        if str(gate[0]) == target:
            indices[idx] = int(cursor)
            cursor += 1
    return indices


def _build_cycle_tail_transducer(
    *,
    cycle: Sequence[tuple[object, ...]],
    lin_order: Mapping[tuple[object, ...], int],
    sector: str,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    n_qubits = int(len(lin_order))
    cycle_len = int(len(cycle))
    meas_index = _measurement_indices_for_cycle(cycle, sector=str(sector))
    tail_state: list[tuple[int, ...]] = [tuple() for _ in range(cycle_len + 1)]
    tail_syn: list[tuple[int, ...]] = [tuple() for _ in range(cycle_len + 1)]

    after_state = [_mask_from_bit(q) for q in range(n_qubits)]
    after_syn = [0 for _ in range(n_qubits)]
    tail_state[cycle_len] = tuple(int(x) for x in after_state)
    tail_syn[cycle_len] = tuple(int(x) for x in after_syn)

    for pos in range(cycle_len - 1, -1, -1):
        gate = cycle[pos]
        op = str(gate[0])
        before_state = after_state.copy()
        before_syn = after_syn.copy()

        if op == "CNOT":
            control = int(lin_order[tuple(gate[1])])
            target = int(lin_order[tuple(gate[2])])
            if str(sector) == "top":
                before_state[target] = int(after_state[target]) ^ int(after_state[control])
                before_syn[target] = int(after_syn[target]) ^ int(after_syn[control])
            else:
                before_state[control] = int(after_state[control]) ^ int(after_state[target])
                before_syn[control] = int(after_syn[control]) ^ int(after_syn[target])
        elif op == "PrepX" and str(sector) == "top":
            q = int(lin_order[tuple(gate[1])])
            before_state[q] = 0
            before_syn[q] = 0
        elif op == "PrepZ" and str(sector) == "bottom":
            q = int(lin_order[tuple(gate[1])])
            before_state[q] = 0
            before_syn[q] = 0
        elif op == "MeasX" and str(sector) == "top":
            q = int(lin_order[tuple(gate[1])])
            tag = _mask_from_bit(int(meas_index[pos]))
            before_syn[q] = int(after_syn[q]) ^ int(tag)
        elif op == "MeasZ" and str(sector) == "bottom":
            q = int(lin_order[tuple(gate[1])])
            tag = _mask_from_bit(int(meas_index[pos]))
            before_syn[q] = int(after_syn[q]) ^ int(tag)

        tail_state[pos] = tuple(int(x) for x in before_state)
        tail_syn[pos] = tuple(int(x) for x in before_syn)
        after_state = before_state
        after_syn = before_syn

    return tuple(tail_state), tuple(tail_syn)


def _build_full_cycle_transducer(
    cycle_state_cols: Sequence[int],
    cycle_raw_cols: Sequence[int],
    *,
    total_rounds: int,
    checks_per_round: int,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    n_qubits = int(len(cycle_state_cols))
    future_state: list[tuple[int, ...]] = [tuple() for _ in range(int(total_rounds) + 1)]
    future_raw: list[tuple[int, ...]] = [tuple() for _ in range(int(total_rounds) + 1)]

    id_cols = tuple(_mask_from_bit(q) for q in range(n_qubits))
    zeros = tuple(0 for _ in range(n_qubits))
    future_state[0] = id_cols
    future_raw[0] = zeros

    shift = int(checks_per_round)
    for rounds in range(1, int(total_rounds) + 1):
        prev_state = future_state[rounds - 1]
        prev_raw = future_raw[rounds - 1]
        state_cols: list[int] = []
        raw_cols: list[int] = []
        for q in range(n_qubits):
            after_cycle = int(cycle_state_cols[q])
            state_cols.append(int(_xor_apply(after_cycle, prev_state)))
            later_raw = int(_xor_apply(after_cycle, prev_raw))
            raw_cols.append(int(cycle_raw_cols[q]) ^ int(later_raw << shift))
        future_state[rounds] = tuple(state_cols)
        future_raw[rounds] = tuple(raw_cols)
    return tuple(future_state), tuple(future_raw)


def _sparsify_round_blocks(raw_mask: int, *, checks_per_round: int, total_rounds: int) -> int:
    out = 0
    block_mask = (1 << int(checks_per_round)) - 1
    prev = 0
    for round_index in range(int(total_rounds)):
        block = int(raw_mask >> (int(round_index) * int(checks_per_round))) & int(block_mask)
        diff = int(block) if int(round_index) == 0 else int(block) ^ int(prev)
        out |= int(diff) << (int(round_index) * int(checks_per_round))
        prev = int(block)
    return int(out)


def _fault_pauli_masks(
    pauli: str,
    qubits: Sequence[tuple[object, ...]],
    *,
    lin_order: Mapping[tuple[object, ...], int],
) -> tuple[int, int]:
    pauli = str(pauli)
    x_mask = 0
    z_mask = 0
    if len(pauli) != int(len(qubits)):
        raise ValueError(f"pauli {pauli!r} and qubits length mismatch")
    for term, q in zip(pauli, qubits, strict=True):
        bit = _mask_from_bit(int(lin_order[tuple(q)]))
        if str(term) in ("X", "Y"):
            x_mask ^= int(bit)
        if str(term) in ("Z", "Y"):
            z_mask ^= int(bit)
    return int(x_mask), int(z_mask)


def _gate_fault_specs(gate: tuple[object, ...], *, error_rate: float) -> tuple[tuple[str, str, tuple[tuple[object, ...], ...], float], ...]:
    op = str(gate[0])
    q1 = tuple(gate[1:2])
    if op == "MeasX":
        return (("before", "Z", q1, float(error_rate)),)
    if op == "MeasZ":
        return (("before", "X", q1, float(error_rate)),)
    if op == "PrepX":
        return (("after", "Z", q1, float(error_rate)),)
    if op == "PrepZ":
        return (("after", "X", q1, float(error_rate)),)
    if op == "IDLE":
        return (
            ("after", "X", q1, float(error_rate) / 3.0),
            ("after", "Y", q1, float(error_rate) / 3.0),
            ("after", "Z", q1, float(error_rate) / 3.0),
        )
    if op == "CNOT":
        q2 = tuple(gate[2:3])
        qubits = q1 + q2
        paulis = ("X", "Y", "Z")
        out: list[tuple[str, str, tuple[tuple[object, ...], ...], float]] = []
        for term in paulis:
            out.append(("after", term, q1, float(error_rate) / 15.0))
        for term in paulis:
            out.append(("after", term, q2, float(error_rate) / 15.0))
        for left in paulis:
            for right in paulis:
                out.append(("after", left + right, qubits, float(error_rate) / 15.0))
        return tuple(out)
    raise ValueError(f"unsupported gate type: {op}")


def _build_fault_groups_and_component_maps(
    *,
    cycle: Sequence[tuple[object, ...]],
    num_cycles: int,
    error_rate: float,
    lin_order: Mapping[tuple[object, ...], int],
    top_tail_state: Sequence[Sequence[int]],
    top_tail_raw: Sequence[Sequence[int]],
    bottom_tail_state: Sequence[Sequence[int]],
    bottom_tail_raw: Sequence[Sequence[int]],
    future_top_state: Sequence[Sequence[int]],
    future_top_raw: Sequence[Sequence[int]],
    future_bottom_state: Sequence[Sequence[int]],
    future_bottom_raw: Sequence[Sequence[int]],
    top_logical_masks: Sequence[int],
    bottom_logical_masks: Sequence[int],
    checks_per_round: int,
    total_rounds: int,
    progress_every: int,
) -> tuple[
    tuple[CorrelatedFaultGroup, ...],
    dict[tuple[int, int], float],
    dict[tuple[int, int], float],
    dict[tuple[tuple[int, int], tuple[int, int]], float],
    float,
]:
    pure_top: dict[tuple[int, int], float] = {}
    pure_bottom: dict[tuple[int, int], float] = {}
    correlated: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
    invisible_probability = 0.0
    groups: list[CorrelatedFaultGroup] = []
    total_groups = int(len(cycle) * int(num_cycles))
    cycle_len = int(len(cycle))
    checks_shift_mask = (1 << int(checks_per_round)) - 1
    if int(checks_shift_mask) <= 0:
        raise ValueError("checks_per_round must be positive")

    for cycle_index in range(int(num_cycles)):
        after_rounds = int(total_rounds - (cycle_index + 1))
        for gate_index, gate in enumerate(cycle):
            choice_rows: list[CorrelatedFaultChoice] = []
            total_choice_probability = 0.0
            for insertion, pauli, qubits, prob in _gate_fault_specs(tuple(gate), error_rate=float(error_rate)):
                x_mask, z_mask = _fault_pauli_masks(str(pauli), qubits, lin_order=lin_order)
                boundary = int(gate_index) if str(insertion) == "before" else int(gate_index) + 1

                top_tail_state_mask = _xor_apply(int(z_mask), top_tail_state[int(boundary)])
                top_tail_raw_mask = _xor_apply(int(z_mask), top_tail_raw[int(boundary)])
                top_future_state_mask = _xor_apply(int(top_tail_state_mask), future_top_state[int(after_rounds)])
                top_future_raw_mask = _xor_apply(int(top_tail_state_mask), future_top_raw[int(after_rounds)])
                top_raw_global = (int(top_tail_raw_mask) << (int(cycle_index) * int(checks_per_round))) ^ (
                    int(top_future_raw_mask) << ((int(cycle_index) + 1) * int(checks_per_round))
                )

                bottom_tail_state_mask = _xor_apply(int(x_mask), bottom_tail_state[int(boundary)])
                bottom_tail_raw_mask = _xor_apply(int(x_mask), bottom_tail_raw[int(boundary)])
                bottom_future_state_mask = _xor_apply(int(bottom_tail_state_mask), future_bottom_state[int(after_rounds)])
                bottom_future_raw_mask = _xor_apply(int(bottom_tail_state_mask), future_bottom_raw[int(after_rounds)])
                bottom_raw_global = (int(bottom_tail_raw_mask) << (int(cycle_index) * int(checks_per_round))) ^ (
                    int(bottom_future_raw_mask) << ((int(cycle_index) + 1) * int(checks_per_round))
                )

                top_syndrome_mask = _sparsify_round_blocks(
                    int(top_raw_global),
                    checks_per_round=int(checks_per_round),
                    total_rounds=int(total_rounds),
                )
                bottom_syndrome_mask = _sparsify_round_blocks(
                    int(bottom_raw_global),
                    checks_per_round=int(checks_per_round),
                    total_rounds=int(total_rounds),
                )
                top_logical_mask = _logical_mask_from_state(int(top_future_state_mask), top_logical_masks)
                bottom_logical_mask = _logical_mask_from_state(int(bottom_future_state_mask), bottom_logical_masks)

                choice_rows.append(
                    CorrelatedFaultChoice(
                        pauli=str(pauli),
                        probability=float(prob),
                        top_syndrome_mask=int(top_syndrome_mask),
                        bottom_syndrome_mask=int(bottom_syndrome_mask),
                        top_logical_mask=int(top_logical_mask),
                        bottom_logical_mask=int(bottom_logical_mask),
                    )
                )
                total_choice_probability += float(prob)

                top_key = (int(top_syndrome_mask), int(top_logical_mask))
                bottom_key = (int(bottom_syndrome_mask), int(bottom_logical_mask))
                top_zero = int(top_key[0]) == 0 and int(top_key[1]) == 0
                bottom_zero = int(bottom_key[0]) == 0 and int(bottom_key[1]) == 0
                if top_zero and bottom_zero:
                    invisible_probability += float(prob)
                elif not top_zero and bottom_zero:
                    pure_top[top_key] = float(pure_top.get(top_key, 0.0) + float(prob))
                elif top_zero and not bottom_zero:
                    pure_bottom[bottom_key] = float(pure_bottom.get(bottom_key, 0.0) + float(prob))
                else:
                    pair_key = (top_key, bottom_key)
                    correlated[pair_key] = float(correlated.get(pair_key, 0.0) + float(prob))

            groups.append(
                CorrelatedFaultGroup(
                    cycle_index=int(cycle_index),
                    gate_index=int(gate_index),
                    gate=tuple(gate),
                    none_probability=float(max(0.0, 1.0 - total_choice_probability)),
                    choices=tuple(choice_rows),
                )
            )
            group_count = len(groups)
            if int(progress_every) > 0 and int(group_count) % int(progress_every) == 0:
                print(
                    "[gross-correlated] "
                    f"groups={group_count}/{total_groups} pure_top={len(pure_top)} pure_bottom={len(pure_bottom)} "
                    f"correlated={len(correlated)} invisible_prob={invisible_probability:.6g}",
                    flush=True,
                )

    return tuple(groups), pure_top, pure_bottom, correlated, float(invisible_probability)


def _summarize_matrix(matrix: sp.csr_matrix) -> dict[str, object]:
    row_w = np.asarray(matrix.getnnz(axis=1), dtype=np.int64).reshape(-1)
    col_w = np.asarray(matrix.getnnz(axis=0), dtype=np.int64).reshape(-1)
    return {
        "rows": int(matrix.shape[0]),
        "cols": int(matrix.shape[1]),
        "nnz": int(matrix.nnz),
        "row_weight_min": int(row_w.min()) if row_w.size else 0,
        "row_weight_max": int(row_w.max()) if row_w.size else 0,
        "row_weight_mean": float(row_w.mean()) if row_w.size else 0.0,
        "col_weight_min": int(col_w.min()) if col_w.size else 0,
        "col_weight_max": int(col_w.max()) if col_w.size else 0,
        "col_weight_mean": float(col_w.mean()) if col_w.size else 0.0,
    }


def paper_gari_aux_marginal_priors_from_arrays(
    *,
    original_priors: np.ndarray,
    paper_gari_matrix: sp.csr_matrix,
    top_class_count: int,
    bottom_class_count: int,
    y_class_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_top = int(top_class_count)
    n_bottom = int(bottom_class_count)
    n_y = int(y_class_count)
    orig = np.asarray(original_priors, dtype=np.float64).reshape(-1)
    expected = int(n_top + n_bottom + n_y)
    if int(orig.size) != int(expected):
        raise ValueError(f"original_priors length mismatch: got {int(orig.size)}, expected {int(expected)}")

    top_marg = np.asarray(orig[:n_top], dtype=np.float64).copy()
    bottom_marg = np.asarray(orig[n_top : n_top + n_bottom], dtype=np.float64).copy()
    if int(n_y) <= 0:
        return top_marg, bottom_marg

    aux_block = paper_gari_matrix[: n_top + n_bottom, n_top + n_bottom : n_top + n_bottom + n_y].tocsc()
    for y_idx in range(int(n_y)):
        a = int(aux_block.indptr[y_idx])
        b = int(aux_block.indptr[y_idx + 1])
        rows = np.asarray(aux_block.indices[a:b], dtype=np.int32)
        if int(rows.size) != 2:
            raise ValueError(f"paper_gari Y column {y_idx} expected 2 auxiliary incidents, found {int(rows.size)}")
        top_hits = rows[rows < int(n_top)]
        bottom_hits = rows[rows >= int(n_top)]
        if int(top_hits.size) != 1 or int(bottom_hits.size) != 1:
            raise ValueError(f"paper_gari Y column {y_idx} malformed auxiliary support: rows={rows.tolist()}")
        top_i = int(top_hits[0])
        bottom_i = int(bottom_hits[0] - int(n_top))
        p = float(orig[int(n_top + n_bottom + y_idx)])
        top_marg[int(top_i)] += float(p)
        bottom_marg[int(bottom_i)] += float(p)
    return top_marg, bottom_marg


def paper_gari_selection_priors_from_arrays(
    *,
    original_priors: np.ndarray,
    paper_gari_matrix: sp.csr_matrix,
    top_class_count: int,
    bottom_class_count: int,
    y_class_count: int,
) -> np.ndarray:
    top_marg, bottom_marg = paper_gari_aux_marginal_priors_from_arrays(
        original_priors=np.asarray(original_priors, dtype=np.float64),
        paper_gari_matrix=paper_gari_matrix,
        top_class_count=int(top_class_count),
        bottom_class_count=int(bottom_class_count),
        y_class_count=int(y_class_count),
    )
    orig = np.asarray(original_priors, dtype=np.float64).reshape(-1)
    return np.concatenate([orig, top_marg, bottom_marg]).astype(np.float64, copy=False)


def load_upstream_gross_setup(
    *,
    pickle_path: str | Path = DEFAULT_UPSTREAM_PICKLE,
) -> Mapping[str, object]:
    path = Path(pickle_path)
    if not path.exists():
        raise FileNotFoundError(f"missing upstream Gross setup pickle: {path}")
    with path.open("rb") as fh:
        data = pickle.load(fh)
    return dict(data)


def build_correlated_gross_problem(
    *,
    error_rate: float = 0.004,
    pickle_path: str | Path = DEFAULT_UPSTREAM_PICKLE,
    progress_every: int = 240,
) -> CorrelatedGrossProblem:
    started = time.perf_counter()
    setup = load_upstream_gross_setup(pickle_path=pickle_path)
    cycle = [tuple(g) for g in setup["cycle"]]
    num_cycles = int(setup["num_cycles"])
    if num_cycles <= 0:
        raise ValueError("num_cycles must be positive")
    x_checks = [tuple(q) for q in setup["Xchecks"]]
    z_checks = [tuple(q) for q in setup["Zchecks"]]
    data_qubits = [tuple(q) for q in setup["data_qubits"]]
    lin_order = {tuple(k): int(v) for k, v in dict(setup["lin_order"]).items()}
    top_logical_masks = _build_logical_row_masks(setup["lx"], data_qubit_order=data_qubits, lin_order=lin_order)
    bottom_logical_masks = _build_logical_row_masks(setup["lz"], data_qubit_order=data_qubits, lin_order=lin_order)

    total_rounds = int(num_cycles + 2)
    checks_per_round = int(len(x_checks))
    if int(checks_per_round) != int(len(z_checks)):
        raise ValueError("X and Z check counts must match for the correlated Gross builder")

    print(
        "[gross-correlated] "
        f"building cycle transducers for num_cycles={num_cycles} total_rounds={total_rounds} error_rate={float(error_rate):.6g}",
        flush=True,
    )
    top_tail_state, top_tail_raw = _build_cycle_tail_transducer(
        cycle=cycle,
        lin_order=lin_order,
        sector="top",
    )
    bottom_tail_state, bottom_tail_raw = _build_cycle_tail_transducer(
        cycle=cycle,
        lin_order=lin_order,
        sector="bottom",
    )
    future_top_state, future_top_raw = _build_full_cycle_transducer(
        top_tail_state[0],
        top_tail_raw[0],
        total_rounds=int(total_rounds),
        checks_per_round=int(checks_per_round),
    )
    future_bottom_state, future_bottom_raw = _build_full_cycle_transducer(
        bottom_tail_state[0],
        bottom_tail_raw[0],
        total_rounds=int(total_rounds),
        checks_per_round=int(checks_per_round),
    )

    print("[gross-correlated] enumerating physical depolarizing fault groups", flush=True)
    groups, pure_top_prob, pure_bottom_prob, correlated_prob, invisible_probability = _build_fault_groups_and_component_maps(
        cycle=cycle,
        num_cycles=int(num_cycles),
        error_rate=float(error_rate),
        lin_order=lin_order,
        top_tail_state=top_tail_state,
        top_tail_raw=top_tail_raw,
        bottom_tail_state=bottom_tail_state,
        bottom_tail_raw=bottom_tail_raw,
        future_top_state=future_top_state,
        future_top_raw=future_top_raw,
        future_bottom_state=future_bottom_state,
        future_bottom_raw=future_bottom_raw,
        top_logical_masks=top_logical_masks,
        bottom_logical_masks=bottom_logical_masks,
        checks_per_round=int(checks_per_round),
        total_rounds=int(total_rounds),
        progress_every=int(progress_every),
    )

    top_keys = set(pure_top_prob.keys())
    bottom_keys = set(pure_bottom_prob.keys())
    for top_key, bottom_key in correlated_prob.keys():
        top_keys.add(tuple(top_key))
        bottom_keys.add(tuple(bottom_key))
    top_keys_sorted = tuple(sorted((tuple(k) for k in top_keys), key=lambda x: (int(x[0]), int(x[1]))))
    bottom_keys_sorted = tuple(sorted((tuple(k) for k in bottom_keys), key=lambda x: (int(x[0]), int(x[1]))))
    y_keys_sorted = tuple(sorted(correlated_prob.keys(), key=lambda x: (int(x[0][0]), int(x[0][1]), int(x[1][0]), int(x[1][1]))))

    top_index = {tuple(k): i for i, k in enumerate(top_keys_sorted)}
    bottom_index = {tuple(k): i for i, k in enumerate(bottom_keys_sorted)}

    top_component_cols = [int(key[0]) for key in top_keys_sorted]
    bottom_component_cols = [int(key[0]) for key in bottom_keys_sorted]
    top_component_matrix = _matrix_from_column_masks(int(checks_per_round * total_rounds), top_component_cols)
    bottom_component_matrix = _matrix_from_column_masks(int(checks_per_round * total_rounds), bottom_component_cols)

    n_top = int(len(top_keys_sorted))
    n_bottom = int(len(bottom_keys_sorted))
    n_y = int(len(y_keys_sorted))
    top_rows = int(checks_per_round * total_rounds)
    bottom_rows = int(checks_per_round * total_rounds)
    total_rows = int(top_rows + bottom_rows)

    original_masks: list[int] = []
    original_top_obs_masks: list[int] = []
    original_bottom_obs_masks: list[int] = []
    original_priors: list[float] = []

    for key in top_keys_sorted:
        original_masks.append(int(key[0]))
        original_top_obs_masks.append(int(key[1]))
        original_bottom_obs_masks.append(0)
        original_priors.append(float(pure_top_prob.get(tuple(key), 0.0)))
    for key in bottom_keys_sorted:
        original_masks.append(int(key[0]) << int(top_rows))
        original_top_obs_masks.append(0)
        original_bottom_obs_masks.append(int(key[1]))
        original_priors.append(float(pure_bottom_prob.get(tuple(key), 0.0)))
    for top_key, bottom_key in y_keys_sorted:
        original_masks.append(int(top_key[0]) ^ (int(bottom_key[0]) << int(top_rows)))
        original_top_obs_masks.append(int(top_key[1]))
        original_bottom_obs_masks.append(int(bottom_key[1]))
        original_priors.append(float(correlated_prob[(tuple(top_key), tuple(bottom_key))]))

    original_matrix = _matrix_from_column_masks(int(total_rows), original_masks)
    original_obs_top = _matrix_from_column_masks(int(len(top_logical_masks)), original_top_obs_masks)
    original_obs_bottom = _matrix_from_column_masks(int(len(bottom_logical_masks)), original_bottom_obs_masks)
    original_variant = CorrelatedVariant(
        name="original_correlated",
        check_matrix=original_matrix,
        observables_top=original_obs_top,
        observables_bottom=original_obs_bottom,
        priors=np.asarray(original_priors, dtype=np.float64),
        selection_priors=np.asarray(original_priors, dtype=np.float64),
        hybrid_aux_row_count=0,
        summary={
            **_summarize_matrix(original_matrix),
            "top_classes": int(n_top),
            "bottom_classes": int(n_bottom),
            "y_classes": int(n_y),
            "invisible_fault_probability": float(invisible_probability),
        },
    )

    gari_rows = int(n_top + n_bottom + total_rows)
    barz_offset = int(n_top + n_bottom + n_y)
    barx_offset = int(barz_offset + n_top)
    gari_masks: list[int] = []
    gari_top_obs_masks: list[int] = []
    gari_bottom_obs_masks: list[int] = []
    gari_priors = list(original_priors) + [0.5] * int(n_top + n_bottom)

    for i in range(n_top):
        gari_masks.append(_mask_from_bit(int(i)))
        gari_top_obs_masks.append(0)
        gari_bottom_obs_masks.append(0)
    for i in range(n_bottom):
        gari_masks.append(_mask_from_bit(int(n_top + i)))
        gari_top_obs_masks.append(0)
        gari_bottom_obs_masks.append(0)
    for top_key, bottom_key in y_keys_sorted:
        top_i = int(top_index[tuple(top_key)])
        bottom_i = int(bottom_index[tuple(bottom_key)])
        gari_masks.append(_mask_from_bit(int(top_i)) ^ _mask_from_bit(int(n_top + bottom_i)))
        gari_top_obs_masks.append(0)
        gari_bottom_obs_masks.append(0)
    for i, key in enumerate(top_keys_sorted):
        row_mask = _mask_from_bit(int(i)) ^ (int(key[0]) << int(n_top + n_bottom))
        gari_masks.append(int(row_mask))
        gari_top_obs_masks.append(int(key[1]))
        gari_bottom_obs_masks.append(0)
    for i, key in enumerate(bottom_keys_sorted):
        row_mask = _mask_from_bit(int(n_top + i)) ^ (int(key[0]) << int(n_top + n_bottom + top_rows))
        gari_masks.append(int(row_mask))
        gari_top_obs_masks.append(0)
        gari_bottom_obs_masks.append(int(key[1]))

    gari_matrix = _matrix_from_column_masks(int(gari_rows), gari_masks)
    gari_obs_top = _matrix_from_column_masks(int(len(top_logical_masks)), gari_top_obs_masks)
    gari_obs_bottom = _matrix_from_column_masks(int(len(bottom_logical_masks)), gari_bottom_obs_masks)
    paper_selection_priors = paper_gari_selection_priors_from_arrays(
        original_priors=np.asarray(original_priors, dtype=np.float64),
        paper_gari_matrix=gari_matrix,
        top_class_count=int(n_top),
        bottom_class_count=int(n_bottom),
        y_class_count=int(n_y),
    )
    paper_variant = CorrelatedVariant(
        name="paper_gari",
        check_matrix=gari_matrix,
        observables_top=gari_obs_top,
        observables_bottom=gari_obs_bottom,
        priors=np.asarray(gari_priors, dtype=np.float64),
        selection_priors=np.asarray(paper_selection_priors, dtype=np.float64),
        hybrid_aux_row_count=int(n_top + n_bottom),
        summary={
            **_summarize_matrix(gari_matrix),
            "top_classes": int(n_top),
            "bottom_classes": int(n_bottom),
            "y_classes": int(n_y),
            "aux_u_rows": int(n_top),
            "aux_v_rows": int(n_bottom),
            "aux_barz_cols": int(n_top),
            "aux_barx_cols": int(n_bottom),
        },
    )

    metadata = {
        "upstream_pickle_path": str(Path(pickle_path)),
        "requested_error_rate": float(error_rate),
        "source_cycle_error_rate": float(setup.get("error_rate", DEFAULT_PICKLE_CONFIG.error_rate)),
        "num_cycles": int(num_cycles),
        "total_rounds": int(total_rounds),
        "cycle_length": int(len(cycle)),
        "gate_groups": int(len(groups)),
        "checks_per_round": int(checks_per_round),
        "top_row_slice_original": [0, int(top_rows)],
        "bottom_row_slice_original": [int(top_rows), int(total_rows)],
        "aux_u_row_slice_gari": [0, int(n_top)],
        "aux_v_row_slice_gari": [int(n_top), int(n_top + n_bottom)],
        "top_row_slice_gari": [int(n_top + n_bottom), int(n_top + n_bottom + top_rows)],
        "bottom_row_slice_gari": [int(n_top + n_bottom + top_rows), int(gari_rows)],
        "barz_col_slice_gari": [int(barz_offset), int(barz_offset + n_top)],
        "barx_col_slice_gari": [int(barx_offset), int(barx_offset + n_bottom)],
        "top_class_count": int(n_top),
        "bottom_class_count": int(n_bottom),
        "y_class_count": int(n_y),
        "zero_column_note": "The upstream split-sector HdecX/HdecZ exports each contain one disconnected all-zero column. The correlated builder omits that column because it has no Tanner edges and no logical support.",
        "build_runtime_sec": float(time.perf_counter() - started),
        "paper_schedule_note": "Rows are ordered as U, V, top detector rows, bottom detector rows. With hybrid_aux_row_count=|U|+|V|, the existing hybrid schedule processes U then V as layered auxiliary blocks before randomized serial updates on the detector rows.",
    }

    return CorrelatedGrossProblem(
        error_rate=float(error_rate),
        num_cycles=int(num_cycles),
        total_rounds=int(total_rounds),
        top_row_count=int(top_rows),
        bottom_row_count=int(bottom_rows),
        logical_qubits=int(len(top_logical_masks)),
        top_class_count=int(n_top),
        bottom_class_count=int(n_bottom),
        y_class_count=int(n_y),
        invisible_fault_probability=float(invisible_probability),
        top_component_matrix=top_component_matrix,
        bottom_component_matrix=bottom_component_matrix,
        original_correlated=original_variant,
        paper_gari=paper_variant,
        fault_groups=tuple(groups),
        metadata=metadata,
    )


@lru_cache(maxsize=4)
def build_correlated_gross_problem_cached(
    *,
    error_rate: float = 0.004,
    pickle_path: str | Path = DEFAULT_UPSTREAM_PICKLE,
) -> CorrelatedGrossProblem:
    return build_correlated_gross_problem(error_rate=float(error_rate), pickle_path=str(Path(pickle_path)), progress_every=0)


def syndrome_mask_to_array(mask: int, n_rows: int) -> np.ndarray:
    return _mask_to_bits(int(mask), int(n_rows))


def logical_mask_to_array(mask: int, n_rows: int) -> np.ndarray:
    return _mask_to_bits(int(mask), int(n_rows))
