from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import scipy.sparse as sp

from grosscode.circuits.tanner_redundant_syndrome import (
    DetectorMode,
    build_redundant_check_matrix,
    load_tanner144_css_matrices,
)
from grosscode.codes.css import derive_css_code_from_checks

try:
    import stim
except Exception:  # pragma: no cover - exercised only on environments without stim
    stim = None


CycleOrder = Literal["XZ", "ZX", "XZXZ", "ZXZX"]
InitialNoiseMode = Literal["depolarizing", "x_only", "z_only", "xz"]
MemorySide = Literal["X", "Z"]


@dataclass(frozen=True, slots=True)
class CircuitBundle:
    circuit: object
    H_ext_X: sp.csr_matrix
    H_ext_Z: sp.csr_matrix
    N_X: sp.csr_matrix
    N_Z: sp.csr_matrix
    schedule_metadata: dict[str, object]
    check_metadata: dict[str, object]
    noise_metadata: dict[str, object]
    detector_metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class MemoryCircuitBundle:
    circuit: object
    side: str
    error_pauli: str
    check_pauli: str
    H_ext: sp.csr_matrix
    relation_matrix: sp.csr_matrix
    logical_matrix: sp.csr_matrix
    schedule_metadata: dict[str, object]
    check_metadata: dict[str, object]
    noise_metadata: dict[str, object]
    detector_metadata: dict[str, object]


def greedy_edge_coloring_schedule(H: sp.spmatrix) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Return CNOT ticks as `(check_row, data_qubit)` edges with no conflicts per tick."""

    csr = sp.csr_matrix(H, dtype=np.uint8)
    edges: list[tuple[int, int]] = []
    for row in range(int(csr.shape[0])):
        start = int(csr.indptr[row])
        stop = int(csr.indptr[row + 1])
        for col in csr.indices[start:stop]:
            edges.append((int(row), int(col)))
    ticks: list[list[tuple[int, int]]] = []
    tick_rows: list[set[int]] = []
    tick_cols: list[set[int]] = []
    for row, col in sorted(edges, key=lambda item: (int(item[0]), int(item[1]))):
        placed = False
        for tick_index in range(len(ticks)):
            if int(row) in tick_rows[tick_index] or int(col) in tick_cols[tick_index]:
                continue
            ticks[tick_index].append((int(row), int(col)))
            tick_rows[tick_index].add(int(row))
            tick_cols[tick_index].add(int(col))
            placed = True
            break
        if not placed:
            ticks.append([(int(row), int(col))])
            tick_rows.append({int(row)})
            tick_cols.append({int(col)})
    return tuple(tuple(tick) for tick in ticks)


def _append_mpp_measurement(circuit: object, *, pauli: str, qubits: Sequence[int]) -> None:
    if stim is None:
        raise RuntimeError("stim is required to build Tanner redundant extraction circuits")
    if not qubits:
        raise ValueError("cannot append an empty MPP measurement")
    targets = []
    for index, qubit in enumerate(qubits):
        if index:
            targets.append(stim.target_combiner())
        if pauli == "X":
            targets.append(stim.target_x(int(qubit)))
        elif pauli == "Z":
            targets.append(stim.target_z(int(qubit)))
        else:
            raise ValueError(f"unsupported MPP pauli {pauli!r}")
    circuit.append("MPP", targets)


def _row_supports(matrix: sp.spmatrix) -> tuple[tuple[int, ...], ...]:
    csr = sp.csr_matrix(matrix, dtype=np.uint8)
    supports: list[tuple[int, ...]] = []
    for row in range(int(csr.shape[0])):
        start = int(csr.indptr[row])
        stop = int(csr.indptr[row + 1])
        supports.append(tuple(int(value) for value in csr.indices[start:stop]))
    return tuple(supports)


def _rec_target(current_measurement_count: int, measurement_index: int) -> object:
    if stim is None:
        raise RuntimeError("stim is required to build Tanner redundant extraction circuits")
    offset = int(measurement_index) - int(current_measurement_count)
    if offset >= 0:
        raise ValueError("measurement index must be in the past")
    return stim.target_rec(int(offset))


def _append_error(circuit: object, name: str, probability: float, targets: Sequence[int]) -> None:
    if float(probability) <= 0.0 or not targets:
        return
    circuit.append(str(name), [int(value) for value in targets], float(probability))


def _append_pair_error(circuit: object, probability: float, pairs: Sequence[tuple[int, int]]) -> None:
    if float(probability) <= 0.0 or not pairs:
        return
    targets: list[int] = []
    for a, b in pairs:
        targets.extend([int(a), int(b)])
    circuit.append("DEPOLARIZE2", targets, float(probability))


def _relation_row_supports(relation_matrix: sp.spmatrix) -> tuple[tuple[int, ...], ...]:
    return _row_supports(relation_matrix)


def _append_detector(
    circuit: object,
    *,
    measurement_count: int,
    measurement_indices: Sequence[int],
    metadata_rows: list[dict[str, object]],
    detector_type: str,
    side: str,
    round_index: int,
    block_index: int,
    row_index: int,
    parent_rows: Sequence[int] = (),
    redundant_row: int | None = None,
) -> None:
    targets = [_rec_target(int(measurement_count), int(value)) for value in measurement_indices]
    circuit.append("DETECTOR", targets, [float(block_index), float(row_index)])
    metadata_rows.append(
        {
            "detector_index": int(len(metadata_rows)),
            "type": str(detector_type),
            "side": str(side),
            "round": int(round_index),
            "block_index": int(block_index),
            "row": int(row_index),
            "measurement_indices": [int(value) for value in measurement_indices],
            "parent_rows": [int(value) for value in parent_rows],
            "redundant_row": None if redundant_row is None else int(redundant_row),
        }
    )


def _apply_initial_noise(
    circuit: object,
    *,
    p_init: float,
    data_qubits: Sequence[int],
    initial_noise: InitialNoiseMode,
) -> None:
    if float(p_init) <= 0.0:
        return
    mode = str(initial_noise)
    if mode == "depolarizing":
        circuit.append("DEPOLARIZE1", [int(q) for q in data_qubits], float(p_init))
    elif mode == "x_only":
        circuit.append("X_ERROR", [int(q) for q in data_qubits], float(p_init))
    elif mode == "z_only":
        circuit.append("Z_ERROR", [int(q) for q in data_qubits], float(p_init))
    elif mode == "xz":
        circuit.append("X_ERROR", [int(q) for q in data_qubits], float(p_init))
        circuit.append("Z_ERROR", [int(q) for q in data_qubits], float(p_init))
    else:
        raise ValueError("initial_noise must be one of depolarizing, x_only, z_only, xz")


def _append_observable(
    circuit: object,
    *,
    measurement_count: int,
    measurement_indices: Sequence[int],
    observable_index: int,
) -> None:
    targets = [_rec_target(int(measurement_count), int(value)) for value in measurement_indices]
    circuit.append("OBSERVABLE_INCLUDE", targets, [int(observable_index)])


def build_tanner144_redundant_memory_extraction_circuit(
    *,
    side: MemorySide,
    rounds: int,
    extra_rows: int,
    max_redundant_row_weight: int,
    max_parent_size: int,
    detector_mode: DetectorMode,
    p_init: float,
    p: float,
    p_meas: float | None = None,
    include_idles: bool = True,
    seed: int = 0,
) -> MemoryCircuitBundle:
    """Build a CSS-side Tanner memory circuit with logical observables.

    ``side="X"`` means an X-error memory experiment: redundant Z checks are
    extracted and final Z-basis data readout defines L_Z observables.  ``side="Z"``
    is the dual Z-error memory experiment using X checks and final X readout.
    This is a detector-side circuit-level path; it is intentionally separate
    from the matrix-level data+measurement surrogate.
    """

    if stim is None:
        raise RuntimeError("stim is required to build Tanner redundant extraction circuits")
    if int(rounds) <= 0:
        raise ValueError("rounds must be >= 1")
    if str(detector_mode) not in {"raw_only", "raw_plus_relations"}:
        raise ValueError("detector_mode must be raw_only or raw_plus_relations")
    if float(p_init) < 0.0 or float(p) < 0.0:
        raise ValueError("noise probabilities must be >= 0")
    meas_p = float(p if p_meas is None else p_meas)
    if meas_p < 0.0:
        raise ValueError("p_meas must be >= 0")

    side_key = str(side).upper()
    if side_key not in {"X", "Z"}:
        raise ValueError("side must be X or Z")

    hx, hz = load_tanner144_css_matrices()
    code = derive_css_code_from_checks(name="Tanner [[144,12,11]]", hx=hx, hz=hz, metadata={})
    if side_key == "X":
        base_h = hz
        logical_matrix = sp.csr_matrix(np.asarray(code.LZ, dtype=np.uint8))
        check_pauli = "Z"
        error_pauli = "X"
        data_reset_gate = "R"
        data_measure_gate = "M"
        measurement_flip_gate = "X_ERROR"
        check_symbol = "H_Z_ext"
        logical_symbol = "L_Z"
        side_seed = int(seed) + 101
    else:
        base_h = hx
        logical_matrix = sp.csr_matrix(np.asarray(code.LX, dtype=np.uint8))
        check_pauli = "X"
        error_pauli = "Z"
        data_reset_gate = "RX"
        data_measure_gate = "MX"
        measurement_flip_gate = "Z_ERROR"
        check_symbol = "H_X_ext"
        logical_symbol = "L_X"
        side_seed = int(seed) + 211

    red = build_redundant_check_matrix(
        base_h,
        target_extra_rows=int(extra_rows),
        max_parent_size=int(max_parent_size),
        max_redundant_row_weight=int(max_redundant_row_weight),
        seed=int(side_seed),
    )
    h_ext = red.H_ext.tocsr()
    relation_matrix = red.relation_matrix.tocsr()
    supports = _row_supports(h_ext)
    relation_supports = _relation_row_supports(relation_matrix)
    logical_supports = _row_supports(logical_matrix)
    schedule = greedy_edge_coloring_schedule(h_ext)

    data_qubits = tuple(range(144))
    ancilla_offset = 144
    row_count = int(h_ext.shape[0])
    ancillas = tuple(int(ancilla_offset + row) for row in range(row_count))
    total_qubits = int(ancilla_offset + row_count)

    circuit = stim.Circuit()
    measurement_count = 0
    detector_rows: list[dict[str, object]] = []
    measurement_blocks: list[dict[str, object]] = []
    reference_measurements: list[int] = []

    circuit.append("QUBIT_COORDS", [int(q) for q in data_qubits], [0.0])
    circuit.append(data_reset_gate, list(data_qubits))
    circuit.append("TICK")

    for support in supports:
        _append_mpp_measurement(circuit, pauli=check_pauli, qubits=support)
        reference_measurements.append(int(measurement_count))
        measurement_count += 1
    circuit.append("TICK")

    if float(p_init) > 0.0:
        circuit.append(f"{error_pauli}_ERROR", list(data_qubits), float(p_init))
        circuit.append("TICK")

    previous_measurements: list[int] | None = None
    for round_index in range(int(rounds)):
        if check_pauli == "X":
            circuit.append("RX", list(ancillas))
        else:
            circuit.append("R", list(ancillas))
        _append_error(circuit, "DEPOLARIZE1", float(p), ancillas)
        circuit.append("TICK")

        for tick in schedule:
            pairs: list[tuple[int, int]] = []
            active_qubits: set[int] = set()
            cnot_targets: list[int] = []
            for check_row, data_qubit in tick:
                ancilla = int(ancilla_offset + int(check_row))
                if check_pauli == "X":
                    cnot_targets.extend([ancilla, int(data_qubit)])
                else:
                    cnot_targets.extend([int(data_qubit), ancilla])
                pairs.append((ancilla, int(data_qubit)))
                active_qubits.add(ancilla)
                active_qubits.add(int(data_qubit))
            if cnot_targets:
                circuit.append("CX", cnot_targets)
            _append_pair_error(circuit, float(p), pairs)
            if include_idles and float(p) > 0.0:
                idle_qubits = [q for q in range(total_qubits) if q not in active_qubits]
                _append_error(circuit, "DEPOLARIZE1", float(p), idle_qubits)
            circuit.append("TICK")

        _append_error(circuit, measurement_flip_gate, meas_p, ancillas)
        circuit.append("MX" if check_pauli == "X" else "M", list(ancillas))
        current_measurements = [int(measurement_count + row) for row in range(row_count)]
        measurement_count += row_count
        measurement_blocks.append(
            {
                "block_index": int(round_index),
                "side": str(side_key),
                "check_pauli": str(check_pauli),
                "round": int(round_index),
                "row_count": int(row_count),
                "measurement_indices": list(current_measurements),
                "ancilla_offset": int(ancilla_offset),
            }
        )

        if round_index == 0:
            for row, measurement_index in enumerate(current_measurements):
                _append_detector(
                    circuit,
                    measurement_count=int(measurement_count),
                    measurement_indices=(int(measurement_index), int(reference_measurements[int(row)])),
                    metadata_rows=detector_rows,
                    detector_type="raw",
                    side=str(side_key),
                    round_index=int(round_index),
                    block_index=int(round_index),
                    row_index=int(row),
                )

        if str(detector_mode) == "raw_plus_relations":
            base_rows = int(red.H_base.shape[0])
            for relation_row, support in enumerate(relation_supports):
                _append_detector(
                    circuit,
                    measurement_count=int(measurement_count),
                    measurement_indices=tuple(int(current_measurements[int(row)]) for row in support),
                    metadata_rows=detector_rows,
                    detector_type="relation",
                    side=str(side_key),
                    round_index=int(round_index),
                    block_index=int(round_index),
                    row_index=int(relation_row),
                    parent_rows=red.redundant_parent_sets[int(relation_row)]
                    if int(relation_row) < len(red.redundant_parent_sets)
                    else (),
                    redundant_row=int(base_rows + relation_row),
                )

        if previous_measurements is not None:
            for row, measurement_index in enumerate(current_measurements):
                _append_detector(
                    circuit,
                    measurement_count=int(measurement_count),
                    measurement_indices=(int(measurement_index), int(previous_measurements[int(row)])),
                    metadata_rows=detector_rows,
                    detector_type="time_diff",
                    side=str(side_key),
                    round_index=int(round_index),
                    block_index=int(round_index),
                    row_index=int(row),
                )
        previous_measurements = list(current_measurements)
        circuit.append("TICK")

    _append_error(circuit, measurement_flip_gate, meas_p, data_qubits)
    circuit.append(data_measure_gate, list(data_qubits))
    final_measurements = [int(measurement_count + q) for q in range(144)]
    measurement_count += 144

    if previous_measurements is None:
        raise AssertionError("previous measurements missing after extraction rounds")
    for row, support in enumerate(supports):
        _append_detector(
            circuit,
            measurement_count=int(measurement_count),
            measurement_indices=(int(previous_measurements[int(row)]), *(int(final_measurements[int(q)]) for q in support)),
            metadata_rows=detector_rows,
            detector_type="final_boundary",
            side=str(side_key),
            round_index=int(rounds),
            block_index=int(rounds),
            row_index=int(row),
        )
    for obs_index, support in enumerate(logical_supports):
        _append_observable(
            circuit,
            measurement_count=int(measurement_count),
            measurement_indices=tuple(int(final_measurements[int(q)]) for q in support),
            observable_index=int(obs_index),
        )

    detector_count_by_type: dict[str, int] = {}
    for row in detector_rows:
        detector_count_by_type[str(row["type"])] = detector_count_by_type.get(str(row["type"]), 0) + 1

    schedule_metadata = {
        "side": str(side_key),
        "rounds": int(rounds),
        "two_qubit_tick_depth": int(len(schedule)),
        "edge_coloring": [[(int(row), int(col)) for row, col in tick] for tick in schedule],
        "schedule_note": "CSS-side deterministic greedy bipartite edge coloring; no data qubit or ancilla is reused inside one CNOT tick.",
    }
    check_metadata = {
        "code": "Tanner [[144,12,11]]",
        "side": str(side_key),
        "error_pauli": str(error_pauli),
        "check_pauli": str(check_pauli),
        "check_symbol": str(check_symbol),
        "logical_symbol": str(logical_symbol),
        "H_ext_shape": tuple(int(value) for value in h_ext.shape),
        "N_shape": tuple(int(value) for value in relation_matrix.shape),
        "logical_shape": tuple(int(value) for value in logical_matrix.shape),
        "redundant_stats": dict(red.stats),
        "parent_sets": [list(parent_set) for parent_set in red.redundant_parent_sets],
        "benchmark_note": "Experimental Tanner circuit-level CSS-side memory path, not the Gross/BB144 public split-sector DEM benchmark.",
    }
    noise_metadata = {
        "p_init": float(p_init),
        "p": float(p),
        "p_meas": float(meas_p),
        "include_idles": bool(include_idles),
        "initial_noise": f"{error_pauli}_ERROR on data after noiseless boundary references",
        "circuit_noise_note": "Low-p extraction noise uses noisy ancilla resets, DEPOLARIZE2 after CNOT ticks, optional idle DEPOLARIZE1, and basis-appropriate pre-measurement flips.",
    }
    detector_metadata = {
        "side": str(side_key),
        "detector_mode": str(detector_mode),
        "detectors": detector_rows,
        "detector_count": int(len(detector_rows)),
        "detector_count_by_type": detector_count_by_type,
        "reference_measurements": list(reference_measurements),
        "measurement_blocks": measurement_blocks,
        "final_measurements": list(final_measurements),
        "observable_count": int(logical_matrix.shape[0]),
        "observable_supports": [list(support) for support in logical_supports],
        "boundary_note": (
            "The circuit is a CSS-side memory experiment: data are reset in the final-readout basis, "
            "noiseless reference checks define the initial boundary, and final data readout supplies "
            "both final-boundary detectors and logical observables."
        ),
    }
    return MemoryCircuitBundle(
        circuit=circuit,
        side=str(side_key),
        error_pauli=str(error_pauli),
        check_pauli=str(check_pauli),
        H_ext=h_ext,
        relation_matrix=relation_matrix,
        logical_matrix=logical_matrix,
        schedule_metadata=schedule_metadata,
        check_metadata=check_metadata,
        noise_metadata=noise_metadata,
        detector_metadata=detector_metadata,
    )


def build_tanner144_redundant_extraction_circuit(
    *,
    rounds: int,
    x_extra_rows: int,
    z_extra_rows: int,
    max_redundant_row_weight: int,
    max_parent_size: int,
    cycle_order: CycleOrder,
    detector_mode: DetectorMode,
    p_init: float,
    p: float,
    p_meas: float | None = None,
    include_idles: bool = True,
    seed: int = 0,
    initial_noise: InitialNoiseMode = "depolarizing",
) -> CircuitBundle:
    if stim is None:
        raise RuntimeError("stim is required to build Tanner redundant extraction circuits")
    if int(rounds) <= 0:
        raise ValueError("rounds must be >= 1")
    order = str(cycle_order).upper()
    if order not in {"XZ", "ZX", "XZXZ", "ZXZX"}:
        raise ValueError("cycle_order must be one of XZ, ZX, XZXZ, ZXZX")
    if str(detector_mode) not in {"raw_only", "raw_plus_relations"}:
        raise ValueError("detector_mode must be raw_only or raw_plus_relations")
    if float(p_init) < 0.0 or float(p) < 0.0:
        raise ValueError("noise probabilities must be >= 0")
    meas_p = float(p if p_meas is None else p_meas)
    if meas_p < 0.0:
        raise ValueError("p_meas must be >= 0")

    hx, hz = load_tanner144_css_matrices()
    x_red = build_redundant_check_matrix(
        hx,
        target_extra_rows=int(x_extra_rows),
        max_parent_size=int(max_parent_size),
        max_redundant_row_weight=int(max_redundant_row_weight),
        seed=int(seed) + 17,
    )
    z_red = build_redundant_check_matrix(
        hz,
        target_extra_rows=int(z_extra_rows),
        max_parent_size=int(max_parent_size),
        max_redundant_row_weight=int(max_redundant_row_weight),
        seed=int(seed) + 31,
    )
    h_by_side = {"X": x_red.H_ext.tocsr(), "Z": z_red.H_ext.tocsr()}
    n_by_side = {"X": x_red.relation_matrix.tocsr(), "Z": z_red.relation_matrix.tocsr()}
    parents_by_side = {"X": x_red.redundant_parent_sets, "Z": z_red.redundant_parent_sets}
    supports_by_side = {side: _row_supports(matrix) for side, matrix in h_by_side.items()}
    relation_supports_by_side = {side: _relation_row_supports(matrix) for side, matrix in n_by_side.items()}
    schedules = {side: greedy_edge_coloring_schedule(matrix) for side, matrix in h_by_side.items()}

    data_qubits = tuple(range(144))
    x_ancilla_offset = 144
    z_ancilla_offset = x_ancilla_offset + int(h_by_side["X"].shape[0])
    ancilla_offset_by_side = {"X": int(x_ancilla_offset), "Z": int(z_ancilla_offset)}
    total_qubits = z_ancilla_offset + int(h_by_side["Z"].shape[0])

    circuit = stim.Circuit()
    measurement_count = 0
    detector_rows: list[dict[str, object]] = []
    measurement_blocks: list[dict[str, object]] = []
    reference_measurements: dict[str, list[int]] = {"X": [], "Z": []}

    circuit.append("QUBIT_COORDS", [int(q) for q in data_qubits], [0.0])

    for side in ("X", "Z"):
        pauli = str(side)
        for support in supports_by_side[side]:
            _append_mpp_measurement(circuit, pauli=pauli, qubits=support)
            reference_measurements[side].append(int(measurement_count))
            measurement_count += 1
    circuit.append("TICK")
    _apply_initial_noise(
        circuit,
        p_init=float(p_init),
        data_qubits=data_qubits,
        initial_noise=initial_noise,
    )
    if float(p_init) > 0.0:
        circuit.append("TICK")

    previous_measurements_by_side: dict[str, list[int] | None] = {"X": None, "Z": None}
    first_extraction_seen = {"X": False, "Z": False}
    block_index = 0
    for round_index in range(int(rounds)):
        for side in order:
            matrix = h_by_side[str(side)]
            row_count = int(matrix.shape[0])
            ancilla_offset = int(ancilla_offset_by_side[str(side)])
            ancillas = tuple(int(ancilla_offset + row) for row in range(row_count))
            if side == "X":
                circuit.append("RX", list(ancillas))
            else:
                circuit.append("R", list(ancillas))
            _append_error(circuit, "DEPOLARIZE1", float(p), ancillas)
            circuit.append("TICK")

            for tick in schedules[str(side)]:
                pairs: list[tuple[int, int]] = []
                active_qubits: set[int] = set()
                cnot_targets: list[int] = []
                for check_row, data_qubit in tick:
                    ancilla = int(ancilla_offset + int(check_row))
                    if side == "X":
                        cnot_targets.extend([ancilla, int(data_qubit)])
                    else:
                        cnot_targets.extend([int(data_qubit), ancilla])
                    pairs.append((ancilla, int(data_qubit)))
                    active_qubits.add(ancilla)
                    active_qubits.add(int(data_qubit))
                if cnot_targets:
                    circuit.append("CX", cnot_targets)
                _append_pair_error(circuit, float(p), pairs)
                if include_idles and float(p) > 0.0:
                    idle_qubits = [q for q in range(total_qubits) if q not in active_qubits]
                    _append_error(circuit, "DEPOLARIZE1", float(p), idle_qubits)
                circuit.append("TICK")

            if side == "X":
                _append_error(circuit, "Z_ERROR", meas_p, ancillas)
                circuit.append("MX", list(ancillas))
            else:
                _append_error(circuit, "X_ERROR", meas_p, ancillas)
                circuit.append("M", list(ancillas))
            current_measurements = [int(measurement_count + row) for row in range(row_count)]
            measurement_count += row_count
            measurement_blocks.append(
                {
                    "block_index": int(block_index),
                    "side": str(side),
                    "round": int(round_index),
                    "row_count": int(row_count),
                    "measurement_indices": list(current_measurements),
                    "ancilla_offset": int(ancilla_offset),
                }
            )

            if not first_extraction_seen[str(side)]:
                for row, measurement_index in enumerate(current_measurements):
                    _append_detector(
                        circuit,
                        measurement_count=int(measurement_count),
                        measurement_indices=(
                            int(measurement_index),
                            int(reference_measurements[str(side)][int(row)]),
                        ),
                        metadata_rows=detector_rows,
                        detector_type="raw",
                        side=str(side),
                        round_index=int(round_index),
                        block_index=int(block_index),
                        row_index=int(row),
                    )
                first_extraction_seen[str(side)] = True

            if str(detector_mode) == "raw_plus_relations":
                base_rows = int(x_red.H_base.shape[0] if side == "X" else z_red.H_base.shape[0])
                for relation_row, support in enumerate(relation_supports_by_side[str(side)]):
                    _append_detector(
                        circuit,
                        measurement_count=int(measurement_count),
                        measurement_indices=tuple(int(current_measurements[int(row)]) for row in support),
                        metadata_rows=detector_rows,
                        detector_type="relation",
                        side=str(side),
                        round_index=int(round_index),
                        block_index=int(block_index),
                        row_index=int(relation_row),
                        parent_rows=parents_by_side[str(side)][int(relation_row)]
                        if int(relation_row) < len(parents_by_side[str(side)])
                        else (),
                        redundant_row=int(base_rows + relation_row),
                    )

            previous = previous_measurements_by_side[str(side)]
            if previous is not None:
                for row, measurement_index in enumerate(current_measurements):
                    _append_detector(
                        circuit,
                        measurement_count=int(measurement_count),
                        measurement_indices=(int(measurement_index), int(previous[int(row)])),
                        metadata_rows=detector_rows,
                        detector_type="time_diff",
                        side=str(side),
                        round_index=int(round_index),
                        block_index=int(block_index),
                        row_index=int(row),
                    )
            previous_measurements_by_side[str(side)] = list(current_measurements)
            circuit.append("TICK")
            block_index += 1

    detector_count_by_type: dict[str, int] = {}
    detector_count_by_side: dict[str, int] = {}
    for row in detector_rows:
        detector_count_by_type[str(row["type"])] = detector_count_by_type.get(str(row["type"]), 0) + 1
        detector_count_by_side[str(row["side"])] = detector_count_by_side.get(str(row["side"]), 0) + 1

    schedule_metadata = {
        "cycle_order": str(order),
        "rounds": int(rounds),
        "two_qubit_tick_depth_X": int(len(schedules["X"])),
        "two_qubit_tick_depth_Z": int(len(schedules["Z"])),
        "total_extraction_blocks": int(block_index),
        "edge_coloring": {
            side: [[(int(row), int(col)) for row, col in tick] for tick in schedules[side]]
            for side in ("X", "Z")
        },
        "schedule_note": "Deterministic greedy bipartite edge coloring; no data qubit or ancilla is reused inside one CNOT tick.",
    }
    check_metadata = {
        "code": "Tanner [[144,12,11]]",
        "benchmark_note": "Experimental redundant extraction path, not the Gross/BB144 public split-sector DEM benchmark.",
        "H_ext_X_shape": tuple(int(value) for value in x_red.H_ext.shape),
        "H_ext_Z_shape": tuple(int(value) for value in z_red.H_ext.shape),
        "N_X_shape": tuple(int(value) for value in x_red.relation_matrix.shape),
        "N_Z_shape": tuple(int(value) for value in z_red.relation_matrix.shape),
        "x_redundant_stats": dict(x_red.stats),
        "z_redundant_stats": dict(z_red.stats),
        "x_parent_sets": [list(parent_set) for parent_set in x_red.redundant_parent_sets],
        "z_parent_sets": [list(parent_set) for parent_set in z_red.redundant_parent_sets],
    }
    noise_metadata = {
        "p_init": float(p_init),
        "p": float(p),
        "p_meas": float(meas_p),
        "include_idles": bool(include_idles),
        "initial_noise": str(initial_noise),
        "initial_noise_note": "Initial data noise is inserted after noiseless reference stabilizer MPPs and before the first extraction block.",
        "circuit_noise_note": "Low-p noise uses DEPOLARIZE1 after resets/idles, DEPOLARIZE2 after CNOT ticks, and basis-appropriate pre-measurement flips.",
    }
    detector_metadata = {
        "detector_mode": str(detector_mode),
        "boundary_reference_note": (
            "Noiseless reference MPPs encode the intended codespace boundary so first-shot raw detectors are "
            "Stim-deterministic without claiming the circuit prepares the Tanner codespace."
        ),
        "detectors": detector_rows,
        "detector_count": int(len(detector_rows)),
        "detector_count_by_type": detector_count_by_type,
        "detector_count_by_side": detector_count_by_side,
        "reference_measurements": {side: list(values) for side, values in reference_measurements.items()},
        "measurement_blocks": measurement_blocks,
    }
    return CircuitBundle(
        circuit=circuit,
        H_ext_X=x_red.H_ext.tocsr(),
        H_ext_Z=z_red.H_ext.tocsr(),
        N_X=x_red.relation_matrix.tocsr(),
        N_Z=z_red.relation_matrix.tocsr(),
        schedule_metadata=schedule_metadata,
        check_metadata=check_metadata,
        noise_metadata=noise_metadata,
        detector_metadata=detector_metadata,
    )


def _detector_pattern_from_measurement_indices(
    bundle: CircuitBundle,
    flipped_measurement_indices: set[int],
) -> np.ndarray:
    out = np.zeros(int(bundle.detector_metadata["detector_count"]), dtype=np.uint8)
    for detector in bundle.detector_metadata["detectors"]:
        detector_index = int(detector["detector_index"])
        parity = 0
        for measurement_index in detector["measurement_indices"]:
            parity ^= int(int(measurement_index) in flipped_measurement_indices)
        out[detector_index] = parity
    return out


def detector_pattern_for_measurement_flip(
    bundle: CircuitBundle,
    *,
    side: str,
    block_index: int,
    check_row: int,
) -> np.ndarray:
    side_key = str(side).upper()
    for block in bundle.detector_metadata["measurement_blocks"]:
        if str(block["side"]) == side_key and int(block["block_index"]) == int(block_index):
            measurement_index = int(block["measurement_indices"][int(check_row)])
            return _detector_pattern_from_measurement_indices(bundle, {int(measurement_index)})
    raise ValueError(f"measurement block not found for side={side_key} block_index={block_index}")


def detector_pattern_for_initial_data_error(
    bundle: CircuitBundle,
    *,
    pauli: Literal["X", "Z"],
    qubit: int,
) -> np.ndarray:
    pauli_key = str(pauli).upper()
    if pauli_key == "X":
        side = "Z"
        matrix = bundle.H_ext_Z.tocsr()
    elif pauli_key == "Z":
        side = "X"
        matrix = bundle.H_ext_X.tocsr()
    else:
        raise ValueError("pauli must be X or Z")
    syndrome = np.asarray(matrix[:, int(qubit)].todense(), dtype=np.uint8).reshape(-1) & 1
    out = np.zeros(int(bundle.detector_metadata["detector_count"]), dtype=np.uint8)
    for detector in bundle.detector_metadata["detectors"]:
        detector_index = int(detector["detector_index"])
        if str(detector["side"]) != side:
            continue
        detector_type = str(detector["type"])
        if detector_type == "raw":
            out[detector_index] = int(syndrome[int(detector["row"])])
        elif detector_type == "relation":
            relation = bundle.N_Z if side == "Z" else bundle.N_X
            row = int(detector["row"])
            value = int((relation.getrow(row) @ syndrome.reshape(-1, 1))[0, 0]) & 1
            out[detector_index] = value
        elif detector_type == "time_diff":
            out[detector_index] = 0
    return out
