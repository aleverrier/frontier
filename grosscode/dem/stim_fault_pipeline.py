from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from typing import Callable
import re

import numpy as np
import stim  # type: ignore

from grosscode.utils.paths import resolve_cache_root


_SINGLE_QUBIT_PAULI_CHANNEL_TERMS: tuple[tuple[str, str], ...] = (
    ("X", "X"),
    ("Y", "Y"),
    ("Z", "Z"),
)

_TWO_QUBIT_DEPOLARIZING_TERMS: tuple[str, ...] = (
    "IX",
    "IY",
    "IZ",
    "XI",
    "XX",
    "XY",
    "XZ",
    "YI",
    "YX",
    "YY",
    "YZ",
    "ZI",
    "ZX",
    "ZY",
    "ZZ",
)

_PURE_NOISE_GATES = {
    "DEPOLARIZE1",
    "DEPOLARIZE2",
    "X_ERROR",
    "Y_ERROR",
    "Z_ERROR",
    "PAULI_CHANNEL_1",
}

_MEASUREMENT_RECORD_GATES = {
    "M",
    "MX",
    "MY",
    "MR",
    "MRX",
    "MRY",
}

_NOISY_MEASUREMENT_GATES = _MEASUREMENT_RECORD_GATES

_ANNOTATION_GATES = {
    "DETECTOR",
    "OBSERVABLE_INCLUDE",
    "QUBIT_COORDS",
    "SHIFT_COORDS",
}

_MEASUREMENT_FLIP_INSERTION = {
    "M": "X",
    "MX": "Z",
    "MY": "X",
    "MR": "X",
    "MRX": "Z",
    "MRY": "X",
}

_MEASUREMENT_NAME_RE = re.compile(
    r"^m(?P<measurement_index>\d+)_(?P<gate>[a-z]+)_i(?P<instruction_index>\d+)_t(?P<target_offset>\d+)_q(?P<qubit>\d+)$"
)

_GROSS_SPLIT_SECTOR_CORRECTION_MAP_CACHE: dict[tuple[str, str, float], tuple[dict[int, int], tuple[int, ...]]] = {}
_GROSS_SPLIT_SECTOR_CORRECTION_MAP_DISK_CACHE_VERSION = 1


@dataclass(frozen=True)
class FaultLocation:
    fault_id: int
    instruction_index: int
    instruction_name: str
    instruction_text: str
    target_group_index: int
    category: str
    qubits: tuple[int, ...]
    pauli_product: str
    probability: float


@dataclass(frozen=True)
class MergedColumn:
    detector_bits: tuple[int, ...]
    logical_bits: tuple[int, ...]
    probability: float
    raw_fault_ids: tuple[int, ...]


@dataclass(frozen=True)
class MergeStatistics:
    raw_fault_count: int
    visible_raw_fault_count: int
    location_count: int
    nonzero_location_signature_count: int
    location_generator_count: int
    unique_raw_detector_signature_count: int
    unique_raw_detector_logical_signature_count: int
    final_merged_column_count: int
    final_merged_detector_signature_count: int
    final_merged_detector_signature_collision_count: int


@dataclass(frozen=True)
class LocationSignatureSummary:
    instruction_index: int
    instruction_name: str
    target_group_index: int
    qubits: tuple[int, ...]
    instruction_text: str
    elementary_fault_count: int
    nonzero_signature_count: int
    generator_count: int


@dataclass(frozen=True)
class CircuitFaultAnalysis:
    faults: tuple[FaultLocation, ...]
    raw_probabilities: np.ndarray
    raw_detector_matrix: np.ndarray
    raw_logical_matrix: np.ndarray
    merged_columns: tuple[MergedColumn, ...]
    stim_columns: tuple[MergedColumn, ...]
    detector_names: tuple[str, ...]
    logical_names: tuple[str, ...]
    measurement_channel_names: tuple[str, ...] = ()
    raw_measurement_channel_matrix: np.ndarray | None = None
    detector_from_channels: np.ndarray | None = None
    logical_from_channels: np.ndarray | None = None
    raw_measurement_reference: np.ndarray | None = None
    measurement_channel_reference: np.ndarray | None = None
    merge_statistics: MergeStatistics | None = None


def _sorted_unique_qubits(qubits: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(q) for q in qubits)


def _extract_qubit(target: stim.GateTarget) -> int:
    if not target.is_qubit_target:
        raise ValueError(f"expected a qubit target, got {target!r}")
    return int(target.qubit_value)


def _append_instruction_without_noise(out: stim.Circuit, inst: stim.CircuitInstruction) -> None:
    name = str(inst.name)
    if name in _PURE_NOISE_GATES:
        return
    if name in _MEASUREMENT_RECORD_GATES:
        out.append(name, inst.targets_copy(), [])
        return
    out.append(name, inst.targets_copy(), inst.gate_args_copy())


def _append_measurement_only_instruction_without_noise(out: stim.Circuit, inst: stim.CircuitInstruction) -> None:
    name = str(inst.name)
    if name in _PURE_NOISE_GATES or name in _ANNOTATION_GATES:
        return
    if name in _MEASUREMENT_RECORD_GATES:
        out.append(name, inst.targets_copy(), [])
        return
    out.append(name, inst.targets_copy(), inst.gate_args_copy())


def _append_pauli_product(out: stim.Circuit, qubits: tuple[int, ...], pauli_product: str) -> None:
    if len(pauli_product) != len(qubits):
        raise ValueError(f"mismatched pauli product {pauli_product!r} for qubits {qubits!r}")
    grouped: dict[str, list[int]] = defaultdict(list)
    for qubit, pauli in zip(qubits, pauli_product):
        if pauli == "I":
            continue
        grouped[pauli].append(int(qubit))
    for pauli in ("X", "Y", "Z"):
        if grouped.get(pauli):
            out.append(pauli, grouped[pauli], [])


def _measurement_target_count(inst: stim.CircuitInstruction) -> int:
    name = str(inst.name)
    if name not in _MEASUREMENT_RECORD_GATES:
        return 0
    return sum(1 for target in inst.targets_copy() if target.is_qubit_target)


def _iter_pauli_channel_1_faults(probability: float) -> tuple[tuple[str, float], ...]:
    p = float(probability)
    if p < 0 or p > 1:
        raise ValueError(f"invalid depolarizing probability: {p}")
    share = p / 3.0
    return (
        ("X", share),
        ("Y", share),
        ("Z", share),
    )


def _iter_depolarize2_faults(probability: float) -> tuple[tuple[str, float], ...]:
    p = float(probability)
    if p < 0 or p > 1:
        raise ValueError(f"invalid depolarizing probability: {p}")
    share = p / 15.0
    return tuple((term, share) for term in _TWO_QUBIT_DEPOLARIZING_TERMS)


def enumerate_fault_locations(circuit: stim.Circuit) -> tuple[FaultLocation, ...]:
    flat = circuit.flattened()
    faults: list[FaultLocation] = []
    for instruction_index, inst in enumerate(flat):
        name = str(inst.name)
        args = tuple(float(x) for x in inst.gate_args_copy())
        targets = tuple(inst.targets_copy())
        if name in {"X_ERROR", "Y_ERROR", "Z_ERROR"}:
            probability = float(args[0])
            pauli = name[0]
            for target_group_index, target in enumerate(targets):
                qubit = _extract_qubit(target)
                faults.append(
                    FaultLocation(
                        fault_id=len(faults),
                        instruction_index=instruction_index,
                        instruction_name=name,
                        instruction_text=str(inst),
                        target_group_index=target_group_index,
                        category="single_pauli_error",
                        qubits=(qubit,),
                        pauli_product=pauli,
                        probability=probability,
                    )
                )
            continue
        if name == "DEPOLARIZE1":
            elementary_faults = _iter_pauli_channel_1_faults(float(args[0]))
            for target_group_index, target in enumerate(targets):
                qubit = _extract_qubit(target)
                for pauli_product, probability in elementary_faults:
                    faults.append(
                        FaultLocation(
                            fault_id=len(faults),
                            instruction_index=instruction_index,
                            instruction_name=name,
                            instruction_text=str(inst),
                            target_group_index=target_group_index,
                            category="single_qubit_depolarizing_term",
                            qubits=(qubit,),
                            pauli_product=pauli_product,
                            probability=probability,
                        )
                    )
            continue
        if name == "DEPOLARIZE2":
            if len(targets) % 2 != 0:
                raise ValueError(f"{name} requires an even number of qubit targets: {inst!r}")
            elementary_faults = _iter_depolarize2_faults(float(args[0]))
            for offset in range(0, len(targets), 2):
                qubits = (_extract_qubit(targets[offset]), _extract_qubit(targets[offset + 1]))
                target_group_index = offset // 2
                for pauli_product, probability in elementary_faults:
                    faults.append(
                        FaultLocation(
                            fault_id=len(faults),
                            instruction_index=instruction_index,
                            instruction_name=name,
                            instruction_text=str(inst),
                            target_group_index=target_group_index,
                            category="two_qubit_depolarizing_term",
                            qubits=qubits,
                            pauli_product=pauli_product,
                            probability=probability,
                        )
                    )
            continue
        if name == "PAULI_CHANNEL_1":
            if len(args) != 3:
                raise ValueError(f"expected 3 probabilities for {name}, got {args!r}")
            pauli_terms = (
                ("X", float(args[0])),
                ("Y", float(args[1])),
                ("Z", float(args[2])),
            )
            for target_group_index, target in enumerate(targets):
                qubit = _extract_qubit(target)
                for pauli_product, probability in pauli_terms:
                    if probability <= 0.0:
                        continue
                    faults.append(
                        FaultLocation(
                            fault_id=len(faults),
                            instruction_index=instruction_index,
                            instruction_name=name,
                            instruction_text=str(inst),
                            target_group_index=target_group_index,
                            category="single_qubit_pauli_channel_term",
                            qubits=(qubit,),
                            pauli_product=pauli_product,
                            probability=probability,
                        )
                    )
            continue
        if name in _NOISY_MEASUREMENT_GATES:
            args = tuple(float(x) for x in inst.gate_args_copy())
            if len(args) == 0:
                continue
            if len(args) != 1:
                raise NotImplementedError(f"unsupported noisy measurement instruction: {inst!r}")
            insertion_pauli = _MEASUREMENT_FLIP_INSERTION[name]
            probability = float(args[0])
            for target_group_index, target in enumerate(targets):
                qubit = _extract_qubit(target)
                faults.append(
                    FaultLocation(
                        fault_id=len(faults),
                        instruction_index=instruction_index,
                        instruction_name=name,
                        instruction_text=str(inst),
                        target_group_index=target_group_index,
                        category="measurement_flip",
                        qubits=(qubit,),
                        pauli_product=insertion_pauli,
                        probability=probability,
                    )
                )
            continue
        if len(args) > 0 and name not in {"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"}:
            if name not in {"RX", "RY", "R", "H", "S", "SQRT_X", "SQRT_Y", "CX", "CY", "CZ", "SWAP", "MPP", "MXX", "MYY", "MZZ", "TICK"}:
                raise NotImplementedError(f"unsupported instruction with gate args in fault enumerator: {inst!r}")
    return tuple(faults)


def build_faulted_circuit(circuit: stim.Circuit, fault: FaultLocation) -> stim.Circuit:
    flat = circuit.flattened()
    out = stim.Circuit()
    for instruction_index, inst in enumerate(flat):
        name = str(inst.name)
        if instruction_index == int(fault.instruction_index):
            if name in _PURE_NOISE_GATES:
                _append_pauli_product(out, fault.qubits, fault.pauli_product)
                continue
            if name in _NOISY_MEASUREMENT_GATES:
                _append_pauli_product(out, fault.qubits, fault.pauli_product)
                out.append(name, inst.targets_copy(), [])
                continue
        _append_instruction_without_noise(out, inst)
    return out


def circuit_measurement_names(circuit: stim.Circuit) -> tuple[str, ...]:
    flat = circuit.flattened()
    names: list[str] = []
    measurement_index = 0
    for instruction_index, inst in enumerate(flat):
        name = str(inst.name)
        if name not in _MEASUREMENT_RECORD_GATES:
            continue
        target_offset = 0
        for target in inst.targets_copy():
            if not target.is_qubit_target:
                continue
            qubit = _extract_qubit(target)
            names.append(f"m{measurement_index:04d}_{name.lower()}_i{instruction_index:05d}_t{target_offset:03d}_q{qubit:03d}")
            measurement_index += 1
            target_offset += 1
    if measurement_index != int(circuit.num_measurements):
        raise AssertionError(
            f"measurement count mismatch while naming records: {measurement_index} vs {int(circuit.num_measurements)}"
        )
    return tuple(names)


def gross_split_sector_final_data_measurement_rows(
    measurement_channel_names: tuple[str, ...] | list[str],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    parsed: list[tuple[int, int, int, int]] = []
    for row_index, name in enumerate(tuple(str(value) for value in measurement_channel_names)):
        match = _MEASUREMENT_NAME_RE.match(str(name))
        if match is None:
            raise ValueError(f"unsupported measurement channel name format: {name!r}")
        parsed.append(
            (
                int(row_index),
                int(match.group("instruction_index")),
                int(match.group("target_offset")),
                int(match.group("qubit")),
            )
        )
    if not parsed:
        raise ValueError("measurement_channel_names must be non-empty")
    final_instruction_index = max(int(item[1]) for item in parsed)
    final_rows = sorted(
        (
            (int(row_index), int(target_offset), int(qubit))
            for row_index, instruction_index, target_offset, qubit in parsed
            if int(instruction_index) == int(final_instruction_index)
        ),
        key=lambda item: (int(item[1]), int(item[0])),
    )
    if not final_rows:
        raise ValueError("could not identify final data measurement rows")
    return (
        tuple(int(row_index) for row_index, _target_offset, _qubit in final_rows),
        tuple(int(qubit) for _row_index, _target_offset, qubit in final_rows),
    )


def circuit_record_to_detector_logical_matrices(
    circuit: stim.Circuit,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    flat = circuit.flattened()
    num_measurements = int(circuit.num_measurements)
    num_detectors = int(circuit.num_detectors)
    num_logicals = int(circuit.num_observables)
    detector_from_measurements = np.zeros((num_detectors, num_measurements), dtype=np.uint8)
    logical_from_measurements = np.zeros((num_logicals, num_measurements), dtype=np.uint8)
    detector_names = tuple(f"D{index}" for index in range(num_detectors))
    logical_names = tuple(f"L{index}" for index in range(num_logicals))

    measurement_so_far = 0
    detector_index = 0
    for inst in flat:
        name = str(inst.name)
        if name in _NOISY_MEASUREMENT_GATES:
            measurement_so_far += _measurement_target_count(inst)
            continue
        if name == "DETECTOR":
            for target in inst.targets_copy():
                if not target.is_measurement_record_target:
                    continue
                absolute_index = measurement_so_far + int(target.value)
                if absolute_index < 0 or absolute_index >= num_measurements:
                    raise ValueError(
                        f"detector target {target!r} resolves outside measurement record "
                        f"(measurement_so_far={measurement_so_far}, num_measurements={num_measurements})"
                    )
                detector_from_measurements[detector_index, absolute_index] ^= 1
            detector_index += 1
            continue
        if name == "OBSERVABLE_INCLUDE":
            gate_args = tuple(float(x) for x in inst.gate_args_copy())
            if len(gate_args) != 1:
                raise ValueError(f"expected one observable index for {inst!r}")
            logical_index = int(gate_args[0])
            for target in inst.targets_copy():
                if not target.is_measurement_record_target:
                    continue
                absolute_index = measurement_so_far + int(target.value)
                if absolute_index < 0 or absolute_index >= num_measurements:
                    raise ValueError(
                        f"observable target {target!r} resolves outside measurement record "
                        f"(measurement_so_far={measurement_so_far}, num_measurements={num_measurements})"
                    )
                logical_from_measurements[logical_index, absolute_index] ^= 1
            continue
    if detector_index != num_detectors:
        raise AssertionError(f"detector count mismatch while parsing record transforms: {detector_index} vs {num_detectors}")
    return detector_from_measurements, logical_from_measurements, detector_names, logical_names


def odd_parity_merge_probability(probabilities: np.ndarray | list[float] | tuple[float, ...]) -> float:
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0
    product = float(np.prod(1.0 - 2.0 * values))
    merged = 0.5 * (1.0 - product)
    return float(min(max(merged, 0.0), 1.0))


def _signature_bitmask(detector_bits: tuple[int, ...], logical_bits: tuple[int, ...]) -> int:
    mask = 0
    bit_index = 0
    for bit in detector_bits:
        if int(bit):
            mask |= 1 << bit_index
        bit_index += 1
    for bit in logical_bits:
        if int(bit):
            mask |= 1 << bit_index
        bit_index += 1
    return mask


def _bitmask_to_signature(mask: int, *, num_detectors: int, num_logicals: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    detector_bits = tuple((int(mask) >> index) & 1 for index in range(num_detectors))
    logical_bits = tuple((int(mask) >> (num_detectors + index)) & 1 for index in range(num_logicals))
    return detector_bits, logical_bits


def _xor_of_basis_subset(basis: tuple[int, ...], subset_mask: int) -> int:
    value = 0
    for index, basis_vector in enumerate(basis):
        if (int(subset_mask) >> index) & 1:
            value ^= int(basis_vector)
    return value


def _is_in_span(basis: tuple[int, ...], value: int) -> bool:
    if int(value) == 0:
        return True
    for subset_mask in range(1 << len(basis)):
        if _xor_of_basis_subset(basis, subset_mask) == int(value):
            return True
    return False


def _independent_generator_terms_for_location(
    *,
    signature_probabilities: dict[int, float],
    raw_fault_ids: tuple[int, ...],
) -> tuple[tuple[int, float, tuple[int, ...]], ...]:
    nonzero_signatures = [int(signature) for signature, probability in signature_probabilities.items() if signature and probability > 0.0]
    if not nonzero_signatures:
        return tuple()

    basis: list[int] = []
    for signature in nonzero_signatures:
        if not _is_in_span(tuple(basis), signature):
            basis.append(int(signature))
    basis_tuple = tuple(basis)
    dimension = len(basis_tuple)
    coord_to_signature = {
        subset_mask: _xor_of_basis_subset(basis_tuple, subset_mask) for subset_mask in range(1 << dimension)
    }
    signature_to_coord = {signature: coord for coord, signature in coord_to_signature.items()}
    q = np.zeros(1 << dimension, dtype=np.float64)
    total_raw = 0.0
    for signature, probability in signature_probabilities.items():
        if probability <= 0.0:
            continue
        coord = signature_to_coord[int(signature)]
        q[coord] += float(probability)
        total_raw += float(probability)
    q[0] += max(0.0, 1.0 - total_raw)

    if dimension == 0:
        return tuple()
    states = list(range(1, 1 << dimension))
    if not states:
        return tuple()
    a_matrix = np.asarray(
        [[((u & y).bit_count() & 1) for y in states] for u in states],
        dtype=np.float64,
    )
    phi = np.zeros(len(states), dtype=np.float64)
    for row, u in enumerate(states):
        total = 0.0
        for x, probability in enumerate(q):
            total += float(probability) * (1.0 if ((u & x).bit_count() & 1) == 0 else -1.0)
        phi[row] = total
    if np.any(phi <= 0.0):
        raise ValueError(f"non-positive characteristic values encountered during location factorization: {phi!r}")
    t = np.linalg.solve(a_matrix, np.log(phi))
    out: list[tuple[int, float, tuple[int, ...]]] = []
    for state, log_term in zip(states, t.tolist()):
        probability = 0.5 * (1.0 - np.exp(float(log_term)))
        probability = float(min(max(probability, 0.0), 1.0))
        if probability <= 1e-18:
            continue
        out.append((int(coord_to_signature[int(state)]), probability, raw_fault_ids))
    return tuple(out)


def _location_signature_groups(
    *,
    faults: tuple[FaultLocation, ...],
    raw_detector_matrix: np.ndarray,
    raw_logical_matrix: np.ndarray,
) -> dict[tuple[int, int, tuple[int, ...]], dict[int, list[int]]]:
    location_signature_groups: dict[tuple[int, int, tuple[int, ...]], dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    n_faults = int(raw_detector_matrix.shape[1])
    for col in range(n_faults):
        detector_bits = tuple(int(x) for x in raw_detector_matrix[:, col].tolist())
        logical_bits = tuple(int(x) for x in raw_logical_matrix[:, col].tolist())
        fault = faults[col]
        location_key = (
            int(fault.instruction_index),
            int(fault.target_group_index),
            tuple(int(q) for q in fault.qubits),
        )
        signature_mask = _signature_bitmask(detector_bits, logical_bits)
        location_signature_groups[location_key][signature_mask].append(col)
    return location_signature_groups


def merge_identical_fault_columns(
    *,
    faults: tuple[FaultLocation, ...],
    raw_detector_matrix: np.ndarray,
    raw_logical_matrix: np.ndarray,
    raw_probabilities: np.ndarray,
    drop_invisible: bool = True,
) -> tuple[MergedColumn, ...]:
    num_detectors = int(raw_detector_matrix.shape[0])
    num_logicals = int(raw_logical_matrix.shape[0])
    location_signature_groups = _location_signature_groups(
        faults=faults,
        raw_detector_matrix=raw_detector_matrix,
        raw_logical_matrix=raw_logical_matrix,
    )

    global_generator_groups: dict[int, list[tuple[float, tuple[int, ...]]]] = defaultdict(list)
    for signature_groups in location_signature_groups.values():
        signature_probabilities = {
            int(signature_mask): float(np.sum(raw_probabilities[raw_fault_ids]))
            for signature_mask, raw_fault_ids in signature_groups.items()
        }
        all_location_raw_ids = tuple(sorted(raw_fault_id for raw_fault_ids in signature_groups.values() for raw_fault_id in raw_fault_ids))
        for signature_mask, probability, source_raw_ids in _independent_generator_terms_for_location(
            signature_probabilities=signature_probabilities,
            raw_fault_ids=all_location_raw_ids,
        ):
            global_generator_groups[int(signature_mask)].append((float(probability), tuple(int(x) for x in source_raw_ids)))
    merged: list[MergedColumn] = []
    for signature_mask, location_items in global_generator_groups.items():
        detector_bits, logical_bits = _bitmask_to_signature(
            int(signature_mask),
            num_detectors=num_detectors,
            num_logicals=num_logicals,
        )
        if drop_invisible and not any(detector_bits) and not any(logical_bits):
            continue
        location_probabilities = [item[0] for item in location_items]
        raw_fault_ids = tuple(sorted(raw_fault_id for _, ids in location_items for raw_fault_id in ids))
        merged.append(
            MergedColumn(
                detector_bits=detector_bits,
                logical_bits=logical_bits,
                probability=odd_parity_merge_probability(location_probabilities),
                raw_fault_ids=raw_fault_ids,
            )
        )
    merged.sort(key=lambda item: (item.detector_bits, item.logical_bits))
    return tuple(merged)


def merge_column_statistics(
    *,
    faults: tuple[FaultLocation, ...],
    raw_detector_matrix: np.ndarray,
    raw_logical_matrix: np.ndarray,
    raw_probabilities: np.ndarray,
    merged_columns: tuple[MergedColumn, ...],
) -> MergeStatistics:
    visible_masks: list[int] = []
    visible_detector_masks: list[int] = []
    num_detectors = int(raw_detector_matrix.shape[0])
    for column in range(int(raw_detector_matrix.shape[1])):
        detector_bits = tuple(int(x) for x in raw_detector_matrix[:, column].tolist())
        logical_bits = tuple(int(x) for x in raw_logical_matrix[:, column].tolist())
        signature_mask = _signature_bitmask(detector_bits, logical_bits)
        if signature_mask == 0:
            continue
        visible_masks.append(int(signature_mask))
        detector_mask = int(signature_mask) & ((1 << num_detectors) - 1)
        visible_detector_masks.append(detector_mask)

    location_signature_groups = _location_signature_groups(
        faults=faults,
        raw_detector_matrix=raw_detector_matrix,
        raw_logical_matrix=raw_logical_matrix,
    )
    nonzero_location_signature_count = sum(
        1 for signature_groups in location_signature_groups.values() for signature_mask in signature_groups if int(signature_mask) != 0
    )
    location_generator_count = 0
    for signature_groups in location_signature_groups.values():
        signature_probabilities = {
            int(signature_mask): float(np.sum(raw_probabilities[raw_fault_ids]))
            for signature_mask, raw_fault_ids in signature_groups.items()
        }
        all_location_raw_ids = tuple(
            sorted(raw_fault_id for raw_fault_ids in signature_groups.values() for raw_fault_id in raw_fault_ids)
        )
        location_generator_count += len(
            _independent_generator_terms_for_location(
                signature_probabilities=signature_probabilities,
                raw_fault_ids=all_location_raw_ids,
            )
        )

    merged_detector_masks = [
        _signature_bitmask(item.detector_bits, tuple(0 for _ in item.logical_bits)) for item in merged_columns
    ]
    merged_detector_counter = Counter(int(mask) for mask in merged_detector_masks)
    merged_detector_collision_count = sum(
        count for count in merged_detector_counter.values() if int(count) > 1
    )
    return MergeStatistics(
        raw_fault_count=len(faults),
        visible_raw_fault_count=len(visible_masks),
        location_count=len(location_signature_groups),
        nonzero_location_signature_count=nonzero_location_signature_count,
        location_generator_count=location_generator_count,
        unique_raw_detector_signature_count=len(set(visible_detector_masks)),
        unique_raw_detector_logical_signature_count=len(set(visible_masks)),
        final_merged_column_count=len(merged_columns),
        final_merged_detector_signature_count=len(set(int(mask) for mask in merged_detector_masks)),
        final_merged_detector_signature_collision_count=merged_detector_collision_count,
    )


def location_signature_summaries(
    *,
    faults: tuple[FaultLocation, ...],
    raw_detector_matrix: np.ndarray,
    raw_logical_matrix: np.ndarray,
    raw_probabilities: np.ndarray,
) -> tuple[LocationSignatureSummary, ...]:
    location_signature_groups = _location_signature_groups(
        faults=faults,
        raw_detector_matrix=raw_detector_matrix,
        raw_logical_matrix=raw_logical_matrix,
    )
    representative_faults: dict[tuple[int, int, tuple[int, ...]], FaultLocation] = {}
    for fault in faults:
        location_key = (
            int(fault.instruction_index),
            int(fault.target_group_index),
            tuple(int(q) for q in fault.qubits),
        )
        representative_faults.setdefault(location_key, fault)

    summaries: list[LocationSignatureSummary] = []
    for location_key in sorted(location_signature_groups):
        signature_groups = location_signature_groups[location_key]
        signature_probabilities = {
            int(signature_mask): float(np.sum(raw_probabilities[raw_fault_ids]))
            for signature_mask, raw_fault_ids in signature_groups.items()
        }
        all_location_raw_ids = tuple(
            sorted(raw_fault_id for raw_fault_ids in signature_groups.values() for raw_fault_id in raw_fault_ids)
        )
        representative = representative_faults[location_key]
        summaries.append(
            LocationSignatureSummary(
                instruction_index=int(representative.instruction_index),
                instruction_name=representative.instruction_name,
                target_group_index=int(representative.target_group_index),
                qubits=tuple(int(q) for q in representative.qubits),
                instruction_text=representative.instruction_text,
                elementary_fault_count=len(all_location_raw_ids),
                nonzero_signature_count=sum(1 for signature_mask in signature_groups if int(signature_mask) != 0),
                generator_count=len(
                    _independent_generator_terms_for_location(
                        signature_probabilities=signature_probabilities,
                        raw_fault_ids=all_location_raw_ids,
                    )
                ),
            )
        )
    return tuple(summaries)


def stim_dem_columns(circuit: stim.Circuit) -> tuple[MergedColumn, ...]:
    dem = circuit.detector_error_model(decompose_errors=False)
    columns: list[MergedColumn] = []
    num_detectors = int(circuit.num_detectors)
    num_logicals = int(circuit.num_observables)
    for inst in dem.flattened():
        if str(inst.type) != "error":
            continue
        detector_bits = np.zeros(num_detectors, dtype=np.uint8)
        logical_bits = np.zeros(num_logicals, dtype=np.uint8)
        for target in inst.targets_copy():
            if target.is_relative_detector_id():
                detector_bits[int(target.val)] ^= 1
            elif target.is_logical_observable_id():
                logical_bits[int(target.val)] ^= 1
            elif target.is_separator():
                continue
            else:
                raise NotImplementedError(f"unsupported DEM target in Stim column parser: {target!r}")
        columns.append(
            MergedColumn(
                detector_bits=tuple(int(x) for x in detector_bits.tolist()),
                logical_bits=tuple(int(x) for x in logical_bits.tolist()),
                probability=float(inst.args_copy()[0]),
                raw_fault_ids=(),
            )
        )
    columns.sort(key=lambda item: (item.detector_bits, item.logical_bits))
    return tuple(columns)


def dem_matrix_reference_columns(circuit: stim.Circuit) -> tuple[MergedColumn, ...]:
    from ldpc.ckt_noise.dem_matrices import detector_error_model_to_check_matrices  # type: ignore

    dem = circuit.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)
    matrices = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True)
    check_matrix = matrices.check_matrix.tocsc()
    observables_matrix = matrices.observables_matrix.tocsc()
    probabilities = np.asarray(matrices.priors, dtype=np.float64).reshape(-1)
    num_columns = int(check_matrix.shape[1])
    columns: list[MergedColumn] = []
    for column_index in range(num_columns):
        det_start = int(check_matrix.indptr[int(column_index)])
        det_stop = int(check_matrix.indptr[int(column_index) + 1])
        detector_bits = np.zeros(int(check_matrix.shape[0]), dtype=np.uint8)
        if int(det_stop) > int(det_start):
            detector_bits[check_matrix.indices[det_start:det_stop].astype(np.int32, copy=False)] = 1
        log_start = int(observables_matrix.indptr[int(column_index)])
        log_stop = int(observables_matrix.indptr[int(column_index) + 1])
        logical_bits = np.zeros(int(observables_matrix.shape[0]), dtype=np.uint8)
        if int(log_stop) > int(log_start):
            logical_bits[observables_matrix.indices[log_start:log_stop].astype(np.int32, copy=False)] = 1
        columns.append(
            MergedColumn(
                detector_bits=tuple(int(x) for x in detector_bits.tolist()),
                logical_bits=tuple(int(x) for x in logical_bits.tolist()),
                probability=float(probabilities[int(column_index)]),
                raw_fault_ids=(),
            )
        )
    columns.sort(key=lambda item: (item.detector_bits, item.logical_bits))
    return tuple(columns)


def assert_columns_match(
    left: tuple[MergedColumn, ...],
    right: tuple[MergedColumn, ...],
    *,
    atol: float = 1e-15,
    rtol: float = 1e-12,
) -> None:
    if len(left) != len(right):
        raise AssertionError(f"column count mismatch: {len(left)} vs {len(right)}")
    for index, (lhs, rhs) in enumerate(zip(left, right)):
        if lhs.detector_bits != rhs.detector_bits or lhs.logical_bits != rhs.logical_bits:
            raise AssertionError(
                f"signature mismatch at column {index}: "
                f"lhs={(lhs.detector_bits, lhs.logical_bits)} rhs={(rhs.detector_bits, rhs.logical_bits)}"
            )
        if not np.isclose(float(lhs.probability), float(rhs.probability), rtol=float(rtol), atol=float(atol)):
            raise AssertionError(
                f"probability mismatch at column {index}: {lhs.probability:.18g} vs {rhs.probability:.18g}"
            )


def _gf2_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lhs = np.asarray(left, dtype=np.uint8)
    rhs = np.asarray(right, dtype=np.uint8)
    if lhs.ndim != 2 or rhs.ndim != 2:
        raise ValueError(f"expected 2-D matrices, got shapes {lhs.shape} and {rhs.shape}")
    if int(lhs.shape[1]) != int(rhs.shape[0]):
        raise ValueError(f"matrix shape mismatch: {lhs.shape} cannot multiply {rhs.shape}")
    try:
        from scipy import sparse
    except ImportError:
        return ((lhs @ rhs) & np.uint8(1)).astype(np.uint8, copy=False)

    product = sparse.csr_matrix(lhs.astype(np.int8, copy=False)) @ rhs.astype(np.int8, copy=False)
    return np.bitwise_and(np.asarray(product), 1).astype(np.uint8, copy=False)


def _next_multiple_of_256(value: int) -> int:
    if value <= 0:
        return 256
    return ((int(value) + 255) // 256) * 256


def _measurement_suffix_cache(
    circuit: stim.Circuit,
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[list[int], dict[int, tuple[int, stim.Circuit]]]:
    flat = list(circuit.flattened())
    measurement_prefix = [0]
    for inst in flat:
        measurement_prefix.append(measurement_prefix[-1] + _measurement_target_count(inst))

    suffix_instruction_indices = tuple(
        instruction_index
        for instruction_index, inst in enumerate(flat)
        if str(inst.name) in _PURE_NOISE_GATES or str(inst.name) in _NOISY_MEASUREMENT_GATES
    )
    total_suffixes = len(suffix_instruction_indices)
    if progress_callback is not None:
        progress_callback(
            f"building measurement suffix cache for {total_suffixes} noisy instructions "
            f"over {len(flat)} flattened circuit instructions"
        )
    cache: dict[int, tuple[int, stim.Circuit]] = {}
    for suffix_index, instruction_index in enumerate(suffix_instruction_indices, start=1):
        inst = flat[instruction_index]
        name = str(inst.name)
        start_index = instruction_index if name in _NOISY_MEASUREMENT_GATES else instruction_index + 1
        suffix = stim.Circuit()
        for later in flat[start_index:]:
            _append_measurement_only_instruction_without_noise(suffix, later)
        cache[instruction_index] = (measurement_prefix[start_index], suffix)
        if progress_callback is not None and (
            suffix_index == 1 or suffix_index == total_suffixes or suffix_index % 5 == 0
        ):
            progress_callback(
                f"measurement suffix cache {suffix_index}/{total_suffixes} "
                f"(instruction {instruction_index}/{len(flat)})"
            )
    return measurement_prefix, cache


def compute_fault_measurement_flip_matrix(
    circuit: stim.Circuit,
    faults: tuple[FaultLocation, ...],
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> np.ndarray:
    num_measurements = int(circuit.num_measurements)
    num_faults = len(faults)
    raw_measurement_matrix = np.zeros((num_measurements, num_faults), dtype=np.uint8)
    faults_by_instruction: dict[int, list[int]] = defaultdict(list)
    for column, fault in enumerate(faults):
        faults_by_instruction[int(fault.instruction_index)].append(column)

    _, suffix_cache = _measurement_suffix_cache(circuit, progress_callback=progress_callback)
    total_groups = len(faults_by_instruction)
    processed_faults = 0
    for group_index, instruction_index in enumerate(sorted(faults_by_instruction), start=1):
        column_indices = faults_by_instruction[instruction_index]
        measurement_offset, suffix = suffix_cache[instruction_index]
        suffix_measurements = int(suffix.num_measurements)
        for start in range(0, len(column_indices), 1024):
            chunk = column_indices[start : start + 1024]
            padded_batch_size = _next_multiple_of_256(len(chunk))
            sim = stim.FlipSimulator(
                batch_size=padded_batch_size,
                disable_stabilizer_randomization=True,
                num_qubits=int(circuit.num_qubits),
            )
            for local_index, column in enumerate(chunk):
                fault = faults[column]
                for qubit, pauli in zip(fault.qubits, fault.pauli_product):
                    if pauli == "I":
                        continue
                    sim.set_pauli_flip(pauli, qubit_index=int(qubit), instance_index=int(local_index))
            sim.do(suffix)
            packed = np.asarray(sim.get_measurement_flips(bit_packed=True), dtype=np.uint8)
            unpacked = np.unpackbits(packed, axis=1, bitorder="little")[:, : len(chunk)].astype(np.uint8, copy=False)
            if suffix_measurements:
                raw_measurement_matrix[measurement_offset : measurement_offset + suffix_measurements, chunk] = unpacked
            processed_faults += len(chunk)
        if progress_callback is not None and (
            group_index == 1 or group_index == total_groups or group_index % 5 == 0
        ):
            progress_callback(
                f"fault groups {group_index}/{total_groups}, processed {processed_faults}/{num_faults} elementary faults"
            )
    return raw_measurement_matrix


def analyze_circuit_faults(
    circuit: stim.Circuit,
    *,
    detector_names: tuple[str, ...] | None = None,
    logical_names: tuple[str, ...] | None = None,
    measurement_channel_names: tuple[str, ...] = (),
    measurement_channel_extractor: Callable[[np.ndarray], np.ndarray] | None = None,
    detector_from_channels: np.ndarray | None = None,
    logical_from_channels: np.ndarray | None = None,
    use_raw_measurements_as_channels: bool = False,
    progress_callback: Callable[[str], None] | None = None,
    reference_columns: tuple[MergedColumn, ...] | None = None,
) -> CircuitFaultAnalysis:
    faults = enumerate_fault_locations(circuit)
    no_noise_circuit = circuit.without_noise()
    reference_measurements = np.asarray(no_noise_circuit.compile_sampler(seed=1).sample(shots=1)[0], dtype=np.uint8)
    reference_channels: np.ndarray | None = None
    if measurement_channel_extractor is not None:
        reference_channels = np.asarray(measurement_channel_extractor(reference_measurements), dtype=np.uint8).reshape(-1)
        if reference_channels.size != len(measurement_channel_names):
            raise ValueError(
                f"channel extractor returned {reference_channels.size} bits, expected {len(measurement_channel_names)}"
            )
    elif use_raw_measurements_as_channels:
        if measurement_channel_names and len(measurement_channel_names) != int(circuit.num_measurements):
            raise ValueError(
                f"raw measurement channel name count mismatch: {len(measurement_channel_names)} vs {int(circuit.num_measurements)}"
            )
        reference_channels = reference_measurements
        if not measurement_channel_names:
            measurement_channel_names = tuple(f"m{index}" for index in range(int(circuit.num_measurements)))

    num_faults = len(faults)
    detector_names = detector_names or tuple(f"D{index}" for index in range(int(circuit.num_detectors)))
    logical_names = logical_names or tuple(f"L{index}" for index in range(int(circuit.num_observables)))
    raw_probabilities = np.asarray([fault.probability for fault in faults], dtype=np.float64)
    raw_measurement_flips = compute_fault_measurement_flip_matrix(
        circuit,
        faults,
        progress_callback=progress_callback,
    )

    raw_measurement_channel_matrix: np.ndarray | None = None
    if use_raw_measurements_as_channels:
        raw_measurement_channel_matrix = raw_measurement_flips
    elif measurement_channel_extractor is not None:
        raw_measurement_channel_matrix = np.zeros((len(measurement_channel_names), num_faults), dtype=np.uint8)
        for column in range(num_faults):
            measurement_outcome = np.bitwise_xor(reference_measurements, raw_measurement_flips[:, column])
            channel_bits = np.asarray(measurement_channel_extractor(measurement_outcome), dtype=np.uint8).reshape(-1)
            raw_measurement_channel_matrix[:, column] = np.bitwise_xor(channel_bits, reference_channels)

    if raw_measurement_channel_matrix is None:
        raise ValueError("analyze_circuit_faults requires either a measurement_channel_extractor or use_raw_measurements_as_channels=True")
    if detector_from_channels is None or logical_from_channels is None:
        raise ValueError("analyze_circuit_faults requires explicit channel-to-detector and channel-to-logical transforms")

    if progress_callback is not None:
        progress_callback("projecting raw faults to detector/logical matrices")
    raw_detector_matrix = _gf2_matmul(np.asarray(detector_from_channels, dtype=np.uint8), raw_measurement_channel_matrix)
    raw_logical_matrix = _gf2_matmul(np.asarray(logical_from_channels, dtype=np.uint8), raw_measurement_channel_matrix)
    if progress_callback is not None:
        progress_callback(
            f"projected raw matrices detector_shape={raw_detector_matrix.shape} "
            f"logical_shape={raw_logical_matrix.shape}"
        )

    merged_columns = merge_identical_fault_columns(
        faults=faults,
        raw_detector_matrix=raw_detector_matrix,
        raw_logical_matrix=raw_logical_matrix,
        raw_probabilities=raw_probabilities,
        drop_invisible=True,
    )
    merge_statistics = merge_column_statistics(
        faults=faults,
        raw_detector_matrix=raw_detector_matrix,
        raw_logical_matrix=raw_logical_matrix,
        raw_probabilities=raw_probabilities,
        merged_columns=merged_columns,
    )
    stim_columns = tuple(reference_columns) if reference_columns is not None else stim_dem_columns(circuit)
    assert_columns_match(merged_columns, stim_columns)
    return CircuitFaultAnalysis(
        faults=faults,
        raw_probabilities=raw_probabilities,
        raw_detector_matrix=raw_detector_matrix,
        raw_logical_matrix=raw_logical_matrix,
        merged_columns=merged_columns,
        stim_columns=stim_columns,
        detector_names=tuple(detector_names),
        logical_names=tuple(logical_names),
        measurement_channel_names=tuple(measurement_channel_names),
        raw_measurement_channel_matrix=raw_measurement_channel_matrix,
        detector_from_channels=None if detector_from_channels is None else np.asarray(detector_from_channels, dtype=np.uint8),
        logical_from_channels=None if logical_from_channels is None else np.asarray(logical_from_channels, dtype=np.uint8),
        raw_measurement_reference=reference_measurements,
        measurement_channel_reference=reference_channels,
        merge_statistics=merge_statistics,
    )


@dataclass(frozen=True)
class Toy422CircuitSpec:
    circuit: stim.Circuit
    raw_measurement_names: tuple[str, ...]
    measurement_channel_names: tuple[str, ...]
    detector_names: tuple[str, ...]
    logical_names: tuple[str, ...]
    detector_from_channels: np.ndarray
    logical_from_channels: np.ndarray


def build_toy_422_memory_z_circuit(*, error_rate: float = 0.004) -> Toy422CircuitSpec:
    p = float(error_rate)
    circuit = stim.Circuit()
    circuit.append("R", [0, 1, 2, 3], [])
    circuit.append("DEPOLARIZE1", [0, 1, 2, 3], [p])
    circuit.append("H", [0], [])
    circuit.append("DEPOLARIZE1", [0], [p])
    for target in (1, 2, 3):
        circuit.append("CX", [0, target], [])
        circuit.append("DEPOLARIZE2", [0, target], [p])

    raw_measurement_names: list[str] = []
    for round_index in range(2):
        circuit.append("RX", [4], [])
        circuit.append("R", [5], [])
        circuit.append("DEPOLARIZE1", [4, 5], [p])
        for data_qubit in (0, 1, 2, 3):
            circuit.append("CX", [4, data_qubit], [])
            circuit.append("DEPOLARIZE2", [4, data_qubit], [p])
        for data_qubit in (0, 1, 2, 3):
            circuit.append("CX", [data_qubit, 5], [])
            circuit.append("DEPOLARIZE2", [data_qubit, 5], [p])
        circuit.append("MX", [4], [p])
        raw_measurement_names.append(f"mx_stab_r{round_index}")
        circuit.append("M", [5], [p])
        raw_measurement_names.append(f"mz_stab_r{round_index}")

    circuit.append("TICK", [], [])
    circuit.append("M", [0, 1, 2, 3], [p])
    raw_measurement_names.extend(("m_data_0", "m_data_1", "m_data_2", "m_data_3"))

    circuit.append("DETECTOR", [stim.target_rec(-8)], [])
    circuit.append("DETECTOR", [stim.target_rec(-7)], [])
    circuit.append("DETECTOR", [stim.target_rec(-8), stim.target_rec(-6)], [])
    circuit.append("DETECTOR", [stim.target_rec(-7), stim.target_rec(-5)], [])
    circuit.append(
        "DETECTOR",
        [
            stim.target_rec(-5),
            stim.target_rec(-4),
            stim.target_rec(-3),
            stim.target_rec(-2),
            stim.target_rec(-1),
        ],
        [],
    )
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-4), stim.target_rec(-3)], [0])
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-4), stim.target_rec(-2)], [1])

    measurement_channel_names = (
        "sx_r0",
        "sz_r0",
        "sx_r1",
        "sz_r1",
        "sz_final",
        "lz0",
        "lz1",
    )
    detector_names = ("D0", "D1", "D2", "D3", "D4")
    logical_names = ("L0", "L1")
    detector_from_channels = np.asarray(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
        ],
        dtype=np.uint8,
    )
    logical_from_channels = np.asarray(
        [
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=np.uint8,
    )
    return Toy422CircuitSpec(
        circuit=circuit,
        raw_measurement_names=tuple(raw_measurement_names),
        measurement_channel_names=measurement_channel_names,
        detector_names=detector_names,
        logical_names=logical_names,
        detector_from_channels=detector_from_channels,
        logical_from_channels=logical_from_channels,
    )


def toy_422_measurement_channels(measurements: np.ndarray) -> np.ndarray:
    bits = np.asarray(measurements, dtype=np.uint8).reshape(-1)
    if bits.size != 8:
        raise ValueError(f"expected 8 raw measurements for the toy [[4,2,2]] circuit, got {bits.size}")
    return np.asarray(
        [
            bits[0],
            bits[1],
            bits[2],
            bits[3],
            bits[4] ^ bits[5] ^ bits[6] ^ bits[7],
            bits[4] ^ bits[5],
            bits[4] ^ bits[6],
        ],
        dtype=np.uint8,
    )


def analyze_toy_422_memory_z(*, error_rate: float = 0.004) -> tuple[Toy422CircuitSpec, CircuitFaultAnalysis]:
    spec = build_toy_422_memory_z_circuit(error_rate=error_rate)
    analysis = analyze_circuit_faults(
        spec.circuit,
        detector_names=spec.detector_names,
        logical_names=spec.logical_names,
        measurement_channel_names=spec.measurement_channel_names,
        measurement_channel_extractor=toy_422_measurement_channels,
        detector_from_channels=spec.detector_from_channels,
        logical_from_channels=spec.logical_from_channels,
    )
    return spec, analysis


@dataclass(frozen=True)
class GrossSplitSectorCircuitSpec:
    backend: str
    sector: str
    error_rate: float
    stim_path: str
    circuit: stim.Circuit
    measurement_channel_names: tuple[str, ...]
    detector_names: tuple[str, ...]
    logical_names: tuple[str, ...]
    detector_from_channels: np.ndarray
    logical_from_channels: np.ndarray


def build_gross_split_sector_circuit_spec(
    *,
    sector: str,
    backend: str = "bravyi_depth7",
    error_rate: float = 0.004,
) -> GrossSplitSectorCircuitSpec:
    from grosscode.circuits.backends import resolve_backend_circuit

    resolved = resolve_backend_circuit(backend=backend, sector=sector, error_rate=error_rate)
    circuit = stim.Circuit.from_file(str(resolved.stim_path))
    measurement_names = circuit_measurement_names(circuit)
    detector_from_measurements, logical_from_measurements, detector_names, logical_names = (
        circuit_record_to_detector_logical_matrices(circuit)
    )
    return GrossSplitSectorCircuitSpec(
        backend=str(backend),
        sector=str(sector),
        error_rate=float(error_rate),
        stim_path=str(resolved.stim_path),
        circuit=circuit,
        measurement_channel_names=measurement_names,
        detector_names=detector_names,
        logical_names=logical_names,
        detector_from_channels=detector_from_measurements,
        logical_from_channels=logical_from_measurements,
    )


def _analyze_gross_split_sector_spec(
    spec: GrossSplitSectorCircuitSpec,
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> CircuitFaultAnalysis:
    return analyze_circuit_faults(
        spec.circuit,
        detector_names=spec.detector_names,
        logical_names=spec.logical_names,
        measurement_channel_names=spec.measurement_channel_names,
        detector_from_channels=spec.detector_from_channels,
        logical_from_channels=spec.logical_from_channels,
        use_raw_measurements_as_channels=True,
        progress_callback=progress_callback,
        reference_columns=dem_matrix_reference_columns(spec.circuit),
    )


def _sanitize_cache_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _cache_rate_tag(value: float) -> str:
    return str(float(value)).replace("-", "m").replace(".", "p")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _measurement_channel_names_sha256(measurement_channel_names: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name in tuple(str(item) for item in measurement_channel_names):
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _gross_split_sector_correction_cache_path(spec: GrossSplitSectorCircuitSpec) -> Path:
    stim_path = Path(str(spec.stim_path))
    fingerprint_payload = {
        "version": int(_GROSS_SPLIT_SECTOR_CORRECTION_MAP_DISK_CACHE_VERSION),
        "stim_sha256": _sha256_file(stim_path),
        "measurement_channel_sha256": _measurement_channel_names_sha256(spec.measurement_channel_names),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    cache_root = resolve_cache_root() / "gross_split_sector_correction_maps"
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root / (
        f"{_sanitize_cache_token(spec.backend)}_{_sanitize_cache_token(spec.sector)}_"
        f"{_cache_rate_tag(float(spec.error_rate))}_{fingerprint}.json"
    )


def gross_split_sector_correction_cache_path(
    *,
    sector: str,
    backend: str = "bravyi_depth7",
    error_rate: float = 0.004,
) -> Path:
    spec = build_gross_split_sector_circuit_spec(
        sector=str(sector),
        backend=str(backend),
        error_rate=float(error_rate),
    )
    return _gross_split_sector_correction_cache_path(spec)


def _load_gross_split_sector_correction_map_from_disk(
    cache_path: Path,
) -> tuple[dict[int, int], tuple[int, ...]] | None:
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if int(payload.get("version", -1)) != int(_GROSS_SPLIT_SECTOR_CORRECTION_MAP_DISK_CACHE_VERSION):
            return None
        final_qubits = tuple(int(value) for value in payload["final_qubits"])
        entries = payload["entries"]
        correction_by_signature = {
            int(str(signature_hex), 16): int(str(correction_hex), 16) for signature_hex, correction_hex in entries
        }
        return correction_by_signature, final_qubits
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return None


def _write_gross_split_sector_correction_map_to_disk(
    cache_path: Path,
    *,
    spec: GrossSplitSectorCircuitSpec,
    correction_by_signature: dict[int, int],
    final_qubits: tuple[int, ...],
) -> None:
    payload = {
        "version": int(_GROSS_SPLIT_SECTOR_CORRECTION_MAP_DISK_CACHE_VERSION),
        "backend": str(spec.backend),
        "sector": str(spec.sector),
        "error_rate": float(spec.error_rate),
        "stim_path": str(spec.stim_path),
        "stim_sha256": _sha256_file(Path(str(spec.stim_path))),
        "measurement_channel_sha256": _measurement_channel_names_sha256(spec.measurement_channel_names),
        "final_qubits": [int(value) for value in final_qubits],
        "entries": [
            [format(int(signature_mask), "x"), format(int(correction_mask), "x")]
            for signature_mask, correction_mask in sorted(correction_by_signature.items(), key=lambda item: int(item[0]))
        ],
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp.{os.getpid()}")
    temp_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    temp_path.replace(cache_path)


def analyze_gross_split_sector_circuit(
    *,
    sector: str,
    backend: str = "bravyi_depth7",
    error_rate: float = 0.004,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[GrossSplitSectorCircuitSpec, CircuitFaultAnalysis]:
    spec = build_gross_split_sector_circuit_spec(
        sector=sector,
        backend=backend,
        error_rate=error_rate,
    )
    analysis = _analyze_gross_split_sector_spec(spec, progress_callback=progress_callback)
    return spec, analysis


def build_gross_split_sector_merged_correction_map(
    *,
    sector: str,
    backend: str = "bravyi_depth7",
    error_rate: float = 0.004,
    progress_callback: Callable[[str], None] | None = None,
    cache_policy: str = "build_if_missing",
) -> tuple[dict[int, int], tuple[int, ...]]:
    cache_policy_key = str(cache_policy).strip().lower()
    if cache_policy_key not in {"build_if_missing", "require_disk"}:
        raise ValueError("cache_policy must be one of ['build_if_missing', 'require_disk']")
    cache_key = (str(sector), str(backend), float(error_rate))
    cached = _GROSS_SPLIT_SECTOR_CORRECTION_MAP_CACHE.get(cache_key)
    if cached is not None:
        return cached
    spec = build_gross_split_sector_circuit_spec(
        sector=str(sector),
        backend=str(backend),
        error_rate=float(error_rate),
    )
    expected_final_qubits = gross_split_sector_final_data_measurement_rows(spec.measurement_channel_names)[1]
    cache_path = _gross_split_sector_correction_cache_path(spec)
    disk_cached = _load_gross_split_sector_correction_map_from_disk(cache_path)
    if disk_cached is not None:
        correction_by_signature, final_qubits = disk_cached
        if tuple(int(value) for value in final_qubits) == tuple(int(value) for value in expected_final_qubits):
            out = (correction_by_signature, tuple(int(value) for value in final_qubits))
            _GROSS_SPLIT_SECTOR_CORRECTION_MAP_CACHE[cache_key] = out
            if progress_callback is not None:
                progress_callback(f"loaded correction map from disk cache {cache_path}")
            return out
        if progress_callback is not None:
            progress_callback(
                f"ignored stale correction map cache {cache_path} because final qubits did not match the current circuit"
            )
    if str(cache_policy_key) == "require_disk":
        raise FileNotFoundError(
            f"required cached correction map not found for sector={sector!r}, backend={backend!r}, error_rate={float(error_rate):.6g}: {cache_path}"
        )
    analysis = _analyze_gross_split_sector_spec(spec, progress_callback=progress_callback)
    raw_measurement_channel_matrix = analysis.raw_measurement_channel_matrix
    if raw_measurement_channel_matrix is None:
        raise ValueError("gross split-sector fault analysis did not produce a raw measurement channel matrix")
    final_row_indices, final_qubits = gross_split_sector_final_data_measurement_rows(
        spec.measurement_channel_names
    )
    final_data_matrix = np.asarray(
        raw_measurement_channel_matrix[np.asarray(final_row_indices, dtype=np.int32), :],
        dtype=np.uint8,
    )
    correction_by_signature: dict[int, int] = {}
    for merged_column in analysis.merged_columns:
        correction_bits = np.zeros(len(final_row_indices), dtype=np.uint8)
        raw_fault_ids = tuple(int(value) for value in merged_column.raw_fault_ids)
        if raw_fault_ids:
            correction_bits = np.bitwise_xor.reduce(
                final_data_matrix[:, np.asarray(raw_fault_ids, dtype=np.int32)],
                axis=1,
            ).astype(np.uint8, copy=False)
        correction_mask = 0
        for bit_index, bit in enumerate(correction_bits.tolist()):
            if int(bit) != 0:
                correction_mask |= 1 << int(bit_index)
        signature_mask = _signature_bitmask(
            tuple(int(value) for value in merged_column.detector_bits),
            tuple(int(value) for value in merged_column.logical_bits),
        )
        correction_by_signature[int(signature_mask)] = int(correction_mask)
    out = (correction_by_signature, tuple(int(value) for value in final_qubits))
    try:
        _write_gross_split_sector_correction_map_to_disk(
            cache_path,
            spec=spec,
            correction_by_signature=correction_by_signature,
            final_qubits=tuple(int(value) for value in final_qubits),
        )
        if progress_callback is not None:
            progress_callback(f"saved correction map to disk cache {cache_path}")
    except OSError as exc:
        if progress_callback is not None:
            progress_callback(f"warning: failed to write correction map disk cache {cache_path}: {exc}")
    _GROSS_SPLIT_SECTOR_CORRECTION_MAP_CACHE[cache_key] = out
    return out


def merged_column_matrices(
    merged_columns: tuple[MergedColumn, ...],
    *,
    num_detectors: int,
    num_logicals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    detector_matrix = np.zeros((num_detectors, len(merged_columns)), dtype=np.uint8)
    logical_matrix = np.zeros((num_logicals, len(merged_columns)), dtype=np.uint8)
    probabilities = np.zeros(len(merged_columns), dtype=np.float64)
    for column, item in enumerate(merged_columns):
        detector_matrix[:, column] = np.asarray(item.detector_bits, dtype=np.uint8)
        logical_matrix[:, column] = np.asarray(item.logical_bits, dtype=np.uint8)
        probabilities[column] = float(item.probability)
    return detector_matrix, logical_matrix, probabilities


def raw_fault_category_counts(faults: tuple[FaultLocation, ...]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for fault in faults:
        counts[(fault.instruction_name, fault.category)] += 1
    return counts
