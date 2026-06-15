from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path

import numpy as np

from grosscode.circuits.backends import resolve_backend_circuit

PROJECTED_COMPONENT_TO_PUBLIC_SECTOR = {
    "X": "Z",
    "Z": "X",
}

PROJECTED_STATE_LABELS = ("II", "XI", "IX", "XX")


def _state_count_to_kind(state_count: int) -> str:
    size = int(state_count)
    if size == 2:
        return "binary"
    if size == 4:
        return "quaternary"
    return f"state{size}"


def _kind_sort_key(kind: str) -> tuple[int, str]:
    text = str(kind)
    if text == "binary":
        return (2, text)
    if text == "quaternary":
        return (4, text)
    if text.startswith("state"):
        try:
            return (int(text[5:]), text)
        except ValueError:
            return (1_000_000, text)
    return (1_000_000, text)


def _popcount(value: int) -> int:
    return int(bin(int(value)).count("1"))


def _canonical_quaternary_pair(q0: int, q1: int) -> tuple[int, int]:
    first = int(q0)
    second = int(q1)
    return (first, second) if first <= second else (second, first)


def _stack_location_key(location: object) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (
            int(frame.instruction_offset),
            int(frame.iteration_index),
            int(frame.instruction_repetitions_arg),
        )
        for frame in getattr(location, "stack_frames", ())
    )


@dataclass(frozen=True)
class ProjectedColumnMetadata:
    index: int
    kind: str
    merged_multiplicity: int
    source_gate_counts: dict[str, int]
    representative_instruction_offset: int
    representative_tick_offset: int
    representative_qubits: tuple[int, ...]
    detector_weight: int
    logical_weight: int
    error_probability: float
    state_probabilities: tuple[float, ...]

    @property
    def state_count(self) -> int:
        return int(len(self.state_probabilities))

    @property
    def local_rank(self) -> int:
        count = int(self.state_count)
        if count <= 0 or count & (count - 1):
            raise ValueError(f"state count must be a positive power of two, got {count}")
        return int(count.bit_length() - 1)


@dataclass(frozen=True)
class ProjectedLocationMetadata:
    backend: str
    projected_component: str
    public_sector: str
    error_rate: float
    noisy_rounds: int
    total_rounds: int
    stim_path: str
    num_detectors: int
    num_observables: int
    raw_location_count: int
    raw_kind_counts: dict[str, int]
    raw_gate_counts: dict[str, int]
    raw_zero_count: int
    zero_gate_counts: dict[str, int]
    raw_duplicate_detector_only: dict[str, int]
    raw_duplicate_detector_plus_logical: dict[str, int]
    merged_kind_counts: dict[str, int]
    merged_multiplicity_hist: dict[int, int]
    duplicate_source_combo_hist: dict[str, int]
    notes: tuple[str, ...]
    grouping_mode: str = "none"
    grouped_raw_location_count: int = 0
    grouped_raw_kind_counts: dict[str, int] = field(default_factory=dict)
    grouped_attached_binary_count: int = 0
    grouped_unmatched_binary_count: int = 0
    grouped_group_size_hist: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class _RawLocation:
    kind: str
    gate: str
    instruction_offset: int
    tick_offset: int
    qubits: tuple[int, ...]
    error_probability: float


@dataclass(frozen=True)
class _EffectiveLocation:
    kind: str
    source_gate_counts: dict[str, int]
    instruction_offset: int
    tick_offset: int
    qubits: tuple[int, ...]
    state_probabilities: tuple[float, ...]


@dataclass(frozen=True)
class ProjectedLocationProblem:
    metadata: ProjectedLocationMetadata
    columns: tuple[ProjectedColumnMetadata, ...]
    detector_edge_row: np.ndarray
    detector_edge_col: np.ndarray
    detector_edge_mask: np.ndarray
    logical_edge_observable: np.ndarray
    logical_edge_col: np.ndarray
    logical_edge_mask: np.ndarray

    @property
    def n(self) -> int:
        return int(len(self.columns))

    @property
    def m(self) -> int:
        return int(self.metadata.num_detectors)

    @property
    def k(self) -> int:
        return int(self.metadata.num_observables)

    @property
    def n_edges(self) -> int:
        return int(self.detector_edge_col.size)

    @cached_property
    def max_state_count(self) -> int:
        return int(max((column.state_count for column in self.columns), default=1))

    @cached_property
    def max_local_rank(self) -> int:
        return int(max((column.local_rank for column in self.columns), default=0))

    @cached_property
    def max_mask_value(self) -> int:
        detector_max = int(np.max(self.detector_edge_mask)) if int(self.detector_edge_mask.size) else 0
        logical_max = int(np.max(self.logical_edge_mask)) if int(self.logical_edge_mask.size) else 0
        return int(max(detector_max, logical_max))

    @cached_property
    def mask_parity_table(self) -> np.ndarray:
        return np.asarray(
            [
                [(_popcount(int(mask) & int(state)) & 1) for state in range(int(self.max_state_count))]
                for mask in range(int(self.max_mask_value) + 1)
            ],
            dtype=np.uint8,
        )

    @cached_property
    def mask_sign_table(self) -> np.ndarray:
        return np.where(np.asarray(self.mask_parity_table, dtype=np.uint8) == 0, 1.0, -1.0).astype(np.float64)

    @cached_property
    def mask_state_split(self) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        table = np.asarray(self.mask_parity_table, dtype=np.uint8)
        return tuple(
            (
                np.flatnonzero(table[mask] == 0).astype(np.int16, copy=False),
                np.flatnonzero(table[mask] == 1).astype(np.int16, copy=False),
            )
            for mask in range(int(table.shape[0]))
        )

    @cached_property
    def prior_state_probabilities(self) -> np.ndarray:
        probs = np.zeros((self.n, int(self.max_state_count)), dtype=np.float64)
        for index, column in enumerate(self.columns):
            raw = tuple(float(value) for value in column.state_probabilities)
            probs[index, : len(raw)] = np.asarray(raw, dtype=np.float64)
        return probs

    @cached_property
    def prior_state_log_probs(self) -> np.ndarray:
        probs = np.asarray(self.prior_state_probabilities, dtype=np.float64)
        with np.errstate(divide="ignore"):
            return np.where(probs > 0.0, np.log(probs), -np.inf).astype(np.float64)

    @cached_property
    def detector_check_to_edges(self) -> tuple[np.ndarray, ...]:
        edges: list[list[int]] = [[] for _ in range(self.m)]
        for edge_index, check_index in enumerate(np.asarray(self.detector_edge_row, dtype=np.int32).tolist()):
            edges[int(check_index)].append(int(edge_index))
        return tuple(np.asarray(group, dtype=np.int32) for group in edges)

    @cached_property
    def detector_var_to_edges(self) -> tuple[np.ndarray, ...]:
        edges: list[list[int]] = [[] for _ in range(self.n)]
        for edge_index, var_index in enumerate(np.asarray(self.detector_edge_col, dtype=np.int32).tolist()):
            edges[int(var_index)].append(int(edge_index))
        return tuple(np.asarray(group, dtype=np.int32) for group in edges)

    def initial_log_scores(self) -> np.ndarray:
        return np.asarray(self.prior_state_log_probs, dtype=np.float64).copy()

    def syndrome_from_symbols(self, symbols: np.ndarray) -> np.ndarray:
        symbol_vec = np.asarray(symbols, dtype=np.uint8).reshape(-1)
        if int(symbol_vec.size) != self.n:
            raise ValueError(f"symbol vector length mismatch: got {symbol_vec.size}, expected {self.n}")
        out = np.zeros(self.m, dtype=np.uint8)
        if self.n_edges == 0:
            return out
        edge_bits = self.mask_parity_table[
            np.asarray(self.detector_edge_mask, dtype=np.uint8),
            symbol_vec[np.asarray(self.detector_edge_col, dtype=np.int32)],
        ].astype(np.uint8, copy=False)
        np.bitwise_xor.at(out, np.asarray(self.detector_edge_row, dtype=np.int32), edge_bits)
        return out

    def logical_action_from_symbols(self, symbols: np.ndarray) -> np.ndarray:
        symbol_vec = np.asarray(symbols, dtype=np.uint8).reshape(-1)
        if int(symbol_vec.size) != self.n:
            raise ValueError(f"symbol vector length mismatch: got {symbol_vec.size}, expected {self.n}")
        out = np.zeros(self.k, dtype=np.uint8)
        if int(self.logical_edge_col.size) == 0:
            return out
        logical_bits = self.mask_parity_table[
            np.asarray(self.logical_edge_mask, dtype=np.uint8),
            symbol_vec[np.asarray(self.logical_edge_col, dtype=np.int32)],
        ].astype(np.uint8, copy=False)
        np.bitwise_xor.at(out, np.asarray(self.logical_edge_observable, dtype=np.int32), logical_bits)
        return out

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.n == 0:
            symbols = np.zeros(0, dtype=np.uint8)
            return symbols, np.zeros(self.m, dtype=np.uint8), np.zeros(self.k, dtype=np.uint8)
        cdf = np.cumsum(np.asarray(self.prior_state_probabilities, dtype=np.float64), axis=1)
        draws = np.asarray(rng.random(self.n), dtype=np.float64).reshape(-1, 1)
        symbols = np.asarray(np.minimum(np.sum(draws > cdf, axis=1), int(self.max_state_count) - 1), dtype=np.uint8)
        return symbols, self.syndrome_from_symbols(symbols), self.logical_action_from_symbols(symbols)

    def detector_mask_matrix(self):  # type: ignore[no-untyped-def]
        import scipy.sparse as sp  # type: ignore

        return sp.csr_matrix(
            (
                np.asarray(self.detector_edge_mask, dtype=np.uint8),
                (
                    np.asarray(self.detector_edge_row, dtype=np.int32),
                    np.asarray(self.detector_edge_col, dtype=np.int32),
                ),
            ),
            shape=(self.m, self.n),
            dtype=np.uint8,
        )

    def logical_mask_matrix(self):  # type: ignore[no-untyped-def]
        import scipy.sparse as sp  # type: ignore

        return sp.csr_matrix(
            (
                np.asarray(self.logical_edge_mask, dtype=np.uint8),
                (
                    np.asarray(self.logical_edge_observable, dtype=np.int32),
                    np.asarray(self.logical_edge_col, dtype=np.int32),
                ),
            ),
            shape=(self.k, self.n),
            dtype=np.uint8,
        )

    def write_matrix_bundle(self, out_dir: str | Path) -> Path:
        import scipy.sparse as sp  # type: ignore
        from scipy.io import mmwrite  # type: ignore

        dest = Path(out_dir).resolve()
        dest.mkdir(parents=True, exist_ok=True)

        detector_matrix = self.detector_mask_matrix().tocoo()
        logical_matrix = self.logical_mask_matrix().tocoo()
        sp.save_npz(dest / "detector_mask_matrix.npz", detector_matrix.tocsr())
        sp.save_npz(dest / "logical_mask_matrix.npz", logical_matrix.tocsr())
        mmwrite(str(dest / "detector_mask_matrix.mtx"), detector_matrix)
        mmwrite(str(dest / "logical_mask_matrix.mtx"), logical_matrix)

        with (dest / "column_index.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "column",
                    "kind",
                    "state_count",
                    "local_rank",
                    "merged_multiplicity",
                    "source_gate_counts_json",
                    "representative_instruction_offset",
                    "representative_tick_offset",
                    "representative_qubits_json",
                    "detector_weight",
                    "logical_weight",
                    "error_probability",
                    "state_probabilities_json",
                ],
            )
            writer.writeheader()
            for column in self.columns:
                writer.writerow(
                    {
                        "column": int(column.index),
                        "kind": str(column.kind),
                        "state_count": int(column.state_count),
                        "local_rank": int(column.local_rank),
                        "merged_multiplicity": int(column.merged_multiplicity),
                        "source_gate_counts_json": json.dumps(column.source_gate_counts, sort_keys=True),
                        "representative_instruction_offset": int(column.representative_instruction_offset),
                        "representative_tick_offset": int(column.representative_tick_offset),
                        "representative_qubits_json": json.dumps(list(column.representative_qubits)),
                        "detector_weight": int(column.detector_weight),
                        "logical_weight": int(column.logical_weight),
                        "error_probability": float(column.error_probability),
                        "state_probabilities_json": json.dumps(list(column.state_probabilities)),
                    }
                )

        metadata_payload = {
            "model_label": "non-default projected-location mixed-alphabet detector-side matrix",
            "backend": str(self.metadata.backend),
            "projected_component": str(self.metadata.projected_component),
            "public_sector": str(self.metadata.public_sector),
            "error_rate": float(self.metadata.error_rate),
            "grouping_mode": str(self.metadata.grouping_mode),
            "stim_path": str(self.metadata.stim_path),
            "detector_matrix_shape": [int(self.m), int(self.n)],
            "logical_matrix_shape": [int(self.k), int(self.n)],
            "raw_location_count": int(self.metadata.raw_location_count),
            "raw_kind_counts": self.metadata.raw_kind_counts,
            "raw_gate_counts": self.metadata.raw_gate_counts,
            "grouped_raw_location_count": int(self.metadata.grouped_raw_location_count),
            "grouped_raw_kind_counts": self.metadata.grouped_raw_kind_counts,
            "grouped_attached_binary_count": int(self.metadata.grouped_attached_binary_count),
            "grouped_unmatched_binary_count": int(self.metadata.grouped_unmatched_binary_count),
            "grouped_group_size_hist": self.metadata.grouped_group_size_hist,
            "raw_zero_count": int(self.metadata.raw_zero_count),
            "zero_gate_counts": self.metadata.zero_gate_counts,
            "raw_duplicate_detector_only": self.metadata.raw_duplicate_detector_only,
            "raw_duplicate_detector_plus_logical": self.metadata.raw_duplicate_detector_plus_logical,
            "merged_kind_counts": self.metadata.merged_kind_counts,
            "merged_multiplicity_hist": self.metadata.merged_multiplicity_hist,
            "duplicate_source_combo_hist": self.metadata.duplicate_source_combo_hist,
            "max_local_rank": int(self.max_local_rank),
            "max_state_count": int(self.max_state_count),
            "entry_semantics": {
                "state_encoding_rule": "Each merged column has a local binary-coordinate alphabet of size 2^r, where r is the column local_rank.",
                "mask_bits_order": "Bit j in a detector/logical mask corresponds to the j-th canonical local subgroup basis vector.",
                "parity_rule": "For any column state s and edge mask m, row contribution is popcount(m & s) mod 2.",
                "legacy_quaternary_labels": {str(index): label for index, label in enumerate(PROJECTED_STATE_LABELS)},
            },
            "notes": [str(item) for item in self.metadata.notes],
        }
        (dest / "metadata.json").write_text(json.dumps(metadata_payload, indent=2) + "\n")

        report_lines = [
            "# Projected-location mixed-alphabet matrix bundle",
            "",
            "- Model: non-default projected-location mixed-alphabet detector-side matrix.",
            f"- Backend: `{self.metadata.backend}`.",
            f"- Projected component: `{self.metadata.projected_component}`.",
            f"- Public split-sector source circuit: `memory_{self.metadata.public_sector}`.",
            f"- Grouping mode: `{self.metadata.grouping_mode}`.",
            f"- Detector matrix: `{self.m} x {self.n}`.",
            f"- Logical matrix: `{self.k} x {self.n}`.",
            f"- Raw physical noisy locations: `{self.metadata.raw_location_count}`.",
            f"- Effective raw locations after local grouping: `{self.metadata.grouped_raw_location_count}`.",
            f"- Dropped trivial zero-signature locations: `{self.metadata.raw_zero_count}`.",
            f"- Raw duplicate audit (`detector + logical`): `{self.metadata.raw_duplicate_detector_plus_logical['duplicate_class_count']}` duplicate classes, `{self.metadata.raw_duplicate_detector_plus_logical['extra_column_count']}` extra raw columns.",
            f"- Final merged columns: `{self.n}` with alphabet histogram `{json.dumps(self.metadata.merged_kind_counts, sort_keys=True)}`.",
            "",
            "## Files",
            "",
            "- `detector_mask_matrix.mtx` / `detector_mask_matrix.npz`: merged detector mask matrix.",
            "- `logical_mask_matrix.mtx` / `logical_mask_matrix.npz`: merged logical mask matrix.",
            "- `column_index.csv`: merged-column metadata and state probabilities.",
            "- `metadata.json`: machine-readable build summary and mask semantics.",
        ]
        (dest / "report.md").write_text("\n".join(report_lines) + "\n")
        return dest


def _duplicate_summary(groups: list[list[int]]) -> dict[str, int]:
    return {
        "duplicate_class_count": int(len(groups)),
        "duplicate_column_count": int(sum(len(group) for group in groups)),
        "extra_column_count": int(sum(len(group) - 1 for group in groups)),
        "largest_group_size": int(max((len(group) for group in groups), default=1)),
    }


def _projected_bit_present(gate_target: object, component: str) -> bool:
    if str(component) == "X":
        return bool(gate_target.is_x_target or gate_target.is_y_target)
    if str(component) == "Z":
        return bool(gate_target.is_z_target or gate_target.is_y_target)
    raise ValueError(f"unsupported projected component: {component}")


def _detector_and_logical_indices(explained_error: object) -> tuple[list[int], list[int]]:
    detectors: list[int] = []
    logicals: list[int] = []
    for term in explained_error.dem_error_terms:
        dem_target = term.dem_target
        if dem_target.is_relative_detector_id():
            detectors.append(int(dem_target.val))
        elif dem_target.is_logical_observable_id():
            logicals.append(int(dem_target.val))
    return detectors, logicals


def _raw_error_probability(kind: str, gate: str, error_rate: float) -> float:
    p = float(error_rate)
    if str(kind) == "quaternary":
        return float(4.0 * p / 5.0)
    if str(gate) == "DEPOLARIZE1":
        return float(2.0 * p / 3.0)
    if str(gate) in {"M", "MX"}:
        return float(p)
    raise ValueError(f"unsupported raw location kind/gate: kind={kind} gate={gate}")


def _raw_location_state_probabilities(raw_location: _RawLocation) -> tuple[float, ...]:
    prob = float(raw_location.error_probability)
    if str(raw_location.kind) == "binary":
        return (float(1.0 - prob), float(prob))
    return (
        float(1.0 - prob),
        float(prob / 3.0),
        float(prob / 3.0),
        float(prob / 3.0),
    )


def _binary_merge_probability(probabilities: list[float]) -> float:
    lam = 1.0
    for prob in probabilities:
        lam *= 1.0 - 2.0 * float(prob)
    return float((1.0 - lam) / 2.0)


def _quaternary_merge_total_probability(probabilities: list[float]) -> float:
    lam = 1.0
    for prob in probabilities:
        lam *= 1.0 - 4.0 * float(prob) / 3.0
    return float((3.0 / 4.0) * (1.0 - lam))


def _convolve_binary_state_probabilities(probabilities: list[tuple[float, ...]]) -> tuple[float, float]:
    out = np.asarray([1.0, 0.0], dtype=np.float64)
    for probs in probabilities:
        arr = np.asarray(probs, dtype=np.float64).reshape(-1)
        if int(arr.size) != 2:
            raise ValueError(f"binary state probability vector must have length 2, got {arr.size}")
        merged = np.zeros(2, dtype=np.float64)
        for a in range(2):
            for b in range(2):
                merged[a ^ b] += float(out[a]) * float(arr[b])
        out = merged
    return (float(out[0]), float(out[1]))


def _convolve_quaternary_state_probabilities(probabilities: list[tuple[float, ...]]) -> tuple[float, float, float, float]:
    out = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    for probs in probabilities:
        arr = np.asarray(probs, dtype=np.float64).reshape(-1)
        if int(arr.size) != 4:
            raise ValueError(f"quaternary state probability vector must have length 4, got {arr.size}")
        merged = np.zeros(4, dtype=np.float64)
        for a in range(4):
            for b in range(4):
                merged[a ^ b] += float(out[a]) * float(arr[b])
        out = merged
    return tuple(float(value) for value in out)


def _convolve_state_probabilities(probabilities: list[tuple[float, ...]]) -> tuple[float, ...]:
    if not probabilities:
        return (1.0,)
    state_count = int(len(probabilities[0]))
    if state_count <= 0 or state_count & (state_count - 1):
        raise ValueError(f"state probability vector length must be a positive power of two, got {state_count}")
    out = np.zeros(state_count, dtype=np.float64)
    out[0] = 1.0
    for probs in probabilities:
        arr = np.asarray(probs, dtype=np.float64).reshape(-1)
        if int(arr.size) != state_count:
            raise ValueError(f"state probability vector length mismatch: expected {state_count}, got {arr.size}")
        merged = np.zeros(state_count, dtype=np.float64)
        for a in range(state_count):
            for b in range(state_count):
                merged[a ^ b] += float(out[a]) * float(arr[b])
        out = merged
    return tuple(float(value) for value in out)


def _translated_quaternary_state_probabilities(
    probabilities: tuple[float, ...], *, error_probability: float, translation: int
) -> tuple[float, float, float, float]:
    arr = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if int(arr.size) != 4:
        raise ValueError(f"quaternary state probability vector must have length 4, got {arr.size}")
    t = int(translation) & 3
    q = float(error_probability)
    out = np.zeros(4, dtype=np.float64)
    for g in range(4):
        out[g] = (1.0 - q) * float(arr[g]) + q * float(arr[g ^ t])
    return tuple(float(value) for value in out)


def _state_response_signature(
    detector_map: dict[int, int], logical_map: dict[int, int], state: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    detector_rows = tuple(
        int(row)
        for row, mask in sorted(detector_map.items())
        if (bin(int(mask) & int(state)).count("1") & 1)
    )
    logical_rows = tuple(
        int(row)
        for row, mask in sorted(logical_map.items())
        if (bin(int(mask) & int(state)).count("1") & 1)
    )
    return detector_rows, logical_rows


def _effective_location_state_probabilities(location: _EffectiveLocation) -> tuple[float, ...]:
    return tuple(float(value) for value in location.state_probabilities)


def _state_response_bitset(
    detector_map: dict[int, int],
    logical_map: dict[int, int],
    state: int,
    *,
    num_detectors: int,
) -> int:
    out = 0
    for row, mask in detector_map.items():
        if (_popcount(int(mask) & int(state)) & 1) != 0:
            out |= 1 << int(row)
    for obs, mask in logical_map.items():
        if (_popcount(int(mask) & int(state)) & 1) != 0:
            out |= 1 << (int(num_detectors) + int(obs))
    return int(out)


def _canonical_basis(vectors: list[int]) -> tuple[int, ...]:
    basis_by_pivot: dict[int, int] = {}
    for value in sorted({int(vec) for vec in vectors if int(vec) != 0}, reverse=True):
        vec = int(value)
        while vec:
            pivot = int(vec.bit_length() - 1)
            pivot_vec = basis_by_pivot.get(int(pivot))
            if pivot_vec is None:
                break
            vec ^= int(pivot_vec)
        if vec == 0:
            continue
        pivot = int(vec.bit_length() - 1)
        for other_pivot in list(basis_by_pivot.keys()):
            if ((int(basis_by_pivot[other_pivot]) >> int(pivot)) & 1) != 0:
                basis_by_pivot[other_pivot] = int(basis_by_pivot[other_pivot]) ^ int(vec)
        basis_by_pivot[int(pivot)] = int(vec)
    return tuple(int(basis_by_pivot[pivot]) for pivot in sorted(basis_by_pivot.keys(), reverse=True))


def _coordinates_in_basis(vector: int, basis: tuple[int, ...]) -> int:
    out = 0
    residual = int(vector)
    for basis_index, basis_vector in enumerate(basis):
        pivot = int(basis_vector.bit_length() - 1)
        if ((int(residual) >> int(pivot)) & 1) == 0:
            continue
        residual ^= int(basis_vector)
        out |= 1 << int(basis_index)
    if residual != 0:
        raise ValueError("vector is not in the supplied basis span")
    return int(out)


def _nearest_quaternary_candidates(
    raw_locations: list[_RawLocation],
) -> dict[int, list[tuple[int, int]]]:
    quaternary_by_qubit: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
    for raw_index, raw_location in enumerate(raw_locations):
        if str(raw_location.kind) != "quaternary":
            continue
        for qubit in raw_location.qubits:
            quaternary_by_qubit[int(qubit)].append((int(raw_location.instruction_offset), int(raw_index)))
    for qubit in list(quaternary_by_qubit.keys()):
        quaternary_by_qubit[int(qubit)] = sorted(quaternary_by_qubit[int(qubit)])
    return {int(key): list(value) for key, value in quaternary_by_qubit.items()}


def _choose_nearest_quaternary_index(
    *,
    binary_location: _RawLocation,
    candidates: list[tuple[int, int]],
    raw_locations: list[_RawLocation],
) -> list[tuple[int, int]]:
    instruction_offset = int(binary_location.instruction_offset)
    min_dist = min(abs(int(instr) - int(instruction_offset)) for instr, _ in candidates)
    nearest = [(int(instr), int(q_index)) for instr, q_index in candidates if abs(int(instr) - int(instruction_offset)) == int(min_dist)]
    nearest.sort(
        key=lambda item: (
            abs(int(item[0]) - int(instruction_offset)),
            0 if int(item[0]) <= int(instruction_offset) else 1,
            abs(int(raw_locations[item[1]].tick_offset) - int(binary_location.tick_offset)),
            int(item[1]),
        )
    )
    return nearest


def _build_exact_grouped_location(
    *,
    member_indices: list[int],
    raw_locations: list[_RawLocation],
    detector_maps: list[dict[int, int]],
    logical_maps: list[dict[int, int]],
    num_detectors: int,
) -> tuple[_EffectiveLocation, dict[int, int], dict[int, int]]:
    representative = raw_locations[int(member_indices[0])]
    response_vectors: list[int] = []
    for member_index in member_indices:
        location = raw_locations[int(member_index)]
        state_count = int(len(_raw_location_state_probabilities(location)))
        for state in range(1, state_count):
            response_vectors.append(
                _state_response_bitset(
                    detector_maps[int(member_index)],
                    logical_maps[int(member_index)],
                    int(state),
                    num_detectors=int(num_detectors),
                )
            )
    basis = _canonical_basis(response_vectors)
    local_rank = int(len(basis))
    state_count = 1 << int(local_rank)
    state_probabilities = np.zeros(state_count, dtype=np.float64)
    member_probabilities = [_raw_location_state_probabilities(raw_locations[int(member_index)]) for member_index in member_indices]
    domain_sizes = [len(item) for item in member_probabilities]
    state_tuple = [0] * int(len(member_indices))
    while True:
        probability = 1.0
        response = 0
        for member_position, member_index in enumerate(member_indices):
            symbol = int(state_tuple[member_position])
            probability *= float(member_probabilities[member_position][symbol])
            if symbol != 0:
                response ^= _state_response_bitset(
                    detector_maps[int(member_index)],
                    logical_maps[int(member_index)],
                    int(symbol),
                    num_detectors=int(num_detectors),
                )
        state_probabilities[_coordinates_in_basis(int(response), basis)] += float(probability)
        carry_index = int(len(state_tuple) - 1)
        while carry_index >= 0:
            state_tuple[carry_index] += 1
            if int(state_tuple[carry_index]) < int(domain_sizes[carry_index]):
                break
            state_tuple[carry_index] = 0
            carry_index -= 1
        if carry_index < 0:
            break

    detector_map: dict[int, int] = {}
    for row in sorted({int(row) for member_index in member_indices for row in detector_maps[int(member_index)].keys()}):
        mask = 0
        for basis_index, basis_vector in enumerate(basis):
            if ((int(basis_vector) >> int(row)) & 1) != 0:
                mask |= 1 << int(basis_index)
        if mask != 0:
            detector_map[int(row)] = int(mask)

    logical_map: dict[int, int] = {}
    for obs in sorted({int(obs) for member_index in member_indices for obs in logical_maps[int(member_index)].keys()}):
        mask = 0
        obs_bit = int(num_detectors) + int(obs)
        for basis_index, basis_vector in enumerate(basis):
            if ((int(basis_vector) >> int(obs_bit)) & 1) != 0:
                mask |= 1 << int(basis_index)
        if mask != 0:
            logical_map[int(obs)] = int(mask)

    gate_counter = Counter(str(raw_locations[int(member_index)].gate) for member_index in member_indices)
    return (
        _EffectiveLocation(
            kind=_state_count_to_kind(int(state_count)),
            source_gate_counts={str(key): int(value) for key, value in sorted(gate_counter.items())},
            instruction_offset=int(representative.instruction_offset),
            tick_offset=int(representative.tick_offset),
            qubits=tuple(int(value) for value in representative.qubits),
            state_probabilities=tuple(float(value) for value in state_probabilities),
        ),
        detector_map,
        logical_map,
    )


def _build_exact_quaternary_raw_response_maps(
    *,
    explained_errors: list[object],
    raw_locations: list[_RawLocation],
    location_lookup: dict[tuple[object, ...], int],
    component: str,
) -> tuple[list[dict[int, int]], list[dict[int, int]]]:
    binary_detector_signatures: defaultdict[int, set[tuple[int, ...]]] = defaultdict(set)
    binary_logical_signatures: defaultdict[int, set[tuple[int, ...]]] = defaultdict(set)
    quaternary_detector_signatures: defaultdict[int, dict[int, set[tuple[int, ...]]]] = defaultdict(
        lambda: {1: set(), 2: set(), 3: set()}
    )
    quaternary_logical_signatures: defaultdict[int, dict[int, set[tuple[int, ...]]]] = defaultdict(
        lambda: {1: set(), 2: set(), 3: set()}
    )

    for explained_error in explained_errors:
        detector_indices, logical_indices = _detector_and_logical_indices(explained_error)
        detector_signature = tuple(sorted(int(value) for value in detector_indices))
        logical_signature = tuple(sorted(int(value) for value in logical_indices))
        for location in explained_error.circuit_error_locations:
            instruction_targets = location.instruction_targets
            name = str(instruction_targets.gate)
            stack_key = _stack_location_key(location)
            instruction_offset = int(location.stack_frames[-1].instruction_offset)
            target_qubits = [int(item.gate_target.value) for item in instruction_targets.targets_in_range]
            if name == "DEPOLARIZE1":
                qubit = int(target_qubits[0])
                mask = 0
                for target_with_coords in location.flipped_pauli_product:
                    gate_target = target_with_coords.gate_target
                    if int(gate_target.qubit_value) != int(qubit):
                        continue
                    if _projected_bit_present(gate_target, component):
                        mask = 1
                        break
                if mask == 0:
                    continue
                raw_index = int(location_lookup[("binary", name, stack_key, int(qubit))])
                binary_detector_signatures[raw_index].add(detector_signature)
                binary_logical_signatures[raw_index].add(logical_signature)
                continue
            if name == "DEPOLARIZE2":
                q0 = int(target_qubits[0])
                q1 = int(target_qubits[1])
                c0, c1 = _canonical_quaternary_pair(q0, q1)
                mask = 0
                for target_with_coords in location.flipped_pauli_product:
                    gate_target = target_with_coords.gate_target
                    if not _projected_bit_present(gate_target, component):
                        continue
                    qubit = int(gate_target.qubit_value)
                    if qubit == int(c0):
                        mask |= 1
                    elif qubit == int(c1):
                        mask |= 2
                if mask == 0:
                    continue
                raw_index = int(location_lookup[("quaternary", name, stack_key, int(c0), int(c1))])
                quaternary_detector_signatures[raw_index][int(mask)].add(detector_signature)
                quaternary_logical_signatures[raw_index][int(mask)].add(logical_signature)
                continue
            if name in {"M", "MX"}:
                qubit = int(target_qubits[0])
                raw_index = int(location_lookup[("binary", name, stack_key, int(qubit))])
                binary_detector_signatures[raw_index].add(detector_signature)
                binary_logical_signatures[raw_index].add(logical_signature)

    detector_maps: list[dict[int, int]] = [defaultdict(int) for _ in raw_locations]
    logical_maps: list[dict[int, int]] = [defaultdict(int) for _ in raw_locations]
    for raw_index, raw_location in enumerate(raw_locations):
        if str(raw_location.kind) == "binary":
            detector_options = set(binary_detector_signatures.get(int(raw_index), set()))
            logical_options = set(binary_logical_signatures.get(int(raw_index), set()))
            if len(detector_options) > 1 or len(logical_options) > 1:
                raise ValueError(
                    f"binary raw location has multiple projected signatures: index={raw_index} location={raw_location}"
                )
            for row in next(iter(detector_options), tuple()):
                detector_maps[raw_index][int(row)] = 1
            for obs in next(iter(logical_options), tuple()):
                logical_maps[raw_index][int(obs)] = 1
            continue

        detector_sets = {
            int(mask): set(next(iter(quaternary_detector_signatures[int(raw_index)][int(mask)]), tuple()))
            for mask in (1, 2, 3)
        }
        logical_sets = {
            int(mask): set(next(iter(quaternary_logical_signatures[int(raw_index)][int(mask)]), tuple()))
            for mask in (1, 2, 3)
        }
        for mask in (1, 2, 3):
            if len(quaternary_detector_signatures[int(raw_index)][int(mask)]) > 1:
                raise ValueError(
                    f"quaternary raw location has multiple detector signatures for projected state {mask}: "
                    f"index={raw_index} location={raw_location}"
                )
            if len(quaternary_logical_signatures[int(raw_index)][int(mask)]) > 1:
                raise ValueError(
                    f"quaternary raw location has multiple logical signatures for projected state {mask}: "
                    f"index={raw_index} location={raw_location}"
                )
        if detector_sets[1] ^ detector_sets[2] != detector_sets[3]:
            raise ValueError(
                f"quaternary detector signatures do not close linearly on projected states: "
                f"index={raw_index} location={raw_location}"
            )
        if logical_sets[1] ^ logical_sets[2] != logical_sets[3]:
            raise ValueError(
                f"quaternary logical signatures do not close linearly on projected states: "
                f"index={raw_index} location={raw_location}"
            )

        for row in sorted(detector_sets[1] | detector_sets[2] | detector_sets[3]):
            mask = (1 if int(row) in detector_sets[1] else 0) | (2 if int(row) in detector_sets[2] else 0)
            if mask != 0:
                detector_maps[raw_index][int(row)] = int(mask)
        for obs in sorted(logical_sets[1] | logical_sets[2] | logical_sets[3]):
            mask = (1 if int(obs) in logical_sets[1] else 0) | (2 if int(obs) in logical_sets[2] else 0)
            if mask != 0:
                logical_maps[raw_index][int(obs)] = int(mask)

    return [dict(item) for item in detector_maps], [dict(item) for item in logical_maps]


def _choose_gate_centered_quaternary_match(
    *,
    binary_index: int,
    raw_locations: list[_RawLocation],
    quaternary_by_qubit: dict[int, list[tuple[int, int]]],
    detector_maps: list[dict[int, int]],
    logical_maps: list[dict[int, int]],
) -> tuple[int, int, str] | None:
    binary_location = raw_locations[int(binary_index)]
    qubit = int(binary_location.qubits[0])
    candidates = list(quaternary_by_qubit.get(int(qubit), []))
    if not candidates:
        return None
    ordered = sorted(
        ((int(instruction_offset), int(q_index)) for instruction_offset, q_index in candidates),
        key=lambda item: (
            abs(int(item[0]) - int(binary_location.instruction_offset)),
            0 if int(item[0]) >= int(binary_location.instruction_offset) else 1,
            abs(int(raw_locations[item[1]].tick_offset) - int(binary_location.tick_offset)),
            int(item[1]),
        ),
    )
    binary_signature = _state_response_signature(detector_maps[int(binary_index)], logical_maps[int(binary_index)], 1)
    for instruction_offset, q_index in ordered:
        direction = "next" if int(instruction_offset) >= int(binary_location.instruction_offset) else "prev"
        for translation in (1, 2, 3):
            if binary_signature == _state_response_signature(detector_maps[int(q_index)], logical_maps[int(q_index)], int(translation)):
                return int(q_index), int(translation), str(direction)
    return None


def _build_effective_locations(
    raw_locations: list[_RawLocation],
    detector_maps: list[dict[int, int]],
    logical_maps: list[dict[int, int]],
    *,
    grouping_mode: str,
    num_detectors: int,
) -> tuple[list[_EffectiveLocation], list[dict[int, int]], list[dict[int, int]], dict[str, object]]:
    mode = str(grouping_mode).strip().lower()
    if mode == "none":
        effective_locations = [
            _EffectiveLocation(
                kind=_state_count_to_kind(len(_raw_location_state_probabilities(raw_location))),
                source_gate_counts={str(raw_location.gate): 1},
                instruction_offset=int(raw_location.instruction_offset),
                tick_offset=int(raw_location.tick_offset),
                qubits=tuple(int(value) for value in raw_location.qubits),
                state_probabilities=_raw_location_state_probabilities(raw_location),
            )
            for raw_location in raw_locations
        ]
        return (
            effective_locations,
            [dict(item) for item in detector_maps],
            [dict(item) for item in logical_maps],
            {
                "grouping_mode": "none",
                "grouped_attached_binary_count": 0,
                "grouped_unmatched_binary_count": int(sum(1 for item in raw_locations if str(item.kind) == "binary")),
                "grouped_group_size_hist": {1: int(len(raw_locations))},
            },
        )
    if mode not in {
        "merge_one_qubit_into_quaternary",
        "merge_all_one_qubit_exact",
        "merge_all_one_qubit_into_quaternary_exact",
    }:
        raise ValueError(
            "grouping_mode must be one of: none, merge_one_qubit_into_quaternary, "
            "merge_all_one_qubit_exact, merge_all_one_qubit_into_quaternary_exact"
        )

    quaternary_by_qubit = _nearest_quaternary_candidates(raw_locations)

    if mode == "merge_all_one_qubit_into_quaternary_exact":
        attached_to_quaternary: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
        attached_binary_indices: set[int] = set()
        zero_gate_counts: Counter[str] = Counter()
        for raw_index, raw_location in enumerate(raw_locations):
            if str(raw_location.kind) != "binary":
                continue
            if not detector_maps[raw_index] and not logical_maps[raw_index]:
                zero_gate_counts[str(raw_location.gate)] += 1
                continue
            match = _choose_gate_centered_quaternary_match(
                binary_index=int(raw_index),
                raw_locations=raw_locations,
                quaternary_by_qubit=quaternary_by_qubit,
                detector_maps=detector_maps,
                logical_maps=logical_maps,
            )
            if match is None:
                raise ValueError(f"nonzero binary raw location could not be attached exactly to an adjacent quaternary anchor: {raw_location}")
            q_index, translation, _ = match
            attached_binary_indices.add(int(raw_index))
            attached_to_quaternary[int(q_index)].append((int(raw_index), int(translation)))

        effective_locations: list[_EffectiveLocation] = []
        effective_detector_maps: list[dict[int, int]] = []
        effective_logical_maps: list[dict[int, int]] = []
        grouped_group_size_hist: Counter[int] = Counter()
        for raw_index, raw_location in enumerate(raw_locations):
            if str(raw_location.kind) != "quaternary":
                continue
            state_probabilities = _raw_location_state_probabilities(raw_location)
            gate_counter = Counter({str(raw_location.gate): 1})
            attachments = sorted(attached_to_quaternary.get(int(raw_index), []), key=lambda item: item[0])
            for binary_index, translation in attachments:
                state_probabilities = _translated_quaternary_state_probabilities(
                    tuple(float(value) for value in state_probabilities),
                    error_probability=float(raw_locations[binary_index].error_probability),
                    translation=int(translation),
                )
                gate_counter[str(raw_locations[binary_index].gate)] += 1
            effective_locations.append(
                _EffectiveLocation(
                    kind="quaternary",
                    source_gate_counts={str(key): int(value) for key, value in sorted(gate_counter.items())},
                    instruction_offset=int(raw_location.instruction_offset),
                    tick_offset=int(raw_location.tick_offset),
                    qubits=tuple(int(value) for value in raw_location.qubits),
                    state_probabilities=tuple(float(value) for value in state_probabilities),
                )
            )
            effective_detector_maps.append(dict(detector_maps[raw_index]))
            effective_logical_maps.append(dict(logical_maps[raw_index]))
            grouped_group_size_hist[1 + len(attachments)] += 1

        grouped_kind_counts = Counter(str(item.kind) for item in effective_locations)
        return (
            effective_locations,
            effective_detector_maps,
            effective_logical_maps,
            {
                "grouping_mode": "merge_all_one_qubit_into_quaternary_exact",
                "grouped_attached_binary_count": int(len(attached_binary_indices)),
                "grouped_unmatched_binary_count": 0,
                "grouped_group_size_hist": {int(key): int(value) for key, value in sorted(grouped_group_size_hist.items())},
                "grouped_raw_location_count": int(len(effective_locations)),
                "grouped_raw_kind_counts": {
                    str(key): int(value)
                    for key, value in sorted(grouped_kind_counts.items(), key=lambda item: _kind_sort_key(str(item[0])))
                },
                "raw_zero_count_override": int(sum(int(value) for value in zero_gate_counts.values())),
                "zero_gate_counts_override": {str(key): int(value) for key, value in sorted(zero_gate_counts.items())},
            },
        )

    attached_to_quaternary: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
    attached_binary_indices: set[int] = set()
    for raw_index, raw_location in enumerate(raw_locations):
        if str(raw_location.kind) != "binary":
            continue
        if not detector_maps[raw_index] and not logical_maps[raw_index]:
            continue
        qubit = int(raw_location.qubits[0])
        candidates = list(quaternary_by_qubit.get(int(qubit), []))
        if not candidates:
            continue
        nearest = _choose_nearest_quaternary_index(binary_location=raw_location, candidates=candidates, raw_locations=raw_locations)
        if mode == "merge_all_one_qubit_exact":
            _, q_index = nearest[0]
            attached_binary_indices.add(int(raw_index))
            attached_to_quaternary[int(q_index)].append((int(raw_index), 0))
            continue
        binary_signature = _state_response_signature(detector_maps[raw_index], logical_maps[raw_index], 1)
        exact_matches: list[tuple[tuple[int, ...], int, int, int]] = []
        for candidate_instruction, q_index in nearest:
            translations = [
                int(state)
                for state in (1, 2, 3)
                if binary_signature == _state_response_signature(detector_maps[q_index], logical_maps[q_index], state)
            ]
            if not translations:
                continue
            sort_key = (
                abs(int(candidate_instruction) - int(raw_location.instruction_offset)),
                0 if int(candidate_instruction) <= int(raw_location.instruction_offset) else 1,
                abs(int(raw_locations[q_index].tick_offset) - int(raw_location.tick_offset)),
                int(q_index),
                int(translations[0]),
            )
            exact_matches.append((sort_key, int(q_index), int(translations[0]), int(len(translations))))
        if exact_matches:
            exact_matches.sort(key=lambda item: item[0])
            _, q_index, translation, _ = exact_matches[0]
            attached_binary_indices.add(int(raw_index))
            attached_to_quaternary[int(q_index)].append((int(raw_index), int(translation)))

    if mode == "merge_all_one_qubit_exact":
        effective_locations = []
        effective_detector_maps = []
        effective_logical_maps = []
        grouped_group_size_hist: Counter[int] = Counter()
        for raw_index, raw_location in enumerate(raw_locations):
            if str(raw_location.kind) != "quaternary":
                continue
            member_indices = [int(raw_index)] + [int(binary_index) for binary_index, _ in sorted(attached_to_quaternary.get(int(raw_index), []), key=lambda item: item[0])]
            effective_location, detector_map, logical_map = _build_exact_grouped_location(
                member_indices=member_indices,
                raw_locations=raw_locations,
                detector_maps=detector_maps,
                logical_maps=logical_maps,
                num_detectors=int(num_detectors),
            )
            effective_locations.append(effective_location)
            effective_detector_maps.append(detector_map)
            effective_logical_maps.append(logical_map)
            grouped_group_size_hist[len(member_indices)] += 1
        for raw_index, raw_location in enumerate(raw_locations):
            if str(raw_location.kind) != "binary":
                continue
            if int(raw_index) in attached_binary_indices:
                continue
            effective_locations.append(
                _EffectiveLocation(
                    kind=_state_count_to_kind(len(_raw_location_state_probabilities(raw_location))),
                    source_gate_counts={str(raw_location.gate): 1},
                    instruction_offset=int(raw_location.instruction_offset),
                    tick_offset=int(raw_location.tick_offset),
                    qubits=tuple(int(value) for value in raw_location.qubits),
                    state_probabilities=_raw_location_state_probabilities(raw_location),
                )
            )
            effective_detector_maps.append(dict(detector_maps[raw_index]))
            effective_logical_maps.append(dict(logical_maps[raw_index]))
            grouped_group_size_hist[1] += 1
        grouped_kind_counts = Counter(str(item.kind) for item in effective_locations)
        return (
            effective_locations,
            effective_detector_maps,
            effective_logical_maps,
            {
                "grouping_mode": "merge_all_one_qubit_exact",
                "grouped_attached_binary_count": int(len(attached_binary_indices)),
                "grouped_unmatched_binary_count": int(
                    sum(1 for raw_index, item in enumerate(raw_locations) if str(item.kind) == "binary" and raw_index not in attached_binary_indices and (detector_maps[raw_index] or logical_maps[raw_index]))
                ),
                "grouped_group_size_hist": {int(key): int(value) for key, value in sorted(grouped_group_size_hist.items())},
                "grouped_raw_location_count": int(len(effective_locations)),
                "grouped_raw_kind_counts": {
                    str(key): int(value)
                    for key, value in sorted(grouped_kind_counts.items(), key=lambda item: _kind_sort_key(str(item[0])))
                },
            },
        )

    effective_locations: list[_EffectiveLocation] = []
    effective_detector_maps: list[dict[int, int]] = []
    effective_logical_maps: list[dict[int, int]] = []
    grouped_group_size_hist: Counter[int] = Counter()
    for raw_index, raw_location in enumerate(raw_locations):
        if str(raw_location.kind) == "quaternary":
            state_probabilities = _raw_location_state_probabilities(raw_location)
            gate_counter = Counter({str(raw_location.gate): 1})
            attachments = sorted(attached_to_quaternary.get(int(raw_index), []), key=lambda item: item[0])
            for binary_index, translation in attachments:
                state_probabilities = _translated_quaternary_state_probabilities(
                    tuple(float(value) for value in state_probabilities),
                    error_probability=float(raw_locations[binary_index].error_probability),
                    translation=int(translation),
                )
                gate_counter[str(raw_locations[binary_index].gate)] += 1
            effective_locations.append(
                _EffectiveLocation(
                    kind="quaternary",
                    source_gate_counts={str(key): int(value) for key, value in sorted(gate_counter.items())},
                    instruction_offset=int(raw_location.instruction_offset),
                    tick_offset=int(raw_location.tick_offset),
                    qubits=tuple(int(value) for value in raw_location.qubits),
                    state_probabilities=tuple(float(value) for value in state_probabilities),
                )
            )
            effective_detector_maps.append(dict(detector_maps[raw_index]))
            effective_logical_maps.append(dict(logical_maps[raw_index]))
            grouped_group_size_hist[1 + len(attachments)] += 1
            continue
        if int(raw_index) in attached_binary_indices:
            continue
        effective_locations.append(
            _EffectiveLocation(
                kind=_state_count_to_kind(len(_raw_location_state_probabilities(raw_location))),
                source_gate_counts={str(raw_location.gate): 1},
                instruction_offset=int(raw_location.instruction_offset),
                tick_offset=int(raw_location.tick_offset),
                qubits=tuple(int(value) for value in raw_location.qubits),
                state_probabilities=_raw_location_state_probabilities(raw_location),
            )
        )
        effective_detector_maps.append(dict(detector_maps[raw_index]))
        effective_logical_maps.append(dict(logical_maps[raw_index]))
        grouped_group_size_hist[1] += 1

    grouped_kind_counts = Counter(str(item.kind) for item in effective_locations)
    return (
        effective_locations,
        effective_detector_maps,
        effective_logical_maps,
        {
            "grouping_mode": "merge_one_qubit_into_quaternary",
            "grouped_attached_binary_count": int(len(attached_binary_indices)),
            "grouped_unmatched_binary_count": int(
                sum(1 for raw_index, item in enumerate(raw_locations) if str(item.kind) == "binary" and raw_index not in attached_binary_indices)
            ),
            "grouped_group_size_hist": {int(key): int(value) for key, value in sorted(grouped_group_size_hist.items())},
            "grouped_raw_location_count": int(len(effective_locations)),
            "grouped_raw_kind_counts": {
                str(key): int(value)
                for key, value in sorted(grouped_kind_counts.items(), key=lambda item: _kind_sort_key(str(item[0])))
            },
        },
    )


@lru_cache(maxsize=16)
def _build_cached(root_text: str, backend: str, projected_component: str, error_rate: float, grouping_mode: str) -> ProjectedLocationProblem:
    import stim  # type: ignore

    component = str(projected_component).upper()
    if component not in PROJECTED_COMPONENT_TO_PUBLIC_SECTOR:
        raise ValueError(f"projected_component must be one of {sorted(PROJECTED_COMPONENT_TO_PUBLIC_SECTOR)}, got {projected_component}")
    public_sector = str(PROJECTED_COMPONENT_TO_PUBLIC_SECTOR[component])
    spec = resolve_backend_circuit(
        backend=str(backend),
        sector=public_sector,
        error_rate=float(error_rate),
        qtanner_root=root_text,
    )
    circuit = stim.Circuit.from_file(str(spec.stim_path))

    dem = circuit.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)
    explained = circuit.explain_detector_error_model_errors(dem_filter=dem, reduce_to_one_representative_error=False)
    raw_locations: list[_RawLocation] = []
    location_lookup: dict[tuple[object, ...], int] = {}
    for explained_error in explained:
        for location in explained_error.circuit_error_locations:
            instruction_targets = location.instruction_targets
            name = str(instruction_targets.gate)
            stack_key = _stack_location_key(location)
            instruction_offset = int(location.stack_frames[-1].instruction_offset)
            target_qubits = [int(item.gate_target.value) for item in instruction_targets.targets_in_range]
            if name == "DEPOLARIZE1":
                qubit = int(target_qubits[0])
                key = ("binary", name, stack_key, int(qubit))
                if key in location_lookup:
                    continue
                location_lookup[key] = len(raw_locations)
                raw_locations.append(
                    _RawLocation(
                        kind="binary",
                        gate=str(name),
                        instruction_offset=int(instruction_offset),
                        tick_offset=int(instruction_offset),
                        qubits=(int(qubit),),
                        error_probability=_raw_error_probability("binary", str(name), float(error_rate)),
                    )
                )
                continue
            if name == "DEPOLARIZE2":
                c0, c1 = _canonical_quaternary_pair(int(target_qubits[0]), int(target_qubits[1]))
                key = ("quaternary", name, stack_key, int(c0), int(c1))
                if key in location_lookup:
                    continue
                location_lookup[key] = len(raw_locations)
                raw_locations.append(
                    _RawLocation(
                        kind="quaternary",
                        gate=str(name),
                        instruction_offset=int(instruction_offset),
                        tick_offset=int(instruction_offset),
                        qubits=(int(c0), int(c1)),
                        error_probability=_raw_error_probability("quaternary", str(name), float(error_rate)),
                    )
                )
                continue
            if name in {"M", "MX"}:
                qubit = int(target_qubits[0])
                key = ("binary", name, stack_key, int(qubit))
                if key in location_lookup:
                    continue
                location_lookup[key] = len(raw_locations)
                raw_locations.append(
                    _RawLocation(
                        kind="binary",
                        gate=str(name),
                        instruction_offset=int(instruction_offset),
                        tick_offset=int(instruction_offset),
                        qubits=(int(qubit),),
                        error_probability=_raw_error_probability("binary", str(name), float(error_rate)),
                    )
                )
    if str(grouping_mode).strip().lower() == "merge_all_one_qubit_into_quaternary_exact":
        detector_maps, logical_maps = _build_exact_quaternary_raw_response_maps(
            explained_errors=list(explained),
            raw_locations=raw_locations,
            location_lookup=location_lookup,
            component=str(component),
        )
    else:
        detector_maps = [defaultdict(int) for _ in raw_locations]
        logical_maps = [defaultdict(int) for _ in raw_locations]
        for explained_error in explained:
            detector_indices, logical_indices = _detector_and_logical_indices(explained_error)
            for location in explained_error.circuit_error_locations:
                instruction_targets = location.instruction_targets
                name = str(instruction_targets.gate)
                stack_key = _stack_location_key(location)
                target_qubits = [int(item.gate_target.value) for item in instruction_targets.targets_in_range]
                mask = 0
                location_index: int | None = None
                if name == "DEPOLARIZE1":
                    qubit = int(target_qubits[0])
                    for target_with_coords in location.flipped_pauli_product:
                        gate_target = target_with_coords.gate_target
                        if int(gate_target.qubit_value) != qubit:
                            continue
                        if _projected_bit_present(gate_target, component):
                            mask = 1
                            break
                    if mask == 0:
                        continue
                    location_index = location_lookup[("binary", name, stack_key, int(qubit))]
                elif name == "DEPOLARIZE2":
                    q0 = int(target_qubits[0])
                    q1 = int(target_qubits[1])
                    c0, c1 = _canonical_quaternary_pair(q0, q1)
                    for target_with_coords in location.flipped_pauli_product:
                        gate_target = target_with_coords.gate_target
                        if not _projected_bit_present(gate_target, component):
                            continue
                        qubit = int(gate_target.qubit_value)
                        if qubit == c0:
                            mask |= 1
                        elif qubit == c1:
                            mask |= 2
                    if mask == 0:
                        continue
                    location_index = location_lookup[("quaternary", name, stack_key, int(c0), int(c1))]
                elif name in {"M", "MX"}:
                    qubit = int(target_qubits[0])
                    mask = 1
                    location_index = location_lookup[("binary", name, stack_key, int(qubit))]
                if location_index is None:
                    continue
                for detector_index in detector_indices:
                    detector_maps[location_index][int(detector_index)] = int(detector_maps[location_index][int(detector_index)]) | int(mask)
                for logical_index in logical_indices:
                    logical_maps[location_index][int(logical_index)] = int(logical_maps[location_index][int(logical_index)]) | int(mask)

    effective_locations, effective_detector_maps, effective_logical_maps, grouping_stats = _build_effective_locations(
        raw_locations,
        detector_maps,
        logical_maps,
        grouping_mode=str(grouping_mode),
        num_detectors=int(dem.num_detectors),
    )

    duplicate_groups_detector_only: defaultdict[tuple[object, ...], list[int]] = defaultdict(list)
    duplicate_groups_full: defaultdict[tuple[object, ...], list[int]] = defaultdict(list)
    merged_groups: defaultdict[tuple[object, ...], list[int]] = defaultdict(list)
    zero_indices: list[int] = []
    for effective_index, effective_location in enumerate(effective_locations):
        detector_signature = tuple(sorted((int(row), int(mask)) for row, mask in effective_detector_maps[effective_index].items()))
        logical_signature = tuple(sorted((int(obs), int(mask)) for obs, mask in effective_logical_maps[effective_index].items()))
        if not detector_signature and not logical_signature:
            zero_indices.append(int(effective_index))
        duplicate_groups_detector_only[(str(effective_location.kind), detector_signature)].append(int(effective_index))
        full_signature = (str(effective_location.kind), detector_signature, logical_signature)
        duplicate_groups_full[full_signature].append(int(effective_index))
        if detector_signature or logical_signature:
            merged_groups[full_signature].append(int(effective_index))

    duplicate_source_combo_hist: Counter[str] = Counter()
    merged_columns: list[ProjectedColumnMetadata] = []
    detector_edge_row: list[int] = []
    detector_edge_col: list[int] = []
    detector_edge_mask: list[int] = []
    logical_edge_observable: list[int] = []
    logical_edge_col: list[int] = []
    logical_edge_mask: list[int] = []
    merged_multiplicity_hist: Counter[int] = Counter()
    for column_index, (signature, group_indices) in enumerate(
        sorted(merged_groups.items(), key=lambda item: (_kind_sort_key(str(item[0][0])), item[0][1], item[0][2]))
    ):
        kind = str(signature[0])
        detector_signature = tuple(signature[1])
        logical_signature = tuple(signature[2])
        merged_multiplicity_hist[len(group_indices)] += 1
        gate_counter: Counter[str] = Counter()
        for idx in group_indices:
            gate_counter.update({str(key): int(value) for key, value in effective_locations[idx].source_gate_counts.items()})
        if len(group_indices) > 1:
            duplicate_source_combo_hist[json.dumps(dict(sorted(gate_counter.items())), sort_keys=True)] += 1
        representative = effective_locations[int(group_indices[0])]
        state_probability_list = [_effective_location_state_probabilities(effective_locations[idx]) for idx in group_indices]
        state_probabilities = _convolve_state_probabilities(state_probability_list)
        error_probability = float(1.0 - float(state_probabilities[0]))
        for row_index, mask in detector_signature:
            detector_edge_row.append(int(row_index))
            detector_edge_col.append(int(column_index))
            detector_edge_mask.append(int(mask))
        for logical_index, mask in logical_signature:
            logical_edge_observable.append(int(logical_index))
            logical_edge_col.append(int(column_index))
            logical_edge_mask.append(int(mask))
        merged_columns.append(
            ProjectedColumnMetadata(
                index=int(column_index),
                kind=str(kind),
                merged_multiplicity=int(len(group_indices)),
                source_gate_counts={str(key): int(value) for key, value in sorted(gate_counter.items())},
                representative_instruction_offset=int(representative.instruction_offset),
                representative_tick_offset=int(representative.tick_offset),
                representative_qubits=tuple(int(value) for value in representative.qubits),
                detector_weight=int(len(detector_signature)),
                logical_weight=int(len(logical_signature)),
                error_probability=float(error_probability),
                state_probabilities=tuple(float(value) for value in state_probabilities),
            )
        )

    grouping_mode_value = str(grouping_stats.get("grouping_mode", "none"))
    zero_gate_counts_default = {
        str(key): int(value)
        for key, value in sorted(
            Counter(
                gate
                for idx in zero_indices
                for gate, count in effective_locations[idx].source_gate_counts.items()
                for _ in range(int(count))
            ).items()
        )
    }
    zero_gate_counts_value = dict(grouping_stats.get("zero_gate_counts_override", zero_gate_counts_default))
    raw_zero_count_value = int(grouping_stats.get("raw_zero_count_override", len(zero_indices)))
    if grouping_mode_value == "merge_one_qubit_into_quaternary":
        grouping_note = (
            "Raw projected one-qubit locations are first absorbed into the nearest raw projected DEPOLARIZE2 location when their detector+logical response "
            "is exactly a quaternary group translation, and the resulting local prior is convolved on the `{II, XI, IX, XX}` alphabet."
        )
    elif grouping_mode_value == "merge_all_one_qubit_exact":
        grouping_note = (
            "Every nonzero projected one-qubit raw location is absorbed into a disjoint nearest raw DEPOLARIZE2 anchor group, then each local group is quotiented "
            "exactly by its detector-plus-logical response subgroup and represented on its minimal independent binary-coordinate alphabet."
        )
    elif grouping_mode_value == "merge_all_one_qubit_into_quaternary_exact":
        grouping_note = (
            "Every nonzero projected one-qubit raw location is absorbed exactly into an adjacent raw DEPOLARIZE2 anchor by forward-first Clifford propagation to the "
            "nearest gate cut. Raw DEPOLARIZE2 columns are built from the exact projected `{XI, IX, XX}` state signatures, so the grouped family stays purely quaternary."
        )
    else:
        grouping_note = (
            "Binary columns come from projected DEPOLARIZE1 targets and noisy measurement targets; quaternary columns come from projected DEPOLARIZE2 CNOT-pair locations."
        )

    metadata = ProjectedLocationMetadata(
        backend=str(spec.backend),
        projected_component=str(component),
        public_sector=str(public_sector),
        error_rate=float(error_rate),
        noisy_rounds=int(spec.noisy_rounds),
        total_rounds=int(spec.noisy_rounds + spec.perfect_rounds),
        stim_path=str(spec.stim_path),
        num_detectors=int(dem.num_detectors),
        num_observables=int(dem.num_observables),
        raw_location_count=int(len(raw_locations)),
        raw_kind_counts={str(key): int(value) for key, value in sorted(Counter(item.kind for item in raw_locations).items())},
        raw_gate_counts={str(key): int(value) for key, value in sorted(Counter(item.gate for item in raw_locations).items())},
        raw_zero_count=int(raw_zero_count_value),
        zero_gate_counts={str(key): int(value) for key, value in sorted(zero_gate_counts_value.items())},
        raw_duplicate_detector_only=_duplicate_summary([group for group in duplicate_groups_detector_only.values() if len(group) > 1]),
        raw_duplicate_detector_plus_logical=_duplicate_summary([group for group in duplicate_groups_full.values() if len(group) > 1]),
        merged_kind_counts={
            str(key): int(value)
            for key, value in sorted(Counter(item.kind for item in merged_columns).items(), key=lambda item: _kind_sort_key(str(item[0])))
        },
        merged_multiplicity_hist={int(key): int(value) for key, value in sorted(merged_multiplicity_hist.items())},
        duplicate_source_combo_hist={str(key): int(value) for key, value in sorted(duplicate_source_combo_hist.items())},
        notes=(
            "This is a non-default detector-side matrix family derived from raw physical noisy locations, not from the accepted binary graphlike DEM columns.",
            f"The projected `{component}` component is extracted from the public `memory_{public_sector}` split-sector Stim circuit because that file carries the relevant side projection.",
            grouping_note,
            "Raw all-zero columns are dropped before duplicate merging because they never affect the side detector record or logical operators.",
        ),
        grouping_mode=str(grouping_mode_value),
        grouped_raw_location_count=int(grouping_stats.get("grouped_raw_location_count", len(effective_locations))),
        grouped_raw_kind_counts={
            str(key): int(value)
            for key, value in sorted(
                dict(grouping_stats.get("grouped_raw_kind_counts", {str(key): int(value) for key, value in Counter(item.kind for item in effective_locations).items()})).items(),
                key=lambda item: _kind_sort_key(str(item[0])),
            )
        },
        grouped_attached_binary_count=int(grouping_stats.get("grouped_attached_binary_count", 0)),
        grouped_unmatched_binary_count=int(grouping_stats.get("grouped_unmatched_binary_count", 0)),
        grouped_group_size_hist={int(key): int(value) for key, value in sorted(dict(grouping_stats.get("grouped_group_size_hist", {1: len(effective_locations)})).items())},
    )
    return ProjectedLocationProblem(
        metadata=metadata,
        columns=tuple(merged_columns),
        detector_edge_row=np.asarray(detector_edge_row, dtype=np.int32),
        detector_edge_col=np.asarray(detector_edge_col, dtype=np.int32),
        detector_edge_mask=np.asarray(detector_edge_mask, dtype=np.uint8),
        logical_edge_observable=np.asarray(logical_edge_observable, dtype=np.int32),
        logical_edge_col=np.asarray(logical_edge_col, dtype=np.int32),
        logical_edge_mask=np.asarray(logical_edge_mask, dtype=np.uint8),
    )


def build_projected_location_problem(
    *,
    projected_component: str = "X",
    backend: str = "bravyi_depth7",
    error_rate: float = 0.004,
    grouping_mode: str = "none",
    qtanner_root: str | Path | None = None,
) -> ProjectedLocationProblem:
    return _build_cached(
        None if qtanner_root is None else str(Path(qtanner_root)),
        str(backend),
        str(projected_component).upper(),
        float(error_rate),
        str(grouping_mode).strip().lower(),
    )


def build_split_binary_control_problem(
    problem: ProjectedLocationProblem,
    *,
    variant_label: str = "split_state_binary_control",
) -> ProjectedLocationProblem:
    detector = problem.detector_mask_matrix().tocsc()
    logical = problem.logical_mask_matrix().tocsc()

    raw_groups: defaultdict[tuple[tuple[int, ...], tuple[int, ...]], list[dict[str, object]]] = defaultdict(list)
    raw_gate_counts: Counter[str] = Counter()
    raw_duplicate_detector_only: defaultdict[tuple[tuple[int, ...], ...], list[int]] = defaultdict(list)
    raw_column_count = 0

    for meta in tuple(problem.columns):
        source_column = int(meta.index)
        det_start = int(detector.indptr[source_column])
        det_stop = int(detector.indptr[source_column + 1])
        det_rows = detector.indices[det_start:det_stop].astype(np.int32, copy=False)
        det_masks = detector.data[det_start:det_stop].astype(np.uint8, copy=False)
        log_start = int(logical.indptr[source_column])
        log_stop = int(logical.indptr[source_column + 1])
        log_rows = logical.indices[log_start:log_stop].astype(np.int32, copy=False)
        log_masks = logical.data[log_start:log_stop].astype(np.uint8, copy=False)

        for state in range(1, int(meta.state_count)):
            detector_signature = tuple(
                int(row)
                for row, mask in zip(det_rows.tolist(), det_masks.tolist(), strict=True)
                if (_popcount(int(mask) & int(state)) & 1) != 0
            )
            logical_signature = tuple(
                int(obs)
                for obs, mask in zip(log_rows.tolist(), log_masks.tolist(), strict=True)
                if (_popcount(int(mask) & int(state)) & 1) != 0
            )
            if not detector_signature and not logical_signature:
                continue
            raw_index = int(raw_column_count)
            raw_column_count += 1
            probability = float(meta.state_probabilities[int(state)])
            entry = {
                "raw_index": int(raw_index),
                "source_column": int(source_column),
                "source_state": int(state),
                "probability": float(probability),
                "source_gate_counts": {str(key): int(value) for key, value in meta.source_gate_counts.items()},
                "representative_instruction_offset": int(meta.representative_instruction_offset),
                "representative_tick_offset": int(meta.representative_tick_offset),
                "representative_qubits": tuple(int(value) for value in meta.representative_qubits),
            }
            raw_groups[(detector_signature, logical_signature)].append(entry)
            raw_duplicate_detector_only[tuple(tuple((int(row),)) for row in detector_signature)].append(int(raw_index))
            raw_gate_counts.update({str(key): int(value) for key, value in meta.source_gate_counts.items()})

    merged_columns: list[ProjectedColumnMetadata] = []
    detector_edge_row: list[int] = []
    detector_edge_col: list[int] = []
    detector_edge_mask: list[int] = []
    logical_edge_observable: list[int] = []
    logical_edge_col: list[int] = []
    logical_edge_mask: list[int] = []
    merged_multiplicity_hist: Counter[int] = Counter()
    duplicate_source_combo_hist: Counter[str] = Counter()
    raw_duplicate_full_groups: list[list[int]] = []

    for column_index, ((detector_signature, logical_signature), group_entries) in enumerate(
        sorted(raw_groups.items(), key=lambda item: (item[0][0], item[0][1]))
    ):
        merged_multiplicity_hist[len(group_entries)] += 1
        raw_duplicate_full_groups.append([int(entry["raw_index"]) for entry in group_entries])
        gate_counter: Counter[str] = Counter()
        probs: list[float] = []
        for entry in group_entries:
            gate_counter.update({str(key): int(value) for key, value in dict(entry["source_gate_counts"]).items()})
            probs.append(float(entry["probability"]))
        if len(group_entries) > 1:
            duplicate_source_combo_hist[json.dumps(dict(sorted(gate_counter.items())), sort_keys=True)] += 1
        merged_probability = float(_binary_merge_probability(probs))
        representative = min(
            group_entries,
            key=lambda entry: (
                int(entry["representative_instruction_offset"]),
                int(entry["representative_tick_offset"]),
                int(entry["source_column"]),
                int(entry["source_state"]),
            ),
        )
        for row_index in detector_signature:
            detector_edge_row.append(int(row_index))
            detector_edge_col.append(int(column_index))
            detector_edge_mask.append(1)
        for logical_index in logical_signature:
            logical_edge_observable.append(int(logical_index))
            logical_edge_col.append(int(column_index))
            logical_edge_mask.append(1)
        merged_columns.append(
            ProjectedColumnMetadata(
                index=int(column_index),
                kind="binary",
                merged_multiplicity=int(len(group_entries)),
                source_gate_counts={str(key): int(value) for key, value in sorted(gate_counter.items())},
                representative_instruction_offset=int(representative["representative_instruction_offset"]),
                representative_tick_offset=int(representative["representative_tick_offset"]),
                representative_qubits=tuple(int(value) for value in tuple(representative["representative_qubits"])),
                detector_weight=int(len(detector_signature)),
                logical_weight=int(len(logical_signature)),
                error_probability=float(merged_probability),
                state_probabilities=(float(1.0 - merged_probability), float(merged_probability)),
            )
        )

    raw_duplicate_detector_groups = [
        list(group)
        for group in raw_duplicate_detector_only.values()
        if len(group) > 1
    ]
    metadata = ProjectedLocationMetadata(
        backend=str(problem.metadata.backend),
        projected_component=str(problem.metadata.projected_component),
        public_sector=str(problem.metadata.public_sector),
        error_rate=float(problem.metadata.error_rate),
        noisy_rounds=int(problem.metadata.noisy_rounds),
        total_rounds=int(problem.metadata.total_rounds),
        stim_path=str(problem.metadata.stim_path),
        num_detectors=int(problem.metadata.num_detectors),
        num_observables=int(problem.metadata.num_observables),
        raw_location_count=int(raw_column_count),
        raw_kind_counts={"binary": int(raw_column_count)},
        raw_gate_counts={str(key): int(value) for key, value in sorted(raw_gate_counts.items())},
        raw_zero_count=0,
        zero_gate_counts={},
        raw_duplicate_detector_only=_duplicate_summary(raw_duplicate_detector_groups),
        raw_duplicate_detector_plus_logical=_duplicate_summary(
            [group for group in raw_duplicate_full_groups if len(group) > 1]
        ),
        merged_kind_counts={"binary": int(len(merged_columns))},
        merged_multiplicity_hist={int(key): int(value) for key, value in sorted(merged_multiplicity_hist.items())},
        duplicate_source_combo_hist={str(key): int(value) for key, value in sorted(duplicate_source_combo_hist.items())},
        notes=(
            "This is a derived split-state binary control built from a projected-location detector-side family.",
            f"The source family grouping mode was `{problem.metadata.grouping_mode}`.",
            "Each nonzero local symbol is expanded into a Bernoulli branch on its exact projected detector-plus-logical action, then exact duplicate branches are merged.",
            "This control preserves the same projected action geometry but is not probabilistically exact when the source local 4-state prior does not factor into independent binary coordinates.",
        ),
        grouping_mode=str(variant_label),
        grouped_raw_location_count=int(raw_column_count),
        grouped_raw_kind_counts={"binary": int(raw_column_count)},
        grouped_attached_binary_count=0,
        grouped_unmatched_binary_count=0,
        grouped_group_size_hist={1: int(raw_column_count)},
    )
    return ProjectedLocationProblem(
        metadata=metadata,
        columns=tuple(merged_columns),
        detector_edge_row=np.asarray(detector_edge_row, dtype=np.int32),
        detector_edge_col=np.asarray(detector_edge_col, dtype=np.int32),
        detector_edge_mask=np.asarray(detector_edge_mask, dtype=np.uint8),
        logical_edge_observable=np.asarray(logical_edge_observable, dtype=np.int32),
        logical_edge_col=np.asarray(logical_edge_col, dtype=np.int32),
        logical_edge_mask=np.asarray(logical_edge_mask, dtype=np.uint8),
    )


__all__ = [
    "PROJECTED_COMPONENT_TO_PUBLIC_SECTOR",
    "PROJECTED_STATE_LABELS",
    "ProjectedColumnMetadata",
    "ProjectedLocationMetadata",
    "ProjectedLocationProblem",
    "build_projected_location_problem",
    "build_split_binary_control_problem",
]
