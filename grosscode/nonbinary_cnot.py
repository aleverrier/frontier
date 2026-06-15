from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from grosscode.circuits.backends import ResolvedBackendCircuit, resolve_backend_circuit
from grosscode.core import DecoderConfig, SideContext
from grosscode.utils.paths import resolve_qtanner_root


def _popcount(value: int) -> int:
    return int(bin(int(value) & 0xFFFF_FFFF).count("1"))


def _state_label(state: int) -> str:
    first = ((int(state) & 1) != 0) + 2 * (((int(state) & 2) != 0))
    second = (((int(state) & 4) != 0)) + 2 * (((int(state) & 8) != 0))
    lut = ("I", "X", "Z", "Y")
    return f"{lut[first]}{lut[second]}"


STATE_LABELS: tuple[str, ...] = tuple(_state_label(state) for state in range(16))
_BASIS_STATE_BITS: tuple[int, ...] = (1, 2, 4, 8)
_MASK_PARITY_TABLE = np.asarray(
    [[(_popcount(mask & state) & 1) for state in range(16)] for mask in range(16)],
    dtype=np.uint8,
)
_MASK_SIGN_TABLE = np.where(_MASK_PARITY_TABLE == 0, 1.0, -1.0).astype(np.float64)
_MASK_STATE_SPLIT = tuple(
    (
        np.flatnonzero(_MASK_PARITY_TABLE[mask] == 0).astype(np.int8, copy=False),
        np.flatnonzero(_MASK_PARITY_TABLE[mask] == 1).astype(np.int8, copy=False),
    )
    for mask in range(16)
)
_MASK_BRANCH_STATE_INDEX = tuple(
    np.asarray([int(state) - 1 for state in _MASK_STATE_SPLIT[mask][1].tolist() if int(state) != 0], dtype=np.int8)
    for mask in range(16)
)


def _small_logsumexp(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return float("-inf")
    vmax = float(np.max(arr))
    if not np.isfinite(vmax):
        return vmax
    return float(vmax + math.log(float(np.sum(np.exp(arr - vmax)))))


def _compress_scores_to_llr_logsumexp(scores: np.ndarray, mask: int, llr_clip: float) -> float:
    zeros, ones = _MASK_STATE_SPLIT[int(mask)]
    llr = _small_logsumexp(np.asarray(scores, dtype=np.float64)[zeros]) - _small_logsumexp(
        np.asarray(scores, dtype=np.float64)[ones]
    )
    return float(np.clip(llr, -float(llr_clip), float(llr_clip)))


def _compress_scores_to_llr_max(scores: np.ndarray, mask: int, llr_clip: float) -> float:
    zeros, ones = _MASK_STATE_SPLIT[int(mask)]
    arr = np.asarray(scores, dtype=np.float64)
    llr = float(np.max(arr[zeros]) - np.max(arr[ones]))
    return float(np.clip(llr, -float(llr_clip), float(llr_clip)))


def _check_update_minsum_binary(
    incoming_v2c: np.ndarray,
    syndrome_bit: int,
    old_c2v: np.ndarray,
    *,
    normalization: float,
    offset: float,
    damping: float,
    llr_clip: float,
) -> np.ndarray:
    degree = int(incoming_v2c.size)
    if degree == 0:
        return np.asarray(old_c2v, dtype=np.float64).copy()

    incoming = np.asarray(incoming_v2c, dtype=np.float64).reshape(-1)
    signs = np.where(incoming >= 0.0, 1.0, -1.0)
    abs_vals = np.abs(incoming)
    parity_sign = -1.0 if (int(syndrome_bit) & 1) else 1.0
    prod_sign = parity_sign * float(np.prod(signs))

    if degree == 1:
        forced = float(llr_clip) if (int(syndrome_bit) & 1) == 0 else -float(llr_clip)
        raw = np.asarray([forced], dtype=np.float64)
    else:
        idx_min = int(np.argmin(abs_vals))
        min1 = float(abs_vals[idx_min])
        masked = abs_vals.copy()
        masked[idx_min] = np.inf
        min2 = float(np.min(masked))
        mags = np.full(degree, float(normalization) * max(min1 - float(offset), 0.0), dtype=np.float64)
        mags[idx_min] = float(normalization) * max(min2 - float(offset), 0.0)
        raw = prod_sign * signs * mags

    raw = np.clip(raw, -float(llr_clip), float(llr_clip))
    if float(damping) > 0.0:
        return np.asarray((1.0 - float(damping)) * raw + float(damping) * np.asarray(old_c2v, dtype=np.float64))
    return raw


def _binary_contribution(mask: int, llr: float) -> np.ndarray:
    return 0.5 * float(llr) * _MASK_SIGN_TABLE[int(mask)]


@dataclass(frozen=True)
class CNOTLocationMetadata:
    backend: str
    sector: str
    error_rate: float
    noisy_rounds: int
    total_rounds: int
    stim_path: str
    num_locations: int
    num_detectors: int
    num_observables: int
    location_instruction_offsets: np.ndarray
    location_tick_offsets: np.ndarray
    location_qubit_pairs: np.ndarray
    visible_nonidentity_state_bitmasks: np.ndarray
    visible_nonidentity_state_counts: np.ndarray
    omitted_noise: tuple[str, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class CNOTLocationBlockMetadata:
    label: str
    detector_offset: int
    num_detectors: int
    observable_offset: int
    num_observables: int
    stim_path: str
    num_edges: int
    visible_nonidentity_state_bitmasks: np.ndarray
    visible_nonidentity_state_counts: np.ndarray

    @property
    def detector_slice(self) -> slice:
        start = int(self.detector_offset)
        return slice(start, start + int(self.num_detectors))

    @property
    def observable_slice(self) -> slice:
        start = int(self.observable_offset)
        return slice(start, start + int(self.num_observables))


@dataclass(frozen=True)
class CNOTLocationUnifiedMetadata:
    backend: str
    error_rate: float
    noisy_rounds: int
    total_rounds: int
    num_locations: int
    num_detectors: int
    num_observables: int
    location_instruction_offsets: np.ndarray
    location_tick_offsets: np.ndarray
    location_qubit_pairs: np.ndarray
    visible_nonidentity_state_bitmasks: np.ndarray
    visible_nonidentity_state_counts: np.ndarray
    blocks: tuple[CNOTLocationBlockMetadata, ...]
    omitted_noise: tuple[str, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class CNOTLocationDecodeResult:
    estimate_symbols: np.ndarray
    posterior_log_scores: np.ndarray
    converged: bool
    iterations: int
    edge_updates: int
    unsatisfied_checks: int
    unsatisfied_vector: np.ndarray
    logical_action: np.ndarray


@dataclass(frozen=True)
class CNOTLocationSplitSyndrome:
    X: np.ndarray
    Z: np.ndarray


@dataclass(frozen=True)
class CNOTLocationSplitDecodeResult:
    X: CNOTLocationDecodeResult
    Z: CNOTLocationDecodeResult
    converged: bool
    logical_frame_action: dict[str, np.ndarray]
    unsatisfied_checks: dict[str, int]
    iterations: dict[str, int]
    edge_updates: dict[str, int]


@dataclass(frozen=True)
class CNOTLocationSideProblem:
    name: str
    metadata: CNOTLocationMetadata
    prior_state_probabilities: np.ndarray
    prior_state_log_probs: np.ndarray
    edge_var: np.ndarray
    edge_check: np.ndarray
    edge_mask: np.ndarray
    check_to_edges: tuple[np.ndarray, ...]
    var_to_edges: tuple[np.ndarray, ...]
    logical_edge_var: np.ndarray
    logical_edge_observable: np.ndarray
    logical_edge_mask: np.ndarray

    @classmethod
    def from_local_masks(
        cls,
        *,
        name: str,
        error_rate: float,
        detector_masks_by_location: Sequence[Mapping[int, int]],
        logical_masks_by_location: Sequence[Mapping[int, int]] | None = None,
        metadata: CNOTLocationMetadata | None = None,
    ) -> "CNOTLocationSideProblem":
        n_locations = int(len(detector_masks_by_location))
        logical_maps = (
            tuple({int(k): int(v) for k, v in mapping.items()} for mapping in logical_masks_by_location)
            if logical_masks_by_location is not None
            else tuple({} for _ in range(n_locations))
        )
        if len(logical_maps) != n_locations:
            raise ValueError("logical_masks_by_location length mismatch")

        num_detectors = 0
        num_observables = 0
        for mapping in detector_masks_by_location:
            for check, mask in mapping.items():
                check_int = int(check)
                mask_int = int(mask)
                if not (0 <= mask_int < 16):
                    raise ValueError(f"detector mask must lie in [0, 15], got {mask_int}")
                num_detectors = max(num_detectors, check_int + 1)
        for mapping in logical_maps:
            for obs, mask in mapping.items():
                obs_int = int(obs)
                mask_int = int(mask)
                if not (0 <= mask_int < 16):
                    raise ValueError(f"logical mask must lie in [0, 15], got {mask_int}")
                num_observables = max(num_observables, obs_int + 1)

        edge_var: list[int] = []
        edge_check: list[int] = []
        edge_mask: list[int] = []
        check_to_edges: list[list[int]] = [[] for _ in range(num_detectors)]
        var_to_edges: list[list[int]] = [[] for _ in range(n_locations)]
        for var_index, mapping in enumerate(detector_masks_by_location):
            for check_index in sorted(int(x) for x in mapping.keys()):
                mask = int(mapping[check_index]) & 15
                if mask == 0:
                    continue
                edge_index = len(edge_var)
                edge_var.append(int(var_index))
                edge_check.append(int(check_index))
                edge_mask.append(int(mask))
                check_to_edges[int(check_index)].append(int(edge_index))
                var_to_edges[int(var_index)].append(int(edge_index))

        logical_edge_var: list[int] = []
        logical_edge_observable: list[int] = []
        logical_edge_mask: list[int] = []
        for var_index, mapping in enumerate(logical_maps):
            for obs_index in sorted(int(x) for x in mapping.keys()):
                mask = int(mapping[obs_index]) & 15
                if mask == 0:
                    continue
                logical_edge_var.append(int(var_index))
                logical_edge_observable.append(int(obs_index))
                logical_edge_mask.append(int(mask))

        p = float(error_rate)
        if not (0.0 <= p <= 1.0):
            raise ValueError("error_rate must lie in [0, 1]")
        prior_state_probabilities = np.full(16, 0.0 if p == 0.0 else p / 15.0, dtype=np.float64)
        prior_state_probabilities[0] = 1.0 - p
        prior_state_log_probs = np.where(
            prior_state_probabilities > 0.0,
            np.log(prior_state_probabilities),
            -np.inf,
        ).astype(np.float64)

        if metadata is None:
            metadata = CNOTLocationMetadata(
                backend="synthetic",
                sector=str(name),
                error_rate=float(error_rate),
                noisy_rounds=0,
                total_rounds=0,
                stim_path="",
                num_locations=n_locations,
                num_detectors=num_detectors,
                num_observables=num_observables,
                location_instruction_offsets=np.arange(n_locations, dtype=np.int32),
                location_tick_offsets=np.zeros(n_locations, dtype=np.int32),
                location_qubit_pairs=np.full((n_locations, 2), -1, dtype=np.int32),
                visible_nonidentity_state_bitmasks=np.zeros(n_locations, dtype=np.uint16),
                visible_nonidentity_state_counts=np.zeros(n_locations, dtype=np.int16),
                omitted_noise=(),
                notes=("Synthetic CNOT-location nonbinary side problem.",),
            )

        return cls(
            name=str(name),
            metadata=metadata,
            prior_state_probabilities=prior_state_probabilities,
            prior_state_log_probs=prior_state_log_probs,
            edge_var=np.asarray(edge_var, dtype=np.int32),
            edge_check=np.asarray(edge_check, dtype=np.int32),
            edge_mask=np.asarray(edge_mask, dtype=np.uint8),
            check_to_edges=tuple(np.asarray(edges, dtype=np.int32) for edges in check_to_edges),
            var_to_edges=tuple(np.asarray(edges, dtype=np.int32) for edges in var_to_edges),
            logical_edge_var=np.asarray(logical_edge_var, dtype=np.int32),
            logical_edge_observable=np.asarray(logical_edge_observable, dtype=np.int32),
            logical_edge_mask=np.asarray(logical_edge_mask, dtype=np.uint8),
        )

    @property
    def n(self) -> int:
        return int(self.metadata.num_locations)

    @property
    def m(self) -> int:
        return int(self.metadata.num_detectors)

    @property
    def k(self) -> int:
        return int(self.metadata.num_observables)

    @property
    def n_edges(self) -> int:
        return int(self.edge_var.size)

    def initial_log_scores(self) -> np.ndarray:
        return np.broadcast_to(self.prior_state_log_probs, (self.n, 16)).copy()

    def syndrome_from_symbols(self, symbols: np.ndarray) -> np.ndarray:
        symbol_vec = np.asarray(symbols, dtype=np.uint8).reshape(-1)
        if int(symbol_vec.size) != self.n:
            raise ValueError(f"symbol vector length mismatch: got {symbol_vec.size}, expected {self.n}")
        out = np.zeros(self.m, dtype=np.uint8)
        if self.edge_var.size == 0:
            return out
        edge_bits = _MASK_PARITY_TABLE[self.edge_mask, symbol_vec[self.edge_var]].astype(np.uint8, copy=False)
        np.bitwise_xor.at(out, self.edge_check, edge_bits)
        return out

    def logical_action_from_symbols(self, symbols: np.ndarray) -> np.ndarray:
        symbol_vec = np.asarray(symbols, dtype=np.uint8).reshape(-1)
        if int(symbol_vec.size) != self.n:
            raise ValueError(f"symbol vector length mismatch: got {symbol_vec.size}, expected {self.n}")
        out = np.zeros(self.k, dtype=np.uint8)
        if self.logical_edge_var.size == 0:
            return out
        logical_bits = _MASK_PARITY_TABLE[self.logical_edge_mask, symbol_vec[self.logical_edge_var]].astype(
            np.uint8, copy=False
        )
        np.bitwise_xor.at(out, self.logical_edge_observable, logical_bits)
        return out

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        symbols = np.asarray(rng.choice(16, size=self.n, p=self.prior_state_probabilities), dtype=np.uint8)
        return symbols, self.syndrome_from_symbols(symbols), self.logical_action_from_symbols(symbols)


@dataclass(frozen=True)
class CNOTLocationProblem:
    graph: CNOTLocationSideProblem
    metadata: CNOTLocationUnifiedMetadata
    block_problems: tuple[CNOTLocationSideProblem, ...]

    @classmethod
    def from_block_problems(
        cls,
        *,
        graph: CNOTLocationSideProblem,
        metadata: CNOTLocationUnifiedMetadata,
        block_problems: Sequence[CNOTLocationSideProblem],
    ) -> "CNOTLocationProblem":
        blocks = tuple(block_problems)
        if tuple(block.label for block in metadata.blocks) != tuple(problem.name for problem in blocks):
            raise ValueError("block metadata labels must match block problem names and order")
        return cls(graph=graph, metadata=metadata, block_problems=blocks)

    @property
    def n(self) -> int:
        return int(self.graph.n)

    @property
    def m(self) -> int:
        return int(self.graph.m)

    @property
    def k(self) -> int:
        return int(self.graph.k)

    @property
    def n_edges(self) -> int:
        return int(self.graph.n_edges)

    @property
    def prior_state_probabilities(self) -> np.ndarray:
        return self.graph.prior_state_probabilities

    @property
    def prior_state_log_probs(self) -> np.ndarray:
        return self.graph.prior_state_log_probs

    @property
    def edge_var(self) -> np.ndarray:
        return self.graph.edge_var

    @property
    def edge_check(self) -> np.ndarray:
        return self.graph.edge_check

    @property
    def edge_mask(self) -> np.ndarray:
        return self.graph.edge_mask

    @property
    def check_to_edges(self) -> tuple[np.ndarray, ...]:
        return self.graph.check_to_edges

    @property
    def var_to_edges(self) -> tuple[np.ndarray, ...]:
        return self.graph.var_to_edges

    @property
    def logical_edge_var(self) -> np.ndarray:
        return self.graph.logical_edge_var

    @property
    def logical_edge_observable(self) -> np.ndarray:
        return self.graph.logical_edge_observable

    @property
    def logical_edge_mask(self) -> np.ndarray:
        return self.graph.logical_edge_mask

    def initial_log_scores(self) -> np.ndarray:
        return self.graph.initial_log_scores()

    def syndrome_from_symbols(self, symbols: np.ndarray) -> np.ndarray:
        return self.graph.syndrome_from_symbols(symbols)

    def logical_action_from_symbols(self, symbols: np.ndarray) -> np.ndarray:
        return self.graph.logical_action_from_symbols(symbols)

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.graph.sample(rng)

    def block_metadata(self, label: str) -> CNOTLocationBlockMetadata:
        key = str(label)
        for block in self.metadata.blocks:
            if str(block.label) == key:
                return block
        raise KeyError(f"unknown CNOT-location block label: {label}")

    def block_problem(self, label: str) -> CNOTLocationSideProblem:
        key = str(label)
        for problem in self.block_problems:
            if str(problem.name) == key:
                return problem
        raise KeyError(f"unknown CNOT-location block label: {label}")

    @property
    def X(self) -> CNOTLocationSideProblem:
        return self.block_problem("memory_X")

    @property
    def Z(self) -> CNOTLocationSideProblem:
        return self.block_problem("memory_Z")

    def block_detector_slice(self, label: str) -> slice:
        return self.block_metadata(label).detector_slice

    def block_observable_slice(self, label: str) -> slice:
        return self.block_metadata(label).observable_slice

    def split_detector_vector(self, values: np.ndarray) -> dict[str, np.ndarray]:
        arr = np.asarray(values, dtype=np.uint8).reshape(-1)
        if int(arr.size) != self.m:
            raise ValueError(f"detector vector length mismatch: got {arr.size}, expected {self.m}")
        return {
            str(block.label): np.asarray(arr[block.detector_slice], dtype=np.uint8)
            for block in self.metadata.blocks
        }

    def split_logical_action(self, values: np.ndarray) -> dict[str, np.ndarray]:
        arr = np.asarray(values, dtype=np.uint8).reshape(-1)
        if int(arr.size) != self.k:
            raise ValueError(f"logical vector length mismatch: got {arr.size}, expected {self.k}")
        return {
            str(block.label): np.asarray(arr[block.observable_slice], dtype=np.uint8)
            for block in self.metadata.blocks
        }

    def combine_detector_blocks(self, blocks: Mapping[str, np.ndarray]) -> np.ndarray:
        out = np.zeros(self.m, dtype=np.uint8)
        for block in self.metadata.blocks:
            key = str(block.label)
            if key not in blocks:
                continue
            arr = np.asarray(blocks[key], dtype=np.uint8).reshape(-1) & 1
            if int(arr.size) != int(block.num_detectors):
                raise ValueError(
                    f"detector block length mismatch for {key}: got {arr.size}, expected {int(block.num_detectors)}"
                )
            out[block.detector_slice] = arr
        return out

    def detector_mask_matrix(self):  # type: ignore[no-untyped-def]
        import scipy.sparse as sp  # type: ignore

        return sp.csr_matrix(
            (
                np.asarray(self.edge_mask, dtype=np.uint8),
                (np.asarray(self.edge_check, dtype=np.int32), np.asarray(self.edge_var, dtype=np.int32)),
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
                    np.asarray(self.logical_edge_var, dtype=np.int32),
                ),
            ),
            shape=(self.k, self.n),
            dtype=np.uint8,
        )

    def duplicate_column_groups(self, *, include_logicals: bool = True) -> tuple[np.ndarray, ...]:
        detector = self.detector_mask_matrix().tocsc()
        logical = self.logical_mask_matrix().tocsc() if include_logicals else None
        groups: dict[tuple[object, ...], list[int]] = defaultdict(list)
        for col in range(self.n):
            d_start = int(detector.indptr[col])
            d_end = int(detector.indptr[col + 1])
            signature: tuple[object, ...] = (
                tuple(int(x) for x in detector.indices[d_start:d_end]),
                tuple(int(x) for x in detector.data[d_start:d_end]),
            )
            if logical is not None:
                l_start = int(logical.indptr[col])
                l_end = int(logical.indptr[col + 1])
                signature += (
                    tuple(int(x) for x in logical.indices[l_start:l_end]),
                    tuple(int(x) for x in logical.data[l_start:l_end]),
                )
            groups[signature].append(int(col))
        return tuple(
            np.asarray(cols, dtype=np.int32)
            for cols in groups.values()
            if len(cols) > 1
        )

    def duplicate_column_summary(self, *, include_logicals: bool = True) -> dict[str, int | bool]:
        groups = self.duplicate_column_groups(include_logicals=include_logicals)
        duplicate_class_count = int(len(groups))
        duplicate_column_count = int(sum(int(group.size) for group in groups))
        extra_column_count = int(sum(int(group.size) - 1 for group in groups))
        largest_group_size = int(max((int(group.size) for group in groups), default=1))
        unique_column_count = int(self.n - extra_column_count)
        return {
            "include_logicals": bool(include_logicals),
            "duplicate_class_count": duplicate_class_count,
            "duplicate_column_count": duplicate_column_count,
            "extra_column_count": extra_column_count,
            "largest_group_size": largest_group_size,
            "unique_column_count": unique_column_count,
        }

    def write_matrix_bundle(self, out_dir: str | Path) -> Path:
        import scipy.sparse as sp  # type: ignore
        from scipy.io import mmwrite  # type: ignore

        dest = Path(out_dir).resolve()
        dest.mkdir(parents=True, exist_ok=True)

        detector_matrix = self.detector_mask_matrix().tocoo()
        logical_matrix = self.logical_mask_matrix().tocoo()
        detector_only_duplicates = self.duplicate_column_summary(include_logicals=False)
        detector_plus_logical_duplicates = self.duplicate_column_summary(include_logicals=True)
        sp.save_npz(dest / "detector_mask_matrix.npz", detector_matrix.tocsr())
        sp.save_npz(dest / "logical_mask_matrix.npz", logical_matrix.tocsr())
        mmwrite(str(dest / "detector_mask_matrix.mtx"), detector_matrix)
        mmwrite(str(dest / "logical_mask_matrix.mtx"), logical_matrix)

        with (dest / "location_index.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["column", "tick_offset", "instruction_offset", "qubit_a", "qubit_b"],
            )
            writer.writeheader()
            for col in range(self.n):
                writer.writerow(
                    {
                        "column": int(col),
                        "tick_offset": int(self.metadata.location_tick_offsets[col]),
                        "instruction_offset": int(self.metadata.location_instruction_offsets[col]),
                        "qubit_a": int(self.metadata.location_qubit_pairs[col, 0]),
                        "qubit_b": int(self.metadata.location_qubit_pairs[col, 1]),
                    }
                )

        bundle_metadata = {
            "model_label": "non-default reduced Gross CNOT-location 16-state detector-side matrix",
            "detector_matrix_shape": [int(self.m), int(self.n)],
            "logical_matrix_shape": [int(self.k), int(self.n)],
            "entry_semantics": {
                "mask_bits_order": ["x1", "z1", "x2", "z2"],
                "state_encoding": {str(idx): label for idx, label in enumerate(STATE_LABELS)},
                "parity_rule": "For symbol state s in {0,...,15}, row contribution is popcount(mask & s) mod 2.",
            },
            "duplicate_columns": {
                "detector_only": detector_only_duplicates,
                "detector_plus_logical": detector_plus_logical_duplicates,
            },
            "blocks": [
                {
                    "label": str(block.label),
                    "detector_offset": int(block.detector_offset),
                    "num_detectors": int(block.num_detectors),
                    "observable_offset": int(block.observable_offset),
                    "num_observables": int(block.num_observables),
                    "stim_path": str(block.stim_path),
                    "num_edges": int(block.num_edges),
                }
                for block in self.metadata.blocks
            ],
            "location_count": int(self.n),
            "omitted_noise": [str(item) for item in self.metadata.omitted_noise],
            "notes": [str(item) for item in self.metadata.notes],
        }
        (dest / "metadata.json").write_text(json.dumps(bundle_metadata, indent=2) + "\n")
        report_lines = [
            "# Reduced Gross CNOT-location matrix bundle",
            "",
            "- Model: non-default reduced Gross CNOT-location 16-state detector-side matrix.",
            f"- Detector matrix: `{self.m} x {self.n}`.",
            f"- Logical matrix: `{self.k} x {self.n}`.",
            "- Entry semantics: each nonzero entry is a 4-bit mask in `{0,...,15}` with bit order `(x1, z1, x2, z2)`.",
            "- Parity rule: for symbol state `s in {0,...,15}`, the row contribution is `popcount(mask & s) mod 2`.",
            f"- Duplicate-column audit (`detector + logical`): `{detector_plus_logical_duplicates['duplicate_class_count']}` duplicate classes, "
            f"`{detector_plus_logical_duplicates['extra_column_count']}` extra columns beyond unique representatives.",
            f"- Duplicate-column audit (`detector only`): `{detector_only_duplicates['duplicate_class_count']}` duplicate classes, "
            f"`{detector_only_duplicates['extra_column_count']}` extra columns beyond unique representatives.",
            "",
            "## Blocks",
            "",
        ]
        for block in self.metadata.blocks:
            report_lines.append(
                f"- `{block.label}`: detector rows `{int(block.detector_offset)}:{int(block.detector_offset + block.num_detectors)}`, "
                f"logical rows `{int(block.observable_offset)}:{int(block.observable_offset + block.num_observables)}`, "
                f"edges `{int(block.num_edges)}`."
            )
        report_lines.extend(
            [
                "",
                "## Files",
                "",
                "- `detector_mask_matrix.mtx` / `detector_mask_matrix.npz`: unified detector mask matrix.",
                "- `logical_mask_matrix.mtx` / `logical_mask_matrix.npz`: unified logical mask matrix.",
                "- `location_index.csv`: shared column index with tick, instruction offset, and CNOT qubit pair.",
                "- `metadata.json`: machine-readable matrix semantics and block layout.",
            ]
        )
        (dest / "report.md").write_text("\n".join(report_lines) + "\n")
        return dest


@dataclass(frozen=True)
class CNOTLocationBernoulliExpansionProblem:
    source_problem: CNOTLocationProblem
    context: SideContext
    branch_location_index: np.ndarray
    branch_state: np.ndarray
    block_edge_counts: dict[str, int]
    visible_branch_count: int
    zero_support_branch_count: int

    @classmethod
    def from_cnot_problem(cls, problem: CNOTLocationProblem) -> "CNOTLocationBernoulliExpansionProblem":
        import scipy.sparse as sp  # type: ignore
        from grosscode.core import binary_csr_mod2

        n_locations = int(problem.n)
        branches_per_location = 15
        total_columns = int(n_locations * branches_per_location)

        detector_nnz = int(sum(int(_MASK_BRANCH_STATE_INDEX[int(mask)].size) for mask in problem.edge_mask.tolist()))
        detector_rows = np.empty(detector_nnz, dtype=np.int32)
        detector_cols = np.empty(detector_nnz, dtype=np.int32)
        cursor = 0
        for edge_index in range(int(problem.n_edges)):
            location_index = int(problem.edge_var[edge_index])
            check_index = int(problem.edge_check[edge_index])
            branch_offsets = _MASK_BRANCH_STATE_INDEX[int(problem.edge_mask[edge_index])]
            count = int(branch_offsets.size)
            if count == 0:
                continue
            detector_rows[cursor : cursor + count] = int(check_index)
            detector_cols[cursor : cursor + count] = int(location_index * branches_per_location) + np.asarray(
                branch_offsets,
                dtype=np.int32,
            )
            cursor += count
        detector_matrix = sp.csr_matrix(
            (
                np.ones(cursor, dtype=np.uint8),
                (detector_rows[:cursor], detector_cols[:cursor]),
            ),
            shape=(int(problem.m), total_columns),
            dtype=np.uint8,
        )
        detector_matrix = binary_csr_mod2(detector_matrix)

        logical_nnz = int(
            sum(int(_MASK_BRANCH_STATE_INDEX[int(mask)].size) for mask in problem.logical_edge_mask.tolist())
        )
        logical_rows = np.empty(logical_nnz, dtype=np.int32)
        logical_cols = np.empty(logical_nnz, dtype=np.int32)
        cursor = 0
        for edge_index in range(int(problem.logical_edge_var.size)):
            location_index = int(problem.logical_edge_var[edge_index])
            logical_index = int(problem.logical_edge_observable[edge_index])
            branch_offsets = _MASK_BRANCH_STATE_INDEX[int(problem.logical_edge_mask[edge_index])]
            count = int(branch_offsets.size)
            if count == 0:
                continue
            logical_rows[cursor : cursor + count] = int(logical_index)
            logical_cols[cursor : cursor + count] = int(location_index * branches_per_location) + np.asarray(
                branch_offsets,
                dtype=np.int32,
            )
            cursor += count
        logical_matrix = sp.csr_matrix(
            (
                np.ones(cursor, dtype=np.uint8),
                (logical_rows[:cursor], logical_cols[:cursor]),
            ),
            shape=(int(problem.k), total_columns),
            dtype=np.uint8,
        )
        logical_matrix = binary_csr_mod2(logical_matrix)

        branch_priors = np.full(total_columns, 0.0, dtype=np.float64)
        if float(problem.metadata.error_rate) > 0.0:
            branch_priors.fill(float(problem.metadata.error_rate) / 15.0)

        branch_location_index = np.repeat(np.arange(n_locations, dtype=np.int32), branches_per_location)
        branch_state = np.tile(np.arange(1, 16, dtype=np.uint8), n_locations)
        visible_branch_count = int(
            sum(int(problem.metadata.visible_nonidentity_state_counts[idx]) for idx in range(n_locations))
        )
        zero_support_branch_count = int(total_columns - visible_branch_count)
        block_edge_counts = {
            str(block.label): int(block.num_edges) * 8
            for block in problem.metadata.blocks
        }
        return cls(
            source_problem=problem,
            context=SideContext.from_matrices(
                name="reduced_cnot_bernoulli15",
                check_matrix=detector_matrix,
                observables=logical_matrix,
                priors=branch_priors,
            ),
            branch_location_index=branch_location_index,
            branch_state=branch_state,
            block_edge_counts=block_edge_counts,
            visible_branch_count=visible_branch_count,
            zero_support_branch_count=zero_support_branch_count,
        )

    @property
    def metadata(self) -> CNOTLocationUnifiedMetadata:
        return self.source_problem.metadata

    @property
    def n(self) -> int:
        return int(self.context.n)

    @property
    def m(self) -> int:
        return int(self.context.m)

    @property
    def k(self) -> int:
        return int(self.context.observables.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.context.graph.n_edges)

    @property
    def branches_per_location(self) -> int:
        return 15

    def branch_index(self, location_index: int, state: int) -> int:
        state_int = int(state)
        if not (0 <= int(location_index) < int(self.source_problem.n)):
            raise ValueError(f"location index out of range: {location_index}")
        if not (1 <= state_int <= 15):
            raise ValueError(f"expanded branch state must lie in [1, 15], got {state}")
        return int(int(location_index) * self.branches_per_location + (state_int - 1))

    def expanded_bits_from_symbols(self, symbols: np.ndarray) -> np.ndarray:
        symbol_vec = np.asarray(symbols, dtype=np.uint8).reshape(-1)
        if int(symbol_vec.size) != int(self.source_problem.n):
            raise ValueError(
                f"symbol vector length mismatch: got {symbol_vec.size}, expected {int(self.source_problem.n)}"
            )
        out = np.zeros(self.n, dtype=np.uint8)
        active_locations = np.flatnonzero(symbol_vec != 0).astype(np.int32, copy=False)
        if active_locations.size:
            active_states = symbol_vec[active_locations].astype(np.int32, copy=False)
            out[active_locations * int(self.branches_per_location) + (active_states - 1)] = 1
        return out

    def syndrome_from_bits(self, bits: np.ndarray) -> np.ndarray:
        return self.context.syndrome(np.asarray(bits, dtype=np.uint8))

    def logical_action_from_bits(self, bits: np.ndarray) -> np.ndarray:
        return self.context.logical_action_for(np.asarray(bits, dtype=np.uint8))

    def syndrome_from_symbols(self, symbols: np.ndarray) -> np.ndarray:
        return self.syndrome_from_bits(self.expanded_bits_from_symbols(symbols))

    def logical_action_from_symbols(self, symbols: np.ndarray) -> np.ndarray:
        return self.logical_action_from_bits(self.expanded_bits_from_symbols(symbols))

    def block_metadata(self, label: str) -> CNOTLocationBlockMetadata:
        return self.source_problem.block_metadata(label)

    def split_detector_vector(self, values: np.ndarray) -> dict[str, np.ndarray]:
        return self.source_problem.split_detector_vector(values)

    def split_logical_action(self, values: np.ndarray) -> dict[str, np.ndarray]:
        return self.source_problem.split_logical_action(values)

    def detector_matrix(self):  # type: ignore[no-untyped-def]
        return self.context.graph.H

    def logical_matrix(self):  # type: ignore[no-untyped-def]
        return self.context.observables

    def write_matrix_bundle(self, out_dir: str | Path) -> Path:
        import scipy.sparse as sp  # type: ignore
        from scipy.io import mmwrite  # type: ignore

        dest = Path(out_dir).resolve()
        dest.mkdir(parents=True, exist_ok=True)

        detector_matrix = self.detector_matrix().tocoo()
        logical_matrix = self.logical_matrix().tocoo()
        sp.save_npz(dest / "detector_matrix.npz", detector_matrix.tocsr())
        sp.save_npz(dest / "logical_matrix.npz", logical_matrix.tocsr())
        mmwrite(str(dest / "detector_matrix.mtx"), detector_matrix)
        mmwrite(str(dest / "logical_matrix.mtx"), logical_matrix)

        with (dest / "branch_index.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "column",
                    "location_index",
                    "state",
                    "state_label",
                    "tick_offset",
                    "instruction_offset",
                    "qubit_a",
                    "qubit_b",
                ],
            )
            writer.writeheader()
            for column in range(self.n):
                location_index = int(self.branch_location_index[column])
                state = int(self.branch_state[column])
                writer.writerow(
                    {
                        "column": int(column),
                        "location_index": int(location_index),
                        "state": int(state),
                        "state_label": str(STATE_LABELS[state]),
                        "tick_offset": int(self.metadata.location_tick_offsets[location_index]),
                        "instruction_offset": int(self.metadata.location_instruction_offsets[location_index]),
                        "qubit_a": int(self.metadata.location_qubit_pairs[location_index, 0]),
                        "qubit_b": int(self.metadata.location_qubit_pairs[location_index, 1]),
                    }
                )

        bundle_metadata = {
            "model_label": "non-default reduced Gross CNOT-location 15-branch Bernoulli expansion",
            "source_model_label": "non-default reduced Gross CNOT-location 16-state detector-side matrix",
            "detector_matrix_shape": [int(self.m), int(self.n)],
            "logical_matrix_shape": [int(self.k), int(self.n)],
            "detector_edge_count": int(self.n_edges),
            "branch_prior": float(self.metadata.error_rate) / 15.0 if float(self.metadata.error_rate) > 0.0 else 0.0,
            "branches_per_location": int(self.branches_per_location),
            "location_count": int(self.source_problem.n),
            "visible_branch_count": int(self.visible_branch_count),
            "zero_support_branch_count": int(self.zero_support_branch_count),
            "block_edge_counts": {str(k): int(v) for k, v in self.block_edge_counts.items()},
            "notes": [
                "Columns are unmerged binary Bernoulli branches, one for each non-identity 2-qubit Pauli state at each CNOT location.",
                "This expansion uses iid branch priors p/15 and does not enforce same-location one-of-16 exclusivity.",
                "The underlying circuit family remains the reduced post-CNOT-only non-default Gross detector-side model.",
                "Reset noise, measurement noise, and DEPOLARIZE1 noise remain omitted.",
            ],
            "omitted_noise": [str(item) for item in self.metadata.omitted_noise],
        }
        (dest / "metadata.json").write_text(json.dumps(bundle_metadata, indent=2) + "\n")
        report_lines = [
            "# Reduced Gross CNOT-location 15-branch Bernoulli expansion bundle",
            "",
            "- Model: non-default reduced Gross detector-side CNOT-location Bernoulli expansion.",
            f"- Detector matrix: `{self.m} x {self.n}` with `{self.n_edges}` binary detector-variable edges.",
            f"- Logical matrix: `{self.k} x {self.n}`.",
            f"- Source locations: `{int(self.source_problem.n)}` shared CNOT locations, each expanded to `{int(self.branches_per_location)}` binary branches.",
            (
                f"- Prior per branch: "
                f"`{(float(self.metadata.error_rate) / 15.0 if float(self.metadata.error_rate) > 0.0 else 0.0):.12g}`."
            ),
            (
                f"- Visible branches on the current reduced Gross build: `{int(self.visible_branch_count)}`; "
                f"zero-support branches kept explicitly: `{int(self.zero_support_branch_count)}`."
            ),
            "- Important caveat: this is a surrogate iid Bernoulli expansion of the reduced CNOT-location model, not the exact one-of-16 physical channel.",
            "",
            "## Files",
            "",
            "- `detector_matrix.mtx` / `detector_matrix.npz`: expanded binary detector matrix.",
            "- `logical_matrix.mtx` / `logical_matrix.npz`: expanded binary logical matrix.",
            "- `branch_index.csv`: branch-to-location/state index with tick, instruction offset, and qubit pair metadata.",
            "- `metadata.json`: machine-readable bundle metadata and modeling caveats.",
        ]
        (dest / "report.md").write_text("\n".join(report_lines) + "\n")
        return dest


@dataclass
class CNOTLocationMergedBernoulliProblem:
    source_problem: CNOTLocationProblem
    source_expansion: CNOTLocationBernoulliExpansionProblem
    context: SideContext
    block_edge_counts: dict[str, int]
    merged_column_multiplicity: np.ndarray
    merged_column_detector_weight: np.ndarray
    merged_column_logical_weight: np.ndarray
    representative_source_column: np.ndarray
    unique_signature_count: int
    dropped_zero_signature: bool
    dropped_zero_signature_multiplicity: int
    multiplicity_histogram: dict[int, int]

    @classmethod
    def from_expansion_problem(
        cls,
        problem: CNOTLocationBernoulliExpansionProblem,
        *,
        drop_zero_signature: bool = True,
    ) -> "CNOTLocationMergedBernoulliProblem":
        import scipy.sparse as sp  # type: ignore

        detector_matrix_csc = problem.detector_matrix().tocsc()
        logical_matrix_csc = problem.logical_matrix().tocsc()
        prior_vec = np.asarray(problem.context.priors, dtype=np.float64).reshape(-1)
        if int(prior_vec.size) != int(problem.n):
            raise ValueError("prior length mismatch in Bernoulli expansion problem")

        signature_to_group: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {}
        detector_signature_rows: list[tuple[int, ...]] = []
        logical_signature_rows: list[tuple[int, ...]] = []
        representative_source_column: list[int] = []
        multiplicity: list[int] = []
        odd_bias_product: list[float] = []

        for column in range(int(problem.n)):
            detector_rows = tuple(
                int(item)
                for item in detector_matrix_csc.indices[
                    detector_matrix_csc.indptr[column] : detector_matrix_csc.indptr[column + 1]
                ].tolist()
            )
            logical_rows = tuple(
                int(item)
                for item in logical_matrix_csc.indices[
                    logical_matrix_csc.indptr[column] : logical_matrix_csc.indptr[column + 1]
                ].tolist()
            )
            signature = (detector_rows, logical_rows)
            group_index = signature_to_group.get(signature)
            if group_index is None:
                group_index = int(len(detector_signature_rows))
                signature_to_group[signature] = group_index
                detector_signature_rows.append(detector_rows)
                logical_signature_rows.append(logical_rows)
                representative_source_column.append(int(column))
                multiplicity.append(0)
                odd_bias_product.append(1.0)
            multiplicity[group_index] += 1
            odd_bias_product[group_index] *= 1.0 - 2.0 * float(prior_vec[column])

        zero_signature_group = signature_to_group.get((tuple(), tuple()))
        keep_group = np.ones(len(detector_signature_rows), dtype=bool)
        dropped_zero_signature_multiplicity = 0
        if bool(drop_zero_signature) and zero_signature_group is not None:
            keep_group[int(zero_signature_group)] = False
            dropped_zero_signature_multiplicity = int(multiplicity[int(zero_signature_group)])

        new_column_count = int(np.count_nonzero(keep_group))
        detector_nnz = int(
            sum(
                len(detector_signature_rows[group_index])
                for group_index in range(len(detector_signature_rows))
                if bool(keep_group[group_index])
            )
        )
        logical_nnz = int(
            sum(
                len(logical_signature_rows[group_index])
                for group_index in range(len(logical_signature_rows))
                if bool(keep_group[group_index])
            )
        )

        detector_rows_out = np.empty(detector_nnz, dtype=np.int32)
        detector_cols_out = np.empty(detector_nnz, dtype=np.int32)
        logical_rows_out = np.empty(logical_nnz, dtype=np.int32)
        logical_cols_out = np.empty(logical_nnz, dtype=np.int32)
        merged_priors = np.empty(new_column_count, dtype=np.float64)
        merged_column_multiplicity = np.empty(new_column_count, dtype=np.int32)
        merged_column_detector_weight = np.empty(new_column_count, dtype=np.int16)
        merged_column_logical_weight = np.empty(new_column_count, dtype=np.int16)
        representative_source_column_out = np.empty(new_column_count, dtype=np.int32)

        detector_cursor = 0
        logical_cursor = 0
        new_column = 0
        for group_index in range(len(detector_signature_rows)):
            if not bool(keep_group[group_index]):
                continue
            detector_rows = detector_signature_rows[group_index]
            logical_rows = logical_signature_rows[group_index]
            detector_weight = int(len(detector_rows))
            logical_weight = int(len(logical_rows))
            if detector_weight:
                detector_rows_out[detector_cursor : detector_cursor + detector_weight] = np.asarray(
                    detector_rows,
                    dtype=np.int32,
                )
                detector_cols_out[detector_cursor : detector_cursor + detector_weight] = int(new_column)
                detector_cursor += detector_weight
            if logical_weight:
                logical_rows_out[logical_cursor : logical_cursor + logical_weight] = np.asarray(
                    logical_rows,
                    dtype=np.int32,
                )
                logical_cols_out[logical_cursor : logical_cursor + logical_weight] = int(new_column)
                logical_cursor += logical_weight
            merged_priors[new_column] = float(np.clip(0.5 * (1.0 - odd_bias_product[group_index]), 0.0, 1.0))
            merged_column_multiplicity[new_column] = int(multiplicity[group_index])
            merged_column_detector_weight[new_column] = np.int16(detector_weight)
            merged_column_logical_weight[new_column] = np.int16(logical_weight)
            representative_source_column_out[new_column] = int(representative_source_column[group_index])
            new_column += 1

        detector_matrix = sp.csr_matrix(
            (
                np.ones(detector_cursor, dtype=np.uint8),
                (detector_rows_out[:detector_cursor], detector_cols_out[:detector_cursor]),
            ),
            shape=(int(problem.m), int(new_column_count)),
            dtype=np.uint8,
        )
        logical_matrix = sp.csr_matrix(
            (
                np.ones(logical_cursor, dtype=np.uint8),
                (logical_rows_out[:logical_cursor], logical_cols_out[:logical_cursor]),
            ),
            shape=(int(problem.k), int(new_column_count)),
            dtype=np.uint8,
        )

        block_edge_counts: dict[str, int] = {}
        for block in problem.metadata.blocks:
            row_slice = slice(int(block.detector_offset), int(block.detector_offset + block.num_detectors))
            block_edge_counts[str(block.label)] = int(detector_matrix[row_slice, :].nnz)

        multiplicity_histogram = {
            int(group_size): int(count)
            for group_size, count in sorted(Counter(multiplicity).items())
        }
        return cls(
            source_problem=problem.source_problem,
            source_expansion=problem,
            context=SideContext.from_matrices(
                name="reduced_cnot_bernoulli15_merged",
                check_matrix=detector_matrix,
                observables=logical_matrix,
                priors=merged_priors,
            ),
            block_edge_counts=block_edge_counts,
            merged_column_multiplicity=np.asarray(merged_column_multiplicity, dtype=np.int32),
            merged_column_detector_weight=np.asarray(merged_column_detector_weight, dtype=np.int16),
            merged_column_logical_weight=np.asarray(merged_column_logical_weight, dtype=np.int16),
            representative_source_column=np.asarray(representative_source_column_out, dtype=np.int32),
            unique_signature_count=int(len(detector_signature_rows)),
            dropped_zero_signature=bool(drop_zero_signature and zero_signature_group is not None),
            dropped_zero_signature_multiplicity=int(dropped_zero_signature_multiplicity),
            multiplicity_histogram=dict(multiplicity_histogram),
        )

    @classmethod
    def from_cnot_problem(
        cls,
        problem: CNOTLocationProblem,
        *,
        drop_zero_signature: bool = True,
    ) -> "CNOTLocationMergedBernoulliProblem":
        return cls.from_expansion_problem(
            CNOTLocationBernoulliExpansionProblem.from_cnot_problem(problem),
            drop_zero_signature=drop_zero_signature,
        )

    @property
    def metadata(self) -> CNOTLocationUnifiedMetadata:
        return self.source_problem.metadata

    @property
    def n(self) -> int:
        return int(self.context.n)

    @property
    def m(self) -> int:
        return int(self.context.m)

    @property
    def k(self) -> int:
        return int(self.context.observables.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.context.graph.n_edges)

    def merge_summary(self) -> dict[str, int | bool | dict[int, int]]:
        return {
            "original_column_count": int(self.source_expansion.n),
            "unique_signature_count": int(self.unique_signature_count),
            "kept_column_count": int(self.n),
            "dropped_column_count": int(self.source_expansion.n - self.n),
            "dropped_zero_signature": bool(self.dropped_zero_signature),
            "dropped_zero_signature_multiplicity": int(self.dropped_zero_signature_multiplicity),
            "detector_edge_count": int(self.n_edges),
            "block_edge_counts": {str(k): int(v) for k, v in self.block_edge_counts.items()},
            "multiplicity_histogram": {int(k): int(v) for k, v in self.multiplicity_histogram.items()},
        }

    def block_metadata(self, label: str) -> CNOTLocationBlockMetadata:
        return self.source_problem.block_metadata(label)

    def split_detector_vector(self, values: np.ndarray) -> dict[str, np.ndarray]:
        return self.source_problem.split_detector_vector(values)

    def split_logical_action(self, values: np.ndarray) -> dict[str, np.ndarray]:
        return self.source_problem.split_logical_action(values)

    def detector_matrix(self):  # type: ignore[no-untyped-def]
        return self.context.graph.H

    def logical_matrix(self):  # type: ignore[no-untyped-def]
        return self.context.observables

    def write_matrix_bundle(self, out_dir: str | Path) -> Path:
        import scipy.sparse as sp  # type: ignore
        from scipy.io import mmwrite  # type: ignore

        dest = Path(out_dir).resolve()
        dest.mkdir(parents=True, exist_ok=True)

        detector_matrix = self.detector_matrix().tocoo()
        logical_matrix = self.logical_matrix().tocoo()
        sp.save_npz(dest / "detector_matrix.npz", detector_matrix.tocsr())
        sp.save_npz(dest / "logical_matrix.npz", logical_matrix.tocsr())
        mmwrite(str(dest / "detector_matrix.mtx"), detector_matrix)
        mmwrite(str(dest / "logical_matrix.mtx"), logical_matrix)

        with (dest / "merged_column_index.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "column",
                    "multiplicity",
                    "prior",
                    "detector_weight",
                    "logical_weight",
                    "representative_source_column",
                    "representative_location_index",
                    "representative_state",
                    "representative_state_label",
                ],
            )
            writer.writeheader()
            for column in range(self.n):
                source_column = int(self.representative_source_column[column])
                state = int(self.source_expansion.branch_state[source_column])
                location_index = int(self.source_expansion.branch_location_index[source_column])
                writer.writerow(
                    {
                        "column": int(column),
                        "multiplicity": int(self.merged_column_multiplicity[column]),
                        "prior": float(self.context.priors[column]),
                        "detector_weight": int(self.merged_column_detector_weight[column]),
                        "logical_weight": int(self.merged_column_logical_weight[column]),
                        "representative_source_column": int(source_column),
                        "representative_location_index": int(location_index),
                        "representative_state": int(state),
                        "representative_state_label": str(STATE_LABELS[state]),
                    }
                )

        bundle_metadata = {
            "model_label": "non-default reduced Gross CNOT-location merged 15-branch Bernoulli expansion",
            "source_model_label": "non-default reduced Gross CNOT-location 15-branch Bernoulli expansion",
            "detector_matrix_shape": [int(self.m), int(self.n)],
            "logical_matrix_shape": [int(self.k), int(self.n)],
            "detector_edge_count": int(self.n_edges),
            "block_edge_counts": {str(k): int(v) for k, v in self.block_edge_counts.items()},
            "original_column_count": int(self.source_expansion.n),
            "unique_signature_count": int(self.unique_signature_count),
            "kept_column_count": int(self.n),
            "dropped_column_count": int(self.source_expansion.n - self.n),
            "dropped_zero_signature": bool(self.dropped_zero_signature),
            "dropped_zero_signature_multiplicity": int(self.dropped_zero_signature_multiplicity),
            "multiplicity_histogram": {str(k): int(v) for k, v in self.multiplicity_histogram.items()},
            "notes": [
                "Columns are exact detector-plus-logical duplicate classes of the 15-branch Bernoulli expansion.",
                "Each merged prior is the odd-parity probability of the original duplicate class: (1 - prod_i (1 - 2 p_i)) / 2.",
                "The disconnected all-zero detector-plus-logical class is dropped by default because it has no effect on decoding.",
                "This remains a surrogate Bernoulli model and still does not restore same-location one-of-16 exclusivity.",
            ],
        }
        (dest / "metadata.json").write_text(json.dumps(bundle_metadata, indent=2) + "\n")
        report_lines = [
            "# Reduced Gross CNOT-location merged Bernoulli bundle",
            "",
            "- Model: non-default reduced Gross detector-side duplicate-collapsed Bernoulli expansion.",
            f"- Detector matrix: `{self.m} x {self.n}` with `{self.n_edges}` binary detector-variable edges.",
            f"- Logical matrix: `{self.k} x {self.n}`.",
            f"- Original Bernoulli columns: `{int(self.source_expansion.n)}`.",
            f"- Unique detector-plus-logical signatures before dropping the all-zero class: `{int(self.unique_signature_count)}`.",
            f"- Kept merged columns: `{int(self.n)}`.",
            f"- Dropped all-zero class multiplicity: `{int(self.dropped_zero_signature_multiplicity)}`.",
            "",
            "## Files",
            "",
            "- `detector_matrix.mtx` / `detector_matrix.npz`: merged binary detector matrix.",
            "- `logical_matrix.mtx` / `logical_matrix.npz`: merged binary logical matrix.",
            "- `merged_column_index.csv`: per-column multiplicity, prior, and representative source branch metadata.",
            "- `metadata.json`: machine-readable merge summary.",
        ]
        (dest / "report.md").write_text("\n".join(report_lines) + "\n")
        return dest


def _convert_flat_target(target: object) -> object:
    import stim  # type: ignore

    if isinstance(target, int):
        return int(target)
    if isinstance(target, tuple) and target and target[0] == "rec":
        return stim.target_rec(int(target[1]))
    raise ValueError(f"unsupported flattened target: {target!r}")


def _reduced_cnot_only_circuit(spec: ResolvedBackendCircuit) -> tuple[object, np.ndarray, np.ndarray, np.ndarray]:
    import stim  # type: ignore

    circuit = stim.Circuit.from_file(str(spec.stim_path))
    reduced = stim.Circuit()
    allowed_deterministic = {
        "CX",
        "CZ",
        "CY",
        "XCX",
        "XCY",
        "XCZ",
        "YCX",
        "YCY",
        "YCZ",
        "H",
        "S",
        "SQRT_X",
        "SQRT_Y",
        "SQRT_Z",
        "R",
        "RX",
        "RY",
        "M",
        "MX",
        "MY",
        "TICK",
        "DETECTOR",
        "OBSERVABLE_INCLUDE",
        "SHIFT_COORDS",
        "QUBIT_COORDS",
    }
    for name, targets, arg in circuit.flattened_operations():
        if name == "DEPOLARIZE1":
            continue
        if name == "DEPOLARIZE2":
            reduced.append(name, [_convert_flat_target(t) for t in targets], arg)
            continue
        if name in {"M", "MX", "MY", "R", "RX", "RY"}:
            reduced.append(name, [_convert_flat_target(t) for t in targets])
            continue
        if name in allowed_deterministic:
            reduced.append(name, [_convert_flat_target(t) for t in targets], arg if name in {"DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "QUBIT_COORDS"} else None)
            continue
        raise ValueError(
            f"unsupported instruction while building reduced CNOT-only circuit for {spec.stim_path}: {name}"
        )

    instruction_offsets: list[int] = []
    tick_offsets: list[int] = []
    qubit_pairs: list[tuple[int, int]] = []
    tick_count = 0
    flattened = list(reduced.flattened_operations())
    for instruction_offset, (name, targets, _arg) in enumerate(flattened):
        if name == "TICK":
            tick_count += 1
            continue
        if name != "DEPOLARIZE2":
            continue
        for k in range(0, len(targets), 2):
            instruction_offsets.append(int(instruction_offset))
            tick_offsets.append(int(tick_count))
            qubit_pairs.append((int(targets[k]), int(targets[k + 1])))

    return (
        reduced,
        np.asarray(instruction_offsets, dtype=np.int32),
        np.asarray(tick_offsets, dtype=np.int32),
        np.asarray(qubit_pairs, dtype=np.int32),
    )


def _dem_terms_to_indices(terms: Sequence[object]) -> tuple[np.ndarray, np.ndarray]:
    detector_indices: list[int] = []
    logical_indices: list[int] = []
    for term in terms:
        dem_target = term.dem_target
        if dem_target.is_relative_detector_id():
            detector_indices.append(int(dem_target.val))
        elif dem_target.is_logical_observable_id():
            logical_indices.append(int(dem_target.val))
    return (
        np.asarray(detector_indices, dtype=np.int32),
        np.asarray(logical_indices, dtype=np.int32),
    )


def _location_and_state_bits(location: object) -> tuple[int, int, int, int]:
    inst = location.instruction_targets
    pair_targets = inst.targets_in_range
    q0 = int(pair_targets[0].gate_target.value)
    q1 = int(pair_targets[1].gate_target.value)
    bits = 0
    for target_with_coords in location.flipped_pauli_product:
        gate_target = target_with_coords.gate_target
        qubit = int(gate_target.qubit_value)
        if qubit == q0:
            if gate_target.is_x_target or gate_target.is_y_target:
                bits ^= 1
            if gate_target.is_z_target or gate_target.is_y_target:
                bits ^= 2
        elif qubit == q1:
            if gate_target.is_x_target or gate_target.is_y_target:
                bits ^= 4
            if gate_target.is_z_target or gate_target.is_y_target:
                bits ^= 8
        else:
            raise ValueError(f"unexpected qubit in flipped_pauli_product: {location}")
    return int(location.stack_frames[-1].instruction_offset), q0, q1, int(bits)


@lru_cache(maxsize=8)
def _build_side_problem_cached(root_text: str, backend: str, sector: str, error_rate: float) -> CNOTLocationSideProblem:
    spec = resolve_backend_circuit(
        backend=str(backend),
        sector=str(sector),
        error_rate=float(error_rate),
        qtanner_root=root_text,
    )
    reduced, instruction_offsets, tick_offsets, qubit_pairs = _reduced_cnot_only_circuit(spec)
    num_locations = int(qubit_pairs.shape[0])
    location_lookup = {
        (int(instruction_offsets[idx]), int(qubit_pairs[idx, 0]), int(qubit_pairs[idx, 1])): int(idx)
        for idx in range(num_locations)
    }

    dem = reduced.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)
    explained = reduced.explain_detector_error_model_errors(dem_filter=dem, reduce_to_one_representative_error=False)

    detector_masks_by_location: list[dict[int, int]] = [dict() for _ in range(num_locations)]
    logical_masks_by_location: list[dict[int, int]] = [dict() for _ in range(num_locations)]
    visible_state_bitmasks = np.zeros(num_locations, dtype=np.uint16)

    for explained_error in explained:
        detector_indices, logical_indices = _dem_terms_to_indices(explained_error.dem_error_terms)
        for location in explained_error.circuit_error_locations:
            instruction_offset, q0, q1, bits = _location_and_state_bits(location)
            location_index = location_lookup[(instruction_offset, q0, q1)]
            if bits == 0:
                continue
            visible_state_bitmasks[location_index] |= np.uint16(1 << int(bits))
            if bits not in _BASIS_STATE_BITS:
                continue
            for detector_index in detector_indices.tolist():
                detector_masks_by_location[location_index][int(detector_index)] = (
                    int(detector_masks_by_location[location_index].get(int(detector_index), 0)) | int(bits)
                )
            for logical_index in logical_indices.tolist():
                logical_masks_by_location[location_index][int(logical_index)] = (
                    int(logical_masks_by_location[location_index].get(int(logical_index), 0)) | int(bits)
                )

    visible_nonidentity_state_counts = np.asarray(
        [_popcount(mask) for mask in visible_state_bitmasks.tolist()],
        dtype=np.int16,
    )
    metadata = CNOTLocationMetadata(
        backend=str(spec.backend),
        sector=f"memory_{spec.sector}",
        error_rate=float(error_rate),
        noisy_rounds=int(spec.noisy_rounds),
        total_rounds=int(spec.noisy_rounds + spec.perfect_rounds),
        stim_path=str(spec.stim_path),
        num_locations=num_locations,
        num_detectors=int(dem.num_detectors),
        num_observables=int(dem.num_observables),
        location_instruction_offsets=instruction_offsets,
        location_tick_offsets=tick_offsets,
        location_qubit_pairs=qubit_pairs,
        visible_nonidentity_state_bitmasks=np.asarray(visible_state_bitmasks, dtype=np.uint16),
        visible_nonidentity_state_counts=visible_nonidentity_state_counts,
        omitted_noise=(
            "DEPOLARIZE1 on data / ancilla / idle locations",
            "measurement noise on M / MX instructions",
            "reset-associated single-qubit noise after R / RX",
        ),
        notes=(
            "This is a non-default reduced Gross detector-side model built from the public bravyi_depth7 Stim circuit.",
            "Only post-CNOT DEPOLARIZE2 noise is retained; all reset, measurement, and single-qubit depolarizing noise is omitted.",
            "Variables are 16-state physical 2-qubit Pauli symbols per CNOT location, not merged DEM columns.",
        ),
    )
    return CNOTLocationSideProblem.from_local_masks(
        name=f"memory_{spec.sector}",
        error_rate=float(error_rate),
        detector_masks_by_location=detector_masks_by_location,
        logical_masks_by_location=logical_masks_by_location,
        metadata=metadata,
    )


def _combine_block_problems(blocks: Sequence[CNOTLocationSideProblem]) -> CNOTLocationProblem:
    problems = tuple(blocks)
    if len(problems) == 0:
        raise ValueError("expected at least one CNOT-location block problem")
    reference = problems[0]
    for other in problems[1:]:
        if int(other.n) != int(reference.n):
            raise ValueError("all CNOT-location blocks must share the same column count")
        if not np.array_equal(reference.metadata.location_tick_offsets, other.metadata.location_tick_offsets):
            raise ValueError("CNOT-location block tick ordering mismatch")
        if not np.array_equal(reference.metadata.location_qubit_pairs, other.metadata.location_qubit_pairs):
            raise ValueError("CNOT-location block qubit-pair ordering mismatch")
        if not np.allclose(reference.prior_state_probabilities, other.prior_state_probabilities):
            raise ValueError("CNOT-location block prior mismatch")

    detector_offset = 0
    logical_offset = 0
    edge_offset = 0
    logical_blocks: list[CNOTLocationBlockMetadata] = []
    edge_var_parts: list[np.ndarray] = []
    edge_check_parts: list[np.ndarray] = []
    edge_mask_parts: list[np.ndarray] = []
    logical_edge_var_parts: list[np.ndarray] = []
    logical_edge_observable_parts: list[np.ndarray] = []
    logical_edge_mask_parts: list[np.ndarray] = []
    check_to_edges: list[np.ndarray] = []
    var_to_edges: list[list[np.ndarray]] = [[] for _ in range(reference.n)]
    combined_visible_bitmasks = np.zeros(reference.n, dtype=np.uint16)

    for problem in problems:
        edge_var_parts.append(np.asarray(problem.edge_var, dtype=np.int32))
        edge_check_parts.append(np.asarray(problem.edge_check + detector_offset, dtype=np.int32))
        edge_mask_parts.append(np.asarray(problem.edge_mask, dtype=np.uint8))
        check_to_edges.extend(
            np.asarray(edges + edge_offset, dtype=np.int32)
            for edges in problem.check_to_edges
        )
        for var_index, edges in enumerate(problem.var_to_edges):
            var_to_edges[var_index].append(np.asarray(edges + edge_offset, dtype=np.int32))

        logical_edge_var_parts.append(np.asarray(problem.logical_edge_var, dtype=np.int32))
        logical_edge_observable_parts.append(np.asarray(problem.logical_edge_observable + logical_offset, dtype=np.int32))
        logical_edge_mask_parts.append(np.asarray(problem.logical_edge_mask, dtype=np.uint8))
        combined_visible_bitmasks |= np.asarray(problem.metadata.visible_nonidentity_state_bitmasks, dtype=np.uint16)
        logical_blocks.append(
            CNOTLocationBlockMetadata(
                label=str(problem.name),
                detector_offset=int(detector_offset),
                num_detectors=int(problem.m),
                observable_offset=int(logical_offset),
                num_observables=int(problem.k),
                stim_path=str(problem.metadata.stim_path),
                num_edges=int(problem.n_edges),
                visible_nonidentity_state_bitmasks=np.asarray(
                    problem.metadata.visible_nonidentity_state_bitmasks,
                    dtype=np.uint16,
                ),
                visible_nonidentity_state_counts=np.asarray(problem.metadata.visible_nonidentity_state_counts, dtype=np.int16),
            )
        )
        detector_offset += int(problem.m)
        logical_offset += int(problem.k)
        edge_offset += int(problem.n_edges)

    combined_visible_counts = np.asarray(
        [_popcount(mask) for mask in combined_visible_bitmasks.tolist()],
        dtype=np.int16,
    )
    combined_graph_metadata = CNOTLocationMetadata(
        backend=str(reference.metadata.backend),
        sector="memory_XZ",
        error_rate=float(reference.metadata.error_rate),
        noisy_rounds=int(reference.metadata.noisy_rounds),
        total_rounds=int(reference.metadata.total_rounds),
        stim_path="",
        num_locations=int(reference.n),
        num_detectors=int(detector_offset),
        num_observables=int(logical_offset),
        location_instruction_offsets=np.asarray(reference.metadata.location_instruction_offsets, dtype=np.int32),
        location_tick_offsets=np.asarray(reference.metadata.location_tick_offsets, dtype=np.int32),
        location_qubit_pairs=np.asarray(reference.metadata.location_qubit_pairs, dtype=np.int32),
        visible_nonidentity_state_bitmasks=combined_visible_bitmasks,
        visible_nonidentity_state_counts=combined_visible_counts,
        omitted_noise=tuple(reference.metadata.omitted_noise),
        notes=(
            "Unified row-stacked reduced Gross detector-side model with shared CNOT-location columns.",
            "Rows are formed by stacking the public memory_X and memory_Z reduced detector blocks while keeping one shared 16-state variable per CNOT location.",
        ),
    )
    graph = CNOTLocationSideProblem(
        name="memory_XZ",
        metadata=combined_graph_metadata,
        prior_state_probabilities=np.asarray(reference.prior_state_probabilities, dtype=np.float64),
        prior_state_log_probs=np.asarray(reference.prior_state_log_probs, dtype=np.float64),
        edge_var=np.concatenate(edge_var_parts).astype(np.int32, copy=False),
        edge_check=np.concatenate(edge_check_parts).astype(np.int32, copy=False),
        edge_mask=np.concatenate(edge_mask_parts).astype(np.uint8, copy=False),
        check_to_edges=tuple(check_to_edges),
        var_to_edges=tuple(
            np.concatenate(parts).astype(np.int32, copy=False) if parts else np.zeros(0, dtype=np.int32)
            for parts in var_to_edges
        ),
        logical_edge_var=np.concatenate(logical_edge_var_parts).astype(np.int32, copy=False),
        logical_edge_observable=np.concatenate(logical_edge_observable_parts).astype(np.int32, copy=False),
        logical_edge_mask=np.concatenate(logical_edge_mask_parts).astype(np.uint8, copy=False),
    )
    unified_metadata = CNOTLocationUnifiedMetadata(
        backend=str(reference.metadata.backend),
        error_rate=float(reference.metadata.error_rate),
        noisy_rounds=int(reference.metadata.noisy_rounds),
        total_rounds=int(reference.metadata.total_rounds),
        num_locations=int(reference.n),
        num_detectors=int(detector_offset),
        num_observables=int(logical_offset),
        location_instruction_offsets=np.asarray(reference.metadata.location_instruction_offsets, dtype=np.int32),
        location_tick_offsets=np.asarray(reference.metadata.location_tick_offsets, dtype=np.int32),
        location_qubit_pairs=np.asarray(reference.metadata.location_qubit_pairs, dtype=np.int32),
        visible_nonidentity_state_bitmasks=combined_visible_bitmasks,
        visible_nonidentity_state_counts=combined_visible_counts,
        blocks=tuple(logical_blocks),
        omitted_noise=tuple(reference.metadata.omitted_noise),
        notes=(
            "This is the native reduced Gross CNOT-location detector matrix with one shared column set of 10368 16-state variables.",
            "The detector rows are stacked as memory_X followed by memory_Z, giving 1872 detector rows in total on the public bravyi_depth7 path.",
            "Only post-CNOT DEPOLARIZE2 noise is retained; reset, measurement, and DEPOLARIZE1 noise remain omitted in this non-default model.",
        ),
    )
    return CNOTLocationProblem.from_block_problems(
        graph=graph,
        metadata=unified_metadata,
        block_problems=problems,
    )


def build_cnot_location_problem(
    *,
    backend: str = "bravyi_depth7",
    error_rate: float = 0.004,
    qtanner_root: str | Path | None = None,
) -> CNOTLocationProblem:
    root = resolve_qtanner_root(qtanner_root)
    x_side = _build_side_problem_cached(str(root), str(backend), "X", float(error_rate))
    z_side = _build_side_problem_cached(str(root), str(backend), "Z", float(error_rate))
    return _combine_block_problems((x_side, z_side))


@lru_cache(maxsize=8)
def _build_bernoulli_expansion_cached(
    root_text: str,
    backend: str,
    error_rate: float,
) -> CNOTLocationBernoulliExpansionProblem:
    return CNOTLocationBernoulliExpansionProblem.from_cnot_problem(
        build_cnot_location_problem(
            backend=str(backend),
            error_rate=float(error_rate),
            qtanner_root=root_text,
        )
    )


def build_cnot_location_bernoulli_expansion_problem(
    *,
    backend: str = "bravyi_depth7",
    error_rate: float = 0.004,
    qtanner_root: str | Path | None = None,
) -> CNOTLocationBernoulliExpansionProblem:
    root = resolve_qtanner_root(qtanner_root)
    return _build_bernoulli_expansion_cached(str(root), str(backend), float(error_rate))


@lru_cache(maxsize=8)
def _build_bernoulli_merged_cached(
    root_text: str,
    backend: str,
    error_rate: float,
    drop_zero_signature: bool,
) -> CNOTLocationMergedBernoulliProblem:
    return CNOTLocationMergedBernoulliProblem.from_expansion_problem(
        build_cnot_location_bernoulli_expansion_problem(
            backend=str(backend),
            error_rate=float(error_rate),
            qtanner_root=root_text,
        ),
        drop_zero_signature=bool(drop_zero_signature),
    )


def build_cnot_location_bernoulli_merged_problem(
    *,
    backend: str = "bravyi_depth7",
    error_rate: float = 0.004,
    qtanner_root: str | Path | None = None,
    drop_zero_signature: bool = True,
) -> CNOTLocationMergedBernoulliProblem:
    root = resolve_qtanner_root(qtanner_root)
    return _build_bernoulli_merged_cached(str(root), str(backend), float(error_rate), bool(drop_zero_signature))


@dataclass
class CNOTLocationMinSumDecoder:
    problem: CNOTLocationSideProblem | CNOTLocationProblem
    config: DecoderConfig = DecoderConfig()
    compression_mode: str = "max"

    def __post_init__(self) -> None:
        self.config.validate("minsum")
        if bool(self.config.self_corrected):
            raise NotImplementedError("self-corrected min-sum is not implemented for the CNOT-location decoder")
        mode = str(self.compression_mode).strip().lower()
        if mode not in {"max", "logsumexp"}:
            raise ValueError("compression_mode must be one of: max, logsumexp")
        self.compression_mode = mode

    def _compress_scores_to_llr(self, scores: np.ndarray, mask: int) -> float:
        if str(self.compression_mode) == "max":
            return _compress_scores_to_llr_max(scores, mask, float(self.config.llr_clip))
        return _compress_scores_to_llr_logsumexp(scores, mask, float(self.config.llr_clip))

    def _decode_flooding(self, target: np.ndarray) -> CNOTLocationDecodeResult:
        belief = self.problem.initial_log_scores()
        c2v = np.zeros(self.problem.n_edges, dtype=np.float64)
        converged = False
        hard = np.zeros(self.problem.n, dtype=np.uint8)
        residual = target.copy()
        iterations = 0
        for it in range(1, int(self.config.max_iter) + 1):
            iterations = int(it)
            v2c = np.zeros(self.problem.n_edges, dtype=np.float64)
            for edge_index in range(self.problem.n_edges):
                var = int(self.problem.edge_var[edge_index])
                mask = int(self.problem.edge_mask[edge_index])
                scores_excluding = belief[var] - _binary_contribution(mask, c2v[edge_index])
                v2c[edge_index] = self._compress_scores_to_llr(scores_excluding, mask)

            new_c2v = c2v.copy()
            for check_index, edges in enumerate(self.problem.check_to_edges):
                if edges.size == 0:
                    continue
                new_c2v[edges] = _check_update_minsum_binary(
                    v2c[edges],
                    int(target[check_index]),
                    c2v[edges],
                    normalization=float(self.config.normalization),
                    offset=float(self.config.offset),
                    damping=float(self.config.damping),
                    llr_clip=float(self.config.llr_clip),
                )

            belief = self.problem.initial_log_scores()
            for edge_index in range(self.problem.n_edges):
                var = int(self.problem.edge_var[edge_index])
                mask = int(self.problem.edge_mask[edge_index])
                belief[var] += _binary_contribution(mask, new_c2v[edge_index])
            belief -= np.max(belief, axis=1, keepdims=True)
            c2v = new_c2v
            hard = np.asarray(np.argmax(belief, axis=1), dtype=np.uint8)
            residual = self.problem.syndrome_from_symbols(hard) ^ target
            if int(np.count_nonzero(residual)) == 0:
                converged = True
                break
        return CNOTLocationDecodeResult(
            estimate_symbols=hard,
            posterior_log_scores=belief,
            converged=bool(converged),
            iterations=int(iterations),
            edge_updates=int(iterations * self.problem.n_edges),
            unsatisfied_checks=int(np.count_nonzero(residual)),
            unsatisfied_vector=np.asarray(residual, dtype=np.uint8),
            logical_action=self.problem.logical_action_from_symbols(hard),
        )

    def _decode_layered(self, target: np.ndarray) -> CNOTLocationDecodeResult:
        belief = self.problem.initial_log_scores()
        c2v = np.zeros(self.problem.n_edges, dtype=np.float64)
        converged = False
        hard = np.zeros(self.problem.n, dtype=np.uint8)
        residual = target.copy()
        iterations = 0
        for it in range(1, int(self.config.max_iter) + 1):
            iterations = int(it)
            for check_index, edges in enumerate(self.problem.check_to_edges):
                if edges.size == 0:
                    continue
                incoming = np.zeros(edges.size, dtype=np.float64)
                for local_edge_index, edge_index in enumerate(edges.tolist()):
                    var = int(self.problem.edge_var[edge_index])
                    mask = int(self.problem.edge_mask[edge_index])
                    scores_excluding = belief[var] - _binary_contribution(mask, c2v[edge_index])
                    incoming[local_edge_index] = self._compress_scores_to_llr(scores_excluding, mask)
                new_c2v = _check_update_minsum_binary(
                    incoming,
                    int(target[check_index]),
                    c2v[edges],
                    normalization=float(self.config.normalization),
                    offset=float(self.config.offset),
                    damping=float(self.config.damping),
                    llr_clip=float(self.config.llr_clip),
                )
                delta = np.asarray(new_c2v - c2v[edges], dtype=np.float64)
                c2v[edges] = new_c2v
                for local_edge_index, edge_index in enumerate(edges.tolist()):
                    if delta[local_edge_index] == 0.0:
                        continue
                    var = int(self.problem.edge_var[edge_index])
                    mask = int(self.problem.edge_mask[edge_index])
                    belief[var] += _binary_contribution(mask, delta[local_edge_index])
            belief -= np.max(belief, axis=1, keepdims=True)
            hard = np.asarray(np.argmax(belief, axis=1), dtype=np.uint8)
            residual = self.problem.syndrome_from_symbols(hard) ^ target
            if int(np.count_nonzero(residual)) == 0:
                converged = True
                break
        return CNOTLocationDecodeResult(
            estimate_symbols=hard,
            posterior_log_scores=belief,
            converged=bool(converged),
            iterations=int(iterations),
            edge_updates=int(iterations * self.problem.n_edges),
            unsatisfied_checks=int(np.count_nonzero(residual)),
            unsatisfied_vector=np.asarray(residual, dtype=np.uint8),
            logical_action=self.problem.logical_action_from_symbols(hard),
        )

    def decode(self, syndrome: np.ndarray) -> CNOTLocationDecodeResult:
        target = np.asarray(syndrome, dtype=np.uint8).reshape(-1) & 1
        if int(target.size) != self.problem.m:
            raise ValueError(f"syndrome length mismatch: got {target.size}, expected {self.problem.m}")
        schedule = self.config.normalized_schedule()
        if schedule == "flooding":
            return self._decode_flooding(target)
        if schedule == "layered":
            return self._decode_layered(target)
        raise ValueError(f"unsupported schedule for CNOT-location min-sum: {self.config.schedule}")


@dataclass
class CNOTLocationSplitMinSumDecoder:
    problem: CNOTLocationProblem
    config: DecoderConfig = DecoderConfig()
    compression_mode: str = "max"

    def __post_init__(self) -> None:
        self.decoder = CNOTLocationMinSumDecoder(self.problem, self.config, compression_mode=self.compression_mode)

    def decode(self, syndrome: CNOTLocationSplitSyndrome) -> CNOTLocationSplitDecodeResult:
        full_syndrome = self.problem.combine_detector_blocks(
            {
                "memory_X": np.asarray(syndrome.X, dtype=np.uint8),
                "memory_Z": np.asarray(syndrome.Z, dtype=np.uint8),
            }
        )
        full_result = self.decoder.decode(full_syndrome)
        residual_blocks = self.problem.split_detector_vector(full_result.unsatisfied_vector)
        logical_blocks = self.problem.split_logical_action(full_result.logical_action)
        x_block = self.problem.block_metadata("memory_X")
        z_block = self.problem.block_metadata("memory_Z")
        x_result = CNOTLocationDecodeResult(
            estimate_symbols=np.asarray(full_result.estimate_symbols, dtype=np.uint8),
            posterior_log_scores=np.asarray(full_result.posterior_log_scores, dtype=np.float64),
            converged=bool(np.count_nonzero(residual_blocks["memory_X"]) == 0),
            iterations=int(full_result.iterations),
            edge_updates=int(full_result.iterations * int(x_block.num_edges)),
            unsatisfied_checks=int(np.count_nonzero(residual_blocks["memory_X"])),
            unsatisfied_vector=np.asarray(residual_blocks["memory_X"], dtype=np.uint8),
            logical_action=np.asarray(logical_blocks["memory_X"], dtype=np.uint8),
        )
        z_result = CNOTLocationDecodeResult(
            estimate_symbols=np.asarray(full_result.estimate_symbols, dtype=np.uint8),
            posterior_log_scores=np.asarray(full_result.posterior_log_scores, dtype=np.float64),
            converged=bool(np.count_nonzero(residual_blocks["memory_Z"]) == 0),
            iterations=int(full_result.iterations),
            edge_updates=int(full_result.iterations * int(z_block.num_edges)),
            unsatisfied_checks=int(np.count_nonzero(residual_blocks["memory_Z"])),
            unsatisfied_vector=np.asarray(residual_blocks["memory_Z"], dtype=np.uint8),
            logical_action=np.asarray(logical_blocks["memory_Z"], dtype=np.uint8),
        )
        return CNOTLocationSplitDecodeResult(
            X=x_result,
            Z=z_result,
            converged=bool(full_result.converged),
            logical_frame_action={
                "x": np.asarray(logical_blocks["memory_X"], dtype=np.uint8),
                "z": np.asarray(logical_blocks["memory_Z"], dtype=np.uint8),
            },
            unsatisfied_checks={
                "x": int(x_result.unsatisfied_checks),
                "z": int(z_result.unsatisfied_checks),
            },
            iterations={
                "x": int(full_result.iterations),
                "z": int(full_result.iterations),
            },
            edge_updates={
                "x": int(x_result.edge_updates),
                "z": int(z_result.edge_updates),
            },
        )


CNOTLocationSideDecodeResult = CNOTLocationDecodeResult


__all__ = [
    "CNOTLocationBernoulliExpansionProblem",
    "CNOTLocationBlockMetadata",
    "CNOTLocationDecodeResult",
    "CNOTLocationMergedBernoulliProblem",
    "CNOTLocationMetadata",
    "CNOTLocationMinSumDecoder",
    "CNOTLocationProblem",
    "CNOTLocationSideDecodeResult",
    "CNOTLocationSideProblem",
    "CNOTLocationSplitDecodeResult",
    "CNOTLocationSplitMinSumDecoder",
    "CNOTLocationSplitSyndrome",
    "CNOTLocationUnifiedMetadata",
    "STATE_LABELS",
    "build_cnot_location_bernoulli_merged_problem",
    "build_cnot_location_bernoulli_expansion_problem",
    "build_cnot_location_problem",
]
