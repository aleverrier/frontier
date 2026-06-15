from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "frontier_mplconfig"))

from grosscode.nonbinary_cnot import CNOTLocationSideProblem, build_cnot_location_problem
from grosscode.utils.paths import resolve_qtanner_root


PROJECTED_STATE_LABELS_X: tuple[str, ...] = ("II", "XI", "IX", "XX")
PROJECTED_STATE_LABELS_Z: tuple[str, ...] = ("II", "ZI", "IZ", "ZZ")

_MASK_PARITY_TABLE = np.asarray(
    [[((int(mask) & int(state)).bit_count() & 1) for state in range(4)] for mask in range(4)],
    dtype=np.uint8,
)
_MASK_STATE_SPLIT = tuple(
    (
        np.flatnonzero(_MASK_PARITY_TABLE[mask] == 0).astype(np.int8, copy=False),
        np.flatnonzero(_MASK_PARITY_TABLE[mask] == 1).astype(np.int8, copy=False),
    )
    for mask in range(4)
)
_FULL_MASK_TO_PROJECTION = {
    (1, 4): "x",
    (2, 8): "z",
}


def masked_parity(mask: int, state: int) -> int:
    mask_int = int(mask)
    state_int = int(state)
    if not (1 <= mask_int <= 3):
        raise ValueError(f"mask must lie in {{1, 2, 3}}, got {mask}")
    if not (0 <= state_int <= 3):
        raise ValueError(f"state must lie in {{0, 1, 2, 3}}, got {state}")
    return int(_MASK_PARITY_TABLE[mask_int, state_int])


def _row_logsumexp4(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or int(arr.shape[1]) != 4:
        raise ValueError(f"expected shape (n, 4), got {arr.shape}")
    ab = np.logaddexp(arr[:, 0], arr[:, 1])
    cd = np.logaddexp(arr[:, 2], arr[:, 3])
    return np.logaddexp(ab, cd)


def _normalize_log_probs(log_probs: np.ndarray) -> np.ndarray:
    arr = np.asarray(log_probs, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False
    if int(arr.shape[1]) != 4:
        raise ValueError(f"expected shape (n, 4), got {arr.shape}")
    norm = _row_logsumexp4(arr)
    if np.any(~np.isfinite(norm)):
        raise FloatingPointError("log-probability normalization encountered an all-zero row")
    out = arr - norm[:, None]
    return out[0] if squeeze else out


def _probabilities_to_log_probs(priors: np.ndarray) -> np.ndarray:
    arr = np.asarray(priors, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False
    if int(arr.shape[1]) != 4:
        raise ValueError(f"expected shape (n, 4), got {arr.shape}")
    if np.any(arr < 0.0):
        raise ValueError("probabilities must be non-negative")
    totals = np.sum(arr, axis=1)
    if np.any(totals <= 0.0):
        raise ValueError("every probability row must have positive mass")
    normalized = arr / totals[:, None]
    out = np.where(normalized > 0.0, np.log(normalized), -np.inf).astype(np.float64)
    return out[0] if squeeze else out


def _broadcast_log_priors(log_priors: np.ndarray, num_vars: int) -> np.ndarray:
    arr = np.asarray(log_priors, dtype=np.float64)
    if arr.ndim == 1:
        if int(arr.size) != 4:
            raise ValueError(f"expected 4 prior states, got {arr.size}")
        return np.broadcast_to(arr.reshape(1, 4), (int(num_vars), 4)).copy()
    if arr.ndim == 2 and int(arr.shape[1]) == 4 and int(arr.shape[0]) == int(num_vars):
        return arr.copy()
    raise ValueError(f"expected log priors of shape (4,) or ({int(num_vars)}, 4), got {arr.shape}")


def _sample_rows(priors: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    probs = np.asarray(priors, dtype=np.float64)
    if probs.ndim != 2 or int(probs.shape[1]) != 4:
        raise ValueError(f"expected shape (n, 4), got {probs.shape}")
    cumulative = np.cumsum(probs, axis=1)
    draws = rng.random((int(probs.shape[0]), 1))
    return np.sum(draws > cumulative, axis=1, dtype=np.uint8).astype(np.uint8, copy=False)


def _detect_active_bits(masks: Sequence[int]) -> tuple[int, int]:
    active = sorted({int(bit) for mask in masks for bit in (1, 2, 4, 8) if int(mask) & int(bit)})
    if len(active) != 2:
        raise ValueError(f"expected exactly two active full-state bits, found {active}")
    low_bit, high_bit = int(active[0]), int(active[1])
    expected = {low_bit, high_bit, low_bit | high_bit}
    observed = {int(mask) for mask in masks if int(mask) != 0}
    if not observed.issubset(expected):
        raise ValueError(f"unexpected full masks {sorted(observed)} for active bits {(low_bit, high_bit)}")
    return low_bit, high_bit


def _project_full_mask(mask: int, low_bit: int, high_bit: int) -> int:
    mask_int = int(mask)
    if mask_int == 0:
        return 0
    projected = 0
    if mask_int & int(low_bit):
        projected |= 1
    if mask_int & int(high_bit):
        projected |= 2
    return int(projected)


def _project_full_state(full_state: int, low_bit: int, high_bit: int) -> int:
    state = 0
    if int(full_state) & int(low_bit):
        state |= 1
    if int(full_state) & int(high_bit):
        state |= 2
    return int(state)


def _project_full_state_probabilities(
    full_probs: np.ndarray,
    *,
    low_bit: int,
    high_bit: int,
    num_vars: int,
) -> np.ndarray:
    arr = np.asarray(full_probs, dtype=np.float64)
    if arr.ndim == 1:
        if int(arr.size) != 16:
            raise ValueError(f"expected 16 full-state priors, got {arr.size}")
        arr = np.broadcast_to(arr.reshape(1, 16), (int(num_vars), 16))
    elif arr.ndim == 2 and int(arr.shape[1]) == 16 and int(arr.shape[0]) == int(num_vars):
        pass
    else:
        raise ValueError(f"expected full-state priors of shape (16,) or ({int(num_vars)}, 16), got {arr.shape}")
    out = np.zeros((int(num_vars), 4), dtype=np.float64)
    for full_state in range(16):
        out[:, _project_full_state(full_state, low_bit, high_bit)] += arr[:, full_state]
    return out


def _lift_log_message(mask: int, llr: float) -> np.ndarray:
    mask_int = int(mask)
    if not (1 <= mask_int <= 3):
        raise ValueError(f"mask must lie in {{1, 2, 3}}, got {mask}")
    log_r0 = -float(np.logaddexp(0.0, -float(llr)))
    log_r1 = -float(np.logaddexp(0.0, float(llr)))
    out = np.empty(4, dtype=np.float64)
    zeros, ones = _MASK_STATE_SPLIT[mask_int]
    out[np.asarray(zeros, dtype=np.intp)] = float(log_r0)
    out[np.asarray(ones, dtype=np.intp)] = float(log_r1)
    return out


def _project_log_message(log_message: np.ndarray, mask: int) -> tuple[float, float]:
    arr = np.asarray(log_message, dtype=np.float64).reshape(4)
    zeros, ones = _MASK_STATE_SPLIT[int(mask)]
    log_q0 = float(np.logaddexp(arr[int(zeros[0])], arr[int(zeros[1])]))
    log_q1 = float(np.logaddexp(arr[int(ones[0])], arr[int(ones[1])]))
    if not np.isfinite(log_q0) and not np.isfinite(log_q1):
        raise FloatingPointError("projected variable-to-check message has zero mass in both parity classes")
    if log_q0 == -np.inf:
        llr = -np.inf
    elif log_q1 == -np.inf:
        llr = np.inf
    else:
        llr = float(log_q0 - log_q1)
    if np.isfinite(llr):
        bias = float(math.tanh(0.5 * llr))
    else:
        bias = 1.0 if llr > 0.0 else -1.0
    return float(llr), float(bias)


def _damp_log_message(
    candidate_log_message: np.ndarray,
    previous_log_message: np.ndarray,
    damping: float,
) -> np.ndarray:
    candidate = _normalize_log_probs(np.asarray(candidate_log_message, dtype=np.float64))
    if float(damping) <= 0.0:
        return candidate
    previous = _normalize_log_probs(np.asarray(previous_log_message, dtype=np.float64))
    keep = float(np.clip(damping, 0.0, 1.0))
    if keep >= 1.0:
        return previous
    log_keep = float(math.log(keep))
    log_fresh = float(math.log1p(-keep))
    mixed = np.logaddexp(log_fresh + candidate, log_keep + previous)
    return _normalize_log_probs(mixed)


def _check_update_from_biases(
    incoming_biases: np.ndarray,
    syndrome_bit: int,
    *,
    atanh_clip_eps: float,
) -> np.ndarray:
    biases = np.asarray(incoming_biases, dtype=np.float64).reshape(-1)
    degree = int(biases.size)
    if degree == 0:
        return np.zeros(0, dtype=np.float64)
    clipped_biases = np.clip(biases, -1.0, 1.0)
    abs_vals = np.abs(clipped_biases)
    zero_mask = abs_vals == 0.0
    total_zero_count = int(np.count_nonzero(zero_mask))
    safe_abs = np.where(zero_mask, 1.0, abs_vals)
    total_logabs = float(np.sum(np.log(safe_abs)))
    signs = np.where(clipped_biases < 0.0, -1.0, 1.0)
    total_sign = float(np.prod(np.where(zero_mask, 1.0, signs)))
    parity_sign = -1.0 if (int(syndrome_bit) & 1) else 1.0
    out = np.empty(degree, dtype=np.float64)
    for idx in range(degree):
        zero_count_except = total_zero_count - (1 if bool(zero_mask[idx]) else 0)
        if zero_count_except > 0:
            delta = 0.0
        else:
            sign_except = parity_sign * total_sign * (1.0 if bool(zero_mask[idx]) else float(signs[idx]))
            logabs_except = total_logabs - (0.0 if bool(zero_mask[idx]) else float(math.log(float(abs_vals[idx]))))
            delta = float(sign_except * math.exp(float(logabs_except)))
        delta = float(np.clip(delta, -1.0 + float(atanh_clip_eps), 1.0 - float(atanh_clip_eps)))
        out[idx] = float(2.0 * math.atanh(delta))
    return out


def _entropy_from_log_probs(log_probs: np.ndarray) -> np.ndarray:
    probs = np.exp(np.asarray(log_probs, dtype=np.float64))
    safe_log = np.where(probs > 0.0, np.asarray(log_probs, dtype=np.float64), 0.0)
    return -np.sum(probs * safe_log, axis=1)


@dataclass(frozen=True)
class MaskedQuaternarySideProblem:
    name: str
    projection: str
    state_labels: tuple[str, ...]
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
        detector_masks_by_variable: Sequence[Mapping[int, int]],
        logical_masks_by_variable: Sequence[Mapping[int, int]] | None = None,
        prior_state_probabilities: np.ndarray | None = None,
        error_rate: float | None = None,
        projection: str = "x",
        state_labels: tuple[str, ...] | None = None,
    ) -> "MaskedQuaternarySideProblem":
        num_vars = int(len(detector_masks_by_variable))
        logical_maps = (
            tuple({int(k): int(v) for k, v in mapping.items()} for mapping in logical_masks_by_variable)
            if logical_masks_by_variable is not None
            else tuple({} for _ in range(num_vars))
        )
        if len(logical_maps) != int(num_vars):
            raise ValueError("logical_masks_by_variable length mismatch")

        num_detectors = 0
        num_logicals = 0
        for mapping in detector_masks_by_variable:
            for check_index, mask in mapping.items():
                mask_int = int(mask)
                if mask_int not in {1, 2, 3}:
                    raise ValueError(f"detector mask must lie in {{1, 2, 3}}, got {mask}")
                num_detectors = max(num_detectors, int(check_index) + 1)
        for mapping in logical_maps:
            for logical_index, mask in mapping.items():
                mask_int = int(mask)
                if mask_int not in {1, 2, 3}:
                    raise ValueError(f"logical mask must lie in {{1, 2, 3}}, got {mask}")
                num_logicals = max(num_logicals, int(logical_index) + 1)

        edge_var: list[int] = []
        edge_check: list[int] = []
        edge_mask: list[int] = []
        check_to_edges: list[list[int]] = [[] for _ in range(num_detectors)]
        var_to_edges: list[list[int]] = [[] for _ in range(num_vars)]
        for var_index, mapping in enumerate(detector_masks_by_variable):
            for check_index in sorted(int(idx) for idx in mapping.keys()):
                mask_int = int(mapping[check_index])
                edge_index = len(edge_var)
                edge_var.append(int(var_index))
                edge_check.append(int(check_index))
                edge_mask.append(mask_int)
                check_to_edges[int(check_index)].append(int(edge_index))
                var_to_edges[int(var_index)].append(int(edge_index))

        logical_edge_var: list[int] = []
        logical_edge_observable: list[int] = []
        logical_edge_mask: list[int] = []
        for var_index, mapping in enumerate(logical_maps):
            for logical_index in sorted(int(idx) for idx in mapping.keys()):
                logical_edge_var.append(int(var_index))
                logical_edge_observable.append(int(logical_index))
                logical_edge_mask.append(int(mapping[logical_index]))

        projection_norm = str(projection).strip().lower()
        labels = (
            PROJECTED_STATE_LABELS_X
            if projection_norm == "x"
            else PROJECTED_STATE_LABELS_Z
            if projection_norm == "z"
            else tuple(state_labels or PROJECTED_STATE_LABELS_X)
        )
        if state_labels is not None:
            labels = tuple(str(item) for item in state_labels)
        if int(len(labels)) != 4:
            raise ValueError("state_labels must contain exactly four labels")

        if prior_state_probabilities is None:
            p = 0.0 if error_rate is None else float(error_rate)
            if not (0.0 <= p <= 1.0):
                raise ValueError("error_rate must lie in [0, 1]")
            base = np.asarray([1.0 - p, p / 3.0, p / 3.0, p / 3.0], dtype=np.float64)
            priors = np.broadcast_to(base.reshape(1, 4), (num_vars, 4)).copy()
        else:
            priors = np.asarray(prior_state_probabilities, dtype=np.float64)
            if priors.ndim == 1:
                if int(priors.size) != 4:
                    raise ValueError(f"expected 4 projected prior states, got {priors.size}")
                priors = np.broadcast_to(priors.reshape(1, 4), (num_vars, 4)).copy()
            elif priors.ndim == 2 and int(priors.shape[0]) == num_vars and int(priors.shape[1]) == 4:
                priors = priors.copy()
            else:
                raise ValueError(f"expected priors of shape (4,) or ({num_vars}, 4), got {priors.shape}")
        prior_log_probs = _broadcast_log_priors(_probabilities_to_log_probs(priors), num_vars)
        normalized_priors = np.exp(prior_log_probs)

        return cls(
            name=str(name),
            projection=projection_norm,
            state_labels=tuple(labels),
            prior_state_probabilities=np.asarray(normalized_priors, dtype=np.float64),
            prior_state_log_probs=np.asarray(prior_log_probs, dtype=np.float64),
            edge_var=np.asarray(edge_var, dtype=np.int32),
            edge_check=np.asarray(edge_check, dtype=np.int32),
            edge_mask=np.asarray(edge_mask, dtype=np.uint8),
            check_to_edges=tuple(np.asarray(edges, dtype=np.int32) for edges in check_to_edges),
            var_to_edges=tuple(np.asarray(edges, dtype=np.int32) for edges in var_to_edges),
            logical_edge_var=np.asarray(logical_edge_var, dtype=np.int32),
            logical_edge_observable=np.asarray(logical_edge_observable, dtype=np.int32),
            logical_edge_mask=np.asarray(logical_edge_mask, dtype=np.uint8),
        )

    @classmethod
    def from_cnot_side_problem(
        cls,
        source: CNOTLocationSideProblem,
        *,
        projection: str | None = None,
        prior_state_probabilities: np.ndarray | None = None,
    ) -> "MaskedQuaternarySideProblem":
        observed_masks = tuple(
            int(mask)
            for mask in np.concatenate(
                (
                    np.asarray(source.edge_mask, dtype=np.uint8),
                    np.asarray(source.logical_edge_mask, dtype=np.uint8),
                )
            ).tolist()
            if int(mask) != 0
        )
        low_bit, high_bit = _detect_active_bits(observed_masks)
        inferred_projection = _FULL_MASK_TO_PROJECTION.get((int(low_bit), int(high_bit)), "projected")
        projection_norm = inferred_projection if projection is None else str(projection).strip().lower()
        if projection_norm == "x":
            labels = PROJECTED_STATE_LABELS_X
        elif projection_norm == "z":
            labels = PROJECTED_STATE_LABELS_Z
        else:
            labels = tuple(f"s{idx}" for idx in range(4))

        if prior_state_probabilities is None:
            projected_priors = _project_full_state_probabilities(
                source.prior_state_probabilities,
                low_bit=int(low_bit),
                high_bit=int(high_bit),
                num_vars=int(source.n),
            )
        else:
            projected_priors = np.asarray(prior_state_probabilities, dtype=np.float64)

        logical_maps: list[dict[int, int]] = [dict() for _ in range(int(source.n))]
        for edge_index in range(int(source.logical_edge_var.size)):
            var_index = int(source.logical_edge_var[edge_index])
            projected_mask = _project_full_mask(
                int(source.logical_edge_mask[edge_index]),
                int(low_bit),
                int(high_bit),
            )
            if int(projected_mask) == 0:
                continue
            logical_maps[var_index][int(source.logical_edge_observable[edge_index])] = int(projected_mask)

        return cls.from_local_masks(
            name=str(source.name),
            detector_masks_by_variable=[
                {
                    int(source.edge_check[edge_index]): _project_full_mask(
                        int(source.edge_mask[edge_index]),
                        int(low_bit),
                        int(high_bit),
                    )
                    for edge_index in edges.tolist()
                }
                for edges in source.var_to_edges
            ],
            logical_masks_by_variable=logical_maps,
            prior_state_probabilities=projected_priors,
            projection=str(projection_norm),
            state_labels=tuple(labels),
        )

    @property
    def num_vars(self) -> int:
        return int(self.prior_state_probabilities.shape[0])

    @property
    def num_detectors(self) -> int:
        return int(len(self.check_to_edges))

    @property
    def num_logicals(self) -> int:
        return int(np.max(self.logical_edge_observable) + 1) if int(self.logical_edge_observable.size) else 0

    @property
    def n_edges(self) -> int:
        return int(self.edge_var.size)

    def resolve_log_priors(
        self,
        *,
        priors: np.ndarray | None = None,
        log_priors: np.ndarray | None = None,
    ) -> np.ndarray:
        if priors is not None and log_priors is not None:
            raise ValueError("provide at most one of priors or log_priors")
        if priors is None and log_priors is None:
            return np.asarray(self.prior_state_log_probs, dtype=np.float64).copy()
        if priors is not None:
            return _broadcast_log_priors(_probabilities_to_log_probs(np.asarray(priors, dtype=np.float64)), self.num_vars)
        return _broadcast_log_priors(np.asarray(log_priors, dtype=np.float64), self.num_vars)

    def syndrome_from_states(self, states: np.ndarray) -> np.ndarray:
        state_vec = np.asarray(states, dtype=np.uint8).reshape(-1)
        if int(state_vec.size) != self.num_vars:
            raise ValueError(f"state vector length mismatch: got {state_vec.size}, expected {self.num_vars}")
        out = np.zeros(self.num_detectors, dtype=np.uint8)
        if self.n_edges == 0:
            return out
        edge_bits = _MASK_PARITY_TABLE[self.edge_mask, state_vec[self.edge_var]].astype(np.uint8, copy=False)
        np.bitwise_xor.at(out, self.edge_check, edge_bits)
        return out

    def logical_action_from_states(self, states: np.ndarray) -> np.ndarray:
        state_vec = np.asarray(states, dtype=np.uint8).reshape(-1)
        if int(state_vec.size) != self.num_vars:
            raise ValueError(f"state vector length mismatch: got {state_vec.size}, expected {self.num_vars}")
        out = np.zeros(self.num_logicals, dtype=np.uint8)
        if int(self.logical_edge_var.size) == 0:
            return out
        logical_bits = _MASK_PARITY_TABLE[
            self.logical_edge_mask,
            state_vec[self.logical_edge_var],
        ].astype(np.uint8, copy=False)
        np.bitwise_xor.at(out, self.logical_edge_observable, logical_bits)
        return out

    def check_syndrome(self, states: np.ndarray, syndrome: np.ndarray) -> bool:
        target = np.asarray(syndrome, dtype=np.uint8).reshape(-1) & 1
        if int(target.size) != self.num_detectors:
            raise ValueError(f"syndrome length mismatch: got {target.size}, expected {self.num_detectors}")
        return bool(np.array_equal(self.syndrome_from_states(states), target))

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        states = _sample_rows(self.prior_state_probabilities, rng)
        return states, self.syndrome_from_states(states), self.logical_action_from_states(states)

    def detector_mask_matrix(self):  # type: ignore[no-untyped-def]
        import scipy.sparse as sp  # type: ignore

        return sp.csr_matrix(
            (
                np.asarray(self.edge_mask, dtype=np.uint8),
                (
                    np.asarray(self.edge_check, dtype=np.int32),
                    np.asarray(self.edge_var, dtype=np.int32),
                ),
            ),
            shape=(self.num_detectors, self.num_vars),
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
            shape=(self.num_logicals, self.num_vars),
            dtype=np.uint8,
        )

    def duplicate_column_summary(self, *, include_logicals: bool = True) -> dict[str, int | bool]:
        detector = self.detector_mask_matrix().tocsc()
        logical = self.logical_mask_matrix().tocsc() if include_logicals else None
        signatures: dict[tuple[object, ...], list[int]] = {}
        for col in range(self.num_vars):
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
            signatures.setdefault(signature, []).append(int(col))
        groups = [cols for cols in signatures.values() if len(cols) > 1]
        extra_column_count = int(sum(len(cols) - 1 for cols in groups))
        return {
            "include_logicals": bool(include_logicals),
            "duplicate_class_count": int(len(groups)),
            "duplicate_column_count": int(sum(len(cols) for cols in groups)),
            "extra_column_count": extra_column_count,
            "unique_column_count": int(self.num_vars - extra_column_count),
        }


@dataclass(frozen=True)
class MaskedQuaternaryBPConfig:
    max_iter: int = 60
    schedule: str = "shuffled_layered"
    damping: float = 0.1
    variable_damping: float = 0.0
    atanh_clip_eps: float = 1e-15
    convergence_tol: float = 1e-9
    random_seed: int | None = 0

    def normalized_schedule(self) -> str:
        schedule = str(self.schedule).strip().lower().replace("-", "_")
        if schedule in {"shuffled", "shuffled_layered", "layered_shuffled"}:
            return "shuffled_layered"
        if schedule in {"residual", "residual_layered", "layered_residual"}:
            return "residual_layered"
        if schedule in {"layered"}:
            return "layered"
        if schedule == "flooding":
            return "flooding"
        return schedule

    def validate(self) -> None:
        if int(self.max_iter) <= 0:
            raise ValueError("max_iter must be > 0")
        if self.normalized_schedule() not in {"flooding", "layered", "shuffled_layered", "residual_layered"}:
            raise ValueError("schedule must be one of flooding, layered, shuffled_layered, residual_layered")
        if not (0.0 <= float(self.damping) < 1.0):
            raise ValueError("damping must lie in [0, 1)")
        if not (0.0 <= float(self.variable_damping) < 1.0):
            raise ValueError("variable_damping must lie in [0, 1)")
        if float(self.atanh_clip_eps) <= 0.0 or float(self.atanh_clip_eps) >= 1.0:
            raise ValueError("atanh_clip_eps must lie in (0, 1)")
        if float(self.convergence_tol) < 0.0:
            raise ValueError("convergence_tol must be >= 0")


@dataclass(frozen=True)
class MaskedQuaternaryBPDecodeResult:
    success: bool
    status: str
    iterations: int
    estimate_states: np.ndarray
    predicted_logicals: np.ndarray
    syndrome_matches: bool
    posterior_log_probs: np.ndarray
    final_check_to_var_llr: np.ndarray
    final_unsatisfied_vector: np.ndarray
    max_message_residual: float
    max_message_residual_by_sweep: tuple[float, ...]
    unsatisfied_checks_by_sweep: tuple[int, ...]
    posterior_entropy_mean: float
    posterior_entropy_min: float
    posterior_entropy_max: float


@dataclass
class MaskedQuaternaryBPDecoder:
    problem: MaskedQuaternarySideProblem
    config: MaskedQuaternaryBPConfig = MaskedQuaternaryBPConfig()

    def __post_init__(self) -> None:
        self.config.validate()

    def check_syndrome(self, states: np.ndarray, syndrome: np.ndarray) -> bool:
        return self.problem.check_syndrome(states, syndrome)

    def predict_logicals(self, states: np.ndarray) -> np.ndarray:
        return self.problem.logical_action_from_states(states)

    def compute_posteriors(
        self,
        *,
        log_priors: np.ndarray,
        check_to_var_llr: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        beliefs = np.asarray(log_priors, dtype=np.float64).copy()
        llr = np.asarray(check_to_var_llr, dtype=np.float64).reshape(-1)
        if int(llr.size) != self.problem.n_edges:
            raise ValueError(f"expected {self.problem.n_edges} check messages, got {llr.size}")
        for edge_index in range(self.problem.n_edges):
            beliefs[int(self.problem.edge_var[edge_index])] += _lift_log_message(
                int(self.problem.edge_mask[edge_index]),
                float(llr[edge_index]),
            )
        return _normalize_log_probs(beliefs) if bool(normalize) else beliefs

    def initialize_state(
        self,
        *,
        priors: np.ndarray | None = None,
        log_priors: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        resolved_log_priors = self.problem.resolve_log_priors(priors=priors, log_priors=log_priors)
        check_to_var_llr = np.zeros(self.problem.n_edges, dtype=np.float64)
        beliefs = self.compute_posteriors(log_priors=resolved_log_priors, check_to_var_llr=check_to_var_llr, normalize=False)
        variable_to_check_log = np.empty((self.problem.n_edges, 4), dtype=np.float64)
        for edge_index in range(self.problem.n_edges):
            variable_to_check_log[edge_index] = _normalize_log_probs(
                np.asarray(resolved_log_priors[int(self.problem.edge_var[edge_index])], dtype=np.float64)
            )
        return resolved_log_priors, beliefs, variable_to_check_log

    def run_bp_iteration(
        self,
        *,
        syndrome: np.ndarray,
        log_priors: np.ndarray,
        beliefs: np.ndarray,
        check_to_var_llr: np.ndarray,
        variable_to_check_log: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        target = np.asarray(syndrome, dtype=np.uint8).reshape(-1) & 1
        if int(target.size) != self.problem.num_detectors:
            raise ValueError(f"syndrome length mismatch: got {target.size}, expected {self.problem.num_detectors}")
        schedule = self.config.normalized_schedule()
        if schedule == "flooding":
            return self._run_flooding_iteration(
                syndrome=target,
                log_priors=log_priors,
                beliefs=beliefs,
                check_to_var_llr=check_to_var_llr,
                variable_to_check_log=variable_to_check_log,
            )
        return self._run_layered_iteration(
            syndrome=target,
            log_priors=log_priors,
            beliefs=beliefs,
            check_to_var_llr=check_to_var_llr,
            variable_to_check_log=variable_to_check_log,
            rng=rng,
            shuffle=bool(schedule == "shuffled_layered"),
            residual_order=bool(schedule == "residual_layered"),
        )

    def _edge_variable_message(
        self,
        *,
        edge_index: int,
        beliefs: np.ndarray,
        check_to_var_llr: np.ndarray,
        variable_to_check_log: np.ndarray,
    ) -> np.ndarray:
        mask = int(self.problem.edge_mask[int(edge_index)])
        var_index = int(self.problem.edge_var[int(edge_index)])
        outgoing = np.asarray(beliefs[var_index], dtype=np.float64) - _lift_log_message(
            mask,
            float(check_to_var_llr[int(edge_index)]),
        )
        return _damp_log_message(
            outgoing,
            np.asarray(variable_to_check_log[int(edge_index)], dtype=np.float64),
            float(self.config.variable_damping),
        )

    def _check_raw_update(
        self,
        *,
        check_index: int,
        syndrome_bit: int,
        beliefs: np.ndarray,
        check_to_var_llr: np.ndarray,
        variable_to_check_log: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        edges = self.problem.check_to_edges[int(check_index)]
        outgoing_messages = np.empty((int(edges.size), 4), dtype=np.float64)
        incoming_bias = np.empty(int(edges.size), dtype=np.float64)
        for local_index, edge_index in enumerate(edges.tolist()):
            outgoing = self._edge_variable_message(
                edge_index=int(edge_index),
                beliefs=beliefs,
                check_to_var_llr=check_to_var_llr,
                variable_to_check_log=variable_to_check_log,
            )
            outgoing_messages[local_index] = outgoing
            _, incoming_bias[local_index] = _project_log_message(outgoing, int(self.problem.edge_mask[edge_index]))
        raw = _check_update_from_biases(
            incoming_bias,
            int(syndrome_bit),
            atanh_clip_eps=float(self.config.atanh_clip_eps),
        )
        return edges, outgoing_messages, raw

    def _run_flooding_iteration(
        self,
        *,
        syndrome: np.ndarray,
        log_priors: np.ndarray,
        beliefs: np.ndarray,
        check_to_var_llr: np.ndarray,
        variable_to_check_log: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        v2c_log = np.empty((self.problem.n_edges, 4), dtype=np.float64)
        projected_bias = np.empty(self.problem.n_edges, dtype=np.float64)
        for edge_index in range(self.problem.n_edges):
            outgoing = self._edge_variable_message(
                edge_index=int(edge_index),
                beliefs=beliefs,
                check_to_var_llr=check_to_var_llr,
                variable_to_check_log=variable_to_check_log,
            )
            v2c_log[edge_index] = outgoing
            _, projected_bias[edge_index] = _project_log_message(outgoing, int(self.problem.edge_mask[edge_index]))

        new_check_to_var = np.asarray(check_to_var_llr, dtype=np.float64).copy()
        max_residual = 0.0
        for check_index, edges in enumerate(self.problem.check_to_edges):
            if int(edges.size) == 0:
                continue
            raw = _check_update_from_biases(
                projected_bias[np.asarray(edges, dtype=np.intp)],
                int(syndrome[check_index]),
                atanh_clip_eps=float(self.config.atanh_clip_eps),
            )
            if float(self.config.damping) > 0.0:
                raw = (1.0 - float(self.config.damping)) * raw + float(self.config.damping) * new_check_to_var[edges]
            residual = float(np.max(np.abs(raw - new_check_to_var[edges])))
            max_residual = max(max_residual, residual)
            new_check_to_var[edges] = raw

        new_beliefs = self.compute_posteriors(
            log_priors=log_priors,
            check_to_var_llr=new_check_to_var,
            normalize=False,
        )
        return new_beliefs, new_check_to_var, v2c_log, float(max_residual)

    def _run_layered_iteration(
        self,
        *,
        syndrome: np.ndarray,
        log_priors: np.ndarray,
        beliefs: np.ndarray,
        check_to_var_llr: np.ndarray,
        variable_to_check_log: np.ndarray,
        rng: np.random.Generator | None,
        shuffle: bool,
        residual_order: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        updated_beliefs = np.asarray(beliefs, dtype=np.float64).copy()
        updated_check_to_var = np.asarray(check_to_var_llr, dtype=np.float64).copy()
        v2c_log = np.asarray(variable_to_check_log, dtype=np.float64).copy()
        order = np.arange(self.problem.num_detectors, dtype=np.int32)
        if bool(residual_order):
            order_pairs: list[tuple[float, int]] = []
            for check_index in order.tolist():
                edges, _outgoing, raw = self._check_raw_update(
                    check_index=int(check_index),
                    syndrome_bit=int(syndrome[int(check_index)]),
                    beliefs=updated_beliefs,
                    check_to_var_llr=updated_check_to_var,
                    variable_to_check_log=v2c_log,
                )
                if int(edges.size) == 0:
                    residual = 0.0
                else:
                    residual = float(np.max(np.abs(raw - np.asarray(updated_check_to_var[edges], dtype=np.float64))))
                order_pairs.append((float(residual), int(check_index)))
            order = np.asarray([idx for _residual, idx in sorted(order_pairs, reverse=True)], dtype=np.int32)
        elif bool(shuffle):
            if rng is None:
                rng = np.random.default_rng()
            rng.shuffle(order)

        max_residual = 0.0
        for check_index in order.tolist():
            edges, outgoing_messages, raw = self._check_raw_update(
                check_index=int(check_index),
                syndrome_bit=int(syndrome[int(check_index)]),
                beliefs=updated_beliefs,
                check_to_var_llr=updated_check_to_var,
                variable_to_check_log=v2c_log,
            )
            if int(edges.size) == 0:
                continue
            for local_index, edge_index in enumerate(edges.tolist()):
                v2c_log[edge_index] = outgoing_messages[local_index]
            old_values = np.asarray(updated_check_to_var[edges], dtype=np.float64)
            if float(self.config.damping) > 0.0:
                raw = (1.0 - float(self.config.damping)) * raw + float(self.config.damping) * old_values
            residual = float(np.max(np.abs(raw - old_values)))
            max_residual = max(max_residual, residual)
            for local_index, edge_index in enumerate(edges.tolist()):
                mask = int(self.problem.edge_mask[edge_index])
                var_index = int(self.problem.edge_var[edge_index])
                old_lift = _lift_log_message(mask, float(updated_check_to_var[edge_index]))
                new_lift = _lift_log_message(mask, float(raw[local_index]))
                updated_beliefs[var_index] += new_lift - old_lift
                updated_check_to_var[edge_index] = float(raw[local_index])

        updated_beliefs -= np.max(updated_beliefs, axis=1, keepdims=True)
        return updated_beliefs, updated_check_to_var, v2c_log, float(max_residual)

    def decode(
        self,
        syndrome: np.ndarray,
        *,
        priors: np.ndarray | None = None,
        log_priors: np.ndarray | None = None,
        target_logicals: np.ndarray | None = None,
    ) -> MaskedQuaternaryBPDecodeResult:
        target = np.asarray(syndrome, dtype=np.uint8).reshape(-1) & 1
        if int(target.size) != self.problem.num_detectors:
            raise ValueError(f"syndrome length mismatch: got {target.size}, expected {self.problem.num_detectors}")

        if target_logicals is not None:
            logical_target = np.asarray(target_logicals, dtype=np.uint8).reshape(-1) & 1
            if int(logical_target.size) != self.problem.num_logicals:
                raise ValueError(
                    f"logical target length mismatch: got {logical_target.size}, expected {self.problem.num_logicals}"
                )
        else:
            logical_target = None

        rng = np.random.default_rng(self.config.random_seed) if self.config.random_seed is not None else np.random.default_rng()
        log_prior_matrix, beliefs, variable_to_check_log = self.initialize_state(priors=priors, log_priors=log_priors)
        check_to_var_llr = np.zeros(self.problem.n_edges, dtype=np.float64)
        residual_history: list[float] = []
        unsatisfied_history: list[int] = []
        posterior_log_probs = _normalize_log_probs(beliefs)
        hard = np.asarray(np.argmax(posterior_log_probs, axis=1), dtype=np.uint8)
        final_unsatisfied = self.problem.syndrome_from_states(hard) ^ target
        predicted_logicals = self.problem.logical_action_from_states(hard)

        try:
            for iteration in range(1, int(self.config.max_iter) + 1):
                beliefs, check_to_var_llr, variable_to_check_log, max_residual = self.run_bp_iteration(
                    syndrome=target,
                    log_priors=log_prior_matrix,
                    beliefs=beliefs,
                    check_to_var_llr=check_to_var_llr,
                    variable_to_check_log=variable_to_check_log,
                    rng=rng,
                )
                if not np.all(np.isfinite(beliefs)):
                    raise FloatingPointError("beliefs contain non-finite values")
                if not np.all(np.isfinite(check_to_var_llr)):
                    raise FloatingPointError("check-to-variable messages contain non-finite values")

                posterior_log_probs = _normalize_log_probs(beliefs)
                hard = np.asarray(np.argmax(posterior_log_probs, axis=1), dtype=np.uint8)
                final_unsatisfied = self.problem.syndrome_from_states(hard) ^ target
                predicted_logicals = self.problem.logical_action_from_states(hard)
                unsatisfied = int(np.count_nonzero(final_unsatisfied))
                residual_history.append(float(max_residual))
                unsatisfied_history.append(int(unsatisfied))

                if unsatisfied == 0:
                    if logical_target is not None and not np.array_equal(predicted_logicals, logical_target):
                        status = "logical_fail"
                        success = False
                    else:
                        status = "success"
                        success = True
                    entropy = _entropy_from_log_probs(posterior_log_probs)
                    return MaskedQuaternaryBPDecodeResult(
                        success=bool(success),
                        status=str(status),
                        iterations=int(iteration),
                        estimate_states=np.asarray(hard, dtype=np.uint8),
                        predicted_logicals=np.asarray(predicted_logicals, dtype=np.uint8),
                        syndrome_matches=True,
                        posterior_log_probs=np.asarray(posterior_log_probs, dtype=np.float64),
                        final_check_to_var_llr=np.asarray(check_to_var_llr, dtype=np.float64),
                        final_unsatisfied_vector=np.asarray(final_unsatisfied, dtype=np.uint8),
                        max_message_residual=float(max_residual),
                        max_message_residual_by_sweep=tuple(float(x) for x in residual_history),
                        unsatisfied_checks_by_sweep=tuple(int(x) for x in unsatisfied_history),
                        posterior_entropy_mean=float(np.mean(entropy)),
                        posterior_entropy_min=float(np.min(entropy)),
                        posterior_entropy_max=float(np.max(entropy)),
                    )
                if float(max_residual) <= float(self.config.convergence_tol):
                    entropy = _entropy_from_log_probs(posterior_log_probs)
                    return MaskedQuaternaryBPDecodeResult(
                        success=False,
                        status="syndrome_fail",
                        iterations=int(iteration),
                        estimate_states=np.asarray(hard, dtype=np.uint8),
                        predicted_logicals=np.asarray(predicted_logicals, dtype=np.uint8),
                        syndrome_matches=False,
                        posterior_log_probs=np.asarray(posterior_log_probs, dtype=np.float64),
                        final_check_to_var_llr=np.asarray(check_to_var_llr, dtype=np.float64),
                        final_unsatisfied_vector=np.asarray(final_unsatisfied, dtype=np.uint8),
                        max_message_residual=float(max_residual),
                        max_message_residual_by_sweep=tuple(float(x) for x in residual_history),
                        unsatisfied_checks_by_sweep=tuple(int(x) for x in unsatisfied_history),
                        posterior_entropy_mean=float(np.mean(entropy)),
                        posterior_entropy_min=float(np.min(entropy)),
                        posterior_entropy_max=float(np.max(entropy)),
                    )
        except FloatingPointError:
            entropy = _entropy_from_log_probs(posterior_log_probs)
            return MaskedQuaternaryBPDecodeResult(
                success=False,
                status="numerical_fail",
                iterations=int(len(residual_history)),
                estimate_states=np.asarray(hard, dtype=np.uint8),
                predicted_logicals=np.asarray(predicted_logicals, dtype=np.uint8),
                syndrome_matches=bool(np.count_nonzero(final_unsatisfied) == 0),
                posterior_log_probs=np.asarray(posterior_log_probs, dtype=np.float64),
                final_check_to_var_llr=np.asarray(check_to_var_llr, dtype=np.float64),
                final_unsatisfied_vector=np.asarray(final_unsatisfied, dtype=np.uint8),
                max_message_residual=float(residual_history[-1]) if residual_history else float("nan"),
                max_message_residual_by_sweep=tuple(float(x) for x in residual_history),
                unsatisfied_checks_by_sweep=tuple(int(x) for x in unsatisfied_history),
                posterior_entropy_mean=float(np.mean(entropy)),
                posterior_entropy_min=float(np.min(entropy)),
                posterior_entropy_max=float(np.max(entropy)),
            )

        entropy = _entropy_from_log_probs(posterior_log_probs)
        return MaskedQuaternaryBPDecodeResult(
            success=False,
            status="max_iter_reached",
            iterations=int(self.config.max_iter),
            estimate_states=np.asarray(hard, dtype=np.uint8),
            predicted_logicals=np.asarray(predicted_logicals, dtype=np.uint8),
            syndrome_matches=bool(np.count_nonzero(final_unsatisfied) == 0),
            posterior_log_probs=np.asarray(posterior_log_probs, dtype=np.float64),
            final_check_to_var_llr=np.asarray(check_to_var_llr, dtype=np.float64),
            final_unsatisfied_vector=np.asarray(final_unsatisfied, dtype=np.uint8),
            max_message_residual=float(residual_history[-1]) if residual_history else float("nan"),
            max_message_residual_by_sweep=tuple(float(x) for x in residual_history),
            unsatisfied_checks_by_sweep=tuple(int(x) for x in unsatisfied_history),
            posterior_entropy_mean=float(np.mean(entropy)),
            posterior_entropy_min=float(np.min(entropy)),
            posterior_entropy_max=float(np.max(entropy)),
        )


@lru_cache(maxsize=8)
def _build_masked_quaternary_cnot_side_problem_cached(
    *,
    backend: str,
    error_rate: float,
    qtanner_root: str,
    side: str,
) -> MaskedQuaternarySideProblem:
    source_problem = build_cnot_location_problem(
        backend=str(backend),
        error_rate=float(error_rate),
        qtanner_root=Path(str(qtanner_root)),
    )
    side_key = str(side).strip()
    if side_key in {"X", "memory_X", "x"}:
        source_side = source_problem.X
    elif side_key in {"Z", "memory_Z", "z"}:
        source_side = source_problem.Z
    else:
        raise ValueError(f"unsupported side {side!r}; expected memory_X or memory_Z")
    return MaskedQuaternarySideProblem.from_cnot_side_problem(source_side)


def build_masked_quaternary_cnot_side_problem(
    *,
    backend: str = "bravyi_depth7",
    error_rate: float = 0.004,
    qtanner_root: str | Path | None = None,
    side: str = "memory_Z",
) -> MaskedQuaternarySideProblem:
    resolved_root = resolve_qtanner_root(qtanner_root)
    return _build_masked_quaternary_cnot_side_problem_cached(
        backend=str(backend),
        error_rate=float(error_rate),
        qtanner_root=str(resolved_root),
        side=str(side),
    )


def build_gross_x_projected_problem(
    *,
    backend: str = "bravyi_depth7",
    error_rate: float = 0.004,
    qtanner_root: str | Path | None = None,
) -> MaskedQuaternarySideProblem:
    return build_masked_quaternary_cnot_side_problem(
        backend=str(backend),
        error_rate=float(error_rate),
        qtanner_root=qtanner_root,
        side="memory_Z",
    )


def build_gross_z_projected_problem(
    *,
    backend: str = "bravyi_depth7",
    error_rate: float = 0.004,
    qtanner_root: str | Path | None = None,
) -> MaskedQuaternarySideProblem:
    return build_masked_quaternary_cnot_side_problem(
        backend=str(backend),
        error_rate=float(error_rate),
        qtanner_root=qtanner_root,
        side="memory_X",
    )


__all__ = [
    "MaskedQuaternaryBPConfig",
    "MaskedQuaternaryBPDecodeResult",
    "MaskedQuaternaryBPDecoder",
    "MaskedQuaternarySideProblem",
    "PROJECTED_STATE_LABELS_X",
    "PROJECTED_STATE_LABELS_Z",
    "build_gross_x_projected_problem",
    "build_gross_z_projected_problem",
    "build_masked_quaternary_cnot_side_problem",
    "masked_parity",
]
