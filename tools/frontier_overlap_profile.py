"""Profile bounded-hypergraph overlap loads and low-weight open polymers.

The tool is deterministic and performs no decoding or Monte Carlo sampling.
It audits the ordered detector matrix used by Frontier, computes the exact
cutwise future-column load entering the Finner condition, and enumerates all
visible open-prefix polymers of sizes one and two together with their
ordering lifetimes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from tools.dem_loader import LoadedProgressiveFamily, load_dem_family


SCHEMA_VERSION = 1
DEFAULT_SCOPES = ("memory_X", "memory_Z")
DEFAULT_RHOS = (0.5, 0.75, 0.99)
DEFAULT_K_VALUES = (16, 512, 1024, 8192)
DEFAULT_OPTIMIZATION_RHOS = tuple(float(value) / 100.0 for value in range(5, 100))


@dataclass(frozen=True)
class LowWeightPolymer:
    """A visible size-1/2 polymer and its exact half-open cut interval."""

    column_indices: tuple[int, ...]
    start: int
    stop: int
    detector_shift_mask: int
    logical_shift_mask: int
    activity: float
    size: int


def chernoff_activity(probability: float, theta: float) -> float:
    """Return the Bernoulli translate Chernoff coefficient."""

    p = float(probability)
    value = float(theta)
    if not 0.0 < p < 1.0:
        raise ValueError("probability must lie strictly between zero and one")
    if not 0.0 <= value <= 1.0:
        raise ValueError("theta must lie in [0, 1]")
    return float(
        (1.0 - p) ** (1.0 - value) * p**value
        + p ** (1.0 - value) * (1.0 - p) ** value
    )


def safe_chernoff_order(*, score_alpha: float, load: int) -> float:
    """Largest theta at most one-half certified by theta*alpha*load <= 1."""

    alpha = float(score_alpha)
    count = int(load)
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("score_alpha must be finite and non-negative")
    if count < 0:
        raise ValueError("load must be non-negative")
    if alpha == 0.0 or count == 0:
        return 0.5
    return float(min(0.5, 1.0 / (alpha * float(count))))


def _iter_set_bits(mask: int) -> Iterable[int]:
    value = int(mask)
    while value:
        bit = int(value & -value)
        yield int(bit.bit_length() - 1)
        value ^= bit


def _column_logical_mask(family: LoadedProgressiveFamily, index: int) -> int:
    masks = tuple(int(value) for value in family.columns[int(index)].logical_response_masks)
    if len(masks) != 2 or masks[0] != 0:
        raise ValueError(
            "overlap profiling currently requires binary columns with outcome 0 as identity"
        )
    return int(masks[1])


def _column_probability(family: LoadedProgressiveFamily, index: int) -> float:
    probabilities = tuple(float(value) for value in family.columns[int(index)].prior_probs)
    if len(probabilities) != 2:
        raise ValueError("overlap profiling currently requires binary columns")
    return float(probabilities[1])


def _family_checksum(family: LoadedProgressiveFamily) -> str:
    digest = hashlib.sha256()
    for index, column in enumerate(family.columns):
        line = (
            f"{int(index)}|{int(column.original_column_index)}|"
            f"{int(column.detector_support_mask)}|{_column_logical_mask(family, index)}|"
            f"{_column_probability(family, index):.17g}\n"
        )
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def _integer_histogram(values: Sequence[int]) -> dict[str, int]:
    return {
        str(int(key)): int(count)
        for key, count in sorted(Counter(int(value) for value in values).items())
    }


def _series_summary(values: Sequence[float]) -> dict[str, float | int]:
    sequence = tuple(float(value) for value in values)
    if not sequence:
        return {
            "cuts": 0,
            "nonzero_cuts": 0,
            "total": 0.0,
            "mean": 0.0,
            "peak": 0.0,
            "peak_cut_after_column": -1,
        }
    peak = max(sequence)
    return {
        "cuts": int(len(sequence)),
        "nonzero_cuts": int(sum(value > 0.0 for value in sequence)),
        "total": float(math.fsum(sequence)),
        "mean": float(math.fsum(sequence) / float(len(sequence))),
        "peak": float(peak),
        "peak_cut_after_column": int(sequence.index(peak)),
    }


def _log10_series_summary(values: Sequence[float]) -> dict[str, float | int]:
    sequence = tuple(float(value) for value in values)
    finite = tuple(
        (index, value) for index, value in enumerate(sequence) if math.isfinite(value)
    )
    if not finite:
        return {
            "cuts": int(len(sequence)),
            "finite_cuts": 0,
            "mean": float("-inf"),
            "minimum": float("-inf"),
            "peak": float("-inf"),
            "peak_cut_after_column": -1,
        }
    peak_index, peak = max(finite, key=lambda item: item[1])
    finite_values = tuple(value for _, value in finite)
    return {
        "cuts": int(len(sequence)),
        "finite_cuts": int(len(finite)),
        "mean": float(math.fsum(finite_values) / len(finite_values)),
        "minimum": float(min(finite_values)),
        "peak": float(peak),
        "peak_cut_after_column": int(peak_index),
    }


def _log_expm1(value: float) -> float:
    x = float(value)
    if x <= 0.0:
        return float("-inf")
    if x > 50.0:
        return float(x + math.log1p(-math.exp(-x)))
    return float(math.log(math.expm1(x)))


def _partial_cap_rhs(
    values: Sequence[float],
    *,
    rho: float,
    K: int,
) -> dict[str, float | None]:
    """Evaluate the size-1/2 contribution to the product-Peierls RHS."""

    exponent = float(rho)
    cap = int(K)
    if not 0.0 < exponent < 1.0:
        raise ValueError("rho must lie strictly between zero and one")
    if cap < 1:
        raise ValueError("K must be positive")
    log_terms = tuple(
        _log_expm1(float(value)) / exponent
        for value in values
        if float(value) > 0.0
    )
    if not log_terms:
        return {"log10": float("-inf"), "value": 0.0}
    maximum = max(log_terms)
    log_sum = float(
        maximum + math.log(math.fsum(math.exp(term - maximum) for term in log_terms))
    )
    log_rhs = float(log_sum - math.log(float(cap)) / exponent)
    return {
        "log10": float(log_rhs / math.log(10.0)),
        "value": float(math.exp(log_rhs)) if log_rhs < 700.0 else None,
    }


def _logsumexp(log_values: Sequence[float]) -> float:
    values = tuple(float(value) for value in log_values)
    if not values:
        return float("-inf")
    maximum = max(values)
    if maximum == float("-inf"):
        return float("-inf")
    return float(
        maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))
    )


def _cap_rhs_from_log_terms(
    log_terms: Sequence[float],
    *,
    rho: float,
    K: int,
) -> dict[str, float | None]:
    """Scale rooted cut terms represented in the log domain by K^(-1/rho)."""

    exponent = float(rho)
    cap = int(K)
    if not 0.0 < exponent < 1.0:
        raise ValueError("rho must lie strictly between zero and one")
    if cap < 1:
        raise ValueError("K must be positive")
    log_sum = _logsumexp(tuple(float(value) for value in log_terms))
    if log_sum == float("-inf"):
        return {"log10": float("-inf"), "value": 0.0}
    log_rhs = float(log_sum - math.log(float(cap)) / exponent)
    return {
        "log10": float(log_rhs / math.log(10.0)),
        "value": float(math.exp(log_rhs)) if log_rhs < 700.0 else None,
    }


def trimmed_spectrum_bound(values: Sequence[float], *, K: int) -> float:
    """Evaluate the exact threshold-head functional on a majorant spectrum."""

    cap = int(K)
    if cap < 1:
        raise ValueError("K must be positive")
    ordered = tuple(
        sorted(
            (float(value) for value in values if float(value) > 0.0),
            reverse=True,
        )
    )
    if len(ordered) < cap:
        return 0.0
    tail = float(math.fsum(ordered))
    best = float(tail / cap)
    for head_size in range(1, cap):
        tail -= float(ordered[head_size - 1])
        best = min(best, float(tail / (cap - head_size)))
    return float(best)


def future_active_load_profile(family: LoadedProgressiveFamily) -> tuple[int, ...]:
    """Exact worst affected-row load of a future column at every cut."""

    count = int(len(family.columns))
    if count == 0:
        return tuple()
    layout = family.layout
    active_mask = 0
    future_load = [0 for _ in range(count)]
    load_histogram: Counter[int] = Counter({0: count})
    result: list[int] = []

    for cut in range(count):
        removed_load = int(future_load[cut])
        load_histogram[removed_load] -= 1
        if load_histogram[removed_load] == 0:
            del load_histogram[removed_load]

        next_active_mask = int(layout.active_masks_after_column[cut])
        changed = int(active_mask ^ next_active_mask)
        for row in _iter_set_bits(changed):
            delta = 1 if (next_active_mask >> int(row)) & 1 else -1
            for future_column in layout.row_touch_columns[int(row)]:
                target = int(future_column)
                if target <= cut:
                    continue
                previous = int(future_load[target])
                load_histogram[previous] -= 1
                if load_histogram[previous] == 0:
                    del load_histogram[previous]
                updated = int(previous + delta)
                if updated < 0:
                    raise AssertionError("future active load became negative")
                future_load[target] = int(updated)
                load_histogram[updated] += 1
        active_mask = int(next_active_mask)
        result.append(int(max(load_histogram, default=0)))

        if sum(load_histogram.values()) != count - cut - 1:
            raise AssertionError("future-column load accounting is inconsistent")

    return tuple(result)


def shift_specific_chernoff_orders(
    family: LoadedProgressiveFamily,
    *,
    cut: int,
    active_rows: Sequence[int],
    score_alpha: float,
) -> tuple[float, ...]:
    """Return the largest safe theta for every active-detector shift."""

    active = tuple(int(row) for row in active_rows)
    active_mask = int(sum(1 << row for row in active))
    future_masks: set[int] = set()
    for column in family.columns[int(cut) + 1 :]:
        support = int(column.detector_support_mask) & active_mask
        local_support = 0
        for local_row, global_row in enumerate(active):
            if (support >> global_row) & 1:
                local_support |= 1 << local_row
        if local_support:
            future_masks.add(int(local_support))

    result: list[float] = []
    for detector_shift in range(1 << len(active)):
        load = max(
            (
                int(detector_shift & future_mask).bit_count()
                for future_mask in future_masks
            ),
            default=0,
        )
        result.append(
            safe_chernoff_order(
                score_alpha=float(score_alpha),
                load=int(load),
            )
        )
    return tuple(result)


def _add_interval(
    differences: list[float],
    *,
    start: int,
    stop: int,
    value: float,
) -> None:
    if int(start) >= int(stop):
        return
    differences[int(start)] += float(value)
    differences[int(stop)] -= float(value)


def _integrate_differences(differences: Sequence[float]) -> tuple[float, ...]:
    current = 0.0
    result: list[float] = []
    for delta in tuple(differences)[:-1]:
        current += float(delta)
        if abs(current) < 1e-12:
            current = 0.0
        result.append(float(current))
    return tuple(result)


def _minimum_last_touch(mask: int, detector_last_column: Sequence[int]) -> int:
    return min(int(detector_last_column[row]) for row in _iter_set_bits(mask))


def enumerate_low_weight_polymers(
    family: LoadedProgressiveFamily,
    *,
    theta: float,
) -> tuple[LowWeightPolymer, ...]:
    """Return every visible size-1/2 polymer with its exact cut interval."""

    column_count = int(len(family.columns))
    supports = tuple(int(column.detector_support_mask) for column in family.columns)
    logical_masks = tuple(
        int(_column_logical_mask(family, index)) for index in range(column_count)
    )
    activities = tuple(
        chernoff_activity(_column_probability(family, index), float(theta))
        for index in range(column_count)
    )
    last = tuple(int(value) for value in family.layout.detector_last_column)
    polymers: list[LowWeightPolymer] = []

    for index, support in enumerate(supports):
        if support:
            stop = _minimum_last_touch(support, last)
        elif logical_masks[index]:
            stop = int(column_count)
        else:
            continue
        start = int(index)
        if start < stop:
            polymers.append(
                LowWeightPolymer(
                    column_indices=(int(index),),
                    start=int(start),
                    stop=int(stop),
                    detector_shift_mask=int(support),
                    logical_shift_mask=int(logical_masks[index]),
                    activity=float(activities[index]),
                    size=1,
                )
            )

    encoded_pairs: set[int] = set()
    for touches in family.layout.row_touch_columns:
        row_columns = tuple(int(value) for value in touches)
        for left_offset, left in enumerate(row_columns):
            for right in row_columns[left_offset + 1 :]:
                a, b = (left, int(right)) if left < int(right) else (int(right), left)
                encoded_pairs.add(int(a * column_count + b))

    for encoded in sorted(encoded_pairs):
        left, right = divmod(int(encoded), column_count)
        common = int(supports[left] & supports[right])
        if common == 0:
            raise AssertionError("candidate pair does not share a detector row")
        detector_shift = int(supports[left] ^ supports[right])
        logical_shift = int(logical_masks[left] ^ logical_masks[right])
        start = _minimum_last_touch(common, last)
        stop = (
            _minimum_last_touch(detector_shift, last)
            if detector_shift
            else int(column_count)
        )
        if (detector_shift or logical_shift) and start < stop:
            polymers.append(
                LowWeightPolymer(
                    column_indices=(int(left), int(right)),
                    start=int(start),
                    stop=int(stop),
                    detector_shift_mask=int(detector_shift),
                    logical_shift_mask=int(logical_shift),
                    activity=float(activities[left] * activities[right]),
                    size=2,
                )
            )

    return tuple(polymers)


def _low_weight_compatibility_groups(
    family: LoadedProgressiveFamily,
    *,
    polymers: Sequence[LowWeightPolymer],
    cut: int,
) -> tuple[int, tuple[LowWeightPolymer, ...], dict[int, tuple[LowWeightPolymer, ...]]]:
    """Group live size-1/2 polymers by completed-row signature."""

    cut_index = int(cut)
    completed_mask = int(
        sum(
            1 << int(row)
            for row, last_touch in enumerate(family.layout.detector_last_column)
            if int(last_touch) <= cut_index
        )
    )
    live = tuple(
        polymer
        for polymer in polymers
        if int(polymer.start) <= cut_index < int(polymer.stop)
    )
    signature_groups: dict[int, list[LowWeightPolymer]] = {}
    singletons: list[LowWeightPolymer] = []
    for polymer in live:
        signatures = tuple(
            int(family.columns[int(column)].detector_support_mask) & completed_mask
            for column in polymer.column_indices
        )
        if len(set(signatures)) != 1:
            raise AssertionError("polymer columns disagree on completed-row signature")
        signature = int(signatures[0])
        if int(polymer.size) == 1:
            if signature != 0:
                raise AssertionError("live singleton touches a completed detector")
            singletons.append(polymer)
            continue
        if int(polymer.size) != 2 or len(polymer.column_indices) != 2:
            raise AssertionError("compatibility audit only supports size-1/2 polymers")
        if signature == 0:
            raise AssertionError("live pair is disconnected in the completed-row graph")
        signature_groups.setdefault(signature, []).append(polymer)
    return (
        int(completed_mask),
        tuple(singletons),
        {
            int(signature): tuple(group)
            for signature, group in signature_groups.items()
        },
    )


def low_weight_compatibility_structure(
    family: LoadedProgressiveFamily,
    *,
    polymers: Sequence[LowWeightPolymer],
    cut: int,
) -> dict[str, object]:
    """Summarize the exact compatibility structure of live size-1/2 polymers.

    At a fixed cut, a live singleton has empty completed-row signature. A
    live pair has two columns with the same nonempty completed-row signature.
    Pair polymers in one signature class form a clique, and two classes
    conflict exactly when their signatures intersect.
    """

    completed_mask, singletons, signature_groups = _low_weight_compatibility_groups(
        family,
        polymers=polymers,
        cut=int(cut),
    )

    signatures = tuple(sorted(signature_groups))
    adjacency = [0 for _ in signatures]
    interaction_edges = 0
    for left, left_signature in enumerate(signatures):
        for right in range(left + 1, len(signatures)):
            if int(left_signature) & int(signatures[right]):
                adjacency[left] |= 1 << right
                adjacency[right] |= 1 << left
                interaction_edges += 1

    unseen = (1 << len(signatures)) - 1
    component_class_counts: list[int] = []
    component_pair_counts: list[int] = []
    while unseen:
        seed = int(unseen & -unseen)
        frontier = int(seed)
        component = 0
        while frontier:
            vertex_bit = int(frontier & -frontier)
            frontier ^= vertex_bit
            vertex = int(vertex_bit.bit_length() - 1)
            component |= vertex_bit
            frontier |= int(adjacency[vertex]) & ~component
        unseen &= ~component
        component_vertices = tuple(_iter_set_bits(component))
        component_class_counts.append(int(len(component_vertices)))
        component_pair_counts.append(
            int(
                sum(
                    len(signature_groups[signatures[vertex]])
                    for vertex in component_vertices
                )
            )
        )

    class_option_counts = tuple(
        int(len(signature_groups[signature])) for signature in signatures
    )
    signature_weights = tuple(int(signature.bit_count()) for signature in signatures)
    used_completed_mask = 0
    for signature in signatures:
        used_completed_mask |= int(signature)
    return {
        "completed_detector_rows": int(completed_mask.bit_count()),
        "used_completed_detector_rows": int(used_completed_mask.bit_count()),
        "live_singletons": int(len(singletons)),
        "live_pairs": int(sum(class_option_counts)),
        "pair_signature_classes": int(len(signatures)),
        "pair_signature_weight_histogram": _integer_histogram(signature_weights),
        "pair_options_per_class_histogram": _integer_histogram(class_option_counts),
        "maximum_pair_options_per_class": int(max(class_option_counts, default=0)),
        "signature_interaction_edges": int(interaction_edges),
        "signature_interaction_components": int(len(component_class_counts)),
        "component_class_count_histogram": _integer_histogram(component_class_counts),
        "component_pair_count_histogram": _integer_histogram(component_pair_counts),
        "maximum_component_classes": int(max(component_class_counts, default=0)),
        "maximum_component_pair_options": int(max(component_pair_counts, default=0)),
    }


def compatible_ungrouped_moment(
    family: LoadedProgressiveFamily,
    *,
    polymers: Sequence[LowWeightPolymer],
    cut: int,
    rho: float,
) -> float:
    """Exact compatible-family moment before grouping by boundary shift.

    Singleton polymers are mutually compatible. Pair polymers are grouped by
    their common nonempty completed-row signature; at most one pair may be
    selected from a class, and selected class signatures must be disjoint.
    """

    exponent = float(rho)
    if not 0.0 < exponent < 1.0:
        raise ValueError("rho must lie strictly between zero and one")
    _, singletons, signature_groups = _low_weight_compatibility_groups(
        family,
        polymers=polymers,
        cut=int(cut),
    )
    singleton_factor = float(
        math.prod(1.0 + float(polymer.activity) ** exponent for polymer in singletons)
    )

    used_global_rows = 0
    for signature in signature_groups:
        used_global_rows |= int(signature)
    global_rows = tuple(_iter_set_bits(used_global_rows))
    local_signatures: list[tuple[int, float]] = []
    for signature, group in sorted(signature_groups.items()):
        local_signature = 0
        for local_row, global_row in enumerate(global_rows):
            if (int(signature) >> int(global_row)) & 1:
                local_signature |= 1 << int(local_row)
        class_weight = float(
            math.fsum(float(polymer.activity) ** exponent for polymer in group)
        )
        local_signatures.append((int(local_signature), float(class_weight)))

    state_count = 1 << len(global_rows)
    partition = np.zeros(state_count, dtype=np.float64)
    partition[0] = 1.0
    row_states = np.arange(state_count, dtype=np.int64)
    for signature, class_weight in local_signatures:
        compatible = row_states[(row_states & int(signature)) == 0]
        updated = partition.copy()
        updated[compatible | int(signature)] += (
            partition[compatible] * float(class_weight)
        )
        partition = updated
    return float(singleton_factor * np.sum(partition, dtype=np.float64) - 1.0)


def _compress_boundary_shift(
    polymer: LowWeightPolymer,
    *,
    active_rows: Sequence[int],
    logical_rows: int,
) -> int:
    active_mask = int(sum(1 << int(row) for row in active_rows))
    if int(polymer.detector_shift_mask) & ~active_mask:
        raise AssertionError("visible polymer detector shift is not active at this cut")
    if int(polymer.logical_shift_mask) >= 1 << int(logical_rows):
        raise AssertionError("polymer logical shift exceeds the declared logical width")
    result = 0
    for local_row, global_row in enumerate(active_rows):
        if (int(polymer.detector_shift_mask) >> int(global_row)) & 1:
            result |= 1 << int(local_row)
    result |= int(polymer.logical_shift_mask) << len(active_rows)
    if result == 0:
        raise AssertionError("visible polymer has zero compressed boundary shift")
    return int(result)


def xor_subset_partition(
    *,
    boundary_bits: int,
    shift_activities: Sequence[tuple[int, float]],
) -> np.ndarray:
    """Exact compatibility-dropped subset partition grouped by XOR shift."""

    bit_count = int(boundary_bits)
    if bit_count < 0:
        raise ValueError("boundary_bits must be non-negative")
    state_count = 1 << bit_count
    indices = np.arange(state_count, dtype=np.int64)
    values = np.zeros(state_count, dtype=np.float64)
    values[0] = 1.0
    for shift, activity in shift_activities:
        encoded_shift = int(shift)
        weight = float(activity)
        if not 0 < encoded_shift < state_count:
            raise ValueError("every shift must be nonzero and fit in boundary_bits")
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("every activity must be finite and non-negative")
        values = values + weight * values[indices ^ encoded_shift]
    return values


def _walsh_hadamard(values: np.ndarray) -> np.ndarray:
    """Return the unnormalized Walsh-Hadamard transform of a 1D vector."""

    transformed = np.array(values, dtype=np.float64, copy=True)
    if transformed.ndim != 1:
        raise ValueError("Walsh-Hadamard input must be one-dimensional")
    count = int(transformed.size)
    if count < 1 or count & (count - 1):
        raise ValueError("Walsh-Hadamard input length must be a positive power of two")
    block = 1
    while block < count:
        view = transformed.reshape(-1, 2 * block)
        left = view[:, :block].copy()
        right = view[:, block:].copy()
        view[:, :block] = left + right
        view[:, block:] = left - right
        block *= 2
    return transformed


def compatible_boundary_partition(
    family: LoadedProgressiveFamily,
    *,
    polymers: Sequence[LowWeightPolymer],
    cut: int,
    active_rows: Sequence[int],
    logical_rows: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Exact size-1/2 compatible-family partition grouped by boundary shift.

    A Walsh-Hadamard transform diagonalizes XOR convolution. For each Fourier
    character, the remaining scalar partition is an exact set-packing dynamic
    program over the completed-row signatures of pair polymers.
    """

    active = tuple(int(row) for row in active_rows)
    boundary_bits = int(len(active) + int(logical_rows))
    boundary_states = 1 << boundary_bits
    _, singletons, signature_groups = _low_weight_compatibility_groups(
        family,
        polymers=polymers,
        cut=int(cut),
    )

    used_global_rows = 0
    for signature in signature_groups:
        used_global_rows |= int(signature)
    global_rows = tuple(_iter_set_bits(used_global_rows))
    row_states = 1 << len(global_rows)
    transformed_partition = np.zeros(
        (row_states, boundary_states),
        dtype=np.float64,
    )
    transformed_partition[0, :] = 1.0
    reachable_rows: set[int] = {0}

    for signature, group in sorted(signature_groups.items()):
        local_signature = 0
        for local_row, global_row in enumerate(global_rows):
            if (int(signature) >> int(global_row)) & 1:
                local_signature |= 1 << int(local_row)
        option_partition = np.zeros(boundary_states, dtype=np.float64)
        for polymer in group:
            shift = _compress_boundary_shift(
                polymer,
                active_rows=active,
                logical_rows=int(logical_rows),
            )
            option_partition[int(shift)] += float(polymer.activity)
        option_transform = _walsh_hadamard(option_partition)

        sources = np.fromiter(
            (
                int(state)
                for state in sorted(reachable_rows)
                if int(state) & int(local_signature) == 0
            ),
            dtype=np.int64,
        )
        if sources.size:
            targets = sources | int(local_signature)
            transformed_partition[targets, :] += (
                transformed_partition[sources, :] * option_transform[np.newaxis, :]
            )
            reachable_rows.update(int(target) for target in targets)

    pair_transform = np.sum(
        transformed_partition[np.fromiter(sorted(reachable_rows), dtype=np.int64), :],
        axis=0,
        dtype=np.float64,
    )
    singleton_partition = xor_subset_partition(
        boundary_bits=int(boundary_bits),
        shift_activities=tuple(
            (
                _compress_boundary_shift(
                    polymer,
                    active_rows=active,
                    logical_rows=int(logical_rows),
                ),
                float(polymer.activity),
            )
            for polymer in singletons
        ),
    )
    total_transform = pair_transform * _walsh_hadamard(singleton_partition)
    partition = _walsh_hadamard(total_transform) / float(boundary_states)
    tolerance = float(
        1e-10 * max(1.0, float(np.max(np.abs(partition), initial=0.0)))
    )
    if float(np.min(partition, initial=0.0)) < -tolerance:
        raise AssertionError("compatible partition has a significant negative entry")
    partition[partition < 0.0] = 0.0
    return partition, {
        "used_completed_rows": int(len(global_rows)),
        "row_mask_states": int(row_states),
        "reachable_row_masks": int(len(reachable_rows)),
        "boundary_states": int(boundary_states),
        "pair_signature_classes": int(len(signature_groups)),
        "live_singletons": int(len(singletons)),
        "live_pairs": int(sum(len(group) for group in signature_groups.values())),
    }


def kernel_boundary_partition_profile(
    family: LoadedProgressiveFamily,
    *,
    theta: float,
) -> tuple[tuple[np.ndarray, ...], tuple[int, ...]]:
    """Enumerate every completed-check kernel vector at every ordered cut.

    The weight of a vector is the product of its column Chernoff activities.
    The state is stored on global detector-row masks while updating, then
    compressed to the active-detector-plus-logical boundary at each cut.
    """

    activities = tuple(
        chernoff_activity(_column_probability(family, index), float(theta))
        for index in range(len(family.columns))
    )
    states: dict[tuple[int, int], float] = {(0, 0): 1.0}
    partitions: list[np.ndarray] = []
    state_counts: list[int] = []
    for cut, column in enumerate(family.columns):
        support = int(column.detector_support_mask)
        logical = int(_column_logical_mask(family, cut))
        activity = float(activities[cut])
        updated: dict[tuple[int, int], float] = {}
        for (detector_mask, logical_mask), weight in states.items():
            key = (int(detector_mask), int(logical_mask))
            updated[key] = float(updated.get(key, 0.0) + float(weight))
            shifted = (
                int(detector_mask) ^ support,
                int(logical_mask) ^ logical,
            )
            updated[shifted] = float(
                updated.get(shifted, 0.0) + float(weight) * activity
            )

        active_mask = int(family.layout.active_masks_after_column[cut])
        states = {
            (int(detector_mask), int(logical_mask)): float(weight)
            for (detector_mask, logical_mask), weight in updated.items()
            if int(detector_mask) & ~active_mask == 0
        }
        active_rows = tuple(_iter_set_bits(active_mask))
        boundary_states = 1 << (len(active_rows) + int(family.logical_rows))
        partition = np.zeros(boundary_states, dtype=np.float64)
        for (detector_mask, logical_mask), weight in states.items():
            compressed = 0
            for local_row, global_row in enumerate(active_rows):
                if (int(detector_mask) >> int(global_row)) & 1:
                    compressed |= 1 << int(local_row)
            compressed |= int(logical_mask) << len(active_rows)
            partition[int(compressed)] += float(weight)
        partitions.append(partition)
        state_counts.append(int(len(states)))
    return tuple(partitions), tuple(state_counts)


def boundary_shift_aggregation_profile(
    family: LoadedProgressiveFamily,
    *,
    theta: float,
    rhos: Sequence[float],
    score_alpha: float = 0.8,
    K_values: Sequence[int] = DEFAULT_K_VALUES,
    optimization_rhos: Sequence[float] = (),
    max_boundary_bits: int = 16,
    progress_every_cuts: int = 25,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Profile grouped low-weight partitions and an all-size kernel majorant."""

    polymers = enumerate_low_weight_polymers(family, theta=float(theta))
    max_future_load = int(max(future_active_load_profile(family), default=0))
    adaptive_thetas = tuple(
        sorted(
            {
                safe_chernoff_order(
                    score_alpha=float(score_alpha),
                    load=int(load),
                )
                for load in range(max_future_load + 1)
            }
        )
    )
    adaptive_polymers = {
        float(adaptive_theta): enumerate_low_weight_polymers(
            family,
            theta=float(adaptive_theta),
        )
        for adaptive_theta in adaptive_thetas
    }
    kernel_profiles = {
        float(adaptive_theta): kernel_boundary_partition_profile(
            family,
            theta=float(adaptive_theta),
        )
        for adaptive_theta in adaptive_thetas
    }

    def topology(polymer: LowWeightPolymer) -> tuple[object, ...]:
        return (
            polymer.column_indices,
            int(polymer.start),
            int(polymer.stop),
            int(polymer.detector_shift_mask),
            int(polymer.logical_shift_mask),
            int(polymer.size),
        )

    expected_topology = tuple(topology(polymer) for polymer in polymers)
    for adaptive_theta, theta_polymers in adaptive_polymers.items():
        if tuple(topology(polymer) for polymer in theta_polymers) != expected_topology:
            raise AssertionError(
                f"polymer topology changed with theta={adaptive_theta:.12g}"
            )
    rho_values = tuple(float(value) for value in rhos)
    optimization_values = tuple(float(value) for value in optimization_rhos)
    calculation_rhos = tuple(dict.fromkeys((*rho_values, *optimization_values)))
    cap_values = tuple(int(value) for value in K_values)
    for rho in calculation_rhos:
        if not 0.0 < rho < 1.0:
            raise ValueError("every rho must lie strictly between zero and one")

    methods = (
        "exponential_product",
        "exact_ungrouped_product",
        "compatibility_ungrouped",
        "shift_grouped",
        "compatibility_shift_grouped",
        "adaptive_compatibility_shift_grouped",
        "adaptive_full_kernel_grouped",
    )
    integrated_log_terms = {
        rho: {method: [] for method in methods} for rho in calculation_rhos
    }
    trimmed_methods = (
        "adaptive_compatibility_shift_grouped",
        "adaptive_full_kernel_grouped",
    )
    trimmed_cap_totals = {
        method: {int(cap): 0.0 for cap in cap_values} for method in trimmed_methods
    }
    per_cut: list[dict[str, object]] = []
    start_time = time.monotonic()
    column_count = int(len(family.columns))

    for cut in range(column_count):
        active_rows = tuple(
            _iter_set_bits(int(family.layout.active_masks_after_column[cut]))
        )
        boundary_bits = int(len(active_rows) + int(family.logical_rows))
        if boundary_bits > int(max_boundary_bits):
            raise ValueError(
                f"cut {cut} needs {boundary_bits} boundary bits, above "
                f"--max-boundary-bits={int(max_boundary_bits)}"
            )
        live = tuple(
            polymer
            for polymer in polymers
            if int(polymer.start) <= cut < int(polymer.stop)
        )
        compatibility_structure = low_weight_compatibility_structure(
            family,
            polymers=polymers,
            cut=int(cut),
        )
        if (
            int(compatibility_structure["live_singletons"])
            + int(compatibility_structure["live_pairs"])
            != len(live)
        ):
            raise AssertionError("compatibility audit and live-polymer count disagree")
        shift_activities = tuple(
            (
                _compress_boundary_shift(
                    polymer,
                    active_rows=active_rows,
                    logical_rows=int(family.logical_rows),
                ),
                float(polymer.activity),
            )
            for polymer in live
        )
        partition = xor_subset_partition(
            boundary_bits=int(boundary_bits),
            shift_activities=shift_activities,
        )
        compatible_partition, compatible_partition_metadata = (
            compatible_boundary_partition(
                family,
                polymers=polymers,
                cut=int(cut),
                active_rows=active_rows,
                logical_rows=int(family.logical_rows),
            )
        )
        detector_shift_thetas = shift_specific_chernoff_orders(
            family,
            cut=int(cut),
            active_rows=active_rows,
            score_alpha=float(score_alpha),
        )
        theta_partitions: dict[float, np.ndarray] = {}
        for adaptive_theta in sorted(set(detector_shift_thetas)):
            if math.isclose(
                float(adaptive_theta),
                float(theta),
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                theta_partitions[float(adaptive_theta)] = compatible_partition
            else:
                theta_partitions[float(adaptive_theta)] = (
                    compatible_boundary_partition(
                        family,
                        polymers=adaptive_polymers[float(adaptive_theta)],
                        cut=int(cut),
                        active_rows=active_rows,
                        logical_rows=int(family.logical_rows),
                    )[0]
                )
        adaptive_partition = np.zeros_like(compatible_partition)
        detector_state_count = 1 << len(active_rows)
        logical_state_count = 1 << int(family.logical_rows)
        for detector_shift, adaptive_theta in enumerate(detector_shift_thetas):
            source = theta_partitions[float(adaptive_theta)]
            for logical_shift in range(logical_state_count):
                encoded = int(detector_shift | (logical_shift * detector_state_count))
                adaptive_partition[encoded] = source[encoded]
        adaptive_kernel_partition = np.zeros_like(compatible_partition)
        for detector_shift, adaptive_theta in enumerate(detector_shift_thetas):
            source = kernel_profiles[float(adaptive_theta)][0][cut]
            if source.shape != adaptive_kernel_partition.shape:
                raise AssertionError("kernel and polymer boundary spaces disagree")
            for logical_shift in range(logical_state_count):
                encoded = int(detector_shift | (logical_shift * detector_state_count))
                adaptive_kernel_partition[encoded] = source[encoded]
        adaptive_tolerance = float(
            1e-10
            * max(
                1.0,
                float(np.max(np.abs(compatible_partition), initial=0.0)),
            )
        )
        if np.any(adaptive_partition > compatible_partition + adaptive_tolerance):
            raise AssertionError("shift-specific theta increased the partition majorant")
        if np.any(adaptive_partition > adaptive_kernel_partition + adaptive_tolerance):
            raise AssertionError("full kernel partition missed a low-weight family")
        cut_trimmed_bounds = {
            "adaptive_compatibility_shift_grouped": {
                str(int(cap)): trimmed_spectrum_bound(
                    adaptive_partition[1:],
                    K=int(cap),
                )
                for cap in cap_values
            },
            "adaptive_full_kernel_grouped": {
                str(int(cap)): trimmed_spectrum_bound(
                    adaptive_kernel_partition[1:],
                    K=int(cap),
                )
                for cap in cap_values
            },
        }
        for method in trimmed_methods:
            for cap in cap_values:
                trimmed_cap_totals[method][int(cap)] += float(
                    cut_trimmed_bounds[method][str(int(cap))]
                )
        if float(np.sum(compatible_partition, dtype=np.float64)) > float(
            np.sum(partition, dtype=np.float64)
        ) + 1e-9:
            raise AssertionError("compatibility increased the subset partition")
        reachable_nonzero = int(np.count_nonzero(partition[1:] > 0.0))
        compatible_reachable_nonzero = int(
            np.count_nonzero(compatible_partition[1:] > 0.0)
        )
        cut_rhos: dict[str, object] = {}

        for rho in calculation_rhos:
            xi_partial = float(math.fsum(activity**rho for _, activity in shift_activities))
            exact_log_partition = float(
                math.fsum(math.log1p(activity**rho) for _, activity in shift_activities)
            )
            compatible_moment = compatible_ungrouped_moment(
                family,
                polymers=polymers,
                cut=int(cut),
                rho=float(rho),
            )
            grouped_moment = float(np.sum(np.power(partition[1:], rho), dtype=np.float64))
            compatible_grouped_moment = float(
                np.sum(np.power(compatible_partition[1:], rho), dtype=np.float64)
            )
            adaptive_grouped_moment = float(
                np.sum(np.power(adaptive_partition[1:], rho), dtype=np.float64)
            )
            adaptive_kernel_moment = float(
                np.sum(
                    np.power(adaptive_kernel_partition[1:], rho),
                    dtype=np.float64,
                )
            )
            rooted_logs = {
                "exponential_product": float(_log_expm1(xi_partial) / rho),
                "exact_ungrouped_product": float(
                    _log_expm1(exact_log_partition) / rho
                ),
                "compatibility_ungrouped": (
                    float(math.log(compatible_moment) / rho)
                    if compatible_moment > 0.0
                    else float("-inf")
                ),
                "shift_grouped": (
                    float(math.log(grouped_moment) / rho)
                    if grouped_moment > 0.0
                    else float("-inf")
                ),
                "compatibility_shift_grouped": (
                    float(math.log(compatible_grouped_moment) / rho)
                    if compatible_grouped_moment > 0.0
                    else float("-inf")
                ),
                "adaptive_compatibility_shift_grouped": (
                    float(math.log(adaptive_grouped_moment) / rho)
                    if adaptive_grouped_moment > 0.0
                    else float("-inf")
                ),
                "adaptive_full_kernel_grouped": (
                    float(math.log(adaptive_kernel_moment) / rho)
                    if adaptive_kernel_moment > 0.0
                    else float("-inf")
                ),
            }
            rooted = {
                method: (
                    float(math.exp(rooted_logs[method]))
                    if rooted_logs[method] < 700.0
                    else None
                )
                for method in methods
            }
            for method in methods:
                integrated_log_terms[rho][method].append(float(rooted_logs[method]))
            if rho in rho_values:
                exponential_moment = (
                    float(math.expm1(xi_partial)) if xi_partial < 700.0 else None
                )
                exact_ungrouped_moment = (
                    float(math.expm1(exact_log_partition))
                    if exact_log_partition < 700.0
                    else None
                )
                cut_rhos[f"{rho:.12g}"] = {
                    "xi_partial": float(xi_partial),
                    "exponential_product_moment": exponential_moment,
                    "exact_ungrouped_product_moment": exact_ungrouped_moment,
                    "compatibility_ungrouped_moment": float(compatible_moment),
                    "shift_grouped_moment": float(grouped_moment),
                    "compatibility_shift_grouped_moment": float(
                        compatible_grouped_moment
                    ),
                    "adaptive_compatibility_shift_grouped_moment": float(
                        adaptive_grouped_moment
                    ),
                    "adaptive_full_kernel_grouped_moment": float(
                        adaptive_kernel_moment
                    ),
                    "cap_term_before_K": rooted,
                    "log10_cap_term_before_K": {
                        method: float(rooted_logs[method] / math.log(10.0))
                        for method in methods
                    },
                }

        per_cut.append(
            {
                "cut_after_column": int(cut),
                "active_detector_rows": int(len(active_rows)),
                "boundary_bits": int(boundary_bits),
                "live_polymers": int(len(live)),
                "reachable_nonzero_shifts": int(reachable_nonzero),
                "partition_total": float(np.sum(partition, dtype=np.float64)),
                "partition_zero_shift": float(partition[0]),
                "compatible_partition_total": float(
                    np.sum(compatible_partition, dtype=np.float64)
                ),
                "compatible_partition_zero_shift": float(compatible_partition[0]),
                "compatible_reachable_nonzero_shifts": int(
                    compatible_reachable_nonzero
                ),
                "compatible_partition_metadata": compatible_partition_metadata,
                "shift_specific_theta_histogram": {
                    f"{adaptive_theta:.12g}": int(count)
                    for adaptive_theta, count in sorted(
                        Counter(detector_shift_thetas).items()
                    )
                },
                "adaptive_full_kernel_partition_total": float(
                    np.sum(adaptive_kernel_partition, dtype=np.float64)
                ),
                "adaptive_full_kernel_partition_zero_shift": float(
                    adaptive_kernel_partition[0]
                ),
                "kernel_state_counts_by_theta": {
                    f"{adaptive_theta:.12g}": int(
                        kernel_profiles[float(adaptive_theta)][1][cut]
                    )
                    for adaptive_theta in adaptive_thetas
                },
                "trimmed_cap_bound_by_K": cut_trimmed_bounds,
                "compatibility_structure": compatibility_structure,
                "rho": cut_rhos,
            }
        )

        if (
            progress is not None
            and int(progress_every_cuts) > 0
            and ((cut + 1) % int(progress_every_cuts) == 0 or cut + 1 == column_count)
        ):
            elapsed = float(time.monotonic() - start_time)
            rate = float((cut + 1) / elapsed) if elapsed > 0.0 else float("inf")
            remaining = int(column_count - cut - 1)
            eta = float(remaining / rate) if rate > 0.0 else 0.0
            progress(
                f"cuts={cut + 1}/{column_count} "
                f"elapsed_s={elapsed:.2f} eta_s={eta:.2f}"
            )

    cap_rhs: dict[str, object] = {}
    reduction: dict[str, object] = {}
    cut_term_log10_summary: dict[str, object] = {}
    for rho in rho_values:
        rho_key = f"{rho:.12g}"
        cap_rhs[rho_key] = {
            method: {
                str(cap): _cap_rhs_from_log_terms(
                    integrated_log_terms[rho][method],
                    rho=float(rho),
                    K=int(cap),
                )
                for cap in cap_values
            }
            for method in methods
        }
        log_totals = {
            method: _logsumexp(integrated_log_terms[rho][method])
            for method in methods
        }
        exact_log_reduction = float(
            log_totals["exact_ungrouped_product"] - log_totals["shift_grouped"]
        )
        compatibility_log_reduction = float(
            log_totals["exact_ungrouped_product"]
            - log_totals["compatibility_ungrouped"]
        )
        combined_log_reduction = float(
            log_totals["shift_grouped"]
            - log_totals["compatibility_shift_grouped"]
        )
        adaptive_log_reduction = float(
            log_totals["compatibility_shift_grouped"]
            - log_totals["adaptive_compatibility_shift_grouped"]
        )
        higher_order_log_factor = float(
            log_totals["adaptive_full_kernel_grouped"]
            - log_totals["adaptive_compatibility_shift_grouped"]
        )
        exponential_log_reduction = float(
            log_totals["exponential_product"] - log_totals["shift_grouped"]
        )
        reduction[rho_key] = {
            "exponential_over_shift_grouped": (
                float(math.exp(exponential_log_reduction))
                if exponential_log_reduction < 700.0
                else None
            ),
            "exact_ungrouped_over_shift_grouped": (
                float(math.exp(exact_log_reduction))
                if exact_log_reduction < 700.0
                else None
            ),
            "log10_exponential_over_shift_grouped": float(
                exponential_log_reduction / math.log(10.0)
            ),
            "log10_exact_ungrouped_over_shift_grouped": float(
                exact_log_reduction / math.log(10.0)
            ),
            "exact_ungrouped_over_compatibility_ungrouped": (
                float(math.exp(compatibility_log_reduction))
                if compatibility_log_reduction < 700.0
                else None
            ),
            "log10_exact_ungrouped_over_compatibility_ungrouped": float(
                compatibility_log_reduction / math.log(10.0)
            ),
            "shift_grouped_over_compatibility_shift_grouped": (
                float(math.exp(combined_log_reduction))
                if combined_log_reduction < 700.0
                else None
            ),
            "log10_shift_grouped_over_compatibility_shift_grouped": float(
                combined_log_reduction / math.log(10.0)
            ),
            "compatibility_shift_grouped_over_adaptive": (
                float(math.exp(adaptive_log_reduction))
                if adaptive_log_reduction < 700.0
                else None
            ),
            "log10_compatibility_shift_grouped_over_adaptive": float(
                adaptive_log_reduction / math.log(10.0)
            ),
            "adaptive_full_kernel_over_size_1_2": (
                float(math.exp(higher_order_log_factor))
                if higher_order_log_factor < 700.0
                else None
            ),
            "log10_adaptive_full_kernel_over_size_1_2": float(
                higher_order_log_factor / math.log(10.0)
            ),
        }
        cut_term_log10_summary[rho_key] = {
            method: _log10_series_summary(
                tuple(
                    float(value / math.log(10.0))
                    for value in integrated_log_terms[rho][method]
                )
            )
            for method in methods
        }

    rho_optimization: dict[str, object] | None = None
    if optimization_values:
        minima: dict[str, object] = {}
        for method in methods:
            minima[method] = {}
            for cap in cap_values:
                candidates = tuple(
                    (
                        float(rho),
                        _cap_rhs_from_log_terms(
                            integrated_log_terms[rho][method],
                            rho=float(rho),
                            K=int(cap),
                        ),
                    )
                    for rho in optimization_values
                )
                best_rho, best_result = min(
                    candidates,
                    key=lambda item: float(item[1]["log10"]),
                )
                minima[method][str(int(cap))] = {
                    "rho": float(best_rho),
                    **best_result,
                }

        critical_K: dict[str, object] = {}
        for method in methods:
            candidates = tuple(
                (
                    float(rho),
                    float(rho * _logsumexp(integrated_log_terms[rho][method])),
                )
                for rho in optimization_values
            )
            best_rho, best_log_K = min(candidates, key=lambda item: item[1])
            best_value = float(math.exp(best_log_K)) if best_log_K < 700.0 else None
            critical_K[method] = {
                "rho": float(best_rho),
                "continuous_threshold_K": best_value,
                "log10_continuous_threshold_K": float(
                    best_log_K / math.log(10.0)
                ),
                "first_integer_strictly_above": (
                    int(math.floor(best_value)) + 1 if best_value is not None else None
                ),
            }

        rho_optimization = {
            "grid": {
                "minimum": float(min(optimization_values)),
                "maximum": float(max(optimization_values)),
                "count": int(len(optimization_values)),
            },
            "minimum_cap_rhs_by_K": minima,
            "critical_K_for_fractional_rhs_below_one": critical_K,
            "critical_K_for_partial_rhs_below_one": {
                method: values
                for method, values in critical_K.items()
                if method != "adaptive_full_kernel_grouped"
            },
        }

    size_1_2_methods = tuple(
        method for method in methods if method != "adaptive_full_kernel_grouped"
    )
    result = {
        "schema_version": 2,
        "description": (
            "Exact size-1/2 compatibility/shift partitions and an all-size "
            "completed-kernel majorant for visible polymer families."
        ),
        "size_1_2_methods_include_higher_polymers": False,
        "adaptive_full_kernel_method_includes_all_vector_weights": True,
        "adaptive_full_kernel_method_overcounts_invisible_components": True,
        "compatibility_enforced_for_ungrouped_method": True,
        "compatibility_enforced_for_shift_grouped_method": False,
        "compatibility_enforced_for_compatibility_shift_grouped_method": True,
        "shift_specific_chernoff_order_used_for_adaptive_method": True,
        "zero_boundary_shift_excluded_from_fractional_moment": True,
        "theta": float(theta),
        "score_alpha": float(score_alpha),
        "shift_specific_thetas": [float(value) for value in adaptive_thetas],
        "max_boundary_bits": int(
            max((int(row["boundary_bits"]) for row in per_cut), default=0)
        ),
        "polymer_universe": {
            "size_1": int(sum(polymer.size == 1 for polymer in polymers)),
            "size_2": int(sum(polymer.size == 2 for polymer in polymers)),
        },
        "fractional_cap_rhs_by_method": cap_rhs,
        "cap_rhs_from_sizes_1_2": {
            rho_key: {
                method: values[method] for method in size_1_2_methods
            }
            for rho_key, values in cap_rhs.items()
        },
        "trimmed_cap_rhs_by_K": {
            method: {
                str(int(cap)): float(trimmed_cap_totals[method][int(cap)])
                for cap in cap_values
            }
            for method in trimmed_methods
        },
        "integrated_reduction_factors": reduction,
        "cut_term_log10_summary": cut_term_log10_summary,
        "per_cut": per_cut,
    }
    if rho_optimization is not None:
        result["rho_optimization"] = rho_optimization
    return result


def low_weight_polymer_profile(
    family: LoadedProgressiveFamily,
    *,
    theta: float,
    rhos: Sequence[float],
    K_values: Sequence[int] = DEFAULT_K_VALUES,
    progress_every_pairs: int = 0,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Enumerate all visible open-prefix polymers of sizes one and two."""

    column_count = int(len(family.columns))
    supports = tuple(int(column.detector_support_mask) for column in family.columns)
    logical_masks = tuple(
        int(_column_logical_mask(family, index)) for index in range(column_count)
    )
    probabilities = tuple(
        float(_column_probability(family, index)) for index in range(column_count)
    )
    activities = tuple(
        chernoff_activity(probability, float(theta)) for probability in probabilities
    )
    last = tuple(int(value) for value in family.layout.detector_last_column)
    rho_values = tuple(float(value) for value in rhos)
    cap_values = tuple(int(value) for value in K_values)
    calculation_rhos = tuple(dict.fromkeys((*rho_values, 1.0)))

    count_differences = {
        1: [0.0 for _ in range(column_count + 1)],
        2: [0.0 for _ in range(column_count + 1)],
    }
    xi_differences = {
        rho: {
            1: [0.0 for _ in range(column_count + 1)],
            2: [0.0 for _ in range(column_count + 1)],
        }
        for rho in calculation_rhos
    }
    lifetime_histograms = {1: Counter(), 2: Counter()}
    unique_counts = {1: 0, 2: 0}

    for index, support in enumerate(supports):
        if support:
            stop = _minimum_last_touch(support, last)
        elif logical_masks[index]:
            stop = int(column_count)
        else:
            continue
        start = int(index)
        if start >= stop:
            continue
        lifetime = int(stop - start)
        unique_counts[1] += 1
        lifetime_histograms[1][lifetime] += 1
        _add_interval(count_differences[1], start=start, stop=stop, value=1.0)
        for rho in calculation_rhos:
            _add_interval(
                xi_differences[rho][1],
                start=start,
                stop=stop,
                value=float(activities[index] ** rho),
            )

    encoded_pairs: set[int] = set()
    for touches in family.layout.row_touch_columns:
        row_columns = tuple(int(value) for value in touches)
        for left_offset, left in enumerate(row_columns):
            for right in row_columns[left_offset + 1 :]:
                a, b = (left, int(right)) if left < int(right) else (int(right), left)
                encoded_pairs.add(int(a * column_count + b))
    ordered_pairs = tuple(sorted(encoded_pairs))

    start_time = time.monotonic()
    for pair_index, encoded in enumerate(ordered_pairs, start=1):
        left, right = divmod(int(encoded), column_count)
        common = int(supports[left] & supports[right])
        if common == 0:
            raise AssertionError("candidate pair does not share a detector row")
        symmetric_difference = int(supports[left] ^ supports[right])
        start = _minimum_last_touch(common, last)
        stop = (
            _minimum_last_touch(symmetric_difference, last)
            if symmetric_difference
            else int(column_count)
        )
        visible = bool(symmetric_difference or (logical_masks[left] ^ logical_masks[right]))
        if visible and start < stop:
            lifetime = int(stop - start)
            unique_counts[2] += 1
            lifetime_histograms[2][lifetime] += 1
            _add_interval(count_differences[2], start=start, stop=stop, value=1.0)
            pair_activity = float(activities[left] * activities[right])
            for rho in calculation_rhos:
                _add_interval(
                    xi_differences[rho][2],
                    start=start,
                    stop=stop,
                    value=float(pair_activity**rho),
                )

        if (
            progress is not None
            and int(progress_every_pairs) > 0
            and (
                pair_index % int(progress_every_pairs) == 0
                or pair_index == len(ordered_pairs)
            )
        ):
            elapsed = float(time.monotonic() - start_time)
            rate = float(pair_index / elapsed) if elapsed > 0.0 else float("inf")
            remaining = int(len(ordered_pairs) - pair_index)
            eta = float(remaining / rate) if rate > 0.0 else 0.0
            progress(
                f"pairs={pair_index}/{len(ordered_pairs)} "
                f"elapsed_s={elapsed:.2f} eta_s={eta:.2f}"
            )

    count_series = {
        size: _integrate_differences(count_differences[size]) for size in (1, 2)
    }
    xi_series = {
        rho: {
            size: _integrate_differences(xi_differences[rho][size]) for size in (1, 2)
        }
        for rho in calculation_rhos
    }

    xi_summary: dict[str, object] = {}
    partial_cap_rhs: dict[str, object] = {}
    for rho in rho_values:
        first = xi_series[rho][1]
        second = xi_series[rho][2]
        combined = tuple(float(a + b) for a, b in zip(first, second, strict=True))
        xi_summary[f"{rho:.12g}"] = {
            "size_1": _series_summary(first),
            "size_2": _series_summary(second),
            "sizes_1_2_lower_bound": _series_summary(combined),
        }
        partial_cap_rhs[f"{rho:.12g}"] = {
            str(int(cap)): _partial_cap_rhs(combined, rho=float(rho), K=int(cap))
            for cap in cap_values
        }
    rho_one_first = xi_series[1.0][1]
    rho_one_second = xi_series[1.0][2]
    rho_one_combined = tuple(
        float(a + b) for a, b in zip(rho_one_first, rho_one_second, strict=True)
    )

    return {
        "theta": float(theta),
        "candidate_connected_pairs": int(len(ordered_pairs)),
        "unique_visible_polymers": {
            "size_1": int(unique_counts[1]),
            "size_2": int(unique_counts[2]),
        },
        "lifetime_histogram_cuts": {
            "size_1": {
                str(int(key)): int(value)
                for key, value in sorted(lifetime_histograms[1].items())
            },
            "size_2": {
                str(int(key)): int(value)
                for key, value in sorted(lifetime_histograms[2].items())
            },
        },
        "visible_polymer_cut_counts": {
            "size_1": _series_summary(count_series[1]),
            "size_2": _series_summary(count_series[2]),
            "sizes_1_2": _series_summary(
                tuple(
                    float(a + b)
                    for a, b in zip(count_series[1], count_series[2], strict=True)
                )
            ),
        },
        "xi_partial_by_rho": xi_summary,
        "lower_bound_on_product_peierls_rhs_from_sizes_1_2": partial_cap_rhs,
        "xi_partial_rho_1_diagnostic": {
            "size_1": _series_summary(rho_one_first),
            "size_2": _series_summary(rho_one_second),
            "sizes_1_2_lower_bound": _series_summary(rho_one_combined),
        },
        "activity_range": {
            "min": float(min(activities, default=0.0)),
            "max": float(max(activities, default=0.0)),
        },
    }


def build_family_profile(
    family: LoadedProgressiveFamily,
    *,
    p_location: float,
    score_alpha: float,
    rhos: Sequence[float] = DEFAULT_RHOS,
    K_values: Sequence[int] = DEFAULT_K_VALUES,
    progress_every_pairs: int = 0,
    aggregate_boundary_shifts: bool = False,
    max_boundary_bits: int = 16,
    progress_every_cuts: int = 25,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Build the deterministic structural and low-weight profile for one scope."""

    column_weights = tuple(
        int(column.detector_support_mask.bit_count()) for column in family.columns
    )
    row_degrees = tuple(len(touches) for touches in family.layout.row_touch_columns)
    probabilities = tuple(
        float(_column_probability(family, index)) for index in range(len(family.columns))
    )
    future_loads = future_active_load_profile(family)
    max_column_weight = int(max(column_weights, default=0))
    max_future_load = int(max(future_loads, default=0))
    theta_column = safe_chernoff_order(
        score_alpha=float(score_alpha),
        load=int(max_column_weight),
    )
    theta_measured = safe_chernoff_order(
        score_alpha=float(score_alpha),
        load=int(max_future_load),
    )
    low_weight = low_weight_polymer_profile(
        family,
        theta=float(theta_measured),
        rhos=tuple(float(value) for value in rhos),
        K_values=tuple(int(value) for value in K_values),
        progress_every_pairs=int(progress_every_pairs),
        progress=progress,
    )
    if bool(aggregate_boundary_shifts):
        aggregation = boundary_shift_aggregation_profile(
            family,
            theta=float(theta_measured),
            rhos=tuple(float(value) for value in rhos),
            score_alpha=float(score_alpha),
            K_values=tuple(int(value) for value in K_values),
            optimization_rhos=DEFAULT_OPTIMIZATION_RHOS,
            max_boundary_bits=int(max_boundary_bits),
            progress_every_cuts=int(progress_every_cuts),
            progress=progress,
        )
        if aggregation["polymer_universe"] != low_weight["unique_visible_polymers"]:
            raise AssertionError("aggregation and lifetime polymer counts disagree")
        low_weight["boundary_shift_aggregation"] = aggregation
    return {
        "backend": str(family.backend),
        "scope": str(family.scope),
        "p_location": float(p_location),
        "column_order": str(family.column_order_name),
        "score_alpha": float(score_alpha),
        "input_checksum_sha256": _family_checksum(family),
        "matrix": {
            "detector_rows": int(family.matrix_rows),
            "logical_rows": int(family.logical_rows),
            "columns": int(family.matrix_cols),
            "detector_edges": int(family.edge_count),
        },
        "probability_range": {
            "min": float(min(probabilities, default=0.0)),
            "max": float(max(probabilities, default=0.0)),
        },
        "detector_column_weight": {
            "histogram": _integer_histogram(column_weights),
            "maximum": int(max_column_weight),
        },
        "detector_row_degree": {
            "histogram": _integer_histogram(row_degrees),
            "maximum": int(max(row_degrees, default=0)),
            "mean": float(math.fsum(row_degrees) / len(row_degrees)) if row_degrees else 0.0,
        },
        "active_width": {
            "maximum": int(family.layout.max_active_detectors),
            "histogram_over_cuts": _integer_histogram(
                tuple(int(value) for value in family.layout.active_width_profile[1:])
            ),
        },
        "future_active_score_load": {
            "maximum": int(max_future_load),
            "histogram_over_cuts": _integer_histogram(future_loads),
        },
        "safe_chernoff_order": {
            "from_full_column_weight": float(theta_column),
            "from_measured_future_active_load": float(theta_measured),
        },
        "low_weight_open_polymers": low_weight,
    }


def _parse_csv_values(text: str, *, cast: Callable[[str], object]) -> tuple[object, ...]:
    values = tuple(part.strip() for part in str(text).split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one comma-separated value")
    return tuple(cast(value) for value in values)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile cutwise Finner loads, visible size-1/2 polymers, exact "
            "compatibility, and optional all-size kernel spectra."
        ),
        epilog=(
            "See docs/COMMANDS.md, docs/BOUNDED_HYPERGRAPH_OVERLAP.md, and "
            "docs/ADAPTIVE_KERNEL_SPECTRUM.md."
        ),
    )
    parser.add_argument("--backend", default="rotated_surface_d3")
    parser.add_argument("--p-location", type=float, default=0.001)
    parser.add_argument("--scopes", default="memory_X,memory_Z")
    parser.add_argument("--column-order", default="deadline_reorder")
    parser.add_argument("--score-alpha", type=float, default=0.8)
    parser.add_argument("--rhos", default="0.5,0.75,0.99")
    parser.add_argument("--K-values", default="16,512,1024,8192")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--progress-every-pairs", type=int, default=250000)
    parser.add_argument("--aggregate-boundary-shifts", action="store_true")
    parser.add_argument("--max-boundary-bits", type=int, default=16)
    parser.add_argument("--progress-every-cuts", type=int, default=25)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        scopes = tuple(str(value) for value in _parse_csv_values(args.scopes, cast=str))
        rhos = tuple(float(value) for value in _parse_csv_values(args.rhos, cast=float))
        K_values = tuple(int(value) for value in _parse_csv_values(args.K_values, cast=int))
        if any(not 0.0 < value < 1.0 for value in rhos):
            raise ValueError("every rho must lie strictly between zero and one")
        if any(value < 1 for value in K_values):
            raise ValueError("every K value must be positive")
        if float(args.p_location) <= 0.0:
            raise ValueError("p_location must be positive")
        if float(args.score_alpha) < 0.0 or not math.isfinite(float(args.score_alpha)):
            raise ValueError("score_alpha must be finite and non-negative")

        profiles: list[dict[str, object]] = []
        total_scopes = len(scopes)
        overall_start = time.monotonic()
        for scope_index, scope in enumerate(scopes, start=1):
            scope_start = time.monotonic()
            print(
                f"[overlap-profile] scope={scope} phase=load "
                f"count={scope_index}/{total_scopes}",
                file=sys.stderr,
                flush=True,
            )
            family = load_dem_family(
                backend=str(args.backend),
                p_location=float(args.p_location),
                scope=str(scope),
                column_order=str(args.column_order),
            )

            def emit_progress(message: str) -> None:
                phase = "aggregation" if str(message).startswith("cuts=") else "pairs"
                print(
                    f"[overlap-profile] scope={scope} phase={phase} {message}",
                    file=sys.stderr,
                    flush=True,
                )

            profile = build_family_profile(
                family,
                p_location=float(args.p_location),
                score_alpha=float(args.score_alpha),
                rhos=rhos,
                K_values=K_values,
                progress_every_pairs=int(args.progress_every_pairs),
                aggregate_boundary_shifts=bool(args.aggregate_boundary_shifts),
                max_boundary_bits=int(args.max_boundary_bits),
                progress_every_cuts=int(args.progress_every_cuts),
                progress=emit_progress,
            )
            profiles.append(profile)
            elapsed = float(time.monotonic() - scope_start)
            overall_elapsed = float(time.monotonic() - overall_start)
            mean_scope = float(overall_elapsed / scope_index)
            eta = float(mean_scope * (total_scopes - scope_index))
            print(
                f"[overlap-profile] scope={scope} phase=complete "
                f"elapsed_s={elapsed:.2f} overall_s={overall_elapsed:.2f} eta_s={eta:.2f}",
                file=sys.stderr,
                flush=True,
            )

        configuration = {
            "backend": str(args.backend),
            "p_location": float(args.p_location),
            "scopes": list(scopes),
            "column_order": str(args.column_order),
            "score_alpha": float(args.score_alpha),
            "rhos": [float(value) for value in rhos],
            "K_values": [int(value) for value in K_values],
        }
        if bool(args.aggregate_boundary_shifts):
            configuration.update(
                {
                    "aggregate_boundary_shifts": True,
                    "max_boundary_bits": int(args.max_boundary_bits),
                }
            )
        payload = {
            "schema_version": int(SCHEMA_VERSION),
            "configuration": configuration,
            "profiles": profiles,
        }
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"[overlap-profile] wrote={output_path} profiles={len(profiles)}",
            file=sys.stderr,
            flush=True,
        )
        return 0
    except Exception as exc:
        print(f"frontier-overlap-profile: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
