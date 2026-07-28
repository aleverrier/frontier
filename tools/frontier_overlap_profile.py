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
from pathlib import Path
from typing import Callable, Iterable, Sequence

from tools.dem_loader import LoadedProgressiveFamily, load_dem_family


SCHEMA_VERSION = 1
DEFAULT_SCOPES = ("memory_X", "memory_Z")
DEFAULT_RHOS = (0.5, 0.75, 0.99)
DEFAULT_K_VALUES = (16, 512, 1024, 8192)


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
            "Profile exact cutwise Finner loads and visible size-1/2 open-prefix "
            "polymers for an ordered Frontier DEM."
        ),
        epilog="See docs/COMMANDS.md and docs/BOUNDED_HYPERGRAPH_OVERLAP.md.",
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
                print(
                    f"[overlap-profile] scope={scope} phase=pairs {message}",
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

        payload = {
            "schema_version": int(SCHEMA_VERSION),
            "configuration": {
                "backend": str(args.backend),
                "p_location": float(args.p_location),
                "scopes": list(scopes),
                "column_order": str(args.column_order),
                "score_alpha": float(args.score_alpha),
                "rhos": [float(value) for value in rhos],
                "K_values": [int(value) for value in K_values],
            },
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
