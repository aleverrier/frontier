from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .arikan import ordering_permutation
from .dynamic_frozen import compute_gap_profile
from .exact_map import exact_map_for_syndrome
from .gf2 import dense_mod2, matvec_mod2, rank_mod2
from .plotting import plot_list_size_scatter
from .sc_posterior import estimate_reliability_monte_carlo
from .scl_decoder import decode_scl


@dataclass(frozen=True)
class SmallBenchmarkConfig:
    block_length: int = 8
    num_instances: int = 6
    trials_per_instance: int = 24
    target_free_bits: int = 4
    num_observables: int = 2
    reliability_samples: int = 128
    max_list_size: int = 16
    seed: int = 20260413
    orderings: tuple[str, ...] = ("natural", "reverse", "bit_reversed", "random0")
    progress_every_trials: int = 20


@dataclass(frozen=True)
class SmallInstance:
    name: str
    m_det: np.ndarray
    o_obs: np.ndarray | None
    priors: np.ndarray


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _best_logical_key(log_masses: dict[int, float]) -> int | None:
    if not log_masses:
        return None
    return min(log_masses.items(), key=lambda item: (-item[1], item[0]))[0]


def generate_random_instance(
    *,
    block_length: int,
    target_free_bits: int,
    num_observables: int,
    rng: np.random.Generator,
    instance_index: int,
) -> SmallInstance:
    target_rank = int(block_length) - int(target_free_bits)
    if target_rank <= 0:
        raise ValueError("target_free_bits must be smaller than block_length")
    rows: list[np.ndarray] = []
    while len(rows) < target_rank:
        candidate = rng.integers(0, 2, size=block_length, dtype=np.uint8)
        if int(candidate.sum()) == 0:
            continue
        trial = candidate.reshape(1, -1) if not rows else np.vstack([rows, candidate.reshape(1, -1)])
        if rank_mod2(trial) > len(rows):
            rows.append(candidate)
    m_det = np.vstack(rows).astype(np.uint8, copy=False)
    o_obs = None
    if int(num_observables) > 0:
        obs_rows: list[np.ndarray] = []
        while len(obs_rows) < int(num_observables):
            candidate = rng.integers(0, 2, size=block_length, dtype=np.uint8)
            if int(candidate.sum()) == 0:
                continue
            trial = candidate.reshape(1, -1) if not obs_rows else np.vstack([obs_rows, candidate.reshape(1, -1)])
            if rank_mod2(trial) > len(obs_rows):
                obs_rows.append(candidate)
        o_obs = np.vstack(obs_rows).astype(np.uint8, copy=False)
    priors = rng.uniform(0.03, 0.22, size=block_length).astype(np.float64)
    return SmallInstance(
        name=f"random_{instance_index:03d}",
        m_det=m_det,
        o_obs=o_obs,
        priors=priors,
    )


def _ordered_view(
    instance: SmallInstance,
    ordering: str,
    *,
    rng_seed: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(rng_seed))
    permutation = ordering_permutation(instance.m_det.shape[1], ordering, rng=rng)
    matrix = instance.m_det[:, permutation].copy()
    observables = None if instance.o_obs is None else instance.o_obs[:, permutation].copy()
    priors = instance.priors[permutation].copy()
    return matrix, observables, priors, permutation


def run_small_exact_benchmark(config: SmallBenchmarkConfig, *, results_dir: str | Path) -> dict[str, object]:
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(config.seed))
    list_sizes = []
    size = 1
    while size <= int(config.max_list_size):
        list_sizes.append(size)
        size *= 2
    trial_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    instance_rows: list[dict[str, object]] = []
    scatter_fault_x: list[int] = []
    scatter_fault_y: list[int] = []
    scatter_logical_x: list[int] = []
    scatter_logical_y: list[int] = []
    aggregate_match: dict[tuple[str, int], dict[str, int]] = {}
    start_time = time.time()

    for instance_index in range(int(config.num_instances)):
        instance = generate_random_instance(
            block_length=config.block_length,
            target_free_bits=config.target_free_bits,
            num_observables=config.num_observables,
            rng=rng,
            instance_index=instance_index,
        )
        for ordering_index, ordering in enumerate(config.orderings):
            matrix, observables, priors, permutation = _ordered_view(
                instance,
                ordering,
                rng_seed=int(config.seed + 1000 * instance_index + ordering_index),
            )
            reliability = estimate_reliability_monte_carlo(
                priors,
                samples=int(config.reliability_samples),
                seed=int(config.seed + 100000 + 97 * instance_index + ordering_index),
            )
            decode_l1 = decode_scl(matrix, priors, matvec_mod2(matrix, np.zeros(matrix.shape[1], dtype=np.uint8)), observables=observables, list_size=1)
            gap = compute_gap_profile(
                length=matrix.shape[1],
                frozen_indices=decode_l1.dynamic_frozen.frozen_indices,
                reliability_scores=reliability.success_probability,
            )
            exhaustive_list_size = 1 << len(decode_l1.dynamic_frozen.info_indices)
            if exhaustive_list_size > int(config.max_list_size):
                raise ValueError(
                    f"instance {instance.name}/{ordering} has exhaustive list size {exhaustive_list_size}, "
                    f"which exceeds max_list_size={config.max_list_size}"
                )
            instance_rows.append(
                {
                    "instance": instance.name,
                    "ordering": ordering,
                    "matrix_rows": int(matrix.shape[0]),
                    "matrix_cols": int(matrix.shape[1]),
                    "observables_rows": 0 if observables is None else int(observables.shape[0]),
                    "rank_q": int(len(decode_l1.dynamic_frozen.frozen_indices)),
                    "free_bits": int(len(decode_l1.dynamic_frozen.info_indices)),
                    "g_max": int(gap.g_max),
                    "predicted_list_size": int(gap.predicted_list_size),
                    "permutation": ",".join(str(int(value)) for value in permutation.tolist()),
                }
            )
            for trial_index in range(int(config.trials_per_instance)):
                truth = (rng.random(matrix.shape[1]) < priors).astype(np.uint8)
                syndrome = matvec_mod2(matrix, truth)
                exact = exact_map_for_syndrome(matrix, priors, syndrome, observables=observables)
                exact_logical_key = _best_logical_key(exact.logical_log_masses)
                min_fault_list = None
                min_logical_list = None
                for list_size in list_sizes:
                    decoded = decode_scl(matrix, priors, syndrome, observables=observables, list_size=list_size)
                    fault_match = bool(np.array_equal(decoded.best_candidate.e, exact.best_error))
                    logical_match = exact_logical_key is None or decoded.best_candidate.logical_key == exact_logical_key
                    stats = aggregate_match.setdefault((ordering, list_size), {"trials": 0, "fault_matches": 0, "logical_matches": 0})
                    stats["trials"] += 1
                    stats["fault_matches"] += int(fault_match)
                    stats["logical_matches"] += int(logical_match)
                    if min_fault_list is None and fault_match:
                        min_fault_list = int(list_size)
                    if min_logical_list is None and logical_match:
                        min_logical_list = int(list_size)
                if min_fault_list is None or min_logical_list is None:
                    raise AssertionError("large-list SCL failed to recover the exact exhaustive answer")
                scatter_fault_x.append(int(gap.predicted_list_size))
                scatter_fault_y.append(int(min_fault_list))
                scatter_logical_x.append(int(gap.predicted_list_size))
                scatter_logical_y.append(int(min_logical_list))
                trial_rows.append(
                    {
                        "instance": instance.name,
                        "ordering": ordering,
                        "trial": int(trial_index),
                        "matrix_rows": int(matrix.shape[0]),
                        "matrix_cols": int(matrix.shape[1]),
                        "observables_rows": 0 if observables is None else int(observables.shape[0]),
                        "g_max": int(gap.g_max),
                        "predicted_list_size": int(gap.predicted_list_size),
                        "min_fault_list_size": int(min_fault_list),
                        "min_logical_list_size": int(min_logical_list),
                        "exact_feasible_count": int(exact.feasible_count),
                        "exact_logical_key": "" if exact_logical_key is None else int(exact_logical_key),
                    }
                )
                if config.progress_every_trials and len(trial_rows) % int(config.progress_every_trials) == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"[polar-dem small] completed_trials={len(trial_rows)} "
                        f"elapsed={elapsed:.1f}s latest_instance={instance.name} ordering={ordering}"
                    )

    for (ordering, list_size), stats in sorted(aggregate_match.items()):
        aggregate_rows.append(
            {
                "ordering": ordering,
                "list_size": int(list_size),
                "trials": int(stats["trials"]),
                "fault_match_rate": float(stats["fault_matches"]) / float(stats["trials"]),
                "logical_match_rate": float(stats["logical_matches"]) / float(stats["trials"]),
            }
        )

    _write_csv(
        out_dir / "small_trials.csv",
        [
            "instance",
            "ordering",
            "trial",
            "matrix_rows",
            "matrix_cols",
            "observables_rows",
            "g_max",
            "predicted_list_size",
            "min_fault_list_size",
            "min_logical_list_size",
            "exact_feasible_count",
            "exact_logical_key",
        ],
        trial_rows,
    )
    _write_csv(
        out_dir / "small_aggregate_by_list_size.csv",
        ["ordering", "list_size", "trials", "fault_match_rate", "logical_match_rate"],
        aggregate_rows,
    )
    _write_csv(
        out_dir / "small_instances.csv",
        ["instance", "ordering", "matrix_rows", "matrix_cols", "observables_rows", "rank_q", "free_bits", "g_max", "predicted_list_size", "permutation"],
        instance_rows,
    )

    plot_list_size_scatter(
        scatter_fault_x,
        scatter_fault_y,
        path=out_dir / "fig_fault_map_list_size_vs_predicted.png",
        title="Fault-MAP: minimum exact list size vs 2^{g_max}",
        ylabel="minimum L matching exact fault-MAP",
    )
    plot_list_size_scatter(
        scatter_logical_x,
        scatter_logical_y,
        path=out_dir / "fig_logical_map_list_size_vs_predicted.png",
        title="Logical-MAP: minimum exact list size vs 2^{g_max}",
        ylabel="minimum L matching exact logical-MAP",
    )

    summary = {
        "config": {
            "block_length": int(config.block_length),
            "num_instances": int(config.num_instances),
            "trials_per_instance": int(config.trials_per_instance),
            "target_free_bits": int(config.target_free_bits),
            "num_observables": int(config.num_observables),
            "reliability_samples": int(config.reliability_samples),
            "max_list_size": int(config.max_list_size),
            "seed": int(config.seed),
            "orderings": list(config.orderings),
        },
        "trial_count": len(trial_rows),
        "fault_predicted_vs_observed_corr": None
        if not trial_rows
        else float(
            np.corrcoef(
                np.log2(np.asarray(scatter_fault_x, dtype=np.float64)),
                np.log2(np.asarray(scatter_fault_y, dtype=np.float64)),
            )[0, 1]
        ),
        "logical_predicted_vs_observed_corr": None
        if not trial_rows
        else float(
            np.corrcoef(
                np.log2(np.asarray(scatter_logical_x, dtype=np.float64)),
                np.log2(np.asarray(scatter_logical_y, dtype=np.float64)),
            )[0, 1]
        ),
        "mean_min_fault_list_size": float(np.mean(np.asarray(scatter_fault_y, dtype=np.float64))),
        "mean_min_logical_list_size": float(np.mean(np.asarray(scatter_logical_y, dtype=np.float64))),
        "median_min_fault_list_size": float(np.median(np.asarray(scatter_fault_y, dtype=np.float64))),
        "median_min_logical_list_size": float(np.median(np.asarray(scatter_logical_y, dtype=np.float64))),
    }
    with (out_dir / "small_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    report_lines = [
        "# Polar DEM Small Exact Benchmark",
        "",
        f"- Standalone synthetic GF(2) instances only; no large-DEM window decoding is claimed here.",
        f"- Block length: `N = {config.block_length}`.",
        f"- Detector matrices: random exact-benchmark instances with `r = {config.block_length - config.target_free_bits}` rows and `N = {config.block_length}` columns.",
        f"- Observable matrices: `{config.num_observables}` rows when enabled.",
        f"- Orderings: `{', '.join(config.orderings)}`.",
        "",
        f"![Fault-MAP list size scatter]({(out_dir / 'fig_fault_map_list_size_vs_predicted.png').name})",
        "",
        (
            "Figure 1. Small exact benchmark on synthetic standalone GF(2) detector systems with "
            f"`N = {config.block_length}`, detector matrices of size `{config.block_length - config.target_free_bits} x {config.block_length}`, "
            f"and observable matrices of size `{config.num_observables} x {config.block_length}` when present. "
            "Each point is one `(instance, ordering, syndrome)` trial. The x-axis is the heuristic prediction `2^{g_max}` from the "
            "dynamic-frozen gap profile; the y-axis is the minimum SCL list size that exactly matches exhaustive fault-MAP. "
            "Both axes use base-2 logarithmic scales. Main quantitative takeaway: the current smoke benchmark is intended as a correctness/diagnostic "
            "check first, and the observed correlation can be read directly from the saved summary JSON."
        ),
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| fault predicted vs observed log2 correlation | `{summary['fault_predicted_vs_observed_corr']:.4f}` |",
        f"| logical predicted vs observed log2 correlation | `{summary['logical_predicted_vs_observed_corr']:.4f}` |",
        f"| mean minimum fault-MAP list size | `{summary['mean_min_fault_list_size']:.3f}` |",
        f"| mean minimum logical-MAP list size | `{summary['mean_min_logical_list_size']:.3f}` |",
        "",
        f"![Logical-MAP list size scatter]({(out_dir / 'fig_logical_map_list_size_vs_predicted.png').name})",
        "",
        (
            "Figure 2. The same small exact benchmark, but the y-axis is the minimum SCL list size needed to match exhaustive logical-MAP after grouping "
            "posterior mass by observable class. Main quantitative takeaway: comparing this figure against Figure 1 shows whether logical-MAP is easier or harder "
            "than full fault-MAP on the same standalone instances."
        ),
        "",
        "| ordering | list size | fault-match rate | logical-match rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in aggregate_rows:
        report_lines.append(
            f"| `{row['ordering']}` | `{row['list_size']}` | `{row['fault_match_rate']:.4f}` | `{row['logical_match_rate']:.4f}` |"
        )
    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n")
    return summary
