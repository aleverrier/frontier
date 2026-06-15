from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .adapters import DenseDemSide
from .arikan import ordering_permutation
from .dynamic_frozen import compute_gap_profile
from .gf2 import matmul_mod2
from .plotting import (
    plot_frozen_free_profile,
    plot_gap_profile,
    plot_gmax_histogram,
    plot_ordering_summary,
    plot_reliability_profile,
)
from .sc_posterior import estimate_reliability_monte_carlo
from .scl_decoder import decode_scl


@dataclass(frozen=True)
class LargeScanConfig:
    window_sizes: tuple[int, ...] = (32, 64)
    stride: int = 32
    max_windows_per_size: int = 3
    reliability_samples: int = 128
    seed: int = 20260413
    orderings: tuple[str, ...] = ("natural", "reverse", "bit_reversed", "random0")
    progress_every_windows: int = 4


def _metadata_window_permutation(metadata: object, window_columns: np.ndarray) -> np.ndarray:
    start = np.asarray(getattr(metadata, "column_round_start"))
    stop = np.asarray(getattr(metadata, "column_round_stop"))
    local = np.arange(window_columns.size, dtype=np.int64)
    cols = window_columns.astype(np.int64, copy=False)
    order = np.lexsort((local, stop[cols], start[cols]))
    return np.asarray(order, dtype=np.int64)


def _ordering_local_permutation(
    ordering: str,
    window_columns: np.ndarray,
    *,
    metadata: object | None,
    seed: int,
) -> np.ndarray:
    key = str(ordering).strip()
    if key in {"metadata_round", "round_span"}:
        if metadata is None:
            raise ValueError("metadata-based ordering requested but metadata is unavailable")
        return _metadata_window_permutation(metadata, window_columns)
    return ordering_permutation(window_columns.size, key, rng=np.random.default_rng(int(seed)))


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_large_structural_scan(
    side: DenseDemSide,
    config: LargeScanConfig,
    *,
    results_dir: str | Path,
    sector_label: str,
) -> dict[str, object]:
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(side.matrix, dtype=np.uint8)
    priors = np.asarray(side.priors, dtype=np.float64)
    rows = int(matrix.shape[0])
    cols = int(matrix.shape[1])
    metrics_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    gmax_values: list[int] = []
    start_time = time.time()
    processed = 0

    for window_size in config.window_sizes:
        if window_size > cols:
            continue
        window_starts = list(range(0, cols - window_size + 1, int(config.stride)))
        window_starts = window_starts[: int(config.max_windows_per_size)]
        for window_index, start_col in enumerate(window_starts):
            natural_columns = np.arange(start_col, start_col + window_size, dtype=np.int64)
            for ordering_index, ordering in enumerate(config.orderings):
                local_perm = _ordering_local_permutation(
                    ordering,
                    natural_columns,
                    metadata=side.metadata,
                    seed=int(config.seed + 10000 * window_size + 100 * window_index + ordering_index),
                )
                ordered_columns = natural_columns[local_perm]
                window_matrix = matrix[:, ordered_columns].copy()
                window_priors = priors[ordered_columns].copy()
                decode_l1 = decode_scl(
                    window_matrix,
                    window_priors,
                    np.zeros(rows, dtype=np.uint8),
                    list_size=1,
                )
                reliability = estimate_reliability_monte_carlo(
                    window_priors,
                    samples=int(config.reliability_samples),
                    seed=int(config.seed + 1000000 + 31 * processed),
                )
                gap = compute_gap_profile(
                    length=window_size,
                    frozen_indices=decode_l1.dynamic_frozen.frozen_indices,
                    reliability_scores=reliability.success_probability,
                )
                gmax_values.append(int(gap.g_max))
                info_indicator = [0 if index in set(decode_l1.dynamic_frozen.frozen_indices) else 1 for index in range(window_size)]
                stem = f"{sector_label.lower()}_w{window_size}_start{start_col:04d}_{ordering}"
                plot_frozen_free_profile(
                    info_indicator,
                    path=out_dir / f"{stem}_free_profile.png",
                    title=f"{sector_label} N={window_size} start={start_col} ordering={ordering}",
                )
                plot_reliability_profile(
                    reliability.success_probability,
                    path=out_dir / f"{stem}_reliability.png",
                    title=f"{sector_label} N={window_size} start={start_col} ordering={ordering}",
                    ylabel="genie-aided SC success probability",
                )
                plot_gap_profile(
                    gap.gap,
                    path=out_dir / f"{stem}_gap.png",
                    title=f"{sector_label} N={window_size} start={start_col} ordering={ordering}",
                )
                row = {
                    "sector": sector_label,
                    "source_matrix_rows": rows,
                    "source_matrix_cols": cols,
                    "window_size": int(window_size),
                    "window_start": int(start_col),
                    "ordering": ordering,
                    "frozen_count": int(len(decode_l1.dynamic_frozen.frozen_indices)),
                    "free_count": int(len(decode_l1.dynamic_frozen.info_indices)),
                    "g_max": int(gap.g_max),
                    "predicted_list_size": int(gap.predicted_list_size),
                    "mean_reliability": float(np.mean(reliability.success_probability)),
                    "min_reliability": float(np.min(reliability.success_probability)),
                    "max_reliability": float(np.max(reliability.success_probability)),
                    "window_columns": ",".join(str(int(value)) for value in ordered_columns.tolist()),
                    "matrix_label": side.matrix_label,
                }
                metrics_rows.append(row)
                with (out_dir / f"{stem}.json").open("w") as handle:
                    json.dump(
                        {
                            "row": row,
                            "reliability_success_probability": reliability.success_probability.tolist(),
                            "conditional_entropy_bits": reliability.conditional_entropy_bits.tolist(),
                            "free_prefix_count": gap.free_prefix_count.tolist(),
                            "ideal_prefix_count": gap.ideal_prefix_count.tolist(),
                            "gap_profile": gap.gap.tolist(),
                            "ideal_info_set": list(gap.ideal_info_set),
                            "frozen_indices": list(decode_l1.dynamic_frozen.frozen_indices),
                            "free_indices": list(decode_l1.dynamic_frozen.info_indices),
                        },
                        handle,
                        indent=2,
                        sort_keys=True,
                    )
                processed += 1
                if config.progress_every_windows and processed % int(config.progress_every_windows) == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"[polar-dem large] processed={processed} sector={sector_label} "
                        f"window_size={window_size} ordering={ordering} elapsed={elapsed:.1f}s"
                    )

    ordering_groups: dict[str, list[int]] = {}
    for row in metrics_rows:
        ordering_groups.setdefault(str(row["ordering"]), []).append(int(row["g_max"]))
    for ordering, values in sorted(ordering_groups.items()):
        summary_rows.append(
            {
                "sector": sector_label,
                "ordering": ordering,
                "windows": len(values),
                "median_g_max": float(np.median(np.asarray(values, dtype=np.float64))),
                "worst_g_max": int(np.max(np.asarray(values, dtype=np.int64))),
            }
        )

    _write_csv(
        out_dir / f"{sector_label.lower()}_window_metrics.csv",
        [
            "sector",
            "source_matrix_rows",
            "source_matrix_cols",
            "window_size",
            "window_start",
            "ordering",
            "frozen_count",
            "free_count",
            "g_max",
            "predicted_list_size",
            "mean_reliability",
            "min_reliability",
            "max_reliability",
            "window_columns",
            "matrix_label",
        ],
        metrics_rows,
    )
    _write_csv(
        out_dir / f"{sector_label.lower()}_ordering_summary.csv",
        ["sector", "ordering", "windows", "median_g_max", "worst_g_max"],
        summary_rows,
    )

    plot_gmax_histogram(
        gmax_values,
        path=out_dir / f"{sector_label.lower()}_gmax_histogram.png",
        title=f"{sector_label} structural scan: g_max histogram",
    )
    plot_ordering_summary(
        [str(row["ordering"]) for row in summary_rows],
        [float(row["median_g_max"]) for row in summary_rows],
        [float(row["worst_g_max"]) for row in summary_rows],
        path=out_dir / f"{sector_label.lower()}_ordering_comparison.png",
        title=f"{sector_label} structural scan ordering comparison",
    )

    natural = next((row for row in summary_rows if row["ordering"] == "natural"), None)
    other_rows = [row for row in summary_rows if row["ordering"] != "natural"]
    if natural is None or not other_rows:
        natural_answer = "insufficient data"
    else:
        best_other_median = min(float(row["median_g_max"]) for row in other_rows)
        best_other_worst = min(int(row["worst_g_max"]) for row in other_rows)
        if float(natural["median_g_max"]) < best_other_median and int(natural["worst_g_max"]) <= best_other_worst:
            natural_answer = "yes"
        elif float(natural["median_g_max"]) <= best_other_median:
            natural_answer = "mixed"
        else:
            natural_answer = "no"

    low_gap_windows = [row for row in metrics_rows if int(row["g_max"]) <= 1]
    best_windows = sorted(metrics_rows, key=lambda row: (int(row["g_max"]), int(row["window_start"])))[:5]
    worst_windows = sorted(metrics_rows, key=lambda row: (-int(row["g_max"]), int(row["window_start"])))[:5]
    summary = {
        "sector": sector_label,
        "matrix_label": side.matrix_label,
        "matrix_rows": rows,
        "matrix_cols": cols,
        "window_count": len(metrics_rows),
        "natural_order_answer": natural_answer,
        "windows_with_g_max_le_1": len(low_gap_windows),
        "best_windows": best_windows,
        "worst_windows": worst_windows,
    }
    with (out_dir / f"{sector_label.lower()}_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    report_lines = [
        f"# Polar DEM Large Structural Scan ({sector_label})",
        "",
        f"- Matrix used: `{side.matrix_label}`.",
        "- This scan is structural only: each window is a local detector-side submatrix profile, not a valid standalone window decoder claim.",
        f"- Reliability estimator: Monte Carlo genie-aided SC success probability with `{config.reliability_samples}` samples per window/order.",
        f"- Tested window sizes: `{', '.join(str(value) for value in config.window_sizes)}`.",
        f"- Tested orderings: `{', '.join(config.orderings)}`.",
        "",
        f"![g_max histogram]({(out_dir / f'{sector_label.lower()}_gmax_histogram.png').name})",
        "",
        (
            f"Figure 1. Structural polar-friendliness scan on `{side.matrix_label}` using consecutive natural-order windows of sizes "
            f"`{', '.join(str(value) for value in config.window_sizes)}` and the ordering set `{', '.join(config.orderings)}`. "
            "For each `(window, ordering)` pair, the detector-side submatrix is multiplied by the binary Arikan transform over GF(2), converted into a "
            "dynamic-frozen form by right-to-left elimination, and scored by the gap statistic `g(t)` / `g_max`. The histogram counts all scanned "
            "window/order combinations. Main quantitative takeaway: the exact `g_max` distribution is saved in the companion CSV and JSON summary."
        ),
        "",
        "| ordering | windows | median g_max | worst-case g_max |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        report_lines.append(
            f"| `{row['ordering']}` | `{row['windows']}` | `{float(row['median_g_max']):.3f}` | `{int(row['worst_g_max'])}` |"
        )
    report_lines.extend(
        [
            "",
            f"![ordering comparison]({(out_dir / f'{sector_label.lower()}_ordering_comparison.png').name})",
            "",
            (
                "Figure 2. Ordering comparison for the same structural scan, showing the median and worst-case `g_max` for each tested within-window column ordering. "
                "Main takeaway: this is the direct comparison used to judge whether the natural order is unusually polar-friendly on the scanned windows."
            ),
            "",
            "| type | window size | start | ordering | g_max | predicted L |",
            "| --- | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for row in best_windows:
        report_lines.append(
            f"| best | `{row['window_size']}` | `{row['window_start']}` | `{row['ordering']}` | `{row['g_max']}` | `{row['predicted_list_size']}` |"
        )
    for row in worst_windows:
        report_lines.append(
            f"| worst | `{row['window_size']}` | `{row['window_start']}` | `{row['ordering']}` | `{row['g_max']}` | `{row['predicted_list_size']}` |"
        )
    (out_dir / f"{sector_label.lower()}_report.md").write_text("\n".join(report_lines) + "\n")
    return summary
