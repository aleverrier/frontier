# SPDX-License-Identifier: Apache-2.0
"""Render the Gross/BB144 transition-evaluation tail curve."""

from __future__ import annotations

import argparse

import plot_utils as u
from matplotlib import pyplot as plt


def _draw(args: argparse.Namespace) -> None:
    u.set_defaults()
    rows = u.load_csv_rows(args.data)
    percentile_path = u.companion_data_path(
        manifest_path=args.manifest,
        figure_id=args.figure_id,
        panel_id="percentile_guides",
    )
    percentiles = u.load_csv_rows(percentile_path) if percentile_path and percentile_path.exists() else []

    ordered = sorted(rows, key=lambda row: u.as_float(row, "bin_left"))
    xs = [u.as_float(row, "bin_left") for row in ordered]
    ys = [u.as_float(row, "tail_fraction_ge_bin_left") for row in ordered]
    lows = [u.as_float(row, "tail_ci_low95") for row in ordered]
    highs = [u.as_float(row, "tail_ci_high95") for row in ordered]

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(xs, ys, color=u.PALETTE[0], linewidth=1.7, label="tail probability")
    u.fill_ci(ax, xs, lows, highs, u.PALETTE[0])
    for row in percentiles:
        metric = row["metric"]
        if not metric.startswith("p") and metric not in {"mean"}:
            continue
        x = u.as_float(row, "transition_evals")
        ax.axvline(x, color="#333333", linestyle="--", linewidth=0.9, alpha=0.65)
        ax.text(x, max(ys), metric, rotation=90, ha="right", va="top", fontsize=7)

    ax.set_xscale("log")
    u.apply_log_y(ax, ys)
    ax.set_xlabel("transition evaluations per full frame")
    ax.set_ylabel("Pr(work >= x)")
    ax.set_title("Gross/BB144 p=0.001 transition-evaluation tail")
    ax.legend(loc="best")
    u.save_figure(fig, args.output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    u.add_common_arguments(parser)
    _draw(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
