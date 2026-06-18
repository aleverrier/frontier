# SPDX-License-Identifier: Apache-2.0
"""Render the Gross/BB144 failure-decomposition figure."""

from __future__ import annotations

import argparse

import plot_utils as u
from matplotlib import pyplot as plt


COMPONENTS = (
    ("empty_terminal_frontier_rate", "empty terminal frontier", "#d62728"),
    ("logical_support_loss_rate", "logical support loss", "#ff7f0e"),
    ("terminal_ranking_failure_rate", "terminal ranking failure", "#9467bd"),
    ("syndrome_fail", "syndrome failure", "#7f7f7f"),
)


def _component_values(rows: list[dict[str, str]], key: str) -> list[float]:
    if key == "syndrome_fail":
        return [u.as_float(row, key, 0.0) / max(1.0, u.as_float(row, "source_trials")) for row in rows]
    return [u.as_float(row, key, 0.0) for row in rows]


def _draw(args: argparse.Namespace) -> None:
    u.set_defaults()
    rows = sorted(u.load_csv_rows(args.data), key=lambda row: u.as_float(row, "avg_retained_list_size"))
    xs = [u.as_float(row, "avg_retained_list_size") for row in rows]
    total = [u.as_float(row, "fer") for row in rows]

    fig, ax = plt.subplots(figsize=(5.8, 4.1))
    cumulative = [0.0 for _ in rows]
    for key, label, color in COMPONENTS:
        values = _component_values(rows, key)
        upper = [a + b for a, b in zip(cumulative, values, strict=True)]
        ax.fill_between(xs, cumulative, upper, color=color, alpha=0.32, linewidth=0, label=label)
        cumulative = upper

    ax.plot(xs, total, color="#111111", marker="o", linewidth=1.5, markersize=3.5, label="total FER")
    ax.set_xscale("log")
    u.apply_log_y(ax, [value for value in total if value > 0.0])
    ax.set_xlabel("average retained list size")
    ax.set_ylabel("logical frame error rate")
    ax.set_title("Gross/BB144 p=0.002 failure decomposition")
    ax.legend(loc="best", fontsize=7)
    u.save_figure(fig, args.output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    u.add_common_arguments(parser)
    _draw(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
