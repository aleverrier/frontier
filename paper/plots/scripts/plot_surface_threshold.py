# SPDX-License-Identifier: Apache-2.0
"""Render rotated-surface code-capacity threshold panels."""

from __future__ import annotations

import argparse

import plot_utils as u
from matplotlib import pyplot as plt


def _draw(args: argparse.Namespace) -> None:
    u.set_defaults()
    rows = u.load_csv_rows(args.data)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    y_values: list[float] = []

    panel = args.panel_id or "left"
    if panel == "left":
        for index, (distance, group) in enumerate(sorted(u.grouped(rows, "distance").items(), key=lambda x: int(x[0]))):
            y_values.extend(
                u.plot_series(
                    ax,
                    group,
                    x_key="p",
                    y_key="fer",
                    label=f"d={distance}",
                    color=u.PALETTE[index % len(u.PALETTE)],
                    marker=u.MARKERS[index % len(u.MARKERS)],
                    low_key="fer_lo95",
                    high_key="fer_hi95",
                )
            )
        u.apply_log_y(ax, y_values)
        ax.set_ylabel("logical frame error rate")
        ax.set_title("Rotated-surface code-capacity threshold")
        u.add_reference_line(ax, 0.189, "pc=0.189")
    else:
        for index, (distance, group) in enumerate(sorted(u.grouped(rows, "distance").items(), key=lambda x: int(x[0]))):
            y_values.extend(
                u.plot_series(
                    ax,
                    group,
                    x_key="p",
                    y_key="mean_states_total",
                    label=f"d={distance}",
                    color=u.PALETTE[index % len(u.PALETTE)],
                    marker=u.MARKERS[index % len(u.MARKERS)],
                )
            )
        u.apply_log_y(ax, y_values)
        ax.set_ylabel("average retained list size")
        ax.set_title("Retained list size")
    ax.set_xlabel("physical error probability p")
    ax.legend(loc="best", ncols=2)
    u.save_figure(fig, args.output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    u.add_common_arguments(parser)
    _draw(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
