# SPDX-License-Identifier: Apache-2.0
"""Render hexagonal color-code threshold panels."""

from __future__ import annotations

import argparse

import plot_utils as u
from matplotlib import pyplot as plt


def _draw(args: argparse.Namespace) -> None:
    u.set_defaults()
    rows = u.load_csv_rows(args.data)
    panel = args.panel_id or "left"
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    y_values: list[float] = []

    for index, (distance, group) in enumerate(sorted(u.grouped(rows, "distance").items(), key=lambda x: int(x[0]))):
        y_key = "plot_y" if "plot_y" in group[0] else "rung0_fer"
        kwargs = {
            "x_key": "plot_x" if "plot_x" in group[0] else "p",
            "y_key": y_key,
            "label": f"d={distance}",
            "color": u.PALETTE[index % len(u.PALETTE)],
            "marker": u.MARKERS[index % len(u.MARKERS)],
        }
        if panel == "left" and group[0].get("plot_ci_low") and group[0].get("plot_ci_high"):
            kwargs["low_key"] = "plot_ci_low"
            kwargs["high_key"] = "plot_ci_high"
        y_values.extend(u.plot_series(ax, group, **kwargs))

    u.apply_log_y(ax, y_values)
    ax.set_xlabel("physical error probability p")
    if panel == "left":
        ax.set_ylabel("logical frame error rate")
        ax.set_title("Hexagonal color-code code capacity")
        u.add_reference_line(ax, 0.109, "pc ~ 0.109")
    else:
        ax.set_ylabel("average retained states")
        ax.set_title("Retained states")
    ax.legend(loc="best", ncols=2)
    u.save_figure(fig, args.output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    u.add_common_arguments(parser)
    _draw(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
