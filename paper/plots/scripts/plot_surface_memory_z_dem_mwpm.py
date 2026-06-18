# SPDX-License-Identifier: Apache-2.0
"""Render surface memory-Z DEM Frontier versus MWPM comparison."""

from __future__ import annotations

import argparse

import plot_utils as u
from matplotlib import pyplot as plt


def _draw(args: argparse.Namespace) -> None:
    u.set_defaults()
    rows = u.load_csv_rows(args.data)
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    y_values: list[float] = []
    for index, (decoder, group) in enumerate(sorted(u.grouped(rows, "decoder").items())):
        y_values.extend(
            u.plot_series(
                ax,
                group,
                x_key="distance",
                y_key="fer_per_round",
                label=decoder,
                color=u.PALETTE[index % len(u.PALETTE)],
                marker=u.MARKERS[index % len(u.MARKERS)],
                low_key="fer_per_round_lo",
                high_key="fer_per_round_hi",
            )
        )
    u.apply_log_y(ax, y_values)
    ax.set_xlabel("surface-code distance")
    ax.set_ylabel("FER per syndrome-extraction round")
    ax.set_title("Rotated-surface memory-Z DEM")
    ax.legend(loc="best")
    u.save_figure(fig, args.output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    u.add_common_arguments(parser)
    _draw(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
