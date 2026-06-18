# SPDX-License-Identifier: Apache-2.0
"""Render BB72 detector-side DEM comparison panels."""

from __future__ import annotations

import argparse

import plot_utils as u
from matplotlib import pyplot as plt


def _has_number(row: dict[str, str], key: str) -> bool:
    return bool(row.get(key, "").strip())


def _label(row: dict[str, str]) -> str:
    return row.get("source_decoder") or row.get("decoder") or row.get("decoder_label") or "reference"


def _draw_references(ax, rows: list[dict[str, str]], x_values: list[float], color_offset: int) -> None:
    if not rows or not x_values:
        return
    xmin, xmax = min(x_values), max(x_values)
    for index, row in enumerate(rows):
        color = u.PALETTE[(index + color_offset) % len(u.PALETTE)]
        y = u.as_float(row, "fer")
        label = _label(row)
        ax.hlines(y, xmin, xmax, color=color, linestyle="--", linewidth=1.2, label=label)
        if row.get("ci_low") and row.get("ci_high"):
            ax.fill_between([xmin, xmax], [u.as_float(row, "ci_low")] * 2, [u.as_float(row, "ci_high")] * 2, color=color, alpha=0.12, linewidth=0)


def _draw(args: argparse.Namespace) -> None:
    u.set_defaults()
    rows = u.load_csv_rows(args.data)
    panel = args.panel_id or "left"
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    y_values: list[float] = []

    if panel == "left":
        key = "decoder_label" if "decoder_label" in rows[0] else "decoder"
        for index, (decoder, group) in enumerate(sorted(u.grouped(rows, key).items())):
            y_values.extend(
                u.plot_series(
                    ax,
                    group,
                    x_key="p",
                    y_key="fer",
                    label=decoder,
                    color=u.PALETTE[index % len(u.PALETTE)],
                    marker=u.MARKERS[index % len(u.MARKERS)],
                    low_key="ci_low95",
                    high_key="ci_high95",
                )
            )
        ax.set_xlabel("physical error probability p")
        ax.set_title("BB72 DEM FER versus p")
        ax.set_xscale("log")
    else:
        x_key = "plot_x"
        curve_rows = [row for row in rows if _has_number(row, x_key)]
        reference_rows = [row for row in rows if not _has_number(row, x_key)]
        for index, (decoder, group) in enumerate(sorted(u.grouped(curve_rows, "decoder").items())):
            y_values.extend(
                u.plot_series(
                    ax,
                    group,
                    x_key=x_key,
                    y_key="plot_y",
                    label=decoder or "Frontier",
                    color=u.PALETTE[index % len(u.PALETTE)],
                    marker=u.MARKERS[index % len(u.MARKERS)],
                    low_key="ci_low",
                    high_key="ci_high",
                )
            )
        _draw_references(
            ax,
            reference_rows,
            [u.as_float(row, x_key) for row in curve_rows],
            color_offset=len(u.grouped(curve_rows, "decoder")),
        )
        ax.set_xlabel("average retained list size")
        ax.set_title("BB72 DEM retained-list sweep")
        ax.set_xscale("log")

    u.apply_log_y(ax, y_values)
    ax.set_ylabel("logical frame error rate")
    ax.legend(loc="best")
    u.save_figure(fig, args.output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    u.add_common_arguments(parser)
    _draw(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
