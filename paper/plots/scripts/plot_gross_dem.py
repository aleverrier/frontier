# SPDX-License-Identifier: Apache-2.0
"""Render Gross/BB144 detector-side DEM panels."""

from __future__ import annotations

import argparse

import plot_utils as u
from matplotlib import pyplot as plt


def _has_number(row: dict[str, str], key: str) -> bool:
    return bool(row.get(key, "").strip())


def _label(row: dict[str, str]) -> str:
    return row.get("source_decoder") or row.get("decoder") or row.get("series") or "reference"


def _draw_references(ax, rows: list[dict[str, str]], x_values: list[float], color_offset: int) -> None:
    if not rows or not x_values:
        return
    xmin, xmax = min(x_values), max(x_values)
    for index, row in enumerate(rows):
        color = u.PALETTE[(index + color_offset) % len(u.PALETTE)]
        y = u.as_float(row, "fer")
        ax.hlines(y, xmin, xmax, color=color, linestyle="--", linewidth=1.2, label=_label(row))
        if row.get("ci_low") and row.get("ci_high"):
            ax.fill_between(
                [xmin, xmax],
                [u.as_float(row, "ci_low")] * 2,
                [u.as_float(row, "ci_high")] * 2,
                color=color,
                alpha=0.12,
                linewidth=0,
            )


def _plot_fer_vs_p(ax, rows: list[dict[str, str]]) -> list[float]:
    values: list[float] = []
    key = "decoder_label" if "decoder_label" in rows[0] else "decoder"
    for index, (decoder, group) in enumerate(sorted(u.grouped(rows, key).items())):
        values.extend(
            u.plot_series(
                ax,
                group,
                x_key="p_location",
                y_key="fer",
                label=decoder,
                color=u.PALETTE[index % len(u.PALETTE)],
                marker=u.MARKERS[index % len(u.MARKERS)],
                low_key="fer_low95",
                high_key="fer_high95",
            )
        )
    ax.set_xlabel("physical error probability p")
    ax.set_title("Gross/BB144 DEM FER versus p")
    ax.set_xscale("log")
    return values


def _plot_fer_vs_retained(ax, rows: list[dict[str, str]]) -> list[float]:
    values: list[float] = []
    key = "series" if "series" in rows[0] else "decoder"
    x_key = "plot_x" if "plot_x" in rows[0] else "x_mean_list_size_per_side"
    curve_rows = [row for row in rows if _has_number(row, x_key)]
    reference_rows = [row for row in rows if not _has_number(row, x_key)]
    for index, (series, group) in enumerate(sorted(u.grouped(curve_rows, key).items())):
        values.extend(
            u.plot_series(
                ax,
                group,
                x_key=x_key,
                y_key="fer",
                label=series,
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
        color_offset=len(u.grouped(curve_rows, key)),
    )
    ax.set_xlabel("average retained list size per side")
    ax.set_title("Gross/BB144 p=0.001 retained-list sweep")
    ax.set_xscale("log")
    return values


def _plot_avg_vs_peak(ax, rows: list[dict[str, str]]) -> list[float]:
    ordered = sorted(rows, key=lambda row: u.as_float(row, "x_avg_retained_list_size_per_side_column"))
    xs = [u.as_float(row, "x_avg_retained_list_size_per_side_column") for row in ordered]
    ys = [u.as_float(row, "y_mean_peak_retained_list_size_per_side_trace") for row in ordered]
    labels = [row.get("point_label") or f"D={u.as_float(row, 'Delta'):g}" for row in ordered]
    ax.plot(xs, ys, marker="o", linewidth=1.6, color=u.PALETTE[0])
    for x, y, label in zip(xs, ys, labels, strict=True):
        ax.text(x, y, label, fontsize=7, ha="left", va="bottom")
    ax.set_xlabel("average retained list size per side")
    ax.set_ylabel("mean peak retained list size per side")
    ax.set_title("Gross/BB144 p=0.001 average versus peak retained states")
    ax.set_xscale("log")
    ax.set_yscale("log")
    return ys


def _draw(args: argparse.Namespace) -> None:
    u.set_defaults()
    rows = u.load_csv_rows(args.data)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    if "p_location" in rows[0] and args.panel_id == "left":
        y_values = _plot_fer_vs_p(ax, rows)
        ax.set_ylabel("logical frame error rate")
        u.apply_log_y(ax, y_values)
        ax.legend(loc="best")
    elif "y_mean_peak_retained_list_size_per_side_trace" in rows[0]:
        _plot_avg_vs_peak(ax, rows)
    else:
        y_values = _plot_fer_vs_retained(ax, rows)
        ax.set_ylabel("logical frame error rate")
        u.apply_log_y(ax, y_values)
        ax.legend(loc="best")

    u.save_figure(fig, args.output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    u.add_common_arguments(parser)
    _draw(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
