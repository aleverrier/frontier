# SPDX-License-Identifier: Apache-2.0
"""Render the data-driven frontier active-boundary schematic."""

from __future__ import annotations

import argparse

import plot_utils as u
from matplotlib import pyplot as plt


def _draw(args: argparse.Namespace) -> None:
    u.set_defaults()
    rows = u.load_csv_rows(args.data)
    by_id = {row["element_id"]: row for row in rows if row["element_id"]}

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.set_aspect("equal")
    ax.axis("off")

    for row in rows:
        if row["element_type"] != "incidence_edge":
            continue
        source = by_id.get(row["source"])
        target = by_id.get(row["target"])
        if not source or not target:
            continue
        color = "#d62728" if u.as_bool(row, "is_active") else "#9aa4ad"
        alpha = 0.9 if u.as_bool(row, "is_processed") or u.as_bool(row, "is_active") else 0.45
        ax.plot(
            [u.as_float(source, "x"), u.as_float(target, "x")],
            [u.as_float(source, "y"), u.as_float(target, "y")],
            color=color,
            linewidth=1.2,
            alpha=alpha,
            zorder=1,
        )

    variable_rows = [row for row in rows if row["element_type"] == "variable_node"]
    processed_xs = [u.as_float(row, "x") for row in variable_rows if u.as_bool(row, "is_processed")]
    unprocessed_xs = [u.as_float(row, "x") for row in variable_rows if not u.as_bool(row, "is_processed")]
    if processed_xs and unprocessed_xs:
        boundary_x = (max(processed_xs) + min(unprocessed_xs)) / 2.0
        ax.axvline(boundary_x, color="#333333", linestyle="--", linewidth=1.0)
        ax.text(boundary_x + 0.04, 4.45, "frontier", ha="left", va="top", fontsize=8)

    for row in variable_rows:
        x = u.as_float(row, "x")
        y = u.as_float(row, "y")
        processed = u.as_bool(row, "is_processed")
        ax.scatter(
            [x],
            [y],
            s=180,
            marker="s",
            color="#88bde6" if processed else "#ffffff",
            edgecolor="#1f77b4",
            linewidth=1.2,
            zorder=3,
        )
        ax.text(x, y, row["element_id"], ha="center", va="center", fontsize=8, zorder=4)

    for row in rows:
        if row["element_type"] not in {"check_node", "logical_node"}:
            continue
        x = u.as_float(row, "x")
        y = u.as_float(row, "y")
        if row["element_type"] == "logical_node":
            color = "#9467bd"
            marker = "D"
            edge = "#5e3c99"
        elif u.as_bool(row, "is_active"):
            color = "#f4a582"
            marker = "o"
            edge = "#b2182b"
        elif u.as_bool(row, "is_closed"):
            color = "#d9d9d9"
            marker = "o"
            edge = "#6b6b6b"
        else:
            color = "#ffffff"
            marker = "o"
            edge = "#333333"
        ax.scatter([x], [y], s=160, marker=marker, color=color, edgecolor=edge, linewidth=1.2, zorder=3)
        ax.text(x, y, row["element_id"], ha="center", va="center", fontsize=8, zorder=4)

    ax.text(0.65, 4.55, "ordered variables", ha="left", va="top", fontsize=9)
    ax.text(2.75, 0.05, "active rows are checks crossing the frontier", ha="left", va="bottom", fontsize=8)
    ax.set_xlim(0.25, 4.25)
    ax.set_ylim(0.0, 4.7)
    u.save_figure(fig, args.output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    u.add_common_arguments(parser)
    _draw(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
