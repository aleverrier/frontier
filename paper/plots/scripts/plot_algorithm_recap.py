# SPDX-License-Identifier: Apache-2.0
"""Render the four algorithm-recap state-plane panels."""

from __future__ import annotations

import argparse

import plot_utils as u
from matplotlib import pyplot as plt


STAGE_BY_PANEL = {
    "step1": "step1_retained",
    "step2": "step2_branch",
    "step3": "step3_merged",
    "step4": "step4_pruned",
}

TITLE_BY_PANEL = {
    "step1": "Retained frontier states",
    "step2": "Branch next variable",
    "step3": "Merge equal states",
    "step4": "Prune by score",
}


def _plot_cutoff(ax, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    alpha = u.as_float(rows[0], "alpha")
    cutoff = u.as_float(rows[0], "cutoff_score_after_next_column")
    xs = [u.as_float(row, "prefix_mass_P") for row in rows]
    xmin, xmax = min(xs), max(xs)
    xline = [xmin, xmax]
    yline = [(cutoff - x) / alpha for x in xline]
    ax.plot(xline, yline, color="#333333", linestyle="--", linewidth=1.0, label="Delta cutoff")


def _draw(args: argparse.Namespace) -> None:
    u.set_defaults()
    rows = u.load_csv_rows(args.data)
    panel_id = args.panel_id or "step1"
    stage = STAGE_BY_PANEL.get(panel_id)
    if stage is None:
        raise ValueError(f"unknown algorithm panel_id={panel_id!r}")

    stage_rows = [row for row in rows if row["stage"] == stage]
    parent_rows = {row["state_key"]: row for row in rows if row["stage"] == "step1_retained"}

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    if panel_id == "step2":
        for row in stage_rows:
            parent = parent_rows.get(row["parent_state_key"])
            if parent:
                ax.plot(
                    [u.as_float(parent, "prefix_mass_P"), u.as_float(row, "prefix_mass_P")],
                    [u.as_float(parent, "future_score_F"), u.as_float(row, "future_score_F")],
                    color="#9aa4ad",
                    linewidth=0.7,
                    alpha=0.45,
                    zorder=1,
                )

    kept = [row for row in stage_rows if u.as_bool(row, "kept")]
    dropped = [row for row in stage_rows if not u.as_bool(row, "kept")]
    for group, label, color, alpha in (
        (dropped, "not retained", "#bdbdbd", 0.45),
        (kept, "retained", "#1f77b4", 0.85),
    ):
        if not group:
            continue
        sizes = [26 + 16 * max(0, u.as_int(row, "merged_from_count", 1) - 1) for row in group]
        ax.scatter(
            [u.as_float(row, "prefix_mass_P") for row in group],
            [u.as_float(row, "future_score_F") for row in group],
            s=sizes,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            alpha=alpha,
            label=label,
            zorder=3,
        )

    if panel_id in {"step2", "step3", "step4"}:
        _plot_cutoff(ax, stage_rows)

    ax.set_title(TITLE_BY_PANEL[panel_id])
    ax.set_xlabel("prefix log mass P")
    ax.set_ylabel("future score F")
    ax.legend(loc="best")
    ax.text(
        0.02,
        0.02,
        f"K={u.as_int(stage_rows[0], 'K')}, Delta={u.as_float(stage_rows[0], 'Delta'):g}, alpha={u.as_float(stage_rows[0], 'alpha'):g}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
    )
    u.save_figure(fig, args.output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    u.add_common_arguments(parser)
    _draw(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
