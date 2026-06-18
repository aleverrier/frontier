# SPDX-License-Identifier: Apache-2.0
"""Small helpers shared by paper-plot renderers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

_cache_root = Path(os.environ.get("MPLCONFIGDIR", Path(tempfile.gettempdir()) / "frontier-matplotlib"))
_cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root.parent))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = REPO_ROOT / "paper" / "plots" / "manifest.csv"

PALETTE = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#7f7f7f",
)
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", required=True, help="Input CSV table.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument("--figure-id", required=True, help="Manifest figure_id.")
    parser.add_argument("--panel-id", default="", help="Manifest panel_id.")
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Paper plot manifest path.",
    )


def load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_sidecar_for_csv(path: str | Path) -> dict[str, object]:
    sidecar_path = Path(path).with_suffix(".json")
    return json.loads(sidecar_path.read_text(encoding="utf-8"))


def load_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def companion_data_path(
    *,
    manifest_path: str | Path,
    figure_id: str,
    panel_id: str,
) -> Path | None:
    for row in load_manifest(manifest_path):
        if row.get("figure_id") == figure_id and row.get("panel_id") == panel_id:
            data_file = row.get("data_file", "")
            return repo_path(data_file) if data_file else None
    return None


def prepare_output(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def set_defaults() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 200,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
        }
    )


def as_float(row: dict[str, str], key: str, default: float | None = None) -> float:
    value = row.get(key, "")
    if value == "" or value is None:
        if default is None:
            raise ValueError(f"missing numeric column {key!r}")
        return default
    return float(value)


def as_int(row: dict[str, str], key: str, default: int | None = None) -> int:
    value = row.get(key, "")
    if value == "" or value is None:
        if default is None:
            raise ValueError(f"missing integer column {key!r}")
        return default
    return int(float(value))


def as_bool(row: dict[str, str], key: str) -> bool:
    return row.get(key, "").strip().lower() in {"1", "true", "yes", "y"}


def grouped(rows: Iterable[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        out[row.get(key, "")].append(row)
    return dict(out)


def finite_positive(values: Iterable[float]) -> list[float]:
    return [x for x in values if math.isfinite(x) and x > 0.0]


def apply_log_y(ax, values: Iterable[float]) -> None:
    positives = finite_positive(values)
    if positives:
        ax.set_yscale("log")
        floor = min(positives) / 2.0
        ax.set_ylim(bottom=max(floor, 1e-12))


def apply_log_x(ax, values: Iterable[float]) -> None:
    all_values = list(values)
    positives = finite_positive(all_values)
    if positives and len(positives) == len(all_values):
        ax.set_xscale("log")


def fill_ci(ax, xs: list[float], lows: list[float], highs: list[float], color: str) -> None:
    if not xs or not lows or not highs:
        return
    ax.fill_between(xs, lows, highs, color=color, alpha=0.15, linewidth=0)


def plot_series(
    ax,
    rows: list[dict[str, str]],
    *,
    x_key: str,
    y_key: str,
    label: str,
    color: str,
    marker: str,
    low_key: str | None = None,
    high_key: str | None = None,
) -> list[float]:
    ordered = sorted(rows, key=lambda r: as_float(r, x_key))
    xs = [as_float(row, x_key) for row in ordered]
    ys = [as_float(row, y_key) for row in ordered]
    ax.plot(xs, ys, marker=marker, linewidth=1.6, markersize=4.0, label=label, color=color)
    if low_key and high_key and all(row.get(low_key, "") and row.get(high_key, "") for row in ordered):
        lows = [as_float(row, low_key) for row in ordered]
        highs = [as_float(row, high_key) for row in ordered]
        fill_ci(ax, xs, lows, highs, color)
    return ys


def save_figure(fig, output: str | Path) -> None:
    path = prepare_output(output)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def add_reference_line(ax, x: float, label: str) -> None:
    ax.axvline(x, color="#333333", linestyle="--", linewidth=1.0, alpha=0.75)
    ymax = ax.get_ylim()[1]
    ax.text(x, ymax, label, rotation=90, va="top", ha="right", color="#333333", fontsize=8)


def label_for_decoder(value: str) -> str:
    return value.replace("IonQ strong beam search", "IonQ strong beam search").replace(
        "FrontierFast", "Frontier"
    )
