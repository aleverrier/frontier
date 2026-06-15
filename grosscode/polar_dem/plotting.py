from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _finalize(fig: plt.Figure, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_frozen_free_profile(info_indicator: Sequence[int], *, path: str | Path, title: str) -> None:
    values = np.asarray(info_indicator, dtype=np.int64).reshape(-1)
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.step(np.arange(values.size), values, where="mid", linewidth=1.6)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("synthetic-bit index i")
    ax.set_ylabel("free=1")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    _finalize(fig, path)


def plot_reliability_profile(scores: Sequence[float], *, path: str | Path, title: str, ylabel: str) -> None:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(np.arange(values.size), values, marker="o", linewidth=1.4, markersize=2.8)
    ax.set_xlabel("synthetic-bit index i")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    _finalize(fig, path)


def plot_gap_profile(gap: Sequence[int], *, path: str | Path, title: str) -> None:
    values = np.asarray(gap, dtype=np.int64).reshape(-1)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.step(np.arange(values.size), values, where="mid", linewidth=1.8)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("synthetic-bit index i")
    ax.set_ylabel("g(i)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    _finalize(fig, path)


def plot_gmax_histogram(gmax_values: Sequence[int], *, path: str | Path, title: str) -> None:
    values = np.asarray(gmax_values, dtype=np.int64).reshape(-1)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bins = np.arange(values.min(initial=0), values.max(initial=0) + 2) - 0.5
    ax.hist(values, bins=bins, edgecolor="black", alpha=0.8)
    ax.set_xlabel("g_max")
    ax.set_ylabel("window/order count")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    _finalize(fig, path)


def plot_list_size_scatter(
    predicted: Sequence[int],
    observed: Sequence[int],
    *,
    path: str | Path,
    title: str,
    ylabel: str,
) -> None:
    x = np.asarray(predicted, dtype=np.float64).reshape(-1)
    y = np.asarray(observed, dtype=np.float64).reshape(-1)
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.scatter(x, y, alpha=0.8)
    max_axis = float(max(np.max(x, initial=1.0), np.max(y, initial=1.0)))
    ax.plot([1.0, max_axis], [1.0, max_axis], linestyle="--", color="black", linewidth=1.0)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("predicted list size 2^{g_max}")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    _finalize(fig, path)


def plot_ordering_summary(
    orderings: Sequence[str],
    medians: Sequence[float],
    worst_cases: Sequence[float],
    *,
    path: str | Path,
    title: str,
) -> None:
    labels = list(orderings)
    x = np.arange(len(labels), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    ax.bar(x - 0.18, medians, width=0.36, label="median g_max")
    ax.bar(x + 0.18, worst_cases, width=0.36, label="worst-case g_max")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("g_max")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    _finalize(fig, path)
