from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable, Sequence

import numpy as np

from grosscode.core import FrameDecodeResult, SideContext


@dataclass(frozen=True)
class SampledFrame:
    x_syndrome: np.ndarray
    z_syndrome: np.ndarray
    x_logical_true: np.ndarray
    z_logical_true: np.ndarray


def sample_frames(
    *,
    x_context: SideContext,
    z_context: SideContext,
    shots: int,
    seed: int,
) -> list[SampledFrame]:
    rng = np.random.default_rng(int(seed))
    out: list[SampledFrame] = []
    for _ in range(int(shots)):
        _, x_syndrome, x_logical = x_context.sample(rng)
        _, z_syndrome, z_logical = z_context.sample(rng)
        out.append(
            SampledFrame(
                x_syndrome=np.asarray(x_syndrome, dtype=np.uint8),
                z_syndrome=np.asarray(z_syndrome, dtype=np.uint8),
                x_logical_true=np.asarray(x_logical, dtype=np.uint8),
                z_logical_true=np.asarray(z_logical, dtype=np.uint8),
            )
        )
    return out


def classify_frame(result: FrameDecodeResult, sample: SampledFrame) -> str:
    if int(result.unsatisfied_checks["x"]) > 0 or int(result.unsatisfied_checks["z"]) > 0:
        return "syndrome_fail"
    if not np.array_equal(np.asarray(result.logical_frame_action["x"], dtype=np.uint8), sample.x_logical_true):
        return "logical_fail"
    if not np.array_equal(np.asarray(result.logical_frame_action["z"], dtype=np.uint8), sample.z_logical_true):
        return "logical_fail"
    return "success"


def suggest_overlap(
    *contexts: SideContext,
    quantile: float = 0.995,
    minimum: int = 32,
    pad: int = 8,
) -> int:
    reaches: list[np.ndarray] = []
    for context in contexts:
        delta = np.asarray(context.col_forward_reach - np.arange(context.n, dtype=np.int32), dtype=np.int32)
        delta = delta[delta >= 0]
        if delta.size:
            reaches.append(delta)
    if not reaches:
        return int(minimum)
    merged = np.concatenate(reaches)
    return int(max(int(minimum), int(np.ceil(np.quantile(merged, float(quantile)))) + int(pad)))


def estimate_runtime_range_seconds(
    *,
    shots: int,
    decoder_count: int,
    max_iter: int,
    contexts: Sequence[SideContext],
    window_multiplier: float = 1.0,
) -> tuple[float, float]:
    total_edges = sum(int(ctx.graph.n_edges) for ctx in contexts)
    work_units = float(shots) * float(decoder_count) * float(max_iter) * float(total_edges) * float(window_multiplier)
    fast = work_units / 2.0e7
    slow = work_units / 3.0e6
    return float(fast), float(slow)


def summarize_decoder_rows(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {
            "shots": 0,
            "success": 0,
            "logical_fail": 0,
            "syndrome_fail": 0,
            "exception_fail": 0,
            "fail_total": 0,
            "logical_error_rate": float("nan"),
            "runtime_ms_mean": float("nan"),
            "iterations_mean": float("nan"),
        }
    status_counts = {
        "success": 0,
        "logical_fail": 0,
        "syndrome_fail": 0,
        "exception_fail": 0,
    }
    runtimes: list[float] = []
    iterations: list[float] = []
    for row in rows:
        status = str(row.get("status", "exception_fail"))
        status_counts[status] = int(status_counts.get(status, 0)) + 1
        runtimes.append(float(row.get("runtime_ms", float("nan"))))
        iterations.append(float(row.get("iterations_total", float("nan"))))
    shots = int(len(rows))
    fail_total = int(shots - status_counts["success"])
    return {
        "shots": shots,
        "success": int(status_counts["success"]),
        "logical_fail": int(status_counts["logical_fail"]),
        "syndrome_fail": int(status_counts["syndrome_fail"]),
        "exception_fail": int(status_counts["exception_fail"]),
        "fail_total": int(fail_total),
        "logical_error_rate": float(fail_total) / float(shots),
        "runtime_ms_mean": float(np.nanmean(np.asarray(runtimes, dtype=float))),
        "iterations_mean": float(np.nanmean(np.asarray(iterations, dtype=float))),
    }


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes = seconds / 60.0
    if minutes < 60.0:
        return f"{minutes:.1f}m"
    return f"{minutes / 60.0:.2f}h"


def timed_decode(decoder: object, sample: SampledFrame) -> tuple[FrameDecodeResult, float]:
    start = time.perf_counter()
    result = decoder.decode(
        x_syndrome=sample.x_syndrome,
        z_syndrome=sample.z_syndrome,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, float(elapsed_ms)
