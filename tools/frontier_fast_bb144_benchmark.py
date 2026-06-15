#!/usr/bin/env python3
"""Focused frontier timing probe for the BB144/Gross split-sector DEM."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MPLCONFIGDIR = _REPO_ROOT / "results" / "mplconfig"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

from tools import frontier_fast_decoder as fast
from tools.gross144_dem_x_progressive_report import _load_dem_family


DEFAULT_SAMPLE_ROWS = Path(
    "results/20260523_circuit_level_fer_psweep_adaptive_v1/"
    "bb144/p0p001/samples/sample_rows_s0_4999.csv"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Time native frontier selected bidirectional-committee decode on "
            "the accepted BB144/Gross split-sector DEM benchmark."
        )
    )
    parser.add_argument("--sample-rows", type=Path, default=DEFAULT_SAMPLE_ROWS)
    parser.add_argument("--backend", default="bravyi_depth7")
    parser.add_argument("--p-location", type=float, default=0.001)
    parser.add_argument("--column-order", default="deadline_reorder")
    parser.add_argument("--K", type=int, default=8192)
    parser.add_argument("--Delta", type=float, default=14.0)
    parser.add_argument("--score-alpha", type=float, default=0.8)
    parser.add_argument(
        "--metric-mode",
        choices=("logsumexp_float", "frontier_lite", "maxlog_int"),
        default="logsumexp_float",
    )
    parser.add_argument("--int-metric-scale", type=int, default=1024)
    parser.add_argument("--rows-per-scope", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument(
        "--payload",
        choices=("replay", "compact", "full"),
        default="replay",
        help="Native selected-committee payload shape to time.",
    )
    parser.add_argument(
        "--scope",
        action="append",
        choices=("memory_X", "memory_Z"),
        help="Scope(s) to benchmark. Defaults to both memory_X and memory_Z.",
    )
    parser.add_argument(
        "--full-payload",
        action="store_true",
        help="Compatibility alias for --payload full.",
    )
    return parser.parse_args()


def _load_syndromes(sample_rows: Path, scopes: tuple[str, ...], rows_per_scope: int) -> dict[str, tuple[int, ...]]:
    rows: dict[str, list[int]] = {scope: [] for scope in scopes}
    with Path(sample_rows).open(newline="") as f:
        for row in csv.DictReader(f):
            scope = str(row.get("scope", ""))
            if scope in rows and len(rows[scope]) < rows_per_scope:
                rows[scope].append(int(row["truth_syndrome"]))
            if all(len(values) >= rows_per_scope for values in rows.values()):
                break
    missing = {scope: rows_per_scope - len(values) for scope, values in rows.items() if len(values) < rows_per_scope}
    if missing:
        raise ValueError(f"sample row file has too few rows for requested benchmark: {missing}")
    return {scope: tuple(values) for scope, values in rows.items()}


def _build_models(*, backend: str, p_location: float, column_order: str, scope: str, syndrome: int):
    family = _load_dem_family(
        backend=str(backend),
        p_location=float(p_location),
        scope=str(scope),
        column_order=str(column_order),
    )
    model = fast.FrontierFastModel(
        columns=family.columns,
        layout=family.layout,
        num_detectors=int(family.matrix_rows),
        num_observables=int(family.logical_rows),
    )
    forward = fast._coerce_model(model, syndrome_int=int(syndrome), direction="forward")
    backward = fast._coerce_model(model, syndrome_int=int(syndrome), direction="backward")
    return family, forward, backward


def _mean_payload_value(payloads: tuple[dict[str, object], ...], key: str) -> float:
    return statistics.mean(float(payload.get(key, 0.0)) for payload in payloads)


def _mean_stats_value(payloads: tuple[dict[str, object], ...], key: str) -> float:
    values = []
    for payload in payloads:
        stats = payload.get("stats")
        if isinstance(stats, dict):
            stats_key = "transition_evals" if str(key) == "selected_transition_evals" else str(key)
            values.append(float(stats.get(stats_key, 0.0)))
        else:
            values.append(float(payload.get(key, 0.0)))
    return statistics.mean(values)


def _decode_payloads(
    *,
    mode: str,
    forward: fast.FrontierFastModel,
    backward: fast.FrontierFastModel,
    syndromes: tuple[int, ...],
    K: int,
    Delta: float,
    score_alpha: float,
    metric_mode: str,
    int_metric_scale: int,
) -> tuple[dict[str, object], ...]:
    if str(mode) == "replay":
        return fast._decode_frontier_fast_native_binary_committee_many_replay_payloads(
            forward,
            backward,
            syndromes,
            K=int(K),
            Delta=float(Delta),
            score_alpha=float(score_alpha),
            metric_mode=str(metric_mode),
            int_metric_scale=int(int_metric_scale),
            _assume_compatible=True,
        )
    return fast._decode_frontier_fast_native_binary_committee_many_payloads(
        forward,
        backward,
        syndromes,
        K=int(K),
        Delta=float(Delta),
        score_alpha=float(score_alpha),
        metric_mode=str(metric_mode),
        int_metric_scale=int(int_metric_scale),
        _assume_compatible=True,
        compact_payload=(str(mode) == "compact"),
    )


def main() -> int:
    args = _parse_args()
    scopes = tuple(args.scope) if args.scope else ("memory_X", "memory_Z")
    payload_mode = "full" if bool(args.full_payload) else str(args.payload)
    if int(args.rows_per_scope) <= 0:
        raise ValueError("--rows-per-scope must be positive")
    if int(args.int_metric_scale) <= 0:
        raise ValueError("--int-metric-scale must be positive")
    if int(args.repeats) <= 0:
        raise ValueError("--repeats must be positive")
    if int(args.warmups) < 0:
        raise ValueError("--warmups must be non-negative")

    syndromes_by_scope = _load_syndromes(
        Path(args.sample_rows),
        scopes=scopes,
        rows_per_scope=int(args.rows_per_scope),
    )

    print(
        "benchmark: BB144/Gross accepted split-sector DEM, "
        "D_X=D_Z=936x8784, O_X=O_Z=12x8784, rounds=12, "
        f"p={float(args.p_location):.6g}, K={int(args.K)}, Delta={float(args.Delta):.6g}, "
        f"score_alpha={float(args.score_alpha):.6g}, "
        f"metric_mode={args.metric_mode}, int_metric_scale={int(args.int_metric_scale)}, "
        f"payload={payload_mode}"
    )
    print(f"sample_rows: {Path(args.sample_rows)}")

    for scope in scopes:
        syndromes = syndromes_by_scope[scope]
        family, forward, backward = _build_models(
            backend=str(args.backend),
            p_location=float(args.p_location),
            column_order=str(args.column_order),
            scope=str(scope),
            syndrome=int(syndromes[0]),
        )
        print(
            f"{scope}: matrix={family.matrix_rows}x{family.matrix_cols}, "
            f"logical={family.logical_rows}x{family.matrix_cols}, rows={len(syndromes)}"
        )

        payloads: tuple[dict[str, object], ...] = tuple()
        for _ in range(int(args.warmups)):
            payloads = _decode_payloads(
                mode=payload_mode,
                forward=forward,
                backward=backward,
                syndromes=syndromes,
                K=int(args.K),
                Delta=float(args.Delta),
                score_alpha=float(args.score_alpha),
                metric_mode=str(args.metric_mode),
                int_metric_scale=int(args.int_metric_scale),
            )

        times: list[float] = []
        for _ in range(int(args.repeats)):
            started = time.perf_counter()
            payloads = _decode_payloads(
                mode=payload_mode,
                forward=forward,
                backward=backward,
                syndromes=syndromes,
                K=int(args.K),
                Delta=float(args.Delta),
                score_alpha=float(args.score_alpha),
                metric_mode=str(args.metric_mode),
                int_metric_scale=int(args.int_metric_scale),
            )
            times.append(time.perf_counter() - started)

        keep = times[2:] if len(times) > 2 else times
        print(
            f"{scope}: times_s={','.join(f'{value:.6f}' for value in times)} "
            f"mean_kept_s={statistics.mean(keep):.6f} "
            f"median_kept_s={statistics.median(keep):.6f} "
            f"ms_per_side_shot={1000.0 * statistics.mean(keep) / len(syndromes):.3f} "
            f"mean_transition_evals_total={_mean_payload_value(payloads, 'transition_evals_total'):.1f} "
            f"mean_selected_transition_evals={_mean_stats_value(payloads, 'selected_transition_evals'):.1f} "
            f"mean_selected_max_post={_mean_stats_value(payloads, 'max_post_prune_state_count'):.1f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
