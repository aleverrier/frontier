#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache_betterbeam")

from tools import frontier_decoder as frontier
from tools import gross144_dem_x_progressive_report as dem_report


PER_SHOT_FIELDS = [
    "decoder",
    "code",
    "scope",
    "shot",
    "source_shot",
    "source_row_identifier",
    "seed",
    "p_location",
    "backend",
    "matrix_rows",
    "matrix_cols",
    "logical_rows",
    "noisy_rounds",
    "K",
    "Delta",
    "selected_K",
    "selected_Delta",
    "score_alpha",
    "metric_mode",
    "int_metric_scale",
    "decoder_mode",
    "direction_mode",
    "pressure_estimator",
    "pressure_beta",
    "pressure_gamma",
    "candidate_pressure_gate",
    "pressure_forward",
    "pressure_backward",
    "engine_requested",
    "selected_engine",
    "forward_engine",
    "backward_engine",
    "selected_direction",
    "selected_direction_matches_committee_direction",
    "forward_decision",
    "backward_decision",
    "primary_selected_direction",
    "primary_status",
    "primary_logical_hat",
    "primary_forward_status",
    "primary_forward_logical_hat",
    "primary_backward_status",
    "primary_backward_logical_hat",
    "forward_log_evidence",
    "backward_log_evidence",
    "forward_terminal_top_log_mass_gap",
    "backward_terminal_top_log_mass_gap",
    "forward_transition_evals",
    "backward_transition_evals",
    "forward_max_post_prune_state_count",
    "backward_max_post_prune_state_count",
    "committee_disagreed",
    "escalated",
    "escalation_reason",
    "escalation_K",
    "escalation_Delta",
    "stage1_status",
    "stage1_accept",
    "stage1_selected_direction",
    "stage1_logical_hat",
    "stage1_forward_status",
    "stage1_forward_logical_hat",
    "stage1_backward_status",
    "stage1_backward_logical_hat",
    "stage1_transition_evals_total",
    "stage2_transition_evals_total",
    "stage1_forward_candidate_cols",
    "stage1_backward_candidate_cols",
    "status",
    "frame_ok",
    "frame_fail_type",
    "logical_hat",
    "truth_logical",
    "truth_detector_weight",
    "truth_logical_weight",
    "log_evidence",
    "terminal_top_log_mass_gap",
    "truth_present_terminal",
    "truth_rank_terminal",
    "failure_diagnosis",
    "decode_s",
    "transition_evals_total",
    "primary_transition_evals_total",
    "escalation_transition_evals_total",
    "selected_transition_evals",
    "max_pre_prune_state_count",
    "max_post_prune_state_count",
    "sum_pre_prune_state_count",
    "sum_post_prune_state_count",
    "processed_columns",
    "exception_message",
]

SUMMARY_FIELDS = [
    "decoder",
    "code",
    "scope",
    "trials",
    "fail_total",
    "fer",
    "fer_per_round",
    "logical_fail",
    "syndrome_fail",
    "exception_fail",
    "syndrome_failure",
    "truth_missing_terminal",
    "truth_present_but_not_selected",
    "bad_ranking",
    "diagnosis_available",
    "success",
    "decode_s_mean",
    "decode_s_p50",
    "decode_s_p95",
    "transition_evals_total_mean",
    "transition_evals_total_p50",
    "transition_evals_total_p95",
    "max_pre_prune_state_count_mean",
    "max_post_prune_state_count_mean",
    "sum_pre_prune_state_count_mean",
    "sum_post_prune_state_count_mean",
    "retained_states_mean",
    "max_pre_prune_state_count_max",
    "max_post_prune_state_count_max",
    "matrix_rows",
    "matrix_cols",
    "logical_rows",
    "noisy_rounds",
    "backend",
    "p_location",
    "K",
    "Delta",
    "escalation_K",
    "escalation_Delta",
    "score_alpha",
    "metric_mode",
    "int_metric_scale",
    "decoder_mode",
    "direction_mode",
    "pressure_estimator",
    "pressure_beta",
    "pressure_gamma",
    "candidate_pressure_gate",
    "pressure_forward_mean",
    "pressure_backward_mean",
    "selected_forward",
    "selected_backward",
    "selected_forward_fraction",
    "selected_backward_fraction",
    "engine_requested",
    "engines_seen",
    "escalated",
    "escalation_fraction",
    "committee_disagreed",
    "committee_disagreement_rate",
    "primary_transition_evals_total_mean",
    "escalation_transition_evals_total_mean",
    "sample_rows",
]


@dataclass(frozen=True, slots=True)
class SampleRow:
    scope: str
    shot: int
    seed: int
    syndrome: int
    logical: int
    detector_weight: int
    logical_weight: int
    source_shot: int = -1
    source_row_identifier: str = ""


@dataclass(frozen=True, slots=True)
class DecodeBundle:
    selected: frontier.FrontierResult
    forward: frontier.FrontierResult | None
    backward: frontier.FrontierResult | None
    selected_direction: str
    transition_evals_total: int
    forward_engine: str = ""
    backward_engine: str = ""
    selected_K: int = 0
    selected_Delta: float = float("nan")
    primary_selected: frontier.FrontierResult | None = None
    primary_forward: frontier.FrontierResult | None = None
    primary_backward: frontier.FrontierResult | None = None
    primary_selected_direction: str = ""
    primary_transition_evals_total: int = 0
    committee_disagreed: bool = False
    escalated: bool = False
    escalation_reason: str = ""
    escalation_K: int = 0
    escalation_Delta: float = float("nan")
    escalation_transition_evals_total: int = 0
    pressure_estimator: str = "none"
    pressure_beta: float = 8.0
    pressure_gamma: float = 2.0
    candidate_pressure_gate: str = "all_but_one"
    pressure_forward: float = float("nan")
    pressure_backward: float = float("nan")


@dataclass(frozen=True, slots=True)
class DirectionPressure:
    estimator: str
    beta: float
    gamma: float
    candidate_gate: str
    forward: float = float("nan")
    backward: float = float("nan")


def _parse_scopes(raw: str) -> tuple[str, ...]:
    return tuple(piece.strip() for piece in str(raw).split(",") if piece.strip())


def _normalize_direction_mode(direction_mode: str | None, decoder_mode: str) -> str:
    if direction_mode is None or not str(direction_mode).strip():
        mode = str(decoder_mode).strip().lower()
        if mode == "bidirectional_committee":
            return "fwd_bwd_committee"
        if mode == "forward":
            return "forward_only"
        if mode == "backward":
            return "backward_only"
        raise ValueError(f"unsupported decoder mode {decoder_mode!r}")
    mode = str(direction_mode).strip().lower()
    aliases = {
        "committee": "fwd_bwd_committee",
        "bidirectional_committee": "fwd_bwd_committee",
        "fwd_bwd": "fwd_bwd_committee",
        "forward": "forward_only",
        "backward": "backward_only",
    }
    normalized = aliases.get(mode, mode)
    if normalized not in {"fwd_bwd_committee", "forward_only", "backward_only"}:
        raise ValueError(
            "unsupported direction mode for this export; expected fwd_bwd_committee, "
            "forward_only, or backward_only"
        )
    return str(normalized)


def _decoder_mode_from_direction_mode(direction_mode: str) -> str:
    mode = str(direction_mode)
    if mode == "fwd_bwd_committee":
        return "bidirectional_committee"
    if mode == "forward_only":
        return "forward"
    if mode == "backward_only":
        return "backward"
    raise ValueError(f"unsupported direction mode {direction_mode!r}")


def _iter_set_bits(mask: int) -> Iterable[int]:
    value = int(mask)
    while value:
        low_bit = int(value) & -int(value)
        yield int(low_bit.bit_length() - 1)
        value ^= int(low_bit)


def _pressure_active_width(
    model: frontier.FrontierModel,
    *,
    syndrome: int,
    beta: float,
    gamma: float,
) -> float:
    active_widths = [
        int((int(active_mask) & int(syndrome)).bit_count())
        for active_mask in tuple(model.layout.active_masks_after_column)
    ]
    closing_active = [
        int((int(close_mask) & int(syndrome)).bit_count())
        for close_mask in tuple(model.layout.closing_masks)
    ]
    return float(sum(active_widths) + float(beta) * max(active_widths, default=0) + float(gamma) * sum(closing_active))


def _pressure_candidate_gate(
    model: frontier.FrontierModel,
    *,
    syndrome: int,
    beta: float,
    gamma: float,
    gate: str,
) -> float:
    gate_key = str(gate).strip().lower()
    if gate_key not in {"all_but_one", "overlap2"}:
        raise ValueError("--candidate-pressure-gate must be 'all_but_one' or 'overlap2'")
    row_touch_columns = tuple(model.layout.row_touch_columns)
    candidate_counts: list[int] = []
    closing_active: list[int] = []
    for column_index, active_mask in enumerate(tuple(model.layout.active_masks_after_column)):
        active_syndrome_mask = int(active_mask) & int(syndrome)
        closing_active.append(
            int((int(model.layout.closing_masks[int(column_index)]) & int(syndrome)).bit_count())
        )
        if int(active_syndrome_mask) == 0:
            candidate_counts.append(0)
            continue
        hit_counts: dict[int, int] = {}
        for row_index in _iter_set_bits(int(active_syndrome_mask)):
            if int(row_index) < 0 or int(row_index) >= len(row_touch_columns):
                continue
            for touched_column in tuple(row_touch_columns[int(row_index)]):
                touched = int(touched_column)
                if touched <= int(column_index):
                    continue
                hit_counts[touched] = int(hit_counts.get(touched, 0)) + 1
        if gate_key == "overlap2":
            candidate_counts.append(sum(1 for value in hit_counts.values() if int(value) >= 2))
        else:
            candidate_counts.append(len(hit_counts))
    return float(
        sum(candidate_counts)
        + float(beta) * max(candidate_counts, default=0)
        + float(gamma) * sum(closing_active)
    )


def _direction_pressure(
    *,
    model: frontier.FrontierModel,
    backward_model: frontier.FrontierModel | None,
    syndrome: int,
    estimator: str,
    beta: float,
    gamma: float,
    candidate_gate: str,
) -> DirectionPressure:
    estimator_key = str(estimator).strip().lower()
    if estimator_key in {"", "none"}:
        return DirectionPressure(
            estimator="none",
            beta=float(beta),
            gamma=float(gamma),
            candidate_gate=str(candidate_gate),
        )
    if backward_model is None:
        backward_model = frontier._coerce_model(model, syndrome_int=int(syndrome), direction="backward")
    if estimator_key == "active_width":
        scorer = _pressure_active_width
        return DirectionPressure(
            estimator=str(estimator_key),
            beta=float(beta),
            gamma=float(gamma),
            candidate_gate=str(candidate_gate),
            forward=float(scorer(model, syndrome=int(syndrome), beta=float(beta), gamma=float(gamma))),
            backward=float(scorer(backward_model, syndrome=int(syndrome), beta=float(beta), gamma=float(gamma))),
        )
    if estimator_key == "candidate_gate_pressure":
        return DirectionPressure(
            estimator=str(estimator_key),
            beta=float(beta),
            gamma=float(gamma),
            candidate_gate=str(candidate_gate),
            forward=float(
                _pressure_candidate_gate(
                    model,
                    syndrome=int(syndrome),
                    beta=float(beta),
                    gamma=float(gamma),
                    gate=str(candidate_gate),
                )
            ),
            backward=float(
                _pressure_candidate_gate(
                    backward_model,
                    syndrome=int(syndrome),
                    beta=float(beta),
                    gamma=float(gamma),
                    gate=str(candidate_gate),
                )
            ),
        )
    raise ValueError("--pressure-estimator must be 'none', 'active_width', or 'candidate_gate_pressure'")


def _copy_pressure(bundle: DecodeBundle, pressure: DirectionPressure) -> DecodeBundle:
    return DecodeBundle(
        selected=bundle.selected,
        forward=bundle.forward,
        backward=bundle.backward,
        selected_direction=str(bundle.selected_direction),
        transition_evals_total=int(bundle.transition_evals_total),
        forward_engine=str(bundle.forward_engine),
        backward_engine=str(bundle.backward_engine),
        selected_K=int(bundle.selected_K),
        selected_Delta=float(bundle.selected_Delta),
        primary_selected=bundle.primary_selected,
        primary_forward=bundle.primary_forward,
        primary_backward=bundle.primary_backward,
        primary_selected_direction=str(bundle.primary_selected_direction),
        primary_transition_evals_total=int(bundle.primary_transition_evals_total),
        committee_disagreed=bool(bundle.committee_disagreed),
        escalated=bool(bundle.escalated),
        escalation_reason=str(bundle.escalation_reason),
        escalation_K=int(bundle.escalation_K),
        escalation_Delta=float(bundle.escalation_Delta),
        escalation_transition_evals_total=int(bundle.escalation_transition_evals_total),
        pressure_estimator=str(pressure.estimator),
        pressure_beta=float(pressure.beta),
        pressure_gamma=float(pressure.gamma),
        candidate_pressure_gate=str(pressure.candidate_gate),
        pressure_forward=float(pressure.forward),
        pressure_backward=float(pressure.backward),
    )


def _load_sample_rows(
    path: Path,
    *,
    scopes: Sequence[str],
    shot_start: int,
    shot_stop: int,
) -> dict[str, list[SampleRow]]:
    wanted = {str(scope) for scope in scopes}
    selected: dict[tuple[str, int], SampleRow] = {}
    with Path(path).expanduser().resolve().open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            scope = str(raw["scope"])
            shot = int(raw["shot"])
            if scope not in wanted or shot < int(shot_start) or shot > int(shot_stop):
                continue
            row = SampleRow(
                scope=scope,
                shot=shot,
                source_shot=int(raw.get("source_shot", shot) or shot),
                source_row_identifier=str(
                    raw.get("source_row_identifier", "")
                    or f"{scope}:seed{int(raw.get('seed', 0))}:shot{shot}"
                ),
                seed=int(raw.get("seed", 0)),
                syndrome=int(raw["truth_syndrome"]),
                logical=int(raw["truth_logical"]),
                detector_weight=int(raw.get("truth_detector_weight", int(raw["truth_syndrome"]).bit_count())),
                logical_weight=int(raw.get("truth_logical_weight", int(raw["truth_logical"]).bit_count())),
            )
            old = selected.get((scope, shot))
            if old is not None and old != row:
                raise ValueError(f"conflicting sample row for {scope}:{shot}")
            selected[(scope, shot)] = row
    missing: list[str] = []
    for scope in sorted(wanted):
        for shot in range(int(shot_start), int(shot_stop) + 1):
            if (scope, shot) not in selected:
                missing.append(f"{scope}:{shot}")
    if missing:
        raise ValueError(f"missing {len(missing)} requested sample rows; first: {', '.join(missing[:20])}")
    by_scope = {scope: [] for scope in sorted(wanted)}
    for scope in sorted(wanted):
        by_scope[scope] = [selected[(scope, shot)] for shot in range(int(shot_start), int(shot_stop) + 1)]
    return by_scope


def _json_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _csv_write(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _split_rows(rows: Sequence[SampleRow], shards: int) -> list[list[SampleRow]]:
    values = list(rows)
    if not values:
        return []
    shard_count = max(1, min(int(shards), len(values)))
    base = len(values) // shard_count
    rem = len(values) % shard_count
    out: list[list[SampleRow]] = []
    start = 0
    for shard in range(shard_count):
        size = base + (1 if shard < rem else 0)
        out.append(values[start : start + size])
        start += size
    return out


def _load_family_pair(task: Mapping[str, object]):
    forward_family = dem_report._load_dem_family(
        backend=str(task["backend"]),
        p_location=float(task["p_location"]),
        scope=str(task["scope"]),
        column_order=str(task["column_order"]),
    )
    backward_family = None
    needs_backward = (
        str(task.get("decoder_mode", "")) in {"backward", "bidirectional_committee"}
        or str(task.get("direction_mode", "")) in {"backward_only", "fwd_bwd_committee"}
        or str(task.get("pressure_estimator", "none")) != "none"
    )
    if bool(needs_backward):
        backward_order = str(task["backward_column_order"]).strip().lower()
        if backward_order in {"bwd_deadline", "backward_deadline_reorder"}:
            backward_family = dem_report._build_backward_deadline_ordered_family(base_family=forward_family)
        else:
            backward_family = dem_report._load_dem_family(
                backend=str(task["backend"]),
                p_location=float(task["p_location"]),
                scope=str(task["scope"]),
                column_order=str(task["backward_column_order"]),
            )
    return forward_family, backward_family


def _make_model(forward_family, backward_family) -> frontier.FrontierModel:
    return frontier.FrontierModel(
        columns=tuple(forward_family.columns),
        layout=forward_family.layout,
        num_detectors=int(forward_family.matrix_rows),
        num_observables=int(forward_family.logical_rows),
        backward_columns=None if backward_family is None else tuple(backward_family.columns),
        backward_layout=None if backward_family is None else backward_family.layout,
    )


def _committee_outcome(result: frontier.FrontierResult | None) -> tuple[str, int | None]:
    if result is None:
        return ("missing", None)
    logical = None if result.logical_hat is None else int(result.logical_hat)
    return (str(result.status), logical)


def _committee_disagreed(
    forward_result: frontier.FrontierResult | None,
    backward_result: frontier.FrontierResult | None,
) -> bool:
    if forward_result is None or backward_result is None:
        return False
    return _committee_outcome(forward_result) != _committee_outcome(backward_result)


def _primary_or_selected(bundle: DecodeBundle) -> frontier.FrontierResult:
    return bundle.primary_selected if bundle.primary_selected is not None else bundle.selected


def _escalated_bundle(
    *,
    primary: DecodeBundle,
    escalated: DecodeBundle,
    reason: str,
    escalation_K: int,
    escalation_Delta: float,
) -> DecodeBundle:
    primary_selected = _primary_or_selected(primary)
    return DecodeBundle(
        selected=escalated.selected,
        forward=escalated.forward,
        backward=escalated.backward,
        selected_direction=str(escalated.selected_direction),
        transition_evals_total=int(primary.transition_evals_total) + int(escalated.transition_evals_total),
        forward_engine=str(escalated.forward_engine),
        backward_engine=str(escalated.backward_engine),
        selected_K=int(escalation_K),
        selected_Delta=float(escalation_Delta),
        primary_selected=primary_selected,
        primary_forward=primary.primary_forward if primary.primary_forward is not None else primary.forward,
        primary_backward=primary.primary_backward if primary.primary_backward is not None else primary.backward,
        primary_selected_direction=str(primary.primary_selected_direction or primary.selected_direction),
        primary_transition_evals_total=int(primary.primary_transition_evals_total or primary.transition_evals_total),
        committee_disagreed=bool(primary.committee_disagreed),
        escalated=True,
        escalation_reason=str(reason),
        escalation_K=int(escalation_K),
        escalation_Delta=float(escalation_Delta),
        escalation_transition_evals_total=int(escalated.transition_evals_total),
        pressure_estimator=str(primary.pressure_estimator),
        pressure_beta=float(primary.pressure_beta),
        pressure_gamma=float(primary.pressure_gamma),
        candidate_pressure_gate=str(primary.candidate_pressure_gate),
        pressure_forward=float(primary.pressure_forward),
        pressure_backward=float(primary.pressure_backward),
    )


def _select_committee(
    *,
    model: frontier.FrontierModel,
    syndrome: int,
    K: int,
    Delta: float,
    score_alpha: float,
    engine: str,
    metric_mode: str = "logsumexp_float",
    int_metric_scale: int = 1024,
) -> DecodeBundle:
    forward_model = frontier._coerce_model(model, syndrome_int=int(syndrome), direction="forward")
    backward_model = frontier._coerce_model(model, syndrome_int=int(syndrome), direction="backward")
    forward_result = frontier.decode_frontier(
        forward_model,
        int(syndrome),
        K=int(K),
        Delta=float(Delta),
        score_alpha=float(score_alpha),
        metric_mode=str(metric_mode),
        int_metric_scale=int(int_metric_scale),
        _engine=str(engine),
    )
    backward_result = frontier.decode_frontier(
        backward_model,
        int(syndrome),
        K=int(K),
        Delta=float(Delta),
        score_alpha=float(score_alpha),
        metric_mode=str(metric_mode),
        int_metric_scale=int(int_metric_scale),
        _engine=str(engine),
    )
    selected_direction, selected = max(
        (("forward", forward_result), ("backward", backward_result)),
        key=lambda item: frontier._committee_selection_key(
            result=item[1],
            direction=str(item[0]),
            preferred_direction="forward",
        ),
    )
    disagreed = _committee_disagreed(forward_result, backward_result)
    return DecodeBundle(
        selected=selected,
        forward=forward_result,
        backward=backward_result,
        selected_direction=str(selected_direction),
        transition_evals_total=int(forward_result.stats.transition_evals)
        + int(backward_result.stats.transition_evals),
        forward_engine=str(forward_result.engine),
        backward_engine=str(backward_result.engine),
        selected_K=int(K),
        selected_Delta=float(Delta),
        primary_selected=selected,
        primary_forward=forward_result,
        primary_backward=backward_result,
        primary_selected_direction=str(selected_direction),
        primary_transition_evals_total=int(forward_result.stats.transition_evals)
        + int(backward_result.stats.transition_evals),
        committee_disagreed=bool(disagreed),
    )


def _bundle_from_committee_results(
    forward_result: frontier.FrontierResult,
    backward_result: frontier.FrontierResult,
    *,
    K: int,
    Delta: float,
) -> DecodeBundle:
    selected_direction, selected = max(
        (("forward", forward_result), ("backward", backward_result)),
        key=lambda item: frontier._committee_selection_key(
            result=item[1],
            direction=str(item[0]),
            preferred_direction="forward",
        ),
    )
    disagreed = _committee_disagreed(forward_result, backward_result)
    return DecodeBundle(
        selected=selected,
        forward=forward_result,
        backward=backward_result,
        selected_direction=str(selected_direction),
        transition_evals_total=int(forward_result.stats.transition_evals)
        + int(backward_result.stats.transition_evals),
        forward_engine=str(forward_result.engine),
        backward_engine=str(backward_result.engine),
        selected_K=int(K),
        selected_Delta=float(Delta),
        primary_selected=selected,
        primary_forward=forward_result,
        primary_backward=backward_result,
        primary_selected_direction=str(selected_direction),
        primary_transition_evals_total=int(forward_result.stats.transition_evals)
        + int(backward_result.stats.transition_evals),
        committee_disagreed=bool(disagreed),
    )


def _native_payload_stats(payload: Mapping[str, object]) -> Mapping[str, object]:
    stats = payload.get("stats", {})
    return stats if isinstance(stats, Mapping) else {}


def _native_payload_transition_evals(payload: Mapping[str, object]) -> int:
    return int(_native_payload_stats(payload).get("transition_evals", 0))


def _native_payload_top1_posterior(payload: Mapping[str, object]) -> float:
    if str(payload.get("status", "no_path")) != "ok":
        return float("-inf")
    log_evidence = float(payload.get("log_evidence", float("-inf")))
    if not math.isfinite(float(log_evidence)):
        return float("-inf")
    masses = dict(payload.get("terminal_log_masses", {}))
    if not masses:
        return float("-inf")
    return max(math.exp(float(value) - float(log_evidence)) for value in masses.values())


def _native_payload_selection_key(
    payload: Mapping[str, object],
    *,
    direction: str,
    preferred_direction: str = "forward",
) -> tuple[float, ...]:
    status_key = str(payload.get("status", "no_path")).strip().lower()
    if status_key == "ok":
        status_rank = 2.0
    elif status_key == "no_path":
        status_rank = 1.0
    else:
        status_rank = 0.0
    log_evidence = (
        float(payload.get("log_evidence", float("-inf")))
        if status_key == "ok" and math.isfinite(float(payload.get("log_evidence", float("-inf"))))
        else float("-inf")
    )
    terminal_gap = float(payload.get("terminal_top_log_mass_gap", float("nan")))
    if math.isnan(float(terminal_gap)):
        terminal_gap = float("-inf")
    top1_key = float(_native_payload_top1_posterior(payload))
    if not math.isfinite(float(top1_key)):
        top1_key = float("-inf")
    preferred_bonus = 1.0 if str(direction) == str(preferred_direction) else 0.0
    return (
        float(status_rank),
        float(log_evidence),
        float(terminal_gap),
        float(top1_key),
        0.0,
        float(preferred_bonus),
    )


def _bundle_from_native_committee_payloads(
    forward_payload: Mapping[str, object],
    backward_payload: Mapping[str, object],
    *,
    K: int,
    Delta: float,
) -> DecodeBundle:
    selected_direction, selected_payload = max(
        (("forward", forward_payload), ("backward", backward_payload)),
        key=lambda item: _native_payload_selection_key(
            item[1],
            direction=str(item[0]),
            preferred_direction="forward",
        ),
    )
    forward_result = frontier._frontier_result_from_native_payload(
        dict(forward_payload),
        direction="forward",
    )
    backward_result = frontier._frontier_result_from_native_payload(
        dict(backward_payload),
        direction="backward",
    )
    selected = frontier._frontier_result_from_native_payload(
        dict(selected_payload),
        direction=str(selected_direction),
    )
    transition_total = int(_native_payload_transition_evals(forward_payload)) + int(
        _native_payload_transition_evals(backward_payload)
    )
    disagreed = _committee_disagreed(forward_result, backward_result)
    return DecodeBundle(
        selected=selected,
        forward=forward_result,
        backward=backward_result,
        selected_direction=str(selected_direction),
        transition_evals_total=int(transition_total),
        forward_engine="native_binary",
        backward_engine="native_binary",
        selected_K=int(K),
        selected_Delta=float(Delta),
        primary_selected=selected,
        primary_forward=forward_result,
        primary_backward=backward_result,
        primary_selected_direction=str(selected_direction),
        primary_transition_evals_total=int(transition_total),
        committee_disagreed=bool(disagreed),
    )


def _bundle_from_native_selected_committee_payload(
    payload: Mapping[str, object],
    *,
    K: int,
    Delta: float,
) -> DecodeBundle:
    selected_direction = str(payload.get("selected_direction", "forward"))
    selected = frontier._frontier_result_from_native_payload(
        dict(payload),
        direction=str(selected_direction),
    )
    transition_total = int(payload.get("transition_evals_total", selected.stats.transition_evals))
    return DecodeBundle(
        selected=selected,
        forward=None,
        backward=None,
        selected_direction=str(selected_direction),
        transition_evals_total=int(transition_total),
        forward_engine="native_binary",
        backward_engine="native_binary",
        selected_K=int(K),
        selected_Delta=float(Delta),
        primary_selected=selected,
        primary_selected_direction=str(selected_direction),
        primary_transition_evals_total=int(transition_total),
    )


def _decode_many_native_replay_payloads(
    *,
    model: frontier.FrontierModel,
    samples: Sequence[SampleRow],
    decoder_mode: str,
    direction_mode: str = "",
    K: int,
    Delta: float,
    score_alpha: float,
    engine: str,
    metric_mode: str = "logsumexp_float",
    int_metric_scale: int = 1024,
    escalate_on_committee_disagreement: bool = False,
) -> tuple[dict[str, object], ...] | None:
    if str(os.environ.get("FRONTIER_SAMPLE_REPLAY_DISABLE_FLAT_NATIVE_REPLAY", "")).strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }:
        return None
    if str(engine) not in {"auto", "native_binary"} or not frontier.native_binary_available():
        return None
    if str(decoder_mode) != "bidirectional_committee" or bool(escalate_on_committee_disagreement):
        return None
    sample_tuple = tuple(samples)
    if not sample_tuple:
        return tuple()
    syndromes = tuple(int(sample.syndrome) for sample in sample_tuple)
    try:
        forward_model = frontier._coerce_model(model, syndrome_int=int(syndromes[0]), direction="forward")
        backward_model = frontier._coerce_model(model, syndrome_int=int(syndromes[0]), direction="backward")
        if not frontier._is_native_binary_compatible(forward_model, syndrome=int(syndromes[0])):
            return None
        if not frontier._is_native_binary_compatible(backward_model, syndrome=int(syndromes[0])):
            return None
        return frontier._decode_frontier_native_binary_committee_many_replay_payloads(
            forward_model,
            backward_model,
            syndromes,
            K=int(K),
            Delta=float(Delta),
            score_alpha=float(score_alpha),
            metric_mode=str(metric_mode),
            int_metric_scale=int(int_metric_scale),
            _assume_compatible=True,
        )
    except (AttributeError, RuntimeError):
        return None


def _native_replay_logical_hat(payload: Mapping[str, object]) -> int | None:
    value = payload.get("logical_hat")
    if value is None or str(payload.get("status", "no_path")) != "ok":
        return None
    return int(value)


def _row_from_native_replay_payload(
    *,
    payload: Mapping[str, object],
    sample: SampleRow,
    task: Mapping[str, object],
    forward_family,
    decode_s: float,
) -> dict[str, object]:
    status = str(payload.get("status", "no_path"))
    logical_hat = _native_replay_logical_hat(payload)
    fail_type = _fail_type(
        status=str(status),
        logical_hat=logical_hat,
        truth_logical=int(sample.logical),
    )
    truth_present = False
    truth_rank = None
    if str(fail_type) == "logical_fail":
        failure_diagnosis = "undiagnosed_failure"
    else:
        failure_diagnosis = _failure_diagnosis(
            fail_type=str(fail_type),
            status=str(status),
            truth_present_terminal=bool(truth_present),
        )
    transition_total = int(payload.get("transition_evals_total", 0))
    forward_status = str(payload.get("forward_status", ""))
    backward_status = str(payload.get("backward_status", ""))
    forward_logical_hat = payload.get("forward_logical_hat")
    backward_logical_hat = payload.get("backward_logical_hat")
    committee_disagreed = False
    if forward_status and backward_status:
        committee_disagreed = (forward_status, forward_logical_hat) != (backward_status, backward_logical_hat)
    if "committee_disagreed" in payload:
        committee_disagreed = _boolish(payload.get("committee_disagreed", False))
    direction_mode = str(task.get("direction_mode", ""))
    selected_direction = str(payload.get("selected_direction", "forward"))
    stage1_status = str(payload.get("stage1_status", ""))
    stage1_logical_hat = payload.get("stage1_logical_hat")
    stage1_forward_logical_hat = payload.get("stage1_forward_logical_hat", forward_logical_hat)
    stage1_backward_logical_hat = payload.get("stage1_backward_logical_hat", backward_logical_hat)
    primary_logical_hat = payload.get("primary_logical_hat", logical_hat)
    escalated = _boolish(payload.get("escalated", False))
    escalation_reason = str(payload.get("escalation_reason", ""))
    primary_transition_total = int(payload.get("primary_transition_evals_total", transition_total))
    escalation_transition_total = int(payload.get("escalation_transition_evals_total", 0))
    decoder_label = (
        "frontier auto fwd/bwd committee"
        if str(task["decoder_mode"]) == "bidirectional_committee"
        else "frontier auto"
    )
    return {
        "decoder": str(decoder_label),
        "code": str(task["code"]),
        "scope": str(task["scope"]),
        "shot": int(sample.shot),
        "source_shot": int(sample.source_shot),
        "source_row_identifier": str(sample.source_row_identifier),
        "seed": int(sample.seed),
        "p_location": float(task["p_location"]),
        "backend": str(task["backend"]),
        "matrix_rows": int(forward_family.matrix_rows),
        "matrix_cols": int(forward_family.matrix_cols),
        "logical_rows": int(forward_family.logical_rows),
        "noisy_rounds": int(forward_family.noisy_rounds),
        "K": int(task["K"]),
        "Delta": float(task["Delta"]),
        "selected_K": int(task["K"]),
        "selected_Delta": float(task["Delta"]),
        "score_alpha": float(task["score_alpha"]),
        "metric_mode": str(task.get("metric_mode", "logsumexp_float")),
        "int_metric_scale": int(task.get("int_metric_scale", 1024) or 1024),
        "decoder_mode": str(task["decoder_mode"]),
        "direction_mode": str(task.get("direction_mode", "")),
        "pressure_estimator": str(task.get("pressure_estimator", "none")),
        "pressure_beta": float(task.get("pressure_beta", 8.0)),
        "pressure_gamma": float(task.get("pressure_gamma", 2.0)),
        "candidate_pressure_gate": str(task.get("candidate_pressure_gate", "all_but_one")),
        "pressure_forward": float("nan"),
        "pressure_backward": float("nan"),
        "engine_requested": str(task["engine"]),
        "selected_engine": "native_binary",
        "forward_engine": "native_binary",
        "backward_engine": "native_binary",
        "selected_direction": str(selected_direction),
        "selected_direction_matches_committee_direction": True,
        "forward_decision": f"{forward_status}:{'' if forward_logical_hat is None else int(forward_logical_hat)}",
        "backward_decision": f"{backward_status}:{'' if backward_logical_hat is None else int(backward_logical_hat)}",
        "primary_selected_direction": str(payload.get("primary_selected_direction", selected_direction)),
        "primary_status": str(payload.get("primary_status", status)),
        "primary_logical_hat": "" if primary_logical_hat is None else int(primary_logical_hat),
        "primary_forward_status": str(payload.get("stage1_forward_status", forward_status)),
        "primary_forward_logical_hat": ""
        if stage1_forward_logical_hat is None
        else int(stage1_forward_logical_hat),
        "primary_backward_status": str(payload.get("stage1_backward_status", backward_status)),
        "primary_backward_logical_hat": ""
        if stage1_backward_logical_hat is None
        else int(stage1_backward_logical_hat),
        "forward_log_evidence": float(payload.get("forward_log_evidence", float("nan"))),
        "backward_log_evidence": float(payload.get("backward_log_evidence", float("nan"))),
        "forward_terminal_top_log_mass_gap": float(payload.get("forward_terminal_top_log_mass_gap", float("nan"))),
        "backward_terminal_top_log_mass_gap": float(payload.get("backward_terminal_top_log_mass_gap", float("nan"))),
        "forward_transition_evals": int(payload.get("forward_transition_evals", 0)),
        "backward_transition_evals": int(payload.get("backward_transition_evals", 0)),
        "forward_max_post_prune_state_count": int(payload.get("forward_max_post_prune_state_count", 0)),
        "backward_max_post_prune_state_count": int(payload.get("backward_max_post_prune_state_count", 0)),
        "committee_disagreed": bool(committee_disagreed),
        "escalated": bool(escalated),
        "escalation_reason": str(escalation_reason),
        "escalation_K": int(task.get("escalation_K", 0) or 0),
        "escalation_Delta": float(task.get("escalation_Delta", float("nan"))),
        "stage1_status": str(stage1_status),
        "stage1_accept": _boolish(payload.get("stage1_accept", False)),
        "stage1_selected_direction": str(payload.get("stage1_selected_direction", "")),
        "stage1_logical_hat": "" if stage1_logical_hat is None else int(stage1_logical_hat),
        "stage1_forward_status": str(payload.get("stage1_forward_status", "")),
        "stage1_forward_logical_hat": ""
        if stage1_forward_logical_hat is None
        else int(stage1_forward_logical_hat),
        "stage1_backward_status": str(payload.get("stage1_backward_status", "")),
        "stage1_backward_logical_hat": ""
        if stage1_backward_logical_hat is None
        else int(stage1_backward_logical_hat),
        "stage1_transition_evals_total": int(payload.get("stage1_transition_evals_total", 0)),
        "stage2_transition_evals_total": int(payload.get("stage2_transition_evals_total", 0)),
        "stage1_forward_candidate_cols": int(payload.get("stage1_forward_candidate_cols", 0)),
        "stage1_backward_candidate_cols": int(payload.get("stage1_backward_candidate_cols", 0)),
        "status": str(status),
        "frame_ok": bool(fail_type == "success"),
        "frame_fail_type": str(fail_type),
        "logical_hat": "" if logical_hat is None else int(logical_hat),
        "truth_logical": int(sample.logical),
        "truth_detector_weight": int(sample.detector_weight),
        "truth_logical_weight": int(sample.logical_weight),
        "log_evidence": float(payload.get("log_evidence", float("-inf"))),
        "terminal_top_log_mass_gap": float(payload.get("terminal_top_log_mass_gap", float("nan"))),
        "truth_present_terminal": bool(truth_present),
        "truth_rank_terminal": "" if truth_rank is None else int(truth_rank),
        "failure_diagnosis": str(failure_diagnosis),
        "decode_s": float(decode_s),
        "transition_evals_total": int(transition_total),
        "primary_transition_evals_total": int(primary_transition_total),
        "escalation_transition_evals_total": int(escalation_transition_total),
        "selected_transition_evals": int(payload.get("selected_transition_evals", 0)),
        "max_pre_prune_state_count": int(payload.get("max_pre_prune_state_count", 0)),
        "max_post_prune_state_count": int(payload.get("max_post_prune_state_count", 0)),
        "sum_pre_prune_state_count": int(payload.get("sum_pre_prune_state_count", 0)),
        "sum_post_prune_state_count": int(payload.get("sum_post_prune_state_count", 0)),
        "processed_columns": int(payload.get("processed_columns", 0)),
        "exception_message": "",
    }


def _decode_one(
    *,
    model: frontier.FrontierModel,
    syndrome: int,
    decoder_mode: str,
    K: int,
    Delta: float,
    score_alpha: float,
    engine: str,
    metric_mode: str = "logsumexp_float",
    int_metric_scale: int = 1024,
    pressure_estimator: str = "none",
    pressure_beta: float = 8.0,
    pressure_gamma: float = 2.0,
    candidate_pressure_gate: str = "all_but_one",
    escalate_on_committee_disagreement: bool = False,
    escalation_K: int = 0,
    escalation_Delta: float = float("nan"),
) -> DecodeBundle:
    if str(decoder_mode) not in {"forward", "backward", "bidirectional_committee"}:
        raise ValueError("this export supports only forward, backward, and bidirectional_committee modes")
    backward_pressure_model = None
    if str(pressure_estimator) != "none":
        backward_pressure_model = frontier._coerce_model(model, syndrome_int=int(syndrome), direction="backward")
    pressure = _direction_pressure(
        model=model,
        backward_model=backward_pressure_model,
        syndrome=int(syndrome),
        estimator=str(pressure_estimator),
        beta=float(pressure_beta),
        gamma=float(pressure_gamma),
        candidate_gate=str(candidate_pressure_gate),
    )
    if str(decoder_mode) == "bidirectional_committee":
        primary = _select_committee(
            model=model,
            syndrome=int(syndrome),
            K=int(K),
            Delta=float(Delta),
            score_alpha=float(score_alpha),
            metric_mode=str(metric_mode),
            int_metric_scale=int(int_metric_scale),
            engine=str(engine),
        )
        if bool(escalate_on_committee_disagreement) and bool(primary.committee_disagreed):
            escalated = _select_committee(
                model=model,
                syndrome=int(syndrome),
                K=int(escalation_K),
                Delta=float(escalation_Delta),
                score_alpha=float(score_alpha),
                metric_mode=str(metric_mode),
                int_metric_scale=int(int_metric_scale),
                engine=str(engine),
            )
            return _escalated_bundle(
                primary=_copy_pressure(primary, pressure),
                escalated=escalated,
                reason="committee_disagreement",
                escalation_K=int(escalation_K),
                escalation_Delta=float(escalation_Delta),
            )
        return _copy_pressure(primary, pressure)
    if str(decoder_mode) == "backward":
        backward_model = frontier._coerce_model(model, syndrome_int=int(syndrome), direction="backward")
        result = frontier.decode_frontier(
            backward_model,
            int(syndrome),
            K=int(K),
            Delta=float(Delta),
            score_alpha=float(score_alpha),
            metric_mode=str(metric_mode),
            int_metric_scale=int(int_metric_scale),
            _engine=str(engine),
        )
        return _copy_pressure(
            DecodeBundle(
                selected=result,
                forward=None,
                backward=result,
                selected_direction="backward",
                transition_evals_total=int(result.stats.transition_evals),
                backward_engine=str(result.engine),
                selected_K=int(K),
                selected_Delta=float(Delta),
                primary_selected=result,
                primary_backward=result,
                primary_selected_direction="backward",
                primary_transition_evals_total=int(result.stats.transition_evals),
            ),
            pressure,
        )
    result = frontier.decode_frontier(
        model,
        int(syndrome),
        K=int(K),
        Delta=float(Delta),
        score_alpha=float(score_alpha),
        metric_mode=str(metric_mode),
        int_metric_scale=int(int_metric_scale),
        _engine=str(engine),
    )
    return _copy_pressure(
        DecodeBundle(
            selected=result,
            forward=result,
            backward=None,
            selected_direction="forward",
            transition_evals_total=int(result.stats.transition_evals),
            forward_engine=str(result.engine),
            selected_K=int(K),
            selected_Delta=float(Delta),
            primary_selected=result,
            primary_forward=result,
            primary_selected_direction="forward",
            primary_transition_evals_total=int(result.stats.transition_evals),
        ),
        pressure,
    )


def _decode_many_native_bundles(
    *,
    model: frontier.FrontierModel,
    samples: Sequence[SampleRow],
    decoder_mode: str,
    K: int,
    Delta: float,
    score_alpha: float,
    engine: str,
    metric_mode: str = "logsumexp_float",
    int_metric_scale: int = 1024,
    pressure_estimator: str = "none",
    pressure_beta: float = 8.0,
    pressure_gamma: float = 2.0,
    candidate_pressure_gate: str = "all_but_one",
    escalate_on_committee_disagreement: bool = False,
    escalation_K: int = 0,
    escalation_Delta: float = float("nan"),
) -> tuple[DecodeBundle, ...] | None:
    if str(decoder_mode) not in {"forward", "backward", "bidirectional_committee"}:
        raise ValueError("this export supports only forward, backward, and bidirectional_committee modes")
    if str(engine) not in {"auto", "native_binary"} or not frontier.native_binary_available():
        return None
    sample_tuple = tuple(samples)
    if not sample_tuple:
        return tuple()
    syndromes = tuple(int(sample.syndrome) for sample in sample_tuple)
    try:
        if str(decoder_mode) == "bidirectional_committee":
            forward_model = frontier._coerce_model(model, syndrome_int=int(syndromes[0]), direction="forward")
            backward_model = frontier._coerce_model(model, syndrome_int=int(syndromes[0]), direction="backward")
            if not frontier._is_native_binary_compatible(forward_model, syndrome=int(syndromes[0])):
                return None
            if not frontier._is_native_binary_compatible(backward_model, syndrome=int(syndromes[0])):
                return None
            pressures = tuple(
                _direction_pressure(
                    model=forward_model,
                    backward_model=backward_model,
                    syndrome=int(syndrome),
                    estimator=str(pressure_estimator),
                    beta=float(pressure_beta),
                    gamma=float(pressure_gamma),
                    candidate_gate=str(candidate_pressure_gate),
                )
                for syndrome in syndromes
            )
            has_pressure = str(pressure_estimator) != "none"
            if not bool(escalate_on_committee_disagreement) and not bool(has_pressure):
                try:
                    selected_payloads = frontier._decode_frontier_native_binary_committee_many_payloads(
                        forward_model,
                        backward_model,
                        syndromes,
                        K=int(K),
                        Delta=float(Delta),
                        score_alpha=float(score_alpha),
                        metric_mode=str(metric_mode),
                        int_metric_scale=int(int_metric_scale),
                        _assume_compatible=True,
                        compact_payload=True,
                    )
                    return tuple(
                        _bundle_from_native_selected_committee_payload(
                            payload,
                            K=int(K),
                            Delta=float(Delta),
                        )
                        for payload in selected_payloads
                    )
                except (AttributeError, RuntimeError):
                    pass
            forward_payloads = frontier._decode_frontier_native_binary_many_payloads(
                forward_model,
                syndromes,
                K=int(K),
                Delta=float(Delta),
                score_alpha=float(score_alpha),
                metric_mode=str(metric_mode),
                int_metric_scale=int(int_metric_scale),
                _assume_compatible=True,
            )
            backward_payloads = frontier._decode_frontier_native_binary_many_payloads(
                backward_model,
                syndromes,
                K=int(K),
                Delta=float(Delta),
                score_alpha=float(score_alpha),
                metric_mode=str(metric_mode),
                int_metric_scale=int(int_metric_scale),
                _assume_compatible=True,
            )
            primary_bundles = tuple(
                _copy_pressure(
                    _bundle_from_native_committee_payloads(
                        forward_payload,
                        backward_payload,
                        K=int(K),
                        Delta=float(Delta),
                    ),
                    pressures[int(index)],
                )
                for index, (forward_payload, backward_payload) in enumerate(
                    zip(forward_payloads, backward_payloads, strict=True)
                )
            )
            if not bool(escalate_on_committee_disagreement):
                return primary_bundles
            escalation_indices = tuple(
                index for index, bundle in enumerate(primary_bundles) if bool(bundle.committee_disagreed)
            )
            if not escalation_indices:
                return primary_bundles
            escalated_syndromes = tuple(int(syndromes[index]) for index in escalation_indices)
            try:
                selected_payloads = frontier._decode_frontier_native_binary_committee_many_payloads(
                    forward_model,
                    backward_model,
                    escalated_syndromes,
                    K=int(escalation_K),
                    Delta=float(escalation_Delta),
                    score_alpha=float(score_alpha),
                    metric_mode=str(metric_mode),
                    int_metric_scale=int(int_metric_scale),
                    _assume_compatible=True,
                    compact_payload=True,
                )
                escalated_bundles = tuple(
                    _bundle_from_native_selected_committee_payload(
                        payload,
                        K=int(escalation_K),
                        Delta=float(escalation_Delta),
                    )
                    for payload in selected_payloads
                )
            except (AttributeError, RuntimeError):
                escalated_forward_payloads = frontier._decode_frontier_native_binary_many_payloads(
                    forward_model,
                    escalated_syndromes,
                    K=int(escalation_K),
                    Delta=float(escalation_Delta),
                    score_alpha=float(score_alpha),
                    metric_mode=str(metric_mode),
                    int_metric_scale=int(int_metric_scale),
                    _assume_compatible=True,
                )
                escalated_backward_payloads = frontier._decode_frontier_native_binary_many_payloads(
                    backward_model,
                    escalated_syndromes,
                    K=int(escalation_K),
                    Delta=float(escalation_Delta),
                    score_alpha=float(score_alpha),
                    metric_mode=str(metric_mode),
                    int_metric_scale=int(int_metric_scale),
                    _assume_compatible=True,
                )
                escalated_bundles = tuple(
                    _bundle_from_native_committee_payloads(
                        forward_payload,
                        backward_payload,
                        K=int(escalation_K),
                        Delta=float(escalation_Delta),
                    )
                    for forward_payload, backward_payload in zip(
                        escalated_forward_payloads,
                        escalated_backward_payloads,
                        strict=True,
                    )
                )
            out = list(primary_bundles)
            for index, escalated_bundle in zip(escalation_indices, escalated_bundles, strict=True):
                out[int(index)] = _escalated_bundle(
                    primary=primary_bundles[int(index)],
                    escalated=escalated_bundle,
                    reason="committee_disagreement",
                    escalation_K=int(escalation_K),
                    escalation_Delta=float(escalation_Delta),
                )
            return tuple(out)
        if str(decoder_mode) == "backward":
            backward_model = frontier._coerce_model(model, syndrome_int=int(syndromes[0]), direction="backward")
            if not frontier._is_native_binary_compatible(backward_model, syndrome=int(syndromes[0])):
                return None
            pressures = tuple(
                _direction_pressure(
                    model=model,
                    backward_model=backward_model,
                    syndrome=int(syndrome),
                    estimator=str(pressure_estimator),
                    beta=float(pressure_beta),
                    gamma=float(pressure_gamma),
                    candidate_gate=str(candidate_pressure_gate),
                )
                for syndrome in syndromes
            )
            results = frontier._decode_frontier_native_binary_many(
                backward_model,
                syndromes,
                K=int(K),
                Delta=float(Delta),
                score_alpha=float(score_alpha),
                metric_mode=str(metric_mode),
                int_metric_scale=int(int_metric_scale),
                direction="backward",
                _assume_compatible=True,
            )
            return tuple(
                _copy_pressure(
                    DecodeBundle(
                        selected=result,
                        forward=None,
                        backward=result,
                        selected_direction="backward",
                        transition_evals_total=int(result.stats.transition_evals),
                        backward_engine=str(result.engine),
                        selected_K=int(K),
                        selected_Delta=float(Delta),
                        primary_selected=result,
                        primary_backward=result,
                        primary_selected_direction="backward",
                        primary_transition_evals_total=int(result.stats.transition_evals),
                    ),
                    pressures[int(index)],
                )
                for index, result in enumerate(results)
            )
        if not frontier._is_native_binary_compatible(model, syndrome=int(syndromes[0])):
            return None
        backward_model = (
            frontier._coerce_model(model, syndrome_int=int(syndromes[0]), direction="backward")
            if str(pressure_estimator) != "none"
            else None
        )
        pressures = tuple(
            _direction_pressure(
                model=model,
                backward_model=backward_model,
                syndrome=int(syndrome),
                estimator=str(pressure_estimator),
                beta=float(pressure_beta),
                gamma=float(pressure_gamma),
                candidate_gate=str(candidate_pressure_gate),
            )
            for syndrome in syndromes
        )
        results = frontier._decode_frontier_native_binary_many(
            model,
            syndromes,
            K=int(K),
            Delta=float(Delta),
            score_alpha=float(score_alpha),
            metric_mode=str(metric_mode),
            int_metric_scale=int(int_metric_scale),
            direction="forward",
            _assume_compatible=True,
        )
        return tuple(
            _copy_pressure(
                DecodeBundle(
                    selected=result,
                    forward=result,
                    backward=None,
                    selected_direction="forward",
                    transition_evals_total=int(result.stats.transition_evals),
                    forward_engine=str(result.engine),
                    selected_K=int(K),
                    selected_Delta=float(Delta),
                    primary_selected=result,
                    primary_forward=result,
                    primary_selected_direction="forward",
                    primary_transition_evals_total=int(result.stats.transition_evals),
                ),
                pressures[int(index)],
            )
            for index, result in enumerate(results)
        )
    except Exception:
        return None


def _fail_type(*, status: str, logical_hat: int | None, truth_logical: int) -> str:
    if str(status) != "ok":
        return "exception_fail" if str(status) == "exception" else "syndrome_fail"
    if logical_hat is None:
        return "logical_fail"
    return "success" if int(logical_hat) == int(truth_logical) else "logical_fail"


def _boolish(value: object) -> bool:
    if isinstance(value, bool):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _terminal_truth_rank(result: frontier.FrontierResult, *, truth_logical: int) -> tuple[bool, int | None]:
    if str(result.status) != "ok":
        return (False, None)
    masses = dict(result.terminal_log_masses)
    truth = int(truth_logical)
    if truth not in masses:
        return (False, None)
    ordered = sorted(masses, key=lambda logical: (-float(masses[int(logical)]), int(logical)))
    return (True, int(ordered.index(truth) + 1))


def _failure_diagnosis(
    *,
    fail_type: str,
    status: str,
    truth_present_terminal: bool,
) -> str:
    if str(fail_type) == "success":
        return "success"
    if str(fail_type) == "exception_fail" or str(status) == "exception":
        return "exception_failure"
    if str(fail_type) == "syndrome_fail":
        return "syndrome_failure"
    if str(fail_type) == "logical_fail":
        if bool(truth_present_terminal):
            return "truth_present_but_not_selected"
        return "truth_missing_terminal"
    return "undiagnosed_failure"


def _task_sample_rows(raw_rows: Sequence[Mapping[str, object]]) -> list[SampleRow]:
    return [
        SampleRow(
            scope=str(raw["scope"]),
            shot=int(raw["shot"]),
            source_shot=int(raw.get("source_shot", raw["shot"]) or raw["shot"]),
            source_row_identifier=str(
                raw.get("source_row_identifier", "")
                or f"{raw['scope']}:seed{int(raw.get('seed', 0))}:shot{int(raw['shot'])}"
            ),
            seed=int(raw["seed"]),
            syndrome=int(raw["syndrome"]),
            logical=int(raw["logical"]),
            detector_weight=int(raw["detector_weight"]),
            logical_weight=int(raw["logical_weight"]),
        )
        for raw in raw_rows
    ]


def _decoder_label(decoder_mode: str) -> str:
    if str(decoder_mode) == "bidirectional_committee":
        return "frontier auto fwd/bwd committee"
    if str(decoder_mode) == "backward":
        return "frontier auto backward"
    return "frontier auto"


def _result_decision(result: frontier.FrontierResult | None) -> str:
    if result is None:
        return ""
    logical = "" if result.logical_hat is None else str(int(result.logical_hat))
    return f"{result.status}:{logical}"


def _result_log_evidence(result: frontier.FrontierResult | None) -> float:
    return float("nan") if result is None else float(result.log_evidence)


def _result_terminal_gap(result: frontier.FrontierResult | None) -> float:
    return float("nan") if result is None else float(result.terminal_top_log_mass_gap)


def _result_transition_evals(result: frontier.FrontierResult | None) -> int:
    return 0 if result is None else int(result.stats.transition_evals)


def _result_max_post(result: frontier.FrontierResult | None) -> int:
    return 0 if result is None else int(result.stats.max_post_prune_state_count)


def _run_shard(task: Mapping[str, object]) -> dict[str, object]:
    started = time.time()
    partial_path = Path(str(task["partial_path"]))
    progress_path = Path(str(task["progress_path"]))
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    forward_family, backward_family = _load_family_pair(task)
    model = _make_model(forward_family, backward_family)
    samples = _task_sample_rows(task["sample_rows"])  # type: ignore[arg-type]
    native_batch_size = max(1, int(task.get("native_batch_size", 32) or 32))
    batch_bundle_by_index: dict[int, object] = {}
    batch_replay_payload_by_index: dict[int, Mapping[str, object]] = {}
    batch_decode_s_by_index: dict[int, float] = {}
    native_batch_disabled = False
    if bool(task.get("warmup_native", False)) and samples:
        warmup_bundles = _decode_many_native_bundles(
            model=model,
            samples=samples[:1],
            decoder_mode=str(task["decoder_mode"]),
            K=int(task["K"]),
            Delta=float(task["Delta"]),
            score_alpha=float(task["score_alpha"]),
            metric_mode=str(task.get("metric_mode", "logsumexp_float")),
            int_metric_scale=int(task.get("int_metric_scale", 1024) or 1024),
            engine=str(task["engine"]),
            pressure_estimator=str(task.get("pressure_estimator", "none")),
            pressure_beta=float(task.get("pressure_beta", 8.0)),
            pressure_gamma=float(task.get("pressure_gamma", 2.0)),
            candidate_pressure_gate=str(task.get("candidate_pressure_gate", "all_but_one")),
            escalate_on_committee_disagreement=bool(task.get("escalate_on_committee_disagreement", False)),
            escalation_K=int(task.get("escalation_K", 0) or 0),
            escalation_Delta=float(task.get("escalation_Delta", float("nan"))),
        )
        if warmup_bundles is None:
            _decode_one(
                model=model,
                syndrome=int(samples[0].syndrome),
                decoder_mode=str(task["decoder_mode"]),
                K=int(task["K"]),
                Delta=float(task["Delta"]),
                score_alpha=float(task["score_alpha"]),
                metric_mode=str(task.get("metric_mode", "logsumexp_float")),
                int_metric_scale=int(task.get("int_metric_scale", 1024) or 1024),
                engine=str(task["engine"]),
                pressure_estimator=str(task.get("pressure_estimator", "none")),
                pressure_beta=float(task.get("pressure_beta", 8.0)),
                pressure_gamma=float(task.get("pressure_gamma", 2.0)),
                candidate_pressure_gate=str(task.get("candidate_pressure_gate", "all_but_one")),
                escalate_on_committee_disagreement=bool(task.get("escalate_on_committee_disagreement", False)),
                escalation_K=int(task.get("escalation_K", 0) or 0),
                escalation_Delta=float(task.get("escalation_Delta", float("nan"))),
            )

    rows: list[dict[str, object]] = []
    with partial_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_SHOT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for local_index, sample in enumerate(samples, start=1):
            if (
                not native_batch_disabled
                and local_index not in batch_bundle_by_index
                and local_index not in batch_replay_payload_by_index
            ):
                chunk_samples = samples[int(local_index) - 1 : int(local_index) - 1 + int(native_batch_size)]
                batch_started = time.perf_counter()
                batch_replay_payloads = None
                if str(task.get("pressure_estimator", "none")) == "none":
                    batch_replay_payloads = _decode_many_native_replay_payloads(
                        model=model,
                        samples=chunk_samples,
                        decoder_mode=str(task["decoder_mode"]),
                        direction_mode=str(task.get("direction_mode", "")),
                        K=int(task["K"]),
                        Delta=float(task["Delta"]),
                        score_alpha=float(task["score_alpha"]),
                        metric_mode=str(task.get("metric_mode", "logsumexp_float")),
                        int_metric_scale=int(task.get("int_metric_scale", 1024) or 1024),
                        engine=str(task["engine"]),
                        escalate_on_committee_disagreement=bool(task.get("escalate_on_committee_disagreement", False)),
                    )
                if batch_replay_payloads is not None:
                    batch_decode_s = float(time.perf_counter() - batch_started) / float(len(chunk_samples))
                    for offset, payload in enumerate(batch_replay_payloads):
                        batch_index = int(local_index) + int(offset)
                        batch_replay_payload_by_index[batch_index] = payload
                        batch_decode_s_by_index[batch_index] = float(batch_decode_s)
                else:
                    batch_bundles = _decode_many_native_bundles(
                        model=model,
                        samples=chunk_samples,
                        decoder_mode=str(task["decoder_mode"]),
                        K=int(task["K"]),
                        Delta=float(task["Delta"]),
                        score_alpha=float(task["score_alpha"]),
                        metric_mode=str(task.get("metric_mode", "logsumexp_float")),
                        int_metric_scale=int(task.get("int_metric_scale", 1024) or 1024),
                        engine=str(task["engine"]),
                        pressure_estimator=str(task.get("pressure_estimator", "none")),
                        pressure_beta=float(task.get("pressure_beta", 8.0)),
                        pressure_gamma=float(task.get("pressure_gamma", 2.0)),
                        candidate_pressure_gate=str(task.get("candidate_pressure_gate", "all_but_one")),
                        escalate_on_committee_disagreement=bool(task.get("escalate_on_committee_disagreement", False)),
                        escalation_K=int(task.get("escalation_K", 0) or 0),
                        escalation_Delta=float(task.get("escalation_Delta", float("nan"))),
                    )
                    if batch_bundles is None:
                        native_batch_disabled = True
                    else:
                        batch_decode_s = float(time.perf_counter() - batch_started) / float(len(chunk_samples))
                        for offset, bundle in enumerate(batch_bundles):
                            batch_index = int(local_index) + int(offset)
                            batch_bundle_by_index[batch_index] = bundle
                            batch_decode_s_by_index[batch_index] = float(batch_decode_s)
            shot_started = time.perf_counter()
            try:
                replay_payload = batch_replay_payload_by_index.pop(int(local_index), None)
                if replay_payload is not None:
                    decode_s = float(batch_decode_s_by_index.pop(int(local_index), 0.0))
                    row = _row_from_native_replay_payload(
                        payload=replay_payload,
                        sample=sample,
                        task=task,
                        forward_family=forward_family,
                        decode_s=float(decode_s),
                    )
                    rows.append(row)
                    writer.writerow(row)
                    if local_index % max(1, int(task.get("progress_every", 100))) == 0 or local_index == len(samples):
                        elapsed = time.time() - started
                        progress = {
                            "scope": str(task["scope"]),
                            "completed": int(local_index),
                            "total": int(len(samples)),
                            "elapsed_s": float(elapsed),
                            "rate_shots_per_s": float(local_index) / float(elapsed) if elapsed > 0 else None,
                            "last_shot": int(sample.shot),
                            "partial_path": str(partial_path),
                        }
                        _json_write(progress_path, progress)
                    continue
                bundle = batch_bundle_by_index.pop(int(local_index), None)
                if bundle is None:
                    bundle = _decode_one(
                        model=model,
                        syndrome=int(sample.syndrome),
                        decoder_mode=str(task["decoder_mode"]),
                        K=int(task["K"]),
                        Delta=float(task["Delta"]),
                        score_alpha=float(task["score_alpha"]),
                        metric_mode=str(task.get("metric_mode", "logsumexp_float")),
                        int_metric_scale=int(task.get("int_metric_scale", 1024) or 1024),
                        engine=str(task["engine"]),
                        pressure_estimator=str(task.get("pressure_estimator", "none")),
                        pressure_beta=float(task.get("pressure_beta", 8.0)),
                        pressure_gamma=float(task.get("pressure_gamma", 2.0)),
                        candidate_pressure_gate=str(task.get("candidate_pressure_gate", "all_but_one")),
                        escalate_on_committee_disagreement=bool(task.get("escalate_on_committee_disagreement", False)),
                        escalation_K=int(task.get("escalation_K", 0) or 0),
                        escalation_Delta=float(task.get("escalation_Delta", float("nan"))),
                    )
                    decode_s = float(time.perf_counter() - shot_started)
                else:
                    decode_s = float(batch_decode_s_by_index.pop(int(local_index), 0.0))
                selected = bundle.selected
                primary_selected = _primary_or_selected(bundle)
                primary_forward = bundle.primary_forward
                primary_backward = bundle.primary_backward
                fail_type = _fail_type(
                    status=str(selected.status),
                    logical_hat=selected.logical_hat,
                    truth_logical=int(sample.logical),
                )
                truth_present, truth_rank = _terminal_truth_rank(selected, truth_logical=int(sample.logical))
                failure_diagnosis = _failure_diagnosis(
                    fail_type=str(fail_type),
                    status=str(selected.status),
                    truth_present_terminal=bool(truth_present),
                )
                row = {
                    "decoder": _decoder_label(str(task["decoder_mode"])),
                    "code": str(task["code"]),
                    "scope": str(task["scope"]),
                    "shot": int(sample.shot),
                    "seed": int(sample.seed),
                    "p_location": float(task["p_location"]),
                    "backend": str(task["backend"]),
                    "matrix_rows": int(forward_family.matrix_rows),
                    "matrix_cols": int(forward_family.matrix_cols),
                    "logical_rows": int(forward_family.logical_rows),
                    "noisy_rounds": int(forward_family.noisy_rounds),
                    "K": int(task["K"]),
                    "Delta": float(task["Delta"]),
                    "selected_K": int(bundle.selected_K or task["K"]),
                    "selected_Delta": (
                        float(bundle.selected_Delta)
                        if math.isfinite(float(bundle.selected_Delta))
                        else float(task["Delta"])
                    ),
                    "score_alpha": float(task["score_alpha"]),
                    "metric_mode": str(task.get("metric_mode", "logsumexp_float")),
                    "int_metric_scale": int(task.get("int_metric_scale", 1024) or 1024),
                    "decoder_mode": str(task["decoder_mode"]),
                    "direction_mode": str(task.get("direction_mode", "")),
                    "pressure_estimator": str(bundle.pressure_estimator),
                    "pressure_beta": float(bundle.pressure_beta),
                    "pressure_gamma": float(bundle.pressure_gamma),
                    "candidate_pressure_gate": str(bundle.candidate_pressure_gate),
                    "pressure_forward": float(bundle.pressure_forward),
                    "pressure_backward": float(bundle.pressure_backward),
                    "engine_requested": str(task["engine"]),
                    "selected_engine": str(selected.engine),
                    "forward_engine": str(bundle.forward_engine)
                    if str(bundle.forward_engine)
                    else ("" if bundle.forward is None else str(bundle.forward.engine)),
                    "backward_engine": str(bundle.backward_engine)
                    if str(bundle.backward_engine)
                    else ("" if bundle.backward is None else str(bundle.backward.engine)),
                    "selected_direction": str(bundle.selected_direction),
                    "selected_direction_matches_committee_direction": (
                        True if str(task["decoder_mode"]) == "bidirectional_committee" else ""
                    ),
                    "forward_decision": _result_decision(primary_forward),
                    "backward_decision": _result_decision(primary_backward),
                    "primary_selected_direction": str(bundle.primary_selected_direction or bundle.selected_direction),
                    "primary_status": str(primary_selected.status),
                    "primary_logical_hat": ""
                    if primary_selected.logical_hat is None
                    else int(primary_selected.logical_hat),
                    "primary_forward_status": "" if primary_forward is None else str(primary_forward.status),
                    "primary_forward_logical_hat": ""
                    if primary_forward is None or primary_forward.logical_hat is None
                    else int(primary_forward.logical_hat),
                    "primary_backward_status": "" if primary_backward is None else str(primary_backward.status),
                    "primary_backward_logical_hat": ""
                    if primary_backward is None or primary_backward.logical_hat is None
                    else int(primary_backward.logical_hat),
                    "forward_log_evidence": _result_log_evidence(primary_forward),
                    "backward_log_evidence": _result_log_evidence(primary_backward),
                    "forward_terminal_top_log_mass_gap": _result_terminal_gap(primary_forward),
                    "backward_terminal_top_log_mass_gap": _result_terminal_gap(primary_backward),
                    "forward_transition_evals": _result_transition_evals(primary_forward),
                    "backward_transition_evals": _result_transition_evals(primary_backward),
                    "forward_max_post_prune_state_count": _result_max_post(primary_forward),
                    "backward_max_post_prune_state_count": _result_max_post(primary_backward),
                    "committee_disagreed": bool(bundle.committee_disagreed),
                    "escalated": bool(bundle.escalated),
                    "escalation_reason": str(bundle.escalation_reason),
                    "escalation_K": int(task.get("escalation_K", 0) or 0),
                    "escalation_Delta": float(task.get("escalation_Delta", float("nan"))),
                    "status": str(selected.status),
                    "frame_ok": bool(fail_type == "success"),
                    "frame_fail_type": str(fail_type),
                    "logical_hat": "" if selected.logical_hat is None else int(selected.logical_hat),
                    "truth_logical": int(sample.logical),
                    "truth_detector_weight": int(sample.detector_weight),
                    "truth_logical_weight": int(sample.logical_weight),
                    "log_evidence": float(selected.log_evidence),
                    "terminal_top_log_mass_gap": float(selected.terminal_top_log_mass_gap),
                    "truth_present_terminal": bool(truth_present),
                    "truth_rank_terminal": "" if truth_rank is None else int(truth_rank),
                    "failure_diagnosis": str(failure_diagnosis),
                    "decode_s": float(decode_s),
                    "transition_evals_total": int(bundle.transition_evals_total),
                    "primary_transition_evals_total": int(
                        bundle.primary_transition_evals_total or bundle.transition_evals_total
                    ),
                    "escalation_transition_evals_total": int(bundle.escalation_transition_evals_total),
                    "selected_transition_evals": int(selected.stats.transition_evals),
                    "max_pre_prune_state_count": int(selected.stats.max_pre_prune_state_count),
                    "max_post_prune_state_count": int(selected.stats.max_post_prune_state_count),
                    "sum_pre_prune_state_count": int(selected.stats.sum_pre_prune_state_count),
                    "sum_post_prune_state_count": int(selected.stats.sum_post_prune_state_count),
                    "processed_columns": int(selected.stats.processed_columns),
                    "exception_message": "",
                }
            except Exception as exc:
                row = {
                    "decoder": _decoder_label(str(task["decoder_mode"])),
                    "code": str(task["code"]),
                    "scope": str(task["scope"]),
                    "shot": int(sample.shot),
                    "seed": int(sample.seed),
                    "p_location": float(task["p_location"]),
                    "backend": str(task["backend"]),
                    "matrix_rows": int(forward_family.matrix_rows),
                    "matrix_cols": int(forward_family.matrix_cols),
                    "logical_rows": int(forward_family.logical_rows),
                    "noisy_rounds": int(forward_family.noisy_rounds),
                    "K": int(task["K"]),
                    "Delta": float(task["Delta"]),
                    "selected_K": int(task["K"]),
                    "selected_Delta": float(task["Delta"]),
                    "score_alpha": float(task["score_alpha"]),
                    "metric_mode": str(task.get("metric_mode", "logsumexp_float")),
                    "int_metric_scale": int(task.get("int_metric_scale", 1024) or 1024),
                    "decoder_mode": str(task["decoder_mode"]),
                    "direction_mode": str(task.get("direction_mode", "")),
                    "pressure_estimator": str(task.get("pressure_estimator", "none")),
                    "pressure_beta": float(task.get("pressure_beta", 8.0)),
                    "pressure_gamma": float(task.get("pressure_gamma", 2.0)),
                    "candidate_pressure_gate": str(task.get("candidate_pressure_gate", "all_but_one")),
                    "pressure_forward": float("nan"),
                    "pressure_backward": float("nan"),
                    "engine_requested": str(task["engine"]),
                    "selected_engine": "",
                    "forward_engine": "",
                    "backward_engine": "",
                    "selected_direction": "",
                    "selected_direction_matches_committee_direction": "",
                    "forward_decision": "",
                    "backward_decision": "",
                    "primary_selected_direction": "",
                    "primary_status": "exception",
                    "primary_logical_hat": "",
                    "primary_forward_status": "",
                    "primary_forward_logical_hat": "",
                    "primary_backward_status": "",
                    "primary_backward_logical_hat": "",
                    "forward_log_evidence": float("nan"),
                    "backward_log_evidence": float("nan"),
                    "forward_terminal_top_log_mass_gap": float("nan"),
                    "backward_terminal_top_log_mass_gap": float("nan"),
                    "forward_transition_evals": 0,
                    "backward_transition_evals": 0,
                    "forward_max_post_prune_state_count": 0,
                    "backward_max_post_prune_state_count": 0,
                    "committee_disagreed": False,
                    "escalated": False,
                    "escalation_reason": "",
                    "escalation_K": int(task.get("escalation_K", 0) or 0),
                    "escalation_Delta": float(task.get("escalation_Delta", float("nan"))),
                    "status": "exception",
                    "frame_ok": False,
                    "frame_fail_type": "exception_fail",
                    "logical_hat": "",
                    "truth_logical": int(sample.logical),
                    "truth_detector_weight": int(sample.detector_weight),
                    "truth_logical_weight": int(sample.logical_weight),
                    "log_evidence": float("-inf"),
                    "terminal_top_log_mass_gap": float("nan"),
                    "truth_present_terminal": False,
                    "truth_rank_terminal": "",
                    "failure_diagnosis": "exception_failure",
                    "decode_s": float(time.perf_counter() - shot_started),
                    "transition_evals_total": 0,
                    "primary_transition_evals_total": 0,
                    "escalation_transition_evals_total": 0,
                    "selected_transition_evals": 0,
                    "max_pre_prune_state_count": 0,
                    "max_post_prune_state_count": 0,
                    "sum_pre_prune_state_count": 0,
                    "sum_post_prune_state_count": 0,
                    "processed_columns": 0,
                    "exception_message": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)
            writer.writerow(row)
            handle.flush()
            _json_write(
                progress_path,
                {
                    "task_id": int(task["task_id"]),
                    "scope": str(task["scope"]),
                    "completed": int(local_index),
                    "total": int(len(samples)),
                    "last_shot": int(sample.shot),
                    "elapsed_s": float(time.time() - started),
                    "partial_path": str(partial_path),
                },
            )
    return {
        "task_id": int(task["task_id"]),
        "scope": str(task["scope"]),
        "rows": rows,
        "shots_completed": int(len(rows)),
        "elapsed_s": float(time.time() - started),
    }


def _quantile(values: Sequence[float], q: float) -> float:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return float("nan")
    return float(np.quantile(np.asarray(finite, dtype=np.float64), float(q)))


def _fer_per_round(fer: float, rounds: int) -> float:
    if int(rounds) <= 0:
        return float("nan")
    value = min(max(float(fer), 0.0), 1.0)
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(1.0 - math.exp(math.log1p(-value) / float(rounds)))


def _summary_row(scope: str, rows: Sequence[Mapping[str, object]], *, sample_rows: Path) -> dict[str, object]:
    row_tuple = tuple(rows)
    trials = int(len(row_tuple))
    fail_total = sum(1 for row in row_tuple if str(row.get("frame_fail_type")) != "success")
    logical_fail = sum(1 for row in row_tuple if str(row.get("frame_fail_type")) == "logical_fail")
    syndrome_fail = sum(1 for row in row_tuple if str(row.get("frame_fail_type")) == "syndrome_fail")
    exception_fail = sum(1 for row in row_tuple if str(row.get("frame_fail_type")) == "exception_fail")
    syndrome_failure = sum(1 for row in row_tuple if str(row.get("failure_diagnosis")) == "syndrome_failure")
    truth_missing_terminal = sum(
        1 for row in row_tuple if str(row.get("failure_diagnosis")) == "truth_missing_terminal"
    )
    truth_present_but_not_selected = sum(
        1 for row in row_tuple if str(row.get("failure_diagnosis")) == "truth_present_but_not_selected"
    )
    first = row_tuple[0] if row_tuple else {}
    decode_values = [float(row.get("decode_s", float("nan"))) for row in row_tuple]
    transition_values = [float(row.get("transition_evals_total", float("nan"))) for row in row_tuple]
    engines = sorted({str(row.get("selected_engine", "")) for row in row_tuple if str(row.get("selected_engine", ""))})
    escalated = sum(1 for row in row_tuple if _boolish(row.get("escalated", False)))
    committee_disagreed = sum(1 for row in row_tuple if _boolish(row.get("committee_disagreed", False)))
    selected_forward = sum(1 for row in row_tuple if str(row.get("selected_direction", "")) == "forward")
    selected_backward = sum(1 for row in row_tuple if str(row.get("selected_direction", "")) == "backward")
    primary_transition_values = [
        float(row.get("primary_transition_evals_total", float("nan"))) for row in row_tuple
    ]
    escalation_transition_values = [
        float(row.get("escalation_transition_evals_total", float("nan"))) for row in row_tuple
    ]
    max_pre_values = [float(row.get("max_pre_prune_state_count", float("nan"))) for row in row_tuple]
    max_post_values = [float(row.get("max_post_prune_state_count", float("nan"))) for row in row_tuple]
    sum_pre_values = [float(row.get("sum_pre_prune_state_count", float("nan"))) for row in row_tuple]
    sum_post_values = [float(row.get("sum_post_prune_state_count", float("nan"))) for row in row_tuple]
    pressure_forward_values = [float(row.get("pressure_forward", float("nan"))) for row in row_tuple]
    pressure_backward_values = [float(row.get("pressure_backward", float("nan"))) for row in row_tuple]
    retained_values = [
        float(row.get("sum_post_prune_state_count", 0) or 0) / float(row.get("processed_columns", 0) or 1)
        for row in row_tuple
    ]
    fer = float(fail_total) / float(trials) if trials else float("nan")
    return {
        "decoder": str(first.get("decoder", "frontier auto")) if first else "frontier auto",
        "code": str(first.get("code", "")) if first else "",
        "scope": str(scope),
        "trials": int(trials),
        "fail_total": int(fail_total),
        "fer": float(fer),
        "fer_per_round": _fer_per_round(float(fer), int(first.get("noisy_rounds", 0) or 0)),
        "logical_fail": int(logical_fail),
        "syndrome_fail": int(syndrome_fail),
        "exception_fail": int(exception_fail),
        "syndrome_failure": int(syndrome_failure),
        "truth_missing_terminal": int(truth_missing_terminal),
        "truth_present_but_not_selected": int(truth_present_but_not_selected),
        "bad_ranking": int(truth_present_but_not_selected),
        "diagnosis_available": 1,
        "success": int(trials - fail_total),
        "decode_s_mean": float(np.mean(np.asarray(decode_values, dtype=np.float64))) if decode_values else float("nan"),
        "decode_s_p50": _quantile(decode_values, 0.50),
        "decode_s_p95": _quantile(decode_values, 0.95),
        "transition_evals_total_mean": (
            float(np.mean(np.asarray(transition_values, dtype=np.float64))) if transition_values else float("nan")
        ),
        "transition_evals_total_p50": _quantile(transition_values, 0.50),
        "transition_evals_total_p95": _quantile(transition_values, 0.95),
        "max_pre_prune_state_count_mean": (
            float(np.mean(np.asarray(max_pre_values, dtype=np.float64))) if max_pre_values else float("nan")
        ),
        "max_post_prune_state_count_mean": (
            float(np.mean(np.asarray(max_post_values, dtype=np.float64))) if max_post_values else float("nan")
        ),
        "sum_pre_prune_state_count_mean": (
            float(np.mean(np.asarray(sum_pre_values, dtype=np.float64))) if sum_pre_values else float("nan")
        ),
        "sum_post_prune_state_count_mean": (
            float(np.mean(np.asarray(sum_post_values, dtype=np.float64))) if sum_post_values else float("nan")
        ),
        "retained_states_mean": (
            float(np.mean(np.asarray(retained_values, dtype=np.float64))) if retained_values else float("nan")
        ),
        "max_pre_prune_state_count_max": max(
            (int(row.get("max_pre_prune_state_count", 0) or 0) for row in row_tuple),
            default=0,
        ),
        "max_post_prune_state_count_max": max(
            (int(row.get("max_post_prune_state_count", 0) or 0) for row in row_tuple),
            default=0,
        ),
        "matrix_rows": int(first.get("matrix_rows", 0) or 0),
        "matrix_cols": int(first.get("matrix_cols", 0) or 0),
        "logical_rows": int(first.get("logical_rows", 0) or 0),
        "noisy_rounds": int(first.get("noisy_rounds", 0) or 0),
        "backend": str(first.get("backend", "")),
        "p_location": float(first.get("p_location", float("nan"))) if first else float("nan"),
        "K": int(first.get("K", 0) or 0),
        "Delta": float(first.get("Delta", float("nan"))) if first else float("nan"),
        "escalation_K": int(first.get("escalation_K", 0) or 0),
        "escalation_Delta": float(first.get("escalation_Delta", float("nan"))) if first else float("nan"),
        "score_alpha": float(first.get("score_alpha", float("nan"))) if first else float("nan"),
        "metric_mode": str(first.get("metric_mode", "logsumexp_float")) if first else "logsumexp_float",
        "int_metric_scale": int(first.get("int_metric_scale", 1024) or 1024) if first else 1024,
        "decoder_mode": str(first.get("decoder_mode", "")),
        "direction_mode": str(first.get("direction_mode", "")),
        "pressure_estimator": str(first.get("pressure_estimator", "none")),
        "pressure_beta": float(first.get("pressure_beta", float("nan"))) if first else float("nan"),
        "pressure_gamma": float(first.get("pressure_gamma", float("nan"))) if first else float("nan"),
        "candidate_pressure_gate": str(first.get("candidate_pressure_gate", "")),
        "pressure_forward_mean": (
            float(np.nanmean(np.asarray(pressure_forward_values, dtype=np.float64)))
            if pressure_forward_values
            else float("nan")
        ),
        "pressure_backward_mean": (
            float(np.nanmean(np.asarray(pressure_backward_values, dtype=np.float64)))
            if pressure_backward_values
            else float("nan")
        ),
        "selected_forward": int(selected_forward),
        "selected_backward": int(selected_backward),
        "selected_forward_fraction": float(selected_forward) / float(trials) if trials else float("nan"),
        "selected_backward_fraction": float(selected_backward) / float(trials) if trials else float("nan"),
        "engine_requested": str(first.get("engine_requested", "")),
        "engines_seen": "|".join(engines),
        "escalated": int(escalated),
        "escalation_fraction": float(escalated) / float(trials) if trials else float("nan"),
        "committee_disagreed": int(committee_disagreed),
        "committee_disagreement_rate": float(committee_disagreed) / float(trials) if trials else float("nan"),
        "primary_transition_evals_total_mean": (
            float(np.mean(np.asarray(primary_transition_values, dtype=np.float64)))
            if primary_transition_values
            else float("nan")
        ),
        "escalation_transition_evals_total_mean": (
            float(np.mean(np.asarray(escalation_transition_values, dtype=np.float64)))
            if escalation_transition_values
            else float("nan")
        ),
        "sample_rows": str(Path(sample_rows).resolve()),
    }


def _combined_failure_diagnosis(x_row: Mapping[str, object], z_row: Mapping[str, object], *, fail_type: str) -> str:
    if str(fail_type) == "success":
        return "success"
    diagnoses = {str(x_row.get("failure_diagnosis", "")), str(z_row.get("failure_diagnosis", ""))}
    for candidate in (
        "exception_failure",
        "syndrome_failure",
        "truth_missing_terminal",
        "truth_present_but_not_selected",
        "undiagnosed_failure",
    ):
        if candidate in diagnoses:
            return str(candidate)
    if str(fail_type) == "syndrome_fail":
        return "syndrome_failure"
    if str(fail_type) == "logical_fail":
        return "undiagnosed_failure"
    return "exception_failure" if str(fail_type) == "exception_fail" else "undiagnosed_failure"


def _combined_rows(per_shot_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    by_scope: dict[str, dict[int, Mapping[str, object]]] = {}
    for row in per_shot_rows:
        by_scope.setdefault(str(row.get("scope")), {})[int(row.get("shot", 0))] = row
    if "memory_X" not in by_scope or "memory_Z" not in by_scope:
        return []
    common = sorted(set(by_scope["memory_X"]) & set(by_scope["memory_Z"]))
    out: list[dict[str, object]] = []
    for shot in common:
        x_row = by_scope["memory_X"][int(shot)]
        z_row = by_scope["memory_Z"][int(shot)]
        fail_types = {str(x_row.get("frame_fail_type")), str(z_row.get("frame_fail_type"))}
        if "exception_fail" in fail_types:
            fail_type = "exception_fail"
        elif "syndrome_fail" in fail_types:
            fail_type = "syndrome_fail"
        elif "logical_fail" in fail_types:
            fail_type = "logical_fail"
        else:
            fail_type = "success"
        failure_diagnosis = _combined_failure_diagnosis(x_row, z_row, fail_type=str(fail_type))
        escalated = _boolish(x_row.get("escalated", False)) or _boolish(z_row.get("escalated", False))
        committee_disagreed = _boolish(x_row.get("committee_disagreed", False)) or _boolish(
            z_row.get("committee_disagreed", False)
        )
        escalation_reasons = sorted(
            {
                str(x_row.get("escalation_reason", "")).strip(),
                str(z_row.get("escalation_reason", "")).strip(),
            }
            - {""}
        )
        out.append(
            {
                **dict(x_row),
                "scope": "combined",
                "frame_ok": bool(fail_type == "success"),
                "frame_fail_type": str(fail_type),
                "selected_K": max(
                    int(float(x_row.get("selected_K", x_row.get("K", 0)) or 0)),
                    int(float(z_row.get("selected_K", z_row.get("K", 0)) or 0)),
                ),
                "selected_Delta": max(
                    float(x_row.get("selected_Delta", x_row.get("Delta", 0.0)) or 0.0),
                    float(z_row.get("selected_Delta", z_row.get("Delta", 0.0)) or 0.0),
                ),
                "committee_disagreed": bool(committee_disagreed),
                "escalated": bool(escalated),
                "escalation_reason": "|".join(escalation_reasons),
                "truth_present_terminal": "|".join(
                    [str(x_row.get("truth_present_terminal", "")), str(z_row.get("truth_present_terminal", ""))]
                ),
                "truth_rank_terminal": "",
                "failure_diagnosis": str(failure_diagnosis),
                "decode_s": float(x_row.get("decode_s", 0.0) or 0.0) + float(z_row.get("decode_s", 0.0) or 0.0),
                "selected_direction": "|".join(
                    [str(x_row.get("selected_direction", "")), str(z_row.get("selected_direction", ""))]
                ),
                "selected_direction_matches_committee_direction": "",
                "pressure_forward": float(x_row.get("pressure_forward", 0.0) or 0.0)
                + float(z_row.get("pressure_forward", 0.0) or 0.0),
                "pressure_backward": float(x_row.get("pressure_backward", 0.0) or 0.0)
                + float(z_row.get("pressure_backward", 0.0) or 0.0),
                "transition_evals_total": int(float(x_row.get("transition_evals_total", 0) or 0))
                + int(float(z_row.get("transition_evals_total", 0) or 0)),
                "primary_transition_evals_total": int(
                    float(x_row.get("primary_transition_evals_total", 0) or 0)
                )
                + int(float(z_row.get("primary_transition_evals_total", 0) or 0)),
                "escalation_transition_evals_total": int(
                    float(x_row.get("escalation_transition_evals_total", 0) or 0)
                )
                + int(float(z_row.get("escalation_transition_evals_total", 0) or 0)),
                "max_pre_prune_state_count": max(
                    int(x_row.get("max_pre_prune_state_count", 0) or 0),
                    int(z_row.get("max_pre_prune_state_count", 0) or 0),
                ),
                "max_post_prune_state_count": max(
                    int(x_row.get("max_post_prune_state_count", 0) or 0),
                    int(z_row.get("max_post_prune_state_count", 0) or 0),
                ),
                "forward_transition_evals": int(float(x_row.get("forward_transition_evals", 0) or 0))
                + int(float(z_row.get("forward_transition_evals", 0) or 0)),
                "backward_transition_evals": int(float(x_row.get("backward_transition_evals", 0) or 0))
                + int(float(z_row.get("backward_transition_evals", 0) or 0)),
                "forward_max_post_prune_state_count": max(
                    int(float(x_row.get("forward_max_post_prune_state_count", 0) or 0)),
                    int(float(z_row.get("forward_max_post_prune_state_count", 0) or 0)),
                ),
                "backward_max_post_prune_state_count": max(
                    int(float(x_row.get("backward_max_post_prune_state_count", 0) or 0)),
                    int(float(z_row.get("backward_max_post_prune_state_count", 0) or 0)),
                ),
                "sum_pre_prune_state_count": int(float(x_row.get("sum_pre_prune_state_count", 0) or 0))
                + int(float(z_row.get("sum_pre_prune_state_count", 0) or 0)),
                "sum_post_prune_state_count": int(float(x_row.get("sum_post_prune_state_count", 0) or 0))
                + int(float(z_row.get("sum_post_prune_state_count", 0) or 0)),
                "processed_columns": int(float(x_row.get("processed_columns", 0) or 0))
                + int(float(z_row.get("processed_columns", 0) or 0)),
                "selected_engine": "|".join(
                    sorted({str(x_row.get("selected_engine", "")), str(z_row.get("selected_engine", ""))} - {""})
                ),
            }
        )
    return out


def _build_tasks(args: argparse.Namespace, by_scope: Mapping[str, Sequence[SampleRow]], out_dir: Path) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    task_id = 0
    for scope in _parse_scopes(str(args.scopes)):
        for shard_rows in _split_rows(tuple(by_scope[str(scope)]), int(args.shards_per_side)):
            task_rows = [
                {
                    "scope": row.scope,
                    "shot": int(row.shot),
                    "seed": int(row.seed),
                    "syndrome": int(row.syndrome),
                    "logical": int(row.logical),
                    "detector_weight": int(row.detector_weight),
                    "logical_weight": int(row.logical_weight),
                }
                for row in shard_rows
            ]
            tasks.append(
                {
                    "task_id": int(task_id),
                    "code": str(args.code),
                    "scope": str(scope),
                    "sample_rows": task_rows,
                    "partial_path": str(out_dir / "shards" / f"shard_{task_id:04d}_{scope}_per_shot.csv"),
                    "progress_path": str(out_dir / "shards" / f"shard_{task_id:04d}_{scope}_progress.json"),
                    "backend": str(args.backend),
                    "p_location": float(args.p_location),
                    "K": int(args.K),
                    "Delta": float(args.Delta),
                    "score_alpha": float(args.score_alpha),
                    "metric_mode": str(args.metric_mode),
                    "int_metric_scale": int(args.int_metric_scale),
                    "decoder_mode": str(args.decoder_mode),
                    "direction_mode": str(args.direction_mode),
                    "pressure_estimator": str(args.pressure_estimator),
                    "pressure_beta": float(args.pressure_beta),
                    "pressure_gamma": float(args.pressure_gamma),
                    "candidate_pressure_gate": str(args.candidate_pressure_gate),
                    "engine": str(args.engine),
                    "native_batch_size": int(args.native_batch_size),
                    "warmup_native": bool(args.warmup_native),
                    "escalate_on_committee_disagreement": bool(args.escalate_on_committee_disagreement),
                    "escalation_K": int(args.escalation_K),
                    "escalation_Delta": float(args.escalation_Delta),
                    "column_order": str(args.column_order),
                    "backward_column_order": str(args.backward_column_order),
                }
            )
            task_id += 1
    return tasks


def _write_report(summary_rows: Sequence[Mapping[str, object]], *, out_dir: Path) -> None:
    rows = [row for row in summary_rows if str(row.get("scope")) in {"memory_X", "memory_Z", "combined"}]
    if not rows:
        return
    first = rows[0]
    lines = [
        "# Frontier Sample Replay",
        "",
        (
            f"- Matrix: `{first['code']}` detector-side DEM, "
            f"`D_X=D_Z={int(first['matrix_rows'])}x{int(first['matrix_cols'])}`, "
            f"`O_X=O_Z={int(first['logical_rows'])}x{int(first['matrix_cols'])}`, "
            f"`{int(first['noisy_rounds'])}` noisy rounds."
        ),
        (
            f"- Decoder: frontier `_engine={first['engine_requested']}`, `K={int(first['K'])}`, "
            f"`Delta={float(first['Delta']):.6g}`, `score_alpha={float(first['score_alpha']):.6g}`, "
            f"`metric_mode={first.get('metric_mode', 'logsumexp_float')}`, "
            f"`int_metric_scale={int(first.get('int_metric_scale', 1024) or 1024)}`, "
            f"direction mode `{first.get('direction_mode', first['decoder_mode'])}`, "
            f"pressure `{first.get('pressure_estimator', 'none')}`."
        ),
        (
            f"- Adaptive escalation: `{int(first.get('escalated', 0))}` side rows escalated "
            f"(`{float(first.get('escalation_fraction', 0.0)):.6g}` fraction in this summary row); "
            f"configured target `K={int(first.get('escalation_K', 0) or 0)}`, "
            f"`Delta={float(first.get('escalation_Delta', float('nan'))):.6g}`."
        ),
        f"- Noise: `backend={first['backend']}`, `p_location={float(first['p_location']):.6g}`.",
        "- FER policy: strict full logical success; syndrome failures and exceptions count as FER.",
        "",
        "| scope | trials | fail_total | FER | FER/round | failures L/S/E | diagnosis syndrome/truth_missing/bad_ranking | escalated | fwd/bwd disagree | mean decode s | mean transitions | engines |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['scope']}` | `{int(row['trials'])}` | `{int(row['fail_total'])}` | "
            f"`{float(row['fer']):.6g}` | `{float(row['fer_per_round']):.6g}` | "
            f"`{int(row['logical_fail'])}/{int(row['syndrome_fail'])}/{int(row['exception_fail'])}` | "
            f"`{int(row['syndrome_failure'])}/{int(row['truth_missing_terminal'])}/{int(row['bad_ranking'])}` | "
            f"`{int(row.get('escalated', 0))}` | "
            f"`{float(row.get('committee_disagreement_rate', float('nan'))):.6g}` | "
            f"`{float(row['decode_s_mean']):.6g}` | `{float(row['transition_evals_total_mean']):.6g}` | "
            f"`{row['engines_seen']}` |"
        )
    lines.extend(["", f"Companion CSVs: `{out_dir / 'summary_by_scope.csv'}`, `{out_dir / 'per_shot_rows.csv'}`."])
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay frontier on matched DEM sample rows.")
    parser.add_argument("--sample-rows", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--code", type=str, required=True)
    parser.add_argument("--backend", type=str, required=True)
    parser.add_argument("--p-location", type=float, required=True)
    parser.add_argument("--shot-start", type=int, default=0)
    parser.add_argument("--shot-stop", type=int, required=True)
    parser.add_argument("--K", "--beam-cap", dest="K", type=int, required=True)
    parser.add_argument("--Delta", "--delta", dest="Delta", type=float, required=True)
    parser.add_argument("--score-alpha", type=float, default=0.8)
    parser.add_argument(
        "--metric-mode",
        choices=("logsumexp_float", "frontier_lite", "maxlog_int"),
        default="logsumexp_float",
    )
    parser.add_argument("--int-metric-scale", type=int, default=1024)
    parser.add_argument(
        "--decoder-mode",
        choices=("forward", "backward", "bidirectional_committee"),
        default="bidirectional_committee",
        help="Backward-compatible decode mode. Prefer --direction-mode for new experiments.",
    )
    parser.add_argument(
        "--direction-mode",
        choices=("fwd_bwd_committee", "forward_only", "backward_only"),
        default=None,
        help="Explicit direction policy. Defaults to the mode implied by --decoder-mode.",
    )
    parser.add_argument(
        "--pressure-estimator",
        choices=("none", "active_width", "candidate_gate_pressure"),
        default="none",
    )
    parser.add_argument("--pressure-beta", type=float, default=8.0)
    parser.add_argument("--pressure-gamma", type=float, default=2.0)
    parser.add_argument(
        "--candidate-pressure-gate",
        choices=("all_but_one", "overlap2"),
        default="all_but_one",
    )
    parser.add_argument("--engine", choices=("auto", "native_binary", "binary", "python"), default="auto")
    parser.add_argument("--scopes", type=str, default="memory_X,memory_Z")
    parser.add_argument("--column-order", type=str, default="deadline_reorder")
    parser.add_argument("--backward-column-order", type=str, default="backward_deadline_reorder")
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--shards-per-side", type=int, default=5)
    parser.add_argument("--native-batch-size", type=int, default=32)
    parser.add_argument("--warmup-native", action="store_true")
    parser.add_argument("--escalate-on-committee-disagreement", action="store_true")
    parser.add_argument("--escalation-K", type=int, default=0)
    parser.add_argument("--escalation-Delta", type=float, default=float("nan"))
    parser.add_argument("--progress-every-shards", type=int, default=1)
    parser.add_argument("--allow-existing", action="store_true")
    args = parser.parse_args(argv)
    args.direction_mode = _normalize_direction_mode(args.direction_mode, args.decoder_mode)
    args.decoder_mode = _decoder_mode_from_direction_mode(str(args.direction_mode))

    if int(args.shot_stop) < int(args.shot_start):
        raise ValueError("--shot-stop must be >= --shot-start")
    if int(args.int_metric_scale) <= 0:
        raise ValueError("--int-metric-scale must be positive")
    if str(args.metric_mode) in {"frontier_lite", "maxlog_int"} and str(args.engine) not in {"auto", "native_binary"}:
        raise ValueError("--metric-mode frontier_lite/maxlog_int requires --engine auto or native_binary")
    if not math.isfinite(float(args.pressure_beta)) or not math.isfinite(float(args.pressure_gamma)):
        raise ValueError("--pressure-beta and --pressure-gamma must be finite")
    if bool(args.escalate_on_committee_disagreement):
        if str(args.decoder_mode) != "bidirectional_committee":
            raise ValueError("--escalate-on-committee-disagreement requires --decoder-mode bidirectional_committee")
        if int(args.escalation_K) <= 0:
            args.escalation_K = int(args.K) * 4
        if not math.isfinite(float(args.escalation_Delta)):
            args.escalation_Delta = float(args.Delta) + 4.0
        if int(args.escalation_K) < int(args.K):
            raise ValueError("--escalation-K must be >= primary --K")
        if float(args.escalation_Delta) < float(args.Delta):
            raise ValueError("--escalation-Delta must be >= primary --Delta")
    else:
        if int(args.escalation_K) <= 0:
            args.escalation_K = 0
    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not bool(args.allow_existing):
        raise FileExistsError(f"output directory exists and is non-empty; pass --allow-existing: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    scopes = _parse_scopes(str(args.scopes))
    by_scope = _load_sample_rows(
        Path(args.sample_rows),
        scopes=scopes,
        shot_start=int(args.shot_start),
        shot_stop=int(args.shot_stop),
    )
    tasks = _build_tasks(args, by_scope, out_dir)
    metadata = {
        "status": "running",
        "code": str(args.code),
        "backend": str(args.backend),
        "p_location": float(args.p_location),
        "shot_start": int(args.shot_start),
        "shot_stop": int(args.shot_stop),
        "shots_per_side": int(args.shot_stop) - int(args.shot_start) + 1,
        "sample_rows": str(Path(args.sample_rows).expanduser().resolve()),
        "K": int(args.K),
        "Delta": float(args.Delta),
        "score_alpha": float(args.score_alpha),
        "metric_mode": str(args.metric_mode),
        "int_metric_scale": int(args.int_metric_scale),
        "decoder_mode": str(args.decoder_mode),
        "direction_mode": str(args.direction_mode),
        "pressure_estimator": str(args.pressure_estimator),
        "pressure_beta": float(args.pressure_beta),
        "pressure_gamma": float(args.pressure_gamma),
        "candidate_pressure_gate": str(args.candidate_pressure_gate),
        "engine": str(args.engine),
        "frontier_native_available": bool(frontier.native_binary_available()),
        "scopes": list(scopes),
        "cpus": int(args.cpus),
        "shards_per_side": int(args.shards_per_side),
        "native_batch_size": int(args.native_batch_size),
        "warmup_native": bool(args.warmup_native),
        "escalate_on_committee_disagreement": bool(args.escalate_on_committee_disagreement),
        "escalation_K": int(args.escalation_K),
        "escalation_Delta": float(args.escalation_Delta),
        "tasks": len(tasks),
        "result_root": str(out_dir),
    }
    _json_write(out_dir / "run_metadata.json", metadata)
    print(
        f"[setup] frontier replay code={args.code} backend={args.backend} p={float(args.p_location):.6g} "
        f"K={int(args.K)} Delta={float(args.Delta):.6g} direction_mode={args.direction_mode} "
        f"decoder_mode={args.decoder_mode} engine={args.engine} "
        f"metric_mode={args.metric_mode} int_metric_scale={int(args.int_metric_scale)} "
        f"pressure={args.pressure_estimator} beta={float(args.pressure_beta):.6g} "
        f"gamma={float(args.pressure_gamma):.6g} gate={args.candidate_pressure_gate} "
        f"shots_per_side={metadata['shots_per_side']} cpus={int(args.cpus)} tasks={len(tasks)}",
        flush=True,
    )
    if bool(args.escalate_on_committee_disagreement):
        print(
            f"[setup] disagreement escalation enabled: primary K={int(args.K)} Delta={float(args.Delta):.6g}; "
            f"committee disagreement -> K={int(args.escalation_K)} Delta={float(args.escalation_Delta):.6g}",
            flush=True,
        )
    print(
        f"[progress] shard partial CSVs and progress JSONs flush after native batches "
        f"of at most {int(args.native_batch_size)} shots",
        flush=True,
    )

    started = time.time()
    all_rows: list[dict[str, object]] = []
    if int(args.cpus) == 1 or len(tasks) <= 1:
        for index, task in enumerate(tasks, start=1):
            result = _run_shard(task)
            all_rows.extend(list(result["rows"]))
            elapsed = time.time() - started
            eta = elapsed / float(index) * float(len(tasks) - index)
            print(
                f"[progress] shard {index}/{len(tasks)} scope={result['scope']} "
                f"side_rows_done={len(all_rows)}/{len(scopes) * metadata['shots_per_side']} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
    else:
        ctx = mp.get_context("fork")
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(args.cpus), mp_context=ctx) as executor:
            future_to_task = {executor.submit(_run_shard, task): task for task in tasks}
            completed = 0
            for future in concurrent.futures.as_completed(future_to_task):
                result = future.result()
                completed += 1
                all_rows.extend(list(result["rows"]))
                if int(args.progress_every_shards) > 0 and (
                    completed == 1 or completed % int(args.progress_every_shards) == 0 or completed == len(tasks)
                ):
                    elapsed = time.time() - started
                    eta = elapsed / float(completed) * float(len(tasks) - completed)
                    print(
                        f"[progress] shard {completed}/{len(tasks)} scope={result['scope']} "
                        f"side_rows_done={len(all_rows)}/{len(scopes) * metadata['shots_per_side']} "
                        f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                        flush=True,
                    )

    all_rows.sort(key=lambda row: (str(row.get("scope")), int(row.get("shot", 0))))
    combined = _combined_rows(all_rows)
    _csv_write(out_dir / "per_shot_rows.csv", all_rows, PER_SHOT_FIELDS)
    if combined:
        _csv_write(out_dir / "combined_per_shot_rows.csv", combined, PER_SHOT_FIELDS)
    summary_rows = [
        _summary_row(scope, [row for row in all_rows if str(row.get("scope")) == str(scope)], sample_rows=Path(args.sample_rows))
        for scope in scopes
    ]
    if combined:
        summary_rows.append(_summary_row("combined", combined, sample_rows=Path(args.sample_rows)))
    _csv_write(out_dir / "summary_by_scope.csv", summary_rows, SUMMARY_FIELDS)
    _write_report(summary_rows, out_dir=out_dir)

    metadata["status"] = "complete"
    metadata["elapsed_s"] = float(time.time() - started)
    metadata["summary_by_scope_csv"] = str(out_dir / "summary_by_scope.csv")
    metadata["per_shot_rows_csv"] = str(out_dir / "per_shot_rows.csv")
    metadata["combined_per_shot_rows_csv"] = str(out_dir / "combined_per_shot_rows.csv") if combined else ""
    metadata["report_md"] = str(out_dir / "report.md")
    _json_write(out_dir / "run_metadata.json", metadata)
    print(f"[done] elapsed_s={float(time.time() - started):.1f} report={out_dir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
