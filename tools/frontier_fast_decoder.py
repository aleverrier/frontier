from __future__ import annotations

import argparse
import heapq
import math
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import steane_progressive_decoder as progressive

try:
    import frontier_fast_native as _frontier_fast_native
except Exception:  # pragma: no cover - optional native extension may be absent.
    _frontier_fast_native = None

_SCORE_ALPHA = 0.8
_NATIVE_MAX_DETECTOR_LIMBS = 64
_NATIVE_MAX_LOGICAL_LIMBS = 8
_BINARY_COMPATIBILITY_CACHE: dict[tuple[tuple[int, ...], int, int], bool] = {}
_NATIVE_COMPATIBILITY_CACHE: dict[tuple[tuple[int, ...], int, int, int, int, int, int], bool] = {}
_NATIVE_MODEL_CACHE: dict[tuple[tuple[int, ...], int, int, int, int, int, int], object] = {}
_NATIVE_CHOICE_COMPATIBILITY_CACHE: dict[tuple[tuple[int, ...], int, int, int, int, int, int], bool] = {}
_NATIVE_CHOICE_MODEL_CACHE: dict[tuple[tuple[int, ...], int, int, int, int, int, int], object] = {}


def _normalize_metric_mode(metric_mode: str) -> str:
    mode = str(metric_mode).strip().lower()
    if mode in {"logsumexp_float", "float", "exact"}:
        return "logsumexp_float"
    if mode in {"frontierlite", "frontier_lite", "frontier-lite", "maxlog_int", "max_log_int", "viterbi_int"}:
        return "frontier_lite"
    raise ValueError("metric_mode must be 'logsumexp_float', 'frontier_lite', or 'maxlog_int'")


def _validate_int_metric_scale(int_metric_scale: int) -> int:
    scale = int(int_metric_scale)
    if scale <= 0:
        raise ValueError("int_metric_scale must be positive")
    return int(scale)


def _score_mode_for_alpha(score_alpha: float) -> str:
    alpha = float(score_alpha)
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("score_alpha must be finite and non-negative")
    token = f"{alpha:.12g}".replace("-", "m").replace(".", "p")
    return f"future_parity_logodds_a{token}"


@dataclass(frozen=True, slots=True)
class FrontierFastModel:
    columns: tuple[progressive.ProgressiveColumn, ...]
    layout: progressive.ProgressiveFrontierLayout
    num_detectors: int
    num_observables: int = 1
    backward_columns: tuple[progressive.ProgressiveColumn, ...] | None = None
    backward_layout: progressive.ProgressiveFrontierLayout | None = None


@dataclass(frozen=True, slots=True)
class FrontierFastStats:
    processed_columns: int
    transition_evals: int
    max_pre_prune_state_count: int
    max_post_prune_state_count: int
    sum_pre_prune_state_count: int
    sum_post_prune_state_count: int
    no_path_count: int
    transition_time_s: float = 0.0
    merge_time_s: float = 0.0
    prune_time_s: float = 0.0
    total_time_s: float = 0.0


@dataclass(frozen=True, slots=True)
class FrontierFastCommitteeMember:
    direction: str
    status: str
    logical_hat: int | None
    log_evidence: float
    terminal_top_log_mass_gap: float
    top1_posterior: float


@dataclass(frozen=True, slots=True)
class FrontierFastResult:
    status: str
    logical_hat: int | None
    log_evidence: float
    terminal_log_masses: dict[int, float]
    stats: FrontierFastStats
    direction: str | None = None
    engine: str = ""
    terminal_top_log_mass_gap: float = float("nan")
    committee_members: tuple[FrontierFastCommitteeMember, ...] = tuple()


def _logaddexp_pair(a: float, b: float) -> float:
    if not math.isfinite(float(a)):
        return float(b)
    if not math.isfinite(float(b)):
        return float(a)
    hi = max(float(a), float(b))
    lo = min(float(a), float(b))
    return float(hi + math.log1p(math.exp(float(lo) - float(hi))))


def _logaddexp_many(values: Iterable[float]) -> float:
    total = float("-inf")
    for value in values:
        total = _logaddexp_pair(total, float(value))
    return float(total)


def _syndrome_to_int(syndrome: int | np.ndarray | Sequence[int]) -> int:
    if isinstance(syndrome, (int, np.integer)):
        return int(syndrome)
    arr = np.asarray(syndrome, dtype=np.uint8).reshape(-1)
    out = 0
    for index, value in enumerate(arr.tolist()):
        if int(value) & 1:
            out |= 1 << int(index)
    return int(out)


def _infer_num_detectors(
    columns: Sequence[progressive.ProgressiveColumn],
    *,
    syndrome_int: int,
) -> int:
    width = max(1, int(syndrome_int).bit_length())
    for column in tuple(columns):
        width = max(width, int(column.detector_support_mask).bit_length())
        for mask in tuple(column.detector_response_masks):
            width = max(width, int(mask).bit_length())
    return int(width)


def _infer_num_observables(columns: Sequence[progressive.ProgressiveColumn]) -> int:
    width = 1
    for column in tuple(columns):
        for mask in tuple(column.logical_response_masks):
            width = max(width, int(mask).bit_length())
    return int(width)


def _looks_like_factor_transition(value: object) -> bool:
    return isinstance(value, progressive.FactorTransition)


def _coerce_columns(problem_or_model: object) -> tuple[progressive.ProgressiveColumn, ...]:
    if isinstance(problem_or_model, FrontierFastModel):
        return tuple(problem_or_model.columns)
    model = getattr(problem_or_model, "family", problem_or_model)
    if hasattr(model, "factor_transitions"):
        return tuple(progressive._columns_from_factor_transitions(tuple(getattr(model, "factor_transitions"))))
    if hasattr(model, "transitions") and not hasattr(model, "columns"):
        maybe_transitions = tuple(getattr(model, "transitions"))
        if maybe_transitions and all(_looks_like_factor_transition(item) for item in maybe_transitions):
            return tuple(progressive._columns_from_factor_transitions(maybe_transitions))
    if hasattr(model, "columns"):
        return tuple(getattr(model, "columns"))
    if isinstance(problem_or_model, Sequence) and not isinstance(problem_or_model, (str, bytes)):
        items = tuple(problem_or_model)
        if items and all(_looks_like_factor_transition(item) for item in items):
            return tuple(progressive._columns_from_factor_transitions(items))
        return tuple(items)
    raise TypeError(
        "decode_frontier_fast expects a FrontierFastModel, an object with .columns, "
        "or a sequence of ProgressiveColumn/FactorTransition objects"
    )


def _coerce_model(
    problem_or_model: object,
    *,
    syndrome_int: int,
    direction: str = "forward",
) -> FrontierFastModel:
    if isinstance(problem_or_model, FrontierFastModel) and str(direction) == "forward":
        return problem_or_model
    if isinstance(problem_or_model, FrontierFastModel) and str(direction) == "backward":
        backward_columns = (
            tuple(problem_or_model.backward_columns)
            if problem_or_model.backward_columns is not None
            else tuple(progressive._reverse_progressive_columns(problem_or_model.columns))
        )
        backward_layout = (
            problem_or_model.backward_layout
            if problem_or_model.backward_layout is not None
            else progressive.build_frontier_layout(
                list(backward_columns),
                num_detectors=int(problem_or_model.num_detectors),
            )
        )
        return FrontierFastModel(
            columns=tuple(backward_columns),
            layout=backward_layout,
            num_detectors=int(problem_or_model.num_detectors),
            num_observables=int(problem_or_model.num_observables),
        )

    model = getattr(problem_or_model, "family", problem_or_model)
    columns = _coerce_columns(problem_or_model)
    num_detectors = int(
        getattr(
            model,
            "num_detectors",
            getattr(model, "matrix_rows", _infer_num_detectors(columns, syndrome_int=int(syndrome_int))),
        )
    )
    num_observables = int(
        getattr(model, "num_observables", getattr(model, "logical_rows", _infer_num_observables(columns)))
    )
    if str(direction) == "backward":
        backward_columns = getattr(model, "backward_columns", None)
        columns = (
            tuple(backward_columns)
            if backward_columns is not None
            else tuple(progressive._reverse_progressive_columns(columns))
        )
        layout = getattr(model, "backward_layout", None)
    else:
        layout = getattr(model, "layout", None)
    if layout is None:
        layout = progressive.build_frontier_layout(list(columns), num_detectors=int(num_detectors))
    return FrontierFastModel(
        columns=tuple(columns),
        layout=layout,
        num_detectors=int(num_detectors),
        num_observables=int(num_observables),
    )


def _select_keys_by_score_gap(
    *,
    score_by_key: Mapping[int, float],
    log_mass_by_key: Mapping[int, float],
    K: int,
    Delta: float,
) -> tuple[int, ...]:
    if int(K) <= 0:
        raise ValueError("K must be positive")
    if float(Delta) < 0.0:
        raise ValueError("Delta must be non-negative")
    if not score_by_key:
        return tuple()
    best_score = max(float(value) for value in score_by_key.values())
    cutoff = float(best_score) - float(Delta)
    survivors = [
        int(key)
        for key, score in score_by_key.items()
        if float(score) >= float(cutoff)
    ]
    rank = lambda key: (
        float(score_by_key[int(key)]),
        float(log_mass_by_key[int(key)]),
        -int(key),
    )
    if len(survivors) > int(K):
        survivors = heapq.nlargest(int(K), survivors, key=rank)
    return tuple(sorted((int(key) for key in survivors), key=rank, reverse=True))


def _frontier_fast_top_posteriors(result: FrontierFastResult) -> tuple[float, float]:
    if str(result.status) != "ok" or not math.isfinite(float(result.log_evidence)):
        return float("nan"), float("nan")
    posteriors = sorted(
        (
            math.exp(float(log_mass) - float(result.log_evidence))
            for log_mass in result.terminal_log_masses.values()
        ),
        reverse=True,
    )
    if not posteriors:
        return float("nan"), float("nan")
    top1 = float(posteriors[0])
    top2 = float(posteriors[1]) if len(posteriors) >= 2 else 0.0
    return float(top1), float(top2)


def _terminal_gap(terminal_log_masses: Mapping[int, float]) -> float:
    ranked = sorted((float(value) for value in terminal_log_masses.values()), reverse=True)
    if not ranked:
        return float("nan")
    if len(ranked) == 1:
        return float("inf")
    return float(ranked[0] - ranked[1])


def _empty_stats(*, started: float, processed_columns: int = 0, no_path_count: int = 1) -> FrontierFastStats:
    return FrontierFastStats(
        processed_columns=int(processed_columns),
        transition_evals=0,
        max_pre_prune_state_count=0,
        max_post_prune_state_count=0,
        sum_pre_prune_state_count=0,
        sum_post_prune_state_count=0,
        no_path_count=int(no_path_count),
        total_time_s=float(time.perf_counter() - float(started)),
    )


def _is_binary_fastpath_compatible(
    problem_or_model: object,
    *,
    syndrome: int | np.ndarray | Sequence[int] = 0,
    direction: str = "forward",
) -> bool:
    """Return True only for columns supported by the existing binary payload path."""
    try:
        syndrome_int = _syndrome_to_int(syndrome)
        model = _coerce_model(problem_or_model, syndrome_int=int(syndrome_int), direction=str(direction))
    except Exception:
        return False
    columns = tuple(model.columns)
    if not columns:
        return False
    if len(tuple(model.layout.column_row_updates)) != len(columns):
        return False
    cache_key = (
        tuple(int(id(column)) for column in columns),
        int(id(model.layout)),
        int(len(columns)),
    )
    cached = _BINARY_COMPATIBILITY_CACHE.get(cache_key)
    if cached is not None:
        return bool(cached)
    compatible = True
    for column, row_updates in zip(columns, tuple(model.layout.column_row_updates), strict=True):
        try:
            payload = progressive._compile_binary_frontierk_column_payload(column, tuple(row_updates))
        except Exception:
            compatible = False
            break
        if payload is None:
            compatible = False
            break
    if len(_BINARY_COMPATIBILITY_CACHE) >= 64:
        _BINARY_COMPATIBILITY_CACHE.clear()
    _BINARY_COMPATIBILITY_CACHE[cache_key] = bool(compatible)
    return bool(compatible)


def native_binary_available() -> bool:
    return _frontier_fast_native is not None and bool(_frontier_fast_native.is_available())


def native_choice_available() -> bool:
    native_module = getattr(_frontier_fast_native, "_native", None)
    return (
        native_binary_available()
        and hasattr(_frontier_fast_native, "NativeChoiceFrontierModel")
        and hasattr(native_module, "make_choice_model")
        and hasattr(native_module, "decode_choice")
    )


def _detector_limb_count(num_detectors: int) -> int:
    return max(1, (int(num_detectors) + 63) // 64)


def _mask_to_limbs(mask: int, n_limbs: int) -> tuple[int, ...]:
    value = int(mask)
    return tuple(int((value >> (64 * limb)) & ((1 << 64) - 1)) for limb in range(int(n_limbs)))


def _single_bit_row_terms(
    masks: Sequence[int],
    weights: Sequence[float],
    *,
    n_limbs: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[float, ...]] | None:
    if len(tuple(masks)) != len(tuple(weights)):
        return None
    limbs: list[int] = []
    bits: list[int] = []
    values: list[float] = []
    for mask, weight in zip(tuple(masks), tuple(weights), strict=True):
        mask_int = int(mask)
        if int(mask_int) <= 0 or int(mask_int).bit_count() != 1:
            return None
        bit_index = int(mask_int.bit_length() - 1)
        limb = int(bit_index // 64)
        if int(limb) < 0 or int(limb) >= int(n_limbs):
            return None
        limbs.append(int(limb))
        bits.append(int(1 << (bit_index % 64)))
        values.append(float(weight))
    return tuple(limbs), tuple(bits), tuple(values)


def _choice_row_terms(
    row_updates: Sequence[progressive.FrontierRowUpdate],
    *,
    active_field: str,
    parity_field: str,
    n_limbs: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[float, ...]] | None:
    masks: list[int] = []
    weights: list[float] = []
    for update in tuple(row_updates):
        if bool(getattr(update, active_field)):
            masks.append(int(update.row_mask))
            weights.append(float(getattr(update, parity_field)))
    return _single_bit_row_terms(tuple(masks), tuple(weights), n_limbs=int(n_limbs))


def _native_collect_phase_timing_enabled() -> bool:
    return str(os.environ.get("FRONTIERFAST_NATIVE_PHASE_TIMING", "0")).strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }


def _native_force_full_key_enabled() -> bool:
    return str(os.environ.get("FRONTIERFAST_NATIVE_FORCE_FULL_KEY", "0")).strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }


def _native_small_pattern_table_disabled() -> bool:
    return str(os.environ.get("FRONTIERFAST_NATIVE_DISABLE_SMALL_PATTERN_TABLE", "0")).strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }


def _native_cache_key(model: FrontierFastModel) -> tuple[tuple[int, ...], int, int, int, int, int, int]:
    return (
        tuple(int(id(column)) for column in tuple(model.columns)),
        int(id(model.layout)),
        int(model.num_detectors),
        int(model.num_observables),
        int(_native_collect_phase_timing_enabled()),
        int(_native_force_full_key_enabled()),
        int(_native_small_pattern_table_disabled()),
    )


def _native_binary_model_spec(model: FrontierFastModel) -> dict[str, object] | None:
    n_limbs = _detector_limb_count(int(model.num_detectors))
    n_logical_limbs = _detector_limb_count(int(model.num_observables))
    if int(n_limbs) > int(_NATIVE_MAX_DETECTOR_LIMBS):
        return None
    if int(model.num_observables) <= 0 or int(n_logical_limbs) > int(_NATIVE_MAX_LOGICAL_LIMBS):
        return None
    columns_spec: list[dict[str, object]] = []
    row_updates_by_column = tuple(model.layout.column_row_updates)
    if len(row_updates_by_column) != len(tuple(model.columns)):
        return None
    for column_index, (column, row_updates) in enumerate(
        zip(tuple(model.columns), row_updates_by_column, strict=True)
    ):
        payload = progressive._compile_binary_frontierk_column_payload(column, tuple(row_updates))
        if payload is None:
            return None
        before_terms = _single_bit_row_terms(
            tuple(int(value) for value in tuple(payload.before_row_bit_masks)),
            tuple(float(value) for value in tuple(payload.before_parity_logodds_weights)),
            n_limbs=int(n_limbs),
        )
        after_terms = _single_bit_row_terms(
            tuple(int(value) for value in tuple(payload.after_row_bit_masks)),
            tuple(float(value) for value in tuple(payload.after_parity_logodds_weights)),
            n_limbs=int(n_limbs),
        )
        if before_terms is None or after_terms is None:
            return None
        before_limbs, before_bits, before_parity = before_terms
        after_limbs, after_bits, after_parity = after_terms
        columns_spec.append(
            {
                "no_error_log_const": float(payload.no_error_log_const),
                "toggle_logodds": float(payload.toggle_logodds),
                "toggle_detector_limbs": _mask_to_limbs(int(payload.toggle_detector_mask), int(n_limbs)),
                "toggle_logical_limbs": _mask_to_limbs(int(payload.toggle_observable_mask), int(n_logical_limbs)),
                "close_limbs": _mask_to_limbs(
                    int(model.layout.closing_masks[int(column_index)]),
                    int(n_limbs),
                ),
                "active_limbs": _mask_to_limbs(
                    int(model.layout.active_masks_after_column[int(column_index)]),
                    int(n_limbs),
                ),
                "before_limbs": before_limbs,
                "before_bits": before_bits,
                "before_parity": before_parity,
                "after_limbs": after_limbs,
                "after_bits": after_bits,
                "after_parity": after_parity,
            }
        )
    return {
        "num_detectors": int(model.num_detectors),
        "num_observables": int(model.num_observables),
        "n_limbs": int(n_limbs),
        "n_logical_limbs": int(n_logical_limbs),
        "collect_phase_timing": bool(_native_collect_phase_timing_enabled()),
        "force_full_key": bool(_native_force_full_key_enabled()),
        "columns": columns_spec,
    }


def _native_choice_model_spec(model: FrontierFastModel) -> dict[str, object] | None:
    n_limbs = _detector_limb_count(int(model.num_detectors))
    n_logical_limbs = _detector_limb_count(int(model.num_observables))
    if int(n_limbs) > int(_NATIVE_MAX_DETECTOR_LIMBS):
        return None
    if int(model.num_observables) <= 0 or int(n_logical_limbs) > int(_NATIVE_MAX_LOGICAL_LIMBS):
        return None
    row_updates_by_column = tuple(model.layout.column_row_updates)
    if len(row_updates_by_column) != len(tuple(model.columns)):
        return None
    columns_spec: list[dict[str, object]] = []
    for column_index, (column, row_updates) in enumerate(
        zip(tuple(model.columns), row_updates_by_column, strict=True)
    ):
        log_priors = tuple(float(value) for value in progressive._column_log_priors(column))
        detector_masks = tuple(int(value) for value in tuple(column.detector_response_masks))
        logical_masks = tuple(int(value) for value in tuple(column.logical_response_masks))
        if not log_priors or len(log_priors) != len(detector_masks) or len(log_priors) != len(logical_masks):
            return None
        before_terms = _choice_row_terms(
            tuple(row_updates),
            active_field="active_before",
            parity_field="parity_logodds_before",
            n_limbs=int(n_limbs),
        )
        after_terms = _choice_row_terms(
            tuple(row_updates),
            active_field="active_after",
            parity_field="parity_logodds_after",
            n_limbs=int(n_limbs),
        )
        if before_terms is None or after_terms is None:
            return None
        before_limbs, before_bits, before_parity = before_terms
        after_limbs, after_bits, after_parity = after_terms
        columns_spec.append(
            {
                "log_priors": tuple(float(value) for value in log_priors),
                "detector_limbs": tuple(_mask_to_limbs(int(mask), int(n_limbs)) for mask in detector_masks),
                "logical_masks": tuple(
                    _mask_to_limbs(int(mask), int(n_logical_limbs)) for mask in logical_masks
                ),
                "close_limbs": _mask_to_limbs(
                    int(model.layout.closing_masks[int(column_index)]),
                    int(n_limbs),
                ),
                "active_limbs": _mask_to_limbs(
                    int(model.layout.active_masks_after_column[int(column_index)]),
                    int(n_limbs),
                ),
                "before_limbs": before_limbs,
                "before_bits": before_bits,
                "before_parity": before_parity,
                "after_limbs": after_limbs,
                "after_bits": after_bits,
                "after_parity": after_parity,
            }
        )
    return {
        "num_detectors": int(model.num_detectors),
        "num_observables": int(model.num_observables),
        "n_limbs": int(n_limbs),
        "n_logical_limbs": int(n_logical_limbs),
        "collect_phase_timing": bool(_native_collect_phase_timing_enabled()),
        "columns": columns_spec,
    }


def _is_native_binary_compatible(
    problem_or_model: object,
    *,
    syndrome: int | np.ndarray | Sequence[int] = 0,
    direction: str = "forward",
) -> bool:
    if not native_binary_available():
        return False
    try:
        syndrome_int = _syndrome_to_int(syndrome)
        model = _coerce_model(problem_or_model, syndrome_int=int(syndrome_int), direction=str(direction))
    except Exception:
        return False
    key = _native_cache_key(model)
    cached = _NATIVE_COMPATIBILITY_CACHE.get(key)
    if cached is not None:
        return bool(cached)
    compatible = bool(_is_binary_fastpath_compatible(model, syndrome=int(syndrome_int)))
    if bool(compatible):
        try:
            compatible = _native_binary_model_spec(model) is not None
        except Exception:
            compatible = False
    if len(_NATIVE_COMPATIBILITY_CACHE) >= 64:
        _NATIVE_COMPATIBILITY_CACHE.clear()
    _NATIVE_COMPATIBILITY_CACHE[key] = bool(compatible)
    return bool(compatible)


def _is_native_choice_compatible(
    problem_or_model: object,
    *,
    syndrome: int | np.ndarray | Sequence[int] = 0,
    direction: str = "forward",
) -> bool:
    if not native_choice_available():
        return False
    try:
        syndrome_int = _syndrome_to_int(syndrome)
        model = _coerce_model(problem_or_model, syndrome_int=int(syndrome_int), direction=str(direction))
    except Exception:
        return False
    key = _native_cache_key(model)
    cached = _NATIVE_CHOICE_COMPATIBILITY_CACHE.get(key)
    if cached is not None:
        return bool(cached)
    try:
        compatible = _native_choice_model_spec(model) is not None
    except Exception:
        compatible = False
    if len(_NATIVE_CHOICE_COMPATIBILITY_CACHE) >= 64:
        _NATIVE_CHOICE_COMPATIBILITY_CACHE.clear()
    _NATIVE_CHOICE_COMPATIBILITY_CACHE[key] = bool(compatible)
    return bool(compatible)


def _get_native_binary_model(model: FrontierFastModel) -> object:
    if not native_binary_available():
        raise RuntimeError("native FrontierFast binary extension is not built")
    key = _native_cache_key(model)
    cached = _NATIVE_MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    spec = _native_binary_model_spec(model)
    if spec is None:
        raise ValueError("native FrontierFast binary engine requested for an unsupported model")
    native_model = _frontier_fast_native.NativeBinaryFrontierModel(spec)
    if len(_NATIVE_MODEL_CACHE) >= 32:
        _NATIVE_MODEL_CACHE.clear()
    _NATIVE_MODEL_CACHE[key] = native_model
    return native_model


def _get_native_choice_model(model: FrontierFastModel) -> object:
    if not native_choice_available():
        raise RuntimeError("native FrontierFast choice extension is not built")
    key = _native_cache_key(model)
    cached = _NATIVE_CHOICE_MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    spec = _native_choice_model_spec(model)
    if spec is None:
        raise ValueError("native FrontierFast choice engine requested for an unsupported model")
    native_model = _frontier_fast_native.NativeChoiceFrontierModel(spec)
    if len(_NATIVE_CHOICE_MODEL_CACHE) >= 32:
        _NATIVE_CHOICE_MODEL_CACHE.clear()
    _NATIVE_CHOICE_MODEL_CACHE[key] = native_model
    return native_model


def _frontier_fast_stats_from_native(payload: Mapping[str, object]) -> FrontierFastStats:
    return FrontierFastStats(
        processed_columns=int(payload.get("processed_columns", 0)),
        transition_evals=int(payload.get("transition_evals", 0)),
        max_pre_prune_state_count=int(payload.get("max_pre_prune_state_count", 0)),
        max_post_prune_state_count=int(payload.get("max_post_prune_state_count", 0)),
        sum_pre_prune_state_count=int(payload.get("sum_pre_prune_state_count", 0)),
        sum_post_prune_state_count=int(payload.get("sum_post_prune_state_count", 0)),
        no_path_count=int(payload.get("no_path_count", 0)),
        transition_time_s=float(payload.get("transition_time_s", 0.0)),
        merge_time_s=float(payload.get("merge_time_s", 0.0)),
        prune_time_s=float(payload.get("prune_time_s", 0.0)),
        total_time_s=float(payload.get("total_time_s", 0.0)),
    )


def _frontier_fast_result_from_native_payload(
    payload: Mapping[str, object],
    *,
    direction: str | None = None,
    engine: str = "native_binary",
) -> FrontierFastResult:
    status = str(payload.get("status", "no_path"))
    terminal_log_masses = {
        int(logical): float(log_mass)
        for logical, log_mass in dict(payload.get("terminal_log_masses", {})).items()
    }
    return FrontierFastResult(
        status=str(status),
        logical_hat=(
            None
            if payload.get("logical_hat") is None or str(status) != "ok"
            else int(payload.get("logical_hat"))
        ),
        log_evidence=float(payload.get("log_evidence", float("-inf"))),
        terminal_log_masses=dict(sorted(terminal_log_masses.items())),
        stats=_frontier_fast_stats_from_native(dict(payload.get("stats", {}))),
        direction=direction,
        engine=str(engine),
        terminal_top_log_mass_gap=float(payload.get("terminal_top_log_mass_gap", float("nan"))),
    )


def _decode_frontier_fast_native_binary(
    problem_or_model: object,
    syndrome: int | np.ndarray | Sequence[int],
    *,
    K: int,
    Delta: float,
    score_alpha: float = _SCORE_ALPHA,
    metric_mode: str = "logsumexp_float",
    int_metric_scale: int = 1024,
    direction: str | None = None,
    _assume_compatible: bool = False,
) -> FrontierFastResult:
    if int(K) <= 0:
        raise ValueError("K must be positive")
    if float(Delta) < 0.0:
        raise ValueError("Delta must be non-negative")
    if not math.isfinite(float(score_alpha)) or float(score_alpha) < 0.0:
        raise ValueError("score_alpha must be finite and non-negative")
    metric_mode = _normalize_metric_mode(metric_mode)
    int_metric_scale = _validate_int_metric_scale(int_metric_scale)
    syndrome_int = _syndrome_to_int(syndrome)
    model = _coerce_model(problem_or_model, syndrome_int=int(syndrome_int))
    if not bool(_assume_compatible) and not _is_native_binary_compatible(model, syndrome=int(syndrome_int)):
        raise ValueError("native FrontierFast binary engine requested for an unsupported model")
    native_model = _get_native_binary_model(model)
    n_limbs = _detector_limb_count(int(model.num_detectors))
    payload = native_model.decode(
        _mask_to_limbs(int(syndrome_int), int(n_limbs)),
        int(K),
        float(Delta),
        float(score_alpha),
        str(metric_mode),
        int(int_metric_scale),
    )
    return _frontier_fast_result_from_native_payload(payload, direction=direction)


def _decode_frontier_fast_native_choice(
    problem_or_model: object,
    syndrome: int | np.ndarray | Sequence[int],
    *,
    K: int,
    Delta: float,
    score_alpha: float = _SCORE_ALPHA,
    direction: str | None = None,
    _assume_compatible: bool = False,
) -> FrontierFastResult:
    if int(K) <= 0:
        raise ValueError("K must be positive")
    if float(Delta) < 0.0:
        raise ValueError("Delta must be non-negative")
    if not math.isfinite(float(score_alpha)) or float(score_alpha) < 0.0:
        raise ValueError("score_alpha must be finite and non-negative")
    syndrome_int = _syndrome_to_int(syndrome)
    model = _coerce_model(problem_or_model, syndrome_int=int(syndrome_int))
    if not bool(_assume_compatible) and not _is_native_choice_compatible(model, syndrome=int(syndrome_int)):
        raise ValueError("native FrontierFast choice engine requested for an unsupported model")
    native_model = _get_native_choice_model(model)
    n_limbs = _detector_limb_count(int(model.num_detectors))
    payload = native_model.decode(
        _mask_to_limbs(int(syndrome_int), int(n_limbs)),
        int(K),
        float(Delta),
        float(score_alpha),
    )
    return _frontier_fast_result_from_native_payload(payload, direction=direction, engine="native_choice")


def _decode_frontier_fast_native_binary_many(
    problem_or_model: object,
    syndromes: Sequence[int | np.ndarray | Sequence[int]],
    *,
    K: int,
    Delta: float,
    score_alpha: float = _SCORE_ALPHA,
    metric_mode: str = "logsumexp_float",
    int_metric_scale: int = 1024,
    direction: str | None = None,
    _assume_compatible: bool = False,
) -> tuple[FrontierFastResult, ...]:
    if int(K) <= 0:
        raise ValueError("K must be positive")
    if float(Delta) < 0.0:
        raise ValueError("Delta must be non-negative")
    if not math.isfinite(float(score_alpha)) or float(score_alpha) < 0.0:
        raise ValueError("score_alpha must be finite and non-negative")
    metric_mode = _normalize_metric_mode(metric_mode)
    int_metric_scale = _validate_int_metric_scale(int_metric_scale)
    syndrome_ints = tuple(_syndrome_to_int(syndrome) for syndrome in tuple(syndromes))
    if not syndrome_ints:
        return tuple()
    model = _coerce_model(problem_or_model, syndrome_int=int(syndrome_ints[0]))
    if not bool(_assume_compatible) and not _is_native_binary_compatible(model, syndrome=int(syndrome_ints[0])):
        raise ValueError("native FrontierFast binary engine requested for an unsupported model")
    native_model = _get_native_binary_model(model)
    n_limbs = _detector_limb_count(int(model.num_detectors))
    payloads = native_model.decode_many(
        tuple(_mask_to_limbs(int(syndrome_int), int(n_limbs)) for syndrome_int in syndrome_ints),
        int(K),
        float(Delta),
        float(score_alpha),
        str(metric_mode),
        int(int_metric_scale),
    )
    return tuple(
        _frontier_fast_result_from_native_payload(dict(payload), direction=direction)
        for payload in tuple(payloads)
    )


def _decode_frontier_fast_native_binary_many_payloads(
    problem_or_model: object,
    syndromes: Sequence[int | np.ndarray | Sequence[int]],
    *,
    K: int,
    Delta: float,
    score_alpha: float = _SCORE_ALPHA,
    metric_mode: str = "logsumexp_float",
    int_metric_scale: int = 1024,
    _assume_compatible: bool = False,
) -> tuple[dict[str, object], ...]:
    if int(K) <= 0:
        raise ValueError("K must be positive")
    if float(Delta) < 0.0:
        raise ValueError("Delta must be non-negative")
    if not math.isfinite(float(score_alpha)) or float(score_alpha) < 0.0:
        raise ValueError("score_alpha must be finite and non-negative")
    metric_mode = _normalize_metric_mode(metric_mode)
    int_metric_scale = _validate_int_metric_scale(int_metric_scale)
    syndrome_ints = tuple(_syndrome_to_int(syndrome) for syndrome in tuple(syndromes))
    if not syndrome_ints:
        return tuple()
    model = _coerce_model(problem_or_model, syndrome_int=int(syndrome_ints[0]))
    if not bool(_assume_compatible) and not _is_native_binary_compatible(model, syndrome=int(syndrome_ints[0])):
        raise ValueError("native FrontierFast binary engine requested for an unsupported model")
    native_model = _get_native_binary_model(model)
    n_limbs = _detector_limb_count(int(model.num_detectors))
    decode_many_payloads = getattr(native_model, "decode_many_payloads", native_model.decode_many)
    payloads = decode_many_payloads(
        tuple(_mask_to_limbs(int(syndrome_int), int(n_limbs)) for syndrome_int in syndrome_ints),
        int(K),
        float(Delta),
        float(score_alpha),
        str(metric_mode),
        int(int_metric_scale),
    )
    return tuple(payload for payload in tuple(payloads))


def _decode_frontier_fast_native_choice_many_payloads(
    problem_or_model: object,
    syndromes: Sequence[int | np.ndarray | Sequence[int]],
    *,
    K: int,
    Delta: float,
    score_alpha: float = _SCORE_ALPHA,
    _assume_compatible: bool = False,
) -> tuple[dict[str, object], ...]:
    if int(K) <= 0:
        raise ValueError("K must be positive")
    if float(Delta) < 0.0:
        raise ValueError("Delta must be non-negative")
    if not math.isfinite(float(score_alpha)) or float(score_alpha) < 0.0:
        raise ValueError("score_alpha must be finite and non-negative")
    syndrome_ints = tuple(_syndrome_to_int(syndrome) for syndrome in tuple(syndromes))
    if not syndrome_ints:
        return tuple()
    model = _coerce_model(problem_or_model, syndrome_int=int(syndrome_ints[0]))
    if not bool(_assume_compatible) and not _is_native_choice_compatible(model, syndrome=int(syndrome_ints[0])):
        raise ValueError("native FrontierFast choice engine requested for an unsupported model")
    native_model = _get_native_choice_model(model)
    n_limbs = _detector_limb_count(int(model.num_detectors))
    payloads = native_model.decode_many(
        tuple(_mask_to_limbs(int(syndrome_int), int(n_limbs)) for syndrome_int in syndrome_ints),
        int(K),
        float(Delta),
        float(score_alpha),
    )
    return tuple(payload for payload in tuple(payloads))


def _decode_frontier_fast_native_choice_many(
    problem_or_model: object,
    syndromes: Sequence[int | np.ndarray | Sequence[int]],
    *,
    K: int,
    Delta: float,
    score_alpha: float = _SCORE_ALPHA,
    direction: str | None = None,
    _assume_compatible: bool = False,
) -> tuple[FrontierFastResult, ...]:
    payloads = _decode_frontier_fast_native_choice_many_payloads(
        problem_or_model,
        syndromes,
        K=int(K),
        Delta=float(Delta),
        score_alpha=float(score_alpha),
        _assume_compatible=bool(_assume_compatible),
    )
    return tuple(
        _frontier_fast_result_from_native_payload(dict(payload), direction=direction, engine="native_choice")
        for payload in tuple(payloads)
    )


def _decode_frontier_fast_native_binary_committee_many_payloads(
    forward_model: FrontierFastModel,
    backward_model: FrontierFastModel,
    syndromes: Sequence[int | np.ndarray | Sequence[int]],
    *,
    K: int,
    Delta: float,
    score_alpha: float = _SCORE_ALPHA,
    metric_mode: str = "logsumexp_float",
    int_metric_scale: int = 1024,
    _assume_compatible: bool = False,
    compact_payload: bool = False,
) -> tuple[dict[str, object], ...]:
    if int(K) <= 0:
        raise ValueError("K must be positive")
    if float(Delta) < 0.0:
        raise ValueError("Delta must be non-negative")
    if not math.isfinite(float(score_alpha)) or float(score_alpha) < 0.0:
        raise ValueError("score_alpha must be finite and non-negative")
    metric_mode = _normalize_metric_mode(metric_mode)
    int_metric_scale = _validate_int_metric_scale(int_metric_scale)
    syndrome_ints = tuple(_syndrome_to_int(syndrome) for syndrome in tuple(syndromes))
    if not syndrome_ints:
        return tuple()
    if not bool(_assume_compatible):
        if not _is_native_binary_compatible(forward_model, syndrome=int(syndrome_ints[0])):
            raise ValueError("native FrontierFast binary engine requested for an unsupported forward model")
        if not _is_native_binary_compatible(backward_model, syndrome=int(syndrome_ints[0])):
            raise ValueError("native FrontierFast binary engine requested for an unsupported backward model")
    native_forward = _get_native_binary_model(forward_model)
    native_backward = _get_native_binary_model(backward_model)
    decode_many_select = getattr(native_forward, "decode_many_select", None)
    if decode_many_select is None:
        raise RuntimeError("native FrontierFast extension does not expose decode_many_select")
    if bool(compact_payload):
        decode_many_select = getattr(native_forward, "decode_many_select_compact", decode_many_select)
    n_limbs_forward = _detector_limb_count(int(forward_model.num_detectors))
    n_limbs_backward = _detector_limb_count(int(backward_model.num_detectors))
    payloads = decode_many_select(
        native_backward,
        tuple(_mask_to_limbs(int(syndrome_int), int(n_limbs_forward)) for syndrome_int in syndrome_ints),
        tuple(_mask_to_limbs(int(syndrome_int), int(n_limbs_backward)) for syndrome_int in syndrome_ints),
        int(K),
        float(Delta),
        float(score_alpha),
        str(metric_mode),
        int(int_metric_scale),
    )
    return tuple(payload for payload in tuple(payloads))


def _decode_frontier_fast_native_binary_committee_many_replay_payloads(
    forward_model: FrontierFastModel,
    backward_model: FrontierFastModel,
    syndromes: Sequence[int | np.ndarray | Sequence[int]],
    *,
    K: int,
    Delta: float,
    score_alpha: float = _SCORE_ALPHA,
    metric_mode: str = "logsumexp_float",
    int_metric_scale: int = 1024,
    _assume_compatible: bool = False,
) -> tuple[dict[str, object], ...]:
    if int(K) <= 0:
        raise ValueError("K must be positive")
    if float(Delta) < 0.0:
        raise ValueError("Delta must be non-negative")
    if not math.isfinite(float(score_alpha)) or float(score_alpha) < 0.0:
        raise ValueError("score_alpha must be finite and non-negative")
    metric_mode = _normalize_metric_mode(metric_mode)
    int_metric_scale = _validate_int_metric_scale(int_metric_scale)
    syndrome_ints = tuple(_syndrome_to_int(syndrome) for syndrome in tuple(syndromes))
    if not syndrome_ints:
        return tuple()
    if not bool(_assume_compatible):
        if not _is_native_binary_compatible(forward_model, syndrome=int(syndrome_ints[0])):
            raise ValueError("native FrontierFast binary engine requested for an unsupported forward model")
        if not _is_native_binary_compatible(backward_model, syndrome=int(syndrome_ints[0])):
            raise ValueError("native FrontierFast binary engine requested for an unsupported backward model")
    native_forward = _get_native_binary_model(forward_model)
    native_backward = _get_native_binary_model(backward_model)
    decode_many_select_replay = getattr(native_forward, "decode_many_select_replay", None)
    if decode_many_select_replay is None:
        raise RuntimeError("native FrontierFast extension does not expose decode_many_select_replay")
    n_limbs_forward = _detector_limb_count(int(forward_model.num_detectors))
    n_limbs_backward = _detector_limb_count(int(backward_model.num_detectors))
    payloads = decode_many_select_replay(
        native_backward,
        tuple(_mask_to_limbs(int(syndrome_int), int(n_limbs_forward)) for syndrome_int in syndrome_ints),
        tuple(_mask_to_limbs(int(syndrome_int), int(n_limbs_backward)) for syndrome_int in syndrome_ints),
        int(K),
        float(Delta),
        float(score_alpha),
        str(metric_mode),
        int(int_metric_scale),
    )
    return tuple(payload for payload in tuple(payloads))


def _progressive_terminal_log_masses(result: progressive.ProgressiveDecodeResult) -> dict[int, float]:
    if tuple(result.terminal_logical_log_mass_items):
        return {
            int(logical): float(log_mass)
            for logical, log_mass in tuple(result.terminal_logical_log_mass_items)
        }
    if tuple(result.terminal_logical_class_summaries):
        return {
            int(summary.logical_mask): float(summary.log_mass)
            for summary in tuple(result.terminal_logical_class_summaries)
        }
    return {}


def _frontier_fast_stats_from_progressive(
    result: progressive.ProgressiveDecodeResult,
    *,
    started: float,
) -> FrontierFastStats:
    expanded_counts = tuple(int(value) for value in tuple(result.expanded_transition_count_by_column))
    candidate_counts = tuple(int(value) for value in tuple(result.beam_candidate_state_count_by_column))
    state_counts = tuple(int(value) for value in tuple(result.state_count_by_column))
    post_counts = state_counts[1:] if len(state_counts) == len(expanded_counts) + 1 else state_counts
    max_pre = int(getattr(result, "max_pre_prune_state_count", 0) or 0)
    if int(max_pre) <= 0 and candidate_counts:
        max_pre = max(candidate_counts)
    return FrontierFastStats(
        processed_columns=int(len(expanded_counts)),
        transition_evals=int(sum(expanded_counts)),
        max_pre_prune_state_count=int(max_pre),
        max_post_prune_state_count=int(max(post_counts, default=0)),
        sum_pre_prune_state_count=int(sum(candidate_counts)),
        sum_post_prune_state_count=int(sum(post_counts)),
        no_path_count=0 if str(result.status) == "ok" else 1,
        transition_time_s=float(getattr(result, "binary_frontierk_transition_expansion_time_s", 0.0) or 0.0),
        merge_time_s=float(getattr(result, "binary_frontierk_merge_time_s", 0.0) or 0.0),
        prune_time_s=float(getattr(result, "binary_frontierk_prune_sort_time_s", 0.0) or 0.0),
        total_time_s=float(time.perf_counter() - float(started)),
    )


def _frontier_fast_result_from_progressive(
    result: progressive.ProgressiveDecodeResult,
    *,
    started: float,
    direction: str | None,
    engine: str,
) -> FrontierFastResult:
    status = "ok" if str(result.status) == "ok" else "no_path"
    terminal_log_masses = _progressive_terminal_log_masses(result) if status == "ok" else {}
    terminal_gap = float(getattr(result, "terminal_top_log_mass_gap", float("nan")))
    if not math.isfinite(float(terminal_gap)) and terminal_log_masses:
        terminal_gap = _terminal_gap(terminal_log_masses)
    return FrontierFastResult(
        status=str(status),
        logical_hat=(int(result.logical_hat) if status == "ok" else None),
        log_evidence=(float(result.log_evidence) if status == "ok" else float("-inf")),
        terminal_log_masses=dict(sorted(terminal_log_masses.items())),
        stats=_frontier_fast_stats_from_progressive(result, started=float(started)),
        direction=direction,
        engine=str(engine),
        terminal_top_log_mass_gap=float(terminal_gap),
    )


def _decode_frontier_fast_binary_adapter(
    problem_or_model: object,
    syndrome: int | np.ndarray | Sequence[int],
    *,
    K: int,
    Delta: float,
    score_alpha: float = _SCORE_ALPHA,
    direction: str | None = None,
    _assume_compatible: bool = False,
) -> FrontierFastResult:
    started = time.perf_counter()
    if int(K) <= 0:
        raise ValueError("K must be positive")
    if float(Delta) < 0.0:
        raise ValueError("Delta must be non-negative")
    syndrome_int = _syndrome_to_int(syndrome)
    model = _coerce_model(problem_or_model, syndrome_int=int(syndrome_int))
    if not bool(_assume_compatible) and not _is_binary_fastpath_compatible(model, syndrome=int(syndrome_int)):
        raise ValueError("binary FrontierFast engine requested for an unsupported model")

    env_overrides = {
        "FRONTIERK_BINARY_TRANSITION_FAST_PATH": "1",
        "FRONTIERK_PRUNE_PRIMARY_TOPK_FAST_PATH": "1",
        "FRONTIERK_FUSED_BINARY_PRIMARY_TOPK_FAST_PATH": "1",
        "FRONTIERK_BINARY_LOCAL_PATTERN_FEATURE_TABLE": "1",
        "FRONTIERK_BINARY_UNIQUE_DETECTOR_SCORE_FAST_PATH": "1",
        "FRONTIERK_BINARY_SMALL_CANDIDATE_DIRECT_GAP": "1",
        "FRONTIERK_BINARY_ZERO_CLOSE_MASK_FAST_PATH": "1",
    }
    previous_env = {name: os.environ.get(name) for name in env_overrides}
    try:
        for name, value in env_overrides.items():
            os.environ[str(name)] = str(value)
        result = progressive.decode_progressive(
            list(model.columns),
            target_syndrome=int(syndrome_int),
            num_detectors=int(model.num_detectors),
            num_observables=int(model.num_observables),
            beam_size=int(K),
            beam_score_gap_threshold=float(Delta),
            score_mode=_score_mode_for_alpha(float(score_alpha)),
            layout=model.layout,
            track_best_path=False,
            merge_duplicate_states=True,
            return_terminal_maps=True,
            future_parity_scorer="cached",
        )
    finally:
        for name, old_value in previous_env.items():
            if old_value is None:
                os.environ.pop(str(name), None)
            else:
                os.environ[str(name)] = str(old_value)

    return _frontier_fast_result_from_progressive(
        result,
        started=float(started),
        direction=direction,
        engine="binary",
    )


def _decode_frontier_fast_python_reference(
    problem_or_model: object,
    syndrome: int | np.ndarray | Sequence[int],
    *,
    K: int,
    Delta: float,
    score_alpha: float = _SCORE_ALPHA,
    direction: str | None = None,
) -> FrontierFastResult:
    """Decode with the pure Python FrontierFast V1 reference implementation.

    V1 uses packed integer state keys, merges by posterior log-mass, and prunes
    by `score >= best_score - Delta` followed by a cap at `K`.
    """
    started = time.perf_counter()
    if int(K) <= 0:
        raise ValueError("K must be positive")
    if float(Delta) < 0.0:
        raise ValueError("Delta must be non-negative")
    syndrome_int = _syndrome_to_int(syndrome)
    model = _coerce_model(problem_or_model, syndrome_int=int(syndrome_int))
    columns = tuple(model.columns)
    if not columns:
        stats = _empty_stats(started=started, no_path_count=0)
        return FrontierFastResult(
            status="ok" if int(syndrome_int) == 0 else "no_path",
            logical_hat=0 if int(syndrome_int) == 0 else None,
            log_evidence=0.0 if int(syndrome_int) == 0 else float("-inf"),
            terminal_log_masses={0: 0.0} if int(syndrome_int) == 0 else {},
            stats=stats if int(syndrome_int) == 0 else replace(stats, no_path_count=1),
            direction=direction,
            engine="python",
            terminal_top_log_mass_gap=float("inf") if int(syndrome_int) == 0 else float("nan"),
        )

    num_detectors = int(model.num_detectors)
    num_observables = int(model.num_observables)
    detector_mask_mask = (1 << int(num_detectors)) - 1
    logical_mask_mask = (1 << int(num_observables)) - 1
    scorer = progressive.FutureParityScorer(
        frontier=model.layout,
        target_syndrome_int=int(syndrome_int),
        mode="cached",
    )

    state_log_mass: dict[int, float] = {0: 0.0}
    processed_columns = 0
    transition_evals = 0
    max_pre_prune = 0
    max_post_prune = 0
    sum_pre_prune = 0
    sum_post_prune = 0
    transition_time_s = 0.0
    merge_time_s = 0.0
    prune_time_s = 0.0

    for column_index, column in enumerate(columns):
        processed_columns = int(column_index) + 1
        close_mask = int(model.layout.closing_masks[int(column_index)])
        active_mask = int(model.layout.active_masks_after_column[int(column_index)])
        log_priors = progressive._column_log_priors(column)
        detector_responses = tuple(int(mask) for mask in column.detector_response_masks)
        logical_responses = tuple(int(mask) for mask in column.logical_response_masks)
        next_log_mass: dict[int, float] = {}

        transition_started = time.perf_counter()
        merge_elapsed = 0.0
        for key, log_mass in state_log_mass.items():
            key_int = int(key)
            det_mask = int(key_int & detector_mask_mask)
            logical_mask = int((key_int >> int(num_detectors)) & logical_mask_mask)
            for local_state, log_prior in enumerate(log_priors):
                if not math.isfinite(float(log_prior)):
                    continue
                transition_evals += 1
                candidate_det = int(det_mask) ^ int(detector_responses[int(local_state)])
                if (((int(candidate_det) ^ int(syndrome_int)) & int(close_mask)) != 0):
                    continue
                next_det = int(candidate_det) & int(active_mask)
                next_logical = int(logical_mask) ^ int(logical_responses[int(local_state)])
                next_key = int(next_det) | (int(next_logical) << int(num_detectors))
                new_log_mass = float(log_mass) + float(log_prior)
                existing = next_log_mass.get(int(next_key))
                if existing is None:
                    next_log_mass[int(next_key)] = float(new_log_mass)
                else:
                    merge_started = time.perf_counter()
                    next_log_mass[int(next_key)] = _logaddexp_pair(float(existing), float(new_log_mass))
                    merge_elapsed += float(time.perf_counter() - merge_started)
        transition_elapsed = float(time.perf_counter() - transition_started)
        transition_time_s += float(max(0.0, transition_elapsed - merge_elapsed))
        merge_time_s += float(merge_elapsed)

        candidate_count = int(len(next_log_mass))
        max_pre_prune = max(int(max_pre_prune), int(candidate_count))
        sum_pre_prune += int(candidate_count)
        if not next_log_mass:
            stats = FrontierFastStats(
                processed_columns=int(processed_columns),
                transition_evals=int(transition_evals),
                max_pre_prune_state_count=int(max_pre_prune),
                max_post_prune_state_count=int(max_post_prune),
                sum_pre_prune_state_count=int(sum_pre_prune),
                sum_post_prune_state_count=int(sum_post_prune),
                no_path_count=1,
                transition_time_s=float(transition_time_s),
                merge_time_s=float(merge_time_s),
                prune_time_s=float(prune_time_s),
                total_time_s=float(time.perf_counter() - started),
            )
            return FrontierFastResult(
                status="no_path",
                logical_hat=None,
                log_evidence=float("-inf"),
                terminal_log_masses={},
                stats=stats,
                direction=direction,
                engine="python",
                terminal_top_log_mass_gap=float("nan"),
            )

        prune_started = time.perf_counter()
        score_cache_by_detector: dict[int, float] = {}
        score_by_key: dict[int, float] = {}
        for key, log_mass in next_log_mass.items():
            det_mask = int(key) & int(detector_mask_mask)
            future_score = score_cache_by_detector.get(int(det_mask))
            if future_score is None:
                future_score = float(score_alpha) * float(
                    scorer.score_from_detector_mask(
                        det_mask=int(det_mask),
                        active_mask=int(active_mask),
                        boundary_column_index=int(column_index),
                    )
                )
                score_cache_by_detector[int(det_mask)] = float(future_score)
            score_by_key[int(key)] = float(log_mass) + float(future_score)
        kept_keys = _select_keys_by_score_gap(
            score_by_key=score_by_key,
            log_mass_by_key=next_log_mass,
            K=int(K),
            Delta=float(Delta),
        )
        state_log_mass = {int(key): float(next_log_mass[int(key)]) for key in kept_keys}
        post_count = int(len(state_log_mass))
        max_post_prune = max(int(max_post_prune), int(post_count))
        sum_post_prune += int(post_count)
        prune_time_s += float(time.perf_counter() - prune_started)

    terminal_log_masses: dict[int, float] = {}
    for key, log_mass in state_log_mass.items():
        det_mask = int(key) & int(detector_mask_mask)
        if int(det_mask) != 0:
            continue
        logical_mask = int((int(key) >> int(num_detectors)) & int(logical_mask_mask))
        terminal_log_masses[int(logical_mask)] = _logaddexp_pair(
            float(terminal_log_masses.get(int(logical_mask), float("-inf"))),
            float(log_mass),
        )

    if not terminal_log_masses:
        status = "no_path"
        logical_hat: int | None = None
        log_evidence = float("-inf")
        no_path_count = 1
    else:
        status = "ok"
        logical_hat = min(
            terminal_log_masses,
            key=lambda logical: (-float(terminal_log_masses[int(logical)]), int(logical)),
        )
        log_evidence = _logaddexp_many(terminal_log_masses.values())
        no_path_count = 0

    stats = FrontierFastStats(
        processed_columns=int(processed_columns),
        transition_evals=int(transition_evals),
        max_pre_prune_state_count=int(max_pre_prune),
        max_post_prune_state_count=int(max_post_prune),
        sum_pre_prune_state_count=int(sum_pre_prune),
        sum_post_prune_state_count=int(sum_post_prune),
        no_path_count=int(no_path_count),
        transition_time_s=float(transition_time_s),
        merge_time_s=float(merge_time_s),
        prune_time_s=float(prune_time_s),
        total_time_s=float(time.perf_counter() - started),
    )
    return FrontierFastResult(
        status=str(status),
        logical_hat=logical_hat,
        log_evidence=float(log_evidence),
        terminal_log_masses=dict(sorted(terminal_log_masses.items())),
        stats=stats,
        direction=direction,
        engine="python",
        terminal_top_log_mass_gap=_terminal_gap(terminal_log_masses),
    )


def decode_frontier_fast(
    problem_or_model: object,
    syndrome: int | np.ndarray | Sequence[int],
    *,
    K: int,
    Delta: float,
    score_alpha: float = _SCORE_ALPHA,
    metric_mode: str = "logsumexp_float",
    int_metric_scale: int = 1024,
    _engine: str = "auto",
) -> FrontierFastResult:
    """Decode with FrontierFast V1.1.

    `_engine` is a private test hook: `auto` dispatches strict binary models to
    the native binary engine when available, otherwise the optimized binary
    adapter, otherwise native multi-choice for compatible non-binary columns or
    the pure Python V1 reference.
    """
    engine = str(_engine).strip().lower()
    if engine not in {"auto", "python", "binary", "native_binary", "native_choice"}:
        raise ValueError(
            "_engine must be one of 'auto', 'python', 'binary', 'native_binary', or 'native_choice'"
        )
    if int(K) <= 0:
        raise ValueError("K must be positive")
    if float(Delta) < 0.0:
        raise ValueError("Delta must be non-negative")
    if not math.isfinite(float(score_alpha)) or float(score_alpha) < 0.0:
        raise ValueError("score_alpha must be finite and non-negative")
    metric_mode = _normalize_metric_mode(metric_mode)
    int_metric_scale = _validate_int_metric_scale(int_metric_scale)
    if metric_mode != "logsumexp_float" and engine not in {"auto", "native_binary"}:
        raise ValueError("frontierLite/maxlog_int metric_mode is only supported by the native_binary engine")
    syndrome_int = _syndrome_to_int(syndrome)
    model = _coerce_model(problem_or_model, syndrome_int=int(syndrome_int))

    if engine == "python":
        return _decode_frontier_fast_python_reference(
            model,
            int(syndrome_int),
            K=int(K),
            Delta=float(Delta),
            score_alpha=float(score_alpha),
        )
    native_compatible = _is_native_binary_compatible(model, syndrome=int(syndrome_int))
    if engine == "native_binary":
        if not native_binary_available():
            raise RuntimeError("native FrontierFast binary extension is not built")
        if not bool(native_compatible):
            raise ValueError("native FrontierFast binary engine requested for an unsupported model")
        return _decode_frontier_fast_native_binary(
            model,
            int(syndrome_int),
            K=int(K),
            Delta=float(Delta),
            score_alpha=float(score_alpha),
            metric_mode=str(metric_mode),
            int_metric_scale=int(int_metric_scale),
            _assume_compatible=True,
        )
    choice_compatible = (
        _is_native_choice_compatible(model, syndrome=int(syndrome_int))
        if engine == "native_choice"
        else False
    )
    if engine == "native_choice":
        if not native_choice_available():
            raise RuntimeError("native FrontierFast choice extension is not built")
        if not bool(choice_compatible):
            raise ValueError("native FrontierFast choice engine requested for an unsupported model")
        return _decode_frontier_fast_native_choice(
            model,
            int(syndrome_int),
            K=int(K),
            Delta=float(Delta),
            score_alpha=float(score_alpha),
            _assume_compatible=True,
        )
    if engine == "auto" and bool(native_compatible):
        return _decode_frontier_fast_native_binary(
            model,
            int(syndrome_int),
            K=int(K),
            Delta=float(Delta),
            score_alpha=float(score_alpha),
            metric_mode=str(metric_mode),
            int_metric_scale=int(int_metric_scale),
            _assume_compatible=True,
        )
    if metric_mode != "logsumexp_float":
        raise ValueError("frontierLite/maxlog_int metric_mode requires a native-binary-compatible model")
    compatible = _is_binary_fastpath_compatible(model, syndrome=int(syndrome_int))
    if engine == "binary":
        if not bool(compatible):
            raise ValueError("binary FrontierFast engine requested for an unsupported model")
        return _decode_frontier_fast_binary_adapter(
            model,
            int(syndrome_int),
            K=int(K),
            Delta=float(Delta),
            score_alpha=float(score_alpha),
            _assume_compatible=True,
        )
    if bool(compatible):
        return _decode_frontier_fast_binary_adapter(
            model,
            int(syndrome_int),
            K=int(K),
            Delta=float(Delta),
            score_alpha=float(score_alpha),
            _assume_compatible=True,
        )
    if engine == "auto":
        choice_compatible = _is_native_choice_compatible(model, syndrome=int(syndrome_int))
        if bool(choice_compatible):
            return _decode_frontier_fast_native_choice(
                model,
                int(syndrome_int),
                K=int(K),
                Delta=float(Delta),
                score_alpha=float(score_alpha),
                _assume_compatible=True,
            )
    return _decode_frontier_fast_python_reference(
        model,
        int(syndrome_int),
        K=int(K),
        Delta=float(Delta),
        score_alpha=float(score_alpha),
    )


def _committee_member_summary(*, direction: str, result: FrontierFastResult) -> FrontierFastCommitteeMember:
    top1, _top2 = _frontier_fast_top_posteriors(result)
    return FrontierFastCommitteeMember(
        direction=str(direction),
        status=str(result.status),
        logical_hat=result.logical_hat,
        log_evidence=float(result.log_evidence),
        terminal_top_log_mass_gap=float(result.terminal_top_log_mass_gap),
        top1_posterior=float(top1),
    )


def _committee_selection_key(
    *,
    result: FrontierFastResult,
    direction: str,
    preferred_direction: str = "forward",
) -> tuple[float, ...]:
    status_key = str(result.status).strip().lower()
    if status_key == "ok":
        status_rank = 2.0
    elif status_key == "no_path":
        status_rank = 1.0
    else:
        status_rank = 0.0
    log_evidence = (
        float(result.log_evidence)
        if status_key == "ok" and math.isfinite(float(result.log_evidence))
        else float("-inf")
    )
    terminal_gap = float(result.terminal_top_log_mass_gap)
    if math.isnan(float(terminal_gap)):
        terminal_gap = float("-inf")
    top1_posterior, _top2_posterior = _frontier_fast_top_posteriors(result)
    top1_key = float(top1_posterior) if math.isfinite(float(top1_posterior)) else float("-inf")
    preferred_bonus = 1.0 if str(direction) == str(preferred_direction) else 0.0
    return (
        float(status_rank),
        float(log_evidence),
        float(terminal_gap),
        float(top1_key),
        0.0,
        float(preferred_bonus),
    )


def decode_frontier_fast_committee(
    problem_or_model: object,
    syndrome: int | np.ndarray | Sequence[int],
    *,
    K: int,
    Delta: float,
    score_alpha: float = _SCORE_ALPHA,
    metric_mode: str = "logsumexp_float",
    int_metric_scale: int = 1024,
    _engine: str = "auto",
) -> FrontierFastResult:
    syndrome_int = _syndrome_to_int(syndrome)
    forward_model = _coerce_model(problem_or_model, syndrome_int=int(syndrome_int), direction="forward")
    backward_model = _coerce_model(problem_or_model, syndrome_int=int(syndrome_int), direction="backward")
    forward_result = replace(
        decode_frontier_fast(
            forward_model,
            int(syndrome_int),
            K=int(K),
            Delta=float(Delta),
            score_alpha=float(score_alpha),
            metric_mode=str(metric_mode),
            int_metric_scale=int(int_metric_scale),
            _engine=str(_engine),
        ),
        direction="forward",
    )
    backward_result = replace(
        decode_frontier_fast(
            backward_model,
            int(syndrome_int),
            K=int(K),
            Delta=float(Delta),
            score_alpha=float(score_alpha),
            metric_mode=str(metric_mode),
            int_metric_scale=int(int_metric_scale),
            _engine=str(_engine),
        ),
        direction="backward",
    )
    selected_direction, selected_result = max(
        (("forward", forward_result), ("backward", backward_result)),
        key=lambda item: _committee_selection_key(
            result=item[1],
            direction=str(item[0]),
            preferred_direction="forward",
        ),
    )
    members = (
        _committee_member_summary(direction="forward", result=forward_result),
        _committee_member_summary(direction="backward", result=backward_result),
    )
    no_path_count = int(forward_result.stats.no_path_count) + int(backward_result.stats.no_path_count)
    return replace(
        selected_result,
        direction=str(selected_direction),
        committee_members=members,
        stats=replace(
            selected_result.stats,
            no_path_count=int(no_path_count) if int(no_path_count) == 2 else int(selected_result.stats.no_path_count),
        ),
    )


def _smoke_columns() -> tuple[progressive.FactorTransition, ...]:
    return (
        progressive.FactorTransition(
            factor_id=0,
            outcomes=(
                progressive.OutcomeTransition(probability=0.55, detector_mask=0, logical_mask=0, label="a0"),
                progressive.OutcomeTransition(probability=0.45, detector_mask=1, logical_mask=1, label="a1"),
            ),
            instruction_offset=0,
            label="f0",
        ),
        progressive.FactorTransition(
            factor_id=1,
            outcomes=(
                progressive.OutcomeTransition(probability=0.70, detector_mask=0, logical_mask=0, label="b0"),
                progressive.OutcomeTransition(probability=0.30, detector_mask=1, logical_mask=0, label="b1"),
            ),
            instruction_offset=1,
            label="f1",
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tiny FrontierFast V1 smoke benchmark")
    parser.add_argument("--K", "--beam-cap", dest="K", type=int, default=8)
    parser.add_argument("--delta", type=float, default=8.0)
    parser.add_argument("--shots", type=int, default=20)
    args = parser.parse_args(argv)
    results: list[FrontierFastResult] = []
    for shot in range(int(args.shots)):
        syndrome = int(shot & 1)
        results.append(decode_frontier_fast(_smoke_columns(), syndrome, K=int(args.K), Delta=float(args.delta)))
    ok_results = [result for result in results if result.status == "ok"]
    mean_transition_evals = (
        float(sum(result.stats.transition_evals for result in results)) / float(max(1, len(results)))
    )
    mean_total_time_s = (
        float(sum(result.stats.total_time_s for result in results)) / float(max(1, len(results)))
    )
    print(f"shots={len(results)} ok={len(ok_results)}")
    print(f"mean_transition_evals={mean_transition_evals:.3f}")
    print(f"max_pre_prune_state_count={max(result.stats.max_pre_prune_state_count for result in results)}")
    print(f"max_post_prune_state_count={max(result.stats.max_post_prune_state_count for result in results)}")
    print(f"mean_total_time_s={mean_total_time_s:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
