#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import multiprocessing as mp
import os
import signal
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache_betterbeam")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grosscode.codes.bivariate_bicycle import get_bivariate_bicycle_backend_spec, is_bivariate_bicycle_backend
from grosscode.codes.generalized_bicycle import get_generalized_bicycle_backend_spec, is_generalized_bicycle_backend
from grosscode.dem.builder import SplitSectorMetadata, build_split_sector_problem, load_dem_side_with_metadata_from_stim
from grosscode.dem.stim_fault_pipeline import build_gross_split_sector_merged_correction_map
from grosscode.dem.triangles import catalog_exact_local_triangles, select_nonoverlapping_triangle_relations
from grosscode.utils.gf2 import dense_mod2, select_independent_rows_mod2
try:
    from tools import frontierk_terminal_signal_analysis as terminal_signal_analysis
except Exception:  # pragma: no cover - optional legacy report helper.
    terminal_signal_analysis = None
from tools import steane_progressive_decoder as progressive


ROUND_COUNT = 12
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "20260401_gross144_dem_x_progressive"
WEAK_OVERLAY_PATH = (
    REPO_ROOT / "results" / "20260223_162450_gross144_p2e3_3e3_4e3_beam_overlay_12k_beam_weak" / "scan_summary.csv"
)
STRONG_OVERLAY_PATH = (
    REPO_ROOT / "results" / "20260223_162450_gross144_p2e3_3e3_4e3_beam_overlay_12k_beam_strong" / "scan_summary.csv"
)
WEAK_REPLAY_P1E3_PATH = (
    REPO_ROOT / "results" / "20260325_weakbeam_stage1_replay_p1e3_p2e3_p3e3_s500" / "p0p001" / "endpoint_summary.csv"
)
COLUMN_ORDER_CHOICES = (
    "fwd_deadline",
    "deadline_reorder",
    "deadline_min_active_w32",
    "deadline_close_first_w32",
    "bwd_deadline",
    "triangle_all_deadline_reorder",
    "triangle_patch10_deadline_reorder",
    "triangle_patch25_deadline_reorder",
    "backward_deadline_reorder",
    "back_deadline_min_active_w32",
    "back_deadline_close_first_w32",
    "bidirectional_deadline_reorder",
    "shared_mitm_order",
    "time_order",
    "span_deadline_reorder",
    "natural_inside_span",
    "bridge_zipper_reorder",
    "rank_gain_per_open_row",
    "logical_frontload_reorder",
    "row_band_round_robin_reorder",
    "row_band_reverse_round_robin_reorder",
    "row_band_center_out_reorder",
    "round_span_zigzag_reorder",
    "round_span_center_out_reorder",
    "mass_desc_deadline_reorder",
    "mass_asc_deadline_reorder",
    "detector_weight_asc_reorder",
    "detector_weight_desc_reorder",
    "frontier_width_greedy_reorder",
    "close_first_greedy_reorder",
    "window96_deadline_reorder",
    "midpoint_joint_reorder",
    "midpoint_backward_reorder",
    "custom_file",
)
FORWARD_DEADLINE_ORDER_ALIASES = {"fwd_deadline", "deadline_reorder"}
BACKWARD_DERIVED_COLUMN_ORDER_CHOICES = (
    "bwd_deadline",
    "backward_deadline_reorder",
    "back_deadline_min_active_w32",
    "back_deadline_close_first_w32",
)
LOCAL_DEADLINE_WINDOW = 96
DEADLINE_PRESSURE_WINDOW = 32
TRIANGLE_PATCH_ORDER_FRACTIONS: dict[str, float] = {
    "triangle_all_deadline_reorder": 1.0,
    "triangle_patch10_deadline_reorder": 0.10,
    "triangle_patch25_deadline_reorder": 0.25,
}
MIDPOINT_JOINT_BOUNDARY_WEIGHT = 2.0
MIDPOINT_BACKWARD_CUT_FRACTIONS = (0.5, 0.625, 0.75, 0.875)
MIDPOINT_BACKWARD_CUT_BOUNDARY_ROW_WEIGHT = 32.0
DEFAULT_MIDDLE_JOIN_CUT_BEAM_FACTOR = 1
DEFAULT_MIDDLE_JOIN_CUT_WINDOW_COLUMNS = 0
DEFAULT_MIDDLE_JOIN_MULTICUT_STRIDE = 0
DEFAULT_MIDDLE_JOIN_MULTICUT_MAX_CUTS = 0
DEFAULT_MIDDLE_JOIN_MULTICUT_WEIGHT_MODE = "compatibility_gap"
DEFAULT_SPLICE_CANDIDATE_COUNT = 8
DEFAULT_SPLICE_CUT_SELECTOR = "middle"
DEFAULT_SPLICE_MAX_CUTS = 1
DEFAULT_SPLICE_AGGREGATE = "median"
TERMINAL_SELECTOR_COST_TILT_LAMBDAS = (0.25, 0.5, 1.0)


@dataclass(frozen=True, slots=True)
class LoadedProgressiveFamily:
    backend: str
    family_key: str
    scope: str
    scope_label: str
    benchmark_title: str
    benchmark_description: str
    benchmark_source_note: str
    detector_symbol: str
    logical_symbol: str
    metadata_symbol: str
    priors_symbol: str
    column_order_name: str
    column_order_source: str
    model_label: str
    decode_label: str
    columns: tuple[progressive.ProgressiveColumn, ...]
    layout: progressive.ProgressiveFrontierLayout
    matrix_rows: int
    matrix_cols: int
    logical_rows: int
    edge_count: int
    noisy_rounds: int
    total_rounds: int
    correction_state_mode: str
    correction_state_bits: int


@dataclass(frozen=True, slots=True)
class JointMiddleJoinOrderedFamilies:
    prefix_columns: int
    suffix_columns: int
    forward_family: LoadedProgressiveFamily
    backward_family: LoadedProgressiveFamily
    cut_boundary_rows: int
    forward_prefix_active_area: int
    backward_prefix_active_area: int


_GLOBAL_FAMILY: LoadedProgressiveFamily | None = None
_GLOBAL_SAMPLE_COLUMNS: tuple[progressive.ProgressiveColumn, ...] = ()
_GLOBAL_SAMPLE_PRIORS: np.ndarray = np.zeros(0, dtype=np.float64)
_GLOBAL_SEED: int = 12345
_GLOBAL_LOOKAHEAD_DEPTH: int = 0
_GLOBAL_LOOKAHEAD_SHORTLIST_SIZE: int = 0
_GLOBAL_DELAYED_PRUNING_GAP_THRESHOLD: float = 0.0
_GLOBAL_DELAYED_PRUNING_FACTOR: int = 1
_GLOBAL_PRUNING_REPLAY_CHECKPOINT_STRIDE: int = 0
_GLOBAL_PRUNING_REPLAY_HORIZON: int = 0
_GLOBAL_TAIL_EXACT_COLUMNS: int = 0
_GLOBAL_SUPERSTEP_MODE: str = "none"
_GLOBAL_SUPERSTEP_PATH_BUDGET: int = 250000
_GLOBAL_SUPERSTEP_STATE_BUDGET: int = 4096
_GLOBAL_SUPERSTEP_TRANSITION_BUDGET: int = 0
_GLOBAL_DETECTOR_BUCKET_PRUNING: bool = False
_GLOBAL_DETECTOR_BUCKET_MAX_LOGICALS: int = 4
_GLOBAL_LOGICAL_CLASS_RESERVE_MIN_CLASSES: int = 0
_GLOBAL_LOGICAL_CLASS_RESERVE_MAX_REPLACEMENTS: int = 0
_GLOBAL_LOGICAL_CLASS_RESERVE_MIN_REMAINING_COLUMNS: int = 0
_GLOBAL_LOGICAL_CLASS_QUOTA_TOP_CLASSES: int = 0
_GLOBAL_LOGICAL_CLASS_QUOTA_RESERVED_SLOTS: int = 0
_GLOBAL_LOGICAL_CLASS_QUOTA_MIN_REMAINING_COLUMNS: int = 0
_GLOBAL_LINEAGE_RESERVE_CHECKPOINT_STRIDE: int = 0
_GLOBAL_LINEAGE_RESERVE_RESERVED_SLOTS: int = 0
_GLOBAL_LOGICAL_RERANK_COLUMNS: int = 0
_GLOBAL_LOGICAL_RERANK_SHORTLIST_SIZE: int = 0
_GLOBAL_LOGICAL_RERANK_MIN_CLASSES: int = 0
_GLOBAL_LOGICAL_RERANK_STATE_BUDGET: int = 1024
_GLOBAL_LOGICAL_RERANK_TRANSITION_BUDGET: int = 100000
_GLOBAL_LOGICAL_RERANK_CHECKPOINT_STRIDE: int = 0
_GLOBAL_LOGICAL_RERANK_MAX_PASSES: int = 1
_GLOBAL_LOGICAL_RERANK_MODE: str = "exact_tail"
_GLOBAL_FINAL_LOGICAL_SELECT_MODE: str = "log_mass"
_GLOBAL_FINAL_LOGICAL_SELECT_REP_COST_WEIGHT: float = 0.0
_GLOBAL_FINAL_LOGICAL_SELECT_MAX_LOG_MASS_GAP: float = float("inf")
_GLOBAL_FINAL_LOGICAL_SELECT_RANK2_VITERBI_TOLERANCE: float = 0.0
_GLOBAL_TRACK_BEST_PATH: bool = False
_GLOBAL_MERGE_DUPLICATE_STATES: bool = True
_GLOBAL_STATE_MERGE_PERIOD_COLUMNS: int = 0
_GLOBAL_SCORE_MODES: tuple[str, ...] = ("prefix",)
_GLOBAL_BEAM_SCORE_GAP_THRESHOLD: float | None = None
_GLOBAL_BEAM_SCORE_GAP_POLICY: progressive.BeamScoreGapPolicy | None = None
_GLOBAL_SELECTIVE_SECONDARY_SCORE_MODE: str = ""
_GLOBAL_SELECTIVE_SECONDARY_TRIGGER_GAP: float = 0.0
_GLOBAL_SELECTIVE_SECONDARY_BAND_SIZE: int = 0
_GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_MODE: str = "none"
_GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_CUTOFF_GAP_THRESHOLD: float = 0.0
_GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_NEAR_CUT_WIDTH: float = 0.0
_GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_MAX_CANDIDATES: int = 0
_GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_CANDIDATE_TOP1_SHARE_THRESHOLD: float = 0.0
_GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_SUPPORT_GAP_THRESHOLD: float = float("inf")
_GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_OVERFLOW_RATIO_THRESHOLD: float = float("inf")
_GLOBAL_EXPORT_STATE_COUNT_PROFILE: bool = False
_GLOBAL_EXPORT_TERMINAL_SELECTOR_SIGNALS: bool = False
_GLOBAL_EXPORT_FRONTIER_PRESSURE_TRACE: bool = False
_GLOBAL_PRODUCTION_FAST_MODE: bool = False
_GLOBAL_FRONTIER_PRESSURE_TRACE_ROWS: list[dict[str, object]] = []
_GLOBAL_COLUMN_ORDER_LABEL: str = "deadline_reorder"
_GLOBAL_DECODER_MODE: str = "forward"
_GLOBAL_BACKWARD_COLUMNS: tuple[progressive.ProgressiveColumn, ...] = ()
_GLOBAL_BACKWARD_LAYOUT: progressive.ProgressiveFrontierLayout | None = None
_GLOBAL_BACKWARD_COLUMN_ORDER: str = ""
_GLOBAL_FORWARD_GUIDANCE_WEIGHT: float = 1.0
_GLOBAL_FORWARD_GUIDANCE_CLIP: float = 6.0
_GLOBAL_FORWARD_GUIDANCE_TRIGGER_GAP: float = 0.0
_GLOBAL_FORWARD_GUIDANCE_WIDEN_FACTOR: float = 2.0
_GLOBAL_FORWARD_GUIDANCE_MIN_INFO_BITS: float = 0.0
_GLOBAL_FORWARD_GUIDANCE_SNAPSHOT_FACTOR: float = 1.0
_GLOBAL_FORWARD_GUIDANCE_SNAPSHOT_GAP: float | None = None
_GLOBAL_FORWARD_GUIDANCE_SNAPSHOT_SOURCE: str = "kept"
_GLOBAL_FORWARD_GUIDANCE_HAMMING_RADIUS: int = 0
_GLOBAL_FORWARD_GUIDANCE_TRIGGER_MODE: str = "top_gap"
_GLOBAL_FORWARD_GUIDANCE_NEARCUT_GAP: float = 0.0
_GLOBAL_FORWARD_GUIDANCE_POOL_TRIGGER_MIN_POSITIVE_NEARCUT: int = 1
_GLOBAL_FORWARD_GUIDANCE_DIVERSITY_FALLBACK: str = "none"
_GLOBAL_FORWARD_GUIDANCE_MODE: str = "detector_penalty"
_GLOBAL_MIDDLE_JOIN_PREFIX_COLUMNS: int | None = None
_GLOBAL_MIDDLE_JOIN_CUT_BEAM_FACTOR: int = 1
_GLOBAL_MIDDLE_JOIN_CUT_WINDOW_COLUMNS: int = 0
_GLOBAL_MIDDLE_JOIN_MULTICUT_PREFIX_COLUMNS: tuple[int, ...] = ()
_GLOBAL_MIDDLE_JOIN_MULTICUT_STRIDE: int = 0
_GLOBAL_MIDDLE_JOIN_MULTICUT_MAX_CUTS: int = 0
_GLOBAL_MIDDLE_JOIN_MULTICUT_WEIGHT_MODE: str = DEFAULT_MIDDLE_JOIN_MULTICUT_WEIGHT_MODE
_GLOBAL_BIDIRECTIONAL_SPLICE_RERANK: bool = False
_GLOBAL_SPLICE_CANDIDATE_COUNT: int = DEFAULT_SPLICE_CANDIDATE_COUNT
_GLOBAL_SPLICE_CUT_SELECTOR: str = DEFAULT_SPLICE_CUT_SELECTOR
_GLOBAL_SPLICE_MAX_CUTS: int = DEFAULT_SPLICE_MAX_CUTS
_GLOBAL_SPLICE_AGGREGATE: str = DEFAULT_SPLICE_AGGREGATE
_GLOBAL_SPLICE_REPLACE_FINAL_SELECTION: bool = False

FINAL_LOGICAL_SELECT_MODE_CHOICES = (
    "log_mass",
    "best_viterbi",
    "log_mass_then_viterbi",
    "log_mass_then_viterbi_then_rep_cost",
    "log_mass_minus_rep_cost",
    "cost_tilted_log_mass",
    "rank2_viterbi_tie_rerank_v1",
)


def _install_worker_signal_handlers() -> None:
    """Let the parent own Ctrl-C; keep worker processes quiet."""
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (AttributeError, ValueError):
        return


def _set_macos_qos_from_env() -> None:
    qos_text = str(os.environ.get("FRONTIERK_MACOS_QOS", "")).strip().lower()
    if not qos_text or sys.platform != "darwin":
        return
    qos_map = {
        "background": 0x09,
        "utility": 0x11,
        "default": 0x15,
        "user_initiated": 0x19,
        "user-initiated": 0x19,
        "interactive": 0x21,
        "user_interactive": 0x21,
        "user-interactive": 0x21,
    }
    qos_class = qos_map.get(qos_text)
    if qos_class is None:
        return
    try:
        relative_priority = int(str(os.environ.get("FRONTIERK_MACOS_QOS_RELPRI", "0")).strip())
    except ValueError:
        relative_priority = 0
    try:
        import ctypes

        pthread = ctypes.CDLL("/usr/lib/libpthread.dylib")
        pthread.pthread_set_qos_class_self_np.argtypes = [ctypes.c_uint, ctypes.c_int]
        pthread.pthread_set_qos_class_self_np.restype = ctypes.c_int
        pthread.pthread_set_qos_class_self_np(ctypes.c_uint(int(qos_class)), ctypes.c_int(int(relative_priority)))
    except Exception:
        return


def _init_progressive_worker() -> None:
    _set_macos_qos_from_env()
    _install_worker_signal_handlers()


def _frame_fer_to_per_round_exact(frame_fer: float, rounds: int = ROUND_COUNT) -> float:
    value = float(frame_fer)
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(1.0 - (1.0 - value) ** (1.0 / float(rounds)))


def _benchmark_descriptor(backend: str) -> tuple[str, str]:
    backend_text = str(backend)
    if backend_text == "bravyi_depth7":
        return (
            "Gross split-sector DEM",
            "accepted public split-sector detector-side DEM benchmark",
        )
    if is_bivariate_bicycle_backend(backend_text):
        spec = get_bivariate_bicycle_backend_spec(backend_text)
        bb_label = f"BB [[{int(spec.n)},{int(spec.k)},{int(spec.distance)}]]"
        return (
            f"{bb_label} split-sector DEM",
            f"non-default {bb_label} detector-side DEM benchmark built locally from SlidingWindowDecoder",
        )
    if is_generalized_bicycle_backend(backend_text):
        spec = get_generalized_bicycle_backend_spec(backend_text)
        return (
            f"{spec.label} split-sector DEM",
            f"non-default {spec.label} detector-side DEM benchmark built locally from the paper schedule",
        )
    if backend_text.startswith("rotated_surface_") or backend_text.startswith("surface_"):
        return (
            f"Rotated surface-code DEM ({backend_text})",
            f"rotated surface-code detector-side DEM benchmark for backend `{backend_text}`",
        )
    return (
        f"Detector-side DEM ({backend_text})",
        f"detector-side DEM benchmark for backend `{backend_text}`",
    )


def _external_benchmark_descriptor(*, backend: str, benchmark_label: str, stim_path: Path) -> tuple[str, str, str]:
    label = str(benchmark_label).strip() or f"External Stim DEM ({stim_path.name})"
    description = f"external detector-side DEM benchmark loaded from Stim circuit `{stim_path}` under backend label `{backend}`"
    source_note = f"{description}."
    return label, description, source_note


def _quantile(values: Sequence[float], q: float) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q, method="linear"))


def _series_total_int(values: Sequence[int]) -> int:
    if not values:
        return 0
    return int(np.sum(np.asarray(tuple(int(value) for value in values), dtype=np.int64)))


def _series_mean(values: Sequence[int]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(tuple(int(value) for value in values), dtype=np.float64)))


def _finite_row_values(rows: Sequence[dict[str, object]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key, float("nan")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(float(value))
    return values


def _finite_diagnostic_values(
    diagnostics: Sequence[progressive.ProgressiveForwardGuidanceDiagnostic],
    key: str,
) -> list[float]:
    values: list[float] = []
    for diagnostic in diagnostics:
        value = float(getattr(diagnostic, key))
        if math.isfinite(value):
            values.append(float(value))
    return values


def _diagnostic_mean(
    diagnostics: Sequence[progressive.ProgressiveForwardGuidanceDiagnostic],
    key: str,
) -> float:
    values = _finite_diagnostic_values(diagnostics, str(key))
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else float("nan")


def _diagnostic_max(
    diagnostics: Sequence[progressive.ProgressiveForwardGuidanceDiagnostic],
    key: str,
) -> float:
    values = _finite_diagnostic_values(diagnostics, str(key))
    return float(max(values)) if values else float("nan")


def _diagnostic_quantile(
    diagnostics: Sequence[progressive.ProgressiveForwardGuidanceDiagnostic],
    key: str,
    q: float,
) -> float:
    values = _finite_diagnostic_values(diagnostics, str(key))
    return _quantile(values, float(q)) if values else float("nan")


def _summarize_forward_guidance_diagnostics(
    result: progressive.ProgressiveDecodeResult,
) -> dict[str, object]:
    diagnostics = tuple(result.forward_guidance_diagnostics)
    empty: dict[str, object] = {
        "forward_guidance_diag_step_count": 0,
        "forward_guidance_triggered_step_count": 0,
        "forward_guidance_triggered_fraction": 0.0,
        "forward_guidance_top_gap_triggered_step_count": 0,
        "forward_guidance_top_gap_triggered_fraction": 0.0,
        "forward_guidance_support_aware_triggered_step_count": 0,
        "forward_guidance_support_aware_triggered_fraction": 0.0,
        "forward_guidance_base_top_primary_gap_mean": float("nan"),
        "forward_guidance_base_top_primary_gap_p10": float("nan"),
        "forward_guidance_base_top_primary_gap_p50": float("nan"),
        "forward_guidance_base_top_primary_gap_p90": float("nan"),
        "forward_guidance_alignment_metadata_step_count": 0,
        "forward_guidance_aligned_step_count": 0,
        "forward_guidance_no_alignment_step_count": 0,
        "forward_guidance_top_rank_changed_count": 0,
        "forward_guidance_top_rank_changed_fraction": 0.0,
        "forward_guidance_top_logical_changed_count": 0,
        "forward_guidance_top_logical_changed_fraction": 0.0,
        "forward_guidance_selected_distance_abs_mean": float("nan"),
        "forward_guidance_selected_distance_abs_max": float("nan"),
        "forward_guidance_selected_state_count_mean": float("nan"),
        "forward_guidance_candidate_interval_row_count_mean": float("nan"),
        "forward_guidance_candidate_snapshot_count_mean": float("nan"),
        "forward_guidance_positive_aligned_snapshot_count_mean": float("nan"),
        "forward_guidance_backward_active_row_count_mean": float("nan"),
        "forward_guidance_common_active_row_count_mean": float("nan"),
        "forward_guidance_aligned_row_count_mean": float("nan"),
        "forward_guidance_aligned_row_count_max": float("nan"),
        "forward_guidance_aligned_fraction_backward_mean": float("nan"),
        "forward_guidance_aligned_fraction_common_mean": float("nan"),
        "forward_guidance_middle_row_count_mean": float("nan"),
        "forward_guidance_overlap_row_count_mean": float("nan"),
        "forward_guidance_zero_support_row_count_mean": float("nan"),
        "forward_guidance_middle_row_fraction_common_mean": float("nan"),
        "forward_guidance_overlap_row_fraction_common_mean": float("nan"),
        "forward_guidance_projected_state_count_mean": float("nan"),
        "forward_guidance_projected_entropy_mean": float("nan"),
        "forward_guidance_projected_effective_support_mean": float("nan"),
        "forward_guidance_projected_top_logprob_mean": float("nan"),
        "forward_guidance_projected_logprob_gap_mean": float("nan"),
        "forward_guidance_candidate_state_count_total": 0,
        "forward_guidance_applied_state_count_total": 0,
        "forward_guidance_missing_mass_count_total": 0,
        "forward_guidance_clipped_state_count_total": 0,
        "forward_guidance_missing_mass_fraction": 0.0,
        "forward_guidance_clipped_fraction": 0.0,
        "forward_guidance_bonus_min_min": float("nan"),
        "forward_guidance_bonus_p10_mean": float("nan"),
        "forward_guidance_bonus_p50_mean": float("nan"),
        "forward_guidance_bonus_mean_mean": float("nan"),
        "forward_guidance_bonus_p90_mean": float("nan"),
        "forward_guidance_bonus_max_max": float("nan"),
        "forward_guidance_weighted_bonus_mean_mean": float("nan"),
        "forward_guidance_guided_top_base_rank_mean": float("nan"),
        "forward_guidance_guided_top_base_rank_p99": float("nan"),
        "forward_guidance_base_top_guided_rank_mean": float("nan"),
        "forward_guidance_base_top_guided_rank_p99": float("nan"),
        "forward_guidance_projected_info_bits_mean": float("nan"),
        "forward_guidance_conditional_shortlist_state_count_mean": float("nan"),
        "forward_guidance_conditional_lookup_radius_mean": float("nan"),
        "forward_guidance_conditional_finite_score_count_mean": float("nan"),
        "forward_guidance_conditional_exact_support_count_mean": float("nan"),
        "forward_guidance_conditional_neighborhood_support_count_mean": float("nan"),
        "forward_guidance_conditional_neighborhood_only_support_count_mean": float("nan"),
        "forward_guidance_conditional_missing_support_count_mean": float("nan"),
        "forward_guidance_conditional_positive_raw_info_count_mean": float("nan"),
        "forward_guidance_conditional_finite_outside_kept_count_mean": float("nan"),
        "forward_guidance_conditional_positive_outside_kept_count_mean": float("nan"),
        "forward_guidance_conditional_nearcut_outside_kept_count_mean": float("nan"),
        "forward_guidance_conditional_positive_nearcut_outside_kept_count_mean": float("nan"),
        "forward_guidance_conditional_missing_logical_class_outside_kept_count_mean": float("nan"),
        "forward_guidance_conditional_positive_bonus_count_mean": float("nan"),
        "forward_guidance_conditional_promoted_state_count_total": 0,
        "forward_guidance_conditional_demoted_state_count_total": 0,
        "forward_guidance_conditional_changed_kept_step_count": 0,
        "forward_guidance_conditional_changed_kept_fraction": 0.0,
        "forward_guidance_conditional_added_logical_class_count_total": 0,
        "forward_guidance_conditional_fallback_candidate_count_total": 0,
        "forward_guidance_conditional_fallback_candidate_count_mean": float("nan"),
        "forward_guidance_conditional_fallback_added_state_count_total": 0,
        "forward_guidance_conditional_fallback_added_logical_class_count_total": 0,
        "forward_guidance_conditional_raw_info_min_min": float("nan"),
        "forward_guidance_conditional_raw_info_p10_mean": float("nan"),
        "forward_guidance_conditional_raw_info_p50_mean": float("nan"),
        "forward_guidance_conditional_raw_info_mean_mean": float("nan"),
        "forward_guidance_conditional_raw_info_p90_mean": float("nan"),
        "forward_guidance_conditional_raw_info_max_max": float("nan"),
        "forward_guidance_conditional_bonus_max_max": float("nan"),
        "forward_guidance_checkpoint_available_step_count": 0,
        "forward_guidance_checkpoint_available_fraction": 0.0,
        "forward_guidance_checkpoint_key_count_mean": float("nan"),
        "forward_guidance_checkpoint_source_state_count_mean": float("nan"),
        "forward_guidance_checkpoint_mass_coverage_after_trim_mean": float("nan"),
        "forward_guidance_checkpoint_band_state_count_mean": float("nan"),
        "forward_guidance_checkpoint_hit_count_mean": float("nan"),
        "forward_guidance_checkpoint_hit_fraction_mean": float("nan"),
        "forward_guidance_checkpoint_rescue_budget_mean": float("nan"),
        "forward_guidance_checkpoint_rescued_state_count_total": 0,
        "forward_guidance_checkpoint_rescued_state_count_mean": float("nan"),
        "forward_guidance_checkpoint_replay_triggered_step_count": 0,
        "forward_guidance_checkpoint_replay_triggered_fraction": 0.0,
        "forward_guidance_checkpoint_replay_prior_available_step_count": 0,
        "forward_guidance_checkpoint_replay_prior_available_fraction": 0.0,
        "forward_guidance_checkpoint_replay_called_step_count": 0,
        "forward_guidance_checkpoint_replay_called_fraction": 0.0,
        "forward_guidance_checkpoint_replay_attempted_step_count": 0,
        "forward_guidance_checkpoint_replay_attempted_fraction": 0.0,
        "forward_guidance_checkpoint_replay_succeeded_step_count": 0,
        "forward_guidance_checkpoint_replay_succeeded_fraction": 0.0,
        "forward_guidance_checkpoint_replay_aborted_no_checkpoint_step_count": 0,
        "forward_guidance_checkpoint_replay_aborted_no_checkpoint_fraction": 0.0,
        "forward_guidance_checkpoint_replay_aborted_window_too_long_step_count": 0,
        "forward_guidance_checkpoint_replay_aborted_window_too_long_fraction": 0.0,
        "forward_guidance_checkpoint_replay_aborted_empty_query_set_step_count": 0,
        "forward_guidance_checkpoint_replay_aborted_empty_query_set_fraction": 0.0,
        "forward_guidance_checkpoint_replay_aborted_budget_cap_step_count": 0,
        "forward_guidance_checkpoint_replay_aborted_budget_cap_fraction": 0.0,
        "forward_guidance_checkpoint_replay_completed_step_count": 0,
        "forward_guidance_checkpoint_replay_completed_fraction": 0.0,
        "forward_guidance_checkpoint_replay_called_given_trigger_fraction": 0.0,
        "forward_guidance_checkpoint_replay_attempted_given_trigger_fraction": 0.0,
        "forward_guidance_checkpoint_replay_completed_given_called_fraction": 0.0,
        "forward_guidance_checkpoint_replay_succeeded_given_called_fraction": 0.0,
        "forward_guidance_checkpoint_replay_missing_target_snapshot_step_count": 0,
        "forward_guidance_checkpoint_replay_missing_target_snapshot_fraction": 0.0,
        "forward_guidance_checkpoint_replay_empty_target_state_log_mass_step_count": 0,
        "forward_guidance_checkpoint_replay_empty_target_state_log_mass_fraction": 0.0,
        "forward_guidance_checkpoint_replay_target_before_start_step_count": 0,
        "forward_guidance_checkpoint_replay_target_before_start_fraction": 0.0,
        "forward_guidance_checkpoint_replay_no_progress_to_next_boundary_step_count": 0,
        "forward_guidance_checkpoint_replay_no_progress_to_next_boundary_fraction": 0.0,
        "forward_guidance_checkpoint_replay_target_not_reached_step_count": 0,
        "forward_guidance_checkpoint_replay_target_not_reached_fraction": 0.0,
        "forward_guidance_checkpoint_replay_target_reached_step_count": 0,
        "forward_guidance_checkpoint_replay_target_reached_fraction": 0.0,
        "forward_guidance_checkpoint_replay_final_processed_columns_mean": float("nan"),
        "forward_guidance_checkpoint_replay_available_snapshot_count_mean": float("nan"),
        "forward_guidance_checkpoint_replay_target_snapshot_present_step_count": 0,
        "forward_guidance_checkpoint_replay_target_snapshot_present_fraction": 0.0,
        "forward_guidance_checkpoint_replay_target_snapshot_state_count_mean": float("nan"),
        "forward_guidance_checkpoint_replay_seed_key_count_mean": float("nan"),
        "forward_guidance_checkpoint_replay_generated_key_count_mean": float("nan"),
        "forward_guidance_checkpoint_replay_new_key_count_mean": float("nan"),
        "forward_guidance_checkpoint_replay_query_key_count_mean": float("nan"),
        "forward_guidance_checkpoint_replay_hit_key_count_mean": float("nan"),
        "forward_guidance_checkpoint_replay_hit_candidate_count_mean": float("nan"),
        "forward_guidance_checkpoint_query_hit_count_before_replay_mean": float("nan"),
        "forward_guidance_checkpoint_query_hit_count_after_replay_mean": float("nan"),
        "forward_guidance_checkpoint_query_new_hit_count_from_replay_mean": float("nan"),
        "forward_guidance_checkpoint_replay_expansion_count_mean": float("nan"),
        "forward_guidance_checkpoint_replay_max_frontier_size_mean": float("nan"),
        "forward_guidance_checkpoint_replay_terminal_state_count_mean": float("nan"),
        "forward_guidance_checkpoint_replay_replayed_column_count_mean": float("nan"),
        "forward_guidance_checkpoint_replay_budget_exhausted_step_count": 0,
        "forward_guidance_checkpoint_replay_budget_exhausted_fraction": 0.0,
        "forward_guidance_checkpoint_replay_generated_key_count_called_mean": float("nan"),
        "forward_guidance_checkpoint_replay_new_key_count_called_mean": float("nan"),
        "forward_guidance_checkpoint_query_new_hit_count_from_replay_called_mean": float("nan"),
        "forward_guidance_local_widen_eligible_step_count": 0,
        "forward_guidance_local_widen_triggered_step_count": 0,
        "forward_guidance_local_widen_triggered_fraction": 0.0,
        "forward_guidance_first_trigger_active_processed_columns": -1,
        "forward_guidance_first_local_widen_triggered_processed_columns": -1,
        "forward_guidance_local_widen_added_state_count_total": 0,
        "forward_guidance_local_widen_kept_count_mean": float("nan"),
        "forward_guidance_truth_cut_state_valid_step_count": 0,
        "forward_guidance_truth_cut_candidate_present_step_count": 0,
        "forward_guidance_truth_cut_candidate_present_fraction": 0.0,
        "forward_guidance_truth_cut_ordinary_kept_step_count": 0,
        "forward_guidance_truth_cut_ordinary_kept_fraction": 0.0,
        "forward_guidance_truth_cut_provisional_present_step_count": 0,
        "forward_guidance_truth_cut_provisional_present_fraction": 0.0,
        "forward_guidance_truth_cut_exact_supported_step_count": 0,
        "forward_guidance_truth_cut_exact_supported_fraction": 0.0,
        "forward_guidance_truth_cut_checkpoint_hit_before_replay_step_count": 0,
        "forward_guidance_truth_cut_checkpoint_hit_before_replay_fraction": 0.0,
        "forward_guidance_truth_cut_checkpoint_replay_queried_step_count": 0,
        "forward_guidance_truth_cut_checkpoint_replay_queried_fraction": 0.0,
        "forward_guidance_truth_cut_checkpoint_replay_hit_step_count": 0,
        "forward_guidance_truth_cut_checkpoint_replay_hit_fraction": 0.0,
        "forward_guidance_truth_cut_prev_checkpoint_exists_step_count": 0,
        "forward_guidance_truth_cut_prev_checkpoint_exists_fraction": 0.0,
        "forward_guidance_truth_cut_prev_checkpoint_ancestor_present_step_count": 0,
        "forward_guidance_truth_cut_prev_checkpoint_ancestor_present_fraction": 0.0,
        "forward_guidance_truth_cut_true_key_in_band_step_count": 0,
        "forward_guidance_truth_cut_true_key_in_band_fraction": 0.0,
        "forward_guidance_truth_cut_true_key_hit_before_replay_step_count": 0,
        "forward_guidance_truth_cut_true_key_hit_before_replay_fraction": 0.0,
        "forward_guidance_truth_cut_true_key_hit_after_replay_step_count": 0,
        "forward_guidance_truth_cut_true_key_hit_after_replay_fraction": 0.0,
        "forward_guidance_truth_cut_true_key_hit_only_via_replay_step_count": 0,
        "forward_guidance_truth_cut_true_key_hit_only_via_replay_fraction": 0.0,
        "forward_guidance_truth_cut_true_key_rescued_step_count": 0,
        "forward_guidance_truth_cut_true_key_rescued_fraction": 0.0,
        "forward_guidance_truth_cut_true_join_rank_in_band_mean": float("nan"),
        "forward_guidance_truth_cut_true_join_rank_in_band_p50": float("nan"),
        "forward_guidance_truth_cut_true_survives_next_prune_step_count": 0,
        "forward_guidance_truth_cut_true_survives_next_prune_fraction": 0.0,
        "forward_guidance_truth_cut_true_survives_next_prune_given_rescued_fraction": 0.0,
        "forward_guidance_truth_cut_true_survives_two_prunes_step_count": 0,
        "forward_guidance_truth_cut_true_survives_two_prunes_fraction": 0.0,
        "forward_guidance_truth_cut_true_survives_two_prunes_given_rescued_fraction": 0.0,
        "forward_guidance_truth_cut_rescued_wrong_class_above_truth_mean": float("nan"),
        "forward_guidance_truth_cut_rescued_wrong_class_above_truth_max": float("nan"),
        "forward_guidance_truth_cut_neighborhood_supported_step_count": 0,
        "forward_guidance_truth_cut_neighborhood_supported_fraction": 0.0,
        "forward_guidance_truth_cut_conditional_supported_step_count": 0,
        "forward_guidance_truth_cut_conditional_supported_fraction": 0.0,
        "forward_guidance_truth_cut_conditional_positive_step_count": 0,
        "forward_guidance_truth_cut_conditional_positive_fraction": 0.0,
        "forward_guidance_truth_cut_added_extra_step_count": 0,
        "forward_guidance_truth_cut_added_extra_fraction": 0.0,
        "forward_guidance_truth_cut_final_kept_step_count": 0,
        "forward_guidance_truth_cut_final_kept_fraction": 0.0,
        "forward_guidance_truth_cut_first_candidate_missing_processed_columns": -1,
        "forward_guidance_truth_cut_first_ordinary_missing_processed_columns": -1,
        "forward_guidance_truth_cut_first_provisional_missing_processed_columns": -1,
        "forward_guidance_truth_cut_first_final_missing_processed_columns": -1,
        "forward_guidance_truth_cut_first_added_extra_processed_columns": -1,
        "forward_guidance_truth_cut_first_ordinary_loss_trigger_active": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_local_widen_triggered": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_provisional_present": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_exact_supported": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_neighborhood_supported": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_conditional_supported": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_hit": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_hit_before_replay": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_queried": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_hit": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_prev_checkpoint_exists": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_prev_checkpoint_ancestor_present": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_base_rank": -1,
        "forward_guidance_truth_cut_first_ordinary_loss_rank_over_beam_size": -1,
        "forward_guidance_truth_cut_first_ordinary_loss_rank_over_ordinary_kept": -1,
        "forward_guidance_truth_cut_first_ordinary_loss_within_2k": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_within_3k": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_within_4k": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_within_2x_ordinary_kept": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_within_3x_ordinary_kept": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_within_4x_ordinary_kept": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_added_extra": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_available": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_key_count": -1,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_source_state_count": -1,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_mass_coverage_after_trim": float("nan"),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_band_state_count": -1,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_hit_count": -1,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_hit_fraction": float("nan"),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_rescue_budget": -1,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_rescued_state_count": -1,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_triggered": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_prior_available": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_called": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_attempted": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_succeeded": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_aborted_no_checkpoint": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_aborted_window_too_long": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_aborted_empty_query_set": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_aborted_budget_cap": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_completed": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_target_before_start": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_no_progress_to_next_boundary": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_target_not_reached": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_target_reached": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_final_processed_columns": -1,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_available_snapshot_count": -1,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_target_snapshot_present": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_target_snapshot_state_count": -1,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_seed_key_count": -1,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_generated_key_count": -1,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_new_key_count": -1,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_query_key_count": -1,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_hit_key_count": -1,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_hit_candidate_count": -1,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_query_hit_count_before_replay": -1,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_query_hit_count_after_replay": -1,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_query_new_hit_count_from_replay": -1,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_expansion_count": -1,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_max_frontier_size": -1,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_terminal_state_count": -1,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_replayed_column_count": -1,
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_budget_exhausted": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_has_checkpoint": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_true_query_key_in_band": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_true_key_hit_before_replay": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_true_key_hit_after_replay": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_true_key_hit_only_via_replay": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_true_key_rescued": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_true_prev_checkpoint_exists": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_true_prev_checkpoint_true_ancestor_key_present": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_true_join_rank_in_band": -1,
        "forward_guidance_truth_cut_first_ordinary_loss_true_survives_next_prune": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_true_survives_two_prunes": 0,
        "forward_guidance_truth_cut_first_ordinary_loss_rescued_wrong_class_above_truth": -1,
        "forward_guidance_truth_cut_raw_info_mean": float("nan"),
        "forward_guidance_truth_cut_raw_info_p50": float("nan"),
    }
    if not diagnostics:
        return empty
    step_count = int(len(diagnostics))

    def _count_where(predicate: Callable[[progressive.ProgressiveForwardGuidanceDiagnostic], bool]) -> int:
        return sum(int(bool(predicate(diagnostic))) for diagnostic in diagnostics)

    def _mean_where(
        predicate: Callable[[progressive.ProgressiveForwardGuidanceDiagnostic], bool],
        attribute: str,
    ) -> float:
        values: list[float] = []
        for diagnostic in diagnostics:
            if not bool(predicate(diagnostic)):
                continue
            value = getattr(diagnostic, str(attribute))
            if isinstance(value, bool):
                values.append(float(int(bool(value))))
            elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append(float(value))
        return float(np.mean(np.asarray(values, dtype=np.float64))) if values else float("nan")

    aligned_step_count = sum(int(int(diagnostic.aligned_row_count) > 0) for diagnostic in diagnostics)
    alignment_metadata_count = sum(
        int(bool(diagnostic.alignment_metadata_available)) for diagnostic in diagnostics
    )
    triggered_count = sum(int(bool(diagnostic.trigger_active)) for diagnostic in diagnostics)
    top_gap_triggered_count = sum(int(bool(diagnostic.top_gap_trigger_active)) for diagnostic in diagnostics)
    support_aware_triggered_count = sum(
        int(bool(diagnostic.support_aware_trigger_active)) for diagnostic in diagnostics
    )
    top_rank_changed_count = sum(int(bool(diagnostic.top_rank_changed)) for diagnostic in diagnostics)
    top_logical_changed_count = sum(int(bool(diagnostic.top_logical_changed)) for diagnostic in diagnostics)
    candidate_state_total = sum(int(diagnostic.guidance_candidate_state_count) for diagnostic in diagnostics)
    applied_state_total = sum(int(diagnostic.guidance_applied_state_count) for diagnostic in diagnostics)
    missing_mass_total = sum(int(diagnostic.guidance_missing_mass_count) for diagnostic in diagnostics)
    clipped_state_total = sum(int(diagnostic.guidance_clipped_state_count) for diagnostic in diagnostics)
    conditional_changed_count = sum(int(bool(diagnostic.conditional_changed_kept_set)) for diagnostic in diagnostics)
    checkpoint_available_count = sum(int(bool(diagnostic.checkpoint_available)) for diagnostic in diagnostics)
    checkpoint_rescued_state_total = sum(
        int(diagnostic.checkpoint_rescued_state_count) for diagnostic in diagnostics
    )
    checkpoint_replay_triggered_count = sum(
        int(bool(diagnostic.checkpoint_replay_triggered)) for diagnostic in diagnostics
    )
    checkpoint_replay_prior_available_count = sum(
        int(bool(diagnostic.checkpoint_replay_prior_available)) for diagnostic in diagnostics
    )
    checkpoint_replay_called_count = sum(
        int(bool(diagnostic.checkpoint_replay_called)) for diagnostic in diagnostics
    )
    checkpoint_replay_attempted_count = sum(
        int(bool(diagnostic.checkpoint_replay_attempted)) for diagnostic in diagnostics
    )
    checkpoint_replay_succeeded_count = sum(
        int(bool(diagnostic.checkpoint_replay_succeeded)) for diagnostic in diagnostics
    )
    checkpoint_replay_aborted_no_checkpoint_count = sum(
        int(bool(diagnostic.checkpoint_replay_aborted_no_checkpoint)) for diagnostic in diagnostics
    )
    checkpoint_replay_aborted_window_too_long_count = sum(
        int(bool(diagnostic.checkpoint_replay_aborted_window_too_long)) for diagnostic in diagnostics
    )
    checkpoint_replay_aborted_empty_query_set_count = sum(
        int(bool(diagnostic.checkpoint_replay_aborted_empty_query_set)) for diagnostic in diagnostics
    )
    checkpoint_replay_aborted_budget_cap_count = sum(
        int(bool(diagnostic.checkpoint_replay_aborted_budget_cap)) for diagnostic in diagnostics
    )
    checkpoint_replay_completed_count = sum(
        int(bool(diagnostic.checkpoint_replay_completed)) for diagnostic in diagnostics
    )
    checkpoint_replay_missing_target_snapshot_count = sum(
        int(str(diagnostic.checkpoint_replay_status) == "missing_target_snapshot")
        for diagnostic in diagnostics
    )
    checkpoint_replay_empty_target_state_log_mass_count = sum(
        int(str(diagnostic.checkpoint_replay_status) == "empty_target_state_log_mass")
        for diagnostic in diagnostics
    )
    checkpoint_replay_target_before_start_count = sum(
        int(str(diagnostic.checkpoint_replay_status) == "nonpositive_window")
        for diagnostic in diagnostics
    )
    checkpoint_replay_no_progress_to_next_boundary_count = sum(
        int(str(diagnostic.checkpoint_replay_status) == "no_progress_to_next_boundary")
        for diagnostic in diagnostics
    )
    checkpoint_replay_target_not_reached_count = sum(
        int(str(diagnostic.checkpoint_replay_status) == "target_not_reached")
        for diagnostic in diagnostics
    )
    checkpoint_replay_target_reached_count = sum(
        int(bool(diagnostic.checkpoint_replay_target_reached)) for diagnostic in diagnostics
    )
    checkpoint_replay_target_snapshot_present_count = sum(
        int(bool(diagnostic.checkpoint_replay_target_snapshot_present))
        for diagnostic in diagnostics
    )
    checkpoint_replay_budget_exhausted_count = sum(
        int(bool(diagnostic.checkpoint_replay_budget_exhausted)) for diagnostic in diagnostics
    )
    local_widen_eligible_count = sum(int(bool(diagnostic.local_widen_eligible)) for diagnostic in diagnostics)
    local_widen_triggered_count = sum(int(bool(diagnostic.local_widen_triggered)) for diagnostic in diagnostics)
    bonus_min_values = _finite_diagnostic_values(diagnostics, "bonus_min")
    bonus_max_values = _finite_diagnostic_values(diagnostics, "bonus_max")
    conditional_raw_min_values = _finite_diagnostic_values(diagnostics, "conditional_raw_info_min")
    conditional_raw_max_values = _finite_diagnostic_values(diagnostics, "conditional_raw_info_max")
    conditional_bonus_max_values = _finite_diagnostic_values(diagnostics, "conditional_bonus_max")
    selected_diagnostics = tuple(
        diagnostic for diagnostic in diagnostics if int(diagnostic.selected_forward_processed_columns) >= 0
    )
    truth_cut_diagnostics = tuple(
        diagnostic for diagnostic in diagnostics if int(diagnostic.truth_cut_state_key) >= 0
    )
    truth_cut_count = int(len(truth_cut_diagnostics))

    def _truth_cut_flag_count(attribute: str) -> int:
        return sum(int(bool(getattr(diagnostic, attribute))) for diagnostic in truth_cut_diagnostics)

    def _truth_cut_fraction(count: int) -> float:
        return float(count) / float(truth_cut_count) if int(truth_cut_count) > 0 else 0.0

    def _truth_cut_first_processed_columns(predicate: Callable[[progressive.ProgressiveForwardGuidanceDiagnostic], bool]) -> int:
        values = [
            int(diagnostic.processed_columns)
            for diagnostic in truth_cut_diagnostics
            if bool(predicate(diagnostic))
        ]
        return int(min(values)) if values else -1

    def _first_processed_columns(predicate: Callable[[progressive.ProgressiveForwardGuidanceDiagnostic], bool]) -> int:
        values = [
            int(diagnostic.processed_columns)
            for diagnostic in diagnostics
            if bool(predicate(diagnostic))
        ]
        return int(min(values)) if values else -1

    first_ordinary_loss_diagnostics = tuple(
        diagnostic
        for diagnostic in truth_cut_diagnostics
        if not bool(diagnostic.truth_cut_state_ordinary_kept)
    )
    first_ordinary_loss_diagnostic = (
        min(first_ordinary_loss_diagnostics, key=lambda diagnostic: int(diagnostic.processed_columns))
        if first_ordinary_loss_diagnostics
        else None
    )

    truth_candidate_count = _truth_cut_flag_count("truth_cut_state_candidate_present")
    truth_ordinary_count = _truth_cut_flag_count("truth_cut_state_ordinary_kept")
    truth_provisional_count = _truth_cut_flag_count("truth_cut_state_provisional_present")
    truth_exact_supported_count = _truth_cut_flag_count("truth_cut_state_exact_supported")
    truth_checkpoint_hit_before_replay_count = _truth_cut_flag_count(
        "truth_cut_state_checkpoint_hit_before_replay"
    )
    truth_checkpoint_replay_queried_count = _truth_cut_flag_count(
        "truth_cut_state_checkpoint_replay_queried"
    )
    truth_checkpoint_replay_hit_count = _truth_cut_flag_count("truth_cut_state_checkpoint_replay_hit")
    truth_prev_checkpoint_exists_count = _truth_cut_flag_count(
        "truth_cut_state_prev_checkpoint_exists"
    )
    truth_prev_checkpoint_ancestor_present_count = _truth_cut_flag_count(
        "truth_cut_state_prev_checkpoint_ancestor_present"
    )
    truth_checkpoint_in_band_count = _truth_cut_flag_count("truth_cut_state_checkpoint_in_band")
    truth_checkpoint_hit_after_replay_count = _truth_cut_flag_count(
        "truth_cut_state_checkpoint_hit_after_replay"
    )
    truth_checkpoint_hit_only_via_replay_count = _truth_cut_flag_count(
        "truth_cut_state_checkpoint_hit_only_via_replay"
    )
    truth_checkpoint_rescued_count = _truth_cut_flag_count(
        "truth_cut_state_checkpoint_rescued"
    )
    truth_join_rank_values = [
        int(diagnostic.truth_cut_state_checkpoint_join_rank_in_band)
        for diagnostic in truth_cut_diagnostics
        if int(diagnostic.truth_cut_state_checkpoint_join_rank_in_band) >= 0
    ]
    truth_wrong_class_above_values = [
        int(diagnostic.truth_cut_state_checkpoint_rescued_wrong_class_above_count)
        for diagnostic in truth_cut_diagnostics
        if bool(diagnostic.truth_cut_state_checkpoint_rescued)
        and int(diagnostic.truth_cut_state_checkpoint_rescued_wrong_class_above_count) >= 0
    ]
    truth_cut_by_processed_columns = {
        int(diagnostic.processed_columns): diagnostic for diagnostic in truth_cut_diagnostics
    }

    def _truth_cut_future_survival(
        diagnostic: progressive.ProgressiveForwardGuidanceDiagnostic,
        steps: int,
    ) -> bool:
        if not bool(diagnostic.truth_cut_state_final_kept):
            return False
        current_processed_columns = int(diagnostic.processed_columns)
        for step in range(1, int(steps) + 1):
            future = truth_cut_by_processed_columns.get(int(current_processed_columns) + int(step))
            if future is None or not bool(future.truth_cut_state_final_kept):
                return False
        return True

    truth_survives_next_prune_count = sum(
        int(
            bool(diagnostic.truth_cut_state_checkpoint_rescued)
            and _truth_cut_future_survival(diagnostic, 1)
        )
        for diagnostic in truth_cut_diagnostics
    )
    truth_survives_two_prunes_count = sum(
        int(
            bool(diagnostic.truth_cut_state_checkpoint_rescued)
            and _truth_cut_future_survival(diagnostic, 2)
        )
        for diagnostic in truth_cut_diagnostics
    )
    truth_neighborhood_supported_count = _truth_cut_flag_count("truth_cut_state_neighborhood_supported")
    truth_supported_count = _truth_cut_flag_count("truth_cut_state_conditional_supported")
    truth_positive_count = _truth_cut_flag_count("truth_cut_state_conditional_positive")
    truth_added_count = _truth_cut_flag_count("truth_cut_state_added_extra")
    truth_final_count = _truth_cut_flag_count("truth_cut_state_final_kept")
    first_ordinary_loss_true_survives_next_prune = (
        int(_truth_cut_future_survival(first_ordinary_loss_diagnostic, 1))
        if first_ordinary_loss_diagnostic is not None
        else 0
    )
    first_ordinary_loss_true_survives_two_prunes = (
        int(_truth_cut_future_survival(first_ordinary_loss_diagnostic, 2))
        if first_ordinary_loss_diagnostic is not None
        else 0
    )
    return {
        **empty,
        "forward_guidance_diag_step_count": int(step_count),
        "forward_guidance_triggered_step_count": int(triggered_count),
        "forward_guidance_triggered_fraction": float(triggered_count) / float(step_count),
        "forward_guidance_top_gap_triggered_step_count": int(top_gap_triggered_count),
        "forward_guidance_top_gap_triggered_fraction": float(top_gap_triggered_count) / float(step_count),
        "forward_guidance_support_aware_triggered_step_count": int(support_aware_triggered_count),
        "forward_guidance_support_aware_triggered_fraction": float(support_aware_triggered_count) / float(step_count),
        "forward_guidance_base_top_primary_gap_mean": _diagnostic_mean(
            diagnostics,
            "base_top_primary_gap",
        ),
        "forward_guidance_base_top_primary_gap_p10": _diagnostic_quantile(
            diagnostics,
            "base_top_primary_gap",
            0.10,
        ),
        "forward_guidance_base_top_primary_gap_p50": _diagnostic_quantile(
            diagnostics,
            "base_top_primary_gap",
            0.50,
        ),
        "forward_guidance_base_top_primary_gap_p90": _diagnostic_quantile(
            diagnostics,
            "base_top_primary_gap",
            0.90,
        ),
        "forward_guidance_alignment_metadata_step_count": int(alignment_metadata_count),
        "forward_guidance_aligned_step_count": int(aligned_step_count),
        "forward_guidance_no_alignment_step_count": int(step_count - aligned_step_count),
        "forward_guidance_top_rank_changed_count": int(top_rank_changed_count),
        "forward_guidance_top_rank_changed_fraction": float(top_rank_changed_count) / float(step_count),
        "forward_guidance_top_logical_changed_count": int(top_logical_changed_count),
        "forward_guidance_top_logical_changed_fraction": float(top_logical_changed_count) / float(step_count),
        "forward_guidance_selected_distance_abs_mean": _diagnostic_mean(
            selected_diagnostics,
            "selected_forward_distance",
        ),
        "forward_guidance_selected_distance_abs_max": _diagnostic_max(
            selected_diagnostics,
            "selected_forward_distance",
        ),
        "forward_guidance_selected_state_count_mean": _diagnostic_mean(
            selected_diagnostics,
            "selected_forward_state_count",
        ),
        "forward_guidance_candidate_interval_row_count_mean": _diagnostic_mean(
            diagnostics,
            "candidate_interval_row_count",
        ),
        "forward_guidance_candidate_snapshot_count_mean": _diagnostic_mean(
            diagnostics,
            "candidate_forward_snapshot_count",
        ),
        "forward_guidance_positive_aligned_snapshot_count_mean": _diagnostic_mean(
            diagnostics,
            "positive_aligned_snapshot_count",
        ),
        "forward_guidance_backward_active_row_count_mean": _diagnostic_mean(
            diagnostics,
            "backward_active_row_count",
        ),
        "forward_guidance_common_active_row_count_mean": _diagnostic_mean(
            diagnostics,
            "common_active_row_count",
        ),
        "forward_guidance_aligned_row_count_mean": _diagnostic_mean(
            diagnostics,
            "aligned_row_count",
        ),
        "forward_guidance_aligned_row_count_max": _diagnostic_max(
            diagnostics,
            "aligned_row_count",
        ),
        "forward_guidance_aligned_fraction_backward_mean": _diagnostic_mean(
            diagnostics,
            "aligned_row_fraction_of_backward_active",
        ),
        "forward_guidance_aligned_fraction_common_mean": _diagnostic_mean(
            diagnostics,
            "aligned_row_fraction_of_common_active",
        ),
        "forward_guidance_middle_row_count_mean": _diagnostic_mean(
            diagnostics,
            "middle_row_count",
        ),
        "forward_guidance_overlap_row_count_mean": _diagnostic_mean(
            diagnostics,
            "overlap_row_count",
        ),
        "forward_guidance_zero_support_row_count_mean": _diagnostic_mean(
            diagnostics,
            "zero_support_row_count",
        ),
        "forward_guidance_middle_row_fraction_common_mean": _diagnostic_mean(
            diagnostics,
            "middle_row_fraction_of_common_active",
        ),
        "forward_guidance_overlap_row_fraction_common_mean": _diagnostic_mean(
            diagnostics,
            "overlap_row_fraction_of_common_active",
        ),
        "forward_guidance_projected_state_count_mean": _diagnostic_mean(
            diagnostics,
            "projected_detector_state_count",
        ),
        "forward_guidance_projected_entropy_mean": _diagnostic_mean(
            diagnostics,
            "projected_entropy",
        ),
        "forward_guidance_projected_effective_support_mean": _diagnostic_mean(
            diagnostics,
            "projected_effective_support",
        ),
        "forward_guidance_projected_top_logprob_mean": _diagnostic_mean(
            diagnostics,
            "projected_top_logprob",
        ),
        "forward_guidance_projected_logprob_gap_mean": _diagnostic_mean(
            diagnostics,
            "projected_logprob_gap",
        ),
        "forward_guidance_candidate_state_count_total": int(candidate_state_total),
        "forward_guidance_applied_state_count_total": int(applied_state_total),
        "forward_guidance_missing_mass_count_total": int(missing_mass_total),
        "forward_guidance_clipped_state_count_total": int(clipped_state_total),
        "forward_guidance_missing_mass_fraction": (
            float(missing_mass_total) / float(applied_state_total)
            if int(applied_state_total) > 0
            else 0.0
        ),
        "forward_guidance_clipped_fraction": (
            float(clipped_state_total) / float(applied_state_total)
            if int(applied_state_total) > 0
            else 0.0
        ),
        "forward_guidance_bonus_min_min": float(min(bonus_min_values)) if bonus_min_values else float("nan"),
        "forward_guidance_bonus_p10_mean": _diagnostic_mean(diagnostics, "bonus_p10"),
        "forward_guidance_bonus_p50_mean": _diagnostic_mean(diagnostics, "bonus_p50"),
        "forward_guidance_bonus_mean_mean": _diagnostic_mean(diagnostics, "bonus_mean"),
        "forward_guidance_bonus_p90_mean": _diagnostic_mean(diagnostics, "bonus_p90"),
        "forward_guidance_bonus_max_max": float(max(bonus_max_values)) if bonus_max_values else float("nan"),
        "forward_guidance_weighted_bonus_mean_mean": _diagnostic_mean(
            diagnostics,
            "weighted_bonus_mean",
        ),
        "forward_guidance_guided_top_base_rank_mean": _diagnostic_mean(
            diagnostics,
            "guided_top_base_rank",
        ),
        "forward_guidance_guided_top_base_rank_p99": _diagnostic_quantile(
            diagnostics,
            "guided_top_base_rank",
            0.99,
        ),
        "forward_guidance_base_top_guided_rank_mean": _diagnostic_mean(
            diagnostics,
            "base_top_guided_rank",
        ),
        "forward_guidance_base_top_guided_rank_p99": _diagnostic_quantile(
            diagnostics,
            "base_top_guided_rank",
            0.99,
        ),
        "forward_guidance_projected_info_bits_mean": _diagnostic_mean(
            diagnostics,
            "projected_info_bits",
        ),
        "forward_guidance_conditional_shortlist_state_count_mean": _diagnostic_mean(
            diagnostics,
            "conditional_shortlist_state_count",
        ),
        "forward_guidance_conditional_lookup_radius_mean": _diagnostic_mean(
            diagnostics,
            "conditional_lookup_radius",
        ),
        "forward_guidance_conditional_finite_score_count_mean": _diagnostic_mean(
            diagnostics,
            "conditional_finite_score_count",
        ),
        "forward_guidance_conditional_exact_support_count_mean": _diagnostic_mean(
            diagnostics,
            "conditional_exact_support_count",
        ),
        "forward_guidance_conditional_neighborhood_support_count_mean": _diagnostic_mean(
            diagnostics,
            "conditional_neighborhood_support_count",
        ),
        "forward_guidance_conditional_neighborhood_only_support_count_mean": _diagnostic_mean(
            diagnostics,
            "conditional_neighborhood_only_support_count",
        ),
        "forward_guidance_conditional_missing_support_count_mean": _diagnostic_mean(
            diagnostics,
            "conditional_missing_support_count",
        ),
        "forward_guidance_conditional_positive_raw_info_count_mean": _diagnostic_mean(
            diagnostics,
            "conditional_positive_raw_info_count",
        ),
        "forward_guidance_conditional_finite_outside_kept_count_mean": _diagnostic_mean(
            diagnostics,
            "conditional_finite_outside_kept_count",
        ),
        "forward_guidance_conditional_positive_outside_kept_count_mean": _diagnostic_mean(
            diagnostics,
            "conditional_positive_outside_kept_count",
        ),
        "forward_guidance_conditional_nearcut_outside_kept_count_mean": _diagnostic_mean(
            diagnostics,
            "conditional_nearcut_outside_kept_count",
        ),
        "forward_guidance_conditional_positive_nearcut_outside_kept_count_mean": _diagnostic_mean(
            diagnostics,
            "conditional_positive_nearcut_outside_kept_count",
        ),
        "forward_guidance_conditional_missing_logical_class_outside_kept_count_mean": _diagnostic_mean(
            diagnostics,
            "conditional_missing_logical_class_outside_kept_count",
        ),
        "forward_guidance_conditional_positive_bonus_count_mean": _diagnostic_mean(
            diagnostics,
            "conditional_positive_bonus_count",
        ),
        "forward_guidance_conditional_promoted_state_count_total": sum(
            int(diagnostic.conditional_promoted_state_count) for diagnostic in diagnostics
        ),
        "forward_guidance_conditional_demoted_state_count_total": sum(
            int(diagnostic.conditional_demoted_state_count) for diagnostic in diagnostics
        ),
        "forward_guidance_conditional_changed_kept_step_count": int(conditional_changed_count),
        "forward_guidance_conditional_changed_kept_fraction": float(conditional_changed_count)
        / float(step_count),
        "forward_guidance_conditional_added_logical_class_count_total": sum(
            int(diagnostic.conditional_added_logical_class_count) for diagnostic in diagnostics
        ),
        "forward_guidance_conditional_fallback_candidate_count_total": sum(
            int(diagnostic.conditional_fallback_candidate_count) for diagnostic in diagnostics
        ),
        "forward_guidance_conditional_fallback_candidate_count_mean": _diagnostic_mean(
            diagnostics,
            "conditional_fallback_candidate_count",
        ),
        "forward_guidance_conditional_fallback_added_state_count_total": sum(
            int(diagnostic.conditional_fallback_added_state_count) for diagnostic in diagnostics
        ),
        "forward_guidance_conditional_fallback_added_logical_class_count_total": sum(
            int(diagnostic.conditional_fallback_added_logical_class_count)
            for diagnostic in diagnostics
        ),
        "forward_guidance_conditional_raw_info_min_min": (
            float(min(conditional_raw_min_values)) if conditional_raw_min_values else float("nan")
        ),
        "forward_guidance_conditional_raw_info_p10_mean": _diagnostic_mean(
            diagnostics,
            "conditional_raw_info_p10",
        ),
        "forward_guidance_conditional_raw_info_p50_mean": _diagnostic_mean(
            diagnostics,
            "conditional_raw_info_p50",
        ),
        "forward_guidance_conditional_raw_info_mean_mean": _diagnostic_mean(
            diagnostics,
            "conditional_raw_info_mean",
        ),
        "forward_guidance_conditional_raw_info_p90_mean": _diagnostic_mean(
            diagnostics,
            "conditional_raw_info_p90",
        ),
        "forward_guidance_conditional_raw_info_max_max": (
            float(max(conditional_raw_max_values)) if conditional_raw_max_values else float("nan")
        ),
        "forward_guidance_conditional_bonus_max_max": (
            float(max(conditional_bonus_max_values)) if conditional_bonus_max_values else float("nan")
        ),
        "forward_guidance_checkpoint_available_step_count": int(checkpoint_available_count),
        "forward_guidance_checkpoint_available_fraction": float(checkpoint_available_count)
        / float(step_count),
        "forward_guidance_checkpoint_key_count_mean": _diagnostic_mean(
            diagnostics,
            "checkpoint_key_count",
        ),
        "forward_guidance_checkpoint_source_state_count_mean": _diagnostic_mean(
            diagnostics,
            "checkpoint_source_state_count",
        ),
        "forward_guidance_checkpoint_mass_coverage_after_trim_mean": _diagnostic_mean(
            diagnostics,
            "checkpoint_mass_coverage_after_trim",
        ),
        "forward_guidance_checkpoint_band_state_count_mean": _diagnostic_mean(
            diagnostics,
            "checkpoint_band_state_count",
        ),
        "forward_guidance_checkpoint_hit_count_mean": _diagnostic_mean(
            diagnostics,
            "checkpoint_hit_count",
        ),
        "forward_guidance_checkpoint_hit_fraction_mean": _diagnostic_mean(
            diagnostics,
            "checkpoint_hit_fraction",
        ),
        "forward_guidance_checkpoint_rescue_budget_mean": _diagnostic_mean(
            diagnostics,
            "checkpoint_rescue_budget",
        ),
        "forward_guidance_checkpoint_rescued_state_count_total": int(checkpoint_rescued_state_total),
        "forward_guidance_checkpoint_rescued_state_count_mean": _diagnostic_mean(
            diagnostics,
            "checkpoint_rescued_state_count",
        ),
        "forward_guidance_checkpoint_replay_triggered_step_count": int(checkpoint_replay_triggered_count),
        "forward_guidance_checkpoint_replay_triggered_fraction": float(checkpoint_replay_triggered_count)
        / float(step_count),
        "forward_guidance_checkpoint_replay_prior_available_step_count": int(
            checkpoint_replay_prior_available_count
        ),
        "forward_guidance_checkpoint_replay_prior_available_fraction": float(
            checkpoint_replay_prior_available_count
        )
        / float(step_count),
        "forward_guidance_checkpoint_replay_called_step_count": int(checkpoint_replay_called_count),
        "forward_guidance_checkpoint_replay_called_fraction": float(checkpoint_replay_called_count)
        / float(step_count),
        "forward_guidance_checkpoint_replay_attempted_step_count": int(checkpoint_replay_attempted_count),
        "forward_guidance_checkpoint_replay_attempted_fraction": float(checkpoint_replay_attempted_count)
        / float(step_count),
        "forward_guidance_checkpoint_replay_succeeded_step_count": int(checkpoint_replay_succeeded_count),
        "forward_guidance_checkpoint_replay_succeeded_fraction": float(checkpoint_replay_succeeded_count)
        / float(step_count),
        "forward_guidance_checkpoint_replay_aborted_no_checkpoint_step_count": int(
            checkpoint_replay_aborted_no_checkpoint_count
        ),
        "forward_guidance_checkpoint_replay_aborted_no_checkpoint_fraction": float(
            checkpoint_replay_aborted_no_checkpoint_count
        )
        / float(step_count),
        "forward_guidance_checkpoint_replay_aborted_window_too_long_step_count": int(
            checkpoint_replay_aborted_window_too_long_count
        ),
        "forward_guidance_checkpoint_replay_aborted_window_too_long_fraction": float(
            checkpoint_replay_aborted_window_too_long_count
        )
        / float(step_count),
        "forward_guidance_checkpoint_replay_aborted_empty_query_set_step_count": int(
            checkpoint_replay_aborted_empty_query_set_count
        ),
        "forward_guidance_checkpoint_replay_aborted_empty_query_set_fraction": float(
            checkpoint_replay_aborted_empty_query_set_count
        )
        / float(step_count),
        "forward_guidance_checkpoint_replay_aborted_budget_cap_step_count": int(
            checkpoint_replay_aborted_budget_cap_count
        ),
        "forward_guidance_checkpoint_replay_aborted_budget_cap_fraction": float(
            checkpoint_replay_aborted_budget_cap_count
        )
        / float(step_count),
        "forward_guidance_checkpoint_replay_completed_step_count": int(
            checkpoint_replay_completed_count
        ),
        "forward_guidance_checkpoint_replay_completed_fraction": float(
            checkpoint_replay_completed_count
        )
        / float(step_count),
        "forward_guidance_checkpoint_replay_called_given_trigger_fraction": (
            float(checkpoint_replay_called_count) / float(checkpoint_replay_triggered_count)
            if int(checkpoint_replay_triggered_count) > 0
            else 0.0
        ),
        "forward_guidance_checkpoint_replay_attempted_given_trigger_fraction": (
            float(checkpoint_replay_attempted_count) / float(checkpoint_replay_triggered_count)
            if int(checkpoint_replay_triggered_count) > 0
            else 0.0
        ),
        "forward_guidance_checkpoint_replay_completed_given_called_fraction": (
            float(checkpoint_replay_completed_count) / float(checkpoint_replay_called_count)
            if int(checkpoint_replay_called_count) > 0
            else 0.0
        ),
        "forward_guidance_checkpoint_replay_succeeded_given_called_fraction": (
            float(checkpoint_replay_succeeded_count) / float(checkpoint_replay_called_count)
            if int(checkpoint_replay_called_count) > 0
            else 0.0
        ),
        "forward_guidance_checkpoint_replay_missing_target_snapshot_step_count": int(
            checkpoint_replay_missing_target_snapshot_count
        ),
        "forward_guidance_checkpoint_replay_missing_target_snapshot_fraction": float(
            checkpoint_replay_missing_target_snapshot_count
        )
        / float(step_count),
        "forward_guidance_checkpoint_replay_empty_target_state_log_mass_step_count": int(
            checkpoint_replay_empty_target_state_log_mass_count
        ),
        "forward_guidance_checkpoint_replay_empty_target_state_log_mass_fraction": float(
            checkpoint_replay_empty_target_state_log_mass_count
        )
        / float(step_count),
        "forward_guidance_checkpoint_replay_target_before_start_step_count": int(
            checkpoint_replay_target_before_start_count
        ),
        "forward_guidance_checkpoint_replay_target_before_start_fraction": float(
            checkpoint_replay_target_before_start_count
        )
        / float(step_count),
        "forward_guidance_checkpoint_replay_no_progress_to_next_boundary_step_count": int(
            checkpoint_replay_no_progress_to_next_boundary_count
        ),
        "forward_guidance_checkpoint_replay_no_progress_to_next_boundary_fraction": float(
            checkpoint_replay_no_progress_to_next_boundary_count
        )
        / float(step_count),
        "forward_guidance_checkpoint_replay_target_not_reached_step_count": int(
            checkpoint_replay_target_not_reached_count
        ),
        "forward_guidance_checkpoint_replay_target_not_reached_fraction": float(
            checkpoint_replay_target_not_reached_count
        )
        / float(step_count),
        "forward_guidance_checkpoint_replay_target_reached_step_count": int(
            checkpoint_replay_target_reached_count
        ),
        "forward_guidance_checkpoint_replay_target_reached_fraction": float(
            checkpoint_replay_target_reached_count
        )
        / float(step_count),
        "forward_guidance_checkpoint_replay_final_processed_columns_mean": _diagnostic_mean(
            diagnostics,
            "checkpoint_replay_final_processed_columns",
        ),
        "forward_guidance_checkpoint_replay_available_snapshot_count_mean": _diagnostic_mean(
            diagnostics,
            "checkpoint_replay_available_snapshot_count",
        ),
        "forward_guidance_checkpoint_replay_target_snapshot_present_step_count": int(
            checkpoint_replay_target_snapshot_present_count
        ),
        "forward_guidance_checkpoint_replay_target_snapshot_present_fraction": float(
            checkpoint_replay_target_snapshot_present_count
        )
        / float(step_count),
        "forward_guidance_checkpoint_replay_target_snapshot_state_count_mean": _diagnostic_mean(
            diagnostics,
            "checkpoint_replay_target_snapshot_state_count",
        ),
        "forward_guidance_checkpoint_replay_seed_key_count_mean": _diagnostic_mean(
            diagnostics,
            "checkpoint_replay_seed_key_count",
        ),
        "forward_guidance_checkpoint_replay_generated_key_count_mean": _diagnostic_mean(
            diagnostics,
            "checkpoint_replay_generated_key_count",
        ),
        "forward_guidance_checkpoint_replay_new_key_count_mean": _diagnostic_mean(
            diagnostics,
            "checkpoint_replay_new_key_count",
        ),
        "forward_guidance_checkpoint_replay_query_key_count_mean": _diagnostic_mean(
            diagnostics,
            "checkpoint_replay_query_key_count",
        ),
        "forward_guidance_checkpoint_replay_hit_key_count_mean": _diagnostic_mean(
            diagnostics,
            "checkpoint_replay_hit_key_count",
        ),
        "forward_guidance_checkpoint_replay_hit_candidate_count_mean": _diagnostic_mean(
            diagnostics,
            "checkpoint_replay_hit_candidate_count",
        ),
        "forward_guidance_checkpoint_query_hit_count_before_replay_mean": _diagnostic_mean(
            diagnostics,
            "checkpoint_query_hit_count_before_replay",
        ),
        "forward_guidance_checkpoint_query_hit_count_after_replay_mean": _diagnostic_mean(
            diagnostics,
            "checkpoint_query_hit_count_after_replay",
        ),
        "forward_guidance_checkpoint_query_new_hit_count_from_replay_mean": _diagnostic_mean(
            diagnostics,
            "checkpoint_query_new_hit_count_from_replay",
        ),
        "forward_guidance_checkpoint_replay_expansion_count_mean": _diagnostic_mean(
            diagnostics,
            "checkpoint_replay_expansion_count",
        ),
        "forward_guidance_checkpoint_replay_max_frontier_size_mean": _diagnostic_mean(
            diagnostics,
            "checkpoint_replay_max_frontier_size",
        ),
        "forward_guidance_checkpoint_replay_terminal_state_count_mean": _diagnostic_mean(
            diagnostics,
            "checkpoint_replay_terminal_state_count",
        ),
        "forward_guidance_checkpoint_replay_replayed_column_count_mean": _diagnostic_mean(
            diagnostics,
            "checkpoint_replay_replayed_column_count",
        ),
        "forward_guidance_checkpoint_replay_budget_exhausted_step_count": int(
            checkpoint_replay_budget_exhausted_count
        ),
        "forward_guidance_checkpoint_replay_budget_exhausted_fraction": float(
            checkpoint_replay_budget_exhausted_count
        )
        / float(step_count),
        "forward_guidance_checkpoint_replay_generated_key_count_called_mean": _mean_where(
            lambda diagnostic: bool(diagnostic.checkpoint_replay_called),
            "checkpoint_replay_generated_key_count",
        ),
        "forward_guidance_checkpoint_replay_new_key_count_called_mean": _mean_where(
            lambda diagnostic: bool(diagnostic.checkpoint_replay_called),
            "checkpoint_replay_new_key_count",
        ),
        "forward_guidance_checkpoint_query_new_hit_count_from_replay_called_mean": _mean_where(
            lambda diagnostic: bool(diagnostic.checkpoint_replay_called),
            "checkpoint_query_new_hit_count_from_replay",
        ),
        "forward_guidance_local_widen_eligible_step_count": int(local_widen_eligible_count),
        "forward_guidance_local_widen_triggered_step_count": int(local_widen_triggered_count),
        "forward_guidance_local_widen_triggered_fraction": float(local_widen_triggered_count)
        / float(step_count),
        "forward_guidance_first_trigger_active_processed_columns": _first_processed_columns(
            lambda diagnostic: bool(diagnostic.trigger_active)
        ),
        "forward_guidance_first_local_widen_triggered_processed_columns": _first_processed_columns(
            lambda diagnostic: bool(diagnostic.local_widen_triggered)
        ),
        "forward_guidance_local_widen_added_state_count_total": sum(
            int(diagnostic.local_widen_added_state_count) for diagnostic in diagnostics
        ),
        "forward_guidance_local_widen_kept_count_mean": _diagnostic_mean(
            diagnostics,
            "local_widen_kept_count",
        ),
        "forward_guidance_truth_cut_state_valid_step_count": int(truth_cut_count),
        "forward_guidance_truth_cut_candidate_present_step_count": int(truth_candidate_count),
        "forward_guidance_truth_cut_candidate_present_fraction": _truth_cut_fraction(int(truth_candidate_count)),
        "forward_guidance_truth_cut_ordinary_kept_step_count": int(truth_ordinary_count),
        "forward_guidance_truth_cut_ordinary_kept_fraction": _truth_cut_fraction(int(truth_ordinary_count)),
        "forward_guidance_truth_cut_provisional_present_step_count": int(truth_provisional_count),
        "forward_guidance_truth_cut_provisional_present_fraction": _truth_cut_fraction(int(truth_provisional_count)),
        "forward_guidance_truth_cut_exact_supported_step_count": int(truth_exact_supported_count),
        "forward_guidance_truth_cut_exact_supported_fraction": _truth_cut_fraction(int(truth_exact_supported_count)),
        "forward_guidance_truth_cut_checkpoint_hit_before_replay_step_count": int(
            truth_checkpoint_hit_before_replay_count
        ),
        "forward_guidance_truth_cut_checkpoint_hit_before_replay_fraction": _truth_cut_fraction(
            int(truth_checkpoint_hit_before_replay_count)
        ),
        "forward_guidance_truth_cut_checkpoint_replay_queried_step_count": int(
            truth_checkpoint_replay_queried_count
        ),
        "forward_guidance_truth_cut_checkpoint_replay_queried_fraction": _truth_cut_fraction(
            int(truth_checkpoint_replay_queried_count)
        ),
        "forward_guidance_truth_cut_checkpoint_replay_hit_step_count": int(truth_checkpoint_replay_hit_count),
        "forward_guidance_truth_cut_checkpoint_replay_hit_fraction": _truth_cut_fraction(
            int(truth_checkpoint_replay_hit_count)
        ),
        "forward_guidance_truth_cut_prev_checkpoint_exists_step_count": int(
            truth_prev_checkpoint_exists_count
        ),
        "forward_guidance_truth_cut_prev_checkpoint_exists_fraction": _truth_cut_fraction(
            int(truth_prev_checkpoint_exists_count)
        ),
        "forward_guidance_truth_cut_prev_checkpoint_ancestor_present_step_count": int(
            truth_prev_checkpoint_ancestor_present_count
        ),
        "forward_guidance_truth_cut_prev_checkpoint_ancestor_present_fraction": _truth_cut_fraction(
            int(truth_prev_checkpoint_ancestor_present_count)
        ),
        "forward_guidance_truth_cut_true_key_in_band_step_count": int(
            truth_checkpoint_in_band_count
        ),
        "forward_guidance_truth_cut_true_key_in_band_fraction": _truth_cut_fraction(
            int(truth_checkpoint_in_band_count)
        ),
        "forward_guidance_truth_cut_true_key_hit_before_replay_step_count": int(
            truth_checkpoint_hit_before_replay_count
        ),
        "forward_guidance_truth_cut_true_key_hit_before_replay_fraction": _truth_cut_fraction(
            int(truth_checkpoint_hit_before_replay_count)
        ),
        "forward_guidance_truth_cut_true_key_hit_after_replay_step_count": int(
            truth_checkpoint_hit_after_replay_count
        ),
        "forward_guidance_truth_cut_true_key_hit_after_replay_fraction": _truth_cut_fraction(
            int(truth_checkpoint_hit_after_replay_count)
        ),
        "forward_guidance_truth_cut_true_key_hit_only_via_replay_step_count": int(
            truth_checkpoint_hit_only_via_replay_count
        ),
        "forward_guidance_truth_cut_true_key_hit_only_via_replay_fraction": _truth_cut_fraction(
            int(truth_checkpoint_hit_only_via_replay_count)
        ),
        "forward_guidance_truth_cut_true_key_rescued_step_count": int(
            truth_checkpoint_rescued_count
        ),
        "forward_guidance_truth_cut_true_key_rescued_fraction": _truth_cut_fraction(
            int(truth_checkpoint_rescued_count)
        ),
        "forward_guidance_truth_cut_true_join_rank_in_band_mean": (
            float(np.mean(np.asarray(truth_join_rank_values, dtype=np.float64)))
            if truth_join_rank_values
            else float("nan")
        ),
        "forward_guidance_truth_cut_true_join_rank_in_band_p50": (
            float(np.quantile(np.asarray(truth_join_rank_values, dtype=np.float64), 0.50))
            if truth_join_rank_values
            else float("nan")
        ),
        "forward_guidance_truth_cut_true_survives_next_prune_step_count": int(
            truth_survives_next_prune_count
        ),
        "forward_guidance_truth_cut_true_survives_next_prune_fraction": _truth_cut_fraction(
            int(truth_survives_next_prune_count)
        ),
        "forward_guidance_truth_cut_true_survives_next_prune_given_rescued_fraction": (
            float(truth_survives_next_prune_count) / float(truth_checkpoint_rescued_count)
            if int(truth_checkpoint_rescued_count) > 0
            else 0.0
        ),
        "forward_guidance_truth_cut_true_survives_two_prunes_step_count": int(
            truth_survives_two_prunes_count
        ),
        "forward_guidance_truth_cut_true_survives_two_prunes_fraction": _truth_cut_fraction(
            int(truth_survives_two_prunes_count)
        ),
        "forward_guidance_truth_cut_true_survives_two_prunes_given_rescued_fraction": (
            float(truth_survives_two_prunes_count) / float(truth_checkpoint_rescued_count)
            if int(truth_checkpoint_rescued_count) > 0
            else 0.0
        ),
        "forward_guidance_truth_cut_rescued_wrong_class_above_truth_mean": (
            float(np.mean(np.asarray(truth_wrong_class_above_values, dtype=np.float64)))
            if truth_wrong_class_above_values
            else float("nan")
        ),
        "forward_guidance_truth_cut_rescued_wrong_class_above_truth_max": (
            float(max(truth_wrong_class_above_values))
            if truth_wrong_class_above_values
            else float("nan")
        ),
        "forward_guidance_truth_cut_neighborhood_supported_step_count": int(truth_neighborhood_supported_count),
        "forward_guidance_truth_cut_neighborhood_supported_fraction": _truth_cut_fraction(
            int(truth_neighborhood_supported_count)
        ),
        "forward_guidance_truth_cut_conditional_supported_step_count": int(truth_supported_count),
        "forward_guidance_truth_cut_conditional_supported_fraction": _truth_cut_fraction(int(truth_supported_count)),
        "forward_guidance_truth_cut_conditional_positive_step_count": int(truth_positive_count),
        "forward_guidance_truth_cut_conditional_positive_fraction": _truth_cut_fraction(int(truth_positive_count)),
        "forward_guidance_truth_cut_added_extra_step_count": int(truth_added_count),
        "forward_guidance_truth_cut_added_extra_fraction": _truth_cut_fraction(int(truth_added_count)),
        "forward_guidance_truth_cut_final_kept_step_count": int(truth_final_count),
        "forward_guidance_truth_cut_final_kept_fraction": _truth_cut_fraction(int(truth_final_count)),
        "forward_guidance_truth_cut_first_candidate_missing_processed_columns": _truth_cut_first_processed_columns(
            lambda diagnostic: not bool(diagnostic.truth_cut_state_candidate_present)
        ),
        "forward_guidance_truth_cut_first_ordinary_missing_processed_columns": _truth_cut_first_processed_columns(
            lambda diagnostic: not bool(diagnostic.truth_cut_state_ordinary_kept)
        ),
        "forward_guidance_truth_cut_first_provisional_missing_processed_columns": _truth_cut_first_processed_columns(
            lambda diagnostic: not bool(diagnostic.truth_cut_state_provisional_present)
        ),
        "forward_guidance_truth_cut_first_final_missing_processed_columns": _truth_cut_first_processed_columns(
            lambda diagnostic: not bool(diagnostic.truth_cut_state_final_kept)
        ),
        "forward_guidance_truth_cut_first_added_extra_processed_columns": _truth_cut_first_processed_columns(
            lambda diagnostic: bool(diagnostic.truth_cut_state_added_extra)
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_trigger_active": (
            int(bool(first_ordinary_loss_diagnostic.trigger_active))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_local_widen_triggered": (
            int(bool(first_ordinary_loss_diagnostic.local_widen_triggered))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_provisional_present": (
            int(bool(first_ordinary_loss_diagnostic.truth_cut_state_provisional_present))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_exact_supported": (
            int(bool(first_ordinary_loss_diagnostic.truth_cut_state_exact_supported))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_neighborhood_supported": (
            int(bool(first_ordinary_loss_diagnostic.truth_cut_state_neighborhood_supported))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_conditional_supported": (
            int(bool(first_ordinary_loss_diagnostic.truth_cut_state_conditional_supported))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_hit": (
            int(bool(first_ordinary_loss_diagnostic.truth_cut_state_checkpoint_hit))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_hit_before_replay": (
            int(bool(first_ordinary_loss_diagnostic.truth_cut_state_checkpoint_hit_before_replay))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_queried": (
            int(bool(first_ordinary_loss_diagnostic.truth_cut_state_checkpoint_replay_queried))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_hit": (
            int(bool(first_ordinary_loss_diagnostic.truth_cut_state_checkpoint_replay_hit))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_prev_checkpoint_exists": (
            int(bool(first_ordinary_loss_diagnostic.truth_cut_state_prev_checkpoint_exists))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_prev_checkpoint_ancestor_present": (
            int(bool(first_ordinary_loss_diagnostic.truth_cut_state_prev_checkpoint_ancestor_present))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_base_rank": (
            int(first_ordinary_loss_diagnostic.truth_cut_state_base_rank)
            if first_ordinary_loss_diagnostic is not None
            else -1
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_rank_over_beam_size": (
            int(first_ordinary_loss_diagnostic.truth_cut_state_rank_over_beam_size)
            if first_ordinary_loss_diagnostic is not None
            else -1
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_rank_over_ordinary_kept": (
            int(first_ordinary_loss_diagnostic.truth_cut_state_rank_over_ordinary_kept)
            if first_ordinary_loss_diagnostic is not None
            else -1
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_within_2k": (
            int(bool(first_ordinary_loss_diagnostic.truth_cut_state_within_2k))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_within_3k": (
            int(bool(first_ordinary_loss_diagnostic.truth_cut_state_within_3k))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_within_4k": (
            int(bool(first_ordinary_loss_diagnostic.truth_cut_state_within_4k))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_within_2x_ordinary_kept": (
            int(bool(first_ordinary_loss_diagnostic.truth_cut_state_within_2x_ordinary_kept))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_within_3x_ordinary_kept": (
            int(bool(first_ordinary_loss_diagnostic.truth_cut_state_within_3x_ordinary_kept))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_within_4x_ordinary_kept": (
            int(bool(first_ordinary_loss_diagnostic.truth_cut_state_within_4x_ordinary_kept))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_added_extra": (
            int(bool(first_ordinary_loss_diagnostic.truth_cut_state_added_extra))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_available": (
            int(bool(first_ordinary_loss_diagnostic.checkpoint_available))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_key_count": (
            int(first_ordinary_loss_diagnostic.checkpoint_key_count)
            if first_ordinary_loss_diagnostic is not None
            else -1
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_source_state_count": (
            int(first_ordinary_loss_diagnostic.checkpoint_source_state_count)
            if first_ordinary_loss_diagnostic is not None
            else -1
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_mass_coverage_after_trim": (
            float(first_ordinary_loss_diagnostic.checkpoint_mass_coverage_after_trim)
            if first_ordinary_loss_diagnostic is not None
            else float("nan")
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_band_state_count": (
            int(first_ordinary_loss_diagnostic.checkpoint_band_state_count)
            if first_ordinary_loss_diagnostic is not None
            else -1
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_hit_count": (
            int(first_ordinary_loss_diagnostic.checkpoint_hit_count)
            if first_ordinary_loss_diagnostic is not None
            else -1
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_hit_fraction": (
            float(first_ordinary_loss_diagnostic.checkpoint_hit_fraction)
            if first_ordinary_loss_diagnostic is not None
            else float("nan")
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_rescue_budget": (
            int(first_ordinary_loss_diagnostic.checkpoint_rescue_budget)
            if first_ordinary_loss_diagnostic is not None
            else -1
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_rescued_state_count": (
            int(first_ordinary_loss_diagnostic.checkpoint_rescued_state_count)
            if first_ordinary_loss_diagnostic is not None
            else -1
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_triggered": (
            int(bool(first_ordinary_loss_diagnostic.checkpoint_replay_triggered))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_prior_available": (
            int(bool(first_ordinary_loss_diagnostic.checkpoint_replay_prior_available))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_called": (
            int(bool(first_ordinary_loss_diagnostic.checkpoint_replay_called))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_attempted": (
            int(bool(first_ordinary_loss_diagnostic.checkpoint_replay_attempted))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_succeeded": (
            int(bool(first_ordinary_loss_diagnostic.checkpoint_replay_succeeded))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_aborted_no_checkpoint": (
            int(bool(first_ordinary_loss_diagnostic.checkpoint_replay_aborted_no_checkpoint))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_aborted_window_too_long": (
            int(bool(first_ordinary_loss_diagnostic.checkpoint_replay_aborted_window_too_long))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_aborted_empty_query_set": (
            int(bool(first_ordinary_loss_diagnostic.checkpoint_replay_aborted_empty_query_set))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_aborted_budget_cap": (
            int(bool(first_ordinary_loss_diagnostic.checkpoint_replay_aborted_budget_cap))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_completed": (
            int(bool(first_ordinary_loss_diagnostic.checkpoint_replay_completed))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_target_before_start": (
            int(str(first_ordinary_loss_diagnostic.checkpoint_replay_status) == "nonpositive_window")
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_no_progress_to_next_boundary": (
            int(str(first_ordinary_loss_diagnostic.checkpoint_replay_status) == "no_progress_to_next_boundary")
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_target_not_reached": (
            int(str(first_ordinary_loss_diagnostic.checkpoint_replay_status) == "target_not_reached")
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_target_reached": (
            int(bool(first_ordinary_loss_diagnostic.checkpoint_replay_target_reached))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_final_processed_columns": (
            int(first_ordinary_loss_diagnostic.checkpoint_replay_final_processed_columns)
            if first_ordinary_loss_diagnostic is not None
            else -1
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_available_snapshot_count": (
            int(first_ordinary_loss_diagnostic.checkpoint_replay_available_snapshot_count)
            if first_ordinary_loss_diagnostic is not None
            else -1
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_target_snapshot_present": (
            int(bool(first_ordinary_loss_diagnostic.checkpoint_replay_target_snapshot_present))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_target_snapshot_state_count": (
            int(first_ordinary_loss_diagnostic.checkpoint_replay_target_snapshot_state_count)
            if first_ordinary_loss_diagnostic is not None
            else -1
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_seed_key_count": (
            int(first_ordinary_loss_diagnostic.checkpoint_replay_seed_key_count)
            if first_ordinary_loss_diagnostic is not None
            else -1
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_generated_key_count": (
            int(first_ordinary_loss_diagnostic.checkpoint_replay_generated_key_count)
            if first_ordinary_loss_diagnostic is not None
            else -1
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_new_key_count": (
            int(first_ordinary_loss_diagnostic.checkpoint_replay_new_key_count)
            if first_ordinary_loss_diagnostic is not None
            else -1
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_query_key_count": (
            int(first_ordinary_loss_diagnostic.checkpoint_replay_query_key_count)
            if first_ordinary_loss_diagnostic is not None
            else -1
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_hit_key_count": (
            int(first_ordinary_loss_diagnostic.checkpoint_replay_hit_key_count)
            if first_ordinary_loss_diagnostic is not None
            else -1
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_hit_candidate_count": (
            int(first_ordinary_loss_diagnostic.checkpoint_replay_hit_candidate_count)
            if first_ordinary_loss_diagnostic is not None
            else -1
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_query_hit_count_before_replay": (
            int(first_ordinary_loss_diagnostic.checkpoint_query_hit_count_before_replay)
            if first_ordinary_loss_diagnostic is not None
            else -1
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_query_hit_count_after_replay": (
            int(first_ordinary_loss_diagnostic.checkpoint_query_hit_count_after_replay)
            if first_ordinary_loss_diagnostic is not None
            else -1
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_query_new_hit_count_from_replay": (
            int(first_ordinary_loss_diagnostic.checkpoint_query_new_hit_count_from_replay)
            if first_ordinary_loss_diagnostic is not None
            else -1
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_expansion_count": (
            int(first_ordinary_loss_diagnostic.checkpoint_replay_expansion_count)
            if first_ordinary_loss_diagnostic is not None
            else -1
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_max_frontier_size": (
            int(first_ordinary_loss_diagnostic.checkpoint_replay_max_frontier_size)
            if first_ordinary_loss_diagnostic is not None
            else -1
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_terminal_state_count": (
            int(first_ordinary_loss_diagnostic.checkpoint_replay_terminal_state_count)
            if first_ordinary_loss_diagnostic is not None
            else -1
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_replayed_column_count": (
            int(first_ordinary_loss_diagnostic.checkpoint_replay_replayed_column_count)
            if first_ordinary_loss_diagnostic is not None
            else -1
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_budget_exhausted": (
            int(bool(first_ordinary_loss_diagnostic.checkpoint_replay_budget_exhausted))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_has_checkpoint": (
            int(bool(first_ordinary_loss_diagnostic.checkpoint_available))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_true_query_key_in_band": (
            int(bool(first_ordinary_loss_diagnostic.truth_cut_state_checkpoint_in_band))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_true_key_hit_before_replay": (
            int(bool(first_ordinary_loss_diagnostic.truth_cut_state_checkpoint_hit_before_replay))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_true_key_hit_after_replay": (
            int(bool(first_ordinary_loss_diagnostic.truth_cut_state_checkpoint_hit_after_replay))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_true_key_hit_only_via_replay": (
            int(bool(first_ordinary_loss_diagnostic.truth_cut_state_checkpoint_hit_only_via_replay))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_true_key_rescued": (
            int(bool(first_ordinary_loss_diagnostic.truth_cut_state_checkpoint_rescued))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_true_prev_checkpoint_exists": (
            int(bool(first_ordinary_loss_diagnostic.truth_cut_state_prev_checkpoint_exists))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_true_prev_checkpoint_true_ancestor_key_present": (
            int(bool(first_ordinary_loss_diagnostic.truth_cut_state_prev_checkpoint_ancestor_present))
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_true_join_rank_in_band": (
            int(first_ordinary_loss_diagnostic.truth_cut_state_checkpoint_join_rank_in_band)
            if first_ordinary_loss_diagnostic is not None
            else -1
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_true_survives_next_prune": (
            int(first_ordinary_loss_true_survives_next_prune)
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_true_survives_two_prunes": (
            int(first_ordinary_loss_true_survives_two_prunes)
            if first_ordinary_loss_diagnostic is not None
            else 0
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_rescued_wrong_class_above_truth": (
            int(first_ordinary_loss_diagnostic.truth_cut_state_checkpoint_rescued_wrong_class_above_count)
            if first_ordinary_loss_diagnostic is not None
            else -1
        ),
        "forward_guidance_truth_cut_raw_info_mean": _diagnostic_mean(
            truth_cut_diagnostics,
            "truth_cut_state_raw_info",
        ),
        "forward_guidance_truth_cut_raw_info_p50": _diagnostic_quantile(
            truth_cut_diagnostics,
            "truth_cut_state_raw_info",
            0.50,
        ),
    }


def _plot_floor_for_values(values: Sequence[float], *, default: float = 1.0e-12) -> float:
    finite_positive = [float(value) for value in values if math.isfinite(float(value)) and float(value) > 0.0]
    if not finite_positive:
        return float(default)
    smallest = min(float(value) for value in finite_positive)
    return float(max(float(default), 0.5 * float(smallest)))


def _normalize_decoder_mode(decoder_mode: str) -> str:
    mode = str(decoder_mode).strip().lower()
    if mode in {"", "forward", "standard_dem_progressive"}:
        return "forward"
    if mode in {"backward", "reverse", "reverse_only"}:
        return "backward"
    if mode in {"bidirectional", "bidirectional_committee", "committee"}:
        return "bidirectional_committee"
    if mode in {"forward_guided_backward", "guided_backward", "forward_informed_backward"}:
        return "forward_guided_backward"
    if mode in {"bidirectional_middle_join", "middle_join", "mitm", "bidirectional_meet_in_middle"}:
        return "bidirectional_middle_join"
    raise ValueError(f"unknown decoder mode: {decoder_mode!r}")


def _default_backward_column_order_label(decoder_mode: str) -> str:
    if _normalize_decoder_mode(str(decoder_mode)) in {
        "backward",
        "bidirectional_committee",
        "bidirectional_middle_join",
        "forward_guided_backward",
    }:
        return "reverse_forward_columns"
    return ""


def _write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fieldnames_from_rows(rows: Sequence[dict[str, object]], fallback: Sequence[str]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            key_text = str(key)
            if key_text in seen:
                continue
            seen.add(key_text)
            fieldnames.append(key_text)
    if fieldnames:
        return fieldnames
    return [str(key) for key in fallback]


def _extend_fieldnames_with_row_keys(
    preferred_fieldnames: Sequence[str], rows: Sequence[dict[str, object]]
) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for key in preferred_fieldnames:
        key_text = str(key)
        if key_text in seen:
            continue
        seen.add(key_text)
        fieldnames.append(key_text)
    for key in _fieldnames_from_rows(rows, []):
        if key in seen:
            continue
        seen.add(key)
        fieldnames.append(key)
    return fieldnames


def _append_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() and path.stat().st_size > 0 else "w"
    with path.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        if mode == "w":
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_compact(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _json_safe_float(value: float | None) -> float | None:
    if value is None:
        return None
    value_f = float(value)
    if not math.isfinite(value_f):
        return None
    return float(value_f)


def _logaddexp_many(values: Sequence[float]) -> float:
    if not values:
        return float("-inf")
    arr = np.asarray(list(values), dtype=np.float64)
    return float(np.logaddexp.reduce(arr))


def _terminal_selector_lambda_key(value: float) -> str:
    return f"lambda_{str(float(value)).replace('.', 'p').replace('-', 'm')}"


def _serialize_terminal_selector_signals(
    result: progressive.ProgressiveDecodeResult,
) -> str:
    summaries = tuple(result.terminal_logical_class_summaries)
    if not summaries:
        return ""
    entries_by_logical = {
        int(logical_mask): tuple(
            (float(log_mass), int(rep_cost))
            for log_mass, rep_cost in tuple(entries)
        )
        for logical_mask, entries in tuple(result.terminal_state_log_mass_rep_cost_by_logical_items)
    }
    summary_records = {
        int(summary.logical_mask): {
            "logical_mask": int(summary.logical_mask),
            "posterior": (
                _json_safe_float(result.logical_posteriors[int(summary.logical_mask)])
                if 0 <= int(summary.logical_mask) < len(tuple(result.logical_posteriors))
                else None
            ),
            "log_mass": float(summary.log_mass),
            "best_viterbi": float(summary.best_viterbi),
            "representative_cost": float(summary.representative_cost),
            "log_mass_minus_best_viterbi": _json_safe_float(
                float(summary.log_mass) - float(summary.best_viterbi)
            ),
            "state_count": int(len(entries_by_logical.get(int(summary.logical_mask), tuple()))),
        }
        for summary in summaries
    }
    for logical_mask, record in summary_records.items():
        entries = tuple(entries_by_logical.get(int(logical_mask), tuple()))
        sorted_entries = tuple(
            sorted(
                entries,
                key=lambda item: (-float(item[0]), int(item[1])),
            )
        )
        record["top_state_log_mass"] = (
            _json_safe_float(sorted_entries[0][0]) if sorted_entries else None
        )
        record["runnerup_state_log_mass"] = (
            _json_safe_float(sorted_entries[1][0]) if len(sorted_entries) >= 2 else None
        )
        record["top_state_rep_cost"] = (
            int(sorted_entries[0][1]) if sorted_entries else None
        )
        record["min_state_rep_cost"] = (
            min(int(rep_cost) for _log_mass, rep_cost in entries) if entries else None
        )
        record["max_state_rep_cost"] = (
            max(int(rep_cost) for _log_mass, rep_cost in entries) if entries else None
        )
        for lambda_value in TERMINAL_SELECTOR_COST_TILT_LAMBDAS:
            lambda_key = _terminal_selector_lambda_key(float(lambda_value))
            tilted_value = _logaddexp_many(
                [
                    float(log_mass) - float(lambda_value) * float(rep_cost)
                    for log_mass, rep_cost in entries
                ]
            )
            record[f"cost_tilted_log_mass_{lambda_key}"] = _json_safe_float(tilted_value)

    def _assign_ranks(
        key_fn: Callable[[dict[str, object]], object],
        field_name: str,
        *,
        reverse: bool,
    ) -> None:
        ordered = sorted(
            summary_records.values(),
            key=lambda record: (
                key_fn(record),
                -int(record["logical_mask"]) if reverse else int(record["logical_mask"]),
            ),
            reverse=bool(reverse),
        )
        for rank, record in enumerate(ordered, start=1):
            record[field_name] = int(rank)

    _assign_ranks(lambda record: float(record["log_mass"]), "log_mass_rank", reverse=True)
    _assign_ranks(lambda record: float(record["best_viterbi"]), "best_viterbi_rank", reverse=True)
    _assign_ranks(
        lambda record: -float(record["representative_cost"]),
        "representative_cost_rank",
        reverse=True,
    )
    for lambda_value in TERMINAL_SELECTOR_COST_TILT_LAMBDAS:
        lambda_key = _terminal_selector_lambda_key(float(lambda_value))
        _assign_ranks(
            lambda record, field=f"cost_tilted_log_mass_{lambda_key}": (
                float("-inf")
                if record[field] is None
                else float(record[field])
            ),
            f"cost_tilted_rank_{lambda_key}",
            reverse=True,
        )

    payload = {
        "schema_version": 1,
        "logical_hat": int(result.logical_hat),
        "final_logical_select_mode": str(result.final_logical_select_mode),
        "final_logical_select_rep_cost_weight": float(result.final_logical_select_rep_cost_weight),
        "final_logical_select_max_log_mass_gap": _json_safe_float(
            float(result.final_logical_select_max_log_mass_gap)
        ),
        "final_logical_select_rank2_viterbi_tolerance": float(
            result.final_logical_select_rank2_viterbi_tolerance
        ),
        "final_logical_select_base_logical": int(result.final_logical_select_base_logical),
        "final_logical_select_gate_triggered": bool(result.final_logical_select_gate_triggered),
        "terminal_top_log_mass_gap": _json_safe_float(float(result.terminal_top_log_mass_gap)),
        "logical_log_mass_items": [
            [int(logical_mask), float(log_mass)]
            for logical_mask, log_mass in tuple(result.terminal_logical_log_mass_items)
        ],
        "correction_log_mass_items": [
            [int(correction_mask), float(log_mass)]
            for correction_mask, log_mass in tuple(result.terminal_correction_log_mass_items)
        ],
        "representative_correction_by_logical_items": [
            [int(logical_mask), int(correction_mask)]
            for logical_mask, correction_mask in tuple(
                result.terminal_representative_correction_by_logical_items
            )
        ],
        "state_log_mass_rep_cost_by_logical": [
            {
                "logical_mask": int(logical_mask),
                "entries": [
                    [float(log_mass), int(rep_cost)]
                    for log_mass, rep_cost in tuple(entries_by_logical.get(int(logical_mask), tuple()))
                ],
            }
            for logical_mask in sorted(summary_records)
        ],
        "class_summaries": [
            summary_records[int(logical_mask)]
            for logical_mask in sorted(
                summary_records,
                key=lambda logical_mask: int(summary_records[int(logical_mask)]["log_mass_rank"]),
            )
        ],
    }
    return _json_compact(payload)


def _serialize_splice_rerank_summary(
    summary: progressive.LogicalSpliceRerankSummary | None,
) -> str:
    if summary is None:
        return ""
    payload = {
        "candidate_logicals": [int(value) for value in tuple(summary.candidate_logicals)],
        "aggregate": str(summary.aggregate_mode),
        "selected_logical": (
            None if summary.selected_logical_mask is None else int(summary.selected_logical_mask)
        ),
        "baseline_logical": int(summary.baseline_logical_mask),
        "selected_score": _json_safe_float(float(summary.selected_score)),
        "baseline_score": _json_safe_float(float(summary.baseline_score)),
        "finite_cut_fraction": _json_safe_float(float(summary.finite_cut_fraction)),
        "missing_support_fraction": _json_safe_float(float(summary.missing_support_fraction)),
        "per_logical_aggregate_items": [
            [int(logical_mask), _json_safe_float(float(score)), int(missing_count)]
            for logical_mask, score, missing_count in tuple(summary.per_logical_aggregate_items)
        ],
        "cuts": [
            {
                "prefix_columns": int(cut.prefix_columns),
                "suffix_columns": int(cut.suffix_columns),
                "boundary_rows": int(cut.boundary_rows),
                "finite_candidate_score_count": int(cut.finite_candidate_score_count),
                "compatible_splice_hit_count": int(cut.compatible_splice_hit_count),
                "missing_support_lookup_count": int(cut.missing_support_lookup_count),
                "selected_logical": cut.selected_logical_mask,
                "selected_score": _json_safe_float(float(cut.selected_score)),
                "class_score_items": [
                    [int(logical_mask), _json_safe_float(float(score))]
                    for logical_mask, score in tuple(cut.class_score_items)
                ],
            }
            for cut in tuple(summary.cut_summaries)
        ],
    }
    return _json_compact(payload)


def _bitmask_from_indices(indices: Sequence[int]) -> int:
    out = 0
    for index in indices:
        out |= 1 << int(index)
    return int(out)


def _log_probs_from_probs(values: Sequence[float]) -> tuple[float, ...]:
    return tuple(float("-inf") if float(value) <= 0.0 else float(math.log(float(value))) for value in values)


def _compact_float_slug(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _beam_score_gap_threshold_enabled(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value)) and float(value) > 0.0


def _format_beam_score_gap_threshold(value: object) -> str:
    if value in {"", None}:
        return ""
    value_f = float(value)
    if not math.isfinite(value_f) or value_f <= 0.0:
        return ""
    return f"{value_f:g}"


def _format_forward_guidance_snapshot_gap(value: object) -> str:
    if value in {"", None}:
        return ""
    value_f = float(value)
    if math.isnan(float(value_f)):
        return ""
    if not math.isfinite(float(value_f)):
        return "off"
    return f"{value_f:g}"


def _forward_guidance_snapshot_gap_slug(value: object) -> str:
    text = _format_forward_guidance_snapshot_gap(value)
    if not text:
        return ""
    if str(text) == "off":
        return "fgsgapoff"
    return f"fgsgap{_compact_float_slug(float(text))}"


def _optional_float(value: object) -> float | None:
    if value in {"", None}:
        return None
    value_f = float(value)
    if math.isnan(value_f):
        return None
    return float(value_f)


def _beam_score_gap_policy_enabled(policy: progressive.BeamScoreGapPolicy | None) -> bool:
    return policy is not None


def _build_beam_score_gap_policy(
    *,
    mode: object,
    base_threshold: object,
    final_threshold: object,
    slope: object,
    reference_count: object,
    min_threshold: object,
    max_threshold: object,
) -> progressive.BeamScoreGapPolicy | None:
    mode_key = str(mode).strip().lower()
    if mode_key in {"", "none", "disabled"}:
        return None
    allowed_modes = {"linear_columns", "active_log", "candidate_log"}
    if mode_key not in allowed_modes:
        raise ValueError(
            "--beam-score-gap-policy-mode must be one of "
            f"{sorted(allowed_modes)}"
        )
    base_value = _optional_float(base_threshold)
    if base_value is None:
        raise ValueError("--beam-score-gap-policy-base-threshold is required when adaptive score-gap policy is enabled")
    final_value = base_value
    slope_value = 0.0
    reference_value = 1.0
    if mode_key == "linear_columns":
        final_value_opt = _optional_float(final_threshold)
        if final_value_opt is None:
            raise ValueError("--beam-score-gap-policy-final-threshold is required for linear_columns")
        final_value = float(final_value_opt)
    else:
        slope_value_opt = _optional_float(slope)
        reference_value_opt = _optional_float(reference_count)
        if slope_value_opt is None:
            raise ValueError(f"--beam-score-gap-policy-slope is required for {mode_key}")
        if reference_value_opt is None:
            raise ValueError(f"--beam-score-gap-policy-reference-count is required for {mode_key}")
        slope_value = float(slope_value_opt)
        reference_value = float(reference_value_opt)
    default_min = min(float(base_value), float(final_value))
    default_max = max(float(base_value), float(final_value))
    min_value = default_min if _optional_float(min_threshold) is None else float(_optional_float(min_threshold))
    max_value = default_max if _optional_float(max_threshold) is None else float(_optional_float(max_threshold))
    return progressive.BeamScoreGapPolicy(
        mode=str(mode_key),
        base_threshold=float(base_value),
        final_threshold=float(final_value),
        slope=float(slope_value),
        reference_count=float(reference_value),
        min_threshold=float(min_value),
        max_threshold=float(max_value),
    )


def _beam_score_gap_policy_from_row(row: dict[str, object]) -> progressive.BeamScoreGapPolicy | None:
    return _build_beam_score_gap_policy(
        mode=row.get("beam_score_gap_policy_mode", ""),
        base_threshold=row.get("beam_score_gap_policy_base_threshold", ""),
        final_threshold=row.get("beam_score_gap_policy_final_threshold", ""),
        slope=row.get("beam_score_gap_policy_slope", ""),
        reference_count=row.get("beam_score_gap_policy_reference_count", ""),
        min_threshold=row.get("beam_score_gap_policy_min_threshold", ""),
        max_threshold=row.get("beam_score_gap_policy_max_threshold", ""),
    )


def _format_beam_score_gap_policy(policy: progressive.BeamScoreGapPolicy | None) -> str:
    if policy is None:
        return ""
    mode = str(policy.mode)
    if mode == "linear_columns":
        return (
            f"{mode}[{float(policy.base_threshold):g}->{float(policy.final_threshold):g}, "
            f"clip={float(policy.min_threshold):g}..{float(policy.max_threshold):g}]"
        )
    return (
        f"{mode}[b={float(policy.base_threshold):g},s={float(policy.slope):g},"
        f"ref={float(policy.reference_count):g},clip={float(policy.min_threshold):g}.."
        f"{float(policy.max_threshold):g}]"
    )


def _beam_score_gap_policy_slug(policy: progressive.BeamScoreGapPolicy | None) -> str:
    if policy is None:
        return ""
    mode = str(policy.mode)
    if mode == "linear_columns":
        return (
            f"sgp_{mode}_b{_compact_float_slug(float(policy.base_threshold))}"
            f"_f{_compact_float_slug(float(policy.final_threshold))}"
            f"_m{_compact_float_slug(float(policy.min_threshold))}"
            f"_M{_compact_float_slug(float(policy.max_threshold))}"
        )
    return (
        f"sgp_{mode}_b{_compact_float_slug(float(policy.base_threshold))}"
        f"_s{_compact_float_slug(float(policy.slope))}"
        f"_r{_compact_float_slug(float(policy.reference_count))}"
        f"_m{_compact_float_slug(float(policy.min_threshold))}"
        f"_M{_compact_float_slug(float(policy.max_threshold))}"
    )


def _format_beam_score_gap_control(
    *,
    beam_score_gap_threshold: float | None,
    beam_score_gap_policy: progressive.BeamScoreGapPolicy | None,
) -> str:
    policy_text = _format_beam_score_gap_policy(beam_score_gap_policy)
    if policy_text:
        return policy_text
    threshold_text = _format_beam_score_gap_threshold(beam_score_gap_threshold)
    return threshold_text if threshold_text else "disabled"


def _selective_secondary_enabled(
    *,
    selective_secondary_score_mode: object,
    selective_secondary_trigger_gap: object,
    selective_secondary_band_size: object,
) -> bool:
    mode_key = str(selective_secondary_score_mode).strip().lower()
    trigger_gap = _optional_float(selective_secondary_trigger_gap)
    try:
        band_size = int(selective_secondary_band_size)
    except (TypeError, ValueError):
        band_size = 0
    return bool(mode_key) and trigger_gap is not None and float(trigger_gap) > 0.0 and int(band_size) > 0


def _format_selective_secondary_control(
    *,
    selective_secondary_score_mode: object,
    selective_secondary_trigger_gap: object,
    selective_secondary_band_size: object,
) -> str:
    if not _selective_secondary_enabled(
        selective_secondary_score_mode=selective_secondary_score_mode,
        selective_secondary_trigger_gap=selective_secondary_trigger_gap,
        selective_secondary_band_size=selective_secondary_band_size,
    ):
        return ""
    return (
        f"{str(selective_secondary_score_mode).strip()}"
        f"[gap<={float(selective_secondary_trigger_gap):g},band={int(selective_secondary_band_size)}]"
    )


def _selective_secondary_slug(
    *,
    selective_secondary_score_mode: object,
    selective_secondary_trigger_gap: object,
    selective_secondary_band_size: object,
) -> str:
    if not _selective_secondary_enabled(
        selective_secondary_score_mode=selective_secondary_score_mode,
        selective_secondary_trigger_gap=selective_secondary_trigger_gap,
        selective_secondary_band_size=selective_secondary_band_size,
    ):
        return ""
    mode_slug = "".join(
        ch if ch.isalnum() or ch in {"_", "-", "."} else "_"
        for ch in str(selective_secondary_score_mode).strip()
    )
    return (
        f"sel_{mode_slug}"
        f"_g{_compact_float_slug(float(selective_secondary_trigger_gap))}"
        f"_b{int(selective_secondary_band_size)}"
    )


def _selective_local_lookahead_enabled(mode: object) -> bool:
    return str(mode).strip().lower() not in {"", "none"}


def _format_selective_local_lookahead_control(
    *,
    mode: object,
    cutoff_gap_threshold: object,
    near_cut_width: object,
    max_candidates: object,
) -> str:
    mode_key = str(mode).strip().lower()
    if not _selective_local_lookahead_enabled(mode_key):
        return ""
    return (
        f"{mode_key}"
        f"[gap<={float(cutoff_gap_threshold):g},W={float(near_cut_width):g},M={int(max_candidates)}]"
    )


def _selective_local_lookahead_steps_json(result: progressive.ProgressiveDecodeResult) -> str:
    steps = []
    for step in tuple(result.selective_local_lookahead_steps):
        steps.append(
            {
                "processed_columns": int(step.processed_columns),
                "boundary_column_index": int(step.boundary_column_index),
                "mode": str(step.mode),
                "score_mode": str(step.score_mode),
                "trigger_reason": str(step.trigger_reason),
                "cutoff_gap": float(step.cutoff_gap),
                "cutoff_gap_threshold": float(step.cutoff_gap_threshold),
                "candidate_top1_share": float(step.candidate_top1_share),
                "candidate_top1_share_threshold": float(step.candidate_top1_share_threshold),
                "kept_top1_share": float(step.kept_top1_share),
                "candidate_effective_support": float(step.candidate_effective_support),
                "kept_effective_support": float(step.kept_effective_support),
                "support_gap": float(step.support_gap),
                "support_gap_threshold": float(step.support_gap_threshold),
                "overflow_ratio": float(step.overflow_ratio),
                "overflow_ratio_threshold": float(step.overflow_ratio_threshold),
                "near_cut_width": float(step.near_cut_width),
                "max_candidates": int(step.max_candidates),
                "near_cut_candidate_count": int(step.near_cut_candidate_count),
                "extra_lookahead_work": int(step.extra_lookahead_work),
                "kept_slots_in_near_cut": int(step.kept_slots_in_near_cut),
                "changed": bool(step.changed),
                "baseline_kept_count": int(step.baseline_kept_count),
                "final_kept_count": int(step.final_kept_count),
                "candidate_state_count": int(step.candidate_state_count),
                "beam_size": int(step.beam_size),
            }
        )
    return json.dumps(steps, sort_keys=True, separators=(",", ":"))


def _serialize_int_series(values: Sequence[int]) -> str:
    return " ".join(str(int(value)) for value in values)


def _parse_int_series(text: object) -> np.ndarray:
    raw = str(text).strip()
    if not raw:
        return np.zeros(0, dtype=np.int32)
    return np.fromstring(raw, sep=" ", dtype=np.int32)


def _profile_slug(
    *,
    decoder_mode: str,
    score_mode: str,
    beam_size: int,
    beam_score_gap_threshold: object,
    backward_column_order: str,
    beam_score_gap_policy: progressive.BeamScoreGapPolicy | None = None,
    selective_secondary_score_mode: str = "",
    selective_secondary_trigger_gap: float = 0.0,
    selective_secondary_band_size: int = 0,
    forward_guidance_trigger_gap: float = 0.0,
    forward_guidance_snapshot_factor: float = 1.0,
    forward_guidance_snapshot_gap: object = None,
    forward_guidance_snapshot_source: str = "kept",
    forward_guidance_hamming_radius: int = 0,
    forward_guidance_trigger_mode: str = "top_gap",
    forward_guidance_nearcut_gap: float = 0.0,
    forward_guidance_pool_trigger_min_positive_nearcut: int = 1,
    forward_guidance_diversity_fallback: str = "none",
    forward_guidance_mode: str = "detector_penalty",
) -> str:
    pieces = [
        _normalize_decoder_mode(str(decoder_mode)),
        f"k{int(beam_size)}",
        str(score_mode),
    ]
    beam_gap_text = _format_beam_score_gap_threshold(beam_score_gap_threshold)
    if beam_gap_text:
        pieces.append(f"sg{_compact_float_slug(float(beam_gap_text))}")
    if _beam_score_gap_policy_enabled(beam_score_gap_policy):
        pieces.append(_beam_score_gap_policy_slug(beam_score_gap_policy))
    selective_slug = _selective_secondary_slug(
        selective_secondary_score_mode=str(selective_secondary_score_mode),
        selective_secondary_trigger_gap=float(selective_secondary_trigger_gap),
        selective_secondary_band_size=int(selective_secondary_band_size),
    )
    if selective_slug:
        pieces.append(selective_slug)
    if float(forward_guidance_trigger_gap) > 0.0:
        pieces.append(f"fgtg{_compact_float_slug(float(forward_guidance_trigger_gap))}")
    if float(forward_guidance_snapshot_factor) > 1.0:
        pieces.append(f"fgsf{_compact_float_slug(float(forward_guidance_snapshot_factor))}")
    snapshot_gap_slug = _forward_guidance_snapshot_gap_slug(forward_guidance_snapshot_gap)
    if snapshot_gap_slug:
        pieces.append(snapshot_gap_slug)
    if str(forward_guidance_snapshot_source).strip().lower() != "kept":
        pieces.append(f"fgss{str(forward_guidance_snapshot_source).strip().lower()}")
    if int(forward_guidance_hamming_radius) > 0:
        pieces.append(f"fghr{int(forward_guidance_hamming_radius)}")
    if str(forward_guidance_trigger_mode).strip().lower() != "top_gap":
        pieces.append(f"fgtm{str(forward_guidance_trigger_mode).strip().lower()}")
    if float(forward_guidance_nearcut_gap) > 0.0:
        pieces.append(f"fgnc{_compact_float_slug(float(forward_guidance_nearcut_gap))}")
    if int(forward_guidance_pool_trigger_min_positive_nearcut) != 1:
        pieces.append(f"fgpos{int(forward_guidance_pool_trigger_min_positive_nearcut)}")
    if str(forward_guidance_diversity_fallback).strip().lower() != "none":
        pieces.append(f"fgdf{str(forward_guidance_diversity_fallback).strip().lower()}")
    if str(forward_guidance_mode).strip().lower() != "detector_penalty":
        pieces.append(f"fg{str(forward_guidance_mode).strip().lower()}")
    backward_label = str(backward_column_order).strip()
    if backward_label:
        pieces.append(backward_label.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "_"))
    slug = "_".join(piece for piece in pieces if piece)
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in slug)


def _is_default_progressive_mode(
    *,
    score_mode: str,
    beam_score_gap_threshold: float | None,
    beam_score_gap_policy: progressive.BeamScoreGapPolicy | None = None,
    lookahead_depth: int,
    lookahead_shortlist_size: int,
    delayed_pruning_gap_threshold: float,
    delayed_pruning_factor: int,
    pruning_replay_checkpoint_stride: int,
    pruning_replay_horizon: int,
    tail_exact_columns: int,
    superstep_mode: str,
    detector_bucket_pruning: bool,
    detector_bucket_max_logicals: int,
    logical_class_reserve_min_classes: int,
    logical_class_reserve_max_replacements: int,
    logical_class_reserve_min_remaining_columns: int,
    logical_class_quota_top_classes: int,
    logical_class_quota_reserved_slots: int,
    logical_class_quota_min_remaining_columns: int,
    lineage_reserve_checkpoint_stride: int,
    lineage_reserve_reserved_slots: int,
    logical_rerank_columns: int,
    logical_rerank_mode: str,
    selective_secondary_score_mode: str = "",
    selective_secondary_trigger_gap: float = 0.0,
    selective_secondary_band_size: int = 0,
) -> bool:
    return (
        str(score_mode) == "prefix"
        and not _beam_score_gap_policy_enabled(beam_score_gap_policy)
        and not _selective_secondary_enabled(
            selective_secondary_score_mode=str(selective_secondary_score_mode),
            selective_secondary_trigger_gap=float(selective_secondary_trigger_gap),
            selective_secondary_band_size=int(selective_secondary_band_size),
        )
        and (
            beam_score_gap_threshold is None
            or not math.isfinite(float(beam_score_gap_threshold))
            or float(beam_score_gap_threshold) <= 0.0
        )
        and int(lookahead_depth) == 0
        and int(lookahead_shortlist_size) == 0
        and float(delayed_pruning_gap_threshold) <= 0.0
        and int(delayed_pruning_factor) <= 1
        and int(pruning_replay_checkpoint_stride) == 0
        and int(pruning_replay_horizon) == 0
        and int(tail_exact_columns) == 0
        and str(superstep_mode) == "none"
        and not bool(detector_bucket_pruning)
        and int(logical_class_reserve_min_classes) == 0
        and int(logical_class_reserve_max_replacements) == 0
        and int(logical_class_reserve_min_remaining_columns) == 0
        and int(logical_class_quota_top_classes) == 0
        and int(logical_class_quota_reserved_slots) == 0
        and int(logical_class_quota_min_remaining_columns) == 0
        and int(lineage_reserve_checkpoint_stride) == 0
        and int(lineage_reserve_reserved_slots) == 0
        and int(logical_rerank_columns) == 0
        and str(logical_rerank_mode) in {"exact_tail", "exact_tail_vector"}
    )


def _decoder_label(
    *,
    family_key: str,
    decoder_mode: str,
    backward_column_order: str,
    beam_size: int,
    score_mode: str,
    beam_score_gap_threshold: float | None,
    beam_score_gap_policy: progressive.BeamScoreGapPolicy | None = None,
    lookahead_depth: int,
    lookahead_shortlist_size: int,
    delayed_pruning_gap_threshold: float,
    delayed_pruning_factor: int,
    pruning_replay_checkpoint_stride: int,
    pruning_replay_horizon: int,
    tail_exact_columns: int,
    superstep_mode: str,
    detector_bucket_pruning: bool,
    detector_bucket_max_logicals: int,
    logical_class_reserve_min_classes: int,
    logical_class_reserve_max_replacements: int,
    logical_class_reserve_min_remaining_columns: int,
    logical_class_quota_top_classes: int,
    logical_class_quota_reserved_slots: int,
    logical_class_quota_min_remaining_columns: int,
    lineage_reserve_checkpoint_stride: int,
    lineage_reserve_reserved_slots: int,
    logical_rerank_columns: int,
    logical_rerank_shortlist_size: int,
    logical_rerank_min_classes: int,
    logical_rerank_state_budget: int,
    logical_rerank_transition_budget: int,
    logical_rerank_checkpoint_stride: int,
    logical_rerank_max_passes: int,
    logical_rerank_mode: str,
    selective_secondary_score_mode: str = "",
    selective_secondary_trigger_gap: float = 0.0,
    selective_secondary_band_size: int = 0,
    forward_guidance_trigger_gap: float = 0.0,
    forward_guidance_snapshot_factor: float = 1.0,
    forward_guidance_snapshot_gap: object = None,
    forward_guidance_snapshot_source: str = "kept",
    forward_guidance_hamming_radius: int = 0,
    forward_guidance_trigger_mode: str = "top_gap",
    forward_guidance_nearcut_gap: float = 0.0,
    forward_guidance_pool_trigger_min_positive_nearcut: int = 1,
    forward_guidance_diversity_fallback: str = "none",
    forward_guidance_mode: str = "detector_penalty",
) -> str:
    base = f"{family_key}_beam{int(beam_size)}"
    decoder_mode_key = _normalize_decoder_mode(str(decoder_mode))
    backward_order_label = str(backward_column_order).strip() or _default_backward_column_order_label(str(decoder_mode))
    if decoder_mode_key == "backward":
        base += "_backward"
        if backward_order_label not in {"", "reverse_forward_columns"}:
            base += f"_back{backward_order_label}"
    elif decoder_mode_key == "bidirectional_committee":
        base += "_bidir"
        if backward_order_label not in {"", "reverse_forward_columns"}:
            base += f"_back{backward_order_label}"
    elif decoder_mode_key == "forward_guided_backward":
        base += "_fgback"
        if backward_order_label not in {"", "reverse_forward_columns"}:
            base += f"_back{backward_order_label}"
    elif decoder_mode_key == "bidirectional_middle_join":
        base += "_mitm"
        if backward_order_label not in {"", "reverse_forward_columns"}:
            base += f"_back{backward_order_label}"
    if _is_default_progressive_mode(
        score_mode=str(score_mode),
        beam_score_gap_threshold=beam_score_gap_threshold,
        beam_score_gap_policy=beam_score_gap_policy,
        selective_secondary_score_mode=str(selective_secondary_score_mode),
        selective_secondary_trigger_gap=float(selective_secondary_trigger_gap),
        selective_secondary_band_size=int(selective_secondary_band_size),
        lookahead_depth=int(lookahead_depth),
        lookahead_shortlist_size=int(lookahead_shortlist_size),
        delayed_pruning_gap_threshold=float(delayed_pruning_gap_threshold),
        delayed_pruning_factor=int(delayed_pruning_factor),
        pruning_replay_checkpoint_stride=int(pruning_replay_checkpoint_stride),
        pruning_replay_horizon=int(pruning_replay_horizon),
        tail_exact_columns=int(tail_exact_columns),
        superstep_mode=str(superstep_mode),
        detector_bucket_pruning=bool(detector_bucket_pruning),
        detector_bucket_max_logicals=int(detector_bucket_max_logicals),
        logical_class_reserve_min_classes=int(logical_class_reserve_min_classes),
        logical_class_reserve_max_replacements=int(logical_class_reserve_max_replacements),
        logical_class_reserve_min_remaining_columns=int(logical_class_reserve_min_remaining_columns),
        logical_class_quota_top_classes=int(logical_class_quota_top_classes),
        logical_class_quota_reserved_slots=int(logical_class_quota_reserved_slots),
        logical_class_quota_min_remaining_columns=int(logical_class_quota_min_remaining_columns),
        lineage_reserve_checkpoint_stride=int(lineage_reserve_checkpoint_stride),
        lineage_reserve_reserved_slots=int(lineage_reserve_reserved_slots),
        logical_rerank_columns=int(logical_rerank_columns),
        logical_rerank_mode=str(logical_rerank_mode),
    ) and float(forward_guidance_trigger_gap) <= 0.0 and float(forward_guidance_snapshot_factor) <= 1.0 and not _format_forward_guidance_snapshot_gap(forward_guidance_snapshot_gap) and str(forward_guidance_snapshot_source).strip().lower() == "kept" and int(forward_guidance_hamming_radius) <= 0 and str(forward_guidance_trigger_mode).strip().lower() == "top_gap" and float(forward_guidance_nearcut_gap) <= 0.0 and int(forward_guidance_pool_trigger_min_positive_nearcut) == 1 and str(forward_guidance_diversity_fallback).strip().lower() == "none" and str(forward_guidance_mode).strip().lower() == "detector_penalty":
        return base
    suffix = f"_{str(score_mode)}"
    if (
        beam_score_gap_threshold is not None
        and math.isfinite(float(beam_score_gap_threshold))
        and float(beam_score_gap_threshold) > 0.0
    ):
        suffix += f"_sg{_compact_float_slug(float(beam_score_gap_threshold))}"
    if _beam_score_gap_policy_enabled(beam_score_gap_policy):
        suffix += f"_{_beam_score_gap_policy_slug(beam_score_gap_policy)}"
    selective_slug = _selective_secondary_slug(
        selective_secondary_score_mode=str(selective_secondary_score_mode),
        selective_secondary_trigger_gap=float(selective_secondary_trigger_gap),
        selective_secondary_band_size=int(selective_secondary_band_size),
    )
    if selective_slug:
        suffix += f"_{selective_slug}"
    if float(forward_guidance_trigger_gap) > 0.0:
        suffix += f"_fgtg{_compact_float_slug(float(forward_guidance_trigger_gap))}"
    if float(forward_guidance_snapshot_factor) > 1.0:
        suffix += f"_fgsf{_compact_float_slug(float(forward_guidance_snapshot_factor))}"
    snapshot_gap_slug = _forward_guidance_snapshot_gap_slug(forward_guidance_snapshot_gap)
    if snapshot_gap_slug:
        suffix += f"_{snapshot_gap_slug}"
    if str(forward_guidance_snapshot_source).strip().lower() != "kept":
        suffix += f"_fgss{str(forward_guidance_snapshot_source).strip().lower()}"
    if int(forward_guidance_hamming_radius) > 0:
        suffix += f"_fghr{int(forward_guidance_hamming_radius)}"
    if str(forward_guidance_trigger_mode).strip().lower() != "top_gap":
        suffix += f"_fgtm{str(forward_guidance_trigger_mode).strip().lower()}"
    if float(forward_guidance_nearcut_gap) > 0.0:
        suffix += f"_fgnc{_compact_float_slug(float(forward_guidance_nearcut_gap))}"
    if int(forward_guidance_pool_trigger_min_positive_nearcut) != 1:
        suffix += f"_fgpos{int(forward_guidance_pool_trigger_min_positive_nearcut)}"
    if str(forward_guidance_diversity_fallback).strip().lower() != "none":
        suffix += f"_fgdf{str(forward_guidance_diversity_fallback).strip().lower()}"
    if str(forward_guidance_mode).strip().lower() != "detector_penalty":
        suffix += f"_fg{str(forward_guidance_mode).strip().lower()}"
    if int(lookahead_depth) > 0:
        suffix += f"_la{int(lookahead_depth)}"
    if int(lookahead_shortlist_size) > 0:
        suffix += f"_ls{int(lookahead_shortlist_size)}"
    if float(delayed_pruning_gap_threshold) > 0.0 and int(delayed_pruning_factor) > 1:
        suffix += f"_dpg{_compact_float_slug(float(delayed_pruning_gap_threshold))}x{int(delayed_pruning_factor)}"
    if int(pruning_replay_checkpoint_stride) > 0 and int(pruning_replay_horizon) > 0:
        suffix += f"_pr{int(pruning_replay_checkpoint_stride)}h{int(pruning_replay_horizon)}"
    if int(tail_exact_columns) > 0:
        suffix += f"_te{int(tail_exact_columns)}"
    if str(superstep_mode) != "none":
        suffix += f"_ss{str(superstep_mode)}"
    if bool(detector_bucket_pruning):
        suffix += "_db"
        suffix += "rr" if int(detector_bucket_max_logicals) <= 0 else str(int(detector_bucket_max_logicals))
    if int(logical_class_reserve_min_classes) > 0 or int(logical_class_reserve_max_replacements) > 0:
        suffix += (
            f"_lcr{int(logical_class_reserve_min_classes)}"
            f"r{int(logical_class_reserve_max_replacements)}"
            f"m{int(logical_class_reserve_min_remaining_columns)}"
        )
    if int(logical_class_quota_top_classes) > 0 or int(logical_class_quota_reserved_slots) > 0:
        suffix += (
            f"_lcq{int(logical_class_quota_top_classes)}"
            f"s{int(logical_class_quota_reserved_slots)}"
            f"m{int(logical_class_quota_min_remaining_columns)}"
        )
    if int(lineage_reserve_checkpoint_stride) > 0 or int(lineage_reserve_reserved_slots) > 0:
        suffix += f"_lin{int(lineage_reserve_checkpoint_stride)}s{int(lineage_reserve_reserved_slots)}"
    if int(logical_rerank_columns) > 0:
        suffix += (
            f"_lr{int(logical_rerank_columns)}x{int(logical_rerank_shortlist_size)}c{int(logical_rerank_min_classes)}"
        )
        if str(logical_rerank_mode) != "exact_tail":
            suffix += f"_{str(logical_rerank_mode)}"
        if int(logical_rerank_state_budget) > 0:
            suffix += f"b{int(logical_rerank_state_budget)}"
        if int(logical_rerank_transition_budget) > 0:
            suffix += f"t{int(logical_rerank_transition_budget)}"
        if int(logical_rerank_checkpoint_stride) > 0:
            suffix += f"s{int(logical_rerank_checkpoint_stride)}"
        if int(logical_rerank_max_passes) != 1:
            suffix += f"m{int(logical_rerank_max_passes)}"
    return f"{base}{suffix}"


def _curve_label(
    *,
    decoder_mode: str,
    backward_column_order: str,
    correction_state_mode: str,
    score_mode: str,
    beam_score_gap_threshold: float | None,
    beam_score_gap_policy: progressive.BeamScoreGapPolicy | None = None,
    lookahead_depth: int,
    lookahead_shortlist_size: int,
    delayed_pruning_gap_threshold: float,
    delayed_pruning_factor: int,
    pruning_replay_checkpoint_stride: int,
    pruning_replay_horizon: int,
    tail_exact_columns: int,
    superstep_mode: str,
    detector_bucket_pruning: bool,
    detector_bucket_max_logicals: int,
    logical_class_reserve_min_classes: int,
    logical_class_reserve_max_replacements: int,
    logical_class_reserve_min_remaining_columns: int,
    logical_class_quota_top_classes: int,
    logical_class_quota_reserved_slots: int,
    logical_class_quota_min_remaining_columns: int,
    lineage_reserve_checkpoint_stride: int,
    lineage_reserve_reserved_slots: int,
    logical_rerank_columns: int,
    logical_rerank_shortlist_size: int,
    logical_rerank_min_classes: int,
    logical_rerank_state_budget: int,
    logical_rerank_transition_budget: int,
    logical_rerank_checkpoint_stride: int,
    logical_rerank_max_passes: int,
    logical_rerank_mode: str,
    selective_secondary_score_mode: str = "",
    selective_secondary_trigger_gap: float = 0.0,
    selective_secondary_band_size: int = 0,
    forward_guidance_trigger_gap: float = 0.0,
    forward_guidance_snapshot_factor: float = 1.0,
    forward_guidance_snapshot_gap: object = None,
    forward_guidance_snapshot_source: str = "kept",
    forward_guidance_hamming_radius: int = 0,
    forward_guidance_trigger_mode: str = "top_gap",
    forward_guidance_nearcut_gap: float = 0.0,
    forward_guidance_pool_trigger_min_positive_nearcut: int = 1,
    forward_guidance_diversity_fallback: str = "none",
    forward_guidance_mode: str = "detector_penalty",
) -> str:
    label = str(score_mode)
    extras: list[str] = []
    if (
        beam_score_gap_threshold is not None
        and math.isfinite(float(beam_score_gap_threshold))
        and float(beam_score_gap_threshold) > 0.0
    ):
        extras.append(f"sg={float(beam_score_gap_threshold):g}")
    if _beam_score_gap_policy_enabled(beam_score_gap_policy):
        extras.append(f"sgp={_format_beam_score_gap_policy(beam_score_gap_policy)}")
    selective_text = _format_selective_secondary_control(
        selective_secondary_score_mode=str(selective_secondary_score_mode),
        selective_secondary_trigger_gap=float(selective_secondary_trigger_gap),
        selective_secondary_band_size=int(selective_secondary_band_size),
    )
    if selective_text:
        extras.append(f"sel={selective_text}")
    if float(forward_guidance_trigger_gap) > 0.0:
        extras.append(f"fgtg={float(forward_guidance_trigger_gap):g}")
    if float(forward_guidance_snapshot_factor) > 1.0:
        extras.append(f"fgsf={float(forward_guidance_snapshot_factor):g}")
    snapshot_gap_text = _format_forward_guidance_snapshot_gap(forward_guidance_snapshot_gap)
    if snapshot_gap_text:
        extras.append(f"fgsgap={snapshot_gap_text}")
    if str(forward_guidance_snapshot_source).strip().lower() != "kept":
        extras.append(f"fgss={str(forward_guidance_snapshot_source).strip().lower()}")
    if int(forward_guidance_hamming_radius) > 0:
        extras.append(f"fghr={int(forward_guidance_hamming_radius)}")
    if str(forward_guidance_trigger_mode).strip().lower() != "top_gap":
        extras.append(f"fgtm={str(forward_guidance_trigger_mode).strip().lower()}")
    if float(forward_guidance_nearcut_gap) > 0.0:
        extras.append(f"fgnc={float(forward_guidance_nearcut_gap):g}")
    if int(forward_guidance_pool_trigger_min_positive_nearcut) != 1:
        extras.append(f"fgpos={int(forward_guidance_pool_trigger_min_positive_nearcut)}")
    if str(forward_guidance_diversity_fallback).strip().lower() != "none":
        extras.append(f"fgdf={str(forward_guidance_diversity_fallback).strip().lower()}")
    if str(forward_guidance_mode).strip().lower() != "detector_penalty":
        extras.append(f"fg={str(forward_guidance_mode).strip().lower()}")
    decoder_mode_key = _normalize_decoder_mode(str(decoder_mode))
    backward_order_label = str(backward_column_order).strip() or _default_backward_column_order_label(str(decoder_mode))
    if decoder_mode_key == "backward":
        extras.append("backward")
        if backward_order_label not in {"", "reverse_forward_columns"}:
            extras.append(f"back={backward_order_label}")
    elif decoder_mode_key == "bidirectional_committee":
        extras.append("bidir")
        if backward_order_label not in {"", "reverse_forward_columns"}:
            extras.append(f"back={backward_order_label}")
    elif decoder_mode_key == "forward_guided_backward":
        extras.append("fgback")
        if backward_order_label not in {"", "reverse_forward_columns"}:
            extras.append(f"back={backward_order_label}")
    elif decoder_mode_key == "bidirectional_middle_join":
        extras.append("mitm")
        if backward_order_label not in {"", "reverse_forward_columns"}:
            extras.append(f"back={backward_order_label}")
    if str(correction_state_mode) != "none":
        extras.append(f"corr={str(correction_state_mode)}")
    if int(lookahead_depth) > 0:
        extras.append(f"la={int(lookahead_depth)}")
    if int(lookahead_shortlist_size) > 0:
        extras.append(f"ls={int(lookahead_shortlist_size)}")
    if float(delayed_pruning_gap_threshold) > 0.0 and int(delayed_pruning_factor) > 1:
        extras.append(f"dp={float(delayed_pruning_gap_threshold):g}x{int(delayed_pruning_factor)}")
    if int(pruning_replay_checkpoint_stride) > 0 and int(pruning_replay_horizon) > 0:
        extras.append(f"pr={int(pruning_replay_checkpoint_stride)}/{int(pruning_replay_horizon)}")
    if int(tail_exact_columns) > 0:
        extras.append(f"te={int(tail_exact_columns)}")
    if str(superstep_mode) != "none":
        extras.append(f"ss={str(superstep_mode)}")
    if bool(detector_bucket_pruning):
        extras.append(
            "db=rr" if int(detector_bucket_max_logicals) <= 0 else f"db={int(detector_bucket_max_logicals)}"
        )
    if int(logical_class_reserve_min_classes) > 0 or int(logical_class_reserve_max_replacements) > 0:
        extras.append(
            f"lcr={int(logical_class_reserve_min_classes)}/{int(logical_class_reserve_max_replacements)}/{int(logical_class_reserve_min_remaining_columns)}"
        )
    if int(logical_class_quota_top_classes) > 0 or int(logical_class_quota_reserved_slots) > 0:
        extras.append(
            f"lcq={int(logical_class_quota_top_classes)}/{int(logical_class_quota_reserved_slots)}/{int(logical_class_quota_min_remaining_columns)}"
        )
    if int(lineage_reserve_checkpoint_stride) > 0 or int(lineage_reserve_reserved_slots) > 0:
        extras.append(f"lin={int(lineage_reserve_checkpoint_stride)}/{int(lineage_reserve_reserved_slots)}")
    if int(logical_rerank_columns) > 0:
        extras.append(
            f"lr={int(logical_rerank_columns)}/{int(logical_rerank_shortlist_size)}/{int(logical_rerank_min_classes)}"
        )
        if str(logical_rerank_mode) != "exact_tail":
            extras.append(f"lr_mode={str(logical_rerank_mode)}")
        if int(logical_rerank_state_budget) > 0:
            extras.append(f"lr_state={int(logical_rerank_state_budget)}")
        if int(logical_rerank_transition_budget) > 0:
            extras.append(f"lr_trans={int(logical_rerank_transition_budget)}")
        if int(logical_rerank_checkpoint_stride) > 0:
            extras.append(f"lr_stride={int(logical_rerank_checkpoint_stride)}")
        if int(logical_rerank_max_passes) != 1:
            extras.append(f"lr_passes={int(logical_rerank_max_passes)}")
    if extras:
        label += " (" + ", ".join(extras) + ")"
    return label


def _correction_state_mode_family_suffix(correction_state_mode: str) -> str:
    mode_key = str(correction_state_mode).strip().lower()
    if mode_key == "none":
        return ""
    if mode_key == "full":
        return "_corrfull"
    if mode_key == "logical_class":
        return "_corrlogical"
    if mode_key == "stabilizer_quotient":
        return "_corrquotient"
    raise ValueError(f"unsupported correction_state_mode {correction_state_mode!r}")


def _row_support_masks(rows: np.ndarray) -> tuple[int, ...]:
    arr = dense_mod2(rows)
    if arr.size == 0:
        return tuple()
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return tuple(
        _bitmask_from_indices(np.flatnonzero(arr[int(row_index)]).astype(np.int32, copy=False).tolist())
        for row_index in range(int(arr.shape[0]))
    )


def _project_correction_mask(mask: int, row_support_masks: Sequence[int]) -> int:
    projected = 0
    raw_mask = int(mask)
    for row_index, row_mask in enumerate(tuple(int(value) for value in row_support_masks)):
        if int((int(raw_mask) & int(row_mask)).bit_count() & 1):
            projected |= 1 << int(row_index)
    return int(projected)


def _project_correction_map_by_signature(
    correction_by_signature: dict[int, int],
    *,
    row_support_masks: Sequence[int],
) -> dict[int, int]:
    masks = tuple(int(value) for value in row_support_masks)
    return {
        int(signature_mask): int(_project_correction_mask(int(correction_mask), masks))
        for signature_mask, correction_mask in correction_by_signature.items()
    }


def _correction_projection_rows(
    *,
    problem,
    scope: str,
    correction_state_mode: str,
) -> np.ndarray | None:
    mode_key = str(correction_state_mode).strip().lower()
    if mode_key in {"none", "full"}:
        return None
    if str(scope) == "memory_X":
        checks = np.asarray(problem.HX.toarray(), dtype=np.uint8)
        logicals = np.asarray(problem.LX, dtype=np.uint8)
    elif str(scope) == "memory_Z":
        checks = np.asarray(problem.HZ.toarray(), dtype=np.uint8)
        logicals = np.asarray(problem.LZ, dtype=np.uint8)
    else:
        raise ValueError(f"unsupported scope: {scope}")
    if mode_key == "logical_class":
        rows = logicals
    elif mode_key == "stabilizer_quotient":
        rows = select_independent_rows_mod2(np.vstack([checks, logicals]).astype(np.uint8, copy=False))
    else:
        raise ValueError(f"unsupported correction_state_mode {correction_state_mode!r}")
    return dense_mod2(rows)


def _deadline_order_key(
    *,
    columns: Sequence[progressive.ProgressiveColumn],
    support_rows_by_column: Sequence[tuple[int, ...]],
    first_touch_by_row: Sequence[int],
    last_touch_by_row: Sequence[int],
    column_index: int,
) -> tuple[int, ...]:
    support_rows = tuple(int(row) for row in support_rows_by_column[int(column_index)])
    if not support_rows:
        sentinel = int(len(columns) + 1)
        return (
            int(sentinel),
            int(sentinel),
            int(sentinel),
            int(columns[int(column_index)].instruction_offset),
            int(columns[int(column_index)].index),
        )
    earliest_last = min(int(last_touch_by_row[int(row)]) for row in support_rows)
    latest_last = max(int(last_touch_by_row[int(row)]) for row in support_rows)
    earliest_first = min(int(first_touch_by_row[int(row)]) for row in support_rows)
    return (
        int(earliest_last),
        int(latest_last),
        int(earliest_first),
        int(columns[int(column_index)].instruction_offset),
        int(columns[int(column_index)].index),
    )


def _precompute_deadline_order_data(
    columns: Sequence[progressive.ProgressiveColumn],
    *,
    num_detectors: int,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...], tuple[int, ...]]:
    support_rows_by_column: list[tuple[int, ...]] = []
    first_touch_by_row = [-1 for _ in range(int(num_detectors))]
    last_touch_by_row = [-1 for _ in range(int(num_detectors))]
    for column_index, column in enumerate(columns):
        support_rows = tuple(int(row) for row in column.detector_support_rows)
        if not support_rows:
            support_rows = progressive._support_rows_from_mask(int(column.detector_support_mask))
        support_rows_by_column.append(tuple(int(row) for row in support_rows))
        for row in support_rows:
            if int(first_touch_by_row[int(row)]) < 0:
                first_touch_by_row[int(row)] = int(column_index)
            last_touch_by_row[int(row)] = int(column_index)
    return (
        tuple(support_rows_by_column),
        tuple(int(value) for value in first_touch_by_row),
        tuple(int(value) for value in last_touch_by_row),
    )


def _column_detector_support_mask(column: progressive.ProgressiveColumn) -> int:
    mask = int(column.detector_support_mask)
    if mask != 0 or not column.detector_support_rows:
        return int(mask)
    return int(_support_mask_from_rows(column.detector_support_rows))


def _column_rank_feature_mask(
    column: progressive.ProgressiveColumn,
    *,
    logical_bit_offset: int,
) -> int:
    feature_mask = int(_column_detector_support_mask(column))
    logical_support_mask = 0
    for logical_mask in tuple(column.logical_response_masks):
        logical_support_mask |= int(logical_mask)
    feature_mask |= int(logical_support_mask) << int(logical_bit_offset)
    return int(feature_mask)


def _reduce_rank_feature_mask(mask: int, basis_by_pivot: dict[int, int]) -> int:
    reduced = int(mask)
    while int(reduced) != 0:
        pivot = int(reduced.bit_length()) - 1
        basis_value = basis_by_pivot.get(int(pivot))
        if basis_value is None:
            break
        reduced ^= int(basis_value)
    return int(reduced)


def _insert_rank_feature_mask(mask: int, basis_by_pivot: dict[int, int]) -> bool:
    reduced = int(_reduce_rank_feature_mask(int(mask), basis_by_pivot))
    if int(reduced) == 0:
        return False
    pivot = int(reduced.bit_length()) - 1
    for other_pivot, other_value in tuple(basis_by_pivot.items()):
        if (int(other_value) >> int(pivot)) & 1:
            basis_by_pivot[int(other_pivot)] = int(other_value) ^ int(reduced)
    basis_by_pivot[int(pivot)] = int(reduced)
    return True


def _greedy_rank_gain_per_open_row_order(
    *,
    columns_ordered: Sequence[progressive.ProgressiveColumn],
    num_detectors: int,
    support_rows_by_column: Sequence[tuple[int, ...]],
    first_touch_by_row: Sequence[int],
    last_touch_by_row: Sequence[int],
) -> tuple[int, ...]:
    total_touch_count_by_row = [0 for _ in range(int(num_detectors))]
    for rows in support_rows_by_column:
        for row in rows:
            total_touch_count_by_row[int(row)] += 1
    processed_by_row = [0 for _ in range(int(num_detectors))]
    remaining_by_row = list(int(value) for value in total_touch_count_by_row)
    current_active_width = 0
    basis_by_pivot: dict[int, int] = {}
    feature_masks_by_column = tuple(
        int(_column_rank_feature_mask(column, logical_bit_offset=int(num_detectors)))
        for column in tuple(columns_ordered)
    )
    remaining = set(range(int(len(columns_ordered))))
    ordering_list: list[int] = []

    def _deadline_key(column_index: int) -> tuple[int, ...]:
        return _deadline_order_key(
            columns=columns_ordered,
            support_rows_by_column=support_rows_by_column,
            first_touch_by_row=first_touch_by_row,
            last_touch_by_row=last_touch_by_row,
            column_index=int(column_index),
        )

    def _candidate_key(column_index: int) -> tuple[float, int, int, int, int, tuple[int, ...]]:
        reduced_feature_mask = int(
            _reduce_rank_feature_mask(int(feature_masks_by_column[int(column_index)]), basis_by_pivot)
        )
        rank_gain = 1 if int(reduced_feature_mask) != 0 else 0
        new_opened_rows = 0
        closed_rows = 0
        for row in support_rows_by_column[int(column_index)]:
            row_index = int(row)
            before_active = bool(
                int(processed_by_row[int(row_index)]) > 0
                and int(remaining_by_row[int(row_index)]) > 0
            )
            after_active = bool(int(remaining_by_row[int(row_index)]) - 1 > 0)
            if bool(after_active) and not bool(before_active):
                new_opened_rows += 1
            elif bool(before_active) and not bool(after_active):
                closed_rows += 1
        active_after_width = int(current_active_width) + int(new_opened_rows) - int(closed_rows)
        gain_per_open_row = float(rank_gain) / float(1 + int(new_opened_rows))
        return (
            -float(gain_per_open_row),
            -int(rank_gain),
            -int(closed_rows),
            int(new_opened_rows),
            int(active_after_width),
            _deadline_key(int(column_index)),
        )

    while remaining:
        best_column_index = min(remaining, key=_candidate_key)
        ordering_list.append(int(best_column_index))
        current_active_width = int(
            current_active_width
            + sum(
                1
                for row in support_rows_by_column[int(best_column_index)]
                if not bool(
                    int(processed_by_row[int(row)]) > 0
                    and int(remaining_by_row[int(row)]) > 0
                )
                and bool(int(remaining_by_row[int(row)]) - 1 > 0)
            )
            - sum(
                1
                for row in support_rows_by_column[int(best_column_index)]
                if bool(
                    int(processed_by_row[int(row)]) > 0
                    and int(remaining_by_row[int(row)]) > 0
                )
                and not bool(int(remaining_by_row[int(row)]) - 1 > 0)
            )
        )
        _insert_rank_feature_mask(int(feature_masks_by_column[int(best_column_index)]), basis_by_pivot)
        for row in support_rows_by_column[int(best_column_index)]:
            processed_by_row[int(row)] += 1
            remaining_by_row[int(row)] -= 1
        remaining.remove(int(best_column_index))
    return tuple(int(value) for value in ordering_list)


def _logical_response_weight(column: progressive.ProgressiveColumn) -> int:
    logical_support_mask = 0
    for logical_mask in tuple(column.logical_response_masks):
        logical_support_mask |= int(logical_mask)
    return int(logical_support_mask.bit_count())


def _logical_frontload_deadline_order(
    *,
    columns_ordered: Sequence[progressive.ProgressiveColumn],
    support_rows_by_column: Sequence[tuple[int, ...]],
    first_touch_by_row: Sequence[int],
    last_touch_by_row: Sequence[int],
) -> tuple[int, ...]:
    def _candidate_key(column_index: int) -> tuple[int, int, tuple[int, ...]]:
        logical_weight = int(_logical_response_weight(columns_ordered[int(column_index)]))
        return (
            0 if logical_weight > 0 else 1,
            -int(logical_weight),
            _deadline_order_key(
                columns=columns_ordered,
                support_rows_by_column=support_rows_by_column,
                first_touch_by_row=first_touch_by_row,
                last_touch_by_row=last_touch_by_row,
                column_index=int(column_index),
            ),
        )

    return tuple(sorted(range(len(columns_ordered)), key=_candidate_key))


def _row_band_round_robin_deadline_order(
    *,
    columns_ordered: Sequence[progressive.ProgressiveColumn],
    num_detectors: int,
    support_rows_by_column: Sequence[tuple[int, ...]],
    first_touch_by_row: Sequence[int],
    last_touch_by_row: Sequence[int],
    band_count: int = 8,
    band_order_mode: str = "ascending",
) -> tuple[int, ...]:
    band_total = max(1, int(band_count))
    no_support_band = int(band_total)

    def _band_for_column(column_index: int) -> int:
        support_rows = tuple(sorted(int(row) for row in support_rows_by_column[int(column_index)]))
        if not support_rows or int(num_detectors) <= 0:
            return int(no_support_band)
        median_row = int(support_rows[len(support_rows) // 2])
        return min(int(band_total) - 1, max(0, int(median_row) * int(band_total) // max(1, int(num_detectors))))

    deadline_order = sorted(
        range(len(columns_ordered)),
        key=lambda column_index: _deadline_order_key(
            columns=columns_ordered,
            support_rows_by_column=support_rows_by_column,
            first_touch_by_row=first_touch_by_row,
            last_touch_by_row=last_touch_by_row,
            column_index=int(column_index),
        ),
    )
    buckets: list[list[int]] = [[] for _ in range(int(band_total) + 1)]
    for column_index in deadline_order:
        buckets[int(_band_for_column(int(column_index)))].append(int(column_index))

    if str(band_order_mode) == "descending":
        band_sequence = tuple(range(int(band_total) - 1, -1, -1)) + (int(no_support_band),)
    elif str(band_order_mode) == "center_out":
        center = (float(band_total) - 1.0) / 2.0
        band_sequence = tuple(
            sorted(range(int(band_total)), key=lambda band: (abs(float(band) - center), int(band)))
        ) + (int(no_support_band),)
    else:
        band_sequence = tuple(range(int(band_total) + 1))

    ordering: list[int] = []
    cursors = [0 for _ in buckets]
    remaining = int(len(deadline_order))
    while int(remaining) > 0:
        progressed = False
        for band_index in band_sequence:
            cursor = int(cursors[int(band_index)])
            bucket = buckets[int(band_index)]
            if cursor >= len(bucket):
                continue
            ordering.append(int(bucket[cursor]))
            cursors[int(band_index)] = cursor + 1
            remaining -= 1
            progressed = True
        if not progressed:
            break

    return tuple(int(value) for value in ordering)


def _column_error_mass(column: progressive.ProgressiveColumn) -> float:
    probabilities = tuple(float(value) for value in tuple(column.prior_probs))
    if len(probabilities) <= 1:
        return 0.0
    return float(sum(float(value) for value in probabilities[1:]))


def _column_detector_weight(
    column: progressive.ProgressiveColumn,
    support_rows: Sequence[int],
) -> int:
    if support_rows:
        return int(len(tuple(support_rows)))
    return int(_column_detector_support_mask(column).bit_count())


def _round_span_group_order(
    group_keys: Sequence[tuple[int, int]],
    *,
    mode: str,
) -> tuple[tuple[int, int], ...]:
    ordered_keys = tuple(sorted((int(start), int(stop)) for start, stop in group_keys))
    if str(mode) == "center_out":
        if not ordered_keys:
            return ()
        midpoint = (float(len(ordered_keys)) - 1.0) / 2.0
        return tuple(
            sorted(
                ordered_keys,
                key=lambda key: (abs(float(ordered_keys.index(key)) - midpoint), int(ordered_keys.index(key))),
            )
        )
    out: list[tuple[int, int]] = []
    left = 0
    right = len(ordered_keys) - 1
    take_left = True
    while left <= right:
        if take_left:
            out.append(ordered_keys[int(left)])
            left += 1
        else:
            out.append(ordered_keys[int(right)])
            right -= 1
        take_left = not take_left
    return tuple(out)


def _greedy_frontier_shape_order(
    *,
    columns_ordered: Sequence[progressive.ProgressiveColumn],
    num_detectors: int,
    support_rows_by_column: Sequence[tuple[int, ...]],
    first_touch_by_row: Sequence[int],
    last_touch_by_row: Sequence[int],
    mode: str,
) -> tuple[int, ...]:
    total_touch_count_by_row = [0 for _ in range(int(num_detectors))]
    for rows in support_rows_by_column:
        for row in rows:
            total_touch_count_by_row[int(row)] += 1
    processed_by_row = [0 for _ in range(int(num_detectors))]
    remaining_by_row = list(int(value) for value in total_touch_count_by_row)
    current_active_width = 0
    remaining = set(range(int(len(columns_ordered))))
    ordering_list: list[int] = []

    def _deadline_key(column_index: int) -> tuple[int, ...]:
        return _deadline_order_key(
            columns=columns_ordered,
            support_rows_by_column=support_rows_by_column,
            first_touch_by_row=first_touch_by_row,
            last_touch_by_row=last_touch_by_row,
            column_index=int(column_index),
        )

    def _shape_counts(column_index: int) -> tuple[int, int, int]:
        new_opened_rows = 0
        closed_rows = 0
        for row in support_rows_by_column[int(column_index)]:
            row_index = int(row)
            before_active = bool(
                int(processed_by_row[int(row_index)]) > 0
                and int(remaining_by_row[int(row_index)]) > 0
            )
            after_active = bool(int(remaining_by_row[int(row_index)]) - 1 > 0)
            if bool(after_active) and not bool(before_active):
                new_opened_rows += 1
            elif bool(before_active) and not bool(after_active):
                closed_rows += 1
        active_after_width = int(current_active_width) + int(new_opened_rows) - int(closed_rows)
        return int(new_opened_rows), int(closed_rows), int(active_after_width)

    def _candidate_key(column_index: int) -> tuple[int, int, int, tuple[int, ...]]:
        new_opened_rows, closed_rows, active_after_width = _shape_counts(int(column_index))
        if str(mode) == "close_first":
            return (
                -int(closed_rows),
                int(active_after_width),
                int(new_opened_rows),
                _deadline_key(int(column_index)),
            )
        return (
            int(active_after_width),
            -int(closed_rows),
            int(new_opened_rows),
            _deadline_key(int(column_index)),
        )

    while remaining:
        best_column_index = min(remaining, key=_candidate_key)
        new_opened_rows, closed_rows, active_after_width = _shape_counts(int(best_column_index))
        ordering_list.append(int(best_column_index))
        current_active_width = int(active_after_width)
        for row in support_rows_by_column[int(best_column_index)]:
            processed_by_row[int(row)] += 1
            remaining_by_row[int(row)] -= 1
        remaining.remove(int(best_column_index))

    return tuple(int(value) for value in ordering_list)


def _frontier_pressure_counts_for_candidate(
    *,
    support_rows: Sequence[int],
    processed_by_row: Sequence[int],
    remaining_by_row: Sequence[int],
    current_active_width: int,
) -> tuple[int, int, int]:
    newly_opened_checks = 0
    closed_checks = 0
    for row in tuple(int(value) for value in support_rows):
        before_active = bool(int(processed_by_row[int(row)]) > 0 and int(remaining_by_row[int(row)]) > 0)
        after_active = bool(int(remaining_by_row[int(row)]) - 1 > 0)
        if bool(after_active) and not bool(before_active):
            newly_opened_checks += 1
        elif bool(before_active) and not bool(after_active):
            closed_checks += 1
    active_checks_after = int(current_active_width) + int(newly_opened_checks) - int(closed_checks)
    return int(active_checks_after), int(newly_opened_checks), int(closed_checks)


def _minmax_normalized_feature(
    values_by_column: Mapping[int, float],
    candidates: Sequence[int],
) -> dict[int, float]:
    candidate_tuple = tuple(int(value) for value in candidates)
    if not candidate_tuple:
        return {}
    values = tuple(float(values_by_column.get(int(column), 0.0)) for column in candidate_tuple)
    lo = min(values)
    hi = max(values)
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or abs(float(hi) - float(lo)) <= 1e-15:
        return {int(column): 0.0 for column in candidate_tuple}
    scale = float(hi) - float(lo)
    return {
        int(column): (float(values_by_column.get(int(column), 0.0)) - float(lo)) / float(scale)
        for column in candidate_tuple
    }


def _deadline_window_pressure_order(
    *,
    columns_ordered: Sequence[progressive.ProgressiveColumn],
    num_detectors: int,
    support_rows_by_column: Sequence[tuple[int, ...]],
    first_touch_by_row: Sequence[int],
    last_touch_by_row: Sequence[int],
    mode: str,
    window: int = DEADLINE_PRESSURE_WINDOW,
    syndrome_by_row: Sequence[int] | None = None,
) -> tuple[int, ...]:
    n = int(len(columns_ordered))
    if n <= 1:
        return tuple(range(n))
    mode_key = str(mode)
    if mode_key not in {"min_active", "close_first"}:
        raise ValueError(f"unsupported deadline pressure mode {mode!r}")

    deadline_order = tuple(
        sorted(
            range(n),
            key=lambda column_index: _deadline_order_key(
                columns=columns_ordered,
                support_rows_by_column=support_rows_by_column,
                first_touch_by_row=first_touch_by_row,
                last_touch_by_row=last_touch_by_row,
                column_index=int(column_index),
            ),
        )
    )
    deadline_rank = [0 for _ in range(n)]
    for rank, column_index in enumerate(deadline_order):
        deadline_rank[int(column_index)] = int(rank)
    base_rank = tuple(range(n))
    original_column_index = tuple(
        int(column.original_column_index) if int(column.original_column_index) >= 0 else int(column.index)
        for column in tuple(columns_ordered)
    )

    total_touch_count_by_row = [0 for _ in range(int(num_detectors))]
    for rows in tuple(support_rows_by_column):
        for row in tuple(rows):
            total_touch_count_by_row[int(row)] += 1
    processed_by_row = [0 for _ in range(int(num_detectors))]
    remaining_by_row = list(int(value) for value in total_touch_count_by_row)

    if syndrome_by_row is None:
        local_syndrome_frac = tuple(0.0 for _ in range(n))
    else:
        syndrome_tuple = tuple(int(value) & 1 for value in syndrome_by_row)
        if len(syndrome_tuple) != int(num_detectors):
            raise ValueError("syndrome_by_row length does not match detector count")
        local_syndrome_frac = tuple(
            0.0
            if not support_rows_by_column[int(column_index)]
            else float(sum(int(syndrome_tuple[int(row)]) for row in support_rows_by_column[int(column_index)]))
            / float(len(support_rows_by_column[int(column_index)]))
            for column_index in range(n)
        )

    placed = [False for _ in range(n)]
    min_remaining_deadline_rank = 0
    current_active_width = 0
    ordering_list: list[int] = []
    weights = (
        {
            "slack": 1.00,
            "active_after": 1.50,
            "newly_opened": 0.75,
            "closed": -1.25,
            "local_syndrome": -0.25,
            "base_rank": 0.05,
        }
        if mode_key == "min_active"
        else {
            "slack": 1.00,
            "active_after": 1.00,
            "newly_opened": 0.50,
            "closed": -2.00,
            "local_syndrome": 0.00,
            "base_rank": 0.05,
        }
    )

    while len(ordering_list) < n:
        while (
            int(min_remaining_deadline_rank) < n
            and bool(placed[int(deadline_order[int(min_remaining_deadline_rank)])])
        ):
            min_remaining_deadline_rank += 1
        if int(min_remaining_deadline_rank) >= n:
            break
        rank_stop = min(n, int(min_remaining_deadline_rank) + max(0, int(window)) + 1)
        eligible = tuple(
            int(deadline_order[int(rank)])
            for rank in range(int(min_remaining_deadline_rank), int(rank_stop))
            if not bool(placed[int(deadline_order[int(rank)])])
        )
        if not eligible:
            raise RuntimeError("deadline pressure ordering found no eligible columns")

        raw_slack: dict[int, float] = {}
        raw_active: dict[int, float] = {}
        raw_newly_opened: dict[int, float] = {}
        raw_closed: dict[int, float] = {}
        raw_local_syndrome: dict[int, float] = {}
        raw_base_rank: dict[int, float] = {}
        for column_index in eligible:
            active_after, newly_opened, closed = _frontier_pressure_counts_for_candidate(
                support_rows=support_rows_by_column[int(column_index)],
                processed_by_row=processed_by_row,
                remaining_by_row=remaining_by_row,
                current_active_width=int(current_active_width),
            )
            raw_slack[int(column_index)] = float(int(deadline_rank[int(column_index)]) - int(min_remaining_deadline_rank))
            raw_active[int(column_index)] = float(active_after)
            raw_newly_opened[int(column_index)] = float(newly_opened)
            raw_closed[int(column_index)] = float(closed)
            raw_local_syndrome[int(column_index)] = float(local_syndrome_frac[int(column_index)])
            raw_base_rank[int(column_index)] = float(base_rank[int(column_index)])

        norm_slack = _minmax_normalized_feature(raw_slack, eligible)
        norm_active = _minmax_normalized_feature(raw_active, eligible)
        norm_newly_opened = _minmax_normalized_feature(raw_newly_opened, eligible)
        norm_closed = _minmax_normalized_feature(raw_closed, eligible)
        norm_local_syndrome = _minmax_normalized_feature(raw_local_syndrome, eligible)
        norm_base_rank = _minmax_normalized_feature(raw_base_rank, eligible)

        def _candidate_key(column_index: int) -> tuple[float, int, int, int]:
            score = (
                float(weights["slack"]) * float(norm_slack[int(column_index)])
                + float(weights["active_after"]) * float(norm_active[int(column_index)])
                + float(weights["newly_opened"]) * float(norm_newly_opened[int(column_index)])
                + float(weights["closed"]) * float(norm_closed[int(column_index)])
                + float(weights["local_syndrome"]) * float(norm_local_syndrome[int(column_index)])
                + float(weights["base_rank"]) * float(norm_base_rank[int(column_index)])
            )
            return (
                float(score),
                int(deadline_rank[int(column_index)]),
                int(base_rank[int(column_index)]),
                int(original_column_index[int(column_index)]),
            )

        best_column_index = min(eligible, key=_candidate_key)
        active_after, _newly_opened, _closed = _frontier_pressure_counts_for_candidate(
            support_rows=support_rows_by_column[int(best_column_index)],
            processed_by_row=processed_by_row,
            remaining_by_row=remaining_by_row,
            current_active_width=int(current_active_width),
        )
        ordering_list.append(int(best_column_index))
        placed[int(best_column_index)] = True
        current_active_width = int(active_after)
        for row in support_rows_by_column[int(best_column_index)]:
            processed_by_row[int(row)] += 1
            remaining_by_row[int(row)] -= 1

    return tuple(int(value) for value in ordering_list)


def _support_mask_from_rows(rows: Sequence[int]) -> int:
    mask = 0
    for row in rows:
        mask |= 1 << int(row)
    return int(mask)


def _stable_reorder_columns(
    columns_time_ordered: Sequence[progressive.ProgressiveColumn],
    ordering: Sequence[int],
) -> tuple[progressive.ProgressiveColumn, ...]:
    return tuple(columns_time_ordered[int(source_index)] for source_index in tuple(ordering))


def load_custom_column_order_file(path: Path, *, column_count: int) -> tuple[int, ...]:
    path = Path(path)
    if str(path.suffix) == ".npy":
        raw = np.load(path)
        values = tuple(int(value) for value in np.asarray(raw, dtype=np.int64).reshape(-1).tolist())
    else:
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            payload = payload.get("order", payload.get("permutation"))
        if not isinstance(payload, list):
            raise ValueError(f"custom column order file {path} must contain a list or an object with `order`")
        values = tuple(int(value) for value in payload)
    if len(values) != int(column_count):
        raise ValueError(
            f"custom column order length {len(values)} does not match matrix column count {int(column_count)}"
        )
    seen = set(values)
    if len(seen) != len(values):
        raise ValueError("custom column order contains duplicate column indices")
    expected = set(range(int(column_count)))
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(
            f"custom column order is not a permutation of 0..{int(column_count) - 1}; "
            f"missing={missing[:8]} extra={extra[:8]}"
        )
    return values


def _backward_deadline_order_key(
    *,
    columns: Sequence[progressive.ProgressiveColumn],
    support_rows_by_column: Sequence[Sequence[int]],
    first_touch_by_row: Sequence[int],
    last_touch_by_row: Sequence[int],
    column_index: int,
) -> tuple[int, ...]:
    support_rows = tuple(int(row) for row in support_rows_by_column[int(column_index)])
    sentinel = int(len(columns) + 1)
    if not support_rows:
        return (
            int(sentinel),
            int(sentinel),
            int(sentinel),
            -int(columns[int(column_index)].instruction_offset),
            -int(columns[int(column_index)].index),
        )
    last_touch_in_backward_order = [
        int(len(columns)) - 1 - int(first_touch_by_row[int(row)])
        for row in support_rows
    ]
    first_touch_in_backward_order = [
        int(len(columns)) - 1 - int(last_touch_by_row[int(row)])
        for row in support_rows
    ]
    return (
        min(last_touch_in_backward_order),
        max(last_touch_in_backward_order),
        min(first_touch_in_backward_order),
        -int(columns[int(column_index)].instruction_offset),
        -int(columns[int(column_index)].index),
    )


def _bidirectional_deadline_order(
    *,
    columns_time_ordered: Sequence[progressive.ProgressiveColumn],
    num_detectors: int,
) -> tuple[int, ...]:
    (
        support_rows_by_column,
        first_touch_by_row,
        last_touch_by_row,
    ) = _precompute_deadline_order_data(columns_time_ordered, num_detectors=int(num_detectors))

    forward_order = sorted(
        range(len(columns_time_ordered)),
        key=lambda column_index: _deadline_order_key(
            columns=columns_time_ordered,
            support_rows_by_column=support_rows_by_column,
            first_touch_by_row=first_touch_by_row,
            last_touch_by_row=last_touch_by_row,
            column_index=int(column_index),
        ),
    )
    backward_order = sorted(
        range(len(columns_time_ordered)),
        key=lambda column_index: _backward_deadline_order_key(
            columns=columns_time_ordered,
            support_rows_by_column=support_rows_by_column,
            first_touch_by_row=first_touch_by_row,
            last_touch_by_row=last_touch_by_row,
            column_index=int(column_index),
        ),
    )

    remaining = set(range(len(columns_time_ordered)))
    forward_prefix: list[int] = []
    backward_prefix: list[int] = []
    forward_cursor = 0
    backward_cursor = 0
    while remaining:
        while (
            forward_cursor < len(forward_order)
            and int(forward_order[int(forward_cursor)]) not in remaining
        ):
            forward_cursor += 1
        if forward_cursor < len(forward_order):
            column_index = int(forward_order[int(forward_cursor)])
            forward_prefix.append(int(column_index))
            remaining.remove(int(column_index))
        if not remaining:
            break
        while (
            backward_cursor < len(backward_order)
            and int(backward_order[int(backward_cursor)]) not in remaining
        ):
            backward_cursor += 1
        if backward_cursor < len(backward_order):
            column_index = int(backward_order[int(backward_cursor)])
            backward_prefix.append(int(column_index))
            remaining.remove(int(column_index))

    return tuple(int(value) for value in (*tuple(forward_prefix), *tuple(reversed(backward_prefix))))


def _midpoint_backward_candidate_prefix_columns(
    *,
    total_columns: int,
    middle_join_prefix_columns: int | None,
) -> tuple[int, ...]:
    if middle_join_prefix_columns is not None:
        return (
            progressive._normalize_progressive_middle_join_prefix_columns(
                middle_join_prefix_columns,
                total_columns=int(total_columns),
            ),
        )
    candidates = {
        progressive._normalize_progressive_middle_join_prefix_columns(
            max(1, min(int(total_columns) - 1, int(round(float(total_columns) * float(fraction))))),
            total_columns=int(total_columns),
        )
        for fraction in MIDPOINT_BACKWARD_CUT_FRACTIONS
    }
    return tuple(sorted(int(value) for value in candidates))


def _greedy_middle_join_backward_suffix_order(
    *,
    columns_ordered: Sequence[progressive.ProgressiveColumn],
    num_detectors: int,
    processed_original_indices: set[int],
) -> tuple[int, ...]:
    (
        support_rows_by_column,
        first_touch_by_row,
        last_touch_by_row,
    ) = _precompute_deadline_order_data(columns_ordered, num_detectors=int(num_detectors))
    total_touch_count_by_row = [0 for _ in range(int(num_detectors))]
    for rows in support_rows_by_column:
        for row in rows:
            total_touch_count_by_row[int(row)] += 1
    processed_by_row = [0 for _ in range(int(num_detectors))]
    remaining_by_row = list(int(value) for value in total_touch_count_by_row)
    current_active_width = 0
    remaining = {
        int(column_index)
        for column_index, column in enumerate(columns_ordered)
        if int(column.index) not in processed_original_indices
    }
    ordered_suffix: list[int] = []

    def _deadline_key(column_index: int) -> tuple[int, ...]:
        return _deadline_order_key(
            columns=columns_ordered,
            support_rows_by_column=support_rows_by_column,
            first_touch_by_row=first_touch_by_row,
            last_touch_by_row=last_touch_by_row,
            column_index=int(column_index),
        )

    def _active_width_after_add(column_index: int) -> int:
        delta = 0
        for row in support_rows_by_column[int(column_index)]:
            row_index = int(row)
            before_active = bool(
                int(processed_by_row[int(row_index)]) > 0
                and int(remaining_by_row[int(row_index)]) > 0
            )
            after_active = bool(int(remaining_by_row[int(row_index)]) - 1 > 0)
            if bool(after_active) and not bool(before_active):
                delta += 1
            elif bool(before_active) and not bool(after_active):
                delta -= 1
        return int(current_active_width) + int(delta)

    while remaining:
        best_column_index = min(
            remaining,
            key=lambda column_index: (
                int(_active_width_after_add(int(column_index))),
                _deadline_key(int(column_index)),
            ),
        )
        ordered_suffix.append(int(best_column_index))
        current_active_width = int(_active_width_after_add(int(best_column_index)))
        for row in support_rows_by_column[int(best_column_index)]:
            processed_by_row[int(row)] += 1
            remaining_by_row[int(row)] -= 1
        remaining.remove(int(best_column_index))
    return tuple(int(value) for value in ordered_suffix)


def _build_forward_anchored_middle_join_ordered_families(
    *,
    base_family: LoadedProgressiveFamily,
    middle_join_prefix_columns: int | None,
) -> JointMiddleJoinOrderedFamilies:
    total_columns = int(len(base_family.columns))
    candidate_prefixes = _midpoint_backward_candidate_prefix_columns(
        total_columns=int(total_columns),
        middle_join_prefix_columns=middle_join_prefix_columns,
    )
    all_original_indices = {int(column.index) for column in tuple(base_family.columns)}
    best_key: tuple[float, int, int, int] | None = None
    best_summary: JointMiddleJoinOrderedFamilies | None = None
    for prefix_columns in candidate_prefixes:
        suffix_columns = int(total_columns) - int(prefix_columns)
        forward_columns = tuple(base_family.columns)
        processed_original_indices = {
            int(column.index)
            for column in tuple(forward_columns)[: int(prefix_columns)]
        }
        backward_prefix_order = _greedy_middle_join_backward_suffix_order(
            columns_ordered=forward_columns,
            num_detectors=int(base_family.matrix_rows),
            processed_original_indices=processed_original_indices,
        )
        backward_suffix_order = tuple(
            int(column_index)
            for column_index, column in enumerate(forward_columns)
            if int(column.index) in processed_original_indices
        )
        backward_columns = _stable_reorder_columns(
            forward_columns,
            (*tuple(backward_prefix_order), *tuple(backward_suffix_order)),
        )
        backward_layout = progressive.build_frontier_layout(list(backward_columns), num_detectors=int(base_family.matrix_rows))
        forward_prefix_active_area = int(
            sum(int(value) for value in tuple(base_family.layout.active_width_profile)[1 : int(prefix_columns) + 1])
        )
        backward_prefix_active_area = int(
            sum(int(value) for value in tuple(backward_layout.active_width_profile)[1 : int(suffix_columns) + 1])
        )
        cut_boundary_rows = int(
            progressive._progressive_middle_join_boundary_mask(
                columns=forward_columns,
                processed_original_indices=processed_original_indices,
                all_original_indices=all_original_indices,
            ).bit_count()
        )
        forward_family = replace(
            base_family,
            columns=forward_columns,
            layout=base_family.layout,
            column_order_name=(
                f"midpoint backward reorder (forward anchor={str(base_family.column_order_name)}, prefix={int(prefix_columns)})"
            ),
            column_order_source=(
                f"{str(base_family.column_order_name)} forward anchor; only the backward processed suffix is greedily "
                "reordered for midpoint join"
            ),
        )
        backward_family = replace(
            base_family,
            columns=backward_columns,
            layout=backward_layout,
            column_order_name=f"midpoint backward reorder (backward prefix={int(suffix_columns)})",
            column_order_source=(
                f"{str(base_family.column_order_name)} forward anchor with greedy backward-only reordering over the "
                "complementary midpoint suffix"
            ),
        )
        objective = (
            float(forward_prefix_active_area)
            + float(backward_prefix_active_area)
            + float(MIDPOINT_BACKWARD_CUT_BOUNDARY_ROW_WEIGHT) * float(cut_boundary_rows)
        )
        candidate_key = (
            float(objective),
            int(backward_prefix_active_area),
            int(cut_boundary_rows),
            -int(prefix_columns),
        )
        summary = JointMiddleJoinOrderedFamilies(
            prefix_columns=int(prefix_columns),
            suffix_columns=int(suffix_columns),
            forward_family=forward_family,
            backward_family=backward_family,
            cut_boundary_rows=int(cut_boundary_rows),
            forward_prefix_active_area=int(forward_prefix_active_area),
            backward_prefix_active_area=int(backward_prefix_active_area),
        )
        if best_key is None or candidate_key < best_key:
            best_key = candidate_key
            best_summary = summary
    if best_summary is None:
        raise AssertionError("forward-anchored midpoint ordering failed to produce a summary")
    return best_summary


def _build_shared_middle_join_ordered_families(
    *,
    base_family: LoadedProgressiveFamily,
    middle_join_prefix_columns: int | None,
) -> JointMiddleJoinOrderedFamilies:
    anchored_summary = _build_forward_anchored_middle_join_ordered_families(
        base_family=base_family,
        middle_join_prefix_columns=middle_join_prefix_columns,
    )
    prefix_columns = int(anchored_summary.prefix_columns)
    suffix_columns = int(anchored_summary.suffix_columns)
    forward_prefix_columns = tuple(anchored_summary.forward_family.columns[:prefix_columns])
    backward_prefix_columns = tuple(anchored_summary.backward_family.columns[:suffix_columns])
    shared_columns = tuple(
        (*tuple(forward_prefix_columns), *tuple(progressive._reverse_progressive_columns(backward_prefix_columns)))
    )
    shared_layout = progressive.build_frontier_layout(list(shared_columns), num_detectors=int(base_family.matrix_rows))
    backward_columns = tuple(progressive._reverse_progressive_columns(shared_columns))
    backward_layout = progressive.build_frontier_layout(list(backward_columns), num_detectors=int(base_family.matrix_rows))
    shared_family = replace(
        base_family,
        columns=shared_columns,
        layout=shared_layout,
        column_order_name=(
            f"shared MITM order (anchor={str(base_family.column_order_name)}, prefix={int(prefix_columns)})"
        ),
        column_order_source=(
            f"{str(base_family.column_order_name)} forward anchor with the complementary midpoint suffix chosen by "
            "forward-anchored backward-prefix optimization, then reversed into one shared matrix order"
        ),
    )
    reverse_family = replace(
        base_family,
        columns=backward_columns,
        layout=backward_layout,
        column_order_name=(
            f"reverse shared MITM order (anchor={str(base_family.column_order_name)}, suffix={int(suffix_columns)})"
        ),
        column_order_source=(
            f"reverse traversal of {str(shared_family.column_order_name)} for same-matrix MITM decoding"
        ),
    )
    return JointMiddleJoinOrderedFamilies(
        prefix_columns=int(prefix_columns),
        suffix_columns=int(suffix_columns),
        forward_family=shared_family,
        backward_family=reverse_family,
        cut_boundary_rows=int(anchored_summary.cut_boundary_rows),
        forward_prefix_active_area=int(anchored_summary.forward_prefix_active_area),
        backward_prefix_active_area=int(anchored_summary.backward_prefix_active_area),
    )


def _build_backward_deadline_ordered_family(
    *,
    base_family: LoadedProgressiveFamily,
) -> LoadedProgressiveFamily:
    reversed_columns = progressive._reverse_progressive_columns(base_family.columns)
    reordered_columns, _ordering = progressive.optimize_column_order(
        list(reversed_columns),
        num_detectors=int(base_family.matrix_rows),
    )
    backward_layout = progressive.build_frontier_layout(
        list(reordered_columns),
        num_detectors=int(base_family.matrix_rows),
    )
    return replace(
        base_family,
        columns=tuple(reordered_columns),
        layout=backward_layout,
        column_order_name=f"backward deadline reorder (anchor={str(base_family.column_order_name)})",
        column_order_source=(
            f"reverse of {str(base_family.column_order_name)}, then closure-aware optimize_column_order on the reversed family"
        ),
    )


def _build_backward_pressure_ordered_family(
    *,
    base_family: LoadedProgressiveFamily,
    column_order: str,
) -> LoadedProgressiveFamily:
    order_key = str(column_order).strip().lower()
    if order_key not in {"back_deadline_min_active_w32", "back_deadline_close_first_w32"}:
        raise ValueError(f"unsupported backward pressure order {column_order!r}")
    reversed_columns = progressive._reverse_progressive_columns(base_family.columns)
    support_rows_by_column, first_touch_by_row, last_touch_by_row = _precompute_deadline_order_data(
        reversed_columns,
        num_detectors=int(base_family.matrix_rows),
    )
    pressure_mode = "close_first" if order_key == "back_deadline_close_first_w32" else "min_active"
    ordering = _deadline_window_pressure_order(
        columns_ordered=reversed_columns,
        num_detectors=int(base_family.matrix_rows),
        support_rows_by_column=support_rows_by_column,
        first_touch_by_row=first_touch_by_row,
        last_touch_by_row=last_touch_by_row,
        mode=str(pressure_mode),
        window=int(DEADLINE_PRESSURE_WINDOW),
    )
    reordered_columns = [
        replace(reversed_columns[int(source_index)], index=int(target_index))
        for target_index, source_index in enumerate(ordering)
    ]
    backward_layout = progressive.build_frontier_layout(
        list(reordered_columns),
        num_detectors=int(base_family.matrix_rows),
    )
    return replace(
        base_family,
        columns=tuple(reordered_columns),
        layout=backward_layout,
        column_order_name=(
            f"backward deadline close-first pressure reorder (window={int(DEADLINE_PRESSURE_WINDOW)})"
            if pressure_mode == "close_first"
            else f"backward deadline min-active pressure reorder (window={int(DEADLINE_PRESSURE_WINDOW)})"
        ),
        column_order_source=(
            f"reverse of {str(base_family.column_order_name)}, then deadline-slack-window "
            f"{str(pressure_mode)} active-frontier pressure reorder; local syndrome feature defaults to zero"
        ),
    )


def _joint_middle_join_order_pair(
    *,
    columns_time_ordered: Sequence[progressive.ProgressiveColumn],
    num_detectors: int,
    middle_join_prefix_columns: int | None,
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    total_columns = int(len(columns_time_ordered))
    prefix_columns = progressive._normalize_progressive_middle_join_prefix_columns(
        middle_join_prefix_columns,
        total_columns=int(total_columns),
    )
    suffix_columns = int(total_columns) - int(prefix_columns)
    (
        support_rows_by_column,
        first_touch_by_row,
        last_touch_by_row,
    ) = _precompute_deadline_order_data(columns_time_ordered, num_detectors=int(num_detectors))
    support_masks_by_column = tuple(
        int(_support_mask_from_rows(rows))
        for rows in support_rows_by_column
    )
    total_touch_count_by_row = [0 for _ in range(int(num_detectors))]
    for rows in support_rows_by_column:
        for row in rows:
            total_touch_count_by_row[int(row)] += 1

    def _deadline_key(column_index: int) -> tuple[int, ...]:
        return _deadline_order_key(
            columns=columns_time_ordered,
            support_rows_by_column=support_rows_by_column,
            first_touch_by_row=first_touch_by_row,
            last_touch_by_row=last_touch_by_row,
            column_index=int(column_index),
        )

    forward_processed_by_row = [0 for _ in range(int(num_detectors))]
    forward_remaining_by_row = list(int(value) for value in total_touch_count_by_row)
    backward_processed_by_row = [0 for _ in range(int(num_detectors))]
    backward_remaining_by_row = list(int(value) for value in total_touch_count_by_row)
    forward_active_width = 0
    backward_active_width = 0
    forward_touched_mask = 0
    backward_touched_mask = 0
    boundary_mask = 0
    remaining = set(range(int(total_columns)))
    forward_order: list[int] = []
    backward_order: list[int] = []

    def _active_width_after_add(
        *,
        column_index: int,
        processed_by_row: list[int],
        remaining_by_row: list[int],
        current_active_width: int,
    ) -> int:
        delta = 0
        for row in support_rows_by_column[int(column_index)]:
            row_index = int(row)
            before_active = bool(
                int(processed_by_row[int(row_index)]) > 0
                and int(remaining_by_row[int(row_index)]) > 0
            )
            after_active = bool(int(remaining_by_row[int(row_index)]) - 1 > 0)
            if bool(after_active) and not bool(before_active):
                delta += 1
            elif bool(before_active) and not bool(after_active):
                delta -= 1
        return int(current_active_width) + int(delta)

    def _select_next_column(side: str) -> int:
        side_key = str(side)
        if side_key == "forward":
            processed_by_row = forward_processed_by_row
            remaining_by_row = forward_remaining_by_row
            current_active_width = int(forward_active_width)
            opposite_touched_mask = int(backward_touched_mask)
        else:
            processed_by_row = backward_processed_by_row
            remaining_by_row = backward_remaining_by_row
            current_active_width = int(backward_active_width)
            opposite_touched_mask = int(forward_touched_mask)
        best_key: tuple[float, int, int, tuple[int, ...]] | None = None
        best_column_index = -1
        for column_index in sorted(remaining):
            next_active_width = _active_width_after_add(
                column_index=int(column_index),
                processed_by_row=processed_by_row,
                remaining_by_row=remaining_by_row,
                current_active_width=int(current_active_width),
            )
            new_boundary_rows = int(
                (int(support_masks_by_column[int(column_index)]) & int(opposite_touched_mask) & ~int(boundary_mask)).bit_count()
            )
            candidate_key = (
                float(next_active_width) + float(MIDPOINT_JOINT_BOUNDARY_WEIGHT) * float(new_boundary_rows),
                int(new_boundary_rows),
                int(next_active_width),
                _deadline_key(int(column_index)),
            )
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_column_index = int(column_index)
        if int(best_column_index) < 0:
            raise AssertionError("joint midpoint ordering failed to select a next column")
        return int(best_column_index)

    while len(forward_order) < int(prefix_columns) or len(backward_order) < int(suffix_columns):
        if len(forward_order) >= int(prefix_columns):
            side_to_fill = "backward"
        elif len(backward_order) >= int(suffix_columns):
            side_to_fill = "forward"
        else:
            forward_fill = float(len(forward_order)) / float(max(1, int(prefix_columns)))
            backward_fill = float(len(backward_order)) / float(max(1, int(suffix_columns)))
            if float(forward_fill) < float(backward_fill):
                side_to_fill = "forward"
            elif float(backward_fill) < float(forward_fill):
                side_to_fill = "backward"
            else:
                forward_candidate = _select_next_column("forward")
                backward_candidate = _select_next_column("backward")
                forward_key = (
                    float(
                        _active_width_after_add(
                            column_index=int(forward_candidate),
                            processed_by_row=forward_processed_by_row,
                            remaining_by_row=forward_remaining_by_row,
                            current_active_width=int(forward_active_width),
                        )
                    )
                    + float(MIDPOINT_JOINT_BOUNDARY_WEIGHT)
                    * float(
                        (
                            int(support_masks_by_column[int(forward_candidate)])
                            & int(backward_touched_mask)
                            & ~int(boundary_mask)
                        ).bit_count()
                    )
                )
                backward_key = (
                    float(
                        _active_width_after_add(
                            column_index=int(backward_candidate),
                            processed_by_row=backward_processed_by_row,
                            remaining_by_row=backward_remaining_by_row,
                            current_active_width=int(backward_active_width),
                        )
                    )
                    + float(MIDPOINT_JOINT_BOUNDARY_WEIGHT)
                    * float(
                        (
                            int(support_masks_by_column[int(backward_candidate)])
                            & int(forward_touched_mask)
                            & ~int(boundary_mask)
                        ).bit_count()
                    )
                )
                side_to_fill = "forward" if float(forward_key) <= float(backward_key) else "backward"
        column_index = _select_next_column(str(side_to_fill))
        remaining.remove(int(column_index))
        support_rows = support_rows_by_column[int(column_index)]
        support_mask = int(support_masks_by_column[int(column_index)])
        if str(side_to_fill) == "forward":
            forward_order.append(int(column_index))
            forward_active_width = _active_width_after_add(
                column_index=int(column_index),
                processed_by_row=forward_processed_by_row,
                remaining_by_row=forward_remaining_by_row,
                current_active_width=int(forward_active_width),
            )
            for row in support_rows:
                forward_processed_by_row[int(row)] += 1
                forward_remaining_by_row[int(row)] -= 1
            forward_touched_mask |= int(support_mask)
            boundary_mask |= int(support_mask) & int(backward_touched_mask)
        else:
            backward_order.append(int(column_index))
            backward_active_width = _active_width_after_add(
                column_index=int(column_index),
                processed_by_row=backward_processed_by_row,
                remaining_by_row=backward_remaining_by_row,
                current_active_width=int(backward_active_width),
            )
            for row in support_rows:
                backward_processed_by_row[int(row)] += 1
                backward_remaining_by_row[int(row)] -= 1
            backward_touched_mask |= int(support_mask)
            boundary_mask |= int(support_mask) & int(forward_touched_mask)

    return (
        tuple(int(value) for value in (*tuple(forward_order), *tuple(backward_order))),
        tuple(int(value) for value in (*tuple(backward_order), *tuple(forward_order))),
        int(prefix_columns),
    )


def _build_joint_middle_join_ordered_families(
    *,
    base_family: LoadedProgressiveFamily,
    middle_join_prefix_columns: int | None,
) -> JointMiddleJoinOrderedFamilies:
    total_columns = int(len(base_family.columns))
    forward_order, backward_order, prefix_columns = _joint_middle_join_order_pair(
        columns_time_ordered=base_family.columns,
        num_detectors=int(base_family.matrix_rows),
        middle_join_prefix_columns=middle_join_prefix_columns,
    )
    suffix_columns = int(total_columns) - int(prefix_columns)
    forward_columns = _stable_reorder_columns(base_family.columns, forward_order)
    backward_columns = _stable_reorder_columns(base_family.columns, backward_order)
    forward_layout = progressive.build_frontier_layout(list(forward_columns), num_detectors=int(base_family.matrix_rows))
    backward_layout = progressive.build_frontier_layout(list(backward_columns), num_detectors=int(base_family.matrix_rows))
    forward_prefix_active_area = int(sum(int(value) for value in tuple(forward_layout.active_width_profile)[1 : int(prefix_columns) + 1]))
    backward_prefix_active_area = int(sum(int(value) for value in tuple(backward_layout.active_width_profile)[1 : int(suffix_columns) + 1]))
    all_original_indices = {int(column.index) for column in tuple(forward_columns)}
    processed_original_indices = {
        int(column.index)
        for column in tuple(forward_columns)[: int(prefix_columns)]
    }
    cut_boundary_rows = int(
        progressive._progressive_middle_join_boundary_mask(
            columns=forward_columns,
            processed_original_indices=processed_original_indices,
            all_original_indices=all_original_indices,
        ).bit_count()
    )
    forward_family = replace(
        base_family,
        columns=forward_columns,
        layout=forward_layout,
        column_order_name=f"midpoint joint reorder (forward prefix={int(prefix_columns)})",
        column_order_source=(
            "time-order columns greedily partitioned to jointly minimize forward-prefix active area, "
            "backward-prefix active area, and cut-boundary rows"
        ),
    )
    backward_family = replace(
        base_family,
        columns=backward_columns,
        layout=backward_layout,
        column_order_name=f"midpoint joint reorder (backward prefix={int(suffix_columns)})",
        column_order_source=(
            "time-order columns greedily partitioned to jointly minimize forward-prefix active area, "
            "backward-prefix active area, and cut-boundary rows"
        ),
    )
    return JointMiddleJoinOrderedFamilies(
        prefix_columns=int(prefix_columns),
        suffix_columns=int(suffix_columns),
        forward_family=forward_family,
        backward_family=backward_family,
        cut_boundary_rows=int(cut_boundary_rows),
        forward_prefix_active_area=int(forward_prefix_active_area),
        backward_prefix_active_area=int(backward_prefix_active_area),
    )


def _triangle_patch_deadline_order(
    *,
    columns_time_ordered: Sequence[progressive.ProgressiveColumn],
    detector_matrix,
    logical_matrix,
    metadata: SplitSectorMetadata,
    ordered_natural_columns: np.ndarray,
    base_deadline_ordering: Sequence[int],
    patch_fraction: float,
) -> tuple[tuple[int, ...], str, str]:
    if detector_matrix is None or logical_matrix is None or metadata is None:
        raise ValueError("triangle-aware column orders require detector/logical matrices plus split-sector metadata")
    ordered_natural = np.asarray(ordered_natural_columns, dtype=np.int32).reshape(-1)
    if ordered_natural.size != len(columns_time_ordered):
        raise ValueError("ordered_natural_columns must match the time-ordered column count")
    deadline_ordering = tuple(int(value) for value in base_deadline_ordering)
    if len(deadline_ordering) != len(columns_time_ordered):
        raise ValueError("base_deadline_ordering must match the time-ordered column count")

    catalog = catalog_exact_local_triangles(
        matrix=detector_matrix,
        observables=logical_matrix,
        metadata=metadata,
        sector="triangle_reorder",
    )
    selection = select_nonoverlapping_triangle_relations(catalog)
    if not selection.selected_relations:
        return (
            deadline_ordering,
            "triangle-aware deadline patch reorder (no local triangles available)",
            "deadline_reorder unchanged because the local exact-triangle catalog is empty",
        )

    natural_to_time = np.empty(int(ordered_natural.size), dtype=np.int32)
    natural_to_time[ordered_natural] = np.arange(int(ordered_natural.size), dtype=np.int32)
    deadline_rank_by_time = np.empty(int(len(deadline_ordering)), dtype=np.int32)
    deadline_rank_by_time[np.asarray(deadline_ordering, dtype=np.int32)] = np.arange(int(len(deadline_ordering)), dtype=np.int32)

    candidates: list[tuple[int, int, tuple[int, int, int, int, int], tuple[int, int, int]]] = []
    for relation in selection.selected_relations:
        time_columns_unsorted = tuple(int(natural_to_time[int(col)]) for col in relation.columns)
        time_columns = tuple(
            sorted(
                time_columns_unsorted,
                key=lambda value: int(deadline_rank_by_time[int(value)]),
            )
        )
        ranks = [int(deadline_rank_by_time[int(value)]) for value in time_columns]
        current_span = int(max(ranks) - min(ranks))
        current_min_rank = int(min(ranks))
        candidates.append((current_span, current_min_rank, relation.selection_key, time_columns))
    candidates.sort(key=lambda item: (int(item[0]), int(item[1]), item[2]))

    if float(patch_fraction) >= 1.0:
        selected_count = int(len(candidates))
    else:
        selected_count = int(math.ceil(float(patch_fraction) * float(len(candidates))))
    selected_count = max(0, min(int(selected_count), int(len(candidates))))
    if selected_count <= 0:
        return (
            deadline_ordering,
            "triangle-aware deadline patch reorder (empty selection)",
            "deadline_reorder unchanged because the requested local-triangle patch fraction selects zero relations",
        )

    selected_blocks = tuple(tuple(int(col) for col in item[3]) for item in candidates[:selected_count])
    block_id_by_column: dict[int, int] = {}
    for block_id, cols in enumerate(selected_blocks):
        for col in cols:
            block_id_by_column[int(col)] = int(block_id)

    patched_ordering: list[int] = []
    emitted_blocks: set[int] = set()
    for col in deadline_ordering:
        block_id = block_id_by_column.get(int(col))
        if block_id is None:
            patched_ordering.append(int(col))
            continue
        if int(block_id) in emitted_blocks:
            continue
        patched_ordering.extend(int(value) for value in selected_blocks[int(block_id)])
        emitted_blocks.add(int(block_id))

    ordering = tuple(int(value) for value in patched_ordering)
    if sorted(ordering) != list(range(len(columns_time_ordered))):
        raise ValueError("triangle-aware deadline patch produced an invalid column permutation")

    span_values = [int(item[0]) for item in candidates[:selected_count]]
    median_span = float(np.median(np.asarray(span_values, dtype=np.float64))) if span_values else float("nan")
    if float(patch_fraction) >= 1.0:
        column_order_name = "triangle-aware deadline patch reorder (all selected local triangles)"
        selection_text = f"all `{int(selected_count)}` selected non-overlapping local exact triangles"
    else:
        fraction_pct = int(round(float(patch_fraction) * 100.0))
        column_order_name = f"triangle-aware deadline patch reorder ({fraction_pct}% local triangles)"
        selection_text = (
            f"the smallest-span `{int(selected_count)}` of `{int(len(candidates))}` selected non-overlapping "
            f"local exact triangles (`{fraction_pct}%` patch fraction)"
        )
    column_order_source = (
        "deadline_reorder followed by a contiguity patch that emits "
        f"{selection_text} as 3-column blocks in their existing deadline-relative order; "
        f"patched-relation median pre-patch span=`{median_span:.1f}` columns"
    )
    return ordering, column_order_name, column_order_source


def _ordered_columns_by_mode(
    *,
    columns_time_ordered: Sequence[progressive.ProgressiveColumn],
    column_round_start: np.ndarray,
    column_round_stop: np.ndarray,
    num_detectors: int,
    column_order: str,
    column_order_file: Path | None = None,
    detector_matrix=None,
    logical_matrix=None,
    metadata: SplitSectorMetadata | None = None,
    ordered_natural_columns: np.ndarray | None = None,
) -> tuple[list[progressive.ProgressiveColumn], tuple[int, ...], str, str]:
    order_key = str(column_order).strip().lower()
    if order_key not in COLUMN_ORDER_CHOICES:
        raise ValueError(f"unsupported column_order {column_order!r}; expected one of {list(COLUMN_ORDER_CHOICES)}")
    if order_key in {"midpoint_joint_reorder", "midpoint_backward_reorder", "shared_mitm_order"} | set(BACKWARD_DERIVED_COLUMN_ORDER_CHOICES):
        raise ValueError(f"{order_key} must be built through the derived backward/midpoint family loader")
    if len(columns_time_ordered) <= 1:
        return (
            list(columns_time_ordered),
            tuple(range(len(columns_time_ordered))),
            "metadata time order",
            "metadata.ordered_column_index",
        )
    if order_key == "custom_file":
        if column_order_file is None:
            raise ValueError("--column-order custom_file requires --column-order-file")
        ordering = load_custom_column_order_file(Path(column_order_file), column_count=len(columns_time_ordered))
        return (
            [
                replace(columns_time_ordered[int(source_index)], index=int(target_index))
                for target_index, source_index in enumerate(ordering)
            ],
            ordering,
            "custom file column order",
            f"explicit permutation loaded from `{Path(column_order_file)}`",
        )
    if order_key == "time_order":
        ordering = tuple(range(len(columns_time_ordered)))
        return (
            [replace(columns_time_ordered[int(index)], index=int(index)) for index in ordering],
            ordering,
            "metadata time order",
            "metadata.ordered_column_index",
        )
    if order_key in TRIANGLE_PATCH_ORDER_FRACTIONS:
        if ordered_natural_columns is None:
            raise ValueError("triangle-aware column orders require ordered_natural_columns")
        _deadline_reordered_columns, deadline_ordering = progressive.optimize_column_order(
            list(columns_time_ordered),
            num_detectors=int(num_detectors),
        )
        ordering, column_order_name, column_order_source = _triangle_patch_deadline_order(
            columns_time_ordered=columns_time_ordered,
            detector_matrix=detector_matrix,
            logical_matrix=logical_matrix,
            metadata=metadata,
            ordered_natural_columns=np.asarray(ordered_natural_columns, dtype=np.int32),
            base_deadline_ordering=deadline_ordering,
            patch_fraction=float(TRIANGLE_PATCH_ORDER_FRACTIONS[order_key]),
        )
        return (
            [replace(columns_time_ordered[int(source_index)], index=int(target_index)) for target_index, source_index in enumerate(ordering)],
            ordering,
            column_order_name,
            column_order_source,
        )
    if order_key in FORWARD_DEADLINE_ORDER_ALIASES:
        reordered, ordering = progressive.optimize_column_order(
            list(columns_time_ordered),
            num_detectors=int(num_detectors),
        )
        return (
            reordered,
            ordering,
            "deadline-style closure-aware reorder",
            "metadata.ordered_column_index re-sorted by earliest detector-row last-touch",
        )
    if order_key == "bidirectional_deadline_reorder":
        ordering = _bidirectional_deadline_order(
            columns_time_ordered=columns_time_ordered,
            num_detectors=int(num_detectors),
        )
        return (
            [
                replace(columns_time_ordered[int(source_index)], index=int(target_index))
                for target_index, source_index in enumerate(ordering)
            ],
            ordering,
            "single-matrix bidirectional deadline reorder",
            (
                "metadata.ordered_column_index two-ended interleave of forward and backward deadline ranks; "
                "use the default reverse_forward_columns backward traversal to keep one shared matrix"
            ),
        )

    support_rows_by_column, first_touch_by_row, last_touch_by_row = _precompute_deadline_order_data(
        columns_time_ordered,
        num_detectors=int(num_detectors),
    )

    def _deadline_key(column_index: int) -> tuple[int, ...]:
        return _deadline_order_key(
            columns=columns_time_ordered,
            support_rows_by_column=support_rows_by_column,
            first_touch_by_row=first_touch_by_row,
            last_touch_by_row=last_touch_by_row,
            column_index=int(column_index),
        )

    ordering_list: list[int] = []
    if order_key == "span_deadline_reorder":
        grouped_indices: dict[tuple[int, int], list[int]] = {}
        for column_index in range(len(columns_time_ordered)):
            group_key = (int(column_round_start[int(column_index)]), int(column_round_stop[int(column_index)]))
            grouped_indices.setdefault(group_key, []).append(int(column_index))
        for group_key in sorted(grouped_indices):
            ordering_list.extend(sorted(grouped_indices[group_key], key=_deadline_key))
        column_order_name = "span-preserving deadline hybrid reorder"
        column_order_source = "metadata.ordered_column_index with deadline sort restricted to each inferred (round_start, round_stop) block"
    elif order_key == "natural_inside_span":
        grouped_indices = {}
        for column_index in range(len(columns_time_ordered)):
            group_key = (int(column_round_start[int(column_index)]), int(column_round_stop[int(column_index)]))
            grouped_indices.setdefault(group_key, []).append(int(column_index))
        for group_key in sorted(grouped_indices):
            ordering_list.extend(grouped_indices[group_key])
        column_order_name = "natural-inside-span reorder"
        column_order_source = (
            "metadata.ordered_column_index with original metadata order preserved inside each inferred "
            "(round_start, round_stop) block"
        )
    elif order_key == "bridge_zipper_reorder":
        local_by_round: dict[int, list[int]] = {}
        bridge_by_round: dict[int, list[int]] = {}
        for column_index in range(len(columns_time_ordered)):
            round_start = int(column_round_start[int(column_index)])
            round_stop = int(column_round_stop[int(column_index)])
            if round_stop <= round_start:
                local_by_round.setdefault(round_start, []).append(int(column_index))
            else:
                bridge_by_round.setdefault(round_start, []).append(int(column_index))
        for round_index in sorted(set(local_by_round) | set(bridge_by_round)):
            local_block = sorted(local_by_round.get(int(round_index), []), key=_deadline_key)
            bridge_block = sorted(bridge_by_round.get(int(round_index), []), key=_deadline_key)
            local_prefix_count = len(local_block) if not bridge_block else (len(local_block) + 1) // 2
            ordering_list.extend(local_block[:local_prefix_count])
            ordering_list.extend(bridge_block)
            ordering_list.extend(local_block[local_prefix_count:])
        column_order_name = "bridge-aware zipper reorder"
        column_order_source = (
            "metadata.ordered_column_index grouped by round spans; within each start-round block, place an "
            "early local-prefix, then bridge columns, then the remaining local columns"
        )
    elif order_key == "rank_gain_per_open_row":
        ordering_list = list(
            _greedy_rank_gain_per_open_row_order(
                columns_ordered=columns_time_ordered,
                num_detectors=int(num_detectors),
                support_rows_by_column=support_rows_by_column,
                first_touch_by_row=first_touch_by_row,
                last_touch_by_row=last_touch_by_row,
            )
        )
        column_order_name = "rank-gain-per-open-row greedy reorder"
        column_order_source = (
            "metadata.ordered_column_index greedily re-sorted to maximize detector/logical rank gain per newly "
            "opened detector row, with closure-aware deadline ties"
        )
    elif order_key == "logical_frontload_reorder":
        ordering_list = list(
            _logical_frontload_deadline_order(
                columns_ordered=columns_time_ordered,
                support_rows_by_column=support_rows_by_column,
                first_touch_by_row=first_touch_by_row,
                last_touch_by_row=last_touch_by_row,
            )
        )
        column_order_name = "logical-frontloaded deadline reorder"
        column_order_source = (
            "metadata.ordered_column_index re-sorted to expose observable/logical-response columns first, "
            "then closure-aware deadline order; uses decoder-visible logical response masks only"
        )
    elif order_key in {
        "row_band_round_robin_reorder",
        "row_band_reverse_round_robin_reorder",
        "row_band_center_out_reorder",
    }:
        band_order_mode = (
            "descending"
            if order_key == "row_band_reverse_round_robin_reorder"
            else "center_out"
            if order_key == "row_band_center_out_reorder"
            else "ascending"
        )
        ordering_list = list(
            _row_band_round_robin_deadline_order(
                columns_ordered=columns_time_ordered,
                num_detectors=int(num_detectors),
                support_rows_by_column=support_rows_by_column,
                first_touch_by_row=first_touch_by_row,
                last_touch_by_row=last_touch_by_row,
                band_order_mode=str(band_order_mode),
            )
        )
        column_order_name = (
            "row-band reverse round-robin deadline reorder"
            if order_key == "row_band_reverse_round_robin_reorder"
            else "row-band center-out round-robin deadline reorder"
            if order_key == "row_band_center_out_reorder"
            else "row-band round-robin deadline reorder"
        )
        column_order_source = (
            f"metadata.ordered_column_index first deadline-sorted, then interleaved {str(band_order_mode)} "
            "round-robin across eight detector-row bands by median support row to avoid spending a long prefix "
            "in one structural band"
        )
    elif order_key in {"round_span_zigzag_reorder", "round_span_center_out_reorder"}:
        grouped_indices = {}
        for column_index in range(len(columns_time_ordered)):
            group_key = (int(column_round_start[int(column_index)]), int(column_round_stop[int(column_index)]))
            grouped_indices.setdefault(group_key, []).append(int(column_index))
        group_mode = "center_out" if order_key == "round_span_center_out_reorder" else "zigzag"
        for group_key in _round_span_group_order(tuple(grouped_indices), mode=str(group_mode)):
            ordering_list.extend(sorted(grouped_indices[group_key], key=_deadline_key))
        column_order_name = (
            "round-span center-out deadline reorder"
            if order_key == "round_span_center_out_reorder"
            else "round-span zigzag deadline reorder"
        )
        column_order_source = (
            f"metadata.ordered_column_index grouped by inferred (round_start, round_stop), visiting groups "
            f"in {str(group_mode)} order and using closure-aware deadline order inside each group"
        )
    elif order_key in {"mass_desc_deadline_reorder", "mass_asc_deadline_reorder"}:
        reverse_mass = order_key == "mass_desc_deadline_reorder"
        ordering_list = sorted(
            range(len(columns_time_ordered)),
            key=lambda column_index: (
                -float(_column_error_mass(columns_time_ordered[int(column_index)]))
                if reverse_mass
                else float(_column_error_mass(columns_time_ordered[int(column_index)])),
                _deadline_key(int(column_index)),
            ),
        )
        column_order_name = (
            "error-mass descending deadline reorder"
            if reverse_mass
            else "error-mass ascending deadline reorder"
        )
        column_order_source = (
            "metadata.ordered_column_index sorted by online column prior non-identity mass "
            f"({'descending' if reverse_mass else 'ascending'}), with closure-aware deadline ties"
        )
    elif order_key in {"detector_weight_asc_reorder", "detector_weight_desc_reorder"}:
        reverse_weight = order_key == "detector_weight_desc_reorder"
        ordering_list = sorted(
            range(len(columns_time_ordered)),
            key=lambda column_index: (
                -int(
                    _column_detector_weight(
                        columns_time_ordered[int(column_index)],
                        support_rows_by_column[int(column_index)],
                    )
                )
                if reverse_weight
                else int(
                    _column_detector_weight(
                        columns_time_ordered[int(column_index)],
                        support_rows_by_column[int(column_index)],
                    )
                ),
                _deadline_key(int(column_index)),
            ),
        )
        column_order_name = (
            "detector-weight descending deadline reorder"
            if reverse_weight
            else "detector-weight ascending deadline reorder"
        )
        column_order_source = (
            "metadata.ordered_column_index sorted by detector support weight "
            f"({'descending' if reverse_weight else 'ascending'}), with closure-aware deadline ties"
        )
    elif order_key in {"deadline_min_active_w32", "deadline_close_first_w32"}:
        pressure_mode = "close_first" if order_key == "deadline_close_first_w32" else "min_active"
        ordering_list = list(
            _deadline_window_pressure_order(
                columns_ordered=columns_time_ordered,
                num_detectors=int(num_detectors),
                support_rows_by_column=support_rows_by_column,
                first_touch_by_row=first_touch_by_row,
                last_touch_by_row=last_touch_by_row,
                mode=str(pressure_mode),
                window=int(DEADLINE_PRESSURE_WINDOW),
            )
        )
        column_order_name = (
            f"deadline close-first pressure reorder (window={int(DEADLINE_PRESSURE_WINDOW)})"
            if pressure_mode == "close_first"
            else f"deadline min-active pressure reorder (window={int(DEADLINE_PRESSURE_WINDOW)})"
        )
        column_order_source = (
            "metadata.ordered_column_index greedily re-sorted within a deadline-slack window using "
            f"{str(pressure_mode)} active-frontier pressure features; local syndrome feature defaults to zero"
        )
    elif order_key in {"frontier_width_greedy_reorder", "close_first_greedy_reorder"}:
        greedy_mode = "close_first" if order_key == "close_first_greedy_reorder" else "frontier_width"
        ordering_list = list(
            _greedy_frontier_shape_order(
                columns_ordered=columns_time_ordered,
                num_detectors=int(num_detectors),
                support_rows_by_column=support_rows_by_column,
                first_touch_by_row=first_touch_by_row,
                last_touch_by_row=last_touch_by_row,
                mode=str(greedy_mode),
            )
        )
        column_order_name = (
            "close-first greedy deadline reorder"
            if order_key == "close_first_greedy_reorder"
            else "frontier-width greedy deadline reorder"
        )
        column_order_source = (
            f"metadata.ordered_column_index greedily re-sorted by {str(greedy_mode)} frontier-shape metrics, "
            "with closure-aware deadline ties"
        )
    else:
        for block_start in range(0, len(columns_time_ordered), int(LOCAL_DEADLINE_WINDOW)):
            block_stop = min(len(columns_time_ordered), int(block_start) + int(LOCAL_DEADLINE_WINDOW))
            block = list(range(int(block_start), int(block_stop)))
            ordering_list.extend(sorted(block, key=_deadline_key))
        column_order_name = f"local-window deadline reorder (window={int(LOCAL_DEADLINE_WINDOW)})"
        column_order_source = (
            f"metadata.ordered_column_index with deadline sort restricted to consecutive {int(LOCAL_DEADLINE_WINDOW)}-column windows"
        )

    ordering = tuple(int(value) for value in ordering_list)
    reordered_columns = [
        replace(columns_time_ordered[int(source_index)], index=int(target_index))
        for target_index, source_index in enumerate(ordering)
    ]
    return reordered_columns, ordering, column_order_name, column_order_source


def _load_dem_family(
    *,
    backend: str,
    p_location: float,
    scope: str,
    initial_data_error_rate: float | None = None,
    correction_state_mode: str = "none",
    require_correction_cache: bool = False,
    column_order: str = "deadline_reorder",
    column_order_file: Path | None = None,
    stim_path: Path | None = None,
    external_benchmark_label: str | None = None,
    external_noisy_rounds: int | None = None,
    external_perfect_rounds: int = 1,
) -> LoadedProgressiveFamily:
    correction_state_mode_key = str(correction_state_mode).strip().lower()
    if correction_state_mode_key not in {"none", "full", "logical_class", "stabilizer_quotient"}:
        raise ValueError(
            "correction_state_mode must be one of ['none', 'full', 'logical_class', 'stabilizer_quotient']"
        )
    problem = None
    loaded_side = None
    metadata = None
    detector = None
    logical = None
    priors = None
    benchmark_source_note = ""
    if stim_path is not None:
        if str(correction_state_mode_key) != "none":
            raise ValueError("external Stim loading currently supports only --correction-state-mode none")
        if bool(require_correction_cache):
            raise ValueError("external Stim loading does not support --require-correction-cache")
        if external_noisy_rounds is None or int(external_noisy_rounds) <= 0:
            raise ValueError("external Stim loading requires --external-noisy-rounds > 0")
        loaded_side = load_dem_side_with_metadata_from_stim(
            stim_path=Path(stim_path),
            backend=str(backend),
            sector=("X" if str(scope) == "memory_X" else "Z"),
            error_rate=float(p_location),
            noisy_rounds=int(external_noisy_rounds),
            perfect_rounds=int(external_perfect_rounds),
        )
        benchmark_title, benchmark_description, benchmark_source_note = _external_benchmark_descriptor(
            backend=str(backend),
            benchmark_label=(
                str(external_benchmark_label)
                if external_benchmark_label is not None and str(external_benchmark_label).strip()
                else f"{str(backend)} external Stim DEM"
            ),
            stim_path=Path(stim_path),
        )
    else:
        problem = build_split_sector_problem(
            backend=str(backend),
            error_rate=float(p_location),
            initial_data_error_rate=initial_data_error_rate,
        )
        benchmark_title, benchmark_description = _benchmark_descriptor(str(backend))
        init_note = (
            ""
            if initial_data_error_rate is None
            else f", initial_data_error_rate={float(initial_data_error_rate):.6g}"
        )
        benchmark_source_note = (
            f"{benchmark_description} from `grosscode.dem.builder.build_split_sector_problem(backend={backend!r}, "
            f"error_rate={float(p_location):.6g}{init_note})`."
        )
    scope_key = str(scope)
    correction_by_signature: dict[int, int] = {}
    correction_state_bits = 0
    family_key_suffix = _correction_state_mode_family_suffix(str(correction_state_mode_key))
    if scope_key == "memory_X":
        if stim_path is not None:
            assert loaded_side is not None
            detector = loaded_side.check_matrix.tocsc()
            logical = loaded_side.observables_matrix.tocsc()
            priors = np.asarray(loaded_side.priors, dtype=np.float64).reshape(-1)
            metadata = loaded_side.metadata
        else:
            detector = problem.D_X.tocsc()
            logical = problem.O_X.tocsc()
            priors = np.asarray(problem.priors_X, dtype=np.float64).reshape(-1)
            metadata = problem.metadata_X
        family_key = f"binary_dem_x{family_key_suffix}"
        scope_label = "X side"
        detector_symbol = "D_X"
        logical_symbol = "O_X"
        metadata_symbol = "metadata_X"
        priors_symbol = "priors_X"
        if str(correction_state_mode_key) != "none":
            correction_by_signature, final_qubits = build_gross_split_sector_merged_correction_map(
                sector="X",
                backend=str(backend),
                error_rate=float(p_location),
                cache_policy=("require_disk" if bool(require_correction_cache) else "build_if_missing"),
            )
            if str(correction_state_mode_key) == "full":
                correction_state_bits = int(len(final_qubits))
            else:
                projection_rows = _correction_projection_rows(
                    problem=problem,
                    scope=str(scope_key),
                    correction_state_mode=str(correction_state_mode_key),
                )
                assert projection_rows is not None
                correction_by_signature = _project_correction_map_by_signature(
                    correction_by_signature,
                    row_support_masks=_row_support_masks(np.asarray(projection_rows, dtype=np.uint8)),
                )
                correction_state_bits = int(np.asarray(projection_rows, dtype=np.uint8).shape[0])
    elif scope_key == "memory_Z":
        if stim_path is not None:
            assert loaded_side is not None
            detector = loaded_side.check_matrix.tocsc()
            logical = loaded_side.observables_matrix.tocsc()
            priors = np.asarray(loaded_side.priors, dtype=np.float64).reshape(-1)
            metadata = loaded_side.metadata
        else:
            detector = problem.D_Z.tocsc()
            logical = problem.O_Z.tocsc()
            priors = np.asarray(problem.priors_Z, dtype=np.float64).reshape(-1)
            metadata = problem.metadata_Z
        family_key = f"binary_dem_z{family_key_suffix}"
        scope_label = "Z side"
        detector_symbol = "D_Z"
        logical_symbol = "O_Z"
        metadata_symbol = "metadata_Z"
        priors_symbol = "priors_Z"
        if str(correction_state_mode_key) != "none":
            correction_by_signature, final_qubits = build_gross_split_sector_merged_correction_map(
                sector="Z",
                backend=str(backend),
                error_rate=float(p_location),
                cache_policy=("require_disk" if bool(require_correction_cache) else "build_if_missing"),
            )
            if str(correction_state_mode_key) == "full":
                correction_state_bits = int(len(final_qubits))
            else:
                projection_rows = _correction_projection_rows(
                    problem=problem,
                    scope=str(scope_key),
                    correction_state_mode=str(correction_state_mode_key),
                )
                assert projection_rows is not None
                correction_by_signature = _project_correction_map_by_signature(
                    correction_by_signature,
                    row_support_masks=_row_support_masks(np.asarray(projection_rows, dtype=np.uint8)),
                )
                correction_state_bits = int(np.asarray(projection_rows, dtype=np.uint8).shape[0])
    else:
        raise ValueError(f"unsupported scope: {scope}")
    ordered_columns = np.asarray(metadata.ordered_column_index, dtype=np.int32)

    columns_time_ordered: list[progressive.ProgressiveColumn] = []
    for order_index, source_column in enumerate(ordered_columns.tolist()):
        q = float(priors[int(source_column)])
        det_start = int(detector.indptr[int(source_column)])
        det_stop = int(detector.indptr[int(source_column) + 1])
        det_rows = detector.indices[det_start:det_stop].astype(np.int32, copy=False)
        log_start = int(logical.indptr[int(source_column)])
        log_stop = int(logical.indptr[int(source_column) + 1])
        log_rows = logical.indices[log_start:log_stop].astype(np.int32, copy=False)
        detector_mask = _bitmask_from_indices(det_rows.tolist())
        logical_mask = _bitmask_from_indices(log_rows.tolist())
        correction_response_masks: tuple[int, ...] | None = None
        if str(correction_state_mode_key) != "none":
            signature_mask = int(detector_mask) | (int(logical_mask) << int(detector.shape[0]))
            correction_mask = correction_by_signature.get(int(signature_mask))
            if correction_mask is None:
                raise ValueError(
                    "missing exact detector-side correction mask for column signature "
                    f"(scope={scope_key}, source_column={int(source_column)}, signature={int(signature_mask)})"
                )
            correction_response_masks = (0, int(correction_mask))
        prior_probs = (float(1.0 - q), float(q))
        round_start = int(metadata.column_round_start[int(source_column)])
        round_stop = int(metadata.column_round_stop[int(source_column)])
        columns_time_ordered.append(
            progressive.ProgressiveColumn(
                family=str(family_key),
                index=int(order_index),
                label=f"r{round_start}_to_r{round_stop}",
                instruction_offset=int(order_index),
                prior_probs=prior_probs,
                detector_response_masks=(0, int(detector_mask)),
                logical_response_masks=(0, int(logical_mask)),
                detector_support_mask=int(detector_mask),
                prior_log_probs=_log_probs_from_probs(prior_probs),
                detector_support_rows=tuple(int(value) for value in det_rows.tolist()),
                correction_response_masks=correction_response_masks,
                original_column_index=int(source_column),
            )
        )

    column_round_start_by_order = np.asarray(
        [int(metadata.column_round_start[int(source_column)]) for source_column in ordered_columns.tolist()],
        dtype=np.int16,
    )
    column_round_stop_by_order = np.asarray(
        [int(metadata.column_round_stop[int(source_column)]) for source_column in ordered_columns.tolist()],
        dtype=np.int16,
    )
    columns, _optimized_order, column_order_name, column_order_source = _ordered_columns_by_mode(
        columns_time_ordered=columns_time_ordered,
        column_round_start=column_round_start_by_order,
        column_round_stop=column_round_stop_by_order,
        num_detectors=int(detector.shape[0]),
        column_order=str(column_order),
        column_order_file=column_order_file,
        detector_matrix=detector,
        logical_matrix=logical,
        metadata=metadata,
        ordered_natural_columns=ordered_columns,
    )
    layout = progressive.build_frontier_layout(columns, num_detectors=int(detector.shape[0]))
    return LoadedProgressiveFamily(
        backend=str(backend),
        family_key=str(family_key),
        scope=str(scope_key),
        scope_label=str(scope_label),
        benchmark_title=str(benchmark_title),
        benchmark_description=str(benchmark_description),
        benchmark_source_note=str(benchmark_source_note),
        detector_symbol=str(detector_symbol),
        logical_symbol=str(logical_symbol),
        metadata_symbol=str(metadata_symbol),
        priors_symbol=str(priors_symbol),
        column_order_name=str(column_order_name),
        column_order_source=str(column_order_source).replace("metadata.", f"{metadata_symbol}."),
        model_label=f"{benchmark_description}, {scope_label.lower()} only",
        decode_label=f"binary DEM progressive ({scope_label.lower()})",
        columns=tuple(columns),
        layout=layout,
        matrix_rows=int(detector.shape[0]),
        matrix_cols=int(detector.shape[1]),
        logical_rows=int(logical.shape[0]),
        edge_count=int(detector.nnz),
        noisy_rounds=int(metadata.noisy_rounds),
        total_rounds=int(metadata.total_rounds),
        correction_state_mode=str(correction_state_mode_key),
        correction_state_bits=int(correction_state_bits),
    )


def _split_shot_indices(shot_indices: Sequence[int], shards: int) -> list[list[int]]:
    shot_index_list = [int(value) for value in shot_indices]
    if not shot_index_list:
        return []
    shard_count = min(int(len(shot_index_list)), int(shards))
    base = int(len(shot_index_list)) // int(shard_count)
    rem = int(len(shot_index_list)) % int(shard_count)
    out: list[list[int]] = []
    start = 0
    for shard_idx in range(int(shard_count)):
        size = int(base + (1 if shard_idx < rem else 0))
        out.append(list(int(value) for value in shot_index_list[start : start + size]))
        start += size
    return out


def _shot_seed(base_seed: int, shot_index: int) -> int:
    return int(base_seed) + int(shot_index) * 1_000_003


def _sample_truth_for_shot(*, shot_index: int) -> tuple[int, int, int, int]:
    rng = np.random.default_rng(_shot_seed(_GLOBAL_SEED, int(shot_index)))
    active = np.flatnonzero(rng.random(_GLOBAL_SAMPLE_PRIORS.size) < _GLOBAL_SAMPLE_PRIORS).astype(np.int32, copy=False)
    syndrome = 0
    logical = 0
    active_identity_mask = 0
    if active.size == 0:
        return 0, 0, 0, 0
    for location_index in active.tolist():
        column = _GLOBAL_SAMPLE_COLUMNS[int(location_index)]
        syndrome ^= int(column.detector_response_masks[1])
        logical ^= int(column.logical_response_masks[1])
        active_identity_mask |= 1 << int(column.instruction_offset)
    return int(syndrome), int(logical), int(active.size), int(active_identity_mask)


def _classify_row(
    *,
    shot_index: int,
    truth_logical: int,
    truth_nonzero_locations: int,
    family: LoadedProgressiveFamily,
    decoder_mode: str,
    backward_column_order: str,
    beam_size: int,
    score_mode: str,
    beam_score_gap_threshold: float | None,
    beam_score_gap_policy: progressive.BeamScoreGapPolicy | None,
    lookahead_depth: int,
    lookahead_shortlist_size: int,
    delayed_pruning_gap_threshold: float,
    delayed_pruning_factor: int,
    pruning_replay_checkpoint_stride: int,
    pruning_replay_horizon: int,
    tail_exact_columns: int,
    superstep_mode: str,
    superstep_path_budget: int,
    superstep_state_budget: int,
    superstep_transition_budget: int,
    detector_bucket_pruning: bool,
    detector_bucket_max_logicals: int,
    logical_class_reserve_min_classes: int,
    logical_class_reserve_max_replacements: int,
    logical_class_reserve_min_remaining_columns: int,
    logical_class_quota_top_classes: int,
    logical_class_quota_reserved_slots: int,
    logical_class_quota_min_remaining_columns: int,
    lineage_reserve_checkpoint_stride: int,
    lineage_reserve_reserved_slots: int,
    logical_rerank_columns: int,
    logical_rerank_shortlist_size: int,
    logical_rerank_min_classes: int,
    logical_rerank_state_budget: int,
    logical_rerank_transition_budget: int,
    logical_rerank_checkpoint_stride: int,
    logical_rerank_max_passes: int,
    logical_rerank_mode: str,
    track_best_path: bool,
    merge_duplicate_states: bool,
    state_merge_period_columns: int,
    decode_s: float,
    result: progressive.ProgressiveDecodeResult,
    state_count_profile_text: str = "",
    exception_message: str = "",
    selective_secondary_score_mode: str = "",
    selective_secondary_trigger_gap: float = 0.0,
    selective_secondary_band_size: int = 0,
) -> dict[str, object]:
    merge_events_total = _series_total_int(result.merge_count_by_column)
    closure_rejects_total = _series_total_int(result.closure_reject_count_by_column)
    top_log_mass_incoming_total = _series_total_int(result.top_log_mass_incoming_count_by_column)
    top_log_mass_merge_total = _series_total_int(result.top_log_mass_merge_count_by_column)
    top_viterbi_incoming_total = _series_total_int(result.top_viterbi_incoming_count_by_column)
    top_viterbi_merge_total = _series_total_int(result.top_viterbi_merge_count_by_column)
    winner_path_incoming_total = (
        float(_series_total_int(result.winning_path_incoming_count_by_column))
        if bool(result.winning_path_incoming_count_by_column)
        else float("nan")
    )
    winner_path_merge_total = (
        float(_series_total_int(result.winning_path_merge_count_by_column))
        if bool(result.winning_path_merge_count_by_column)
        else float("nan")
    )
    diagnosis = result.truth_logical_diagnosis
    if diagnosis is None or int(diagnosis.truth_logical_mask) != int(truth_logical):
        diagnosis = progressive.diagnose_progressive_truth_logical(
            result=result,
            truth_logical_mask=int(truth_logical),
        )
    diagnostics = progressive.summarize_progressive_beam_diagnostics(result)
    committee_members = tuple(result.committee_members)
    forward_committee_members = tuple(
        member for member in committee_members if str(member.direction) == "forward"
    )
    backward_committee_members = tuple(
        member for member in committee_members if str(member.direction) == "backward"
    )
    forward_committee_member = (
        forward_committee_members[0] if len(forward_committee_members) == 1 else None
    )
    backward_committee_member = (
        backward_committee_members[0] if len(backward_committee_members) == 1 else None
    )
    backward_order_label = str(backward_column_order).strip() or _default_backward_column_order_label(str(decoder_mode))
    threshold_trace_values = [
        float(value)
        for value in result.beam_score_gap_threshold_by_column
        if math.isfinite(float(value))
    ]
    forward_guidance_diag_summary = _summarize_forward_guidance_diagnostics(result)
    committee_select_top1_posterior, committee_select_top2_posterior = (
        progressive._progressive_committee_top_posteriors(result)
    )
    is_middle_join = str(result.committee_mode) == "middle_join" or str(result.sweep_direction) == "middle_join"
    splice_summary = result.splice_rerank_summary
    splice_enabled = splice_summary is not None and bool(splice_summary.enabled)
    splice_selected_logical = (
        None if splice_summary is None else splice_summary.selected_logical_mask
    )
    final_select_base_logical = int(result.final_logical_select_base_logical)
    baseline_selected_logical = (
        int(splice_summary.baseline_logical_mask)
        if splice_summary is not None
        else (
            int(final_select_base_logical)
            if int(final_select_base_logical) >= 0
            else int(result.logical_hat)
        )
    )

    def _fail_type_for_logical(logical_mask: int | None) -> str:
        if str(result.status) != "ok":
            return "exception_fail" if str(result.status) == "exception" else "syndrome_fail"
        if logical_mask is None:
            return "logical_fail"
        return "success" if int(logical_mask) == int(truth_logical) else "logical_fail"

    baseline_fail_type = _fail_type_for_logical(int(baseline_selected_logical))
    splice_fail_type = (
        _fail_type_for_logical(
            None if splice_selected_logical is None else int(splice_selected_logical)
        )
        if bool(splice_enabled)
        else ""
    )
    splice_truth_present_in_candidates = (
        bool(splice_enabled)
        and splice_summary is not None
        and int(truth_logical) in set(int(value) for value in tuple(splice_summary.candidate_logicals))
    )
    splice_changed = (
        bool(splice_enabled)
        and splice_selected_logical is not None
        and int(splice_selected_logical) != int(splice_summary.baseline_logical_mask)
    )
    splice_fixed = (
        bool(splice_enabled)
        and splice_selected_logical is not None
        and int(splice_summary.baseline_logical_mask) != int(truth_logical)
        and int(splice_selected_logical) == int(truth_logical)
    )
    splice_broken = (
        bool(splice_enabled)
        and splice_selected_logical is not None
        and int(splice_summary.baseline_logical_mask) == int(truth_logical)
        and int(splice_selected_logical) != int(truth_logical)
    )
    row: dict[str, object] = {
        "shot": int(shot_index),
        "scope": str(family.scope),
        "family": str(family.family_key),
        "decoder_mode": _normalize_decoder_mode(str(decoder_mode)),
        "backward_column_order": str(backward_order_label),
        "correction_state_mode": str(family.correction_state_mode),
        "correction_state_bits": int(family.correction_state_bits),
        "track_best_path": bool(track_best_path),
        "state_merge_mode": _state_merge_mode_label(
            merge_duplicate_states=bool(merge_duplicate_states),
            state_merge_period_columns=int(state_merge_period_columns),
        ),
        "state_merge_period_columns": int(state_merge_period_columns),
        "decoder": _decoder_label(
            family_key=str(family.family_key),
            decoder_mode=str(decoder_mode),
            backward_column_order=str(backward_order_label),
            beam_size=int(beam_size),
            score_mode=str(score_mode),
            beam_score_gap_threshold=beam_score_gap_threshold,
            beam_score_gap_policy=beam_score_gap_policy,
            selective_secondary_score_mode=str(selective_secondary_score_mode),
            selective_secondary_trigger_gap=float(selective_secondary_trigger_gap),
            selective_secondary_band_size=int(selective_secondary_band_size),
            forward_guidance_trigger_gap=float(result.forward_guidance_trigger_gap),
            forward_guidance_snapshot_factor=float(result.forward_guidance_snapshot_factor),
            forward_guidance_snapshot_gap=result.forward_guidance_snapshot_gap,
            forward_guidance_snapshot_source=str(result.forward_guidance_snapshot_source),
            forward_guidance_hamming_radius=int(result.forward_guidance_hamming_radius),
            forward_guidance_trigger_mode=str(result.forward_guidance_trigger_mode),
            forward_guidance_nearcut_gap=float(result.forward_guidance_nearcut_gap),
            forward_guidance_pool_trigger_min_positive_nearcut=int(
                result.forward_guidance_pool_trigger_min_positive_nearcut
            ),
            forward_guidance_diversity_fallback=str(result.forward_guidance_diversity_fallback),
            forward_guidance_mode=str(result.forward_guidance_mode),
            lookahead_depth=int(lookahead_depth),
            lookahead_shortlist_size=int(lookahead_shortlist_size),
            delayed_pruning_gap_threshold=float(delayed_pruning_gap_threshold),
            delayed_pruning_factor=int(delayed_pruning_factor),
            pruning_replay_checkpoint_stride=int(result.pruning_replay_checkpoint_stride),
            pruning_replay_horizon=int(result.pruning_replay_horizon),
            tail_exact_columns=int(tail_exact_columns),
            superstep_mode=str(superstep_mode),
            detector_bucket_pruning=bool(detector_bucket_pruning),
            detector_bucket_max_logicals=int(detector_bucket_max_logicals),
            logical_class_reserve_min_classes=int(logical_class_reserve_min_classes),
            logical_class_reserve_max_replacements=int(logical_class_reserve_max_replacements),
            logical_class_reserve_min_remaining_columns=int(logical_class_reserve_min_remaining_columns),
            logical_class_quota_top_classes=int(logical_class_quota_top_classes),
            logical_class_quota_reserved_slots=int(logical_class_quota_reserved_slots),
            logical_class_quota_min_remaining_columns=int(logical_class_quota_min_remaining_columns),
            lineage_reserve_checkpoint_stride=int(lineage_reserve_checkpoint_stride),
            lineage_reserve_reserved_slots=int(lineage_reserve_reserved_slots),
            logical_rerank_columns=int(logical_rerank_columns),
            logical_rerank_shortlist_size=int(logical_rerank_shortlist_size),
            logical_rerank_min_classes=int(logical_rerank_min_classes),
            logical_rerank_state_budget=int(logical_rerank_state_budget),
            logical_rerank_transition_budget=int(logical_rerank_transition_budget),
            logical_rerank_checkpoint_stride=int(logical_rerank_checkpoint_stride),
            logical_rerank_max_passes=int(logical_rerank_max_passes),
            logical_rerank_mode=str(logical_rerank_mode),
        ),
        "beam_size": int(beam_size),
        "score_mode": str(score_mode),
        "beam_score_gap_threshold": (
            float(beam_score_gap_threshold)
            if beam_score_gap_threshold is not None and math.isfinite(float(beam_score_gap_threshold))
            else ""
        ),
        "beam_score_gap_policy_mode": (
            str(result.beam_score_gap_policy_mode)
            if _beam_score_gap_policy_enabled(beam_score_gap_policy)
            else ""
        ),
        "beam_score_gap_policy_base_threshold": (
            float(beam_score_gap_policy.base_threshold) if beam_score_gap_policy is not None else ""
        ),
        "beam_score_gap_policy_final_threshold": (
            float(beam_score_gap_policy.final_threshold) if beam_score_gap_policy is not None else ""
        ),
        "beam_score_gap_policy_slope": (
            float(beam_score_gap_policy.slope) if beam_score_gap_policy is not None else ""
        ),
        "beam_score_gap_policy_reference_count": (
            float(beam_score_gap_policy.reference_count) if beam_score_gap_policy is not None else ""
        ),
        "beam_score_gap_policy_min_threshold": (
            float(beam_score_gap_policy.min_threshold) if beam_score_gap_policy is not None else ""
        ),
        "beam_score_gap_policy_max_threshold": (
            float(beam_score_gap_policy.max_threshold) if beam_score_gap_policy is not None else ""
        ),
        "beam_score_gap_threshold_trace_mean": (
            float(np.mean(np.asarray(threshold_trace_values, dtype=np.float64)))
            if threshold_trace_values
            else ""
        ),
        "beam_score_gap_threshold_trace_min": (
            float(min(threshold_trace_values)) if threshold_trace_values else ""
        ),
        "beam_score_gap_threshold_trace_max": (
            float(max(threshold_trace_values)) if threshold_trace_values else ""
        ),
        "selective_secondary_score_mode": (
            str(result.selective_secondary_score_mode)
            if _selective_secondary_enabled(
                selective_secondary_score_mode=str(selective_secondary_score_mode),
                selective_secondary_trigger_gap=float(selective_secondary_trigger_gap),
                selective_secondary_band_size=int(selective_secondary_band_size),
            )
            else ""
        ),
        "selective_secondary_trigger_gap": (
            float(selective_secondary_trigger_gap)
            if _selective_secondary_enabled(
                selective_secondary_score_mode=str(selective_secondary_score_mode),
                selective_secondary_trigger_gap=float(selective_secondary_trigger_gap),
                selective_secondary_band_size=int(selective_secondary_band_size),
            )
            else ""
        ),
        "selective_secondary_band_size": (
            int(selective_secondary_band_size)
            if _selective_secondary_enabled(
                selective_secondary_score_mode=str(selective_secondary_score_mode),
                selective_secondary_trigger_gap=float(selective_secondary_trigger_gap),
                selective_secondary_band_size=int(selective_secondary_band_size),
            )
            else ""
        ),
        "selective_secondary_trigger_count": int(result.selective_secondary_trigger_count),
        "selective_secondary_changed_count": int(result.selective_secondary_changed_count),
        "selective_secondary_reranked_state_count": int(result.selective_secondary_reranked_state_count),
        "selective_local_lookahead_mode": str(result.selective_local_lookahead_mode),
        "selective_local_lookahead_score_mode": str(result.selective_local_lookahead_score_mode),
        "selective_local_lookahead_cutoff_gap_threshold": (
            float(result.selective_local_lookahead_cutoff_gap_threshold)
            if _selective_local_lookahead_enabled(result.selective_local_lookahead_mode)
            else ""
        ),
        "selective_local_lookahead_near_cut_width": (
            float(result.selective_local_lookahead_near_cut_width)
            if _selective_local_lookahead_enabled(result.selective_local_lookahead_mode)
            else ""
        ),
        "selective_local_lookahead_max_candidates": (
            int(result.selective_local_lookahead_max_candidates)
            if _selective_local_lookahead_enabled(result.selective_local_lookahead_mode)
            else ""
        ),
        "selective_local_lookahead_candidate_top1_share_threshold": (
            float(result.selective_local_lookahead_candidate_top1_share_threshold)
            if _selective_local_lookahead_enabled(result.selective_local_lookahead_mode)
            else ""
        ),
        "selective_local_lookahead_support_gap_threshold": (
            float(result.selective_local_lookahead_support_gap_threshold)
            if _selective_local_lookahead_enabled(result.selective_local_lookahead_mode)
            else ""
        ),
        "selective_local_lookahead_overflow_ratio_threshold": (
            float(result.selective_local_lookahead_overflow_ratio_threshold)
            if _selective_local_lookahead_enabled(result.selective_local_lookahead_mode)
            else ""
        ),
        "selective_local_lookahead_trigger_count": int(result.selective_local_lookahead_trigger_count),
        "selective_local_lookahead_changed_count": int(result.selective_local_lookahead_changed_count),
        "selective_local_lookahead_candidate_count": int(result.selective_local_lookahead_candidate_count),
        "selective_local_lookahead_extra_work": int(result.selective_local_lookahead_extra_work),
        "selective_local_lookahead_steps_json": _selective_local_lookahead_steps_json(result),
        "lookahead_depth": int(lookahead_depth),
        "lookahead_shortlist_size": int(lookahead_shortlist_size),
        "delayed_pruning_gap_threshold": float(delayed_pruning_gap_threshold),
        "delayed_pruning_factor": int(delayed_pruning_factor),
        "pruning_replay_checkpoint_stride": int(result.pruning_replay_checkpoint_stride),
        "pruning_replay_horizon": int(result.pruning_replay_horizon),
        "pruning_replay_attempt_count": int(result.pruning_replay_attempt_count),
        "pruning_replay_applied_count": int(result.pruning_replay_applied_count),
        "pruning_replay_replaced_state_count": int(result.pruning_replay_replaced_state_count),
        "pruning_replay_replayed_column_count": int(result.pruning_replay_replayed_column_count),
        "pruning_replay_extra_transition_evals": int(result.pruning_replay_extra_transition_evals),
        "tail_exact_columns": int(tail_exact_columns),
        "superstep_mode": str(superstep_mode),
        "superstep_path_budget": int(superstep_path_budget),
        "superstep_state_budget": int(superstep_state_budget),
        "superstep_transition_budget": int(superstep_transition_budget),
        "detector_bucket_pruning": bool(detector_bucket_pruning),
        "detector_bucket_max_logicals": int(detector_bucket_max_logicals),
        "logical_class_reserve_min_classes": int(logical_class_reserve_min_classes),
        "logical_class_reserve_max_replacements": int(logical_class_reserve_max_replacements),
        "logical_class_reserve_min_remaining_columns": int(logical_class_reserve_min_remaining_columns),
        "logical_class_reserve_applied_count": int(result.logical_class_reserve_applied_count),
        "logical_class_reserve_replaced_state_count": int(result.logical_class_reserve_replaced_state_count),
        "logical_class_quota_top_classes": int(logical_class_quota_top_classes),
        "logical_class_quota_reserved_slots": int(logical_class_quota_reserved_slots),
        "logical_class_quota_min_remaining_columns": int(logical_class_quota_min_remaining_columns),
        "logical_class_quota_applied_count": int(result.logical_class_quota_applied_count),
        "logical_class_quota_kept_state_count": int(result.logical_class_quota_kept_state_count),
        "lineage_reserve_checkpoint_stride": int(lineage_reserve_checkpoint_stride),
        "lineage_reserve_reserved_slots": int(lineage_reserve_reserved_slots),
        "lineage_reserve_applied_count": int(result.lineage_reserve_applied_count),
        "lineage_reserve_kept_state_count": int(result.lineage_reserve_kept_state_count),
        "logical_rerank_columns": int(logical_rerank_columns),
        "logical_rerank_shortlist_size": int(logical_rerank_shortlist_size),
        "logical_rerank_min_classes": int(logical_rerank_min_classes),
        "logical_rerank_state_budget": int(logical_rerank_state_budget),
        "logical_rerank_transition_budget": int(logical_rerank_transition_budget),
        "logical_rerank_checkpoint_stride": int(logical_rerank_checkpoint_stride),
        "logical_rerank_max_passes": int(logical_rerank_max_passes),
        "logical_rerank_mode": str(logical_rerank_mode),
        "final_logical_select_mode": str(result.final_logical_select_mode),
        "final_logical_select_rep_cost_weight": float(result.final_logical_select_rep_cost_weight),
        "final_logical_select_max_log_mass_gap": (
            float(result.final_logical_select_max_log_mass_gap)
            if math.isfinite(float(result.final_logical_select_max_log_mass_gap))
            else ""
        ),
        "final_logical_select_rank2_viterbi_tolerance": float(
            result.final_logical_select_rank2_viterbi_tolerance
        ),
        "final_logical_select_base_logical": int(result.final_logical_select_base_logical),
        "final_logical_select_gate_triggered": bool(result.final_logical_select_gate_triggered),
        "terminal_top_log_mass_gap": float(result.terminal_top_log_mass_gap),
        "log_evidence": float(result.log_evidence),
        "committee_select_top1_posterior": float(committee_select_top1_posterior),
        "committee_select_top2_posterior": float(committee_select_top2_posterior),
        "forward_guidance_weight": float(result.forward_guidance_weight),
        "forward_guidance_clip": float(result.forward_guidance_clip),
        "forward_guidance_trigger_gap": float(result.forward_guidance_trigger_gap),
        "forward_guidance_widen_factor": float(result.forward_guidance_widen_factor),
        "forward_guidance_min_info_bits": float(result.forward_guidance_min_info_bits),
        "forward_guidance_snapshot_factor": float(result.forward_guidance_snapshot_factor),
        "forward_guidance_snapshot_gap": (
            ""
            if result.forward_guidance_snapshot_gap is None
            else _format_forward_guidance_snapshot_gap(result.forward_guidance_snapshot_gap)
        ),
        "forward_guidance_snapshot_source": str(result.forward_guidance_snapshot_source),
        "forward_guidance_hamming_radius": int(result.forward_guidance_hamming_radius),
        "forward_guidance_trigger_mode": str(result.forward_guidance_trigger_mode),
        "forward_guidance_nearcut_gap": float(result.forward_guidance_nearcut_gap),
        "forward_guidance_pool_trigger_min_positive_nearcut": int(
            result.forward_guidance_pool_trigger_min_positive_nearcut
        ),
        "forward_guidance_diversity_fallback": str(result.forward_guidance_diversity_fallback),
        "forward_guidance_mode": str(result.forward_guidance_mode),
        **forward_guidance_diag_summary,
        "logical_rerank_pass_count": int(result.logical_rerank_pass_count),
        "delayed_pruning_trigger_count": int(result.delayed_pruning_trigger_count),
        "delayed_pruning_active_prune_count": int(result.delayed_pruning_active_prune_count),
        "delayed_pruning_peak_beam_size": int(result.delayed_pruning_peak_beam_size),
        "noisy_rounds": int(family.noisy_rounds),
        "total_rounds": int(family.total_rounds),
        "matrix_rows": int(family.matrix_rows),
        "matrix_cols": int(family.matrix_cols),
        "logical_rows": int(family.logical_rows),
        "edge_count": int(family.edge_count),
        "frontier_max_active_detectors": int(family.layout.max_active_detectors),
        "truth_nonzero_locations": int(truth_nonzero_locations),
        "exception_message": str(exception_message),
        "status": str(result.status),
        "sweep_direction": str(result.sweep_direction),
        "committee_mode": str(result.committee_mode),
        "committee_selected_direction": str(result.committee_selected_direction),
        "committee_selected_score_mode": str(result.committee_selected_score_mode),
        "committee_member_count": int(len(committee_members)),
        "committee_forward_member_count": int(len(forward_committee_members)),
        "committee_backward_member_count": int(len(backward_committee_members)),
        "committee_member_score_modes": "|".join(str(member.score_mode) for member in committee_members),
        "committee_forward_logical_hat": (
            int(forward_committee_member.logical_hat) if forward_committee_member is not None else -1
        ),
        "committee_backward_logical_hat": (
            int(backward_committee_member.logical_hat) if backward_committee_member is not None else -1
        ),
        "committee_forward_log_evidence": (
            float(forward_committee_member.log_evidence) if forward_committee_member is not None else float("nan")
        ),
        "committee_backward_log_evidence": (
            float(backward_committee_member.log_evidence) if backward_committee_member is not None else float("nan")
        ),
        "middle_join_prefix_columns": (
            int(result.middle_join_prefix_columns) if bool(is_middle_join) else ""
        ),
        "middle_join_suffix_columns": (
            int(result.middle_join_suffix_columns) if bool(is_middle_join) else ""
        ),
        "middle_join_boundary_rows": (
            int(result.middle_join_boundary_rows) if bool(is_middle_join) else ""
        ),
        "middle_join_forward_cut_state_count": (
            int(result.middle_join_forward_cut_state_count) if bool(is_middle_join) else ""
        ),
        "middle_join_backward_cut_state_count": (
            int(result.middle_join_backward_cut_state_count) if bool(is_middle_join) else ""
        ),
        "middle_join_forward_cut_detector_count": (
            int(result.middle_join_forward_cut_detector_count) if bool(is_middle_join) else ""
        ),
        "middle_join_backward_cut_detector_count": (
            int(result.middle_join_backward_cut_detector_count) if bool(is_middle_join) else ""
        ),
        "middle_join_compatible_forward_state_count": (
            int(result.middle_join_compatible_forward_state_count) if bool(is_middle_join) else ""
        ),
        "middle_join_compatible_backward_state_count": (
            int(result.middle_join_compatible_backward_state_count) if bool(is_middle_join) else ""
        ),
        "middle_join_compatible_pair_count": (
            int(result.middle_join_compatible_pair_count) if bool(is_middle_join) else ""
        ),
        "middle_join_forward_cut_log_mass": (
            float(result.middle_join_forward_cut_log_mass) if bool(is_middle_join) else ""
        ),
        "middle_join_backward_cut_log_mass": (
            float(result.middle_join_backward_cut_log_mass) if bool(is_middle_join) else ""
        ),
        "middle_join_compatible_forward_log_mass": (
            float(result.middle_join_compatible_forward_log_mass) if bool(is_middle_join) else ""
        ),
        "middle_join_compatible_backward_log_mass": (
            float(result.middle_join_compatible_backward_log_mass) if bool(is_middle_join) else ""
        ),
        "middle_join_multicut_requested_cut_count": (
            int(result.middle_join_multicut_requested_cut_count) if bool(is_middle_join) else ""
        ),
        "middle_join_multicut_used_cut_count": (
            int(result.middle_join_multicut_used_cut_count) if bool(is_middle_join) else ""
        ),
        "middle_join_multicut_weight_mode": (
            str(result.middle_join_multicut_weight_mode) if bool(is_middle_join) else ""
        ),
        "middle_join_multicut_prefix_columns_items": (
            "|".join(str(int(value)) for value in result.middle_join_multicut_prefix_columns_items)
            if bool(is_middle_join)
            else ""
        ),
        "middle_join_multicut_weight_items": (
            "|".join(
                f"{int(prefix_columns)}:{float(weight):.12g}"
                for prefix_columns, weight in result.middle_join_multicut_weight_items
            )
            if bool(is_middle_join)
            else ""
        ),
        "splice_enabled": bool(splice_enabled),
        "splice_cut_selector": (
            str(_GLOBAL_SPLICE_CUT_SELECTOR) if bool(splice_enabled) else ""
        ),
        "splice_candidate_count": (
            int(splice_summary.candidate_count) if splice_summary is not None else ""
        ),
        "splice_candidate_logical_class_count": (
            int(splice_summary.candidate_count) if splice_summary is not None else ""
        ),
        "splice_cut_count": (
            int(splice_summary.cut_count) if splice_summary is not None else ""
        ),
        "splice_requested_cut_count": (
            int(splice_summary.requested_cut_count) if splice_summary is not None else ""
        ),
        "splice_invalid_cut_count": (
            int(splice_summary.invalid_cut_count) if splice_summary is not None else ""
        ),
        "splice_aggregate": (
            str(splice_summary.aggregate_mode) if splice_summary is not None else ""
        ),
        "splice_selected_logical": (
            int(splice_selected_logical) if splice_selected_logical is not None else ""
        ),
        "splice_baseline_logical": (
            int(splice_summary.baseline_logical_mask) if splice_summary is not None else ""
        ),
        "baseline_selected_logical": int(baseline_selected_logical),
        "baseline_fail_type": str(baseline_fail_type),
        "splice_fail_type": str(splice_fail_type),
        "splice_changed": bool(splice_changed),
        "splice_selected_score": (
            float(splice_summary.selected_score)
            if splice_summary is not None and math.isfinite(float(splice_summary.selected_score))
            else ""
        ),
        "splice_baseline_score": (
            float(splice_summary.baseline_score)
            if splice_summary is not None and math.isfinite(float(splice_summary.baseline_score))
            else ""
        ),
        "splice_finite_cut_fraction": (
            float(splice_summary.finite_cut_fraction) if splice_summary is not None else ""
        ),
        "splice_missing_support_fraction": (
            float(splice_summary.missing_support_fraction) if splice_summary is not None else ""
        ),
        "splice_hit_count_mean": (
            float(splice_summary.hit_count_mean) if splice_summary is not None else ""
        ),
        "splice_truth_present_in_candidates": bool(splice_truth_present_in_candidates),
        "splice_fixed": bool(splice_fixed),
        "splice_broken": bool(splice_broken),
        "splice_candidate_missing_truth": (
            bool(splice_enabled) and not bool(splice_truth_present_in_candidates)
        ),
        "splice_cut_scores_json": _serialize_splice_rerank_summary(splice_summary),
        "logical_hat": int(result.logical_hat),
        "truth_logical": int(truth_logical),
        "truth_logical_retained_terminal": bool(diagnosis.truth_present_terminal),
        "truth_logical_failure_mode": str(diagnosis.failure_mode),
        "forward_guidance_truth_cut_terminal_truth_present": int(bool(diagnosis.truth_present_terminal)),
        "forward_guidance_truth_cut_first_ordinary_loss_truth_present_terminal": int(
            bool(diagnosis.truth_present_terminal)
        ),
        "truth_terminal_log_mass": (
            float(diagnosis.truth_terminal_log_mass)
            if math.isfinite(float(diagnosis.truth_terminal_log_mass))
            else ""
        ),
        "truth_terminal_best_viterbi": (
            float(diagnosis.truth_terminal_best_viterbi)
            if math.isfinite(float(diagnosis.truth_terminal_best_viterbi))
            else ""
        ),
        "truth_terminal_representative_cost": (
            float(diagnosis.truth_terminal_representative_cost)
            if math.isfinite(float(diagnosis.truth_terminal_representative_cost))
            else ""
        ),
        "truth_terminal_log_mass_rank": int(diagnosis.truth_terminal_log_mass_rank),
        "truth_terminal_best_viterbi_rank": int(diagnosis.truth_terminal_best_viterbi_rank),
        "terminal_distinct_logical_classes": int(diagnosis.terminal_class_count),
        "discard_step_count": int(diagnostics.discard_step_count),
        "cumulative_discarded_prefix_mass": float(result.cumulative_discarded_prefix_mass),
        "max_discarded_prefix_mass": float(diagnostics.max_discarded_prefix_mass),
        "mean_discarded_prefix_mass": float(diagnostics.mean_discarded_prefix_mass),
        "max_discarded_prefix_fraction": float(diagnostics.max_discarded_prefix_fraction),
        "mean_discarded_prefix_fraction": float(diagnostics.mean_discarded_prefix_fraction),
        "truth_logical_discard_step_count": int(diagnostics.truth_logical_discard_step_count),
        "cumulative_truth_logical_discarded_prefix_mass": float(
            result.cumulative_truth_logical_discarded_prefix_mass
        ),
        "max_truth_logical_discarded_prefix_mass": float(
            diagnostics.max_truth_logical_discarded_prefix_mass
        ),
        "mean_truth_logical_discarded_prefix_mass": float(
            diagnostics.mean_truth_logical_discarded_prefix_mass
        ),
        "max_truth_logical_discarded_prefix_fraction": float(
            diagnostics.max_truth_logical_discarded_prefix_fraction
        ),
        "mean_truth_logical_discarded_prefix_fraction": float(
            diagnostics.mean_truth_logical_discarded_prefix_fraction
        ),
        "transition_evals": int(np.sum(np.asarray(result.expanded_transition_count_by_column, dtype=np.int64))),
        "lookahead_transition_evals": int(np.sum(np.asarray(result.lookahead_transition_count_by_column, dtype=np.int64))),
        "transition_evals_total": int(
            np.sum(np.asarray(result.expanded_transition_count_by_column, dtype=np.int64))
            + np.sum(np.asarray(result.lookahead_transition_count_by_column, dtype=np.int64))
        ),
        "transition_evals_physical_total": int(
            np.sum(np.asarray(result.expanded_transition_count_by_column, dtype=np.int64))
            + np.sum(np.asarray(result.lookahead_transition_count_by_column, dtype=np.int64))
            + int(result.pruning_replay_extra_transition_evals)
        ),
        "decode_s": float(decode_s),
        "log_mass_primary_production_path": str(result.log_mass_primary_production_path),
        "log_mass_primary_production_path_used": bool(result.log_mass_primary_production_path_used),
        "log_mass_primary_guard_fallback_count": int(result.log_mass_primary_guard_fallback_count),
        "binary_frontierk_hotloop_instrumentation_enabled": bool(
            getattr(result, "binary_frontierk_hotloop_instrumentation_enabled", False)
        ),
        "binary_frontierk_parent_state_count": int(
            getattr(result, "binary_frontierk_parent_state_count", 0)
        ),
        "binary_frontierk_parent_detector_key_unique_count": int(
            getattr(result, "binary_frontierk_parent_detector_key_unique_count", 0)
        ),
        "binary_frontierk_parent_detector_key_duplicate_count": int(
            getattr(result, "binary_frontierk_parent_detector_key_duplicate_count", 0)
        ),
        "binary_frontierk_candidate_detector_key_unique_count": int(
            getattr(result, "binary_frontierk_candidate_detector_key_unique_count", 0)
        ),
        "binary_frontierk_candidate_detector_key_duplicate_count": int(
            getattr(result, "binary_frontierk_candidate_detector_key_duplicate_count", 0)
        ),
        "binary_frontierk_candidate_detector_key_full_state_duplicate_count": int(
            getattr(result, "binary_frontierk_candidate_detector_key_full_state_duplicate_count", 0)
        ),
        "binary_frontierk_local_pattern_sample_count": int(
            getattr(result, "binary_frontierk_local_pattern_sample_count", 0)
        ),
        "binary_frontierk_local_pattern_unique_count": int(
            getattr(result, "binary_frontierk_local_pattern_unique_count", 0)
        ),
        "binary_frontierk_local_pattern_duplicate_count": int(
            getattr(result, "binary_frontierk_local_pattern_duplicate_count", 0)
        ),
        "binary_frontierk_local_pattern_full_table_entry_count": int(
            getattr(result, "binary_frontierk_local_pattern_full_table_entry_count", 0)
        ),
        "binary_frontierk_local_pattern_full_table_oversized_count": int(
            getattr(result, "binary_frontierk_local_pattern_full_table_oversized_count", 0)
        ),
        "binary_frontierk_local_pattern_max_unique_per_column": int(
            getattr(result, "binary_frontierk_local_pattern_max_unique_per_column", 0)
        ),
        "binary_frontierk_local_pattern_max_degree": int(
            getattr(result, "binary_frontierk_local_pattern_max_degree", 0)
        ),
        "binary_frontierk_local_pattern_max_table_entries_per_column": int(
            getattr(result, "binary_frontierk_local_pattern_max_table_entries_per_column", 0)
        ),
        "binary_frontierk_local_pattern_feature_table_enabled": bool(
            getattr(result, "binary_frontierk_local_pattern_feature_table_enabled", False)
        ),
        "binary_frontierk_local_pattern_feature_table_boundary_count": int(
            getattr(result, "binary_frontierk_local_pattern_feature_table_boundary_count", 0)
        ),
        "binary_frontierk_local_pattern_feature_table_lookup_count": int(
            getattr(result, "binary_frontierk_local_pattern_feature_table_lookup_count", 0)
        ),
        "binary_frontierk_local_pattern_feature_table_entry_count": int(
            getattr(result, "binary_frontierk_local_pattern_feature_table_entry_count", 0)
        ),
        "binary_frontierk_local_pattern_feature_table_fallback_count": int(
            getattr(result, "binary_frontierk_local_pattern_feature_table_fallback_count", 0)
        ),
        "binary_frontierk_local_pattern_feature_table_max_entries_per_column": int(
            getattr(result, "binary_frontierk_local_pattern_feature_table_max_entries_per_column", 0)
        ),
        "binary_frontierk_unique_detector_score_fast_path_enabled": bool(
            getattr(result, "binary_frontierk_unique_detector_score_fast_path_enabled", False)
        ),
        "binary_frontierk_unique_detector_score_fast_path_boundary_count": int(
            getattr(result, "binary_frontierk_unique_detector_score_fast_path_boundary_count", 0)
        ),
        "binary_frontierk_unique_detector_score_fast_path_candidate_count": int(
            getattr(result, "binary_frontierk_unique_detector_score_fast_path_candidate_count", 0)
        ),
        "binary_frontierk_streaming_rank_selector_enabled": bool(
            getattr(result, "binary_frontierk_streaming_rank_selector_enabled", False)
        ),
        "binary_frontierk_streaming_rank_selector_boundary_count": int(
            getattr(result, "binary_frontierk_streaming_rank_selector_boundary_count", 0)
        ),
        "binary_frontierk_streaming_rank_selector_stored_candidate_count": int(
            getattr(result, "binary_frontierk_streaming_rank_selector_stored_candidate_count", 0)
        ),
        "binary_frontierk_streaming_rank_selector_skipped_candidate_count": int(
            getattr(result, "binary_frontierk_streaming_rank_selector_skipped_candidate_count", 0)
        ),
        "binary_frontierk_streaming_rank_selector_max_window_candidate_count": int(
            getattr(result, "binary_frontierk_streaming_rank_selector_max_window_candidate_count", 0)
        ),
        "binary_frontierk_deferred_feature_materialization_enabled": bool(
            getattr(result, "binary_frontierk_deferred_feature_materialization_enabled", False)
        ),
        "binary_frontierk_deferred_feature_materialization_boundary_count": int(
            getattr(result, "binary_frontierk_deferred_feature_materialization_boundary_count", 0)
        ),
        "binary_frontierk_deferred_feature_materialization_parity_feature_count": int(
            getattr(result, "binary_frontierk_deferred_feature_materialization_parity_feature_count", 0)
        ),
        "binary_frontierk_deferred_feature_materialization_full_feature_count": int(
            getattr(result, "binary_frontierk_deferred_feature_materialization_full_feature_count", 0)
        ),
        "binary_frontierk_deferred_feature_materialization_fallback_count": int(
            getattr(result, "binary_frontierk_deferred_feature_materialization_fallback_count", 0)
        ),
        "binary_frontierk_small_candidate_direct_gap_boundary_count": int(
            getattr(result, "binary_frontierk_small_candidate_direct_gap_boundary_count", 0)
        ),
        "binary_frontierk_small_candidate_direct_gap_candidate_count": int(
            getattr(result, "binary_frontierk_small_candidate_direct_gap_candidate_count", 0)
        ),
        "binary_frontierk_zero_close_mask_fast_path_boundary_count": int(
            getattr(result, "binary_frontierk_zero_close_mask_fast_path_boundary_count", 0)
        ),
        "binary_frontierk_zero_close_mask_fast_path_state_count": int(
            getattr(result, "binary_frontierk_zero_close_mask_fast_path_state_count", 0)
        ),
        "pruned_states_total": int(np.sum(np.asarray(result.pruned_state_count_by_column, dtype=np.int64))),
        "mean_states": float(np.mean(np.asarray(result.state_count_by_column, dtype=np.float64))),
        "max_states_seen": int(result.max_state_count),
        "merge_events_total": int(merge_events_total),
        "merge_events_per_column": float(_series_mean(result.merge_count_by_column)),
        "closure_rejects_total": int(closure_rejects_total),
        "closure_rejects_per_column": float(_series_mean(result.closure_reject_count_by_column)),
        "top_log_mass_incoming_per_column": float(_series_mean(result.top_log_mass_incoming_count_by_column)),
        "top_log_mass_merge_per_column": float(_series_mean(result.top_log_mass_merge_count_by_column)),
        "top_log_mass_incoming_total": int(top_log_mass_incoming_total),
        "top_log_mass_merge_total": int(top_log_mass_merge_total),
        "top_viterbi_incoming_per_column": float(_series_mean(result.top_viterbi_incoming_count_by_column)),
        "top_viterbi_merge_per_column": float(_series_mean(result.top_viterbi_merge_count_by_column)),
        "top_viterbi_incoming_total": int(top_viterbi_incoming_total),
        "top_viterbi_merge_total": int(top_viterbi_merge_total),
        "winner_path_incoming_per_column": (
            float(_series_mean(result.winning_path_incoming_count_by_column))
            if bool(result.winning_path_incoming_count_by_column)
            else float("nan")
        ),
        "winner_path_merge_per_column": (
            float(_series_mean(result.winning_path_merge_count_by_column))
            if bool(result.winning_path_merge_count_by_column)
            else float("nan")
        ),
        "winner_path_incoming_total": float(winner_path_incoming_total),
        "winner_path_merge_total": float(winner_path_merge_total),
        "tail_exact_start_column": int(result.tail_exact_start_column),
        "frame_ok": False,
        "frame_fail_type": "success",
    }
    if str(state_count_profile_text):
        row["state_count_by_column_profile"] = str(state_count_profile_text)
    if bool(_GLOBAL_EXPORT_TERMINAL_SELECTOR_SIGNALS):
        row["terminal_selector_signals_json"] = _serialize_terminal_selector_signals(result)
    if str(result.status) != "ok":
        row["frame_ok"] = False
        row["frame_fail_type"] = "exception_fail" if str(result.status) == "exception" else "syndrome_fail"
        return row
    if int(result.logical_hat) == int(truth_logical):
        row["frame_ok"] = True
        row["frame_fail_type"] = "success"
    else:
        row["frame_ok"] = False
        row["frame_fail_type"] = "logical_fail"
    return row


def _decode_one_shot(shot_index: int, beam_sizes: tuple[int, ...]) -> list[dict[str, object]]:
    family = _GLOBAL_FAMILY
    if family is None:
        raise RuntimeError("global family not initialized")
    truth_sample = tuple(_sample_truth_for_shot(shot_index=int(shot_index)))
    if len(truth_sample) == 3:
        syndrome, truth_logical, truth_nonzero_locations = truth_sample
        truth_column_identity_mask = 0
    elif len(truth_sample) == 4:
        syndrome, truth_logical, truth_nonzero_locations, truth_column_identity_mask = truth_sample
    else:
        raise ValueError("_sample_truth_for_shot must return 3 or 4 values")
    out: list[dict[str, object]] = []
    for score_mode in _GLOBAL_SCORE_MODES:
        for beam_size in beam_sizes:
            started = time.perf_counter()
            exception_message = ""
            try:
                decode_kwargs: dict[str, object] = {
                    "target_syndrome": int(syndrome),
                    "num_detectors": int(family.matrix_rows),
                    "num_observables": int(family.logical_rows),
                    "beam_size": int(beam_size),
                    "score_mode": str(score_mode),
                    "beam_score_gap_threshold": _GLOBAL_BEAM_SCORE_GAP_THRESHOLD,
                    "beam_score_gap_policy": _GLOBAL_BEAM_SCORE_GAP_POLICY,
                    "selective_secondary_score_mode": str(_GLOBAL_SELECTIVE_SECONDARY_SCORE_MODE),
                    "selective_secondary_trigger_gap": float(_GLOBAL_SELECTIVE_SECONDARY_TRIGGER_GAP),
                    "selective_secondary_band_size": int(_GLOBAL_SELECTIVE_SECONDARY_BAND_SIZE),
                    "selective_local_lookahead_mode": str(_GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_MODE),
                    "selective_local_lookahead_cutoff_gap_threshold": float(
                        _GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_CUTOFF_GAP_THRESHOLD
                    ),
                    "selective_local_lookahead_near_cut_width": float(
                        _GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_NEAR_CUT_WIDTH
                    ),
                    "selective_local_lookahead_max_candidates": int(
                        _GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_MAX_CANDIDATES
                    ),
                    "selective_local_lookahead_candidate_top1_share_threshold": float(
                        _GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_CANDIDATE_TOP1_SHARE_THRESHOLD
                    ),
                    "selective_local_lookahead_support_gap_threshold": float(
                        _GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_SUPPORT_GAP_THRESHOLD
                    ),
                    "selective_local_lookahead_overflow_ratio_threshold": float(
                        _GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_OVERFLOW_RATIO_THRESHOLD
                    ),
                    "lookahead_depth": int(_GLOBAL_LOOKAHEAD_DEPTH),
                    "lookahead_shortlist_size": int(_GLOBAL_LOOKAHEAD_SHORTLIST_SIZE),
                    "delayed_pruning_gap_threshold": float(_GLOBAL_DELAYED_PRUNING_GAP_THRESHOLD),
                    "delayed_pruning_factor": int(_GLOBAL_DELAYED_PRUNING_FACTOR),
                    "pruning_replay_checkpoint_stride": int(_GLOBAL_PRUNING_REPLAY_CHECKPOINT_STRIDE),
                    "pruning_replay_horizon": int(_GLOBAL_PRUNING_REPLAY_HORIZON),
                    "tail_exact_columns": int(_GLOBAL_TAIL_EXACT_COLUMNS),
                    "superstep_mode": str(_GLOBAL_SUPERSTEP_MODE),
                    "superstep_path_budget": int(_GLOBAL_SUPERSTEP_PATH_BUDGET),
                    "superstep_state_budget": int(_GLOBAL_SUPERSTEP_STATE_BUDGET),
                    "superstep_transition_budget": int(_GLOBAL_SUPERSTEP_TRANSITION_BUDGET),
                    "detector_bucket_pruning": bool(_GLOBAL_DETECTOR_BUCKET_PRUNING),
                    "detector_bucket_max_logicals": int(_GLOBAL_DETECTOR_BUCKET_MAX_LOGICALS),
                    "logical_class_reserve_min_classes": int(_GLOBAL_LOGICAL_CLASS_RESERVE_MIN_CLASSES),
                    "logical_class_reserve_max_replacements": int(_GLOBAL_LOGICAL_CLASS_RESERVE_MAX_REPLACEMENTS),
                    "logical_class_reserve_min_remaining_columns": int(_GLOBAL_LOGICAL_CLASS_RESERVE_MIN_REMAINING_COLUMNS),
                    "logical_class_quota_top_classes": int(_GLOBAL_LOGICAL_CLASS_QUOTA_TOP_CLASSES),
                    "logical_class_quota_reserved_slots": int(_GLOBAL_LOGICAL_CLASS_QUOTA_RESERVED_SLOTS),
                    "logical_class_quota_min_remaining_columns": int(_GLOBAL_LOGICAL_CLASS_QUOTA_MIN_REMAINING_COLUMNS),
                    "lineage_reserve_checkpoint_stride": int(_GLOBAL_LINEAGE_RESERVE_CHECKPOINT_STRIDE),
                    "lineage_reserve_reserved_slots": int(_GLOBAL_LINEAGE_RESERVE_RESERVED_SLOTS),
                    "logical_rerank_columns": int(_GLOBAL_LOGICAL_RERANK_COLUMNS),
                    "logical_rerank_shortlist_size": int(_GLOBAL_LOGICAL_RERANK_SHORTLIST_SIZE),
                    "logical_rerank_min_classes": int(_GLOBAL_LOGICAL_RERANK_MIN_CLASSES),
                    "logical_rerank_state_budget": int(_GLOBAL_LOGICAL_RERANK_STATE_BUDGET),
                    "logical_rerank_transition_budget": int(_GLOBAL_LOGICAL_RERANK_TRANSITION_BUDGET),
                    "logical_rerank_checkpoint_stride": int(_GLOBAL_LOGICAL_RERANK_CHECKPOINT_STRIDE),
                    "logical_rerank_max_passes": int(_GLOBAL_LOGICAL_RERANK_MAX_PASSES),
                    "logical_rerank_mode": str(_GLOBAL_LOGICAL_RERANK_MODE),
                    "final_logical_select_mode": str(_GLOBAL_FINAL_LOGICAL_SELECT_MODE),
                    "final_logical_select_rep_cost_weight": float(_GLOBAL_FINAL_LOGICAL_SELECT_REP_COST_WEIGHT),
                    "final_logical_select_max_log_mass_gap": float(_GLOBAL_FINAL_LOGICAL_SELECT_MAX_LOG_MASS_GAP),
                    "final_logical_select_rank2_viterbi_tolerance": float(
                        _GLOBAL_FINAL_LOGICAL_SELECT_RANK2_VITERBI_TOLERANCE
                    ),
                    "log_mass_primary_production_path": (
                        "guarded" if bool(_GLOBAL_PRODUCTION_FAST_MODE) else "off"
                    ),
                    "merge_duplicate_states": bool(_GLOBAL_MERGE_DUPLICATE_STATES),
                    "state_merge_period_columns": int(_GLOBAL_STATE_MERGE_PERIOD_COLUMNS),
                    "track_best_path": bool(_GLOBAL_TRACK_BEST_PATH),
                    "return_terminal_maps": bool(_GLOBAL_EXPORT_TERMINAL_SELECTOR_SIGNALS),
                    "return_correction": bool(str(family.correction_state_mode) != "none"),
                    "correction_merge_mode": "exact",
                    "diagnostic_truth_logical_mask": int(truth_logical),
                    "diagnostic_truth_column_identity_mask": int(truth_column_identity_mask),
                }
                if (
                    bool(_GLOBAL_BIDIRECTIONAL_SPLICE_RERANK)
                    and str(_GLOBAL_SPLICE_CUT_SELECTOR).strip().lower()
                    in {"smallest_cutoff_gap", "flat_cutoff", "cutoff_gap", "min_cutoff_gap"}
                ):
                    decode_kwargs["pre_prune_score_trace_checkpoint_stride"] = 1
                if bool(_GLOBAL_EXPORT_FRONTIER_PRESSURE_TRACE):
                    decode_kwargs["pre_prune_score_trace_checkpoint_stride"] = 1
                decoder_mode_key = _normalize_decoder_mode(str(_GLOBAL_DECODER_MODE))
                if decoder_mode_key == "backward":
                    result = progressive.decode_progressive_directional(
                        family.columns,
                        sweep_direction="backward",
                        layout=family.layout,
                        backward_columns=(_GLOBAL_BACKWARD_COLUMNS or None),
                        backward_layout=_GLOBAL_BACKWARD_LAYOUT,
                        **decode_kwargs,
                    )
                elif decoder_mode_key == "bidirectional_committee":
                    result = progressive.decode_progressive_bidirectional_committee(
                        family.columns,
                        layout=family.layout,
                        backward_columns=(_GLOBAL_BACKWARD_COLUMNS or None),
                        backward_layout=_GLOBAL_BACKWARD_LAYOUT,
                        **decode_kwargs,
                    )
                elif decoder_mode_key == "forward_guided_backward":
                    result = progressive.decode_progressive_forward_guided_backward(
                        family.columns,
                        layout=family.layout,
                        backward_columns=(_GLOBAL_BACKWARD_COLUMNS or None),
                        backward_layout=_GLOBAL_BACKWARD_LAYOUT,
                        forward_guidance_weight=float(_GLOBAL_FORWARD_GUIDANCE_WEIGHT),
                        forward_guidance_clip=float(_GLOBAL_FORWARD_GUIDANCE_CLIP),
                        forward_guidance_trigger_gap=float(_GLOBAL_FORWARD_GUIDANCE_TRIGGER_GAP),
                        forward_guidance_widen_factor=float(_GLOBAL_FORWARD_GUIDANCE_WIDEN_FACTOR),
                        forward_guidance_min_info_bits=float(_GLOBAL_FORWARD_GUIDANCE_MIN_INFO_BITS),
                        forward_guidance_snapshot_factor=float(_GLOBAL_FORWARD_GUIDANCE_SNAPSHOT_FACTOR),
                        forward_guidance_snapshot_gap=_GLOBAL_FORWARD_GUIDANCE_SNAPSHOT_GAP,
                        forward_guidance_snapshot_source=str(_GLOBAL_FORWARD_GUIDANCE_SNAPSHOT_SOURCE),
                        forward_guidance_hamming_radius=int(_GLOBAL_FORWARD_GUIDANCE_HAMMING_RADIUS),
                        forward_guidance_trigger_mode=str(_GLOBAL_FORWARD_GUIDANCE_TRIGGER_MODE),
                        forward_guidance_nearcut_gap=float(_GLOBAL_FORWARD_GUIDANCE_NEARCUT_GAP),
                        forward_guidance_pool_trigger_min_positive_nearcut=int(
                            _GLOBAL_FORWARD_GUIDANCE_POOL_TRIGGER_MIN_POSITIVE_NEARCUT
                        ),
                        forward_guidance_diversity_fallback=str(_GLOBAL_FORWARD_GUIDANCE_DIVERSITY_FALLBACK),
                        forward_guidance_mode=str(_GLOBAL_FORWARD_GUIDANCE_MODE),
                        **decode_kwargs,
                    )
                elif decoder_mode_key == "bidirectional_middle_join":
                    result = progressive.decode_progressive_bidirectional_middle_join(
                        family.columns,
                        layout=family.layout,
                        backward_columns=(_GLOBAL_BACKWARD_COLUMNS or None),
                        backward_layout=_GLOBAL_BACKWARD_LAYOUT,
                        middle_join_prefix_columns=_GLOBAL_MIDDLE_JOIN_PREFIX_COLUMNS,
                        middle_join_multicut_prefix_columns=_GLOBAL_MIDDLE_JOIN_MULTICUT_PREFIX_COLUMNS,
                        middle_join_multicut_stride=_GLOBAL_MIDDLE_JOIN_MULTICUT_STRIDE,
                        middle_join_multicut_max_cuts=_GLOBAL_MIDDLE_JOIN_MULTICUT_MAX_CUTS,
                        middle_join_multicut_weight_mode=_GLOBAL_MIDDLE_JOIN_MULTICUT_WEIGHT_MODE,
                        middle_join_cut_window_columns=_GLOBAL_MIDDLE_JOIN_CUT_WINDOW_COLUMNS,
                        middle_join_cut_beam_factor=_GLOBAL_MIDDLE_JOIN_CUT_BEAM_FACTOR,
                        **decode_kwargs,
                    )
                else:
                    result = progressive.decode_progressive(
                        family.columns,
                        layout=family.layout,
                        **decode_kwargs,
                    )
                if bool(_GLOBAL_BIDIRECTIONAL_SPLICE_RERANK) and str(result.status) == "ok":
                    splice_summary = progressive.diagnose_bidirectional_splice_rerank(
                        family.columns,
                        baseline_result=result,
                        layout=family.layout,
                        backward_columns=(_GLOBAL_BACKWARD_COLUMNS or None),
                        backward_layout=_GLOBAL_BACKWARD_LAYOUT,
                        splice_candidate_count=int(_GLOBAL_SPLICE_CANDIDATE_COUNT),
                        splice_cut_selector=str(_GLOBAL_SPLICE_CUT_SELECTOR),
                        splice_max_cuts=int(_GLOBAL_SPLICE_MAX_CUTS),
                        splice_aggregate=str(_GLOBAL_SPLICE_AGGREGATE),
                        **decode_kwargs,
                    )
                    replacement_logical = splice_summary.selected_logical_mask
                    result = replace(result, splice_rerank_summary=splice_summary)
                    if (
                        bool(_GLOBAL_SPLICE_REPLACE_FINAL_SELECTION)
                        and replacement_logical is not None
                    ):
                        result = replace(result, logical_hat=int(replacement_logical))
            except Exception as exc:
                exception_message = f"{type(exc).__name__}: {exc}"
                result = progressive.ProgressiveDecodeResult(
                    status="exception",
                    target_syndrome=int(syndrome),
                    logical_hat=0,
                    logical_posteriors=tuple(),
                    log_evidence=float("-inf"),
                    state_count_by_column=tuple(),
                    pruned_state_count_by_column=tuple(),
                    expanded_transition_count_by_column=tuple(),
                    lookahead_transition_count_by_column=tuple(),
                    max_state_count=0,
                    beam_size=int(beam_size),
                    lookahead_depth=int(_GLOBAL_LOOKAHEAD_DEPTH),
                    score_mode=str(score_mode),
                    beam_score_gap_policy_mode=(
                        ""
                        if _GLOBAL_BEAM_SCORE_GAP_POLICY is None
                        else str(_GLOBAL_BEAM_SCORE_GAP_POLICY.mode)
                    ),
                    selective_secondary_score_mode=str(_GLOBAL_SELECTIVE_SECONDARY_SCORE_MODE),
                    selective_secondary_trigger_gap=float(_GLOBAL_SELECTIVE_SECONDARY_TRIGGER_GAP),
                    selective_secondary_band_size=int(_GLOBAL_SELECTIVE_SECONDARY_BAND_SIZE),
                    selective_local_lookahead_mode=str(_GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_MODE),
                    selective_local_lookahead_score_mode=progressive._selective_local_lookahead_score_mode(
                        str(_GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_MODE)
                    ),
                    selective_local_lookahead_cutoff_gap_threshold=float(
                        _GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_CUTOFF_GAP_THRESHOLD
                    ),
                    selective_local_lookahead_near_cut_width=float(
                        _GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_NEAR_CUT_WIDTH
                    ),
                    selective_local_lookahead_max_candidates=int(
                        _GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_MAX_CANDIDATES
                    ),
                    selective_local_lookahead_candidate_top1_share_threshold=float(
                        _GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_CANDIDATE_TOP1_SHARE_THRESHOLD
                    ),
                    selective_local_lookahead_support_gap_threshold=float(
                        _GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_SUPPORT_GAP_THRESHOLD
                    ),
                    selective_local_lookahead_overflow_ratio_threshold=float(
                        _GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_OVERFLOW_RATIO_THRESHOLD
                    ),
                    delayed_pruning_gap_threshold=float(_GLOBAL_DELAYED_PRUNING_GAP_THRESHOLD),
                    delayed_pruning_factor=int(_GLOBAL_DELAYED_PRUNING_FACTOR),
                    pruning_replay_checkpoint_stride=int(_GLOBAL_PRUNING_REPLAY_CHECKPOINT_STRIDE),
                    pruning_replay_horizon=int(_GLOBAL_PRUNING_REPLAY_HORIZON),
                    lineage_reserve_checkpoint_stride=int(_GLOBAL_LINEAGE_RESERVE_CHECKPOINT_STRIDE),
                    lineage_reserve_reserved_slots=int(_GLOBAL_LINEAGE_RESERVE_RESERVED_SLOTS),
                    superstep_mode=str(_GLOBAL_SUPERSTEP_MODE),
                    superstep_path_budget=int(_GLOBAL_SUPERSTEP_PATH_BUDGET),
                    superstep_state_budget=int(_GLOBAL_SUPERSTEP_STATE_BUDGET),
                    superstep_transition_budget=int(_GLOBAL_SUPERSTEP_TRANSITION_BUDGET),
                    detector_bucket_pruning=bool(_GLOBAL_DETECTOR_BUCKET_PRUNING),
                    detector_bucket_max_logicals=int(_GLOBAL_DETECTOR_BUCKET_MAX_LOGICALS),
                    lookahead_shortlist_size=int(_GLOBAL_LOOKAHEAD_SHORTLIST_SIZE),
                    tail_exact_columns=int(_GLOBAL_TAIL_EXACT_COLUMNS),
                    tail_exact_start_column=-1,
                    logical_rerank_columns=0,
                    logical_rerank_shortlist_size=0,
                    logical_rerank_min_classes=0,
                    logical_rerank_state_budget=0,
                    logical_rerank_transition_budget=0,
                    logical_rerank_checkpoint_stride=0,
                    logical_rerank_max_passes=1,
                    logical_rerank_mode="exact_tail",
                    logical_rerank_pass_count=0,
                    final_logical_select_mode=str(_GLOBAL_FINAL_LOGICAL_SELECT_MODE),
                    final_logical_select_rep_cost_weight=float(_GLOBAL_FINAL_LOGICAL_SELECT_REP_COST_WEIGHT),
                    final_logical_select_max_log_mass_gap=float(_GLOBAL_FINAL_LOGICAL_SELECT_MAX_LOG_MASS_GAP),
                    final_logical_select_rank2_viterbi_tolerance=float(
                        _GLOBAL_FINAL_LOGICAL_SELECT_RANK2_VITERBI_TOLERANCE
                    ),
                    final_logical_select_gate_triggered=False,
                    terminal_top_log_mass_gap=float("inf"),
                    delayed_pruning_trigger_count=0,
                    delayed_pruning_active_prune_count=0,
                    delayed_pruning_peak_beam_size=0 if int(beam_size) <= 0 else int(beam_size),
                    best_path_states=tuple(),
                    best_path_log_prob=float("-inf"),
                )
            decode_s = float(time.perf_counter() - started)
            state_count_profile_text = (
                _serialize_int_series(result.state_count_by_column)
                if bool(_GLOBAL_EXPORT_STATE_COUNT_PROFILE) and bool(result.state_count_by_column)
                else ""
            )
            if bool(_GLOBAL_EXPORT_FRONTIER_PRESSURE_TRACE) and tuple(result.pre_prune_score_trace):
                trace_order = (
                    str(_GLOBAL_BACKWARD_COLUMN_ORDER)
                    if str(_GLOBAL_DECODER_MODE) == "backward" and str(_GLOBAL_BACKWARD_COLUMN_ORDER)
                    else str(_GLOBAL_COLUMN_ORDER_LABEL)
                )
                _GLOBAL_FRONTIER_PRESSURE_TRACE_ROWS.extend(
                    _frontier_pressure_rows_from_result(
                        result=result,
                        backend=str(family.backend),
                        scope=str(family.scope),
                        shot=int(shot_index),
                        beam_size=int(beam_size),
                        delta=_GLOBAL_BEAM_SCORE_GAP_THRESHOLD,
                        score_mode=str(score_mode),
                        column_order=str(trace_order),
                        diagnostic_truth_logical_mask=int(truth_logical),
                    )
                )
            out.append(
                _classify_row(
                    shot_index=int(shot_index),
                    truth_logical=int(truth_logical),
                    truth_nonzero_locations=int(truth_nonzero_locations),
                    family=family,
                    decoder_mode=str(_GLOBAL_DECODER_MODE),
                    backward_column_order=str(_GLOBAL_BACKWARD_COLUMN_ORDER),
                    beam_size=int(beam_size),
                    score_mode=str(score_mode),
                    beam_score_gap_threshold=_GLOBAL_BEAM_SCORE_GAP_THRESHOLD,
                    beam_score_gap_policy=_GLOBAL_BEAM_SCORE_GAP_POLICY,
                    selective_secondary_score_mode=str(_GLOBAL_SELECTIVE_SECONDARY_SCORE_MODE),
                    selective_secondary_trigger_gap=float(_GLOBAL_SELECTIVE_SECONDARY_TRIGGER_GAP),
                    selective_secondary_band_size=int(_GLOBAL_SELECTIVE_SECONDARY_BAND_SIZE),
                    lookahead_depth=int(_GLOBAL_LOOKAHEAD_DEPTH),
                    lookahead_shortlist_size=int(_GLOBAL_LOOKAHEAD_SHORTLIST_SIZE),
                    delayed_pruning_gap_threshold=float(_GLOBAL_DELAYED_PRUNING_GAP_THRESHOLD),
                    delayed_pruning_factor=int(_GLOBAL_DELAYED_PRUNING_FACTOR),
                    pruning_replay_checkpoint_stride=int(_GLOBAL_PRUNING_REPLAY_CHECKPOINT_STRIDE),
                    pruning_replay_horizon=int(_GLOBAL_PRUNING_REPLAY_HORIZON),
                    tail_exact_columns=int(_GLOBAL_TAIL_EXACT_COLUMNS),
                    superstep_mode=str(_GLOBAL_SUPERSTEP_MODE),
                    superstep_path_budget=int(_GLOBAL_SUPERSTEP_PATH_BUDGET),
                    superstep_state_budget=int(_GLOBAL_SUPERSTEP_STATE_BUDGET),
                    superstep_transition_budget=int(_GLOBAL_SUPERSTEP_TRANSITION_BUDGET),
                    detector_bucket_pruning=bool(_GLOBAL_DETECTOR_BUCKET_PRUNING),
                    detector_bucket_max_logicals=int(_GLOBAL_DETECTOR_BUCKET_MAX_LOGICALS),
                    logical_class_reserve_min_classes=int(_GLOBAL_LOGICAL_CLASS_RESERVE_MIN_CLASSES),
                    logical_class_reserve_max_replacements=int(_GLOBAL_LOGICAL_CLASS_RESERVE_MAX_REPLACEMENTS),
                    logical_class_reserve_min_remaining_columns=int(_GLOBAL_LOGICAL_CLASS_RESERVE_MIN_REMAINING_COLUMNS),
                    logical_class_quota_top_classes=int(_GLOBAL_LOGICAL_CLASS_QUOTA_TOP_CLASSES),
                    logical_class_quota_reserved_slots=int(_GLOBAL_LOGICAL_CLASS_QUOTA_RESERVED_SLOTS),
                    logical_class_quota_min_remaining_columns=int(_GLOBAL_LOGICAL_CLASS_QUOTA_MIN_REMAINING_COLUMNS),
                    lineage_reserve_checkpoint_stride=int(_GLOBAL_LINEAGE_RESERVE_CHECKPOINT_STRIDE),
                    lineage_reserve_reserved_slots=int(_GLOBAL_LINEAGE_RESERVE_RESERVED_SLOTS),
                    logical_rerank_columns=int(_GLOBAL_LOGICAL_RERANK_COLUMNS),
                    logical_rerank_shortlist_size=int(_GLOBAL_LOGICAL_RERANK_SHORTLIST_SIZE),
                    logical_rerank_min_classes=int(_GLOBAL_LOGICAL_RERANK_MIN_CLASSES),
                    logical_rerank_state_budget=int(_GLOBAL_LOGICAL_RERANK_STATE_BUDGET),
                    logical_rerank_transition_budget=int(_GLOBAL_LOGICAL_RERANK_TRANSITION_BUDGET),
                    logical_rerank_checkpoint_stride=int(_GLOBAL_LOGICAL_RERANK_CHECKPOINT_STRIDE),
                    logical_rerank_max_passes=int(_GLOBAL_LOGICAL_RERANK_MAX_PASSES),
                    logical_rerank_mode=str(_GLOBAL_LOGICAL_RERANK_MODE),
                    track_best_path=bool(_GLOBAL_TRACK_BEST_PATH),
                    merge_duplicate_states=bool(_GLOBAL_MERGE_DUPLICATE_STATES),
                    state_merge_period_columns=int(_GLOBAL_STATE_MERGE_PERIOD_COLUMNS),
                    decode_s=float(decode_s),
                    result=result,
                    state_count_profile_text=str(state_count_profile_text),
                    exception_message=str(exception_message),
                )
            )
    return out


def _run_shard(task: dict[str, object]) -> dict[str, object]:
    global _GLOBAL_FRONTIER_PRESSURE_TRACE_ROWS
    beam_sizes = tuple(int(value) for value in task["beam_sizes"])
    start = time.time()
    rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    _GLOBAL_FRONTIER_PRESSURE_TRACE_ROWS = []
    shot_indices = [int(value) for value in task["shot_indices"]]
    partial_path_text = str(task.get("partial_path", "")).strip()
    progress_path_text = str(task.get("progress_path", "")).strip()
    partial_path = Path(partial_path_text) if partial_path_text else None
    progress_path = Path(progress_path_text) if progress_path_text else None
    if partial_path is not None and partial_path.exists():
        partial_path.unlink()
    if progress_path is not None and progress_path.exists():
        progress_path.unlink()
    if progress_path is not None:
        _write_json(
            progress_path,
            {
                "task_id": int(task["task_id"]),
                "shots_total": int(len(shot_indices)),
                "shots_completed": 0,
                "rows_written": 0,
                "elapsed_s": 0.0,
                "status": "running",
            },
        )
    for local_index, shot_index in enumerate(shot_indices, start=1):
        trace_start = int(len(_GLOBAL_FRONTIER_PRESSURE_TRACE_ROWS))
        shot_rows = _decode_one_shot(int(shot_index), beam_sizes=beam_sizes)
        if len(_GLOBAL_FRONTIER_PRESSURE_TRACE_ROWS) > int(trace_start):
            trace_rows.extend(_GLOBAL_FRONTIER_PRESSURE_TRACE_ROWS[int(trace_start):])
        rows.extend(shot_rows)
        if partial_path is not None and shot_rows:
            _append_csv(partial_path, shot_rows, _fieldnames_from_rows(shot_rows, ["shot"]))
        if progress_path is not None:
            _write_json(
                progress_path,
                {
                    "task_id": int(task["task_id"]),
                    "shots_total": int(len(shot_indices)),
                    "shots_completed": int(local_index),
                    "rows_written": int(len(rows)),
                    "elapsed_s": float(time.time() - start),
                    "status": "running",
                },
            )
    elapsed_s = float(time.time() - start)
    if progress_path is not None:
        _write_json(
            progress_path,
            {
                "task_id": int(task["task_id"]),
                "shots_total": int(len(shot_indices)),
                "shots_completed": int(len(shot_indices)),
                "rows_written": int(len(rows)),
                "elapsed_s": float(elapsed_s),
                "status": "completed",
            },
        )
    return {
        "task_id": int(task["task_id"]),
        "shots_completed": int(len(shot_indices)),
        "elapsed_s": float(elapsed_s),
        "rows": rows,
        "pressure_trace_rows": trace_rows,
    }


def _state_merge_mode_label(*, merge_duplicate_states: bool, state_merge_period_columns: int) -> str:
    if bool(merge_duplicate_states):
        return "exact"
    if int(state_merge_period_columns) > 0:
        return f"periodic_{int(state_merge_period_columns)}"
    return "disabled"


def _summary_row(
    *,
    rows: Sequence[dict[str, object]],
    decoder: str,
    family: str,
    decoder_mode: str,
    backward_column_order: str,
    correction_state_mode: str,
    correction_state_bits: int,
    state_merge_mode: str,
    beam_size: int,
    score_mode: str,
    beam_score_gap_threshold: float | None,
    lookahead_depth: int,
    lookahead_shortlist_size: int,
    delayed_pruning_gap_threshold: float,
    delayed_pruning_factor: int,
    pruning_replay_checkpoint_stride: int,
    pruning_replay_horizon: int,
    tail_exact_columns: int,
    superstep_mode: str,
    detector_bucket_pruning: bool,
    detector_bucket_max_logicals: int,
    logical_class_reserve_min_classes: int,
    logical_class_reserve_max_replacements: int,
    logical_class_reserve_min_remaining_columns: int,
    logical_class_quota_top_classes: int,
    logical_class_quota_reserved_slots: int,
    logical_class_quota_min_remaining_columns: int,
    lineage_reserve_checkpoint_stride: int,
    lineage_reserve_reserved_slots: int,
    logical_rerank_columns: int,
    logical_rerank_shortlist_size: int,
    logical_rerank_min_classes: int,
    logical_rerank_state_budget: int,
    logical_rerank_transition_budget: int,
    logical_rerank_checkpoint_stride: int,
    logical_rerank_max_passes: int,
    logical_rerank_mode: str,
    track_best_path: bool,
    final_logical_select_mode: str = "log_mass",
    final_logical_select_rep_cost_weight: float = 0.0,
    final_logical_select_max_log_mass_gap: float = float("inf"),
    final_logical_select_rank2_viterbi_tolerance: float = 0.0,
    selective_secondary_score_mode: str = "",
    selective_secondary_trigger_gap: float = 0.0,
    selective_secondary_band_size: int = 0,
) -> dict[str, object]:
    shots = len(rows)
    round_count = int(rows[0].get("noisy_rounds", ROUND_COUNT)) if rows else int(ROUND_COUNT)
    fail_total = sum(int(not bool(row["frame_ok"])) for row in rows)
    logical_fail = sum(int(str(row["frame_fail_type"]) == "logical_fail") for row in rows)
    logical_fail_truth_missing_terminal = sum(
        int(str(row.get("truth_logical_failure_mode", "")) == "truth_missing_terminal")
        for row in rows
    )
    logical_fail_truth_present_but_not_selected = sum(
        int(str(row.get("truth_logical_failure_mode", "")) == "truth_present_but_not_selected")
        for row in rows
    )
    syndrome_fail = sum(int(str(row["frame_fail_type"]) == "syndrome_fail") for row in rows)
    exception_fail = sum(int(str(row["frame_fail_type"]) == "exception_fail") for row in rows)
    baseline_fail_total = sum(
        int(str(row.get("baseline_fail_type", row.get("frame_fail_type", ""))) != "success")
        for row in rows
    )
    baseline_logical_fail = sum(
        int(str(row.get("baseline_fail_type", "")) == "logical_fail") for row in rows
    )
    baseline_syndrome_fail = sum(
        int(str(row.get("baseline_fail_type", "")) == "syndrome_fail") for row in rows
    )
    baseline_exception_fail = sum(
        int(str(row.get("baseline_fail_type", "")) == "exception_fail") for row in rows
    )
    splice_enabled_rows = [row for row in rows if bool(row.get("splice_enabled", False))]
    splice_fail_total = sum(
        int(str(row.get("splice_fail_type", "")) != "success")
        for row in splice_enabled_rows
    )
    splice_logical_fail = sum(
        int(str(row.get("splice_fail_type", "")) == "logical_fail")
        for row in splice_enabled_rows
    )
    splice_syndrome_fail = sum(
        int(str(row.get("splice_fail_type", "")) == "syndrome_fail")
        for row in splice_enabled_rows
    )
    splice_exception_fail = sum(
        int(str(row.get("splice_fail_type", "")) == "exception_fail")
        for row in splice_enabled_rows
    )
    splice_unchanged_failure = sum(
        int(
            str(row.get("baseline_fail_type", "")) != "success"
            and str(row.get("splice_fail_type", "")) != "success"
        )
        for row in splice_enabled_rows
    )
    splice_truth_present_but_not_selected = sum(
        int(
            bool(row.get("splice_truth_present_in_candidates", False))
            and str(row.get("splice_fail_type", "")) == "logical_fail"
        )
        for row in splice_enabled_rows
    )
    fer = float(fail_total) / float(shots) if shots else float("nan")
    total_decode_s = float(sum(float(row["decode_s"]) for row in rows)) if rows else float("nan")
    total_transition_evals = float(sum(float(row["transition_evals_total"]) for row in rows)) if rows else 0.0
    total_transition_evals_physical = (
        float(sum(float(row.get("transition_evals_physical_total", 0.0)) for row in rows)) if rows else 0.0
    )
    matrix_cols = int(rows[0]["matrix_cols"]) if rows else 0
    trace_mean_values = [
        value
        for row in rows
        if (value := _optional_float(row.get("beam_score_gap_threshold_trace_mean", float("nan")))) is not None
        and math.isfinite(value)
    ]
    trace_min_values = [
        value
        for row in rows
        if (value := _optional_float(row.get("beam_score_gap_threshold_trace_min", float("nan")))) is not None
        and math.isfinite(value)
    ]
    trace_max_values = [
        value
        for row in rows
        if (value := _optional_float(row.get("beam_score_gap_threshold_trace_max", float("nan")))) is not None
        and math.isfinite(value)
    ]
    def _row_mean(key: str) -> float:
        values = _finite_row_values(rows, str(key))
        return float(np.mean(np.asarray(values, dtype=np.float64))) if values else float("nan")

    def _row_mean_nonnegative(key: str) -> float:
        values = [
            float(value)
            for value in _finite_row_values(rows, str(key))
            if int(value) >= 0
        ]
        return float(np.mean(np.asarray(values, dtype=np.float64))) if values else float("nan")

    def _row_mean_positive(key: str) -> float:
        values = [
            float(value)
            for value in _finite_row_values(rows, str(key))
            if int(value) > 0
        ]
        return float(np.mean(np.asarray(values, dtype=np.float64))) if values else float("nan")

    def _row_positive_count(key: str) -> int:
        return sum(
            int(float(value) > 0.0)
            for value in _finite_row_values(rows, str(key))
        )

    def _row_eq_count(key: str, target: int) -> int:
        return sum(
            int(int(value) == int(target))
            for value in _finite_row_values(rows, str(key))
        )

    def _row_max(key: str) -> float:
        values = _finite_row_values(rows, str(key))
        return float(max(values)) if values else float("nan")

    def _row_min(key: str) -> float:
        values = _finite_row_values(rows, str(key))
        return float(min(values)) if values else float("nan")

    def _row_min_nonnegative(key: str) -> int:
        values = [
            int(value)
            for value in _finite_row_values(rows, str(key))
            if int(value) >= 0
        ]
        return int(min(values)) if values else -1

    def _row_sum(key: str) -> float:
        total = 0.0
        for row in rows:
            value = _optional_float(row.get(str(key), 0.0))
            if value is not None and math.isfinite(float(value)):
                total += float(value)
        return float(total)

    def _row_quantile(key: str, q: float) -> float:
        values = _finite_row_values(rows, str(key))
        return _quantile(values, float(q)) if values else float("nan")

    return {
        "decoder": str(decoder),
        "family": str(family),
        "decoder_mode": _normalize_decoder_mode(str(decoder_mode)),
        "backward_column_order": str(backward_column_order),
        "correction_state_mode": str(correction_state_mode),
        "correction_state_bits": int(correction_state_bits),
        "state_merge_mode": str(state_merge_mode),
        "state_merge_period_columns": int(rows[0].get("state_merge_period_columns", 0)) if rows else 0,
        "beam_size": int(beam_size),
        "score_mode": str(score_mode),
        "beam_score_gap_threshold": (
            float(beam_score_gap_threshold)
            if beam_score_gap_threshold is not None and math.isfinite(float(beam_score_gap_threshold))
            else ""
        ),
        "beam_score_gap_policy_mode": str(rows[0].get("beam_score_gap_policy_mode", "")) if rows else "",
        "beam_score_gap_policy_base_threshold": rows[0].get("beam_score_gap_policy_base_threshold", "") if rows else "",
        "beam_score_gap_policy_final_threshold": rows[0].get("beam_score_gap_policy_final_threshold", "") if rows else "",
        "beam_score_gap_policy_slope": rows[0].get("beam_score_gap_policy_slope", "") if rows else "",
        "beam_score_gap_policy_reference_count": rows[0].get("beam_score_gap_policy_reference_count", "") if rows else "",
        "beam_score_gap_policy_min_threshold": rows[0].get("beam_score_gap_policy_min_threshold", "") if rows else "",
        "beam_score_gap_policy_max_threshold": rows[0].get("beam_score_gap_policy_max_threshold", "") if rows else "",
        "beam_score_gap_threshold_trace_mean": (
            float(np.mean(np.asarray(trace_mean_values, dtype=np.float64)))
            if trace_mean_values
            else ""
        ),
        "beam_score_gap_threshold_trace_min": (
            float(min(trace_min_values)) if trace_min_values else ""
        ),
        "beam_score_gap_threshold_trace_max": (
            float(max(trace_max_values)) if trace_max_values else ""
        ),
        "selective_secondary_score_mode": str(rows[0].get("selective_secondary_score_mode", "")) if rows else "",
        "selective_secondary_trigger_gap": rows[0].get("selective_secondary_trigger_gap", "") if rows else "",
        "selective_secondary_band_size": rows[0].get("selective_secondary_band_size", "") if rows else "",
        "selective_local_lookahead_mode": str(rows[0].get("selective_local_lookahead_mode", "none")) if rows else "none",
        "selective_local_lookahead_score_mode": str(rows[0].get("selective_local_lookahead_score_mode", "")) if rows else "",
        "selective_local_lookahead_cutoff_gap_threshold": rows[0].get("selective_local_lookahead_cutoff_gap_threshold", "") if rows else "",
        "selective_local_lookahead_near_cut_width": rows[0].get("selective_local_lookahead_near_cut_width", "") if rows else "",
        "selective_local_lookahead_max_candidates": rows[0].get("selective_local_lookahead_max_candidates", "") if rows else "",
        "selective_local_lookahead_candidate_top1_share_threshold": rows[0].get(
            "selective_local_lookahead_candidate_top1_share_threshold",
            "",
        ) if rows else "",
        "selective_local_lookahead_support_gap_threshold": rows[0].get(
            "selective_local_lookahead_support_gap_threshold",
            "",
        ) if rows else "",
        "selective_local_lookahead_overflow_ratio_threshold": rows[0].get(
            "selective_local_lookahead_overflow_ratio_threshold",
            "",
        ) if rows else "",
        "forward_guidance_weight": rows[0].get("forward_guidance_weight", "") if rows else "",
        "forward_guidance_clip": rows[0].get("forward_guidance_clip", "") if rows else "",
        "forward_guidance_trigger_gap": rows[0].get("forward_guidance_trigger_gap", "") if rows else "",
        "forward_guidance_widen_factor": rows[0].get("forward_guidance_widen_factor", "") if rows else "",
        "forward_guidance_min_info_bits": rows[0].get("forward_guidance_min_info_bits", "") if rows else "",
        "forward_guidance_snapshot_factor": rows[0].get("forward_guidance_snapshot_factor", "") if rows else "",
        "forward_guidance_snapshot_gap": rows[0].get("forward_guidance_snapshot_gap", "") if rows else "",
        "forward_guidance_snapshot_source": rows[0].get("forward_guidance_snapshot_source", "") if rows else "",
        "forward_guidance_hamming_radius": rows[0].get("forward_guidance_hamming_radius", "") if rows else "",
        "forward_guidance_trigger_mode": rows[0].get("forward_guidance_trigger_mode", "") if rows else "",
        "forward_guidance_nearcut_gap": rows[0].get("forward_guidance_nearcut_gap", "") if rows else "",
        "forward_guidance_pool_trigger_min_positive_nearcut": rows[0].get(
            "forward_guidance_pool_trigger_min_positive_nearcut",
            "",
        ) if rows else "",
        "forward_guidance_diversity_fallback": rows[0].get(
            "forward_guidance_diversity_fallback",
            "",
        ) if rows else "",
        "forward_guidance_mode": rows[0].get("forward_guidance_mode", "") if rows else "",
        "forward_guidance_diag_step_count_mean": _row_mean("forward_guidance_diag_step_count"),
        "forward_guidance_triggered_step_count_mean": _row_mean(
            "forward_guidance_triggered_step_count"
        ),
        "forward_guidance_triggered_fraction_mean": _row_mean(
            "forward_guidance_triggered_fraction"
        ),
        "forward_guidance_top_gap_triggered_step_count_mean": _row_mean(
            "forward_guidance_top_gap_triggered_step_count"
        ),
        "forward_guidance_top_gap_triggered_fraction_mean": _row_mean(
            "forward_guidance_top_gap_triggered_fraction"
        ),
        "forward_guidance_support_aware_triggered_step_count_mean": _row_mean(
            "forward_guidance_support_aware_triggered_step_count"
        ),
        "forward_guidance_support_aware_triggered_fraction_mean": _row_mean(
            "forward_guidance_support_aware_triggered_fraction"
        ),
        "forward_guidance_base_top_primary_gap_mean": _row_mean(
            "forward_guidance_base_top_primary_gap_mean"
        ),
        "forward_guidance_base_top_primary_gap_p10": _row_quantile(
            "forward_guidance_base_top_primary_gap_p10",
            0.10,
        ),
        "forward_guidance_base_top_primary_gap_p50": _row_quantile(
            "forward_guidance_base_top_primary_gap_p50",
            0.50,
        ),
        "forward_guidance_base_top_primary_gap_p90": _row_quantile(
            "forward_guidance_base_top_primary_gap_p90",
            0.90,
        ),
        "forward_guidance_alignment_metadata_step_count_mean": _row_mean(
            "forward_guidance_alignment_metadata_step_count"
        ),
        "forward_guidance_aligned_step_count_mean": _row_mean("forward_guidance_aligned_step_count"),
        "forward_guidance_no_alignment_step_count_mean": _row_mean(
            "forward_guidance_no_alignment_step_count"
        ),
        "forward_guidance_top_rank_changed_count_mean": _row_mean(
            "forward_guidance_top_rank_changed_count"
        ),
        "forward_guidance_top_rank_changed_fraction_mean": _row_mean(
            "forward_guidance_top_rank_changed_fraction"
        ),
        "forward_guidance_top_logical_changed_count_mean": _row_mean(
            "forward_guidance_top_logical_changed_count"
        ),
        "forward_guidance_top_logical_changed_fraction_mean": _row_mean(
            "forward_guidance_top_logical_changed_fraction"
        ),
        "forward_guidance_selected_distance_abs_mean": _row_mean(
            "forward_guidance_selected_distance_abs_mean"
        ),
        "forward_guidance_selected_distance_abs_max": _row_max(
            "forward_guidance_selected_distance_abs_max"
        ),
        "forward_guidance_selected_state_count_mean": _row_mean(
            "forward_guidance_selected_state_count_mean"
        ),
        "forward_guidance_candidate_interval_row_count_mean": _row_mean(
            "forward_guidance_candidate_interval_row_count_mean"
        ),
        "forward_guidance_candidate_snapshot_count_mean": _row_mean(
            "forward_guidance_candidate_snapshot_count_mean"
        ),
        "forward_guidance_positive_aligned_snapshot_count_mean": _row_mean(
            "forward_guidance_positive_aligned_snapshot_count_mean"
        ),
        "forward_guidance_backward_active_row_count_mean": _row_mean(
            "forward_guidance_backward_active_row_count_mean"
        ),
        "forward_guidance_common_active_row_count_mean": _row_mean(
            "forward_guidance_common_active_row_count_mean"
        ),
        "forward_guidance_aligned_row_count_mean": _row_mean(
            "forward_guidance_aligned_row_count_mean"
        ),
        "forward_guidance_aligned_row_count_max": _row_max("forward_guidance_aligned_row_count_max"),
        "forward_guidance_aligned_fraction_backward_mean": _row_mean(
            "forward_guidance_aligned_fraction_backward_mean"
        ),
        "forward_guidance_aligned_fraction_common_mean": _row_mean(
            "forward_guidance_aligned_fraction_common_mean"
        ),
        "forward_guidance_middle_row_count_mean": _row_mean("forward_guidance_middle_row_count_mean"),
        "forward_guidance_overlap_row_count_mean": _row_mean("forward_guidance_overlap_row_count_mean"),
        "forward_guidance_zero_support_row_count_mean": _row_mean(
            "forward_guidance_zero_support_row_count_mean"
        ),
        "forward_guidance_middle_row_fraction_common_mean": _row_mean(
            "forward_guidance_middle_row_fraction_common_mean"
        ),
        "forward_guidance_overlap_row_fraction_common_mean": _row_mean(
            "forward_guidance_overlap_row_fraction_common_mean"
        ),
        "forward_guidance_projected_state_count_mean": _row_mean(
            "forward_guidance_projected_state_count_mean"
        ),
        "forward_guidance_projected_entropy_mean": _row_mean(
            "forward_guidance_projected_entropy_mean"
        ),
        "forward_guidance_projected_effective_support_mean": _row_mean(
            "forward_guidance_projected_effective_support_mean"
        ),
        "forward_guidance_projected_top_logprob_mean": _row_mean(
            "forward_guidance_projected_top_logprob_mean"
        ),
        "forward_guidance_projected_logprob_gap_mean": _row_mean(
            "forward_guidance_projected_logprob_gap_mean"
        ),
        "forward_guidance_candidate_state_count_total": _row_sum(
            "forward_guidance_candidate_state_count_total"
        ),
        "forward_guidance_applied_state_count_total": _row_sum(
            "forward_guidance_applied_state_count_total"
        ),
        "forward_guidance_missing_mass_count_total": _row_sum(
            "forward_guidance_missing_mass_count_total"
        ),
        "forward_guidance_clipped_state_count_total": _row_sum(
            "forward_guidance_clipped_state_count_total"
        ),
        "forward_guidance_missing_mass_fraction_mean": _row_mean(
            "forward_guidance_missing_mass_fraction"
        ),
        "forward_guidance_clipped_fraction_mean": _row_mean("forward_guidance_clipped_fraction"),
        "forward_guidance_bonus_min_min": _row_min("forward_guidance_bonus_min_min"),
        "forward_guidance_bonus_p10_mean": _row_mean("forward_guidance_bonus_p10_mean"),
        "forward_guidance_bonus_p50_mean": _row_mean("forward_guidance_bonus_p50_mean"),
        "forward_guidance_bonus_mean_mean": _row_mean("forward_guidance_bonus_mean_mean"),
        "forward_guidance_bonus_p90_mean": _row_mean("forward_guidance_bonus_p90_mean"),
        "forward_guidance_bonus_max_max": _row_max("forward_guidance_bonus_max_max"),
        "forward_guidance_weighted_bonus_mean_mean": _row_mean(
            "forward_guidance_weighted_bonus_mean_mean"
        ),
        "forward_guidance_guided_top_base_rank_mean": _row_mean(
            "forward_guidance_guided_top_base_rank_mean"
        ),
        "forward_guidance_guided_top_base_rank_p99": _row_quantile(
            "forward_guidance_guided_top_base_rank_p99",
            0.99,
        ),
        "forward_guidance_base_top_guided_rank_mean": _row_mean(
            "forward_guidance_base_top_guided_rank_mean"
        ),
        "forward_guidance_base_top_guided_rank_p99": _row_quantile(
            "forward_guidance_base_top_guided_rank_p99",
            0.99,
        ),
        "forward_guidance_projected_info_bits_mean": _row_mean(
            "forward_guidance_projected_info_bits_mean"
        ),
        "forward_guidance_conditional_shortlist_state_count_mean": _row_mean(
            "forward_guidance_conditional_shortlist_state_count_mean"
        ),
        "forward_guidance_conditional_lookup_radius_mean": _row_mean(
            "forward_guidance_conditional_lookup_radius_mean"
        ),
        "forward_guidance_conditional_finite_score_count_mean": _row_mean(
            "forward_guidance_conditional_finite_score_count_mean"
        ),
        "forward_guidance_conditional_exact_support_count_mean": _row_mean(
            "forward_guidance_conditional_exact_support_count_mean"
        ),
        "forward_guidance_conditional_neighborhood_support_count_mean": _row_mean(
            "forward_guidance_conditional_neighborhood_support_count_mean"
        ),
        "forward_guidance_conditional_neighborhood_only_support_count_mean": _row_mean(
            "forward_guidance_conditional_neighborhood_only_support_count_mean"
        ),
        "forward_guidance_conditional_missing_support_count_mean": _row_mean(
            "forward_guidance_conditional_missing_support_count_mean"
        ),
        "forward_guidance_conditional_positive_raw_info_count_mean": _row_mean(
            "forward_guidance_conditional_positive_raw_info_count_mean"
        ),
        "forward_guidance_conditional_finite_outside_kept_count_mean": _row_mean(
            "forward_guidance_conditional_finite_outside_kept_count_mean"
        ),
        "forward_guidance_conditional_positive_outside_kept_count_mean": _row_mean(
            "forward_guidance_conditional_positive_outside_kept_count_mean"
        ),
        "forward_guidance_conditional_nearcut_outside_kept_count_mean": _row_mean(
            "forward_guidance_conditional_nearcut_outside_kept_count_mean"
        ),
        "forward_guidance_conditional_positive_nearcut_outside_kept_count_mean": _row_mean(
            "forward_guidance_conditional_positive_nearcut_outside_kept_count_mean"
        ),
        "forward_guidance_conditional_missing_logical_class_outside_kept_count_mean": _row_mean(
            "forward_guidance_conditional_missing_logical_class_outside_kept_count_mean"
        ),
        "forward_guidance_conditional_positive_bonus_count_mean": _row_mean(
            "forward_guidance_conditional_positive_bonus_count_mean"
        ),
        "forward_guidance_conditional_promoted_state_count_total": _row_sum(
            "forward_guidance_conditional_promoted_state_count_total"
        ),
        "forward_guidance_conditional_demoted_state_count_total": _row_sum(
            "forward_guidance_conditional_demoted_state_count_total"
        ),
        "forward_guidance_conditional_changed_kept_step_count": _row_sum(
            "forward_guidance_conditional_changed_kept_step_count"
        ),
        "forward_guidance_conditional_changed_kept_fraction_mean": _row_mean(
            "forward_guidance_conditional_changed_kept_fraction"
        ),
        "forward_guidance_conditional_added_logical_class_count_total": _row_sum(
            "forward_guidance_conditional_added_logical_class_count_total"
        ),
        "forward_guidance_conditional_fallback_candidate_count_total": _row_sum(
            "forward_guidance_conditional_fallback_candidate_count_total"
        ),
        "forward_guidance_conditional_fallback_candidate_count_mean": _row_mean(
            "forward_guidance_conditional_fallback_candidate_count_mean"
        ),
        "forward_guidance_conditional_fallback_added_state_count_total": _row_sum(
            "forward_guidance_conditional_fallback_added_state_count_total"
        ),
        "forward_guidance_conditional_fallback_added_logical_class_count_total": _row_sum(
            "forward_guidance_conditional_fallback_added_logical_class_count_total"
        ),
        "forward_guidance_conditional_raw_info_min_min": _row_min(
            "forward_guidance_conditional_raw_info_min_min"
        ),
        "forward_guidance_conditional_raw_info_p10_mean": _row_mean(
            "forward_guidance_conditional_raw_info_p10_mean"
        ),
        "forward_guidance_conditional_raw_info_p50_mean": _row_mean(
            "forward_guidance_conditional_raw_info_p50_mean"
        ),
        "forward_guidance_conditional_raw_info_mean_mean": _row_mean(
            "forward_guidance_conditional_raw_info_mean_mean"
        ),
        "forward_guidance_conditional_raw_info_p90_mean": _row_mean(
            "forward_guidance_conditional_raw_info_p90_mean"
        ),
        "forward_guidance_conditional_raw_info_max_max": _row_max(
            "forward_guidance_conditional_raw_info_max_max"
        ),
        "forward_guidance_conditional_bonus_max_max": _row_max(
            "forward_guidance_conditional_bonus_max_max"
        ),
        "forward_guidance_checkpoint_available_step_count": _row_sum(
            "forward_guidance_checkpoint_available_step_count"
        ),
        "forward_guidance_checkpoint_available_fraction_mean": _row_mean(
            "forward_guidance_checkpoint_available_fraction"
        ),
        "forward_guidance_checkpoint_key_count_mean": _row_mean(
            "forward_guidance_checkpoint_key_count_mean"
        ),
        "forward_guidance_checkpoint_source_state_count_mean": _row_mean(
            "forward_guidance_checkpoint_source_state_count_mean"
        ),
        "forward_guidance_checkpoint_mass_coverage_after_trim_mean": _row_mean(
            "forward_guidance_checkpoint_mass_coverage_after_trim_mean"
        ),
        "forward_guidance_checkpoint_band_state_count_mean": _row_mean(
            "forward_guidance_checkpoint_band_state_count_mean"
        ),
        "forward_guidance_checkpoint_hit_count_mean": _row_mean(
            "forward_guidance_checkpoint_hit_count_mean"
        ),
        "forward_guidance_checkpoint_hit_fraction_mean": _row_mean(
            "forward_guidance_checkpoint_hit_fraction_mean"
        ),
        "forward_guidance_checkpoint_rescue_budget_mean": _row_mean(
            "forward_guidance_checkpoint_rescue_budget_mean"
        ),
        "forward_guidance_checkpoint_rescued_state_count_total": _row_sum(
            "forward_guidance_checkpoint_rescued_state_count_total"
        ),
        "forward_guidance_checkpoint_rescued_state_count_mean": _row_mean(
            "forward_guidance_checkpoint_rescued_state_count_mean"
        ),
        "forward_guidance_checkpoint_replay_triggered_step_count": _row_sum(
            "forward_guidance_checkpoint_replay_triggered_step_count"
        ),
        "forward_guidance_checkpoint_replay_triggered_fraction_mean": _row_mean(
            "forward_guidance_checkpoint_replay_triggered_fraction"
        ),
        "forward_guidance_checkpoint_replay_prior_available_step_count": _row_sum(
            "forward_guidance_checkpoint_replay_prior_available_step_count"
        ),
        "forward_guidance_checkpoint_replay_prior_available_fraction_mean": _row_mean(
            "forward_guidance_checkpoint_replay_prior_available_fraction"
        ),
        "forward_guidance_checkpoint_replay_called_step_count": _row_sum(
            "forward_guidance_checkpoint_replay_called_step_count"
        ),
        "forward_guidance_checkpoint_replay_called_fraction_mean": _row_mean(
            "forward_guidance_checkpoint_replay_called_fraction"
        ),
        "forward_guidance_checkpoint_replay_attempted_step_count": _row_sum(
            "forward_guidance_checkpoint_replay_attempted_step_count"
        ),
        "forward_guidance_checkpoint_replay_attempted_fraction_mean": _row_mean(
            "forward_guidance_checkpoint_replay_attempted_fraction"
        ),
        "forward_guidance_checkpoint_replay_succeeded_step_count": _row_sum(
            "forward_guidance_checkpoint_replay_succeeded_step_count"
        ),
        "forward_guidance_checkpoint_replay_succeeded_fraction_mean": _row_mean(
            "forward_guidance_checkpoint_replay_succeeded_fraction"
        ),
        "forward_guidance_checkpoint_replay_aborted_no_checkpoint_step_count": _row_sum(
            "forward_guidance_checkpoint_replay_aborted_no_checkpoint_step_count"
        ),
        "forward_guidance_checkpoint_replay_aborted_no_checkpoint_fraction_mean": _row_mean(
            "forward_guidance_checkpoint_replay_aborted_no_checkpoint_fraction"
        ),
        "forward_guidance_checkpoint_replay_aborted_window_too_long_step_count": _row_sum(
            "forward_guidance_checkpoint_replay_aborted_window_too_long_step_count"
        ),
        "forward_guidance_checkpoint_replay_aborted_window_too_long_fraction_mean": _row_mean(
            "forward_guidance_checkpoint_replay_aborted_window_too_long_fraction"
        ),
        "forward_guidance_checkpoint_replay_aborted_empty_query_set_step_count": _row_sum(
            "forward_guidance_checkpoint_replay_aborted_empty_query_set_step_count"
        ),
        "forward_guidance_checkpoint_replay_aborted_empty_query_set_fraction_mean": _row_mean(
            "forward_guidance_checkpoint_replay_aborted_empty_query_set_fraction"
        ),
        "forward_guidance_checkpoint_replay_aborted_budget_cap_step_count": _row_sum(
            "forward_guidance_checkpoint_replay_aborted_budget_cap_step_count"
        ),
        "forward_guidance_checkpoint_replay_aborted_budget_cap_fraction_mean": _row_mean(
            "forward_guidance_checkpoint_replay_aborted_budget_cap_fraction"
        ),
        "forward_guidance_checkpoint_replay_completed_step_count": _row_sum(
            "forward_guidance_checkpoint_replay_completed_step_count"
        ),
        "forward_guidance_checkpoint_replay_completed_fraction_mean": _row_mean(
            "forward_guidance_checkpoint_replay_completed_fraction"
        ),
        "forward_guidance_checkpoint_replay_target_before_start_step_count": _row_sum(
            "forward_guidance_checkpoint_replay_target_before_start_step_count"
        ),
        "forward_guidance_checkpoint_replay_target_before_start_fraction_mean": _row_mean(
            "forward_guidance_checkpoint_replay_target_before_start_fraction"
        ),
        "forward_guidance_checkpoint_replay_no_progress_to_next_boundary_step_count": _row_sum(
            "forward_guidance_checkpoint_replay_no_progress_to_next_boundary_step_count"
        ),
        "forward_guidance_checkpoint_replay_no_progress_to_next_boundary_fraction_mean": _row_mean(
            "forward_guidance_checkpoint_replay_no_progress_to_next_boundary_fraction"
        ),
        "forward_guidance_checkpoint_replay_target_not_reached_step_count": _row_sum(
            "forward_guidance_checkpoint_replay_target_not_reached_step_count"
        ),
        "forward_guidance_checkpoint_replay_target_not_reached_fraction_mean": _row_mean(
            "forward_guidance_checkpoint_replay_target_not_reached_fraction"
        ),
        "forward_guidance_checkpoint_replay_target_reached_step_count": _row_sum(
            "forward_guidance_checkpoint_replay_target_reached_step_count"
        ),
        "forward_guidance_checkpoint_replay_target_reached_fraction_mean": _row_mean(
            "forward_guidance_checkpoint_replay_target_reached_fraction"
        ),
        "forward_guidance_checkpoint_replay_final_processed_columns_mean": _row_mean(
            "forward_guidance_checkpoint_replay_final_processed_columns_mean"
        ),
        "forward_guidance_checkpoint_replay_available_snapshot_count_mean": _row_mean(
            "forward_guidance_checkpoint_replay_available_snapshot_count_mean"
        ),
        "forward_guidance_checkpoint_replay_target_snapshot_present_step_count": _row_sum(
            "forward_guidance_checkpoint_replay_target_snapshot_present_step_count"
        ),
        "forward_guidance_checkpoint_replay_target_snapshot_present_fraction_mean": _row_mean(
            "forward_guidance_checkpoint_replay_target_snapshot_present_fraction"
        ),
        "forward_guidance_checkpoint_replay_target_snapshot_state_count_mean": _row_mean(
            "forward_guidance_checkpoint_replay_target_snapshot_state_count_mean"
        ),
        "forward_guidance_checkpoint_replay_seed_key_count_mean": _row_mean(
            "forward_guidance_checkpoint_replay_seed_key_count_mean"
        ),
        "forward_guidance_checkpoint_replay_generated_key_count_mean": _row_mean(
            "forward_guidance_checkpoint_replay_generated_key_count_mean"
        ),
        "forward_guidance_checkpoint_replay_new_key_count_mean": _row_mean(
            "forward_guidance_checkpoint_replay_new_key_count_mean"
        ),
        "forward_guidance_checkpoint_replay_query_key_count_mean": _row_mean(
            "forward_guidance_checkpoint_replay_query_key_count_mean"
        ),
        "forward_guidance_checkpoint_replay_hit_key_count_mean": _row_mean(
            "forward_guidance_checkpoint_replay_hit_key_count_mean"
        ),
        "forward_guidance_checkpoint_replay_hit_candidate_count_mean": _row_mean(
            "forward_guidance_checkpoint_replay_hit_candidate_count_mean"
        ),
        "forward_guidance_checkpoint_query_hit_count_before_replay_mean": _row_mean(
            "forward_guidance_checkpoint_query_hit_count_before_replay_mean"
        ),
        "forward_guidance_checkpoint_query_hit_count_after_replay_mean": _row_mean(
            "forward_guidance_checkpoint_query_hit_count_after_replay_mean"
        ),
        "forward_guidance_checkpoint_query_new_hit_count_from_replay_mean": _row_mean(
            "forward_guidance_checkpoint_query_new_hit_count_from_replay_mean"
        ),
        "forward_guidance_checkpoint_replay_expansion_count_mean": _row_mean(
            "forward_guidance_checkpoint_replay_expansion_count_mean"
        ),
        "forward_guidance_checkpoint_replay_max_frontier_size_mean": _row_mean(
            "forward_guidance_checkpoint_replay_max_frontier_size_mean"
        ),
        "forward_guidance_checkpoint_replay_terminal_state_count_mean": _row_mean(
            "forward_guidance_checkpoint_replay_terminal_state_count_mean"
        ),
        "forward_guidance_checkpoint_replay_replayed_column_count_mean": _row_mean(
            "forward_guidance_checkpoint_replay_replayed_column_count_mean"
        ),
        "forward_guidance_checkpoint_replay_budget_exhausted_step_count": _row_sum(
            "forward_guidance_checkpoint_replay_budget_exhausted_step_count"
        ),
        "forward_guidance_checkpoint_replay_budget_exhausted_fraction_mean": _row_mean(
            "forward_guidance_checkpoint_replay_budget_exhausted_fraction"
        ),
        "forward_guidance_local_widen_eligible_step_count": _row_sum(
            "forward_guidance_local_widen_eligible_step_count"
        ),
        "forward_guidance_local_widen_triggered_step_count": _row_sum(
            "forward_guidance_local_widen_triggered_step_count"
        ),
        "forward_guidance_local_widen_triggered_fraction_mean": _row_mean(
            "forward_guidance_local_widen_triggered_fraction"
        ),
        "forward_guidance_first_trigger_active_processed_columns_min": _row_min_nonnegative(
            "forward_guidance_first_trigger_active_processed_columns"
        ),
        "forward_guidance_first_local_widen_triggered_processed_columns_min": _row_min_nonnegative(
            "forward_guidance_first_local_widen_triggered_processed_columns"
        ),
        "forward_guidance_local_widen_added_state_count_total": _row_sum(
            "forward_guidance_local_widen_added_state_count_total"
        ),
        "forward_guidance_local_widen_kept_count_mean": _row_mean(
            "forward_guidance_local_widen_kept_count_mean"
        ),
        "forward_guidance_truth_cut_state_valid_step_count": _row_sum(
            "forward_guidance_truth_cut_state_valid_step_count"
        ),
        "forward_guidance_truth_cut_candidate_present_step_count": _row_sum(
            "forward_guidance_truth_cut_candidate_present_step_count"
        ),
        "forward_guidance_truth_cut_candidate_present_fraction_mean": _row_mean(
            "forward_guidance_truth_cut_candidate_present_fraction"
        ),
        "forward_guidance_truth_cut_ordinary_kept_step_count": _row_sum(
            "forward_guidance_truth_cut_ordinary_kept_step_count"
        ),
        "forward_guidance_truth_cut_ordinary_kept_fraction_mean": _row_mean(
            "forward_guidance_truth_cut_ordinary_kept_fraction"
        ),
        "forward_guidance_truth_cut_provisional_present_step_count": _row_sum(
            "forward_guidance_truth_cut_provisional_present_step_count"
        ),
        "forward_guidance_truth_cut_provisional_present_fraction_mean": _row_mean(
            "forward_guidance_truth_cut_provisional_present_fraction"
        ),
        "forward_guidance_truth_cut_exact_supported_step_count": _row_sum(
            "forward_guidance_truth_cut_exact_supported_step_count"
        ),
        "forward_guidance_truth_cut_exact_supported_fraction_mean": _row_mean(
            "forward_guidance_truth_cut_exact_supported_fraction"
        ),
        "forward_guidance_truth_cut_checkpoint_hit_before_replay_step_count": _row_sum(
            "forward_guidance_truth_cut_checkpoint_hit_before_replay_step_count"
        ),
        "forward_guidance_truth_cut_checkpoint_hit_before_replay_fraction_mean": _row_mean(
            "forward_guidance_truth_cut_checkpoint_hit_before_replay_fraction"
        ),
        "forward_guidance_truth_cut_checkpoint_replay_queried_step_count": _row_sum(
            "forward_guidance_truth_cut_checkpoint_replay_queried_step_count"
        ),
        "forward_guidance_truth_cut_checkpoint_replay_queried_fraction_mean": _row_mean(
            "forward_guidance_truth_cut_checkpoint_replay_queried_fraction"
        ),
        "forward_guidance_truth_cut_checkpoint_replay_hit_step_count": _row_sum(
            "forward_guidance_truth_cut_checkpoint_replay_hit_step_count"
        ),
        "forward_guidance_truth_cut_checkpoint_replay_hit_fraction_mean": _row_mean(
            "forward_guidance_truth_cut_checkpoint_replay_hit_fraction"
        ),
        "forward_guidance_truth_cut_prev_checkpoint_exists_step_count": _row_sum(
            "forward_guidance_truth_cut_prev_checkpoint_exists_step_count"
        ),
        "forward_guidance_truth_cut_prev_checkpoint_exists_fraction_mean": _row_mean(
            "forward_guidance_truth_cut_prev_checkpoint_exists_fraction"
        ),
        "forward_guidance_truth_cut_prev_checkpoint_ancestor_present_step_count": _row_sum(
            "forward_guidance_truth_cut_prev_checkpoint_ancestor_present_step_count"
        ),
        "forward_guidance_truth_cut_prev_checkpoint_ancestor_present_fraction_mean": _row_mean(
            "forward_guidance_truth_cut_prev_checkpoint_ancestor_present_fraction"
        ),
        "forward_guidance_truth_cut_neighborhood_supported_step_count": _row_sum(
            "forward_guidance_truth_cut_neighborhood_supported_step_count"
        ),
        "forward_guidance_truth_cut_neighborhood_supported_fraction_mean": _row_mean(
            "forward_guidance_truth_cut_neighborhood_supported_fraction"
        ),
        "forward_guidance_truth_cut_conditional_supported_step_count": _row_sum(
            "forward_guidance_truth_cut_conditional_supported_step_count"
        ),
        "forward_guidance_truth_cut_conditional_supported_fraction_mean": _row_mean(
            "forward_guidance_truth_cut_conditional_supported_fraction"
        ),
        "forward_guidance_truth_cut_conditional_positive_step_count": _row_sum(
            "forward_guidance_truth_cut_conditional_positive_step_count"
        ),
        "forward_guidance_truth_cut_conditional_positive_fraction_mean": _row_mean(
            "forward_guidance_truth_cut_conditional_positive_fraction"
        ),
        "forward_guidance_truth_cut_added_extra_step_count": _row_sum(
            "forward_guidance_truth_cut_added_extra_step_count"
        ),
        "forward_guidance_truth_cut_added_extra_fraction_mean": _row_mean(
            "forward_guidance_truth_cut_added_extra_fraction"
        ),
        "forward_guidance_truth_cut_final_kept_step_count": _row_sum(
            "forward_guidance_truth_cut_final_kept_step_count"
        ),
        "forward_guidance_truth_cut_final_kept_fraction_mean": _row_mean(
            "forward_guidance_truth_cut_final_kept_fraction"
        ),
        "forward_guidance_truth_cut_first_candidate_missing_processed_columns_min": _row_min_nonnegative(
            "forward_guidance_truth_cut_first_candidate_missing_processed_columns"
        ),
        "forward_guidance_truth_cut_first_ordinary_missing_processed_columns_min": _row_min_nonnegative(
            "forward_guidance_truth_cut_first_ordinary_missing_processed_columns"
        ),
        "forward_guidance_truth_cut_first_provisional_missing_processed_columns_min": _row_min_nonnegative(
            "forward_guidance_truth_cut_first_provisional_missing_processed_columns"
        ),
        "forward_guidance_truth_cut_first_final_missing_processed_columns_min": _row_min_nonnegative(
            "forward_guidance_truth_cut_first_final_missing_processed_columns"
        ),
        "forward_guidance_truth_cut_first_added_extra_processed_columns_min": _row_min_nonnegative(
            "forward_guidance_truth_cut_first_added_extra_processed_columns"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_trigger_active_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_trigger_active"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_local_widen_triggered_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_local_widen_triggered"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_provisional_present_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_provisional_present"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_exact_supported_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_exact_supported"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_neighborhood_supported_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_neighborhood_supported"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_conditional_supported_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_conditional_supported"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_hit_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_hit"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_hit_before_replay_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_hit_before_replay"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_queried_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_queried"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_hit_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_hit"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_prev_checkpoint_exists_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_prev_checkpoint_exists"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_prev_checkpoint_ancestor_present_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_prev_checkpoint_ancestor_present"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_base_rank_mean": _row_mean_nonnegative(
            "forward_guidance_truth_cut_first_ordinary_loss_base_rank"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_rank_over_beam_size_mean": _row_mean_nonnegative(
            "forward_guidance_truth_cut_first_ordinary_loss_rank_over_beam_size"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_rank_over_ordinary_kept_mean": _row_mean_nonnegative(
            "forward_guidance_truth_cut_first_ordinary_loss_rank_over_ordinary_kept"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_within_2k_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_within_2k"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_within_3k_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_within_3k"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_within_4k_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_within_4k"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_within_2x_ordinary_kept_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_within_2x_ordinary_kept"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_within_3x_ordinary_kept_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_within_3x_ordinary_kept"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_within_4x_ordinary_kept_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_within_4x_ordinary_kept"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_added_extra_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_added_extra"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_available_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_available"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_key_count_mean": _row_mean_nonnegative(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_key_count"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_source_state_count_mean": _row_mean_nonnegative(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_source_state_count"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_mass_coverage_after_trim_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_mass_coverage_after_trim"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_band_state_count_mean": _row_mean_nonnegative(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_band_state_count"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_hit_count_mean": _row_mean_nonnegative(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_hit_count"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_hit_fraction_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_hit_fraction"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_rescue_budget_mean": _row_mean_nonnegative(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_rescue_budget"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_rescued_state_count_mean": _row_mean_nonnegative(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_rescued_state_count"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_triggered_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_triggered"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_prior_available_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_prior_available"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_called_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_called"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_attempted_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_attempted"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_succeeded_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_succeeded"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_aborted_no_checkpoint_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_aborted_no_checkpoint"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_aborted_window_too_long_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_aborted_window_too_long"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_aborted_empty_query_set_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_aborted_empty_query_set"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_aborted_budget_cap_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_aborted_budget_cap"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_completed_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_completed"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_target_before_start_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_target_before_start"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_no_progress_to_next_boundary_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_no_progress_to_next_boundary"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_target_not_reached_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_target_not_reached"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_target_reached_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_target_reached"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_final_processed_columns_mean": _row_mean_nonnegative(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_final_processed_columns"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_available_snapshot_count_mean": _row_mean_nonnegative(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_available_snapshot_count"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_target_snapshot_present_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_target_snapshot_present"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_target_snapshot_state_count_mean": _row_mean_nonnegative(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_target_snapshot_state_count"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_seed_key_count_mean": _row_mean_nonnegative(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_seed_key_count"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_generated_key_count_mean": _row_mean_nonnegative(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_generated_key_count"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_new_key_count_mean": _row_mean_nonnegative(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_new_key_count"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_query_key_count_mean": _row_mean_nonnegative(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_query_key_count"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_hit_key_count_mean": _row_mean_nonnegative(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_hit_key_count"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_hit_candidate_count_mean": _row_mean_nonnegative(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_hit_candidate_count"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_query_hit_count_before_replay_mean": _row_mean_nonnegative(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_query_hit_count_before_replay"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_query_hit_count_after_replay_mean": _row_mean_nonnegative(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_query_hit_count_after_replay"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_query_new_hit_count_from_replay_mean": _row_mean_nonnegative(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_query_new_hit_count_from_replay"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_expansion_count_mean": _row_mean_nonnegative(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_expansion_count"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_max_frontier_size_mean": _row_mean_nonnegative(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_max_frontier_size"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_terminal_state_count_mean": _row_mean_nonnegative(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_terminal_state_count"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_replayed_column_count_mean": _row_mean_nonnegative(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_replayed_column_count"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_budget_exhausted_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_budget_exhausted"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_true_join_rank_in_band_mean": _row_mean_nonnegative(
            "forward_guidance_truth_cut_first_ordinary_loss_true_join_rank_in_band"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_true_survives_next_prune_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_true_survives_next_prune"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_true_survives_two_prunes_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_true_survives_two_prunes"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_rescued_wrong_class_above_truth_mean": _row_mean_nonnegative(
            "forward_guidance_truth_cut_first_ordinary_loss_rescued_wrong_class_above_truth"
        ),
        "forward_guidance_truth_cut_terminal_truth_present_mean": _row_mean(
            "forward_guidance_truth_cut_terminal_truth_present"
        ),
        "forward_guidance_truth_cut_first_ordinary_loss_truth_present_terminal_mean": _row_mean(
            "forward_guidance_truth_cut_first_ordinary_loss_truth_present_terminal"
        ),
        "forward_guidance_truth_cut_true_join_rank_in_band_mean_mean": _row_mean_nonnegative(
            "forward_guidance_truth_cut_true_join_rank_in_band_mean"
        ),
        "forward_guidance_truth_cut_true_survives_next_prune_fraction_mean": _row_mean(
            "forward_guidance_truth_cut_true_survives_next_prune_fraction"
        ),
        "forward_guidance_truth_cut_true_survives_next_prune_given_rescued_fraction_mean": _row_mean(
            "forward_guidance_truth_cut_true_survives_next_prune_given_rescued_fraction"
        ),
        "forward_guidance_truth_cut_true_survives_two_prunes_fraction_mean": _row_mean(
            "forward_guidance_truth_cut_true_survives_two_prunes_fraction"
        ),
        "forward_guidance_truth_cut_true_survives_two_prunes_given_rescued_fraction_mean": _row_mean(
            "forward_guidance_truth_cut_true_survives_two_prunes_given_rescued_fraction"
        ),
        "forward_guidance_truth_cut_rescued_wrong_class_above_truth_mean_mean": _row_mean_nonnegative(
            "forward_guidance_truth_cut_rescued_wrong_class_above_truth_mean"
        ),
        "forward_guidance_truth_cut_raw_info_mean": _row_mean(
            "forward_guidance_truth_cut_raw_info_mean"
        ),
        "forward_guidance_truth_cut_raw_info_p50_mean": _row_mean(
            "forward_guidance_truth_cut_raw_info_p50"
        ),
        "track_best_path": bool(track_best_path),
        "lookahead_depth": int(lookahead_depth),
        "lookahead_shortlist_size": int(lookahead_shortlist_size),
        "delayed_pruning_gap_threshold": float(delayed_pruning_gap_threshold),
        "delayed_pruning_factor": int(delayed_pruning_factor),
        "pruning_replay_checkpoint_stride": int(pruning_replay_checkpoint_stride),
        "pruning_replay_horizon": int(pruning_replay_horizon),
        "tail_exact_columns": int(tail_exact_columns),
        "superstep_mode": str(superstep_mode),
        "superstep_path_budget": int(rows[0].get("superstep_path_budget", 0)) if rows else 0,
        "superstep_state_budget": int(rows[0].get("superstep_state_budget", 0)) if rows else 0,
        "superstep_transition_budget": int(rows[0].get("superstep_transition_budget", 0)) if rows else 0,
        "detector_bucket_pruning": bool(detector_bucket_pruning),
        "detector_bucket_max_logicals": int(detector_bucket_max_logicals),
        "logical_class_reserve_min_classes": int(logical_class_reserve_min_classes),
        "logical_class_reserve_max_replacements": int(logical_class_reserve_max_replacements),
        "logical_class_reserve_min_remaining_columns": int(logical_class_reserve_min_remaining_columns),
        "logical_class_reserve_applied_count_mean": float(np.mean([float(row.get("logical_class_reserve_applied_count", 0)) for row in rows])) if rows else float("nan"),
        "logical_class_reserve_replaced_state_count_mean": float(np.mean([float(row.get("logical_class_reserve_replaced_state_count", 0)) for row in rows])) if rows else float("nan"),
        "logical_class_quota_top_classes": int(logical_class_quota_top_classes),
        "logical_class_quota_reserved_slots": int(logical_class_quota_reserved_slots),
        "logical_class_quota_min_remaining_columns": int(logical_class_quota_min_remaining_columns),
        "logical_class_quota_applied_count_mean": float(np.mean([float(row.get("logical_class_quota_applied_count", 0)) for row in rows])) if rows else float("nan"),
        "logical_class_quota_kept_state_count_mean": float(np.mean([float(row.get("logical_class_quota_kept_state_count", 0)) for row in rows])) if rows else float("nan"),
        "lineage_reserve_checkpoint_stride": int(lineage_reserve_checkpoint_stride),
        "lineage_reserve_reserved_slots": int(lineage_reserve_reserved_slots),
        "lineage_reserve_applied_count_mean": float(np.mean([float(row.get("lineage_reserve_applied_count", 0)) for row in rows])) if rows else float("nan"),
        "lineage_reserve_kept_state_count_mean": float(np.mean([float(row.get("lineage_reserve_kept_state_count", 0)) for row in rows])) if rows else float("nan"),
        "logical_rerank_columns": int(logical_rerank_columns),
        "logical_rerank_shortlist_size": int(logical_rerank_shortlist_size),
        "logical_rerank_min_classes": int(logical_rerank_min_classes),
        "logical_rerank_state_budget": int(logical_rerank_state_budget),
        "logical_rerank_transition_budget": int(logical_rerank_transition_budget),
        "logical_rerank_checkpoint_stride": int(logical_rerank_checkpoint_stride),
        "logical_rerank_max_passes": int(logical_rerank_max_passes),
        "logical_rerank_mode": str(logical_rerank_mode),
        "final_logical_select_mode": str(final_logical_select_mode),
        "final_logical_select_rep_cost_weight": float(final_logical_select_rep_cost_weight),
        "final_logical_select_max_log_mass_gap": (
            float(final_logical_select_max_log_mass_gap)
            if math.isfinite(float(final_logical_select_max_log_mass_gap))
            else ""
        ),
        "final_logical_select_rank2_viterbi_tolerance": float(
            final_logical_select_rank2_viterbi_tolerance
        ),
        "final_logical_select_gate_triggered_count": sum(
            int(bool(row.get("final_logical_select_gate_triggered", False))) for row in rows
        ),
        "terminal_top_log_mass_gap_mean": (
            float(np.mean(_finite_row_values(rows, "terminal_top_log_mass_gap")))
            if _finite_row_values(rows, "terminal_top_log_mass_gap")
            else float("nan")
        ),
        "terminal_top_log_mass_gap_p99": _quantile(
            _finite_row_values(rows, "terminal_top_log_mass_gap"),
            0.99,
        ),
        "logical_rerank_pass_count_mean": float(np.mean([float(row.get("logical_rerank_pass_count", 0)) for row in rows])) if rows else float("nan"),
        "delayed_pruning_trigger_count_mean": float(np.mean([float(row.get("delayed_pruning_trigger_count", 0)) for row in rows])) if rows else float("nan"),
        "delayed_pruning_active_prune_count_mean": float(np.mean([float(row.get("delayed_pruning_active_prune_count", 0)) for row in rows])) if rows else float("nan"),
        "delayed_pruning_peak_beam_size_max": int(max(int(row.get("delayed_pruning_peak_beam_size", 0)) for row in rows)) if rows else 0,
        "selective_secondary_trigger_count_mean": float(np.mean([float(row.get("selective_secondary_trigger_count", 0)) for row in rows])) if rows else float("nan"),
        "selective_secondary_changed_count_mean": float(np.mean([float(row.get("selective_secondary_changed_count", 0)) for row in rows])) if rows else float("nan"),
        "selective_secondary_reranked_state_count_mean": float(np.mean([float(row.get("selective_secondary_reranked_state_count", 0)) for row in rows])) if rows else float("nan"),
        "selective_local_lookahead_trigger_count_mean": _row_mean("selective_local_lookahead_trigger_count"),
        "selective_local_lookahead_changed_count_mean": _row_mean("selective_local_lookahead_changed_count"),
        "selective_local_lookahead_candidate_count_mean": _row_mean("selective_local_lookahead_candidate_count"),
        "selective_local_lookahead_extra_work_mean": _row_mean("selective_local_lookahead_extra_work"),
        "pruning_replay_attempt_count_mean": float(np.mean([float(row.get("pruning_replay_attempt_count", 0)) for row in rows])) if rows else float("nan"),
        "pruning_replay_applied_count_mean": float(np.mean([float(row.get("pruning_replay_applied_count", 0)) for row in rows])) if rows else float("nan"),
        "pruning_replay_replaced_state_count_mean": float(np.mean([float(row.get("pruning_replay_replaced_state_count", 0)) for row in rows])) if rows else float("nan"),
        "pruning_replay_replayed_column_count_mean": float(np.mean([float(row.get("pruning_replay_replayed_column_count", 0)) for row in rows])) if rows else float("nan"),
        "pruning_replay_extra_transition_evals_mean": float(np.mean([float(row.get("pruning_replay_extra_transition_evals", 0)) for row in rows])) if rows else float("nan"),
        "pruning_replay_replaced_states_per_apply_mean": (
            float(
                np.mean(
                    [
                        float(row.get("pruning_replay_replaced_state_count", 0.0))
                        / float(row.get("pruning_replay_applied_count", 0.0))
                        for row in rows
                        if float(row.get("pruning_replay_applied_count", 0.0)) > 0.0
                    ]
                )
            )
            if any(float(row.get("pruning_replay_applied_count", 0.0)) > 0.0 for row in rows)
            else float("nan")
        ),
        "splice_enabled_count": sum(int(bool(row.get("splice_enabled", False))) for row in rows),
        "splice_changed_count": sum(int(bool(row.get("splice_changed", False))) for row in rows),
        "splice_fixed_count": sum(int(bool(row.get("splice_fixed", False))) for row in rows),
        "splice_broken_count": sum(int(bool(row.get("splice_broken", False))) for row in rows),
        "splice_unchanged_failure_count": int(splice_unchanged_failure),
        "baseline_fail_total": int(baseline_fail_total),
        "baseline_logical_fail": int(baseline_logical_fail),
        "baseline_syndrome_fail": int(baseline_syndrome_fail),
        "baseline_exception_fail": int(baseline_exception_fail),
        "splice_fail_total": int(splice_fail_total),
        "splice_logical_fail": int(splice_logical_fail),
        "splice_syndrome_fail": int(splice_syndrome_fail),
        "splice_exception_fail": int(splice_exception_fail),
        "splice_truth_present_but_not_selected_count": int(splice_truth_present_but_not_selected),
        "splice_candidate_missing_truth_count": sum(
            int(bool(row.get("splice_candidate_missing_truth", False))) for row in rows
        ),
        "splice_candidate_count_mean": _row_mean("splice_candidate_count"),
        "splice_candidate_logical_class_count_mean": _row_mean(
            "splice_candidate_logical_class_count"
        ),
        "splice_cut_count_mean": _row_mean("splice_cut_count"),
        "splice_finite_cut_fraction_mean": _row_mean("splice_finite_cut_fraction"),
        "splice_missing_support_fraction_mean": _row_mean("splice_missing_support_fraction"),
        "splice_hit_count_mean": _row_mean("splice_hit_count_mean"),
        "noisy_rounds": int(round_count),
        "total_rounds": int(rows[0].get("total_rounds", round_count)) if rows else int(round_count),
        "shots": int(shots),
        "committee_selected_forward_count": sum(
            int(str(row.get("committee_selected_direction", "")) == "forward") for row in rows
        ),
        "committee_selected_backward_count": sum(
            int(str(row.get("committee_selected_direction", "")) == "backward") for row in rows
        ),
        "fail_total": int(fail_total),
        "logical_fail": int(logical_fail),
        "logical_fail_truth_missing_terminal": int(logical_fail_truth_missing_terminal),
        "logical_fail_truth_present_but_not_selected": int(logical_fail_truth_present_but_not_selected),
        "truth_terminal_present_count": int(_row_positive_count("truth_terminal_log_mass_rank")),
        "truth_terminal_log_mass_rank_mean_present": _row_mean_positive(
            "truth_terminal_log_mass_rank"
        ),
        "truth_terminal_best_viterbi_rank_mean_present": _row_mean_positive(
            "truth_terminal_best_viterbi_rank"
        ),
        "truth_terminal_log_mass_rank1_count": int(_row_eq_count("truth_terminal_log_mass_rank", 1)),
        "truth_terminal_best_viterbi_rank1_count": int(
            _row_eq_count("truth_terminal_best_viterbi_rank", 1)
        ),
        "syndrome_fail": int(syndrome_fail),
        "exception_fail": int(exception_fail),
        "fer": float(fer),
        "fer_per_round": float(_frame_fer_to_per_round_exact(float(fer), int(round_count))) if shots else float("nan"),
        "discard_step_count_mean": float(np.mean([float(row.get("discard_step_count", 0)) for row in rows])) if rows else float("nan"),
        "cumulative_discarded_prefix_mass_mean": float(
            np.mean([float(row.get("cumulative_discarded_prefix_mass", 0.0)) for row in rows])
        )
        if rows
        else float("nan"),
        "cumulative_discarded_prefix_mass_p99": _quantile(
            [float(row.get("cumulative_discarded_prefix_mass", 0.0)) for row in rows],
            0.99,
        ),
        "max_discarded_prefix_mass_mean": float(
            np.mean([float(row.get("max_discarded_prefix_mass", 0.0)) for row in rows])
        )
        if rows
        else float("nan"),
        "max_discarded_prefix_mass_p99": _quantile(
            [float(row.get("max_discarded_prefix_mass", 0.0)) for row in rows],
            0.99,
        ),
        "mean_discarded_prefix_mass_mean": float(
            np.mean([float(row.get("mean_discarded_prefix_mass", 0.0)) for row in rows])
        )
        if rows
        else float("nan"),
        "mean_discarded_prefix_mass_p99": _quantile(
            [float(row.get("mean_discarded_prefix_mass", 0.0)) for row in rows],
            0.99,
        ),
        "max_discarded_prefix_fraction_mean": float(
            np.mean([float(row.get("max_discarded_prefix_fraction", 0.0)) for row in rows])
        )
        if rows
        else float("nan"),
        "max_discarded_prefix_fraction_p99": _quantile(
            [float(row.get("max_discarded_prefix_fraction", 0.0)) for row in rows],
            0.99,
        ),
        "mean_discarded_prefix_fraction_mean": float(
            np.mean([float(row.get("mean_discarded_prefix_fraction", 0.0)) for row in rows])
        )
        if rows
        else float("nan"),
        "mean_discarded_prefix_fraction_p99": _quantile(
            [float(row.get("mean_discarded_prefix_fraction", 0.0)) for row in rows],
            0.99,
        ),
        "truth_logical_discard_step_count_mean": float(
            np.mean([float(row.get("truth_logical_discard_step_count", 0)) for row in rows])
        )
        if rows
        else float("nan"),
        "cumulative_truth_logical_discarded_prefix_mass_mean": float(
            np.mean([float(row.get("cumulative_truth_logical_discarded_prefix_mass", 0.0)) for row in rows])
        )
        if rows
        else float("nan"),
        "cumulative_truth_logical_discarded_prefix_mass_p99": _quantile(
            [float(row.get("cumulative_truth_logical_discarded_prefix_mass", 0.0)) for row in rows],
            0.99,
        ),
        "max_truth_logical_discarded_prefix_mass_mean": float(
            np.mean([float(row.get("max_truth_logical_discarded_prefix_mass", 0.0)) for row in rows])
        )
        if rows
        else float("nan"),
        "max_truth_logical_discarded_prefix_mass_p99": _quantile(
            [float(row.get("max_truth_logical_discarded_prefix_mass", 0.0)) for row in rows],
            0.99,
        ),
        "mean_truth_logical_discarded_prefix_mass_mean": float(
            np.mean([float(row.get("mean_truth_logical_discarded_prefix_mass", 0.0)) for row in rows])
        )
        if rows
        else float("nan"),
        "mean_truth_logical_discarded_prefix_mass_p99": _quantile(
            [float(row.get("mean_truth_logical_discarded_prefix_mass", 0.0)) for row in rows],
            0.99,
        ),
        "max_truth_logical_discarded_prefix_fraction_mean": float(
            np.mean([float(row.get("max_truth_logical_discarded_prefix_fraction", 0.0)) for row in rows])
        )
        if rows
        else float("nan"),
        "max_truth_logical_discarded_prefix_fraction_p99": _quantile(
            [float(row.get("max_truth_logical_discarded_prefix_fraction", 0.0)) for row in rows],
            0.99,
        ),
        "mean_truth_logical_discarded_prefix_fraction_mean": float(
            np.mean([float(row.get("mean_truth_logical_discarded_prefix_fraction", 0.0)) for row in rows])
        )
        if rows
        else float("nan"),
        "mean_truth_logical_discarded_prefix_fraction_p99": _quantile(
            [float(row.get("mean_truth_logical_discarded_prefix_fraction", 0.0)) for row in rows],
            0.99,
        ),
        "decode_s_mean": float(np.mean([float(row["decode_s"]) for row in rows])) if rows else float("nan"),
        "decode_s_p99": _quantile([float(row["decode_s"]) for row in rows], 0.99),
        "transition_evals_mean": float(np.mean([float(row["transition_evals"]) for row in rows])) if rows else float("nan"),
        "transition_evals_p99": _quantile([float(row["transition_evals"]) for row in rows], 0.99),
        "lookahead_transition_evals_mean": float(np.mean([float(row["lookahead_transition_evals"]) for row in rows])) if rows else float("nan"),
        "lookahead_transition_evals_p99": _quantile([float(row["lookahead_transition_evals"]) for row in rows], 0.99),
        "transition_evals_total_mean": float(np.mean([float(row["transition_evals_total"]) for row in rows])) if rows else float("nan"),
        "transition_evals_total_p99": _quantile([float(row["transition_evals_total"]) for row in rows], 0.99),
        "transition_evals_physical_total_mean": float(np.mean([float(row.get("transition_evals_physical_total", 0.0)) for row in rows])) if rows else float("nan"),
        "transition_evals_physical_total_p99": _quantile([float(row.get("transition_evals_physical_total", 0.0)) for row in rows], 0.99),
        "us_per_transition_mean": (
            float(1.0e6 * total_decode_s / total_transition_evals) if rows and float(total_transition_evals) > 0.0 else float("nan")
        ),
        "us_per_transition_physical_mean": (
            float(1.0e6 * total_decode_s / total_transition_evals_physical)
            if rows and float(total_transition_evals_physical) > 0.0
            else float("nan")
        ),
        "us_per_column_mean": (
            float(1.0e6 * total_decode_s / float(shots * matrix_cols)) if rows and int(shots) > 0 and int(matrix_cols) > 0 else float("nan")
        ),
        "mean_states_mean": float(np.mean([float(row["mean_states"]) for row in rows])) if rows else float("nan"),
        "mean_states_p99": _quantile([float(row["mean_states"]) for row in rows], 0.99),
        "merge_events_total_mean": float(np.mean([float(row["merge_events_total"]) for row in rows])) if rows else float("nan"),
        "merge_events_per_column_mean": float(np.mean([float(row["merge_events_per_column"]) for row in rows])) if rows else float("nan"),
        "closure_rejects_total_mean": float(np.mean([float(row["closure_rejects_total"]) for row in rows])) if rows else float("nan"),
        "closure_rejects_per_column_mean": float(np.mean([float(row["closure_rejects_per_column"]) for row in rows])) if rows else float("nan"),
        "top_log_mass_incoming_per_column_mean": float(np.mean([float(row["top_log_mass_incoming_per_column"]) for row in rows])) if rows else float("nan"),
        "top_log_mass_merge_per_column_mean": float(np.mean([float(row["top_log_mass_merge_per_column"]) for row in rows])) if rows else float("nan"),
        "top_viterbi_incoming_per_column_mean": float(np.mean([float(row["top_viterbi_incoming_per_column"]) for row in rows])) if rows else float("nan"),
        "top_viterbi_merge_per_column_mean": float(np.mean([float(row["top_viterbi_merge_per_column"]) for row in rows])) if rows else float("nan"),
        "winner_path_incoming_per_column_mean": (
            float(np.mean(_finite_row_values(rows, "winner_path_incoming_per_column")))
            if _finite_row_values(rows, "winner_path_incoming_per_column")
            else float("nan")
        ),
        "winner_path_merge_per_column_mean": (
            float(np.mean(_finite_row_values(rows, "winner_path_merge_per_column")))
            if _finite_row_values(rows, "winner_path_merge_per_column")
            else float("nan")
        ),
        "max_states_seen": int(max(int(row["max_states_seen"]) for row in rows)) if rows else 0,
        "matrix_rows": int(rows[0]["matrix_rows"]) if rows else 0,
        "matrix_cols": int(matrix_cols),
        "logical_rows": int(rows[0]["logical_rows"]) if rows else 0,
        "edge_count": int(rows[0]["edge_count"]) if rows else 0,
        "frontier_max_active_detectors": int(rows[0]["frontier_max_active_detectors"]) if rows else 0,
    }


def _build_summary_rows(per_shot_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in per_shot_rows:
        key = (
            str(row["decoder"]),
            str(row["family"]),
            str(row.get("decoder_mode", "forward")),
            str(row.get("backward_column_order", "")),
            str(row.get("correction_state_mode", "none")),
            int(row.get("correction_state_bits", 0)),
            str(row.get("state_merge_mode", "exact")),
            int(row["beam_size"]),
            str(row["score_mode"]),
            (
                None
                if row.get("beam_score_gap_threshold", "") in {"", None}
                else float(row["beam_score_gap_threshold"])
            ),
            str(row.get("selective_secondary_score_mode", "")),
            (
                None
                if row.get("selective_secondary_trigger_gap", "") in {"", None}
                else float(row["selective_secondary_trigger_gap"])
            ),
            (
                0
                if row.get("selective_secondary_band_size", "") in {"", None}
                else int(row["selective_secondary_band_size"])
            ),
            bool(row.get("track_best_path", False)),
            int(row["lookahead_depth"]),
            int(row["lookahead_shortlist_size"]),
            float(row.get("delayed_pruning_gap_threshold", 0.0)),
            int(row.get("delayed_pruning_factor", 1)),
            int(row.get("pruning_replay_checkpoint_stride", 0)),
            int(row.get("pruning_replay_horizon", 0)),
            int(row.get("tail_exact_columns", 0)),
            str(row.get("superstep_mode", "none")),
            bool(row.get("detector_bucket_pruning", False)),
            int(row.get("detector_bucket_max_logicals", 0)),
            int(row.get("logical_class_reserve_min_classes", 0)),
            int(row.get("logical_class_reserve_max_replacements", 0)),
            int(row.get("logical_class_reserve_min_remaining_columns", 0)),
            int(row.get("logical_class_quota_top_classes", 0)),
            int(row.get("logical_class_quota_reserved_slots", 0)),
            int(row.get("logical_class_quota_min_remaining_columns", 0)),
            int(row.get("lineage_reserve_checkpoint_stride", 0)),
            int(row.get("lineage_reserve_reserved_slots", 0)),
            int(row.get("logical_rerank_columns", 0)),
            int(row.get("logical_rerank_shortlist_size", 0)),
            int(row.get("logical_rerank_min_classes", 0)),
            int(row.get("logical_rerank_state_budget", 0)),
            int(row.get("logical_rerank_transition_budget", 0)),
            int(row.get("logical_rerank_checkpoint_stride", 0)),
            int(row.get("logical_rerank_max_passes", 1)),
            str(row.get("logical_rerank_mode", "exact_tail")),
            str(row.get("final_logical_select_mode", "log_mass")),
            float(row.get("final_logical_select_rep_cost_weight", 0.0)),
            (
                float("inf")
                if row.get("final_logical_select_max_log_mass_gap", "") in {"", None}
                else float(row["final_logical_select_max_log_mass_gap"])
            ),
            float(row.get("final_logical_select_rank2_viterbi_tolerance", 0.0) or 0.0),
        )
        grouped.setdefault(key, []).append(dict(row))
    out: list[dict[str, object]] = []
    for decoder, family, decoder_mode, backward_column_order, correction_state_mode, correction_state_bits, state_merge_mode, beam_size, score_mode, beam_score_gap_threshold, selective_secondary_score_mode, selective_secondary_trigger_gap, selective_secondary_band_size, track_best_path, lookahead_depth, lookahead_shortlist_size, delayed_pruning_gap_threshold, delayed_pruning_factor, pruning_replay_checkpoint_stride, pruning_replay_horizon, tail_exact_columns, superstep_mode, detector_bucket_pruning, detector_bucket_max_logicals, logical_class_reserve_min_classes, logical_class_reserve_max_replacements, logical_class_reserve_min_remaining_columns, logical_class_quota_top_classes, logical_class_quota_reserved_slots, logical_class_quota_min_remaining_columns, lineage_reserve_checkpoint_stride, lineage_reserve_reserved_slots, logical_rerank_columns, logical_rerank_shortlist_size, logical_rerank_min_classes, logical_rerank_state_budget, logical_rerank_transition_budget, logical_rerank_checkpoint_stride, logical_rerank_max_passes, logical_rerank_mode, final_logical_select_mode, final_logical_select_rep_cost_weight, final_logical_select_max_log_mass_gap, final_logical_select_rank2_viterbi_tolerance in sorted(grouped, key=lambda item: item):
        rows = sorted(
            grouped[(decoder, family, decoder_mode, backward_column_order, correction_state_mode, correction_state_bits, state_merge_mode, beam_size, score_mode, beam_score_gap_threshold, selective_secondary_score_mode, selective_secondary_trigger_gap, selective_secondary_band_size, track_best_path, lookahead_depth, lookahead_shortlist_size, delayed_pruning_gap_threshold, delayed_pruning_factor, pruning_replay_checkpoint_stride, pruning_replay_horizon, tail_exact_columns, superstep_mode, detector_bucket_pruning, detector_bucket_max_logicals, logical_class_reserve_min_classes, logical_class_reserve_max_replacements, logical_class_reserve_min_remaining_columns, logical_class_quota_top_classes, logical_class_quota_reserved_slots, logical_class_quota_min_remaining_columns, lineage_reserve_checkpoint_stride, lineage_reserve_reserved_slots, logical_rerank_columns, logical_rerank_shortlist_size, logical_rerank_min_classes, logical_rerank_state_budget, logical_rerank_transition_budget, logical_rerank_checkpoint_stride, logical_rerank_max_passes, logical_rerank_mode, final_logical_select_mode, final_logical_select_rep_cost_weight, final_logical_select_max_log_mass_gap, final_logical_select_rank2_viterbi_tolerance)],
            key=lambda row: int(row["shot"]),
        )
        out.append(
            _summary_row(
                rows=rows,
                decoder=str(decoder),
                family=str(family),
                decoder_mode=str(decoder_mode),
                backward_column_order=str(backward_column_order),
                correction_state_mode=str(correction_state_mode),
                correction_state_bits=int(correction_state_bits),
                state_merge_mode=str(state_merge_mode),
                beam_size=int(beam_size),
                score_mode=str(score_mode),
                beam_score_gap_threshold=beam_score_gap_threshold,
                selective_secondary_score_mode=str(selective_secondary_score_mode),
                selective_secondary_trigger_gap=(
                    0.0 if selective_secondary_trigger_gap is None else float(selective_secondary_trigger_gap)
                ),
                selective_secondary_band_size=int(selective_secondary_band_size),
                track_best_path=bool(track_best_path),
                lookahead_depth=int(lookahead_depth),
                lookahead_shortlist_size=int(lookahead_shortlist_size),
                delayed_pruning_gap_threshold=float(delayed_pruning_gap_threshold),
                delayed_pruning_factor=int(delayed_pruning_factor),
                pruning_replay_checkpoint_stride=int(pruning_replay_checkpoint_stride),
                pruning_replay_horizon=int(pruning_replay_horizon),
                tail_exact_columns=int(tail_exact_columns),
                superstep_mode=str(superstep_mode),
                detector_bucket_pruning=bool(detector_bucket_pruning),
                detector_bucket_max_logicals=int(detector_bucket_max_logicals),
                logical_class_reserve_min_classes=int(logical_class_reserve_min_classes),
                logical_class_reserve_max_replacements=int(logical_class_reserve_max_replacements),
                logical_class_reserve_min_remaining_columns=int(logical_class_reserve_min_remaining_columns),
                logical_class_quota_top_classes=int(logical_class_quota_top_classes),
                logical_class_quota_reserved_slots=int(logical_class_quota_reserved_slots),
                logical_class_quota_min_remaining_columns=int(logical_class_quota_min_remaining_columns),
                lineage_reserve_checkpoint_stride=int(lineage_reserve_checkpoint_stride),
                lineage_reserve_reserved_slots=int(lineage_reserve_reserved_slots),
                logical_rerank_columns=int(logical_rerank_columns),
                logical_rerank_shortlist_size=int(logical_rerank_shortlist_size),
                logical_rerank_min_classes=int(logical_rerank_min_classes),
                logical_rerank_state_budget=int(logical_rerank_state_budget),
                logical_rerank_transition_budget=int(logical_rerank_transition_budget),
                logical_rerank_checkpoint_stride=int(logical_rerank_checkpoint_stride),
                logical_rerank_max_passes=int(logical_rerank_max_passes),
                logical_rerank_mode=str(logical_rerank_mode),
                final_logical_select_mode=str(final_logical_select_mode),
                final_logical_select_rep_cost_weight=float(final_logical_select_rep_cost_weight),
                final_logical_select_max_log_mass_gap=(
                    float(final_logical_select_max_log_mass_gap)
                ),
                final_logical_select_rank2_viterbi_tolerance=float(
                    final_logical_select_rank2_viterbi_tolerance
                ),
            )
        )
    return out


def _frontier_rows(family: LoadedProgressiveFamily) -> list[dict[str, object]]:
    return [
        {
            "family": str(family.family_key),
            "model_label": str(family.model_label),
            "matrix_rows": int(family.matrix_rows),
            "matrix_cols": int(family.matrix_cols),
            "logical_rows": int(family.logical_rows),
            "edge_count": int(family.edge_count),
            "frontier_max_active_detectors": int(family.layout.max_active_detectors),
            "correction_state_mode": str(family.correction_state_mode),
            "correction_state_bits": int(family.correction_state_bits),
        }
    ]


FRONTIER_PRESSURE_FIELDNAMES = (
    "backend",
    "scope",
    "shot",
    "K",
    "beam_cap",
    "Delta",
    "score_mode",
    "column_order",
    "boundary_index",
    "pre_prune_candidate_count",
    "kept_count",
    "cap_pressure",
    "best_log_weight_pre",
    "best_log_weight_kept",
    "candidate_N_tau_0p5",
    "candidate_N_tau_1",
    "candidate_N_tau_2",
    "candidate_N_tau_3",
    "candidate_N_tau_5",
    "candidate_N_tau_8",
    "kept_N_tau_0p5",
    "kept_N_tau_1",
    "kept_N_tau_2",
    "kept_N_tau_3",
    "kept_N_tau_5",
    "kept_N_tau_8",
    "candidate_effective_support",
    "kept_effective_support",
    "pruned_log_mass",
    "kept_log_mass",
    "pruned_mass_fraction",
    "terminal_class_count_so_far",
    "first_loss_boundary",
    "truth_present_before_prune",
    "truth_present_after_prune",
    "truth_disappeared_here",
)


def _tau_field_suffix(tau: float) -> str:
    text = str(float(tau)).replace(".", "p")
    return text[:-2] if text.endswith("p0") else text


def _frontier_pressure_rows_from_result(
    *,
    result: progressive.ProgressiveDecodeResult,
    backend: str,
    scope: str,
    shot: int,
    beam_size: int,
    delta: float | None,
    score_mode: str,
    column_order: str,
    diagnostic_truth_logical_mask: int | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    first_loss_boundary: int | None = None
    for snapshot in tuple(result.pre_prune_score_trace):
        candidate_scores = tuple(float(state.rank_primary) for state in tuple(snapshot.states))
        kept_scores = tuple(float(state.rank_primary) for state in tuple(snapshot.states) if bool(state.kept))
        candidate_prefix = tuple(float(state.prefix_log_mass) for state in tuple(snapshot.states))
        kept_prefix = tuple(float(state.prefix_log_mass) for state in tuple(snapshot.states) if bool(state.kept))
        pruned_prefix = tuple(float(state.prefix_log_mass) for state in tuple(snapshot.states) if not bool(state.kept))
        kept_log_mass = progressive.frontier_pressure_log_mass(kept_prefix)
        pruned_log_mass = progressive.frontier_pressure_log_mass(pruned_prefix)
        candidate_log_mass = progressive.frontier_pressure_log_mass(candidate_prefix)
        pruned_fraction = (
            float(math.exp(float(pruned_log_mass) - float(candidate_log_mass)))
            if math.isfinite(float(pruned_log_mass)) and math.isfinite(float(candidate_log_mass))
            else 0.0
        )
        truth_present_before: object = ""
        truth_present_after: object = ""
        truth_disappeared_here: object = ""
        first_loss_value: object = ""
        if diagnostic_truth_logical_mask is not None:
            truth_present_before_bool = any(
                int(state.logical_mask) == int(diagnostic_truth_logical_mask)
                for state in tuple(snapshot.states)
            )
            truth_present_after_bool = any(
                int(state.logical_mask) == int(diagnostic_truth_logical_mask) and bool(state.kept)
                for state in tuple(snapshot.states)
            )
            disappeared = bool(truth_present_before_bool and not truth_present_after_bool)
            if disappeared and first_loss_boundary is None:
                first_loss_boundary = int(snapshot.boundary_column_index)
            truth_present_before = bool(truth_present_before_bool)
            truth_present_after = bool(truth_present_after_bool)
            truth_disappeared_here = bool(disappeared)
            first_loss_value = "" if first_loss_boundary is None else int(first_loss_boundary)
        row: dict[str, object] = {
            "backend": str(backend),
            "scope": str(scope),
            "shot": int(shot),
            "K": int(beam_size),
            "beam_cap": int(beam_size),
            "Delta": "" if delta is None else float(delta),
            "score_mode": str(score_mode),
            "column_order": str(column_order),
            "boundary_index": int(snapshot.boundary_column_index),
            "pre_prune_candidate_count": int(snapshot.candidate_state_count),
            "kept_count": int(snapshot.kept_state_count),
            "cap_pressure": float(snapshot.candidate_state_count) / float(max(1, int(beam_size))),
            "best_log_weight_pre": float(max(candidate_scores)) if candidate_scores else float("-inf"),
            "best_log_weight_kept": float(max(kept_scores)) if kept_scores else float("-inf"),
            "candidate_effective_support": progressive.frontier_pressure_effective_support(candidate_scores),
            "kept_effective_support": progressive.frontier_pressure_effective_support(kept_scores),
            "pruned_log_mass": "" if not math.isfinite(float(pruned_log_mass)) else float(pruned_log_mass),
            "kept_log_mass": "" if not math.isfinite(float(kept_log_mass)) else float(kept_log_mass),
            "pruned_mass_fraction": float(pruned_fraction),
            "terminal_class_count_so_far": "",
            "first_loss_boundary": first_loss_value,
            "truth_present_before_prune": truth_present_before,
            "truth_present_after_prune": truth_present_after,
            "truth_disappeared_here": truth_disappeared_here,
        }
        for tau in progressive.FRONTIER_PRESSURE_TAUS:
            suffix = _tau_field_suffix(float(tau))
            row[f"candidate_N_tau_{suffix}"] = progressive.frontier_pressure_n_tau(
                candidate_scores,
                float(tau),
            )
            row[f"kept_N_tau_{suffix}"] = progressive.frontier_pressure_n_tau(
                kept_scores,
                float(tau),
            )
        rows.append(row)
    return rows


def _write_frontier_pressure_trace_artifacts(
    *,
    trace_rows: Sequence[dict[str, object]],
    out_dir: Path,
) -> None:
    if not trace_rows:
        return
    _write_csv(out_dir / "frontier_pressure_trace.csv", trace_rows, FRONTIER_PRESSURE_FIELDNAMES)
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in trace_rows:
        key = (
            str(row.get("backend", "")),
            str(row.get("scope", "")),
            int(row.get("K", 0)),
            row.get("Delta", ""),
            str(row.get("score_mode", "")),
            str(row.get("column_order", "")),
        )
        grouped.setdefault(key, []).append(dict(row))
    summary_rows: list[dict[str, object]] = []
    for key, rows in sorted(grouped.items(), key=lambda item: item[0]):
        backend, scope, beam_size, delta, score_mode, column_order = key
        cap_values = [float(row.get("cap_pressure", 0.0)) for row in rows]
        n3_values = [float(row.get("candidate_N_tau_3", 0.0)) for row in rows]
        n5_values = [float(row.get("candidate_N_tau_5", 0.0)) for row in rows]
        eff_values = [float(row.get("candidate_effective_support", 0.0)) for row in rows]
        summary_rows.append(
            {
                "backend": str(backend),
                "scope": str(scope),
                "K": int(beam_size),
                "beam_cap": int(beam_size),
                "Delta": delta,
                "score_mode": str(score_mode),
                "column_order": str(column_order),
                "trace_rows": int(len(rows)),
                "shots": int(len({int(row.get("shot", -1)) for row in rows})),
                "cap_pressure_mean": float(np.mean(cap_values)) if cap_values else float("nan"),
                "cap_pressure_p99": _quantile(cap_values, 0.99),
                "candidate_N_tau_3_mean": float(np.mean(n3_values)) if n3_values else float("nan"),
                "candidate_N_tau_5_mean": float(np.mean(n5_values)) if n5_values else float("nan"),
                "candidate_effective_support_mean": float(np.mean(eff_values)) if eff_values else float("nan"),
                "truth_disappeared_count": sum(
                    int(str(row.get("truth_disappeared_here", "")).lower() == "true")
                    for row in rows
                ),
            }
        )
    _write_csv(
        out_dir / "frontier_pressure_summary.csv",
        summary_rows,
        (
            "backend",
            "scope",
            "K",
            "beam_cap",
            "Delta",
            "score_mode",
            "column_order",
            "trace_rows",
            "shots",
            "cap_pressure_mean",
            "cap_pressure_p99",
            "candidate_N_tau_3_mean",
            "candidate_N_tau_5_mean",
            "candidate_effective_support_mean",
            "truth_disappeared_count",
        ),
    )


def _baseline_circuit_name(scope: str) -> str:
    return "gross_mem_x" if str(scope) == "memory_X" else "gross_mem_z"


def _load_overlay_baseline(path: Path, *, p_location: float, label: str, scope: str) -> dict[str, object] | None:
    if not path.exists():
        return None
    for row in _read_csv_rows(path):
        if str(row.get("circuit", "")) != _baseline_circuit_name(str(scope)):
            continue
        if abs(float(row["p_location"]) - float(p_location)) > 1e-15:
            continue
        return {
            "label": str(label),
            "source": str(path),
            "fer": float(row["p_logical"]),
            "fer_upper": float(row["p_logical"]),
            "fer_per_round": float(_frame_fer_to_per_round_exact(float(row["p_logical"]), ROUND_COUNT)),
            "work_mean": float(row["beam_api_iters_mean"]),
            "work_tail": float(row["beam_api_iters_p999"]),
            "work_metric": "beam_api_iters",
        }
    return None


def _load_replay_weak_p1e3_baseline(*, scope: str) -> dict[str, object] | None:
    if not WEAK_REPLAY_P1E3_PATH.exists():
        return None
    for row in _read_csv_rows(WEAK_REPLAY_P1E3_PATH):
        if str(row.get("curve", "")) != str(scope):
            continue
        fer = float(row["fer"])
        fer_upper = float(row["fer_hi95"]) if "fer_hi95" in row else fer
        return {
            "label": "beam weak (500-shot upper bound)",
            "source": str(WEAK_REPLAY_P1E3_PATH),
            "fer": float(fer),
            "fer_upper": float(max(fer, fer_upper)),
            "fer_per_round": float(_frame_fer_to_per_round_exact(float(max(fer, fer_upper)), ROUND_COUNT)),
            "work_mean": float(row["beam_all_iters_mean"]),
            "work_tail": float(row["beam_all_iters_p999"]),
            "work_metric": "beam_all_iters",
        }
    return None


def _beam_baselines(*, p_location: float, scope: str) -> list[dict[str, object]]:
    return _beam_baselines_for_backend(backend="bravyi_depth7", p_location=float(p_location), scope=str(scope))


def _beam_baselines_for_backend(*, backend: str, p_location: float, scope: str) -> list[dict[str, object]]:
    if str(backend) != "bravyi_depth7":
        return []
    out: list[dict[str, object]] = []
    if abs(float(p_location) - 0.001) <= 1e-15:
        weak = _load_replay_weak_p1e3_baseline(scope=str(scope))
        if weak is not None:
            out.append(weak)
        return out
    weak = _load_overlay_baseline(WEAK_OVERLAY_PATH, p_location=float(p_location), label="beam weak", scope=str(scope))
    strong = _load_overlay_baseline(STRONG_OVERLAY_PATH, p_location=float(p_location), label="beam strong", scope=str(scope))
    if weak is not None:
        out.append(weak)
    if strong is not None:
        out.append(strong)
    return out


def _plot_frontier_profile(*, family: LoadedProgressiveFamily, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=200)
    xs = np.arange(len(family.layout.active_width_profile), dtype=np.int32)
    ys = np.asarray(family.layout.active_width_profile, dtype=np.int32)
    ax.plot(
        xs,
        ys,
        linewidth=2.1,
        color="#2b6cb0",
        label=f"binary DEM {family.scope_label} ({family.matrix_rows} x {family.matrix_cols})",
    )
    ax.set_xlabel("processed column prefix length")
    ax.set_ylabel("active detector bits in frontier state")
    ax.set_title(f"{family.benchmark_title} {family.scope_label} Progressive Frontier Width")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _write_state_count_profile_artifacts(
    *,
    per_shot_rows: Sequence[dict[str, object]],
    out_dir: Path,
) -> None:
    profile_rows = [
        row
        for row in per_shot_rows
        if str(row.get("state_count_by_column_profile", "")).strip()
    ]
    if not profile_rows:
        return

    grouped: dict[tuple[object, ...], list[np.ndarray]] = {}
    for row in profile_rows:
        series = _parse_int_series(row.get("state_count_by_column_profile", ""))
        if series.size == 0:
            continue
        key = (
            str(row.get("decoder_mode", "forward")),
            str(row.get("backward_column_order", "")),
            str(row.get("score_mode", "")),
            int(row.get("beam_size", 0)),
            row.get("beam_score_gap_threshold", ""),
            str(row.get("selective_secondary_score_mode", "")),
            row.get("selective_secondary_trigger_gap", ""),
            row.get("selective_secondary_band_size", ""),
            row.get("forward_guidance_trigger_gap", ""),
            row.get("forward_guidance_snapshot_factor", ""),
            row.get("forward_guidance_snapshot_gap", ""),
            row.get("forward_guidance_snapshot_source", "kept"),
            row.get("forward_guidance_hamming_radius", ""),
            row.get("forward_guidance_trigger_mode", "top_gap"),
            row.get("forward_guidance_nearcut_gap", ""),
            row.get("forward_guidance_pool_trigger_min_positive_nearcut", ""),
            row.get("forward_guidance_diversity_fallback", "none"),
            row.get("forward_guidance_mode", "detector_penalty"),
        )
        grouped.setdefault(key, []).append(series.astype(np.int32, copy=False))
    if not grouped:
        return

    out_rows: list[dict[str, object]] = []
    palette = ["#2b6cb0", "#d97706", "#059669", "#7c3aed", "#dc2626", "#0f766e"]
    for index, (key, traces) in enumerate(sorted(grouped.items(), key=lambda item: item[0])):
        (
            decoder_mode,
            backward_column_order,
            score_mode,
            beam_size,
            beam_score_gap_threshold,
            selective_secondary_score_mode,
            selective_secondary_trigger_gap,
            selective_secondary_band_size,
            forward_guidance_trigger_gap,
            forward_guidance_snapshot_factor,
            forward_guidance_snapshot_gap,
            forward_guidance_snapshot_source,
            forward_guidance_hamming_radius,
            forward_guidance_trigger_mode,
            forward_guidance_nearcut_gap,
            forward_guidance_pool_trigger_min_positive_nearcut,
            forward_guidance_diversity_fallback,
            forward_guidance_mode,
        ) = key
        stacked = np.vstack(traces)
        mean_profile = np.mean(stacked, axis=0)
        p90_profile = np.quantile(stacked, 0.90, axis=0)
        p99_profile = np.quantile(stacked, 0.99, axis=0)
        max_profile = np.max(stacked, axis=0)
        columns = np.arange(mean_profile.shape[0], dtype=np.int32)
        score_gap_text = _format_beam_score_gap_threshold(beam_score_gap_threshold)
        beam_score_gap_policy = None
        label = _curve_label(
            decoder_mode=str(decoder_mode),
            backward_column_order=str(backward_column_order),
            correction_state_mode="none",
            score_mode=str(score_mode),
            beam_score_gap_threshold=(
                None if not score_gap_text else float(score_gap_text)
            ),
            beam_score_gap_policy=beam_score_gap_policy,
            lookahead_depth=0,
            lookahead_shortlist_size=0,
            delayed_pruning_gap_threshold=0.0,
            delayed_pruning_factor=1,
            pruning_replay_checkpoint_stride=0,
            pruning_replay_horizon=0,
            tail_exact_columns=0,
            superstep_mode="none",
            detector_bucket_pruning=False,
            detector_bucket_max_logicals=0,
            logical_class_reserve_min_classes=0,
            logical_class_reserve_max_replacements=0,
            logical_class_reserve_min_remaining_columns=0,
            logical_class_quota_top_classes=0,
            logical_class_quota_reserved_slots=0,
            logical_class_quota_min_remaining_columns=0,
            lineage_reserve_checkpoint_stride=0,
            lineage_reserve_reserved_slots=0,
            logical_rerank_columns=0,
            logical_rerank_shortlist_size=0,
            logical_rerank_min_classes=0,
            logical_rerank_state_budget=0,
            logical_rerank_transition_budget=0,
            logical_rerank_checkpoint_stride=0,
            logical_rerank_max_passes=1,
            logical_rerank_mode="exact_tail",
            selective_secondary_score_mode=str(selective_secondary_score_mode),
            selective_secondary_trigger_gap=(
                0.0
                if selective_secondary_trigger_gap in {"", None}
                else float(selective_secondary_trigger_gap)
            ),
            selective_secondary_band_size=(
                0 if selective_secondary_band_size in {"", None} else int(selective_secondary_band_size)
            ),
            forward_guidance_trigger_gap=(
                0.0 if forward_guidance_trigger_gap in {"", None} else float(forward_guidance_trigger_gap)
            ),
            forward_guidance_snapshot_factor=(
                1.0 if forward_guidance_snapshot_factor in {"", None} else float(forward_guidance_snapshot_factor)
            ),
            forward_guidance_snapshot_gap=forward_guidance_snapshot_gap,
            forward_guidance_snapshot_source=str(forward_guidance_snapshot_source),
            forward_guidance_hamming_radius=(
                0 if forward_guidance_hamming_radius in {"", None} else int(forward_guidance_hamming_radius)
            ),
            forward_guidance_trigger_mode=str(forward_guidance_trigger_mode),
            forward_guidance_nearcut_gap=(
                0.0 if forward_guidance_nearcut_gap in {"", None} else float(forward_guidance_nearcut_gap)
            ),
            forward_guidance_pool_trigger_min_positive_nearcut=(
                1
                if forward_guidance_pool_trigger_min_positive_nearcut in {"", None}
                else int(forward_guidance_pool_trigger_min_positive_nearcut)
            ),
            forward_guidance_diversity_fallback=str(forward_guidance_diversity_fallback),
            forward_guidance_mode=str(forward_guidance_mode),
        )
        slug = _profile_slug(
            decoder_mode=str(decoder_mode),
            score_mode=str(score_mode),
            beam_size=int(beam_size),
            beam_score_gap_threshold=beam_score_gap_threshold,
            backward_column_order=str(backward_column_order),
            beam_score_gap_policy=beam_score_gap_policy,
            selective_secondary_score_mode=str(selective_secondary_score_mode),
            selective_secondary_trigger_gap=(
                0.0
                if selective_secondary_trigger_gap in {"", None}
                else float(selective_secondary_trigger_gap)
            ),
            selective_secondary_band_size=(
                0 if selective_secondary_band_size in {"", None} else int(selective_secondary_band_size)
            ),
            forward_guidance_trigger_gap=(
                0.0 if forward_guidance_trigger_gap in {"", None} else float(forward_guidance_trigger_gap)
            ),
            forward_guidance_snapshot_factor=(
                1.0 if forward_guidance_snapshot_factor in {"", None} else float(forward_guidance_snapshot_factor)
            ),
            forward_guidance_snapshot_gap=forward_guidance_snapshot_gap,
            forward_guidance_snapshot_source=str(forward_guidance_snapshot_source),
            forward_guidance_hamming_radius=(
                0 if forward_guidance_hamming_radius in {"", None} else int(forward_guidance_hamming_radius)
            ),
            forward_guidance_trigger_mode=str(forward_guidance_trigger_mode),
            forward_guidance_nearcut_gap=(
                0.0 if forward_guidance_nearcut_gap in {"", None} else float(forward_guidance_nearcut_gap)
            ),
            forward_guidance_pool_trigger_min_positive_nearcut=(
                1
                if forward_guidance_pool_trigger_min_positive_nearcut in {"", None}
                else int(forward_guidance_pool_trigger_min_positive_nearcut)
            ),
            forward_guidance_diversity_fallback=str(forward_guidance_diversity_fallback),
            forward_guidance_mode=str(forward_guidance_mode),
        )
        for column_index in range(mean_profile.shape[0]):
            out_rows.append(
                {
                    "profile_slug": str(slug),
                    "label": str(label),
                    "decoder_mode": str(decoder_mode),
                    "backward_column_order": str(backward_column_order),
                    "score_mode": str(score_mode),
                    "beam_size": int(beam_size),
                    "beam_score_gap_threshold": score_gap_text,
                    "forward_guidance_snapshot_source": str(forward_guidance_snapshot_source),
                    "forward_guidance_mode": str(forward_guidance_mode),
                    "forward_guidance_trigger_gap": str(forward_guidance_trigger_gap),
                    "forward_guidance_snapshot_factor": str(forward_guidance_snapshot_factor),
                    "forward_guidance_snapshot_gap": str(forward_guidance_snapshot_gap),
                    "forward_guidance_hamming_radius": str(forward_guidance_hamming_radius),
                    "forward_guidance_trigger_mode": str(forward_guidance_trigger_mode),
                    "forward_guidance_nearcut_gap": str(forward_guidance_nearcut_gap),
                    "forward_guidance_pool_trigger_min_positive_nearcut": str(
                        forward_guidance_pool_trigger_min_positive_nearcut
                    ),
                    "forward_guidance_diversity_fallback": str(forward_guidance_diversity_fallback),
                    "column_index": int(column_index),
                    "state_count_mean": float(mean_profile[column_index]),
                    "state_count_p90": float(p90_profile[column_index]),
                    "state_count_p99": float(p99_profile[column_index]),
                    "state_count_max": float(max_profile[column_index]),
                    "shots": int(stacked.shape[0]),
                }
            )

        fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=200)
        color = palette[int(index) % len(palette)]
        ax.plot(columns, mean_profile, linewidth=2.0, color=color, label="mean states")
        ax.plot(columns, p90_profile, linewidth=1.4, color=color, linestyle="--", label="p90 states")
        ax.plot(columns, p99_profile, linewidth=1.2, color=color, linestyle=":", label="p99 states")
        ax.axhline(float(beam_size), linewidth=1.2, color="#111827", linestyle="-.", label=f"K cap = {int(beam_size)}")
        ax.set_xlabel("processed column boundary")
        ax.set_ylabel("retained frontier states")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False)
        ax.set_title(f"Beam Profile: {label}")
        fig.tight_layout()
        fig.savefig(out_dir / f"fig_state_count_profile_{slug}.png", bbox_inches="tight")
        plt.close(fig)

    _write_csv(
        out_dir / "state_count_profile_summary.csv",
        out_rows,
        [
            "profile_slug",
            "label",
            "decoder_mode",
            "backward_column_order",
            "score_mode",
            "beam_size",
            "beam_score_gap_threshold",
            "forward_guidance_mode",
            "forward_guidance_trigger_gap",
            "forward_guidance_snapshot_factor",
            "forward_guidance_snapshot_gap",
            "forward_guidance_snapshot_source",
            "forward_guidance_hamming_radius",
            "forward_guidance_trigger_mode",
            "forward_guidance_nearcut_gap",
            "forward_guidance_pool_trigger_min_positive_nearcut",
            "forward_guidance_diversity_fallback",
            "column_index",
            "state_count_mean",
            "state_count_p90",
            "state_count_p99",
            "state_count_max",
            "shots",
        ],
    )


def _write_selective_local_lookahead_artifacts(
    *,
    per_shot_rows: Sequence[dict[str, object]],
    summary_rows: Sequence[dict[str, object]],
    out_dir: Path,
) -> None:
    if not any(
        _selective_local_lookahead_enabled(row.get("selective_local_lookahead_mode", "none"))
        or str(row.get("selective_local_lookahead_steps_json", "")).strip() not in {"", "[]"}
        for row in per_shot_rows
    ):
        return
    _write_csv(
        out_dir / "selective_lookahead_rows.csv",
        per_shot_rows,
        _fieldnames_from_rows(per_shot_rows, ["shot"]),
    )
    _write_csv(
        out_dir / "selective_lookahead_summary.csv",
        summary_rows,
        _fieldnames_from_rows(summary_rows, ["decoder", "beam_size", "score_mode"]),
    )
    trigger_rows: list[dict[str, object]] = []
    for row in per_shot_rows:
        raw_steps = str(row.get("selective_local_lookahead_steps_json", "")).strip()
        if not raw_steps:
            continue
        try:
            steps = json.loads(raw_steps)
        except json.JSONDecodeError:
            continue
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            trigger_row = {
                "shot": row.get("shot", ""),
                "family": row.get("family", ""),
                "scope": row.get("scope", ""),
                "decoder_mode": row.get("decoder_mode", ""),
                "beam_size": row.get("beam_size", ""),
                "score_mode": row.get("score_mode", ""),
                "frame_ok": row.get("frame_ok", ""),
                "frame_fail_type": row.get("frame_fail_type", ""),
                "logical_hat": row.get("logical_hat", ""),
                "truth_logical": row.get("truth_logical", ""),
                "truth_terminal_present": row.get("truth_logical_terminal_present", ""),
                "decode_s": row.get("decode_s", ""),
            }
            trigger_row.update({str(key): value for key, value in step.items()})
            trigger_rows.append(trigger_row)
    _write_csv(
        out_dir / "selective_lookahead_trigger_rows.csv",
        trigger_rows,
        _fieldnames_from_rows(
            trigger_rows,
            ["shot", "family", "scope", "beam_size", "score_mode", "boundary_column_index"],
        ),
    )


def _plot_fer_vs_beam(
    *,
    summary_rows: Sequence[dict[str, object]],
    beam_baselines: Sequence[dict[str, object]],
    out_path: Path,
    value_key: str,
    ylabel: str,
    round_count: int,
) -> None:
    shots = int(summary_rows[0]["shots"]) if summary_rows else 1
    frame_floor = 0.5 / float(max(1, shots))
    per_round_floor = _frame_fer_to_per_round_exact(frame_floor, int(round_count))
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=200)
    floor = per_round_floor if str(value_key) == "fer_per_round" else frame_floor
    score_mode_groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in summary_rows:
        key = (
            str(row.get("decoder_mode", "forward")),
            str(row.get("backward_column_order", "")),
            str(row.get("correction_state_mode", "none")),
            str(row["score_mode"]),
            (
                None
                if row.get("beam_score_gap_threshold", "") in {"", None}
                else float(row["beam_score_gap_threshold"])
            ),
            str(row.get("selective_secondary_score_mode", "")),
            (
                None
                if row.get("selective_secondary_trigger_gap", "") in {"", None}
                else float(row.get("selective_secondary_trigger_gap", 0.0))
            ),
            (
                0
                if row.get("selective_secondary_band_size", "") in {"", None}
                else int(row.get("selective_secondary_band_size", 0))
            ),
            (
                0.0
                if row.get("forward_guidance_trigger_gap", "") in {"", None}
                else float(row.get("forward_guidance_trigger_gap", 0.0))
            ),
            (
                1.0
                if row.get("forward_guidance_snapshot_factor", "") in {"", None}
                else float(row.get("forward_guidance_snapshot_factor", 1.0))
            ),
            row.get("forward_guidance_snapshot_gap", ""),
            str(row.get("forward_guidance_snapshot_source", "kept")),
            (
                0
                if row.get("forward_guidance_hamming_radius", "") in {"", None}
                else int(row.get("forward_guidance_hamming_radius", 0))
            ),
            str(row.get("forward_guidance_trigger_mode", "top_gap")),
            (
                0.0
                if row.get("forward_guidance_nearcut_gap", "") in {"", None}
                else float(row.get("forward_guidance_nearcut_gap", 0.0))
            ),
            (
                1
                if row.get("forward_guidance_pool_trigger_min_positive_nearcut", "") in {"", None}
                else int(row.get("forward_guidance_pool_trigger_min_positive_nearcut", 1))
            ),
            str(row.get("forward_guidance_diversity_fallback", "none")),
            str(row.get("forward_guidance_mode", "detector_penalty")),
            int(row["lookahead_depth"]),
            int(row["lookahead_shortlist_size"]),
            float(row.get("delayed_pruning_gap_threshold", 0.0)),
            int(row.get("delayed_pruning_factor", 1)),
            int(row.get("pruning_replay_checkpoint_stride", 0)),
            int(row.get("pruning_replay_horizon", 0)),
            int(row.get("tail_exact_columns", 0)),
            str(row.get("superstep_mode", "none")),
            bool(row.get("detector_bucket_pruning", False)),
            int(row.get("detector_bucket_max_logicals", 0)),
            int(row.get("logical_class_reserve_min_classes", 0)),
            int(row.get("logical_class_reserve_max_replacements", 0)),
            int(row.get("logical_class_reserve_min_remaining_columns", 0)),
            int(row.get("logical_class_quota_top_classes", 0)),
            int(row.get("logical_class_quota_reserved_slots", 0)),
            int(row.get("logical_class_quota_min_remaining_columns", 0)),
            int(row.get("lineage_reserve_checkpoint_stride", 0)),
            int(row.get("lineage_reserve_reserved_slots", 0)),
            int(row.get("logical_rerank_columns", 0)),
            int(row.get("logical_rerank_shortlist_size", 0)),
            int(row.get("logical_rerank_min_classes", 0)),
            int(row.get("logical_rerank_state_budget", 0)),
            int(row.get("logical_rerank_transition_budget", 0)),
            int(row.get("logical_rerank_checkpoint_stride", 0)),
            int(row.get("logical_rerank_max_passes", 1)),
            str(row.get("logical_rerank_mode", "exact_tail")),
        )
        score_mode_groups.setdefault(key, []).append(dict(row))
    palette = ["#2b6cb0", "#d97706", "#059669", "#7c3aed", "#dc2626", "#0f766e"]
    for index, (key, rows) in enumerate(
        sorted(score_mode_groups.items(), key=lambda item: item[0])
    ):
        decoder_mode, backward_column_order, correction_state_mode, score_mode, beam_score_gap_threshold, selective_secondary_score_mode, selective_secondary_trigger_gap, selective_secondary_band_size, forward_guidance_trigger_gap, forward_guidance_snapshot_factor, forward_guidance_snapshot_gap, forward_guidance_snapshot_source, forward_guidance_hamming_radius, forward_guidance_trigger_mode, forward_guidance_nearcut_gap, forward_guidance_pool_trigger_min_positive_nearcut, forward_guidance_diversity_fallback, forward_guidance_mode, lookahead_depth, lookahead_shortlist_size, delayed_pruning_gap_threshold, delayed_pruning_factor, pruning_replay_checkpoint_stride, pruning_replay_horizon, tail_exact_columns, superstep_mode, detector_bucket_pruning, detector_bucket_max_logicals, logical_class_reserve_min_classes, logical_class_reserve_max_replacements, logical_class_reserve_min_remaining_columns, logical_class_quota_top_classes, logical_class_quota_reserved_slots, logical_class_quota_min_remaining_columns, lineage_reserve_checkpoint_stride, lineage_reserve_reserved_slots, logical_rerank_columns, logical_rerank_shortlist_size, logical_rerank_min_classes, logical_rerank_state_budget, logical_rerank_transition_budget, logical_rerank_checkpoint_stride, logical_rerank_max_passes, logical_rerank_mode = key
        beam_score_gap_policy = _beam_score_gap_policy_from_row(rows[0]) if rows else None
        xs = np.asarray([int(row["beam_size"]) for row in rows], dtype=np.float64)
        ys = np.asarray([max(float(row[value_key]), float(floor)) for row in rows], dtype=np.float64)
        color = palette[int(index) % len(palette)]
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.1,
            color=str(color),
            label=_curve_label(
                decoder_mode=str(decoder_mode),
                backward_column_order=str(backward_column_order),
                correction_state_mode=str(correction_state_mode),
                score_mode=str(score_mode),
                beam_score_gap_threshold=beam_score_gap_threshold,
                beam_score_gap_policy=beam_score_gap_policy,
                lookahead_depth=int(lookahead_depth),
                lookahead_shortlist_size=int(lookahead_shortlist_size),
                delayed_pruning_gap_threshold=float(delayed_pruning_gap_threshold),
                delayed_pruning_factor=int(delayed_pruning_factor),
                pruning_replay_checkpoint_stride=int(pruning_replay_checkpoint_stride),
                pruning_replay_horizon=int(pruning_replay_horizon),
                tail_exact_columns=int(tail_exact_columns),
                superstep_mode=str(superstep_mode),
                detector_bucket_pruning=bool(detector_bucket_pruning),
                detector_bucket_max_logicals=int(detector_bucket_max_logicals),
                logical_class_reserve_min_classes=int(logical_class_reserve_min_classes),
                logical_class_reserve_max_replacements=int(logical_class_reserve_max_replacements),
                logical_class_reserve_min_remaining_columns=int(logical_class_reserve_min_remaining_columns),
                logical_class_quota_top_classes=int(logical_class_quota_top_classes),
                logical_class_quota_reserved_slots=int(logical_class_quota_reserved_slots),
                logical_class_quota_min_remaining_columns=int(logical_class_quota_min_remaining_columns),
                lineage_reserve_checkpoint_stride=int(lineage_reserve_checkpoint_stride),
                lineage_reserve_reserved_slots=int(lineage_reserve_reserved_slots),
                logical_rerank_columns=int(logical_rerank_columns),
                logical_rerank_shortlist_size=int(logical_rerank_shortlist_size),
                logical_rerank_min_classes=int(logical_rerank_min_classes),
                logical_rerank_state_budget=int(logical_rerank_state_budget),
                logical_rerank_transition_budget=int(logical_rerank_transition_budget),
                logical_rerank_checkpoint_stride=int(logical_rerank_checkpoint_stride),
                logical_rerank_max_passes=int(logical_rerank_max_passes),
                logical_rerank_mode=str(logical_rerank_mode),
                selective_secondary_score_mode=str(selective_secondary_score_mode),
                selective_secondary_trigger_gap=(
                    0.0 if selective_secondary_trigger_gap is None else float(selective_secondary_trigger_gap)
                ),
                selective_secondary_band_size=int(selective_secondary_band_size),
                forward_guidance_trigger_gap=float(forward_guidance_trigger_gap),
                forward_guidance_snapshot_factor=float(forward_guidance_snapshot_factor),
                forward_guidance_snapshot_gap=forward_guidance_snapshot_gap,
                forward_guidance_snapshot_source=str(forward_guidance_snapshot_source),
                forward_guidance_hamming_radius=int(forward_guidance_hamming_radius),
                forward_guidance_trigger_mode=str(forward_guidance_trigger_mode),
                forward_guidance_nearcut_gap=float(forward_guidance_nearcut_gap),
                forward_guidance_pool_trigger_min_positive_nearcut=int(
                    forward_guidance_pool_trigger_min_positive_nearcut
                ),
                forward_guidance_diversity_fallback=str(forward_guidance_diversity_fallback),
                forward_guidance_mode=str(forward_guidance_mode),
            ),
        )

    baseline_styles = {
        "beam weak": ("--", "#6b7280"),
        "beam weak (500-shot upper bound)": ("--", "#6b7280"),
        "beam strong": (":", "#111827"),
    }
    for baseline in beam_baselines:
        label = str(baseline["label"])
        linestyle, color = baseline_styles.get(label, ("-.", "#4b5563"))
        raw_value = float(baseline["fer_per_round"] if str(value_key) == "fer_per_round" else baseline["fer_upper"])
        ax.axhline(max(raw_value, float(floor)), color=color, linestyle=linestyle, linewidth=1.5, label=label)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("beam size K")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_pruning_diagnostics_vs_beam(
    *,
    summary_rows: Sequence[dict[str, object]],
    out_path: Path,
    left_key: str = "cumulative_discarded_prefix_mass_mean",
    right_key: str = "max_discarded_prefix_fraction_mean",
    left_ylabel: str = "mean cumulative discarded prefix mass",
    right_ylabel: str = "mean max discarded prefix fraction",
    left_title: str = "Pruning-Loss Certificate",
    right_title: str = "Worst-Step Discard Fraction",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=200)
    score_mode_groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in summary_rows:
        key = (
            str(row.get("decoder_mode", "forward")),
            str(row.get("backward_column_order", "")),
            str(row.get("correction_state_mode", "none")),
            str(row["score_mode"]),
            (
                None
                if row.get("beam_score_gap_threshold", "") in {"", None}
                else float(row["beam_score_gap_threshold"])
            ),
            str(row.get("selective_secondary_score_mode", "")),
            (
                None
                if row.get("selective_secondary_trigger_gap", "") in {"", None}
                else float(row.get("selective_secondary_trigger_gap", 0.0))
            ),
            (
                0
                if row.get("selective_secondary_band_size", "") in {"", None}
                else int(row.get("selective_secondary_band_size", 0))
            ),
            (
                0.0
                if row.get("forward_guidance_trigger_gap", "") in {"", None}
                else float(row.get("forward_guidance_trigger_gap", 0.0))
            ),
            (
                1.0
                if row.get("forward_guidance_snapshot_factor", "") in {"", None}
                else float(row.get("forward_guidance_snapshot_factor", 1.0))
            ),
            row.get("forward_guidance_snapshot_gap", ""),
            str(row.get("forward_guidance_snapshot_source", "kept")),
            (
                0
                if row.get("forward_guidance_hamming_radius", "") in {"", None}
                else int(row.get("forward_guidance_hamming_radius", 0))
            ),
            str(row.get("forward_guidance_trigger_mode", "top_gap")),
            (
                0.0
                if row.get("forward_guidance_nearcut_gap", "") in {"", None}
                else float(row.get("forward_guidance_nearcut_gap", 0.0))
            ),
            (
                1
                if row.get("forward_guidance_pool_trigger_min_positive_nearcut", "") in {"", None}
                else int(row.get("forward_guidance_pool_trigger_min_positive_nearcut", 1))
            ),
            str(row.get("forward_guidance_diversity_fallback", "none")),
            str(row.get("forward_guidance_mode", "detector_penalty")),
            int(row["lookahead_depth"]),
            int(row["lookahead_shortlist_size"]),
            float(row.get("delayed_pruning_gap_threshold", 0.0)),
            int(row.get("delayed_pruning_factor", 1)),
            int(row.get("pruning_replay_checkpoint_stride", 0)),
            int(row.get("pruning_replay_horizon", 0)),
            int(row.get("tail_exact_columns", 0)),
            str(row.get("superstep_mode", "none")),
            bool(row.get("detector_bucket_pruning", False)),
            int(row.get("detector_bucket_max_logicals", 0)),
            int(row.get("logical_class_reserve_min_classes", 0)),
            int(row.get("logical_class_reserve_max_replacements", 0)),
            int(row.get("logical_class_reserve_min_remaining_columns", 0)),
            int(row.get("logical_class_quota_top_classes", 0)),
            int(row.get("logical_class_quota_reserved_slots", 0)),
            int(row.get("logical_class_quota_min_remaining_columns", 0)),
            int(row.get("lineage_reserve_checkpoint_stride", 0)),
            int(row.get("lineage_reserve_reserved_slots", 0)),
            int(row.get("logical_rerank_columns", 0)),
            int(row.get("logical_rerank_shortlist_size", 0)),
            int(row.get("logical_rerank_min_classes", 0)),
            int(row.get("logical_rerank_state_budget", 0)),
            int(row.get("logical_rerank_transition_budget", 0)),
            int(row.get("logical_rerank_checkpoint_stride", 0)),
            int(row.get("logical_rerank_max_passes", 1)),
            str(row.get("logical_rerank_mode", "exact_tail")),
        )
        score_mode_groups.setdefault(key, []).append(dict(row))

    palette = ["#2b6cb0", "#d97706", "#059669", "#7c3aed", "#dc2626", "#0f766e"]
    left_values = [float(row.get(str(left_key), 0.0)) for row in summary_rows]
    right_values = [float(row.get(str(right_key), 0.0)) for row in summary_rows]
    left_floor = _plot_floor_for_values(left_values)
    right_floor = _plot_floor_for_values(right_values)

    for index, (key, rows) in enumerate(
        sorted(score_mode_groups.items(), key=lambda item: item[0])
    ):
        decoder_mode, backward_column_order, correction_state_mode, score_mode, beam_score_gap_threshold, selective_secondary_score_mode, selective_secondary_trigger_gap, selective_secondary_band_size, forward_guidance_trigger_gap, forward_guidance_snapshot_factor, forward_guidance_snapshot_gap, forward_guidance_snapshot_source, forward_guidance_hamming_radius, forward_guidance_trigger_mode, forward_guidance_nearcut_gap, forward_guidance_pool_trigger_min_positive_nearcut, forward_guidance_diversity_fallback, forward_guidance_mode, lookahead_depth, lookahead_shortlist_size, delayed_pruning_gap_threshold, delayed_pruning_factor, pruning_replay_checkpoint_stride, pruning_replay_horizon, tail_exact_columns, superstep_mode, detector_bucket_pruning, detector_bucket_max_logicals, logical_class_reserve_min_classes, logical_class_reserve_max_replacements, logical_class_reserve_min_remaining_columns, logical_class_quota_top_classes, logical_class_quota_reserved_slots, logical_class_quota_min_remaining_columns, lineage_reserve_checkpoint_stride, lineage_reserve_reserved_slots, logical_rerank_columns, logical_rerank_shortlist_size, logical_rerank_min_classes, logical_rerank_state_budget, logical_rerank_transition_budget, logical_rerank_checkpoint_stride, logical_rerank_max_passes, logical_rerank_mode = key
        beam_score_gap_policy = _beam_score_gap_policy_from_row(rows[0]) if rows else None
        xs = np.asarray([int(row["beam_size"]) for row in rows], dtype=np.float64)
        cumulative_mass = np.asarray(
            [max(float(row.get(str(left_key), 0.0)), float(left_floor)) for row in rows],
            dtype=np.float64,
        )
        max_fraction = np.asarray(
            [max(float(row.get(str(right_key), 0.0)), float(right_floor)) for row in rows],
            dtype=np.float64,
        )
        color = palette[int(index) % len(palette)]
        label = _curve_label(
            decoder_mode=str(decoder_mode),
            backward_column_order=str(backward_column_order),
            correction_state_mode=str(correction_state_mode),
            score_mode=str(score_mode),
            beam_score_gap_threshold=beam_score_gap_threshold,
            beam_score_gap_policy=beam_score_gap_policy,
            lookahead_depth=int(lookahead_depth),
            lookahead_shortlist_size=int(lookahead_shortlist_size),
            delayed_pruning_gap_threshold=float(delayed_pruning_gap_threshold),
            delayed_pruning_factor=int(delayed_pruning_factor),
            pruning_replay_checkpoint_stride=int(pruning_replay_checkpoint_stride),
            pruning_replay_horizon=int(pruning_replay_horizon),
            tail_exact_columns=int(tail_exact_columns),
            superstep_mode=str(superstep_mode),
            detector_bucket_pruning=bool(detector_bucket_pruning),
            detector_bucket_max_logicals=int(detector_bucket_max_logicals),
            logical_class_reserve_min_classes=int(logical_class_reserve_min_classes),
            logical_class_reserve_max_replacements=int(logical_class_reserve_max_replacements),
            logical_class_reserve_min_remaining_columns=int(logical_class_reserve_min_remaining_columns),
            logical_class_quota_top_classes=int(logical_class_quota_top_classes),
            logical_class_quota_reserved_slots=int(logical_class_quota_reserved_slots),
            logical_class_quota_min_remaining_columns=int(logical_class_quota_min_remaining_columns),
            lineage_reserve_checkpoint_stride=int(lineage_reserve_checkpoint_stride),
            lineage_reserve_reserved_slots=int(lineage_reserve_reserved_slots),
            logical_rerank_columns=int(logical_rerank_columns),
            logical_rerank_shortlist_size=int(logical_rerank_shortlist_size),
            logical_rerank_min_classes=int(logical_rerank_min_classes),
            logical_rerank_state_budget=int(logical_rerank_state_budget),
            logical_rerank_transition_budget=int(logical_rerank_transition_budget),
            logical_rerank_checkpoint_stride=int(logical_rerank_checkpoint_stride),
            logical_rerank_max_passes=int(logical_rerank_max_passes),
            logical_rerank_mode=str(logical_rerank_mode),
            selective_secondary_score_mode=str(selective_secondary_score_mode),
            selective_secondary_trigger_gap=(
                0.0 if selective_secondary_trigger_gap is None else float(selective_secondary_trigger_gap)
            ),
            selective_secondary_band_size=int(selective_secondary_band_size),
            forward_guidance_trigger_gap=float(forward_guidance_trigger_gap),
            forward_guidance_snapshot_factor=float(forward_guidance_snapshot_factor),
            forward_guidance_snapshot_gap=forward_guidance_snapshot_gap,
            forward_guidance_snapshot_source=str(forward_guidance_snapshot_source),
            forward_guidance_hamming_radius=int(forward_guidance_hamming_radius),
            forward_guidance_trigger_mode=str(forward_guidance_trigger_mode),
            forward_guidance_nearcut_gap=float(forward_guidance_nearcut_gap),
            forward_guidance_pool_trigger_min_positive_nearcut=int(
                forward_guidance_pool_trigger_min_positive_nearcut
            ),
            forward_guidance_diversity_fallback=str(forward_guidance_diversity_fallback),
            forward_guidance_mode=str(forward_guidance_mode),
        )
        axes[0].plot(xs, cumulative_mass, marker="o", linewidth=2.1, color=str(color), label=label)
        axes[1].plot(xs, max_fraction, marker="o", linewidth=2.1, color=str(color), label=label)

    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("beam size K")
    axes[0].set_ylabel(str(left_ylabel))
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].set_title(str(left_title))

    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("beam size K")
    axes[1].set_ylabel(str(right_ylabel))
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].set_title(str(right_title))
    axes[1].legend(frameon=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    *,
    out_dir: Path,
    family: LoadedProgressiveFamily,
    summary_rows: Sequence[dict[str, object]],
    beam_baselines: Sequence[dict[str, object]],
    p_location: float,
    shots: int,
    backend: str,
    terminal_failedrank_analysis: dict[str, object] | None = None,
) -> None:
    frontier_csv_rows = _frontier_rows(family)
    frontier = frontier_csv_rows[0]
    lookahead_depth = int(summary_rows[0].get("lookahead_depth", 0)) if summary_rows else 0
    lookahead_shortlist_size = int(summary_rows[0].get("lookahead_shortlist_size", 0)) if summary_rows else 0
    beam_score_gap_threshold = (
        None
        if not summary_rows or summary_rows[0].get("beam_score_gap_threshold", "") in {"", None}
        else float(summary_rows[0]["beam_score_gap_threshold"])
    )
    beam_score_gap_policy = _beam_score_gap_policy_from_row(summary_rows[0]) if summary_rows else None
    beam_score_gap_enabled = _beam_score_gap_threshold_enabled(beam_score_gap_threshold)
    beam_score_gap_text = _format_beam_score_gap_threshold(beam_score_gap_threshold)
    beam_score_gap_policy_text = _format_beam_score_gap_policy(beam_score_gap_policy)
    score_gap_control_text = _format_beam_score_gap_control(
        beam_score_gap_threshold=beam_score_gap_threshold,
        beam_score_gap_policy=beam_score_gap_policy,
    )
    delayed_pruning_gap_threshold = float(summary_rows[0].get("delayed_pruning_gap_threshold", 0.0)) if summary_rows else 0.0
    delayed_pruning_factor = int(summary_rows[0].get("delayed_pruning_factor", 1)) if summary_rows else 1
    pruning_replay_checkpoint_stride = int(summary_rows[0].get("pruning_replay_checkpoint_stride", 0)) if summary_rows else 0
    pruning_replay_horizon = int(summary_rows[0].get("pruning_replay_horizon", 0)) if summary_rows else 0
    tail_exact_columns = int(summary_rows[0].get("tail_exact_columns", 0)) if summary_rows else 0
    superstep_mode = str(summary_rows[0].get("superstep_mode", "none")) if summary_rows else "none"
    superstep_path_budget = int(summary_rows[0].get("superstep_path_budget", 0)) if summary_rows else 0
    superstep_state_budget = int(summary_rows[0].get("superstep_state_budget", 0)) if summary_rows else 0
    superstep_transition_budget = int(summary_rows[0].get("superstep_transition_budget", 0)) if summary_rows else 0
    detector_bucket_pruning = bool(summary_rows[0].get("detector_bucket_pruning", False)) if summary_rows else False
    detector_bucket_max_logicals = int(summary_rows[0].get("detector_bucket_max_logicals", 0)) if summary_rows else 0
    logical_class_reserve_min_classes = int(summary_rows[0].get("logical_class_reserve_min_classes", 0)) if summary_rows else 0
    logical_class_reserve_max_replacements = int(summary_rows[0].get("logical_class_reserve_max_replacements", 0)) if summary_rows else 0
    logical_class_reserve_min_remaining_columns = int(summary_rows[0].get("logical_class_reserve_min_remaining_columns", 0)) if summary_rows else 0
    logical_class_quota_top_classes = int(summary_rows[0].get("logical_class_quota_top_classes", 0)) if summary_rows else 0
    logical_class_quota_reserved_slots = int(summary_rows[0].get("logical_class_quota_reserved_slots", 0)) if summary_rows else 0
    logical_class_quota_min_remaining_columns = int(summary_rows[0].get("logical_class_quota_min_remaining_columns", 0)) if summary_rows else 0
    lineage_reserve_checkpoint_stride = int(summary_rows[0].get("lineage_reserve_checkpoint_stride", 0)) if summary_rows else 0
    lineage_reserve_reserved_slots = int(summary_rows[0].get("lineage_reserve_reserved_slots", 0)) if summary_rows else 0
    score_modes = sorted({str(row.get("score_mode", "")) for row in summary_rows}) if summary_rows else ["prefix"]
    terminal_failedrank_rows = (
        list(terminal_failedrank_analysis.get("rows", []))
        if terminal_failedrank_analysis is not None
        else []
    )
    terminal_failedrank_aggregate = (
        dict(terminal_failedrank_analysis.get("aggregate", {}))
        if terminal_failedrank_analysis is not None
        else {}
    )
    frontier_fig = out_dir / "fig_frontier_width_profile.png"
    fer_fig = out_dir / "fig_progressive_fer_vs_beam.png"
    fer_round_fig = out_dir / "fig_progressive_fer_per_round_vs_beam.png"
    pruning_fig = out_dir / "fig_progressive_pruning_diagnostics_vs_beam.png"
    truth_pruning_fig = out_dir / "fig_progressive_truth_logical_pruning_diagnostics_vs_beam.png"
    logical_diversity_clauses: list[str] = []
    if int(logical_class_reserve_min_classes) > 0 or int(logical_class_reserve_max_replacements) > 0:
        logical_diversity_clauses.append(
            f"logical-class reserve `{int(logical_class_reserve_min_classes)}/{int(logical_class_reserve_max_replacements)}/{int(logical_class_reserve_min_remaining_columns)}`"
        )
    if int(logical_class_quota_top_classes) > 0 or int(logical_class_quota_reserved_slots) > 0:
        logical_diversity_clauses.append(
            f"logical-class quota `{int(logical_class_quota_top_classes)}/{int(logical_class_quota_reserved_slots)}/{int(logical_class_quota_min_remaining_columns)}`"
        )
    if int(lineage_reserve_checkpoint_stride) > 0 or int(lineage_reserve_reserved_slots) > 0:
        logical_diversity_clauses.append(
            f"checkpoint-lineage reserve `{int(lineage_reserve_checkpoint_stride)}/{int(lineage_reserve_reserved_slots)}`"
        )
    logical_diversity_note = ""
    if logical_diversity_clauses:
        logical_diversity_note = ", " + ", ".join(logical_diversity_clauses)
    pruning_replay_note = ""
    if int(pruning_replay_checkpoint_stride) > 0 and int(pruning_replay_horizon) > 0:
        pruning_replay_note = (
            f", pruning replay stride `{int(pruning_replay_checkpoint_stride)}` horizon "
            f"`{int(pruning_replay_horizon)}`"
        )
    default_mode_enabled = (
        int(summary_rows[0].get("lookahead_depth", 0)) <= 0
        and int(summary_rows[0].get("tail_exact_columns", 0)) <= 0
        and not bool(beam_score_gap_enabled)
        and not _beam_score_gap_policy_enabled(beam_score_gap_policy)
        and float(delayed_pruning_gap_threshold) <= 0.0
        and int(delayed_pruning_factor) <= 1
        and int(pruning_replay_checkpoint_stride) <= 0
        and int(pruning_replay_horizon) <= 0
        and str(superstep_mode) == "none"
        and not bool(detector_bucket_pruning)
        and not logical_diversity_clauses
    )
    decoder_mode_clause = (
        f"- Decoder: progressive frontier/list decoder on a `{family.column_order_name}` derived from "
        f"`{family.column_order_source}`."
        if default_mode_enabled
        else f"- Decoder: progressive frontier/list decoder on a `{family.column_order_name}` derived from "
        f"`{family.column_order_source}`, with score-gap control "
        f"`{score_gap_control_text}`, pruning lookahead depth "
        f"`{int(summary_rows[0].get('lookahead_depth', 0))}` columns, shortlist size "
        f"`{int(summary_rows[0].get('lookahead_shortlist_size', 0))}`, delayed pruning "
        f"`{float(delayed_pruning_gap_threshold):g} x {int(delayed_pruning_factor)}`{pruning_replay_note}, exact tail solve "
        f"`{int(summary_rows[0].get('tail_exact_columns', 0))}` columns, and super-step mode "
        f"`{str(superstep_mode)}` (path budget `{int(superstep_path_budget)}`, state budget "
        f"`{int(superstep_state_budget)}`, transition budget `{int(superstep_transition_budget)}`), "
        f"detector-bucket pruning `{bool(detector_bucket_pruning)}` with per-detector logical cap "
        f"`{int(detector_bucket_max_logicals)}`{logical_diversity_note}."
    )
    figure2_mode_clause = (
        "The run uses the default prefix scorer with no score-gap pruning, shortlist, lookahead, delayed-pruning beam widening, pruning replay, exact tail solve, super-step grouping, detector-bucket pruning, or logical-class / checkpoint-lineage diversity gate. "
        if (
            int(lookahead_depth) <= 0
            and int(lookahead_shortlist_size) <= 0
            and not bool(beam_score_gap_enabled)
            and not _beam_score_gap_policy_enabled(beam_score_gap_policy)
            and float(delayed_pruning_gap_threshold) <= 0.0
            and int(delayed_pruning_factor) <= 1
            and int(pruning_replay_checkpoint_stride) <= 0
            and int(pruning_replay_horizon) <= 0
            and int(tail_exact_columns) <= 0
            and str(superstep_mode) == "none"
            and not bool(detector_bucket_pruning)
            and not logical_diversity_clauses
            and score_modes == ["prefix"]
        )
        else f"The run uses score-gap control `{score_gap_control_text}`, lookahead depth `{int(lookahead_depth)}`, shortlist size `{int(lookahead_shortlist_size)}`, delayed pruning threshold `{float(delayed_pruning_gap_threshold):g}` with widening factor `{int(delayed_pruning_factor)}`, pruning replay `{int(pruning_replay_checkpoint_stride)}/{int(pruning_replay_horizon)}`, exact tail solve `{int(tail_exact_columns)}` columns, super-step mode `{str(superstep_mode)}` with path budget `{int(superstep_path_budget)}`, state budget `{int(superstep_state_budget)}`, and transition budget `{int(superstep_transition_budget)}`, detector-bucket pruning `{bool(detector_bucket_pruning)}` with per-detector logical cap `{int(detector_bucket_max_logicals)}`{logical_diversity_note}. "
    )

    lines = [
        f"# {family.benchmark_title} {family.scope_label} progressive frontier report",
        "",
        f"- Benchmark: {family.benchmark_source_note}",
        f"- Scope: {family.scope_label} only, with the standard binary DEM matrix and independent Bernoulli detector-model faults drawn from `{family.priors_symbol}`.",
        f"- Matrix used: detector matrix `{int(family.matrix_rows)} x {int(family.matrix_cols)}`, logical matrix `{int(family.logical_rows)} x {int(family.matrix_cols)}`.",
        decoder_mode_clause,
        f"- FER policy: strict full logical frame error on the {family.scope_label.lower()} benchmark; failures are decomposed into `logical_fail`, `syndrome_fail`, and `exception_fail`.",
        f"- Noisy syndrome-extraction rounds: `{int(family.noisy_rounds)}`.",
        f"- Correction-state mode: `{family.correction_state_mode}` with `{int(family.correction_state_bits)}` tracked correction bits.",
        f"- Score modes compared: {', '.join(f'`{mode}`' for mode in score_modes)}.",
        "- Terminal ranking diagnostics: per-shot terminal selector signals are exported automatically and the `truth_present_but_not_selected` subset is analyzed automatically whenever present.",
        "",
        f"![{family.benchmark_title} {family.scope_label} frontier width]({frontier_fig})",
        "",
        (
            "Caption. Figure 1. Active detector-boundary width versus processed column prefix for progressive decoding "
            f"on the {family.benchmark_description} restricted to the {family.scope_label.lower()} task. The detector matrix is "
            f"`{int(frontier['matrix_rows'])} x {int(frontier['matrix_cols'])}`, the logical matrix is "
            f"`{int(frontier['logical_rows'])} x {int(frontier['matrix_cols'])}`, and the column order is a "
            f"`{family.column_order_name}` derived from `{family.column_order_source}`. The x-axis is processed prefix length; the y-axis is the "
            "number of detector rows still live on the frontier because some later column can still flip them. Main "
            "quantitative takeaway: this peak frontier width is the online detector memory that the progressive beam "
            "must carry before pruning quality becomes the dominant issue."
        ),
        "",
        "| family | model | detector matrix | logical matrix | detector edges | max frontier width | correction state | correction bits |",
        "| --- | --- | --- | --- | ---: | ---: | --- | ---: |",
        (
            f"| `{frontier['family']}` | `{frontier['model_label']}` | "
            f"`{int(frontier['matrix_rows'])} x {int(frontier['matrix_cols'])}` | "
            f"`{int(frontier['logical_rows'])} x {int(frontier['matrix_cols'])}` | "
            f"`{int(frontier['edge_count'])}` | `{int(frontier['frontier_max_active_detectors'])}` | "
            f"`{frontier['correction_state_mode']}` | `{int(frontier['correction_state_bits'])}` |"
        ),
        "",
        f"![{family.benchmark_title} {family.scope_label} FER vs beam]({fer_fig})",
        "",
        (
            f"Caption. Figure 2. Strict full logical frame error rate versus progressive beam size `K` on the "
            f"{family.benchmark_description} at detector-model fault rate `p={float(p_location):.6g}`, "
            f"using `{family.detector_symbol} = {int(frontier['matrix_rows'])} x {int(frontier['matrix_cols'])}`, "
            f"`{family.logical_symbol} = {int(frontier['logical_rows'])} x {int(frontier['matrix_cols'])}`, `{int(family.noisy_rounds)}` noisy syndrome-extraction "
            f"rounds, and `{int(shots)}` sampled {family.scope_label.lower()} frames. Each colored curve corresponds to one score mode "
            f"({', '.join(f'`{mode}`' for mode in score_modes)}), with beam size `K` on the x-axis. "
            f"{figure2_mode_clause}"
            "Horizontal beam-search reference lines are included when a stored anchor exists at the same `p`; at "
            "`p=0.001`, the repo currently only has a same-benchmark weak-beam replay upper bound from a `500`-shot "
            "exact replay bundle. The x-axis is beam size on a log base-2 scale; the y-axis is strict full logical FER "
            "on a logarithmic scale. Main quantitative takeaway: whether the alternative score modes reduce syndrome-"
            "fail pruning without paying too much extra work."
        ),
        "",
        "| decoder | correction state | correction bits | score_mode | beam_score_gap | beam_score_gap_policy | lookahead_depth | lookahead_shortlist_size | delayed_pruning | tail_exact_columns | superstep_mode | detector_bucket_pruning | detector_bucket_max_logicals | shots | fail_total | logical_fail | truth_missing_terminal | truth_present_but_not_selected | syndrome_fail | exception_fail | FER | mean decode s | us/transition | mean total evals | matrix |",
        "| --- | --- | ---: | --- | ---: | --- | ---: | ---: | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        row_policy_text = _format_beam_score_gap_policy(_beam_score_gap_policy_from_row(row))
        lines.append(
            f"| `{row['decoder']}` | `{row.get('correction_state_mode', 'none')}` | `{int(row.get('correction_state_bits', 0))}` | `{row['score_mode']}` | `{_format_beam_score_gap_threshold(row.get('beam_score_gap_threshold', ''))}` | `{row_policy_text if row_policy_text else '-'}` | `{int(row['lookahead_depth'])}` | `{int(row['lookahead_shortlist_size'])}` | `{float(row.get('delayed_pruning_gap_threshold', 0.0)):g} x {int(row.get('delayed_pruning_factor', 1))}` | `{int(row.get('tail_exact_columns', 0))}` | `{row.get('superstep_mode', 'none')}` | `{bool(row.get('detector_bucket_pruning', False))}` | `{int(row.get('detector_bucket_max_logicals', 0))}` | "
            f"`{int(row['shots'])}` | `{int(row['fail_total'])}` | `{int(row['logical_fail'])}` | "
            f"`{int(row.get('logical_fail_truth_missing_terminal', 0))}` | `{int(row.get('logical_fail_truth_present_but_not_selected', 0))}` | "
            f"`{int(row['syndrome_fail'])}` | `{int(row['exception_fail'])}` | `{float(row['fer']):.6g}` | "
            f"`{float(row['decode_s_mean']):.3f}` | `{float(row['us_per_transition_mean']):.3f}` | "
            f"`{float(row['transition_evals_total_mean']):.3f}` | "
            f"`{int(row['matrix_rows'])} x {int(row['matrix_cols'])}` |"
        )
    splice_summary_rows = [
        row for row in summary_rows if int(row.get("splice_enabled_count", 0)) > 0
    ]
    if splice_summary_rows:
        lines.extend(
            [
                "",
                "## Bidirectional Splice Rerank Audit",
                "",
                "- This audit is offline unless `--splice-replace-final-selection` was explicitly used; the splice reranker does not change online pruning.",
                "",
                "| decoder | score_mode | K | baseline failures | baseline logical | baseline syndrome | splice failures | splice logical | splice syndrome | fixes | breaks | unchanged failures | truth missing candidates | truth present not selected | finite cut frac | missing support frac |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in splice_summary_rows:
            lines.append(
                f"| `{row['decoder']}` | `{row['score_mode']}` | `{int(row['beam_size'])}` | "
                f"`{int(row.get('baseline_fail_total', 0))}` | "
                f"`{int(row.get('baseline_logical_fail', 0))}` | "
                f"`{int(row.get('baseline_syndrome_fail', 0))}` | "
                f"`{int(row.get('splice_fail_total', 0))}` | "
                f"`{int(row.get('splice_logical_fail', 0))}` | "
                f"`{int(row.get('splice_syndrome_fail', 0))}` | "
                f"`{int(row.get('splice_fixed_count', 0))}` | "
                f"`{int(row.get('splice_broken_count', 0))}` | "
                f"`{int(row.get('splice_unchanged_failure_count', 0))}` | "
                f"`{int(row.get('splice_candidate_missing_truth_count', 0))}` | "
                f"`{int(row.get('splice_truth_present_but_not_selected_count', 0))}` | "
                f"`{float(row.get('splice_finite_cut_fraction_mean', float('nan'))):.3f}` | "
                f"`{float(row.get('splice_missing_support_fraction_mean', float('nan'))):.3f}` |"
            )
    if beam_baselines:
        lines.extend(
            [
                "",
                "| archived beam baseline | FER used in plot | raw FER point estimate | work mean | work tail | work metric | note |",
                "| --- | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for baseline in beam_baselines:
            lines.append(
                f"| `{baseline['label']}` | `{float(baseline['fer_upper']):.6g}` | `{float(baseline['fer']):.6g}` | "
                f"`{float(baseline['work_mean']):.6g}` | `{float(baseline['work_tail']):.6g}` | "
                f"`{baseline['work_metric']}` | `{Path(str(baseline['source'])).name}` |"
            )
    else:
        lines.extend(
            [
                "",
                f"- Archived beam baseline note: no stored same-backend same-`p` {family.scope_label.lower()} beam-search anchor was found for this plot, so no horizontal beam line is shown.",
            ]
        )

    lines.extend(
        [
            "",
        f"![{family.benchmark_title} {family.scope_label} FER per round vs beam]({fer_round_fig})",
        "",
        (
            f"Caption. Figure 3. The same progressive {family.scope_label.lower()} detector-side DEM comparison as Figure 2 after exact "
            f"conversion from frame FER to FER per noisy syndrome-extraction round using the backend-specific `{int(family.noisy_rounds)}`-round "
            f"convention `1 - (1 - FER)^(1/{int(family.noisy_rounds)})`. The decoder family, matrix, corpus, and beam baselines are unchanged. The "
            "x-axis is beam size `K` on a log base-2 scale; the y-axis is strict FER per round on a logarithmic "
            "scale. Main quantitative takeaway: whether any frame-level gap to the beam-search target persists "
            "after standard per-round normalization for each score mode."
            ),
            "",
            "| decoder | correction state | correction bits | score_mode | beam_score_gap | beam_score_gap_policy | selective_secondary | lookahead_depth | lookahead_shortlist_size | delayed_pruning | tail_exact_columns | superstep_mode | detector_bucket_pruning | detector_bucket_max_logicals | FER/frame | FER/round | mean states | p99 mean states | max states seen | frontier max active |",
            "| --- | --- | ---: | --- | ---: | --- | --- | ---: | ---: | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary_rows:
        row_policy_text = _format_beam_score_gap_policy(_beam_score_gap_policy_from_row(row))
        selective_text = _format_selective_secondary_control(
            selective_secondary_score_mode=row.get("selective_secondary_score_mode", ""),
            selective_secondary_trigger_gap=row.get("selective_secondary_trigger_gap", ""),
            selective_secondary_band_size=row.get("selective_secondary_band_size", ""),
        )
        lines.append(
            f"| `{row['decoder']}` | `{row.get('correction_state_mode', 'none')}` | `{int(row.get('correction_state_bits', 0))}` | `{row['score_mode']}` | `{_format_beam_score_gap_threshold(row.get('beam_score_gap_threshold', ''))}` | `{row_policy_text if row_policy_text else '-'}` | `{selective_text if selective_text else '-'}` | `{int(row['lookahead_depth'])}` | `{int(row['lookahead_shortlist_size'])}` | `{float(row.get('delayed_pruning_gap_threshold', 0.0)):g} x {int(row.get('delayed_pruning_factor', 1))}` | `{int(row.get('tail_exact_columns', 0))}` | `{row.get('superstep_mode', 'none')}` | `{bool(row.get('detector_bucket_pruning', False))}` | `{int(row.get('detector_bucket_max_logicals', 0))}` | "
            f"`{float(row['fer']):.6g}` | `{float(row['fer_per_round']):.6g}` | "
            f"`{float(row['mean_states_mean']):.3f}` | `{float(row['mean_states_p99']):.3f}` | "
            f"`{int(row['max_states_seen'])}` | `{int(row['frontier_max_active_detectors'])}` |"
        )
    replay_summary_rows = [
        row
        for row in summary_rows
        if int(row.get("pruning_replay_checkpoint_stride", 0)) > 0
        and int(row.get("pruning_replay_horizon", 0)) > 0
    ]
    if replay_summary_rows:
        lines.extend(
            [
                "",
                "Replay telemetry",
                "",
                "| decoder | replay stride | replay horizon | attempts / shot | applies / shot | replaced states / shot | replaced states / apply | replayed columns / shot | extra replay evals / shot | physical total evals / shot | us / physical transition |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in replay_summary_rows:
            lines.append(
                f"| `{row['decoder']}` | `{int(row.get('pruning_replay_checkpoint_stride', 0))}` | "
                f"`{int(row.get('pruning_replay_horizon', 0))}` | "
                f"`{float(row.get('pruning_replay_attempt_count_mean', float('nan'))):.3f}` | "
                f"`{float(row.get('pruning_replay_applied_count_mean', float('nan'))):.3f}` | "
                f"`{float(row.get('pruning_replay_replaced_state_count_mean', float('nan'))):.3f}` | "
                f"`{float(row.get('pruning_replay_replaced_states_per_apply_mean', float('nan'))):.3f}` | "
                f"`{float(row.get('pruning_replay_replayed_column_count_mean', float('nan'))):.3f}` | "
                f"`{float(row.get('pruning_replay_extra_transition_evals_mean', float('nan'))):.3f}` | "
                f"`{float(row.get('transition_evals_physical_total_mean', float('nan'))):.3f}` | "
                f"`{float(row.get('us_per_transition_physical_mean', float('nan'))):.3f}` |"
            )
    selective_summary_rows = [
        row
        for row in summary_rows
        if _selective_secondary_enabled(
            selective_secondary_score_mode=row.get("selective_secondary_score_mode", ""),
            selective_secondary_trigger_gap=row.get("selective_secondary_trigger_gap", ""),
            selective_secondary_band_size=row.get("selective_secondary_band_size", ""),
        )
    ]
    if selective_summary_rows:
        lines.extend(
            [
                "",
                "Selective Boundary Rerank Telemetry",
                "",
                "| decoder | selective_secondary | triggers / shot | changed keeps / shot | reranked states / shot |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in selective_summary_rows:
            selective_text = _format_selective_secondary_control(
                selective_secondary_score_mode=row.get("selective_secondary_score_mode", ""),
                selective_secondary_trigger_gap=row.get("selective_secondary_trigger_gap", ""),
                selective_secondary_band_size=row.get("selective_secondary_band_size", ""),
            )
            lines.append(
                f"| `{row['decoder']}` | `{selective_text}` | "
                f"`{float(row.get('selective_secondary_trigger_count_mean', float('nan'))):.3f}` | "
                f"`{float(row.get('selective_secondary_changed_count_mean', float('nan'))):.3f}` | "
                f"`{float(row.get('selective_secondary_reranked_state_count_mean', float('nan'))):.3f}` |"
            )
    lineage_summary_rows = [
        row
        for row in summary_rows
        if int(row.get("lineage_reserve_checkpoint_stride", 0)) > 0
        and int(row.get("lineage_reserve_reserved_slots", 0)) > 0
    ]
    if lineage_summary_rows:
        lines.extend(
            [
                "",
                "Checkpoint-Lineage Reserve Telemetry",
                "",
                "| decoder | checkpoint stride | reserved slots | lineage reserve applies / shot | lineage reserve kept states / shot | kept states / apply |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in lineage_summary_rows:
            kept_mean = float(row.get("lineage_reserve_kept_state_count_mean", float("nan")))
            applied_mean = float(row.get("lineage_reserve_applied_count_mean", float("nan")))
            kept_per_apply = float("nan")
            if math.isfinite(applied_mean) and applied_mean > 0.0 and math.isfinite(kept_mean):
                kept_per_apply = float(kept_mean / applied_mean)
            lines.append(
                f"| `{row['decoder']}` | `{int(row.get('lineage_reserve_checkpoint_stride', 0))}` | "
                f"`{int(row.get('lineage_reserve_reserved_slots', 0))}` | "
                f"`{applied_mean:.3f}` | "
                f"`{kept_mean:.3f}` | "
                f"`{kept_per_apply:.3f}` |"
            )
    if beam_baselines:
        lines.extend(
            [
                "",
                "| archived beam baseline | FER/round used in plot | note |",
                "| --- | ---: | --- |",
            ]
        )
        for baseline in beam_baselines:
            lines.append(
                f"| `{baseline['label']}` | `{float(baseline['fer_per_round']):.6g}` | "
                f"`{Path(str(baseline['source'])).name}` |"
            )

    lines.extend(
        [
            "",
            f"![{family.benchmark_title} {family.scope_label} pruning diagnostics vs beam]({pruning_fig})",
            "",
            (
                f"Caption. Figure 4. Pruning-loss diagnostics versus beam size `K` for the same {family.scope_label.lower()} detector-side DEM sweep as Figures 2 and 3 on "
                f"`{family.detector_symbol} = {int(frontier['matrix_rows'])} x {int(frontier['matrix_cols'])}`, `{family.logical_symbol} = {int(frontier['logical_rows'])} x {int(frontier['matrix_cols'])}`, "
                f"`{int(family.noisy_rounds)}` noisy syndrome-extraction rounds, and `{int(shots)}` sampled frames at `p={float(p_location):.6g}`. The left panel plots the shot-mean cumulative discarded prefix mass, i.e. the sum over boundary prune steps of the exact prefix probability mass removed by truncation; the right panel plots the shot-mean worst single-step discarded prefix fraction. Both axes use beam size `K` on a log base-2 scale and the y-axes are logarithmic with a small positive floor for exact zeros. Main quantitative takeaway: if finite-`K` frontier decoding works for structural reasons rather than luck, both curves should fall rapidly with `K`, and the practically good `K` regime should already sit at very small discarded-mass values."
            ),
            "",
            "| decoder | score_mode | K | discard steps/shot | mean cumulative discarded mass | p99 cumulative discarded mass | mean max discarded mass | mean max discarded fraction | p99 max discarded fraction |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| `{row['decoder']}` | `{row['score_mode']}` | `{int(row['beam_size'])}` | "
            f"`{float(row.get('discard_step_count_mean', float('nan'))):.3f}` | "
            f"`{float(row.get('cumulative_discarded_prefix_mass_mean', float('nan'))):.6g}` | "
            f"`{float(row.get('cumulative_discarded_prefix_mass_p99', float('nan'))):.6g}` | "
            f"`{float(row.get('max_discarded_prefix_mass_mean', float('nan'))):.6g}` | "
            f"`{float(row.get('max_discarded_prefix_fraction_mean', float('nan'))):.6g}` | "
            f"`{float(row.get('max_discarded_prefix_fraction_p99', float('nan'))):.6g}` |"
        )

    lines.extend(
        [
            "",
            f"![{family.benchmark_title} {family.scope_label} true-logical pruning diagnostics vs beam]({truth_pruning_fig})",
            "",
            (
                f"Caption. Figure 5. Offline oracle pruning diagnostics versus beam size `K` for the same {family.scope_label.lower()} detector-side DEM sweep as Figures 2 to 4 on "
                f"`{family.detector_symbol} = {int(frontier['matrix_rows'])} x {int(frontier['matrix_cols'])}`, `{family.logical_symbol} = {int(frontier['logical_rows'])} x {int(frontier['matrix_cols'])}`, "
                f"`{int(family.noisy_rounds)}` noisy syndrome-extraction rounds, and `{int(shots)}` sampled frames at `p={float(p_location):.6g}`. This figure is diagnostic only: it uses the known true logical class of each sampled frame for reporting, not for any online decoder decision. The left panel plots the shot-mean cumulative discarded prefix mass restricted to states in the true logical class; the right panel plots the shot-mean worst single-step fraction of currently present true-class prefix mass discarded by pruning. Both axes use beam size `K` on a log base-2 scale and the y-axes are logarithmic with a small positive floor for exact zeros. Main quantitative takeaway: if modest finite-`K` works because pruning mostly removes wrong logical classes, the true-class discarded-mass curves should become small much faster than the raw discarded-mass curves from Figure 4."
            ),
            "",
            "| decoder | score_mode | K | truth discard steps/shot | mean cumulative true-class discarded mass | p99 cumulative true-class discarded mass | mean max true-class discarded mass | mean max true-class discarded fraction | p99 max true-class discarded fraction |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| `{row['decoder']}` | `{row['score_mode']}` | `{int(row['beam_size'])}` | "
            f"`{float(row.get('truth_logical_discard_step_count_mean', float('nan'))):.3f}` | "
            f"`{float(row.get('cumulative_truth_logical_discarded_prefix_mass_mean', float('nan'))):.6g}` | "
            f"`{float(row.get('cumulative_truth_logical_discarded_prefix_mass_p99', float('nan'))):.6g}` | "
            f"`{float(row.get('max_truth_logical_discarded_prefix_mass_mean', float('nan'))):.6g}` | "
            f"`{float(row.get('max_truth_logical_discarded_prefix_fraction_mean', float('nan'))):.6g}` | "
            f"`{float(row.get('max_truth_logical_discarded_prefix_fraction_p99', float('nan'))):.6g}` |"
        )

    lines.extend(["", "## Terminal Ranking Diagnostics", ""])
    if terminal_failedrank_analysis is None:
        lines.append(
            "- Terminal ranking analysis was not run because terminal selector signal export was disabled."
        )
    else:
        lines.extend(
            [
                f"- Analyzed subset: `{int(terminal_failedrank_aggregate.get('row_count', 0))}` `truth_present_but_not_selected` shots out of `{int(terminal_failedrank_aggregate.get('source_row_count', 0))}` total per-shot rows and `{int(terminal_failedrank_aggregate.get('signal_row_count', 0))}` `ok` rows with exported terminal signals.",
                f"- Matrix used for the diagnostic: `{int(family.matrix_rows)} x {int(family.matrix_cols)}` detector matrix with `{int(family.logical_rows)}` logical rows and `{int(family.noisy_rounds)}` noisy rounds.",
            ]
        )
        if terminal_failedrank_rows:
            failedrank_fig = Path(str(terminal_failedrank_analysis["plot_path"])).name
            summary_csv_name = Path(str(terminal_failedrank_analysis["summary_csv"])).name
            aggregate_json_name = Path(str(terminal_failedrank_analysis["aggregate_json"])).name
            lines.extend(
                [
                    f"- Main outcome: `{int(terminal_failedrank_aggregate.get('monotone_mvc_certificate_count', 0))} / {int(terminal_failedrank_aggregate.get('row_count', 0))}` failed-rank shots satisfy the monotone `(M,V,C)` certificate, and `truth_delta > winner_delta` on `{int(terminal_failedrank_aggregate.get('truth_delta_gt_winner_count', 0))}` shots.",
                    (
                        f"- Latent-state read: truth is more diffuse than the selected winner on "
                        f"`{int(terminal_failedrank_aggregate.get('truth_more_diffuse_than_winner_count', 0))} / {int(terminal_failedrank_aggregate.get('row_count', 0))}` shots, "
                        f"with mean effective support `{float(terminal_failedrank_aggregate.get('truth_effective_support_mean', float('nan'))):.3f}` for truth versus "
                        f"`{float(terminal_failedrank_aggregate.get('winner_effective_support_mean', float('nan'))):.3f}` for the winner; truth top-state share is lower on "
                        f"`{int(terminal_failedrank_aggregate.get('truth_lower_top_state_share_count', 0))}` shots."
                    ),
                    "",
                    f"![{family.benchmark_title} {family.scope_label} terminal failed-rank diagnostics]({failedrank_fig})",
                    "",
                    (
                        f"Caption. Figure 6. Terminal failed-rank diagnostics for the `truth_present_but_not_selected` subset from the same progressive frontier run on "
                        f"`{family.detector_symbol} = {int(family.matrix_rows)} x {int(family.matrix_cols)}`, `{family.logical_symbol} = {int(family.logical_rows)} x {int(family.matrix_cols)}`, "
                        f"`{int(family.noisy_rounds)}` noisy rounds, and `{int(shots)}` sampled {family.scope_label.lower()} frames at `p={float(p_location):.6g}`. "
                        "Top left: truth ranks under terminal `log_mass`, `best_viterbi`, and post hoc cost-tilted summary scores. "
                        "Top right: truth-class and winner-class merged state counts together with their effective support sizes. "
                        "Bottom left: monotone dominance checks for `M = log_mass`, `V = best_viterbi`, `C = representative_cost`, the combined monotone `(M,V,C)` certificate, and whether `truth_delta = log_mass - best_viterbi` exceeds the winner's value. "
                        "Bottom right: truth and winner `Delta` values together with `Delta_truth - Delta_winner`. "
                        "Main quantitative takeaway: when every failed-rank shot satisfies the monotone `(M,V,C)` certificate, terminal-summary-only rerankers are structurally blocked on that subset; the new effective-support traces show whether the truth class survives mainly as a more diffuse latent-state mixture than the selected winner."
                    ),
                    "",
                    "| shot | bucket | mass rank | viterbi rank | truth eff. supp. | winner eff. supp. | truth top share | `truth_delta - winner_delta` | MVC cert. |",
                    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            display_rows = terminal_failedrank_rows[: min(len(terminal_failedrank_rows), 12)]
            for row in display_rows:
                lines.append(
                    f"| `{int(row['shot'])}` | `{row['truth_rank_bucket']}` | `{int(row['truth_log_mass_rank'])}` | "
                    f"`{int(row['truth_best_viterbi_rank'])}` | `{float(row['truth_effective_support']):.3f}` | "
                    f"`{float(row['winner_effective_support']):.3f}` | `{float(row['truth_top_state_share']):.3f}` | "
                    f"`{float(row['truth_delta_minus_winner']):+.3f}` | `{int(row['monotone_mvc_certificate'])}` |"
                )
            if len(terminal_failedrank_rows) > len(display_rows):
                lines.append(
                    f"- Showing the first `{len(display_rows)}` failed-rank shots above; full details are in `{summary_csv_name}`."
                )
            else:
                lines.append(f"- Full per-shot failed-rank details are in `{summary_csv_name}`.")
            lines.append(f"- Aggregate failed-rank summary JSON: `{aggregate_json_name}`.")
        else:
            lines.append(
                "- No `truth_present_but_not_selected` shots were present in this run, so the failed-rank terminal analysis is vacuous for this corpus."
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- This report is the corrected standard-matrix rerun on `{family.detector_symbol}`, not a non-default reduced-CNOT or projected-location bundle.",
            f"- Because the benchmark is {family.scope_label.lower()} only, every reported FER is already the strict full logical frame error for this one-side task.",
            "- If the failure decomposition is dominated by `syndrome_fail`, the current prefix-mass pruning is still dropping all valid continuations and the next step is a stronger lookahead score, not only larger `K`.",
            "- The new pruning-loss diagnostics are the main certificate lane: small cumulative discarded mass suggests pruning is not removing much exact posterior weight, while a large worst-step discarded fraction points to a specific boundary where the beam cap is still too aggressive.",
            "- Figure 5 is offline-oracle-only and non-comparable for decoder ranking; it is only there to test whether pruning mainly discards wrong logical classes instead of the true one.",
            "- The terminal ranking section is the selector-side certificate lane: all-monotone `(M,V,C)` dominance means terminal-summary-only reranking is exhausted on the failed-rank subset, while mixed `Delta` values point toward richer latent-state instrumentation rather than more post hoc selector swaps.",
            f"- Matrix status: exploratory progressive decoder architecture on the true detector-side DEM {family.scope_label.lower()} benchmark for backend `{family.backend}`.",
        ]
    )

    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Progressive frontier/list-decoding report on a detector-side DEM benchmark."
        ),
        epilog=(
            "Smoke example:\n"
            "  tools/py tools/gross144_dem_x_progressive_report.py --backend bb_72_12_6 "
            "--scope memory_Z --p-location 0.004 --shots 1 --seed 20260418 "
            "--decoder-mode bidirectional_committee --column-order shared_mitm_order "
            "--beam-sizes 64 "
            "--score-modes prefix --bidirectional-splice-rerank --splice-candidate-count 4 "
            "--splice-cut-selector middle --splice-max-cuts 1 --results-dir /tmp/bb72_splice_smoke "
            "--cpus 1 --shards 1 --progress-every-shards 1 --no-write-plots --write-report"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--backend", type=str, default="bravyi_depth7")
    parser.add_argument("--scope", type=str, choices=["memory_X", "memory_Z"], default="memory_X")
    parser.add_argument("--stim-path", type=Path, default=None)
    parser.add_argument("--external-benchmark-label", type=str, default=None)
    parser.add_argument("--external-noisy-rounds", type=int, default=None)
    parser.add_argument("--external-perfect-rounds", type=int, default=1)
    parser.add_argument(
        "--initial-data-error-rate",
        type=float,
        default=None,
        help=(
            "Optional replacement probability for the initial BB data-qubit preparation error layer. "
            "This is a non-default bivariate-bicycle matrix variant; all non-initial locations keep --p-location."
        ),
    )
    parser.add_argument(
        "--correction-state-mode",
        type=str,
        choices=["none", "full", "logical_class", "stabilizer_quotient"],
        default="none",
    )
    parser.add_argument("--require-correction-cache", action="store_true")
    parser.add_argument("--p-location", type=float, default=0.001)
    parser.add_argument("--shots", type=int, default=32)
    parser.add_argument(
        "--shot-start",
        type=int,
        default=0,
        help=(
            "First absolute shot index for the default contiguous corpus. Ignored when "
            "--shot-indices is provided. Use --shot-start N --shots M to run N..N+M-1."
        ),
    )
    parser.add_argument(
        "--shot-indices",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Explicit absolute shot indices to run. When provided, these indices replace the default "
            "contiguous range `0..shots-1` and let you rerun targeted shots from an existing corpus."
        ),
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--beam-sizes", type=int, nargs="*", default=[4, 8, 16, 32])
    parser.add_argument("--column-order", type=str, choices=COLUMN_ORDER_CHOICES, default="deadline_reorder")
    parser.add_argument(
        "--column-order-file",
        type=Path,
        default=None,
        help="Explicit permutation for --column-order custom_file; entries are positions in metadata time order.",
    )
    parser.add_argument(
        "--decoder-mode",
        type=str,
        choices=[
            "forward",
            "backward",
            "bidirectional_committee",
            "bidirectional_middle_join",
            "forward_guided_backward",
        ],
        default="forward",
    )
    parser.add_argument("--backward-column-order", type=str, choices=COLUMN_ORDER_CHOICES, default=None)
    parser.add_argument("--middle-join-prefix-columns", type=int, default=None)
    parser.add_argument("--middle-join-multicut-prefix-columns", type=int, nargs="+", default=None)
    parser.add_argument("--middle-join-multicut-stride", type=int, default=None)
    parser.add_argument("--middle-join-multicut-max-cuts", type=int, default=None)
    parser.add_argument(
        "--middle-join-multicut-weight-mode",
        type=str,
        choices=["uniform", "compatibility", "compatibility_gap"],
        default=None,
    )
    parser.add_argument("--middle-join-cut-window-columns", type=int, default=None)
    parser.add_argument("--middle-join-cut-beam-factor", type=int, default=None)
    parser.add_argument(
        "--bidirectional-splice-rerank",
        action="store_true",
        help="Run an offline bidirectional cut-splice logical reranker and report its selected logical class.",
    )
    parser.add_argument("--splice-candidate-count", type=int, default=DEFAULT_SPLICE_CANDIDATE_COUNT)
    parser.add_argument(
        "--splice-cut-selector",
        type=str,
        choices=[
            "middle",
            "linspace",
            "evenly_spaced",
            "max_cap_pressure",
            "smallest_cutoff_gap",
            "first_truth_pruned",
        ],
        default=DEFAULT_SPLICE_CUT_SELECTOR,
    )
    parser.add_argument("--splice-max-cuts", type=int, default=DEFAULT_SPLICE_MAX_CUTS)
    parser.add_argument(
        "--splice-aggregate",
        type=str,
        choices=["median", "mean", "trimmed_mean"],
        default=DEFAULT_SPLICE_AGGREGATE,
    )
    parser.add_argument(
        "--splice-replace-final-selection",
        action="store_true",
        help="Use the splice-selected logical class as the official report selection.",
    )
    parser.add_argument("--score-modes", type=str, nargs="+", default=["prefix"])
    parser.add_argument("--beam-score-gap-threshold", type=float, default=None)
    parser.add_argument(
        "--beam-score-gap-policy-mode",
        type=str,
        choices=["linear_columns", "active_log", "candidate_log"],
        default=None,
    )
    parser.add_argument("--beam-score-gap-policy-base-threshold", type=float, default=None)
    parser.add_argument("--beam-score-gap-policy-final-threshold", type=float, default=None)
    parser.add_argument("--beam-score-gap-policy-slope", type=float, default=None)
    parser.add_argument("--beam-score-gap-policy-reference-count", type=float, default=None)
    parser.add_argument("--beam-score-gap-policy-min-threshold", type=float, default=None)
    parser.add_argument("--beam-score-gap-policy-max-threshold", type=float, default=None)
    parser.add_argument("--selective-secondary-score-mode", type=str, default="")
    parser.add_argument("--selective-secondary-trigger-gap", type=float, default=0.0)
    parser.add_argument("--selective-secondary-band-size", type=int, default=0)
    parser.add_argument(
        "--selective-local-lookahead-mode",
        type=str,
        choices=progressive.SELECTIVE_LOCAL_LOOKAHEAD_MODES,
        default="none",
    )
    parser.add_argument("--selective-local-lookahead-cutoff-gap-threshold", type=float, default=0.0)
    parser.add_argument("--selective-local-lookahead-near-cut-width", type=float, default=0.0)
    parser.add_argument("--selective-local-lookahead-max-candidates", type=int, default=0)
    parser.add_argument("--selective-local-lookahead-candidate-top1-share-threshold", type=float, default=0.0)
    parser.add_argument("--selective-local-lookahead-support-gap-threshold", type=float, default=float("inf"))
    parser.add_argument("--selective-local-lookahead-overflow-ratio-threshold", type=float, default=float("inf"))
    parser.add_argument("--forward-guidance-weight", type=float, default=1.0)
    parser.add_argument("--forward-guidance-clip", type=float, default=6.0)
    parser.add_argument(
        "--forward-guidance-mode",
        type=str,
        choices=[
            "detector_penalty",
            "conditional_rescue",
            "state_overlap_rescue",
            "checkpoint_exact_rescue",
            "checkpoint_exact_replay_rescue",
            "conditional_widen",
            "conditional_diversity_widen",
        ],
        default="detector_penalty",
        help=(
            "Forward-guided backward message mode. conditional_rescue uses a logical-aware one-sided "
            "shortlist rerank; state_overlap_rescue uses exact projected forward-state overlap for a "
            "small positive shortlist rescue; checkpoint_exact_rescue uses sparse exact same-key "
            "forward checkpoint masses only to rescue a few locally supported near-cutoff states "
            "without changing the ordinary top-K ordering; checkpoint_exact_replay_rescue keeps the "
            "same rescue-only policy but locally replays forward from the latest sparse checkpoint "
            "anchor when the queried near-cutoff separator key is missing at the current cut; "
            "conditional_widen temporarily widens the local beam on informative cuts; "
            "conditional_diversity_widen adds finite positive conditional-support extras, prioritizing "
            "logical classes absent from the normal kept set."
        ),
    )
    parser.add_argument(
        "--forward-guidance-diversity-fallback",
        type=str,
        choices=["none", "missing_logical_base_rank"],
        default="none",
        help=(
            "Extra selector for conditional_diversity_widen. "
            "missing_logical_base_rank spends remaining local-widen slots on at most one "
            "base-ranked provisional candidate from each logical class absent from the ordinary kept set."
        ),
    )
    parser.add_argument("--forward-guidance-widen-factor", type=float, default=2.0)
    parser.add_argument("--forward-guidance-min-info-bits", type=float, default=0.0)
    parser.add_argument(
        "--forward-guidance-snapshot-factor",
        type=float,
        default=1.0,
        help="Multiply only the forward snapshot-building beam width; backward pruning keeps the requested beam size.",
    )
    parser.add_argument(
        "--forward-guidance-snapshot-gap",
        type=float,
        default=None,
        help=(
            "Override the beam score-gap threshold only for the forward snapshot-building pass; "
            "use inf to disable score-gap pruning for that pass."
        ),
    )
    parser.add_argument(
        "--forward-guidance-snapshot-source",
        type=str,
        choices=["kept", "pre_prune"],
        default="kept",
        help=(
            "Choose which forward frontier is aggregated into guidance snapshots. kept uses the "
            "post-prune retained frontier; pre_prune uses the candidate frontier before forward "
            "beam/score-gap truncation at each snapshot boundary."
        ),
    )
    parser.add_argument(
        "--forward-guidance-hamming-radius",
        type=int,
        default=0,
        help="Use conditional support within this Hamming radius on aligned detector rows; 0 is exact lookup.",
    )
    parser.add_argument(
        "--forward-guidance-trigger-mode",
        type=str,
        choices=["top_gap", "support_aware", "top_gap_or_support_aware"],
        default="top_gap",
        help=(
            "Trigger rule for conditional forward guidance. support_aware fires from pool-level "
            "conditional support outside the kept set; top_gap_or_support_aware uses either rule."
        ),
    )
    parser.add_argument(
        "--forward-guidance-nearcut-gap",
        type=float,
        default=0.0,
        help="Base-score band below the kept cutoff used by support-aware trigger counts.",
    )
    parser.add_argument(
        "--forward-guidance-pool-trigger-min-positive-nearcut",
        type=int,
        default=1,
        help="Minimum positive supported outside-kept near-cut candidates needed for the support-aware trigger arm.",
    )
    parser.add_argument(
        "--forward-guidance-trigger-gap",
        type=float,
        default=0.0,
        help="Apply forward guidance only when the backward base top-primary gap is <= this value; 0 disables trigger gating.",
    )
    parser.add_argument("--lookahead-depth", type=int, default=0)
    parser.add_argument("--lookahead-shortlist-size", type=int, default=0)
    parser.add_argument("--delayed-pruning-gap-threshold", type=float, default=0.0)
    parser.add_argument("--delayed-pruning-factor", type=int, default=1)
    parser.add_argument("--pruning-replay-checkpoint-stride", type=int, default=0)
    parser.add_argument("--pruning-replay-horizon", type=int, default=0)
    parser.add_argument("--tail-exact-columns", type=int, default=0)
    parser.add_argument(
        "--superstep-mode",
        type=str,
        choices=["none", "closure_blocks", "disjoint_detector_runs"],
        default="none",
    )
    parser.add_argument("--superstep-path-budget", type=int, default=250000)
    parser.add_argument("--superstep-state-budget", type=int, default=4096)
    parser.add_argument("--superstep-transition-budget", type=int, default=0)
    parser.add_argument("--detector-bucket-pruning", action="store_true")
    parser.add_argument("--detector-bucket-max-logicals", type=int, default=4)
    parser.add_argument("--logical-class-reserve-min-classes", type=int, default=0)
    parser.add_argument("--logical-class-reserve-max-replacements", type=int, default=0)
    parser.add_argument("--logical-class-reserve-min-remaining-columns", type=int, default=0)
    parser.add_argument("--logical-class-quota-top-classes", type=int, default=0)
    parser.add_argument("--logical-class-quota-reserved-slots", type=int, default=0)
    parser.add_argument("--logical-class-quota-min-remaining-columns", type=int, default=0)
    parser.add_argument("--lineage-reserve-checkpoint-stride", type=int, default=0)
    parser.add_argument("--lineage-reserve-reserved-slots", type=int, default=0)
    parser.add_argument("--logical-rerank-columns", type=int, default=0)
    parser.add_argument("--logical-rerank-shortlist-size", type=int, default=0)
    parser.add_argument("--logical-rerank-min-classes", type=int, default=0)
    parser.add_argument("--logical-rerank-state-budget", type=int, default=1024)
    parser.add_argument("--logical-rerank-transition-budget", type=int, default=100000)
    parser.add_argument("--logical-rerank-checkpoint-stride", type=int, default=0)
    parser.add_argument("--logical-rerank-max-passes", type=int, default=1)
    parser.add_argument("--logical-rerank-mode", type=str, default="exact_tail")
    parser.add_argument(
        "--final-logical-select-mode",
        type=str,
        choices=FINAL_LOGICAL_SELECT_MODE_CHOICES,
        default="log_mass",
    )
    parser.add_argument("--final-logical-select-rep-cost-weight", type=float, default=0.0)
    parser.add_argument("--final-logical-select-max-log-mass-gap", type=float, default=float("inf"))
    parser.add_argument("--final-logical-select-rank2-viterbi-tolerance", type=float, default=0.0)
    parser.add_argument("--track-best-path", action="store_true")
    parser.add_argument("--disable-state-merging", action="store_true")
    parser.add_argument(
        "--state-merge-period-columns",
        type=int,
        default=0,
        help=(
            "Experimental delayed-merge mode. Requires --disable-state-merging; duplicate "
            "detector/logical states are kept separate between periodic post-prune collapse "
            "boundaries. A value of 10 merges after every 10 processed columns and at terminal."
        ),
    )
    parser.add_argument(
        "--production-fast-mode",
        action="store_true",
        help=(
            "Run the exact same decoder decision path while suppressing expensive diagnostic "
            "postprocessing by default. Explicit diagnostic flags still re-enable their artifacts."
        ),
    )
    parser.add_argument("--export-state-count-profile", action="store_true")
    parser.add_argument(
        "--export-frontier-pressure-trace",
        action="store_true",
        help="Opt-in pre-prune live-set pressure trace with N_tau/effective-support diagnostics.",
    )
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--write-plots",
        dest="write_plots",
        action="store_true",
        help="Write report plots even when --production-fast-mode is set.",
    )
    plot_group.add_argument(
        "--no-write-plots",
        dest="write_plots",
        action="store_false",
        help="Skip plot generation.",
    )
    report_group = parser.add_mutually_exclusive_group()
    report_group.add_argument(
        "--write-report",
        dest="write_report",
        action="store_true",
        help="Write report.md even when --production-fast-mode is set.",
    )
    report_group.add_argument(
        "--no-write-report",
        dest="write_report",
        action="store_false",
        help="Skip report.md generation.",
    )
    terminal_signal_group = parser.add_mutually_exclusive_group()
    terminal_signal_group.add_argument(
        "--export-terminal-selector-signals",
        dest="export_terminal_selector_signals",
        action="store_true",
        help=(
            "Emit one JSON payload column per shot/config row with richer terminal logical-class signals, "
            "including per-logical terminal state (log_mass, rep_cost) entries and derived selector features. "
            "This is enabled by default."
        ),
    )
    terminal_signal_group.add_argument(
        "--no-export-terminal-selector-signals",
        dest="export_terminal_selector_signals",
        action="store_false",
        help="Disable terminal selector signal export and the automatic failed-rank terminal analysis section.",
    )
    parser.set_defaults(
        export_terminal_selector_signals=None,
        write_plots=None,
        write_report=None,
    )
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--shards", type=int, default=None)
    parser.add_argument("--progress-every-shards", type=int, default=1)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()
    if args.export_terminal_selector_signals is None:
        args.export_terminal_selector_signals = not bool(args.production_fast_mode)
    if args.write_plots is None:
        args.write_plots = not bool(args.production_fast_mode)
    if args.write_report is None:
        args.write_report = not bool(args.production_fast_mode)
    if args.shot_indices is not None:
        unique_shot_indices: list[int] = []
        seen_shot_indices: set[int] = set()
        for value in args.shot_indices:
            shot_index = int(value)
            if shot_index < 0:
                raise ValueError("--shot-indices entries must be >= 0")
            if shot_index in seen_shot_indices:
                continue
            seen_shot_indices.add(int(shot_index))
            unique_shot_indices.append(int(shot_index))
        if not unique_shot_indices:
            raise ValueError("--shot-indices must not be empty")
        args.selected_shot_indices = tuple(int(value) for value in unique_shot_indices)
        args.shots = int(len(args.selected_shot_indices))
    else:
        if int(args.shots) <= 0:
            raise ValueError("--shots must be > 0")
        if int(args.shot_start) < 0:
            raise ValueError("--shot-start must be >= 0")
        args.selected_shot_indices = tuple(
            int(value) for value in range(int(args.shot_start), int(args.shot_start) + int(args.shots))
        )
    if not args.beam_sizes:
        raise ValueError("--beam-sizes must not be empty")
    if any(int(value) <= 0 for value in args.beam_sizes):
        raise ValueError("--beam-sizes entries must be >= 1")
    if not args.score_modes:
        raise ValueError("--score-modes must not be empty")
    args.score_modes = list(dict.fromkeys(str(value) for value in args.score_modes))
    if args.beam_score_gap_threshold is not None:
        if math.isnan(float(args.beam_score_gap_threshold)):
            raise ValueError("--beam-score-gap-threshold must not be NaN")
        if math.isfinite(float(args.beam_score_gap_threshold)) and float(args.beam_score_gap_threshold) < 0.0:
            raise ValueError("--beam-score-gap-threshold must be >= 0")
    args.beam_score_gap_policy = _build_beam_score_gap_policy(
        mode=args.beam_score_gap_policy_mode,
        base_threshold=args.beam_score_gap_policy_base_threshold,
        final_threshold=args.beam_score_gap_policy_final_threshold,
        slope=args.beam_score_gap_policy_slope,
        reference_count=args.beam_score_gap_policy_reference_count,
        min_threshold=args.beam_score_gap_policy_min_threshold,
        max_threshold=args.beam_score_gap_policy_max_threshold,
    )
    if args.beam_score_gap_threshold is not None and args.beam_score_gap_policy is not None:
        raise ValueError("--beam-score-gap-threshold and adaptive score-gap policy are mutually exclusive")
    if math.isnan(float(args.selective_secondary_trigger_gap)):
        raise ValueError("--selective-secondary-trigger-gap must not be NaN")
    if math.isfinite(float(args.selective_secondary_trigger_gap)) and float(args.selective_secondary_trigger_gap) < 0.0:
        raise ValueError("--selective-secondary-trigger-gap must be >= 0")
    if int(args.selective_secondary_band_size) < 0:
        raise ValueError("--selective-secondary-band-size must be >= 0")
    selective_secondary_enabled = _selective_secondary_enabled(
        selective_secondary_score_mode=str(args.selective_secondary_score_mode),
        selective_secondary_trigger_gap=float(args.selective_secondary_trigger_gap),
        selective_secondary_band_size=int(args.selective_secondary_band_size),
    )
    if bool(selective_secondary_enabled) != bool(
        str(args.selective_secondary_score_mode).strip()
        and float(args.selective_secondary_trigger_gap) > 0.0
        and int(args.selective_secondary_band_size) > 0
    ):
        raise ValueError(
            "--selective-secondary-score-mode, --selective-secondary-trigger-gap > 0, and "
            "--selective-secondary-band-size > 0 must be provided together"
        )
    args.selective_local_lookahead_mode = progressive._normalize_selective_local_lookahead_mode(
        str(args.selective_local_lookahead_mode)
    )
    if math.isnan(float(args.selective_local_lookahead_cutoff_gap_threshold)):
        raise ValueError("--selective-local-lookahead-cutoff-gap-threshold must not be NaN")
    if (
        math.isfinite(float(args.selective_local_lookahead_cutoff_gap_threshold))
        and float(args.selective_local_lookahead_cutoff_gap_threshold) < 0.0
    ):
        raise ValueError("--selective-local-lookahead-cutoff-gap-threshold must be >= 0")
    if (
        math.isnan(float(args.selective_local_lookahead_near_cut_width))
        or not math.isfinite(float(args.selective_local_lookahead_near_cut_width))
        or float(args.selective_local_lookahead_near_cut_width) < 0.0
    ):
        raise ValueError("--selective-local-lookahead-near-cut-width must be finite and >= 0")
    if int(args.selective_local_lookahead_max_candidates) < 0:
        raise ValueError("--selective-local-lookahead-max-candidates must be >= 0")
    if (
        str(args.selective_local_lookahead_mode) != "none"
        and int(args.selective_local_lookahead_max_candidates) <= 0
    ):
        raise ValueError("--selective-local-lookahead-max-candidates must be > 0 when mode is not none")
    if (
        math.isnan(float(args.selective_local_lookahead_candidate_top1_share_threshold))
        or not math.isfinite(float(args.selective_local_lookahead_candidate_top1_share_threshold))
        or float(args.selective_local_lookahead_candidate_top1_share_threshold) < 0.0
        or float(args.selective_local_lookahead_candidate_top1_share_threshold) > 1.0
    ):
        raise ValueError("--selective-local-lookahead-candidate-top1-share-threshold must be in [0, 1]")
    if (
        math.isnan(float(args.selective_local_lookahead_support_gap_threshold))
        or float(args.selective_local_lookahead_support_gap_threshold) < 0.0
    ):
        raise ValueError("--selective-local-lookahead-support-gap-threshold must be >= 0 when finite")
    if (
        math.isnan(float(args.selective_local_lookahead_overflow_ratio_threshold))
        or float(args.selective_local_lookahead_overflow_ratio_threshold) < 0.0
    ):
        raise ValueError("--selective-local-lookahead-overflow-ratio-threshold must be >= 0 when finite")
    if math.isnan(float(args.forward_guidance_weight)):
        raise ValueError("--forward-guidance-weight must not be NaN")
    if not math.isfinite(float(args.forward_guidance_weight)) or float(args.forward_guidance_weight) < 0.0:
        raise ValueError("--forward-guidance-weight must be finite and >= 0")
    if math.isnan(float(args.forward_guidance_clip)):
        raise ValueError("--forward-guidance-clip must not be NaN")
    if not math.isfinite(float(args.forward_guidance_clip)) or float(args.forward_guidance_clip) < 0.0:
        raise ValueError("--forward-guidance-clip must be finite and >= 0")
    if math.isnan(float(args.forward_guidance_trigger_gap)):
        raise ValueError("--forward-guidance-trigger-gap must not be NaN")
    if not math.isfinite(float(args.forward_guidance_trigger_gap)) or float(args.forward_guidance_trigger_gap) < 0.0:
        raise ValueError("--forward-guidance-trigger-gap must be finite and >= 0")
    if math.isnan(float(args.forward_guidance_widen_factor)):
        raise ValueError("--forward-guidance-widen-factor must not be NaN")
    if not math.isfinite(float(args.forward_guidance_widen_factor)) or float(args.forward_guidance_widen_factor) < 1.0:
        raise ValueError("--forward-guidance-widen-factor must be finite and >= 1")
    if math.isnan(float(args.forward_guidance_min_info_bits)):
        raise ValueError("--forward-guidance-min-info-bits must not be NaN")
    if not math.isfinite(float(args.forward_guidance_min_info_bits)) or float(args.forward_guidance_min_info_bits) < 0.0:
        raise ValueError("--forward-guidance-min-info-bits must be finite and >= 0")
    if math.isnan(float(args.forward_guidance_snapshot_factor)):
        raise ValueError("--forward-guidance-snapshot-factor must not be NaN")
    if not math.isfinite(float(args.forward_guidance_snapshot_factor)) or float(args.forward_guidance_snapshot_factor) < 1.0:
        raise ValueError("--forward-guidance-snapshot-factor must be finite and >= 1")
    if args.forward_guidance_snapshot_gap is not None:
        if math.isnan(float(args.forward_guidance_snapshot_gap)):
            raise ValueError("--forward-guidance-snapshot-gap must not be NaN")
        if math.isfinite(float(args.forward_guidance_snapshot_gap)) and float(args.forward_guidance_snapshot_gap) < 0.0:
            raise ValueError("--forward-guidance-snapshot-gap must be >= 0 when finite")
    args.forward_guidance_snapshot_source = progressive._normalize_forward_guidance_snapshot_source(
        str(args.forward_guidance_snapshot_source)
    )
    if int(args.forward_guidance_hamming_radius) < 0 or int(args.forward_guidance_hamming_radius) > 2:
        raise ValueError("--forward-guidance-hamming-radius must be in [0, 2]")
    if math.isnan(float(args.forward_guidance_nearcut_gap)):
        raise ValueError("--forward-guidance-nearcut-gap must not be NaN")
    if not math.isfinite(float(args.forward_guidance_nearcut_gap)) or float(args.forward_guidance_nearcut_gap) < 0.0:
        raise ValueError("--forward-guidance-nearcut-gap must be finite and >= 0")
    if int(args.forward_guidance_pool_trigger_min_positive_nearcut) < 0:
        raise ValueError("--forward-guidance-pool-trigger-min-positive-nearcut must be >= 0")
    args.forward_guidance_diversity_fallback = progressive._normalize_forward_guidance_diversity_fallback(
        str(args.forward_guidance_diversity_fallback)
    )
    if int(args.lookahead_depth) < 0:
        raise ValueError("--lookahead-depth must be >= 0")
    if int(args.lookahead_shortlist_size) < 0:
        raise ValueError("--lookahead-shortlist-size must be >= 0")
    if math.isnan(float(args.delayed_pruning_gap_threshold)):
        raise ValueError("--delayed-pruning-gap-threshold must not be NaN")
    if math.isfinite(float(args.delayed_pruning_gap_threshold)) and float(args.delayed_pruning_gap_threshold) < 0.0:
        raise ValueError("--delayed-pruning-gap-threshold must be >= 0")
    if (
        args.beam_score_gap_threshold is not None
        and math.isfinite(float(args.beam_score_gap_threshold))
        and float(args.beam_score_gap_threshold) > 0.0
        and float(args.delayed_pruning_gap_threshold) > 0.0
        and int(args.delayed_pruning_factor) > 1
    ):
        raise ValueError("--beam-score-gap-threshold is not supported with delayed pruning")
    if int(args.delayed_pruning_factor) < 1:
        raise ValueError("--delayed-pruning-factor must be >= 1")
    if int(args.pruning_replay_checkpoint_stride) < 0:
        raise ValueError("--pruning-replay-checkpoint-stride must be >= 0")
    if int(args.pruning_replay_horizon) < 0:
        raise ValueError("--pruning-replay-horizon must be >= 0")
    if bool(int(args.pruning_replay_checkpoint_stride) > 0) != bool(int(args.pruning_replay_horizon) > 0):
        raise ValueError("--pruning-replay-checkpoint-stride and --pruning-replay-horizon must both be > 0 to enable replay")
    if int(args.tail_exact_columns) < 0:
        raise ValueError("--tail-exact-columns must be >= 0")
    if int(args.superstep_path_budget) < 0:
        raise ValueError("--superstep-path-budget must be >= 0")
    if int(args.superstep_state_budget) < 0:
        raise ValueError("--superstep-state-budget must be >= 0")
    if int(args.superstep_transition_budget) < 0:
        raise ValueError("--superstep-transition-budget must be >= 0")
    if int(args.detector_bucket_max_logicals) < 0:
        raise ValueError("--detector-bucket-max-logicals must be >= 0")
    if int(args.logical_class_reserve_min_classes) < 0:
        raise ValueError("--logical-class-reserve-min-classes must be >= 0")
    if int(args.logical_class_reserve_max_replacements) < 0:
        raise ValueError("--logical-class-reserve-max-replacements must be >= 0")
    if int(args.logical_class_reserve_min_remaining_columns) < 0:
        raise ValueError("--logical-class-reserve-min-remaining-columns must be >= 0")
    if int(args.logical_class_quota_top_classes) < 0:
        raise ValueError("--logical-class-quota-top-classes must be >= 0")
    if int(args.logical_class_quota_reserved_slots) < 0:
        raise ValueError("--logical-class-quota-reserved-slots must be >= 0")
    if int(args.logical_class_quota_min_remaining_columns) < 0:
        raise ValueError("--logical-class-quota-min-remaining-columns must be >= 0")
    if int(args.lineage_reserve_checkpoint_stride) < 0:
        raise ValueError("--lineage-reserve-checkpoint-stride must be >= 0")
    if int(args.lineage_reserve_reserved_slots) < 0:
        raise ValueError("--lineage-reserve-reserved-slots must be >= 0")
    if bool(int(args.lineage_reserve_checkpoint_stride) > 0) != bool(int(args.lineage_reserve_reserved_slots) > 0):
        raise ValueError("--lineage-reserve-checkpoint-stride and --lineage-reserve-reserved-slots must both be > 0 to enable lineage reserve")
    if int(args.logical_rerank_columns) < 0:
        raise ValueError("--logical-rerank-columns must be >= 0")
    if int(args.logical_rerank_shortlist_size) < 0:
        raise ValueError("--logical-rerank-shortlist-size must be >= 0")
    if int(args.logical_rerank_min_classes) < 0:
        raise ValueError("--logical-rerank-min-classes must be >= 0")
    if int(args.logical_rerank_state_budget) < 0:
        raise ValueError("--logical-rerank-state-budget must be >= 0")
    if int(args.logical_rerank_transition_budget) < 0:
        raise ValueError("--logical-rerank-transition-budget must be >= 0")
    if int(args.logical_rerank_checkpoint_stride) < 0:
        raise ValueError("--logical-rerank-checkpoint-stride must be >= 0")
    if int(args.logical_rerank_max_passes) < 0:
        raise ValueError("--logical-rerank-max-passes must be >= 0")
    if str(args.logical_rerank_mode) not in {"exact_tail", "exact_tail_vector", "local_cone"}:
        raise ValueError("--logical-rerank-mode must be one of ['exact_tail', 'exact_tail_vector', 'local_cone']")
    args.final_logical_select_mode = str(args.final_logical_select_mode).strip().lower()
    if str(args.final_logical_select_mode) not in FINAL_LOGICAL_SELECT_MODE_CHOICES:
        raise ValueError(
            "--final-logical-select-mode must be one of "
            f"{list(FINAL_LOGICAL_SELECT_MODE_CHOICES)}"
        )
    if math.isnan(float(args.final_logical_select_rep_cost_weight)):
        raise ValueError("--final-logical-select-rep-cost-weight must not be NaN")
    if not math.isfinite(float(args.final_logical_select_rep_cost_weight)):
        raise ValueError("--final-logical-select-rep-cost-weight must be finite")
    if float(args.final_logical_select_rep_cost_weight) < 0.0:
        raise ValueError("--final-logical-select-rep-cost-weight must be >= 0")
    if math.isnan(float(args.final_logical_select_max_log_mass_gap)):
        raise ValueError("--final-logical-select-max-log-mass-gap must not be NaN")
    if math.isfinite(float(args.final_logical_select_max_log_mass_gap)) and float(args.final_logical_select_max_log_mass_gap) < 0.0:
        raise ValueError("--final-logical-select-max-log-mass-gap must be >= 0 when finite")
    if math.isnan(float(args.final_logical_select_rank2_viterbi_tolerance)):
        raise ValueError("--final-logical-select-rank2-viterbi-tolerance must not be NaN")
    if (
        not math.isfinite(float(args.final_logical_select_rank2_viterbi_tolerance))
        or float(args.final_logical_select_rank2_viterbi_tolerance) < 0.0
    ):
        raise ValueError("--final-logical-select-rank2-viterbi-tolerance must be finite and >= 0")
    if int(args.cpus) <= 0:
        raise ValueError("--cpus must be >= 1")
    if args.shards is None:
        args.shards = min(int(args.shots), max(int(args.cpus), int(args.cpus) * 8))
    if int(args.shards) <= 0:
        raise ValueError("--shards must be >= 1")
    if int(args.progress_every_shards) <= 0:
        raise ValueError("--progress-every-shards must be >= 1")
    if args.backward_column_order is not None and _normalize_decoder_mode(str(args.decoder_mode)) not in {
        "backward",
        "bidirectional_committee",
        "forward_guided_backward",
        "bidirectional_middle_join",
    }:
        raise ValueError("--backward-column-order requires a backward-capable decoder mode")
    if args.middle_join_prefix_columns is not None and int(args.middle_join_prefix_columns) <= 0:
        raise ValueError("--middle-join-prefix-columns must be >= 1 when provided")
    if args.middle_join_multicut_prefix_columns is not None:
        if not args.middle_join_multicut_prefix_columns:
            raise ValueError("--middle-join-multicut-prefix-columns must not be empty when provided")
        if any(int(value) <= 0 for value in args.middle_join_multicut_prefix_columns):
            raise ValueError("--middle-join-multicut-prefix-columns entries must be >= 1")
    if args.middle_join_multicut_stride is not None and int(args.middle_join_multicut_stride) < 0:
        raise ValueError("--middle-join-multicut-stride must be >= 0 when provided")
    if args.middle_join_multicut_max_cuts is not None and int(args.middle_join_multicut_max_cuts) < 0:
        raise ValueError("--middle-join-multicut-max-cuts must be >= 0 when provided")
    if args.middle_join_multicut_weight_mode is not None:
        args.middle_join_multicut_weight_mode = progressive._normalize_progressive_middle_join_multicut_weight_mode(
            str(args.middle_join_multicut_weight_mode)
        )
    if args.middle_join_cut_window_columns is not None and int(args.middle_join_cut_window_columns) < 0:
        raise ValueError("--middle-join-cut-window-columns must be >= 0 when provided")
    if args.middle_join_cut_beam_factor is not None and int(args.middle_join_cut_beam_factor) < 1:
        raise ValueError("--middle-join-cut-beam-factor must be >= 1 when provided")
    if (
        args.middle_join_multicut_prefix_columns is not None
        and _normalize_decoder_mode(str(args.decoder_mode)) != "bidirectional_middle_join"
    ):
        raise ValueError("--middle-join-multicut-prefix-columns requires --decoder-mode bidirectional_middle_join")
    if (
        args.middle_join_multicut_stride is not None
        and _normalize_decoder_mode(str(args.decoder_mode)) != "bidirectional_middle_join"
    ):
        raise ValueError("--middle-join-multicut-stride requires --decoder-mode bidirectional_middle_join")
    if (
        args.middle_join_multicut_max_cuts is not None
        and _normalize_decoder_mode(str(args.decoder_mode)) != "bidirectional_middle_join"
    ):
        raise ValueError("--middle-join-multicut-max-cuts requires --decoder-mode bidirectional_middle_join")
    if (
        args.middle_join_multicut_weight_mode is not None
        and _normalize_decoder_mode(str(args.decoder_mode)) != "bidirectional_middle_join"
    ):
        raise ValueError("--middle-join-multicut-weight-mode requires --decoder-mode bidirectional_middle_join")
    if args.middle_join_cut_window_columns is not None and _normalize_decoder_mode(str(args.decoder_mode)) != "bidirectional_middle_join":
        raise ValueError("--middle-join-cut-window-columns requires --decoder-mode bidirectional_middle_join")
    if args.middle_join_cut_beam_factor is not None and _normalize_decoder_mode(str(args.decoder_mode)) != "bidirectional_middle_join":
        raise ValueError("--middle-join-cut-beam-factor requires --decoder-mode bidirectional_middle_join")
    if (
        args.middle_join_cut_beam_factor is not None
        and int(args.middle_join_cut_beam_factor) > 1
        and (args.middle_join_cut_window_columns is None or int(args.middle_join_cut_window_columns) <= 0)
    ):
        raise ValueError(
            "--middle-join-cut-beam-factor > 1 requires --middle-join-cut-window-columns > 0"
        )
    requested_middle_join_multicut = bool(args.middle_join_multicut_prefix_columns) or (
        args.middle_join_multicut_stride is not None and int(args.middle_join_multicut_stride) > 0
    ) or (
        args.middle_join_multicut_max_cuts is not None and int(args.middle_join_multicut_max_cuts) > 1
    )
    if requested_middle_join_multicut and (
        (args.middle_join_cut_window_columns is not None and int(args.middle_join_cut_window_columns) > 0)
        or (args.middle_join_cut_beam_factor is not None and int(args.middle_join_cut_beam_factor) > 1)
    ):
        raise ValueError(
            "middle-join multicut mode does not support cut-window beam widening"
        )
    if int(args.splice_candidate_count) <= 0:
        raise ValueError("--splice-candidate-count must be >= 1")
    if int(args.splice_max_cuts) <= 0:
        raise ValueError("--splice-max-cuts must be >= 1")
    args.splice_aggregate = progressive._normalize_splice_aggregate_mode(str(args.splice_aggregate))
    if bool(args.splice_replace_final_selection) and not bool(args.bidirectional_splice_rerank):
        raise ValueError("--splice-replace-final-selection requires --bidirectional-splice-rerank")
    if bool(args.require_correction_cache) and str(args.correction_state_mode) == "none":
        raise ValueError("--require-correction-cache requires --correction-state-mode != none")
    if str(args.column_order) == "custom_file" and args.column_order_file is None:
        raise ValueError("--column-order custom_file requires --column-order-file")
    if str(args.column_order) != "custom_file" and args.column_order_file is not None:
        raise ValueError("--column-order-file is only valid with --column-order custom_file")
    if args.stim_path is not None:
        if int(args.external_perfect_rounds) < 0:
            raise ValueError("--external-perfect-rounds must be >= 0")
        if args.external_noisy_rounds is None or int(args.external_noisy_rounds) <= 0:
            raise ValueError("--stim-path requires --external-noisy-rounds > 0")
        if args.initial_data_error_rate is not None:
            raise ValueError("--initial-data-error-rate is only valid when the BB Stim path is built by backend")
    elif args.external_noisy_rounds is not None or int(args.external_perfect_rounds) != 1 or args.external_benchmark_label is not None:
        raise ValueError(
            "--external-benchmark-label, --external-noisy-rounds, and --external-perfect-rounds require --stim-path"
        )
    return args


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.results_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    decoder_mode = _normalize_decoder_mode(str(args.decoder_mode))
    start = time.time()
    backward_family: LoadedProgressiveFamily | None = None
    backward_column_order = _default_backward_column_order_label(str(decoder_mode))
    effective_middle_join_prefix_columns: int | None = (
        None if args.middle_join_prefix_columns is None else int(args.middle_join_prefix_columns)
    )
    effective_middle_join_cut_window_columns = (
        int(args.middle_join_cut_window_columns)
        if args.middle_join_cut_window_columns is not None
        else int(DEFAULT_MIDDLE_JOIN_CUT_WINDOW_COLUMNS)
    )
    effective_middle_join_cut_beam_factor = (
        int(args.middle_join_cut_beam_factor)
        if args.middle_join_cut_beam_factor is not None
        else int(DEFAULT_MIDDLE_JOIN_CUT_BEAM_FACTOR)
    )
    effective_middle_join_multicut_prefix_columns = tuple(
        int(value) for value in (args.middle_join_multicut_prefix_columns or tuple())
    )
    effective_middle_join_multicut_stride = (
        int(args.middle_join_multicut_stride)
        if args.middle_join_multicut_stride is not None
        else int(DEFAULT_MIDDLE_JOIN_MULTICUT_STRIDE)
    )
    effective_middle_join_multicut_max_cuts = (
        int(args.middle_join_multicut_max_cuts)
        if args.middle_join_multicut_max_cuts is not None
        else int(DEFAULT_MIDDLE_JOIN_MULTICUT_MAX_CUTS)
    )
    effective_middle_join_multicut_weight_mode = (
        str(args.middle_join_multicut_weight_mode)
        if args.middle_join_multicut_weight_mode is not None
        else str(DEFAULT_MIDDLE_JOIN_MULTICUT_WEIGHT_MODE)
    )
    joint_middle_join_summary: JointMiddleJoinOrderedFamilies | None = None
    if str(args.column_order) == "midpoint_joint_reorder":
        if str(decoder_mode) != "bidirectional_middle_join":
            raise ValueError("--column-order midpoint_joint_reorder currently requires --decoder-mode bidirectional_middle_join")
        if args.backward_column_order is not None:
            raise ValueError("--backward-column-order is not supported with --column-order midpoint_joint_reorder")
        base_family = _load_dem_family(
            backend=str(args.backend),
            p_location=float(args.p_location),
            initial_data_error_rate=args.initial_data_error_rate,
            scope=str(args.scope),
            correction_state_mode=str(args.correction_state_mode),
            require_correction_cache=bool(args.require_correction_cache),
            column_order="time_order",
            stim_path=(None if args.stim_path is None else Path(args.stim_path)),
            external_benchmark_label=(None if args.external_benchmark_label is None else str(args.external_benchmark_label)),
            external_noisy_rounds=(None if args.external_noisy_rounds is None else int(args.external_noisy_rounds)),
            external_perfect_rounds=int(args.external_perfect_rounds),
        )
        joint_middle_join_summary = _build_joint_middle_join_ordered_families(
            base_family=base_family,
            middle_join_prefix_columns=effective_middle_join_prefix_columns,
        )
        family = joint_middle_join_summary.forward_family
        backward_family = joint_middle_join_summary.backward_family
        backward_column_order = str(backward_family.column_order_name)
        effective_middle_join_prefix_columns = int(joint_middle_join_summary.prefix_columns)
    elif str(args.column_order) == "midpoint_backward_reorder":
        if str(decoder_mode) != "bidirectional_middle_join":
            raise ValueError("--column-order midpoint_backward_reorder currently requires --decoder-mode bidirectional_middle_join")
        if args.backward_column_order is not None:
            raise ValueError("--backward-column-order is not supported with --column-order midpoint_backward_reorder")
        base_family = _load_dem_family(
            backend=str(args.backend),
            p_location=float(args.p_location),
            initial_data_error_rate=args.initial_data_error_rate,
            scope=str(args.scope),
            correction_state_mode=str(args.correction_state_mode),
            require_correction_cache=bool(args.require_correction_cache),
            column_order="deadline_reorder",
            stim_path=(None if args.stim_path is None else Path(args.stim_path)),
            external_benchmark_label=(None if args.external_benchmark_label is None else str(args.external_benchmark_label)),
            external_noisy_rounds=(None if args.external_noisy_rounds is None else int(args.external_noisy_rounds)),
            external_perfect_rounds=int(args.external_perfect_rounds),
        )
        joint_middle_join_summary = _build_forward_anchored_middle_join_ordered_families(
            base_family=base_family,
            middle_join_prefix_columns=effective_middle_join_prefix_columns,
        )
        family = joint_middle_join_summary.forward_family
        backward_family = joint_middle_join_summary.backward_family
        backward_column_order = str(backward_family.column_order_name)
        effective_middle_join_prefix_columns = int(joint_middle_join_summary.prefix_columns)
    elif str(args.column_order) == "shared_mitm_order":
        if args.backward_column_order is not None:
            raise ValueError("--backward-column-order is not supported with --column-order shared_mitm_order")
        base_family = _load_dem_family(
            backend=str(args.backend),
            p_location=float(args.p_location),
            initial_data_error_rate=args.initial_data_error_rate,
            scope=str(args.scope),
            correction_state_mode=str(args.correction_state_mode),
            require_correction_cache=bool(args.require_correction_cache),
            column_order="deadline_reorder",
            stim_path=(None if args.stim_path is None else Path(args.stim_path)),
            external_benchmark_label=(None if args.external_benchmark_label is None else str(args.external_benchmark_label)),
            external_noisy_rounds=(None if args.external_noisy_rounds is None else int(args.external_noisy_rounds)),
            external_perfect_rounds=int(args.external_perfect_rounds),
        )
        joint_middle_join_summary = _build_shared_middle_join_ordered_families(
            base_family=base_family,
            middle_join_prefix_columns=effective_middle_join_prefix_columns,
        )
        family = joint_middle_join_summary.forward_family
        if decoder_mode in {"backward", "bidirectional_committee", "forward_guided_backward", "bidirectional_middle_join"}:
            backward_family = joint_middle_join_summary.backward_family
            backward_column_order = str(backward_family.column_order_name)
        else:
            backward_column_order = ""
        effective_middle_join_prefix_columns = int(joint_middle_join_summary.prefix_columns)
    elif str(args.column_order) in set(BACKWARD_DERIVED_COLUMN_ORDER_CHOICES):
        raise ValueError(f"--column-order {str(args.column_order)} is only supported via --backward-column-order")
    else:
        family = _load_dem_family(
            backend=str(args.backend),
            p_location=float(args.p_location),
            initial_data_error_rate=args.initial_data_error_rate,
            scope=str(args.scope),
            correction_state_mode=str(args.correction_state_mode),
            require_correction_cache=bool(args.require_correction_cache),
            column_order=str(args.column_order),
            column_order_file=(None if args.column_order_file is None else Path(args.column_order_file)),
            stim_path=(None if args.stim_path is None else Path(args.stim_path)),
            external_benchmark_label=(None if args.external_benchmark_label is None else str(args.external_benchmark_label)),
            external_noisy_rounds=(None if args.external_noisy_rounds is None else int(args.external_noisy_rounds)),
            external_perfect_rounds=int(args.external_perfect_rounds),
        )
    load_elapsed = time.time() - start
    print(
        f"[setup] loaded DEM {family.scope_label} family in {load_elapsed:.2f}s; "
        f"matrix={family.matrix_rows}x{family.matrix_cols} logical={family.logical_rows}x{family.matrix_cols} "
        f"initial_data_error_rate={'' if args.initial_data_error_rate is None else float(args.initial_data_error_rate)} "
        f"correction_state_mode={family.correction_state_mode} correction_state_bits={family.correction_state_bits} "
        f"state_merge_mode={_state_merge_mode_label(merge_duplicate_states=not bool(args.disable_state_merging), state_merge_period_columns=int(args.state_merge_period_columns))}",
        flush=True,
    )
    print(
        f"[setup] column_order={family.column_order_name} source={family.column_order_source} "
        f"frontier_max_active_detectors={family.layout.max_active_detectors}",
        flush=True,
    )
    if effective_middle_join_prefix_columns is not None:
        if int(effective_middle_join_prefix_columns) >= len(family.columns):
            raise ValueError("--middle-join-prefix-columns must be <= len(columns) - 1")
    if joint_middle_join_summary is None and decoder_mode in {"backward", "bidirectional_committee", "forward_guided_backward", "bidirectional_middle_join"} and args.backward_column_order is not None:
        if str(args.backward_column_order) == "midpoint_backward_reorder":
            joint_middle_join_summary = _build_forward_anchored_middle_join_ordered_families(
                base_family=family,
                middle_join_prefix_columns=effective_middle_join_prefix_columns,
            )
            backward_family = joint_middle_join_summary.backward_family
            backward_column_order = str(backward_family.column_order_name)
            effective_middle_join_prefix_columns = int(joint_middle_join_summary.prefix_columns)
        elif str(args.backward_column_order) in {"bwd_deadline", "backward_deadline_reorder"}:
            backward_family = _build_backward_deadline_ordered_family(
                base_family=family,
            )
            backward_column_order = str(backward_family.column_order_name)
        elif str(args.backward_column_order) in {"back_deadline_min_active_w32", "back_deadline_close_first_w32"}:
            backward_family = _build_backward_pressure_ordered_family(
                base_family=family,
                column_order=str(args.backward_column_order),
            )
            backward_column_order = str(backward_family.column_order_name)
        else:
            backward_family = _load_dem_family(
                backend=str(args.backend),
                p_location=float(args.p_location),
                initial_data_error_rate=args.initial_data_error_rate,
                scope=str(args.scope),
                correction_state_mode=str(args.correction_state_mode),
                require_correction_cache=bool(args.require_correction_cache),
                column_order=str(args.backward_column_order),
                stim_path=(None if args.stim_path is None else Path(args.stim_path)),
                external_benchmark_label=(None if args.external_benchmark_label is None else str(args.external_benchmark_label)),
                external_noisy_rounds=(None if args.external_noisy_rounds is None else int(args.external_noisy_rounds)),
                external_perfect_rounds=int(args.external_perfect_rounds),
            )
            backward_column_order = str(backward_family.column_order_name)
        print(
            f"[setup] decoder_mode={decoder_mode} backward_column_order={backward_family.column_order_name} "
            f"backward_source={backward_family.column_order_source} "
            f"backward_frontier_max_active_detectors={backward_family.layout.max_active_detectors}",
            flush=True,
        )
    elif joint_middle_join_summary is not None:
        if backward_family is not None:
            print(
                f"[setup] decoder_mode={decoder_mode} backward_column_order={backward_family.column_order_name} "
                f"backward_source={backward_family.column_order_source} "
                f"backward_frontier_max_active_detectors={backward_family.layout.max_active_detectors}",
                flush=True,
            )
        else:
            print(
                f"[setup] decoder_mode={decoder_mode} backward_column_order={backward_column_order or 'n/a'}",
                flush=True,
            )
        print(
            f"[setup] {str(args.column_order)} prefix={int(joint_middle_join_summary.prefix_columns)} "
            f"suffix={int(joint_middle_join_summary.suffix_columns)} "
            f"forward_prefix_active_area={int(joint_middle_join_summary.forward_prefix_active_area)} "
            f"backward_prefix_active_area={int(joint_middle_join_summary.backward_prefix_active_area)} "
            f"cut_boundary_rows={int(joint_middle_join_summary.cut_boundary_rows)} "
            f"middle_join_cut_window_columns={int(effective_middle_join_cut_window_columns)} "
            f"middle_join_cut_beam_factor={int(effective_middle_join_cut_beam_factor)}",
            flush=True,
        )
    else:
        print(
            f"[setup] decoder_mode={decoder_mode} backward_column_order={backward_column_order or 'n/a'}",
            flush=True,
        )
    if decoder_mode == "bidirectional_middle_join":
        print(
            f"[setup] middle_join prefix={effective_middle_join_prefix_columns if effective_middle_join_prefix_columns is not None else 'auto'} "
            f"multicut_prefixes={list(effective_middle_join_multicut_prefix_columns)} "
            f"multicut_stride={int(effective_middle_join_multicut_stride)} "
            f"multicut_max_cuts={int(effective_middle_join_multicut_max_cuts)} "
            f"multicut_weight_mode={str(effective_middle_join_multicut_weight_mode)} "
            f"cut_window_columns={int(effective_middle_join_cut_window_columns)} "
            f"cut_beam_factor={int(effective_middle_join_cut_beam_factor)}",
            flush=True,
        )
    print(
        f"[setup] score_gap_control={_format_beam_score_gap_control(beam_score_gap_threshold=(None if args.beam_score_gap_threshold is None else float(args.beam_score_gap_threshold)), beam_score_gap_policy=args.beam_score_gap_policy)}",
        flush=True,
    )
    print(
        f"[setup] selective_secondary={_format_selective_secondary_control(selective_secondary_score_mode=str(args.selective_secondary_score_mode), selective_secondary_trigger_gap=float(args.selective_secondary_trigger_gap), selective_secondary_band_size=int(args.selective_secondary_band_size)) or 'disabled'}",
        flush=True,
    )
    print(
        f"[setup] selective_local_lookahead={_format_selective_local_lookahead_control(mode=str(args.selective_local_lookahead_mode), cutoff_gap_threshold=float(args.selective_local_lookahead_cutoff_gap_threshold), near_cut_width=float(args.selective_local_lookahead_near_cut_width), max_candidates=int(args.selective_local_lookahead_max_candidates)) or 'disabled'}",
        flush=True,
    )
    print(
        f"[setup] forward_guidance=mode:{str(args.forward_guidance_mode)} "
        f"weight:{float(args.forward_guidance_weight):g} "
        f"clip:{float(args.forward_guidance_clip):g} "
        f"trigger_gap:{float(args.forward_guidance_trigger_gap):g} "
        f"widen_factor:{float(args.forward_guidance_widen_factor):g} "
        f"min_info_bits:{float(args.forward_guidance_min_info_bits):g} "
        f"snapshot_factor:{float(args.forward_guidance_snapshot_factor):g} "
        f"snapshot_gap:{_format_forward_guidance_snapshot_gap(args.forward_guidance_snapshot_gap) or 'inherit'} "
        f"snapshot_source:{str(args.forward_guidance_snapshot_source)} "
        f"hamming_radius:{int(args.forward_guidance_hamming_radius)} "
        f"trigger_mode:{str(args.forward_guidance_trigger_mode)} "
        f"nearcut_gap:{float(args.forward_guidance_nearcut_gap):g} "
        f"pool_min_pos_nearcut:{int(args.forward_guidance_pool_trigger_min_positive_nearcut)} "
        f"diversity_fallback:{str(args.forward_guidance_diversity_fallback)}",
        flush=True,
    )

    global _GLOBAL_FAMILY, _GLOBAL_SAMPLE_COLUMNS, _GLOBAL_SAMPLE_PRIORS, _GLOBAL_SEED, _GLOBAL_LOOKAHEAD_DEPTH, _GLOBAL_LOOKAHEAD_SHORTLIST_SIZE, _GLOBAL_DELAYED_PRUNING_GAP_THRESHOLD, _GLOBAL_DELAYED_PRUNING_FACTOR, _GLOBAL_PRUNING_REPLAY_CHECKPOINT_STRIDE, _GLOBAL_PRUNING_REPLAY_HORIZON, _GLOBAL_TAIL_EXACT_COLUMNS, _GLOBAL_SUPERSTEP_MODE, _GLOBAL_SUPERSTEP_PATH_BUDGET, _GLOBAL_SUPERSTEP_STATE_BUDGET, _GLOBAL_SUPERSTEP_TRANSITION_BUDGET, _GLOBAL_DETECTOR_BUCKET_PRUNING, _GLOBAL_DETECTOR_BUCKET_MAX_LOGICALS, _GLOBAL_LOGICAL_CLASS_RESERVE_MIN_CLASSES, _GLOBAL_LOGICAL_CLASS_RESERVE_MAX_REPLACEMENTS, _GLOBAL_LOGICAL_CLASS_RESERVE_MIN_REMAINING_COLUMNS, _GLOBAL_LOGICAL_CLASS_QUOTA_TOP_CLASSES, _GLOBAL_LOGICAL_CLASS_QUOTA_RESERVED_SLOTS, _GLOBAL_LOGICAL_CLASS_QUOTA_MIN_REMAINING_COLUMNS, _GLOBAL_LINEAGE_RESERVE_CHECKPOINT_STRIDE, _GLOBAL_LINEAGE_RESERVE_RESERVED_SLOTS, _GLOBAL_LOGICAL_RERANK_COLUMNS, _GLOBAL_LOGICAL_RERANK_SHORTLIST_SIZE, _GLOBAL_LOGICAL_RERANK_MIN_CLASSES, _GLOBAL_LOGICAL_RERANK_STATE_BUDGET, _GLOBAL_LOGICAL_RERANK_TRANSITION_BUDGET, _GLOBAL_LOGICAL_RERANK_CHECKPOINT_STRIDE, _GLOBAL_LOGICAL_RERANK_MAX_PASSES, _GLOBAL_LOGICAL_RERANK_MODE, _GLOBAL_FINAL_LOGICAL_SELECT_MODE, _GLOBAL_FINAL_LOGICAL_SELECT_REP_COST_WEIGHT, _GLOBAL_FINAL_LOGICAL_SELECT_MAX_LOG_MASS_GAP, _GLOBAL_FINAL_LOGICAL_SELECT_RANK2_VITERBI_TOLERANCE, _GLOBAL_TRACK_BEST_PATH, _GLOBAL_MERGE_DUPLICATE_STATES, _GLOBAL_STATE_MERGE_PERIOD_COLUMNS, _GLOBAL_SCORE_MODES, _GLOBAL_BEAM_SCORE_GAP_THRESHOLD, _GLOBAL_BEAM_SCORE_GAP_POLICY, _GLOBAL_SELECTIVE_SECONDARY_SCORE_MODE, _GLOBAL_SELECTIVE_SECONDARY_TRIGGER_GAP, _GLOBAL_SELECTIVE_SECONDARY_BAND_SIZE, _GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_MODE, _GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_CUTOFF_GAP_THRESHOLD, _GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_NEAR_CUT_WIDTH, _GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_MAX_CANDIDATES, _GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_CANDIDATE_TOP1_SHARE_THRESHOLD, _GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_SUPPORT_GAP_THRESHOLD, _GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_OVERFLOW_RATIO_THRESHOLD, _GLOBAL_FORWARD_GUIDANCE_WEIGHT, _GLOBAL_FORWARD_GUIDANCE_CLIP, _GLOBAL_FORWARD_GUIDANCE_TRIGGER_GAP, _GLOBAL_FORWARD_GUIDANCE_WIDEN_FACTOR, _GLOBAL_FORWARD_GUIDANCE_MIN_INFO_BITS, _GLOBAL_FORWARD_GUIDANCE_SNAPSHOT_FACTOR, _GLOBAL_FORWARD_GUIDANCE_SNAPSHOT_GAP, _GLOBAL_FORWARD_GUIDANCE_SNAPSHOT_SOURCE, _GLOBAL_FORWARD_GUIDANCE_HAMMING_RADIUS, _GLOBAL_FORWARD_GUIDANCE_TRIGGER_MODE, _GLOBAL_FORWARD_GUIDANCE_NEARCUT_GAP, _GLOBAL_FORWARD_GUIDANCE_POOL_TRIGGER_MIN_POSITIVE_NEARCUT, _GLOBAL_FORWARD_GUIDANCE_DIVERSITY_FALLBACK, _GLOBAL_FORWARD_GUIDANCE_MODE, _GLOBAL_EXPORT_STATE_COUNT_PROFILE, _GLOBAL_EXPORT_TERMINAL_SELECTOR_SIGNALS, _GLOBAL_EXPORT_FRONTIER_PRESSURE_TRACE, _GLOBAL_PRODUCTION_FAST_MODE, _GLOBAL_FRONTIER_PRESSURE_TRACE_ROWS, _GLOBAL_COLUMN_ORDER_LABEL, _GLOBAL_DECODER_MODE, _GLOBAL_BACKWARD_COLUMNS, _GLOBAL_BACKWARD_LAYOUT, _GLOBAL_BACKWARD_COLUMN_ORDER, _GLOBAL_MIDDLE_JOIN_PREFIX_COLUMNS, _GLOBAL_MIDDLE_JOIN_CUT_BEAM_FACTOR, _GLOBAL_MIDDLE_JOIN_CUT_WINDOW_COLUMNS, _GLOBAL_MIDDLE_JOIN_MULTICUT_PREFIX_COLUMNS, _GLOBAL_MIDDLE_JOIN_MULTICUT_STRIDE, _GLOBAL_MIDDLE_JOIN_MULTICUT_MAX_CUTS, _GLOBAL_MIDDLE_JOIN_MULTICUT_WEIGHT_MODE, _GLOBAL_BIDIRECTIONAL_SPLICE_RERANK, _GLOBAL_SPLICE_CANDIDATE_COUNT, _GLOBAL_SPLICE_CUT_SELECTOR, _GLOBAL_SPLICE_MAX_CUTS, _GLOBAL_SPLICE_AGGREGATE, _GLOBAL_SPLICE_REPLACE_FINAL_SELECTION
    _GLOBAL_FAMILY = family
    _GLOBAL_SAMPLE_COLUMNS = tuple(family.columns)
    _GLOBAL_SAMPLE_PRIORS = np.asarray([float(column.prior_probs[1]) for column in family.columns], dtype=np.float64)
    _GLOBAL_SEED = int(args.seed)
    _GLOBAL_LOOKAHEAD_DEPTH = int(args.lookahead_depth)
    _GLOBAL_LOOKAHEAD_SHORTLIST_SIZE = int(args.lookahead_shortlist_size)
    _GLOBAL_DELAYED_PRUNING_GAP_THRESHOLD = float(args.delayed_pruning_gap_threshold)
    _GLOBAL_DELAYED_PRUNING_FACTOR = int(args.delayed_pruning_factor)
    _GLOBAL_PRUNING_REPLAY_CHECKPOINT_STRIDE = int(args.pruning_replay_checkpoint_stride)
    _GLOBAL_PRUNING_REPLAY_HORIZON = int(args.pruning_replay_horizon)
    _GLOBAL_TAIL_EXACT_COLUMNS = int(args.tail_exact_columns)
    _GLOBAL_SUPERSTEP_MODE = str(args.superstep_mode)
    _GLOBAL_SUPERSTEP_PATH_BUDGET = int(args.superstep_path_budget)
    _GLOBAL_SUPERSTEP_STATE_BUDGET = int(args.superstep_state_budget)
    _GLOBAL_SUPERSTEP_TRANSITION_BUDGET = int(args.superstep_transition_budget)
    _GLOBAL_DETECTOR_BUCKET_PRUNING = bool(args.detector_bucket_pruning)
    _GLOBAL_DETECTOR_BUCKET_MAX_LOGICALS = int(args.detector_bucket_max_logicals)
    _GLOBAL_LOGICAL_CLASS_RESERVE_MIN_CLASSES = int(args.logical_class_reserve_min_classes)
    _GLOBAL_LOGICAL_CLASS_RESERVE_MAX_REPLACEMENTS = int(args.logical_class_reserve_max_replacements)
    _GLOBAL_LOGICAL_CLASS_RESERVE_MIN_REMAINING_COLUMNS = int(args.logical_class_reserve_min_remaining_columns)
    _GLOBAL_LOGICAL_CLASS_QUOTA_TOP_CLASSES = int(args.logical_class_quota_top_classes)
    _GLOBAL_LOGICAL_CLASS_QUOTA_RESERVED_SLOTS = int(args.logical_class_quota_reserved_slots)
    _GLOBAL_LOGICAL_CLASS_QUOTA_MIN_REMAINING_COLUMNS = int(args.logical_class_quota_min_remaining_columns)
    _GLOBAL_LINEAGE_RESERVE_CHECKPOINT_STRIDE = int(args.lineage_reserve_checkpoint_stride)
    _GLOBAL_LINEAGE_RESERVE_RESERVED_SLOTS = int(args.lineage_reserve_reserved_slots)
    _GLOBAL_LOGICAL_RERANK_COLUMNS = int(args.logical_rerank_columns)
    _GLOBAL_LOGICAL_RERANK_SHORTLIST_SIZE = int(args.logical_rerank_shortlist_size)
    _GLOBAL_LOGICAL_RERANK_MIN_CLASSES = int(args.logical_rerank_min_classes)
    _GLOBAL_LOGICAL_RERANK_STATE_BUDGET = int(args.logical_rerank_state_budget)
    _GLOBAL_LOGICAL_RERANK_TRANSITION_BUDGET = int(args.logical_rerank_transition_budget)
    _GLOBAL_LOGICAL_RERANK_CHECKPOINT_STRIDE = int(args.logical_rerank_checkpoint_stride)
    _GLOBAL_LOGICAL_RERANK_MAX_PASSES = int(args.logical_rerank_max_passes)
    _GLOBAL_LOGICAL_RERANK_MODE = str(args.logical_rerank_mode)
    _GLOBAL_FINAL_LOGICAL_SELECT_MODE = str(args.final_logical_select_mode)
    _GLOBAL_FINAL_LOGICAL_SELECT_REP_COST_WEIGHT = float(args.final_logical_select_rep_cost_weight)
    _GLOBAL_FINAL_LOGICAL_SELECT_MAX_LOG_MASS_GAP = float(args.final_logical_select_max_log_mass_gap)
    _GLOBAL_FINAL_LOGICAL_SELECT_RANK2_VITERBI_TOLERANCE = float(
        args.final_logical_select_rank2_viterbi_tolerance
    )
    _GLOBAL_TRACK_BEST_PATH = bool(args.track_best_path)
    _GLOBAL_MERGE_DUPLICATE_STATES = not bool(args.disable_state_merging)
    _GLOBAL_STATE_MERGE_PERIOD_COLUMNS = int(args.state_merge_period_columns)
    if int(_GLOBAL_STATE_MERGE_PERIOD_COLUMNS) < 0:
        raise ValueError("--state-merge-period-columns must be >= 0")
    if int(_GLOBAL_STATE_MERGE_PERIOD_COLUMNS) > 0 and bool(_GLOBAL_MERGE_DUPLICATE_STATES):
        raise ValueError("--state-merge-period-columns requires --disable-state-merging")
    _GLOBAL_SCORE_MODES = tuple(str(value) for value in args.score_modes)
    _GLOBAL_BEAM_SCORE_GAP_THRESHOLD = (
        None if args.beam_score_gap_threshold is None else float(args.beam_score_gap_threshold)
    )
    _GLOBAL_BEAM_SCORE_GAP_POLICY = args.beam_score_gap_policy
    _GLOBAL_SELECTIVE_SECONDARY_SCORE_MODE = str(args.selective_secondary_score_mode).strip().lower()
    _GLOBAL_SELECTIVE_SECONDARY_TRIGGER_GAP = float(args.selective_secondary_trigger_gap)
    _GLOBAL_SELECTIVE_SECONDARY_BAND_SIZE = int(args.selective_secondary_band_size)
    _GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_MODE = str(args.selective_local_lookahead_mode).strip().lower()
    _GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_CUTOFF_GAP_THRESHOLD = float(
        args.selective_local_lookahead_cutoff_gap_threshold
    )
    _GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_NEAR_CUT_WIDTH = float(
        args.selective_local_lookahead_near_cut_width
    )
    _GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_MAX_CANDIDATES = int(
        args.selective_local_lookahead_max_candidates
    )
    _GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_CANDIDATE_TOP1_SHARE_THRESHOLD = float(
        args.selective_local_lookahead_candidate_top1_share_threshold
    )
    _GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_SUPPORT_GAP_THRESHOLD = float(
        args.selective_local_lookahead_support_gap_threshold
    )
    _GLOBAL_SELECTIVE_LOCAL_LOOKAHEAD_OVERFLOW_RATIO_THRESHOLD = float(
        args.selective_local_lookahead_overflow_ratio_threshold
    )
    _GLOBAL_FORWARD_GUIDANCE_WEIGHT = float(args.forward_guidance_weight)
    _GLOBAL_FORWARD_GUIDANCE_CLIP = float(args.forward_guidance_clip)
    _GLOBAL_FORWARD_GUIDANCE_TRIGGER_GAP = float(args.forward_guidance_trigger_gap)
    _GLOBAL_FORWARD_GUIDANCE_WIDEN_FACTOR = float(args.forward_guidance_widen_factor)
    _GLOBAL_FORWARD_GUIDANCE_MIN_INFO_BITS = float(args.forward_guidance_min_info_bits)
    _GLOBAL_FORWARD_GUIDANCE_SNAPSHOT_FACTOR = float(args.forward_guidance_snapshot_factor)
    _GLOBAL_FORWARD_GUIDANCE_SNAPSHOT_GAP = (
        None if args.forward_guidance_snapshot_gap is None else float(args.forward_guidance_snapshot_gap)
    )
    _GLOBAL_FORWARD_GUIDANCE_SNAPSHOT_SOURCE = str(args.forward_guidance_snapshot_source)
    _GLOBAL_FORWARD_GUIDANCE_HAMMING_RADIUS = int(args.forward_guidance_hamming_radius)
    _GLOBAL_FORWARD_GUIDANCE_TRIGGER_MODE = str(args.forward_guidance_trigger_mode)
    _GLOBAL_FORWARD_GUIDANCE_NEARCUT_GAP = float(args.forward_guidance_nearcut_gap)
    _GLOBAL_FORWARD_GUIDANCE_POOL_TRIGGER_MIN_POSITIVE_NEARCUT = int(
        args.forward_guidance_pool_trigger_min_positive_nearcut
    )
    _GLOBAL_FORWARD_GUIDANCE_DIVERSITY_FALLBACK = str(args.forward_guidance_diversity_fallback)
    _GLOBAL_FORWARD_GUIDANCE_MODE = str(args.forward_guidance_mode)
    _GLOBAL_EXPORT_STATE_COUNT_PROFILE = bool(args.export_state_count_profile)
    _GLOBAL_EXPORT_TERMINAL_SELECTOR_SIGNALS = bool(args.export_terminal_selector_signals)
    _GLOBAL_EXPORT_FRONTIER_PRESSURE_TRACE = bool(args.export_frontier_pressure_trace)
    _GLOBAL_PRODUCTION_FAST_MODE = bool(args.production_fast_mode)
    _GLOBAL_FRONTIER_PRESSURE_TRACE_ROWS = []
    _GLOBAL_COLUMN_ORDER_LABEL = str(args.column_order)
    _GLOBAL_DECODER_MODE = str(decoder_mode)
    _GLOBAL_BACKWARD_COLUMNS = tuple(backward_family.columns) if backward_family is not None else tuple()
    _GLOBAL_BACKWARD_LAYOUT = backward_family.layout if backward_family is not None else None
    _GLOBAL_BACKWARD_COLUMN_ORDER = str(backward_column_order)
    _GLOBAL_MIDDLE_JOIN_PREFIX_COLUMNS = effective_middle_join_prefix_columns
    _GLOBAL_MIDDLE_JOIN_CUT_BEAM_FACTOR = int(effective_middle_join_cut_beam_factor)
    _GLOBAL_MIDDLE_JOIN_CUT_WINDOW_COLUMNS = int(effective_middle_join_cut_window_columns)
    _GLOBAL_MIDDLE_JOIN_MULTICUT_PREFIX_COLUMNS = tuple(
        int(value) for value in effective_middle_join_multicut_prefix_columns
    )
    _GLOBAL_MIDDLE_JOIN_MULTICUT_STRIDE = int(effective_middle_join_multicut_stride)
    _GLOBAL_MIDDLE_JOIN_MULTICUT_MAX_CUTS = int(effective_middle_join_multicut_max_cuts)
    _GLOBAL_MIDDLE_JOIN_MULTICUT_WEIGHT_MODE = str(effective_middle_join_multicut_weight_mode)
    _GLOBAL_BIDIRECTIONAL_SPLICE_RERANK = bool(args.bidirectional_splice_rerank)
    _GLOBAL_SPLICE_CANDIDATE_COUNT = int(args.splice_candidate_count)
    _GLOBAL_SPLICE_CUT_SELECTOR = str(args.splice_cut_selector)
    _GLOBAL_SPLICE_MAX_CUTS = int(args.splice_max_cuts)
    _GLOBAL_SPLICE_AGGREGATE = str(args.splice_aggregate)
    _GLOBAL_SPLICE_REPLACE_FINAL_SELECTION = bool(args.splice_replace_final_selection)

    frontier_rows = _frontier_rows(family)
    _write_csv(
        out_dir / "frontier_summary.csv",
        frontier_rows,
        ["family", "model_label", "matrix_rows", "matrix_cols", "logical_rows", "edge_count", "frontier_max_active_detectors", "correction_state_mode", "correction_state_bits"],
    )
    if bool(args.write_plots):
        _plot_frontier_profile(family=family, out_path=out_dir / "fig_frontier_width_profile.png")
    print("[setup] shard partial CSVs will flush after each completed shot", flush=True)

    metadata_path = out_dir / "run_metadata.json"
    metadata: dict[str, object] = {
        "benchmark": f"progressive_dem_{family.backend}_{family.scope}",
        "backend": str(args.backend),
        "stim_path": ("" if args.stim_path is None else str(Path(args.stim_path).resolve())),
        "external_benchmark_label": ("" if args.external_benchmark_label is None else str(args.external_benchmark_label)),
        "external_noisy_rounds": ("" if args.external_noisy_rounds is None else int(args.external_noisy_rounds)),
        "external_perfect_rounds": int(args.external_perfect_rounds),
        "initial_data_error_rate": (
            "" if args.initial_data_error_rate is None else float(args.initial_data_error_rate)
        ),
        "scope": str(args.scope),
        "p_location": float(args.p_location),
        "noisy_rounds": int(family.noisy_rounds),
        "total_rounds": int(family.total_rounds),
        "shots": int(args.shots),
        "shot_start": int(args.shot_start),
        "shot_indices": [int(value) for value in tuple(args.selected_shot_indices)],
        "seed": int(args.seed),
        "beam_sizes": [int(value) for value in args.beam_sizes],
        "column_order": str(args.column_order),
        "column_order_file": "" if args.column_order_file is None else str(Path(args.column_order_file).resolve()),
        "decoder_mode": str(decoder_mode),
        "backward_column_order": str(backward_column_order),
        "beam_score_gap_threshold": (
            "" if args.beam_score_gap_threshold is None else float(args.beam_score_gap_threshold)
        ),
        "beam_score_gap_policy_mode": (
            "" if args.beam_score_gap_policy is None else str(args.beam_score_gap_policy.mode)
        ),
        "beam_score_gap_policy_base_threshold": (
            "" if args.beam_score_gap_policy is None else float(args.beam_score_gap_policy.base_threshold)
        ),
        "beam_score_gap_policy_final_threshold": (
            "" if args.beam_score_gap_policy is None else float(args.beam_score_gap_policy.final_threshold)
        ),
        "beam_score_gap_policy_slope": (
            "" if args.beam_score_gap_policy is None else float(args.beam_score_gap_policy.slope)
        ),
        "beam_score_gap_policy_reference_count": (
            "" if args.beam_score_gap_policy is None else float(args.beam_score_gap_policy.reference_count)
        ),
        "beam_score_gap_policy_min_threshold": (
            "" if args.beam_score_gap_policy is None else float(args.beam_score_gap_policy.min_threshold)
        ),
        "beam_score_gap_policy_max_threshold": (
            "" if args.beam_score_gap_policy is None else float(args.beam_score_gap_policy.max_threshold)
        ),
        "selective_secondary_score_mode": str(args.selective_secondary_score_mode).strip().lower(),
        "selective_secondary_trigger_gap": float(args.selective_secondary_trigger_gap),
        "selective_secondary_band_size": int(args.selective_secondary_band_size),
        "selective_local_lookahead_mode": str(args.selective_local_lookahead_mode).strip().lower(),
        "selective_local_lookahead_cutoff_gap_threshold": float(
            args.selective_local_lookahead_cutoff_gap_threshold
        ),
        "selective_local_lookahead_near_cut_width": float(args.selective_local_lookahead_near_cut_width),
        "selective_local_lookahead_max_candidates": int(args.selective_local_lookahead_max_candidates),
        "selective_local_lookahead_candidate_top1_share_threshold": float(
            args.selective_local_lookahead_candidate_top1_share_threshold
        ),
        "selective_local_lookahead_support_gap_threshold": (
            ""
            if not math.isfinite(float(args.selective_local_lookahead_support_gap_threshold))
            else float(args.selective_local_lookahead_support_gap_threshold)
        ),
        "selective_local_lookahead_overflow_ratio_threshold": (
            ""
            if not math.isfinite(float(args.selective_local_lookahead_overflow_ratio_threshold))
            else float(args.selective_local_lookahead_overflow_ratio_threshold)
        ),
        "forward_guidance_weight": float(args.forward_guidance_weight),
        "forward_guidance_clip": float(args.forward_guidance_clip),
        "forward_guidance_trigger_gap": float(args.forward_guidance_trigger_gap),
        "forward_guidance_widen_factor": float(args.forward_guidance_widen_factor),
        "forward_guidance_min_info_bits": float(args.forward_guidance_min_info_bits),
        "forward_guidance_snapshot_factor": float(args.forward_guidance_snapshot_factor),
        "forward_guidance_snapshot_gap": (
            ""
            if args.forward_guidance_snapshot_gap is None
            else _format_forward_guidance_snapshot_gap(args.forward_guidance_snapshot_gap)
        ),
        "forward_guidance_snapshot_source": str(args.forward_guidance_snapshot_source),
        "forward_guidance_hamming_radius": int(args.forward_guidance_hamming_radius),
        "forward_guidance_trigger_mode": str(args.forward_guidance_trigger_mode),
        "forward_guidance_nearcut_gap": float(args.forward_guidance_nearcut_gap),
        "forward_guidance_pool_trigger_min_positive_nearcut": int(
            args.forward_guidance_pool_trigger_min_positive_nearcut
        ),
        "forward_guidance_diversity_fallback": str(args.forward_guidance_diversity_fallback),
        "forward_guidance_mode": str(args.forward_guidance_mode),
        "middle_join_prefix_columns": (
            "" if effective_middle_join_prefix_columns is None else int(effective_middle_join_prefix_columns)
        ),
        "middle_join_multicut_prefix_columns": [
            int(value) for value in effective_middle_join_multicut_prefix_columns
        ],
        "middle_join_multicut_stride": int(effective_middle_join_multicut_stride),
        "middle_join_multicut_max_cuts": int(effective_middle_join_multicut_max_cuts),
        "middle_join_multicut_weight_mode": str(effective_middle_join_multicut_weight_mode),
        "middle_join_cut_window_columns": int(effective_middle_join_cut_window_columns),
        "middle_join_cut_beam_factor": int(effective_middle_join_cut_beam_factor),
        "bidirectional_splice_rerank": bool(args.bidirectional_splice_rerank),
        "splice_candidate_count": int(args.splice_candidate_count),
        "splice_cut_selector": str(args.splice_cut_selector),
        "splice_max_cuts": int(args.splice_max_cuts),
        "splice_aggregate": str(args.splice_aggregate),
        "splice_replace_final_selection": bool(args.splice_replace_final_selection),
        "correction_state_mode": str(family.correction_state_mode),
        "correction_state_bits": int(family.correction_state_bits),
        "state_merge_mode": _state_merge_mode_label(
            merge_duplicate_states=not bool(args.disable_state_merging),
            state_merge_period_columns=int(args.state_merge_period_columns),
        ),
        "state_merge_period_columns": int(args.state_merge_period_columns),
        "production_fast_mode": bool(args.production_fast_mode),
        "write_plots": bool(args.write_plots),
        "write_report": bool(args.write_report),
        "export_state_count_profile": bool(args.export_state_count_profile),
        "export_terminal_selector_signals": bool(args.export_terminal_selector_signals),
        "export_frontier_pressure_trace": bool(args.export_frontier_pressure_trace),
        "require_correction_cache": bool(args.require_correction_cache),
        "score_modes": [str(value) for value in args.score_modes],
        "lookahead_depth": int(args.lookahead_depth),
        "lookahead_shortlist_size": int(args.lookahead_shortlist_size),
        "delayed_pruning_gap_threshold": float(args.delayed_pruning_gap_threshold),
        "delayed_pruning_factor": int(args.delayed_pruning_factor),
        "pruning_replay_checkpoint_stride": int(args.pruning_replay_checkpoint_stride),
        "pruning_replay_horizon": int(args.pruning_replay_horizon),
        "tail_exact_columns": int(args.tail_exact_columns),
        "superstep_mode": str(args.superstep_mode),
        "superstep_path_budget": int(args.superstep_path_budget),
        "superstep_state_budget": int(args.superstep_state_budget),
        "superstep_transition_budget": int(args.superstep_transition_budget),
        "detector_bucket_pruning": bool(args.detector_bucket_pruning),
        "detector_bucket_max_logicals": int(args.detector_bucket_max_logicals),
        "logical_class_reserve_min_classes": int(args.logical_class_reserve_min_classes),
        "logical_class_reserve_max_replacements": int(args.logical_class_reserve_max_replacements),
        "logical_class_reserve_min_remaining_columns": int(args.logical_class_reserve_min_remaining_columns),
        "logical_class_quota_top_classes": int(args.logical_class_quota_top_classes),
        "logical_class_quota_reserved_slots": int(args.logical_class_quota_reserved_slots),
        "logical_class_quota_min_remaining_columns": int(args.logical_class_quota_min_remaining_columns),
        "lineage_reserve_checkpoint_stride": int(args.lineage_reserve_checkpoint_stride),
        "lineage_reserve_reserved_slots": int(args.lineage_reserve_reserved_slots),
        "logical_rerank_columns": int(args.logical_rerank_columns),
        "logical_rerank_shortlist_size": int(args.logical_rerank_shortlist_size),
        "logical_rerank_min_classes": int(args.logical_rerank_min_classes),
        "logical_rerank_state_budget": int(args.logical_rerank_state_budget),
        "logical_rerank_transition_budget": int(args.logical_rerank_transition_budget),
        "logical_rerank_checkpoint_stride": int(args.logical_rerank_checkpoint_stride),
        "logical_rerank_max_passes": int(args.logical_rerank_max_passes),
        "logical_rerank_mode": str(args.logical_rerank_mode),
        "final_logical_select_mode": str(args.final_logical_select_mode),
        "final_logical_select_rep_cost_weight": float(args.final_logical_select_rep_cost_weight),
        "final_logical_select_max_log_mass_gap": (
            float(args.final_logical_select_max_log_mass_gap)
            if math.isfinite(float(args.final_logical_select_max_log_mass_gap))
            else ""
        ),
        "final_logical_select_rank2_viterbi_tolerance": float(
            args.final_logical_select_rank2_viterbi_tolerance
        ),
        "track_best_path": bool(args.track_best_path),
        "cpus": int(args.cpus),
        "shards": int(args.shards),
        "matrix_rows": int(family.matrix_rows),
        "matrix_cols": int(family.matrix_cols),
        "logical_rows": int(family.logical_rows),
        "column_order_name": str(family.column_order_name),
        "column_order_source": str(family.column_order_source),
        "frontier_max_active_detectors": int(family.layout.max_active_detectors),
        "backward_frontier_max_active_detectors": (
            int(backward_family.layout.max_active_detectors) if backward_family is not None else ""
        ),
        "joint_order_forward_prefix_active_area": (
            "" if joint_middle_join_summary is None else int(joint_middle_join_summary.forward_prefix_active_area)
        ),
        "joint_order_backward_prefix_active_area": (
            "" if joint_middle_join_summary is None else int(joint_middle_join_summary.backward_prefix_active_area)
        ),
        "joint_order_cut_boundary_rows": (
            "" if joint_middle_join_summary is None else int(joint_middle_join_summary.cut_boundary_rows)
        ),
        "status": "running",
        "setup_elapsed_s": float(time.time() - start),
    }
    _write_json(metadata_path, metadata)

    tasks = [
        {
            "task_id": int(task_id),
            "shot_indices": shot_indices,
            "beam_sizes": [int(value) for value in args.beam_sizes],
            "partial_path": str(shards_dir / f"shard_{int(task_id):04d}_per_shot.csv"),
            "progress_path": str(shards_dir / f"shard_{int(task_id):04d}_progress.json"),
        }
        for task_id, shot_indices in enumerate(
            _split_shot_indices(tuple(args.selected_shot_indices), int(args.shards))
        )
    ]

    per_shot_rows: list[dict[str, object]] = []
    pressure_trace_rows: list[dict[str, object]] = []
    run_start = time.time()
    if int(args.cpus) == 1:
        for task_index, task in enumerate(tasks, start=1):
            result = _run_shard(task)
            shard_rows = list(result["rows"])
            per_shot_rows.extend(shard_rows)
            pressure_trace_rows.extend(list(result.get("pressure_trace_rows", [])))
            shard_path = shards_dir / f"shard_{int(result['task_id']):04d}_per_shot.csv"
            _write_csv(shard_path, shard_rows, _fieldnames_from_rows(shard_rows, ["shot"]))
            elapsed_s = time.time() - run_start
            remaining = len(tasks) - int(task_index)
            mean_per_task = elapsed_s / float(task_index)
            eta_s = mean_per_task * float(remaining)
            print(
                f"[progress] shard {int(task_index)}/{len(tasks)} shots={int(result['shots_completed'])} "
                f"elapsed={elapsed_s:.1f}s eta={eta_s:.1f}s",
                flush=True,
            )
    else:
        ctx = mp.get_context("fork")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=int(args.cpus),
            mp_context=ctx,
            initializer=_init_progressive_worker,
        ) as executor:
            future_to_task = {executor.submit(_run_shard, task): task for task in tasks}
            completed = 0
            shots_completed = 0
            for future in concurrent.futures.as_completed(future_to_task):
                result = future.result()
                shard_rows = list(result["rows"])
                per_shot_rows.extend(shard_rows)
                pressure_trace_rows.extend(list(result.get("pressure_trace_rows", [])))
                shard_path = shards_dir / f"shard_{int(result['task_id']):04d}_per_shot.csv"
                _write_csv(shard_path, shard_rows, _fieldnames_from_rows(shard_rows, ["shot"]))
                completed += 1
                shots_completed += int(result["shots_completed"])
                if int(args.progress_every_shards) > 0 and (
                    int(completed) == 1
                    or (int(completed) % int(args.progress_every_shards) == 0)
                    or int(completed) == int(len(tasks))
                ):
                    elapsed_s = time.time() - run_start
                    mean_per_shard = elapsed_s / float(completed)
                    eta_s = mean_per_shard * float(len(tasks) - completed)
                    print(
                        f"[progress] shard {int(completed)}/{len(tasks)} "
                        f"shots_done={int(shots_completed)}/{int(args.shots)} "
                        f"elapsed={elapsed_s:.1f}s eta={eta_s:.1f}s",
                        flush=True,
                    )

    per_shot_rows.sort(key=lambda row: (int(row["shot"]), str(row["score_mode"]), int(row["beam_size"])))
    if per_shot_rows:
        _write_csv(out_dir / "per_shot_rows.csv", per_shot_rows, _fieldnames_from_rows(per_shot_rows, ["shot"]))
    terminal_failedrank_analysis: dict[str, object] | None = None
    if bool(args.export_terminal_selector_signals) and per_shot_rows:
        terminal_failedrank_analysis = terminal_signal_analysis.run_terminal_signal_analysis(
            results_dir=out_dir,
            input_csv=out_dir / "per_shot_rows.csv",
            summary_csv=out_dir / "terminal_signal_failedrank_summary.csv",
            aggregate_json=out_dir / "terminal_signal_failedrank_aggregate.json",
            plot_path=out_dir / "fig_terminal_signal_failedrank_subset.png",
            matrix_rows=int(family.matrix_rows),
            matrix_cols=int(family.matrix_cols),
            logical_rows=int(family.logical_rows),
            noisy_rounds=int(family.noisy_rounds),
            failure_mode_filter=terminal_signal_analysis.DEFAULT_FAILURE_MODE_FILTER,
        )
    if bool(args.export_state_count_profile):
        _write_state_count_profile_artifacts(per_shot_rows=per_shot_rows, out_dir=out_dir)
    if bool(args.export_frontier_pressure_trace):
        pressure_trace_rows.sort(
            key=lambda row: (
                str(row.get("scope", "")),
                int(row.get("shot", 0)),
                int(row.get("K", 0)),
                str(row.get("score_mode", "")),
                int(row.get("boundary_index", 0)),
            )
        )
        _write_frontier_pressure_trace_artifacts(trace_rows=pressure_trace_rows, out_dir=out_dir)
    summary_rows = _build_summary_rows(per_shot_rows)
    _write_csv(
        out_dir / "summary.csv",
        summary_rows,
        _extend_fieldnames_with_row_keys(
            [
            "decoder",
            "family",
            "decoder_mode",
            "backward_column_order",
            "correction_state_mode",
            "correction_state_bits",
            "state_merge_mode",
            "state_merge_period_columns",
            "beam_size",
            "score_mode",
            "beam_score_gap_threshold",
            "beam_score_gap_policy_mode",
            "beam_score_gap_policy_base_threshold",
            "beam_score_gap_policy_final_threshold",
            "beam_score_gap_policy_slope",
            "beam_score_gap_policy_reference_count",
            "beam_score_gap_policy_min_threshold",
            "beam_score_gap_policy_max_threshold",
            "beam_score_gap_threshold_trace_mean",
            "beam_score_gap_threshold_trace_min",
            "beam_score_gap_threshold_trace_max",
            "selective_secondary_score_mode",
            "selective_secondary_trigger_gap",
            "selective_secondary_band_size",
            "selective_local_lookahead_mode",
            "selective_local_lookahead_score_mode",
            "selective_local_lookahead_cutoff_gap_threshold",
            "selective_local_lookahead_near_cut_width",
            "selective_local_lookahead_max_candidates",
            "selective_local_lookahead_candidate_top1_share_threshold",
            "selective_local_lookahead_support_gap_threshold",
            "selective_local_lookahead_overflow_ratio_threshold",
            "forward_guidance_weight",
            "forward_guidance_clip",
            "forward_guidance_trigger_gap",
            "forward_guidance_widen_factor",
            "forward_guidance_min_info_bits",
            "forward_guidance_snapshot_factor",
            "forward_guidance_snapshot_gap",
            "forward_guidance_snapshot_source",
            "forward_guidance_hamming_radius",
            "forward_guidance_trigger_mode",
            "forward_guidance_nearcut_gap",
            "forward_guidance_pool_trigger_min_positive_nearcut",
            "forward_guidance_diversity_fallback",
            "forward_guidance_mode",
            "forward_guidance_diag_step_count_mean",
            "forward_guidance_triggered_step_count_mean",
            "forward_guidance_triggered_fraction_mean",
            "forward_guidance_top_gap_triggered_step_count_mean",
            "forward_guidance_top_gap_triggered_fraction_mean",
            "forward_guidance_support_aware_triggered_step_count_mean",
            "forward_guidance_support_aware_triggered_fraction_mean",
            "forward_guidance_base_top_primary_gap_mean",
            "forward_guidance_base_top_primary_gap_p10",
            "forward_guidance_base_top_primary_gap_p50",
            "forward_guidance_base_top_primary_gap_p90",
            "forward_guidance_alignment_metadata_step_count_mean",
            "forward_guidance_aligned_step_count_mean",
            "forward_guidance_no_alignment_step_count_mean",
            "forward_guidance_top_rank_changed_count_mean",
            "forward_guidance_top_rank_changed_fraction_mean",
            "forward_guidance_top_logical_changed_count_mean",
            "forward_guidance_top_logical_changed_fraction_mean",
            "forward_guidance_selected_distance_abs_mean",
            "forward_guidance_selected_distance_abs_max",
            "forward_guidance_selected_state_count_mean",
            "forward_guidance_candidate_interval_row_count_mean",
            "forward_guidance_candidate_snapshot_count_mean",
            "forward_guidance_positive_aligned_snapshot_count_mean",
            "forward_guidance_backward_active_row_count_mean",
            "forward_guidance_common_active_row_count_mean",
            "forward_guidance_aligned_row_count_mean",
            "forward_guidance_aligned_row_count_max",
            "forward_guidance_aligned_fraction_backward_mean",
            "forward_guidance_aligned_fraction_common_mean",
            "forward_guidance_middle_row_count_mean",
            "forward_guidance_overlap_row_count_mean",
            "forward_guidance_zero_support_row_count_mean",
            "forward_guidance_middle_row_fraction_common_mean",
            "forward_guidance_overlap_row_fraction_common_mean",
            "forward_guidance_projected_state_count_mean",
            "forward_guidance_projected_entropy_mean",
            "forward_guidance_projected_effective_support_mean",
            "forward_guidance_projected_top_logprob_mean",
            "forward_guidance_projected_logprob_gap_mean",
            "forward_guidance_candidate_state_count_total",
            "forward_guidance_applied_state_count_total",
            "forward_guidance_missing_mass_count_total",
            "forward_guidance_clipped_state_count_total",
            "forward_guidance_missing_mass_fraction_mean",
            "forward_guidance_clipped_fraction_mean",
            "forward_guidance_bonus_min_min",
            "forward_guidance_bonus_p10_mean",
            "forward_guidance_bonus_p50_mean",
            "forward_guidance_bonus_mean_mean",
            "forward_guidance_bonus_p90_mean",
            "forward_guidance_bonus_max_max",
            "forward_guidance_weighted_bonus_mean_mean",
            "forward_guidance_guided_top_base_rank_mean",
            "forward_guidance_guided_top_base_rank_p99",
            "forward_guidance_base_top_guided_rank_mean",
            "forward_guidance_base_top_guided_rank_p99",
            "forward_guidance_projected_info_bits_mean",
            "forward_guidance_conditional_shortlist_state_count_mean",
            "forward_guidance_conditional_lookup_radius_mean",
            "forward_guidance_conditional_finite_score_count_mean",
            "forward_guidance_conditional_exact_support_count_mean",
            "forward_guidance_conditional_neighborhood_support_count_mean",
            "forward_guidance_conditional_neighborhood_only_support_count_mean",
            "forward_guidance_conditional_missing_support_count_mean",
            "forward_guidance_conditional_positive_raw_info_count_mean",
            "forward_guidance_conditional_finite_outside_kept_count_mean",
            "forward_guidance_conditional_positive_outside_kept_count_mean",
            "forward_guidance_conditional_nearcut_outside_kept_count_mean",
            "forward_guidance_conditional_positive_nearcut_outside_kept_count_mean",
            "forward_guidance_conditional_missing_logical_class_outside_kept_count_mean",
            "forward_guidance_conditional_positive_bonus_count_mean",
            "forward_guidance_conditional_promoted_state_count_total",
            "forward_guidance_conditional_demoted_state_count_total",
            "forward_guidance_conditional_changed_kept_step_count",
            "forward_guidance_conditional_changed_kept_fraction_mean",
            "forward_guidance_conditional_added_logical_class_count_total",
            "forward_guidance_conditional_fallback_candidate_count_total",
            "forward_guidance_conditional_fallback_candidate_count_mean",
            "forward_guidance_conditional_fallback_added_state_count_total",
            "forward_guidance_conditional_fallback_added_logical_class_count_total",
            "forward_guidance_conditional_raw_info_min_min",
            "forward_guidance_conditional_raw_info_p10_mean",
            "forward_guidance_conditional_raw_info_p50_mean",
            "forward_guidance_conditional_raw_info_mean_mean",
            "forward_guidance_conditional_raw_info_p90_mean",
            "forward_guidance_conditional_raw_info_max_max",
            "forward_guidance_conditional_bonus_max_max",
            "forward_guidance_checkpoint_available_step_count",
            "forward_guidance_checkpoint_available_fraction_mean",
            "forward_guidance_checkpoint_key_count_mean",
            "forward_guidance_checkpoint_source_state_count_mean",
            "forward_guidance_checkpoint_mass_coverage_after_trim_mean",
            "forward_guidance_checkpoint_band_state_count_mean",
            "forward_guidance_checkpoint_hit_count_mean",
            "forward_guidance_checkpoint_hit_fraction_mean",
            "forward_guidance_checkpoint_rescue_budget_mean",
            "forward_guidance_checkpoint_rescued_state_count_total",
            "forward_guidance_checkpoint_rescued_state_count_mean",
            "forward_guidance_checkpoint_replay_triggered_step_count",
            "forward_guidance_checkpoint_replay_triggered_fraction_mean",
            "forward_guidance_checkpoint_replay_prior_available_step_count",
            "forward_guidance_checkpoint_replay_prior_available_fraction_mean",
            "forward_guidance_checkpoint_replay_called_step_count",
            "forward_guidance_checkpoint_replay_called_fraction_mean",
            "forward_guidance_checkpoint_replay_attempted_step_count",
            "forward_guidance_checkpoint_replay_attempted_fraction_mean",
            "forward_guidance_checkpoint_replay_succeeded_step_count",
            "forward_guidance_checkpoint_replay_succeeded_fraction_mean",
            "forward_guidance_checkpoint_replay_aborted_no_checkpoint_step_count",
            "forward_guidance_checkpoint_replay_aborted_no_checkpoint_fraction_mean",
            "forward_guidance_checkpoint_replay_aborted_window_too_long_step_count",
            "forward_guidance_checkpoint_replay_aborted_window_too_long_fraction_mean",
            "forward_guidance_checkpoint_replay_aborted_empty_query_set_step_count",
            "forward_guidance_checkpoint_replay_aborted_empty_query_set_fraction_mean",
            "forward_guidance_checkpoint_replay_aborted_budget_cap_step_count",
            "forward_guidance_checkpoint_replay_aborted_budget_cap_fraction_mean",
            "forward_guidance_checkpoint_replay_completed_step_count",
            "forward_guidance_checkpoint_replay_completed_fraction_mean",
            "forward_guidance_checkpoint_replay_target_reached_step_count",
            "forward_guidance_checkpoint_replay_target_reached_fraction_mean",
            "forward_guidance_checkpoint_replay_final_processed_columns_mean",
            "forward_guidance_checkpoint_replay_available_snapshot_count_mean",
            "forward_guidance_checkpoint_replay_target_snapshot_present_step_count",
            "forward_guidance_checkpoint_replay_target_snapshot_present_fraction_mean",
            "forward_guidance_checkpoint_replay_target_snapshot_state_count_mean",
            "forward_guidance_checkpoint_replay_seed_key_count_mean",
            "forward_guidance_checkpoint_replay_generated_key_count_mean",
            "forward_guidance_checkpoint_replay_new_key_count_mean",
            "forward_guidance_checkpoint_replay_query_key_count_mean",
            "forward_guidance_checkpoint_replay_hit_key_count_mean",
            "forward_guidance_checkpoint_replay_hit_candidate_count_mean",
            "forward_guidance_checkpoint_query_hit_count_before_replay_mean",
            "forward_guidance_checkpoint_query_hit_count_after_replay_mean",
            "forward_guidance_checkpoint_query_new_hit_count_from_replay_mean",
            "forward_guidance_checkpoint_replay_expansion_count_mean",
            "forward_guidance_checkpoint_replay_max_frontier_size_mean",
            "forward_guidance_checkpoint_replay_terminal_state_count_mean",
            "forward_guidance_checkpoint_replay_replayed_column_count_mean",
            "forward_guidance_checkpoint_replay_budget_exhausted_step_count",
            "forward_guidance_checkpoint_replay_budget_exhausted_fraction_mean",
            "forward_guidance_local_widen_eligible_step_count",
            "forward_guidance_local_widen_triggered_step_count",
            "forward_guidance_local_widen_triggered_fraction_mean",
            "forward_guidance_first_trigger_active_processed_columns_min",
            "forward_guidance_first_local_widen_triggered_processed_columns_min",
            "forward_guidance_local_widen_added_state_count_total",
            "forward_guidance_local_widen_kept_count_mean",
            "forward_guidance_truth_cut_state_valid_step_count",
            "forward_guidance_truth_cut_candidate_present_step_count",
            "forward_guidance_truth_cut_candidate_present_fraction_mean",
            "forward_guidance_truth_cut_ordinary_kept_step_count",
            "forward_guidance_truth_cut_ordinary_kept_fraction_mean",
            "forward_guidance_truth_cut_provisional_present_step_count",
            "forward_guidance_truth_cut_provisional_present_fraction_mean",
            "forward_guidance_truth_cut_exact_supported_step_count",
            "forward_guidance_truth_cut_exact_supported_fraction_mean",
            "forward_guidance_truth_cut_checkpoint_hit_before_replay_step_count",
            "forward_guidance_truth_cut_checkpoint_hit_before_replay_fraction_mean",
            "forward_guidance_truth_cut_checkpoint_replay_queried_step_count",
            "forward_guidance_truth_cut_checkpoint_replay_queried_fraction_mean",
            "forward_guidance_truth_cut_checkpoint_replay_hit_step_count",
            "forward_guidance_truth_cut_checkpoint_replay_hit_fraction_mean",
            "forward_guidance_truth_cut_prev_checkpoint_exists_step_count",
            "forward_guidance_truth_cut_prev_checkpoint_exists_fraction_mean",
            "forward_guidance_truth_cut_prev_checkpoint_ancestor_present_step_count",
            "forward_guidance_truth_cut_prev_checkpoint_ancestor_present_fraction_mean",
            "forward_guidance_truth_cut_neighborhood_supported_step_count",
            "forward_guidance_truth_cut_neighborhood_supported_fraction_mean",
            "forward_guidance_truth_cut_conditional_supported_step_count",
            "forward_guidance_truth_cut_conditional_supported_fraction_mean",
            "forward_guidance_truth_cut_conditional_positive_step_count",
            "forward_guidance_truth_cut_conditional_positive_fraction_mean",
            "forward_guidance_truth_cut_added_extra_step_count",
            "forward_guidance_truth_cut_added_extra_fraction_mean",
            "forward_guidance_truth_cut_final_kept_step_count",
            "forward_guidance_truth_cut_final_kept_fraction_mean",
            "forward_guidance_truth_cut_first_candidate_missing_processed_columns_min",
            "forward_guidance_truth_cut_first_ordinary_missing_processed_columns_min",
            "forward_guidance_truth_cut_first_provisional_missing_processed_columns_min",
            "forward_guidance_truth_cut_first_final_missing_processed_columns_min",
            "forward_guidance_truth_cut_first_added_extra_processed_columns_min",
            "forward_guidance_truth_cut_first_ordinary_loss_trigger_active_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_local_widen_triggered_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_provisional_present_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_exact_supported_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_neighborhood_supported_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_conditional_supported_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_hit_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_hit_before_replay_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_queried_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_hit_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_prev_checkpoint_exists_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_prev_checkpoint_ancestor_present_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_base_rank_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_rank_over_beam_size_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_rank_over_ordinary_kept_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_within_2k_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_within_3k_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_within_4k_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_within_2x_ordinary_kept_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_within_3x_ordinary_kept_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_within_4x_ordinary_kept_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_added_extra_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_available_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_key_count_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_source_state_count_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_mass_coverage_after_trim_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_band_state_count_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_hit_count_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_hit_fraction_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_rescue_budget_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_rescued_state_count_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_triggered_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_prior_available_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_called_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_attempted_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_succeeded_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_aborted_no_checkpoint_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_aborted_window_too_long_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_aborted_empty_query_set_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_aborted_budget_cap_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_completed_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_target_reached_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_final_processed_columns_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_available_snapshot_count_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_target_snapshot_present_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_target_snapshot_state_count_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_seed_key_count_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_generated_key_count_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_new_key_count_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_query_key_count_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_hit_key_count_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_hit_candidate_count_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_query_hit_count_before_replay_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_query_hit_count_after_replay_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_query_new_hit_count_from_replay_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_expansion_count_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_max_frontier_size_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_terminal_state_count_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_replayed_column_count_mean",
            "forward_guidance_truth_cut_first_ordinary_loss_checkpoint_replay_budget_exhausted_mean",
            "forward_guidance_truth_cut_raw_info_mean",
            "forward_guidance_truth_cut_raw_info_p50_mean",
            "lookahead_depth",
            "lookahead_shortlist_size",
            "delayed_pruning_gap_threshold",
            "delayed_pruning_factor",
            "pruning_replay_checkpoint_stride",
            "pruning_replay_horizon",
            "tail_exact_columns",
            "superstep_mode",
            "superstep_path_budget",
            "superstep_state_budget",
            "superstep_transition_budget",
            "detector_bucket_pruning",
            "detector_bucket_max_logicals",
            "logical_class_reserve_min_classes",
            "logical_class_reserve_max_replacements",
            "logical_class_reserve_min_remaining_columns",
            "logical_class_reserve_applied_count_mean",
            "logical_class_reserve_replaced_state_count_mean",
            "logical_class_quota_top_classes",
            "logical_class_quota_reserved_slots",
            "logical_class_quota_min_remaining_columns",
            "logical_class_quota_applied_count_mean",
            "logical_class_quota_kept_state_count_mean",
            "lineage_reserve_checkpoint_stride",
            "lineage_reserve_reserved_slots",
            "lineage_reserve_applied_count_mean",
            "lineage_reserve_kept_state_count_mean",
            "logical_rerank_columns",
            "logical_rerank_shortlist_size",
            "logical_rerank_min_classes",
            "logical_rerank_state_budget",
            "logical_rerank_transition_budget",
            "logical_rerank_checkpoint_stride",
            "logical_rerank_max_passes",
            "logical_rerank_mode",
            "final_logical_select_mode",
            "final_logical_select_rep_cost_weight",
            "final_logical_select_max_log_mass_gap",
            "final_logical_select_rank2_viterbi_tolerance",
            "final_logical_select_gate_triggered_count",
            "terminal_top_log_mass_gap_mean",
            "terminal_top_log_mass_gap_p99",
            "track_best_path",
            "logical_rerank_pass_count_mean",
            "delayed_pruning_trigger_count_mean",
            "delayed_pruning_active_prune_count_mean",
            "delayed_pruning_peak_beam_size_max",
            "selective_secondary_trigger_count_mean",
            "selective_secondary_changed_count_mean",
            "selective_secondary_reranked_state_count_mean",
            "selective_local_lookahead_trigger_count_mean",
            "selective_local_lookahead_changed_count_mean",
            "selective_local_lookahead_candidate_count_mean",
            "selective_local_lookahead_extra_work_mean",
            "pruning_replay_attempt_count_mean",
            "pruning_replay_applied_count_mean",
            "pruning_replay_replaced_state_count_mean",
            "pruning_replay_replayed_column_count_mean",
            "pruning_replay_extra_transition_evals_mean",
            "pruning_replay_replaced_states_per_apply_mean",
            "splice_enabled_count",
            "splice_changed_count",
            "splice_fixed_count",
            "splice_broken_count",
            "splice_unchanged_failure_count",
            "baseline_fail_total",
            "baseline_logical_fail",
            "baseline_syndrome_fail",
            "baseline_exception_fail",
            "splice_fail_total",
            "splice_logical_fail",
            "splice_syndrome_fail",
            "splice_exception_fail",
            "splice_truth_present_but_not_selected_count",
            "splice_candidate_missing_truth_count",
            "splice_candidate_count_mean",
            "splice_candidate_logical_class_count_mean",
            "splice_cut_count_mean",
            "splice_finite_cut_fraction_mean",
            "splice_missing_support_fraction_mean",
            "splice_hit_count_mean",
            "shots",
            "committee_selected_forward_count",
            "committee_selected_backward_count",
            "fail_total",
            "logical_fail",
            "logical_fail_truth_missing_terminal",
            "logical_fail_truth_present_but_not_selected",
            "truth_terminal_present_count",
            "truth_terminal_log_mass_rank_mean_present",
            "truth_terminal_best_viterbi_rank_mean_present",
            "truth_terminal_log_mass_rank1_count",
            "truth_terminal_best_viterbi_rank1_count",
            "syndrome_fail",
            "exception_fail",
            "fer",
            "fer_per_round",
            "discard_step_count_mean",
            "cumulative_discarded_prefix_mass_mean",
            "cumulative_discarded_prefix_mass_p99",
            "max_discarded_prefix_mass_mean",
            "max_discarded_prefix_mass_p99",
            "mean_discarded_prefix_mass_mean",
            "mean_discarded_prefix_mass_p99",
            "max_discarded_prefix_fraction_mean",
            "max_discarded_prefix_fraction_p99",
            "mean_discarded_prefix_fraction_mean",
            "mean_discarded_prefix_fraction_p99",
            "truth_logical_discard_step_count_mean",
            "cumulative_truth_logical_discarded_prefix_mass_mean",
            "cumulative_truth_logical_discarded_prefix_mass_p99",
            "max_truth_logical_discarded_prefix_mass_mean",
            "max_truth_logical_discarded_prefix_mass_p99",
            "mean_truth_logical_discarded_prefix_mass_mean",
            "mean_truth_logical_discarded_prefix_mass_p99",
            "max_truth_logical_discarded_prefix_fraction_mean",
            "max_truth_logical_discarded_prefix_fraction_p99",
            "mean_truth_logical_discarded_prefix_fraction_mean",
            "mean_truth_logical_discarded_prefix_fraction_p99",
            "decode_s_mean",
            "decode_s_p99",
            "transition_evals_mean",
            "transition_evals_p99",
            "lookahead_transition_evals_mean",
            "lookahead_transition_evals_p99",
            "transition_evals_total_mean",
            "transition_evals_total_p99",
            "transition_evals_physical_total_mean",
            "transition_evals_physical_total_p99",
            "us_per_transition_mean",
            "us_per_transition_physical_mean",
            "us_per_column_mean",
            "mean_states_mean",
            "mean_states_p99",
            "merge_events_total_mean",
            "merge_events_per_column_mean",
            "closure_rejects_total_mean",
            "closure_rejects_per_column_mean",
            "top_log_mass_incoming_per_column_mean",
            "top_log_mass_merge_per_column_mean",
            "top_viterbi_incoming_per_column_mean",
            "top_viterbi_merge_per_column_mean",
            "winner_path_incoming_per_column_mean",
            "winner_path_merge_per_column_mean",
            "max_states_seen",
            "noisy_rounds",
            "total_rounds",
            "matrix_rows",
            "matrix_cols",
            "logical_rows",
            "edge_count",
            "frontier_max_active_detectors",
            ],
            summary_rows,
        ),
    )
    _write_selective_local_lookahead_artifacts(
        per_shot_rows=per_shot_rows,
        summary_rows=summary_rows,
        out_dir=out_dir,
    )

    beam_baselines = _beam_baselines_for_backend(
        backend=str(args.backend),
        p_location=float(args.p_location),
        scope=str(args.scope),
    )
    if bool(args.write_plots):
        _plot_fer_vs_beam(
            summary_rows=summary_rows,
            beam_baselines=beam_baselines,
            out_path=out_dir / "fig_progressive_fer_vs_beam.png",
            value_key="fer",
            ylabel="strict full logical FER",
            round_count=int(family.noisy_rounds),
        )
        _plot_fer_vs_beam(
            summary_rows=summary_rows,
            beam_baselines=beam_baselines,
            out_path=out_dir / "fig_progressive_fer_per_round_vs_beam.png",
            value_key="fer_per_round",
            ylabel="strict full logical FER per round",
            round_count=int(family.noisy_rounds),
        )
        _plot_pruning_diagnostics_vs_beam(
            summary_rows=summary_rows,
            out_path=out_dir / "fig_progressive_pruning_diagnostics_vs_beam.png",
        )
        _plot_pruning_diagnostics_vs_beam(
            summary_rows=summary_rows,
            out_path=out_dir / "fig_progressive_truth_logical_pruning_diagnostics_vs_beam.png",
            left_key="cumulative_truth_logical_discarded_prefix_mass_mean",
            right_key="max_truth_logical_discarded_prefix_fraction_mean",
            left_ylabel="mean cumulative true-class discarded mass",
            right_ylabel="mean max true-class discarded fraction",
            left_title="True-Class Pruning-Loss",
            right_title="Worst-Step True-Class Discard Fraction",
        )
    if bool(args.write_report):
        _write_report(
            out_dir=out_dir,
            family=family,
            summary_rows=summary_rows,
            beam_baselines=beam_baselines,
            p_location=float(args.p_location),
            shots=int(args.shots),
            backend=str(args.backend),
            terminal_failedrank_analysis=terminal_failedrank_analysis,
        )

    metadata["status"] = "completed"
    metadata["elapsed_s"] = float(time.time() - start)
    metadata["completed_shards"] = int(len(tasks))
    metadata["completed_rows"] = int(len(per_shot_rows))
    if terminal_failedrank_analysis is not None:
        terminal_failedrank_aggregate = dict(terminal_failedrank_analysis.get("aggregate", {}))
        metadata["terminal_failedrank_analysis"] = {
            "failure_mode_filter": terminal_failedrank_aggregate.get("failure_mode_filter"),
            "row_count": int(terminal_failedrank_aggregate.get("row_count", 0)),
            "monotone_mvc_certificate_count": int(
                terminal_failedrank_aggregate.get("monotone_mvc_certificate_count", 0)
            ),
            "truth_delta_gt_winner_count": int(
                terminal_failedrank_aggregate.get("truth_delta_gt_winner_count", 0)
            ),
            "summary_csv": terminal_failedrank_aggregate.get("summary_csv"),
            "aggregate_json": terminal_failedrank_aggregate.get("aggregate_json"),
            "plot_path": terminal_failedrank_aggregate.get("plot_path"),
        }
    _write_json(metadata_path, metadata)
    print(f"[done] results_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()
