from __future__ import annotations

import csv
import json
import math
import os
import shlex
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - exercised in the repo toolchain
    from grosscode.utils import pandas_lite as pd

from grosscode.decoders.api import DecoderWindow, SplitSectorPriors, SplitSectorSyndrome, SplitXZDecoder
from grosscode.dem.builder import SplitSectorProblem, build_split_sector_problem
from grosscode.utils.gf2 import csr_matvec_mod2
from grosscode.utils.paths import DEFAULT_RESULTS_ROOT, ensure_mplconfigdir


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DECODERS = ("baseline_bposd", "bp", "minsum", "local_round")
SUPPORTED_DECODERS = DEFAULT_DECODERS + (
    "separator_wavefront",
    "round_wavefront",
    "relay_minsum",
    "separator_wavefront_sa",
    "round_wavefront_sa",
)


def float_tag(value: float) -> str:
    return str(float(value)).replace("-", "m").replace(".", "p")


def format_seconds(seconds: float) -> str:
    value = max(0.0, float(seconds))
    if value < 60.0:
        return f"{value:.1f}s"
    if value < 3600.0:
        return f"{value / 60.0:.1f}m"
    return f"{value / 3600.0:.2f}h"


def parse_csv_floats(raw: str) -> tuple[float, ...]:
    values = tuple(float(token.strip()) for token in str(raw).split(",") if token.strip())
    if not values:
        raise ValueError("expected at least one float value")
    return values


def parse_csv_decoders(raw: str) -> tuple[str, ...]:
    values = tuple(token.strip() for token in str(raw).split(",") if token.strip())
    if not values:
        raise ValueError("expected at least one decoder name")
    invalid = sorted(set(values) - set(SUPPORTED_DECODERS))
    if invalid:
        raise ValueError(f"unsupported decoders: {invalid}")
    return values


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=np.float64), float(q)))


def _frame_fer_to_per_round_exact(frame_fer: float, noisy_rounds: int) -> float:
    rounds = max(1, int(noisy_rounds))
    clipped = float(max(0.0, min(0.5, frame_fer)))
    return float((1.0 - math.pow(1.0 - 2.0 * clipped, 1.0 / float(rounds))) * 0.5)


def _default_shard_shots(shots_per_point: int, cpus: int) -> int:
    if int(shots_per_point) <= 0:
        raise ValueError("shots_per_point must be positive")
    return max(1, min(int(shots_per_point), int(math.ceil(shots_per_point / max(1, cpus * 2)))))


def _frame_seed(base_seed: int, p_index: int, shot_index: int) -> int:
    return int(base_seed + 1_000_003 * int(p_index) + int(shot_index))


def _stable_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(_stable_json(payload), encoding="utf-8")
    tmp_path.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_df(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    try:
        frame.to_parquet(path.with_suffix(".parquet"), index=False)
    except Exception:
        pass


@dataclass(frozen=True)
class MatchedBenchmarkConfig:
    backend: str = "bravyi_depth7"
    p_values: tuple[float, ...] = (0.003, 0.004)
    shots_per_point: int = 4
    decoders: tuple[str, ...] = DEFAULT_DECODERS
    cpus: str | int = "auto"
    shard_shots: int = 0
    seed: int = 20260312
    baseline_max_iter: int = 25
    baseline_osd_method: str = "OSD_0"
    baseline_osd_order: int = 0
    baseline_minsum_scale: float = 0.75
    bp_max_iter: int = 25
    minsum_max_iter: int = 40
    minsum_scale: float = 0.75
    window_round_radius: int = 1
    window_max_iter: int = 40
    local_round_passes: int = 3
    separator_window_rounds: int = 3
    separator_overlap_rounds: int = 1
    separator_topk: int = 2
    separator_max_branches: int = 4
    separator_max_window_expansions: int = 2
    separator_mean_tail: int = 8
    separator_reliable_shell_hops: int = 1
    separator_reliable_topk: int = 6
    separator_reliable_abs_mean_threshold: float = 6.0
    relay_enable: bool = False
    relay_trigger_residual_stall_rounds: int = 4
    relay_trigger_rebound: int = 1
    relay_gamma_stable: float = 0.15
    relay_gamma_bulk: float = 0.0
    relay_gamma_frontier: float = -0.15
    relay_clip_B: float = 12.0
    relay_frontier_shell_radius: int = 1
    relay_flip_mode: str = "erase"
    relay_flip_kappa: float = 0.25
    relay_flip_eta: float = 0.5
    relay_candidate_score: str = "instability_plus_residual"
    relay_num_legs: int = 1
    relay_leg_iters: int = 8
    triangle_factorization: str = "off"
    layered_schedule: str = "off"
    gauge_descent: bool = False
    gauge_descent_max_iterations: int = 128
    progress_every_shards: int = 1
    results_dir: Path | None = None
    command: str | None = None

    def resolved_cpus(self) -> int:
        if str(self.cpus) == "auto":
            return max(1, int(os.cpu_count() or 1))
        return max(1, int(self.cpus))

    def resolved_shard_shots(self) -> int:
        if int(self.shard_shots) > 0:
            return int(self.shard_shots)
        return _default_shard_shots(int(self.shots_per_point), int(self.resolved_cpus()))

    def run_slug(self) -> str:
        p_slug = "-".join(float_tag(value) for value in self.p_values)
        return (
            f"gross144_matched_{self.backend}_p{p_slug}_s{int(self.shots_per_point)}_seed{int(self.seed)}"
            f"_bo{int(self.baseline_osd_order)}_bp{int(self.bp_max_iter)}"
            f"_ms{int(self.minsum_max_iter)}_wr{int(self.window_round_radius)}"
            f"_lp{int(self.local_round_passes)}"
            f"_sww{int(self.separator_window_rounds)}"
            f"_swo{int(self.separator_overlap_rounds)}"
            f"_swk{int(self.separator_topk)}"
            f"_swb{int(self.separator_max_branches)}"
            f"_swx{int(self.separator_max_window_expansions)}"
            f"_swt{int(self.separator_mean_tail)}"
            f"_swh{int(self.separator_reliable_shell_hops)}"
            f"_swr{int(self.separator_reliable_topk)}"
            f"_re{int(bool(self.relay_enable))}"
        )

    def resolved_results_dir(self) -> Path:
        if self.results_dir is not None:
            return Path(self.results_dir)
        return DEFAULT_RESULTS_ROOT / "benchmarks_matched" / self.run_slug()

    def to_json_ready(self) -> dict[str, Any]:
        return {
            "backend": str(self.backend),
            "p_values": [float(value) for value in self.p_values],
            "shots_per_point": int(self.shots_per_point),
            "decoders": [str(name) for name in self.decoders],
            "cpus": str(self.cpus),
            "resolved_cpus": int(self.resolved_cpus()),
            "shard_shots": int(self.resolved_shard_shots()),
            "seed": int(self.seed),
            "baseline_max_iter": int(self.baseline_max_iter),
            "baseline_osd_method": str(self.baseline_osd_method),
            "baseline_osd_order": int(self.baseline_osd_order),
            "baseline_minsum_scale": float(self.baseline_minsum_scale),
            "bp_max_iter": int(self.bp_max_iter),
            "minsum_max_iter": int(self.minsum_max_iter),
            "minsum_scale": float(self.minsum_scale),
            "window_round_radius": int(self.window_round_radius),
            "window_max_iter": int(self.window_max_iter),
            "local_round_passes": int(self.local_round_passes),
            "separator_window_rounds": int(self.separator_window_rounds),
            "separator_overlap_rounds": int(self.separator_overlap_rounds),
            "separator_topk": int(self.separator_topk),
            "separator_max_branches": int(self.separator_max_branches),
            "separator_max_window_expansions": int(self.separator_max_window_expansions),
            "separator_mean_tail": int(self.separator_mean_tail),
            "separator_reliable_shell_hops": int(self.separator_reliable_shell_hops),
            "separator_reliable_topk": int(self.separator_reliable_topk),
            "separator_reliable_abs_mean_threshold": float(self.separator_reliable_abs_mean_threshold),
            "relay_enable": bool(self.relay_enable),
            "relay_trigger_residual_stall_rounds": int(self.relay_trigger_residual_stall_rounds),
            "relay_trigger_rebound": int(self.relay_trigger_rebound),
            "relay_gamma_stable": float(self.relay_gamma_stable),
            "relay_gamma_bulk": float(self.relay_gamma_bulk),
            "relay_gamma_frontier": float(self.relay_gamma_frontier),
            "relay_clip_B": float(self.relay_clip_B),
            "relay_frontier_shell_radius": int(self.relay_frontier_shell_radius),
            "relay_flip_mode": str(self.relay_flip_mode),
            "relay_flip_kappa": float(self.relay_flip_kappa),
            "relay_flip_eta": float(self.relay_flip_eta),
            "relay_candidate_score": str(self.relay_candidate_score),
            "relay_num_legs": int(self.relay_num_legs),
            "relay_leg_iters": int(self.relay_leg_iters),
            "triangle_factorization": str(self.triangle_factorization),
            "layered_schedule": str(self.layered_schedule),
            "gauge_descent": bool(self.gauge_descent),
            "gauge_descent_max_iterations": int(self.gauge_descent_max_iterations),
            "progress_every_shards": int(self.progress_every_shards),
            "results_dir": str(self.resolved_results_dir()),
            "command": str(self.command) if self.command else None,
        }


@dataclass(frozen=True)
class _ShardTask:
    backend: str
    p_index: int
    error_rate: float
    shot_start: int
    shot_stop: int
    seed: int
    decoders: tuple[str, ...]
    baseline_max_iter: int
    baseline_osd_method: str
    baseline_osd_order: int
    baseline_minsum_scale: float
    bp_max_iter: int
    minsum_max_iter: int
    minsum_scale: float
    window_round_radius: int
    window_max_iter: int
    local_round_passes: int
    separator_window_rounds: int
    separator_overlap_rounds: int
    separator_topk: int
    separator_max_branches: int
    separator_max_window_expansions: int
    separator_mean_tail: int
    separator_reliable_shell_hops: int
    separator_reliable_topk: int
    separator_reliable_abs_mean_threshold: float
    relay_enable: bool
    relay_trigger_residual_stall_rounds: int
    relay_trigger_rebound: int
    relay_gamma_stable: float
    relay_gamma_bulk: float
    relay_gamma_frontier: float
    relay_clip_B: float
    relay_frontier_shell_radius: int
    relay_flip_mode: str
    relay_flip_kappa: float
    relay_flip_eta: float
    relay_candidate_score: str
    relay_num_legs: int
    relay_leg_iters: int
    triangle_factorization: str
    layered_schedule: str
    gauge_descent: bool
    gauge_descent_max_iterations: int
    config_json: dict[str, Any]


def _sample_problem_frame(problem: SplitSectorProblem, *, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    faults_x = (rng.random(problem.priors_X.size) < np.asarray(problem.priors_X, dtype=np.float64)).astype(np.uint8)
    faults_z = (rng.random(problem.priors_Z.size) < np.asarray(problem.priors_Z, dtype=np.float64)).astype(np.uint8)
    syndrome_x = csr_matvec_mod2(problem.D_X, faults_x)
    syndrome_z = csr_matvec_mod2(problem.D_Z, faults_z)
    logical_x = csr_matvec_mod2(problem.O_X, faults_x)
    logical_z = csr_matvec_mod2(problem.O_Z, faults_z)
    return syndrome_x, syndrome_z, logical_x, logical_z


def _status_from_outputs(
    *,
    syndrome_x: np.ndarray,
    syndrome_z: np.ndarray,
    truth_obs_x: np.ndarray,
    truth_obs_z: np.ndarray,
    correction_x: np.ndarray,
    correction_z: np.ndarray,
    logical_action_x: np.ndarray,
    logical_action_z: np.ndarray,
    problem: SplitSectorProblem,
) -> str:
    pred_syndrome_x = csr_matvec_mod2(problem.D_X, correction_x)
    pred_syndrome_z = csr_matvec_mod2(problem.D_Z, correction_z)
    if not np.array_equal(pred_syndrome_x, syndrome_x) or not np.array_equal(pred_syndrome_z, syndrome_z):
        return "syndrome_fail"
    if not np.array_equal(np.asarray(logical_action_x, dtype=np.uint8), truth_obs_x):
        return "logical_fail"
    if not np.array_equal(np.asarray(logical_action_z, dtype=np.uint8), truth_obs_z):
        return "logical_fail"
    return "success"


def _build_bposd_decoder(matrix, *, max_iter: int, ms_scale: float, osd_order: int, osd_method: str = "OSD_0"):
    import ldpc  # type: ignore

    base_kwargs = {
        "error_rate": 0.05,
        "max_iter": int(max_iter),
        "bp_method": "minimum_sum",
        "ms_scaling_factor": float(ms_scale),
        "osd_method": str(osd_method),
        "osd_order": int(osd_order),
    }
    try:
        return ldpc.BpOsdDecoder(
            matrix,
            schedule="serial",
            omp_thread_count=1,
            input_vector_type="syndrome",
            **base_kwargs,
        )
    except TypeError:
        return ldpc.BpOsdDecoder(matrix, **base_kwargs)


def _decode_sector_bposd(
    *,
    matrix,
    observables,
    syndrome: np.ndarray,
    priors: np.ndarray,
    max_iter: int,
    ms_scale: float,
    osd_method: str,
    osd_order: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    decoder = _build_bposd_decoder(
        matrix,
        max_iter=max_iter,
        ms_scale=ms_scale,
        osd_method=str(osd_method),
        osd_order=osd_order,
    )
    if hasattr(decoder, "update_channel_probs"):
        decoder.update_channel_probs(np.asarray(priors, dtype=np.float64))
    correction = np.asarray(decoder.decode(np.asarray(syndrome, dtype=np.uint8).copy()), dtype=np.uint8).reshape(-1) & 1
    logical = csr_matvec_mod2(observables, correction)
    return correction, logical, {
        "iterations": int(getattr(decoder, "iter", 0)),
        "converged": bool(getattr(decoder, "converge", False)),
        "syndrome_ok": bool(np.array_equal(csr_matvec_mod2(matrix, correction), syndrome)),
        "logical_weight": int(np.asarray(logical, dtype=np.uint8).sum()),
        "osd_method": str(osd_method),
        "osd_order": int(osd_order),
        "method": "bposd",
    }


def _decode_frame_baseline(
    *,
    problem: SplitSectorProblem,
    syndrome: SplitSectorSyndrome,
    priors: SplitSectorPriors,
    truth_obs_x: np.ndarray,
    truth_obs_z: np.ndarray,
    max_iter: int,
    ms_scale: float,
    osd_method: str,
    osd_order: int,
) -> tuple[str, dict[str, Any]]:
    correction_x, logical_x, diag_x = _decode_sector_bposd(
        matrix=problem.D_X,
        observables=problem.O_X,
        syndrome=np.asarray(syndrome.X, dtype=np.uint8),
        priors=np.asarray(priors.X, dtype=np.float64),
        max_iter=max_iter,
        ms_scale=ms_scale,
        osd_method=str(osd_method),
        osd_order=osd_order,
    )
    correction_z, logical_z, diag_z = _decode_sector_bposd(
        matrix=problem.D_Z,
        observables=problem.O_Z,
        syndrome=np.asarray(syndrome.Z, dtype=np.uint8),
        priors=np.asarray(priors.Z, dtype=np.float64),
        max_iter=max_iter,
        ms_scale=ms_scale,
        osd_method=str(osd_method),
        osd_order=osd_order,
    )
    status = _status_from_outputs(
        syndrome_x=np.asarray(syndrome.X, dtype=np.uint8),
        syndrome_z=np.asarray(syndrome.Z, dtype=np.uint8),
        truth_obs_x=np.asarray(truth_obs_x, dtype=np.uint8),
        truth_obs_z=np.asarray(truth_obs_z, dtype=np.uint8),
        correction_x=correction_x,
        correction_z=correction_z,
        logical_action_x=logical_x,
        logical_action_z=logical_z,
        problem=problem,
    )
    return status, {
        "decoder": "baseline_bposd",
        "sector_X": diag_x,
        "sector_Z": diag_z,
        "logical_frame_action_X": logical_x,
        "logical_frame_action_Z": logical_z,
    }


def _decode_frame_with_runner(
    *,
    runner: SplitXZDecoder,
    problem: SplitSectorProblem,
    syndrome: SplitSectorSyndrome,
    priors: SplitSectorPriors,
    truth_obs_x: np.ndarray,
    truth_obs_z: np.ndarray,
    decoder_name: str,
    shot_seed: int,
    bp_max_iter: int,
    minsum_max_iter: int,
    minsum_scale: float,
    window: DecoderWindow,
) -> tuple[str, dict[str, Any]]:
    runner.seed = int(shot_seed)
    if decoder_name == "bp":
        result = runner.decode_split_xz(syndrome, priors, window=None, decoder="bp")
    elif decoder_name == "minsum":
        result = runner.decode_split_xz(syndrome, priors, window=window, decoder="minsum")
    elif decoder_name == "local_round":
        result = runner.decode_split_xz(syndrome, priors, window=window, decoder="local_round")
    elif decoder_name == "separator_wavefront":
        result = runner.decode_split_xz(syndrome, priors, window=window, decoder="separator_wavefront")
    elif decoder_name == "separator_wavefront_sa":
        result = runner.decode_split_xz(syndrome, priors, window=window, decoder="separator_wavefront_sa")
    elif decoder_name == "round_wavefront":
        result = runner.decode_split_xz(syndrome, priors, window=window, decoder="round_wavefront")
    elif decoder_name == "round_wavefront_sa":
        result = runner.decode_split_xz(syndrome, priors, window=window, decoder="round_wavefront_sa")
    elif decoder_name == "relay_minsum":
        result = runner.decode_split_xz(syndrome, priors, window=window, decoder="relay_minsum")
    else:
        raise ValueError(f"unsupported decoder_name={decoder_name}")
    status = _status_from_outputs(
        syndrome_x=np.asarray(syndrome.X, dtype=np.uint8),
        syndrome_z=np.asarray(syndrome.Z, dtype=np.uint8),
        truth_obs_x=np.asarray(truth_obs_x, dtype=np.uint8),
        truth_obs_z=np.asarray(truth_obs_z, dtype=np.uint8),
        correction_x=result.correction_X,
        correction_z=result.correction_Z,
        logical_action_x=result.logical_frame_action_X,
        logical_action_z=result.logical_frame_action_Z,
        problem=problem,
    )
    return status, result.diagnostics


def _run_one_decoder(
    *,
    runner: SplitXZDecoder,
    decoder_name: str,
    problem: SplitSectorProblem,
    syndrome: SplitSectorSyndrome,
    priors: SplitSectorPriors,
    truth_obs_x: np.ndarray,
    truth_obs_z: np.ndarray,
    shot_seed: int,
    task: _ShardTask,
    window: DecoderWindow,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        if decoder_name == "baseline_bposd":
            status, diagnostics = _decode_frame_baseline(
                problem=problem,
                syndrome=syndrome,
                priors=priors,
                truth_obs_x=truth_obs_x,
                truth_obs_z=truth_obs_z,
                max_iter=int(task.baseline_max_iter),
                ms_scale=float(task.baseline_minsum_scale),
                osd_method=str(task.baseline_osd_method),
                osd_order=int(task.baseline_osd_order),
            )
        else:
            status, diagnostics = _decode_frame_with_runner(
                runner=runner,
                problem=problem,
                syndrome=syndrome,
                priors=priors,
                truth_obs_x=truth_obs_x,
                truth_obs_z=truth_obs_z,
                decoder_name=decoder_name,
                shot_seed=shot_seed,
                bp_max_iter=int(task.bp_max_iter),
                minsum_max_iter=int(task.minsum_max_iter),
                minsum_scale=float(task.minsum_scale),
                window=window,
            )
        elapsed_ms = 1000.0 * (time.perf_counter() - started)
        diag_x = dict(diagnostics.get("sector_X", {}))
        diag_z = dict(diagnostics.get("sector_Z", {}))
        return {
            "decoder": str(decoder_name),
            "status": str(status),
            "decode_ms": float(elapsed_ms),
            "iterations_x": int(diag_x.get("iterations", 0)),
            "iterations_z": int(diag_z.get("iterations", 0)),
            "iterations_total": int(diag_x.get("iterations", 0)) + int(diag_z.get("iterations", 0)),
            "work_ms_iters": int(diag_x.get("iterations", 0)) + int(diag_z.get("iterations", 0)),
            "converged_x": int(bool(diag_x.get("converged", False))),
            "converged_z": int(bool(diag_z.get("converged", False))),
            "frame_converged": int(bool(diag_x.get("converged", False) and diag_z.get("converged", False))),
            "syndrome_ok_x": int(bool(diag_x.get("syndrome_ok", False))),
            "syndrome_ok_z": int(bool(diag_z.get("syndrome_ok", False))),
            "frame_syndrome_ok": int(bool(diag_x.get("syndrome_ok", False) and diag_z.get("syndrome_ok", False))),
            "rescue_used_x": int(bool(diag_x.get("rescue_used", False))),
            "rescue_used_z": int(bool(diag_z.get("rescue_used", False))),
            "frame_rescue_used": int(bool(diag_x.get("rescue_used", False) or diag_z.get("rescue_used", False))),
            "active_columns_x": int(diag_x.get("active_columns", 0)),
            "active_columns_z": int(diag_z.get("active_columns", 0)),
            "residual_weight_final_x": int(diag_x.get("residual_weight_final", 0)),
            "residual_weight_final_z": int(diag_z.get("residual_weight_final", 0)),
            "triangle_catalog_size_x": int(diag_x.get("triangle_catalog_size", 0)),
            "triangle_catalog_size_z": int(diag_z.get("triangle_catalog_size", 0)),
            "triangle_selected_count_x": int(diag_x.get("triangle_selected_count", 0)),
            "triangle_selected_count_z": int(diag_z.get("triangle_selected_count", 0)),
            "triangle_reduced_count_x": int(diag_x.get("reduced_triangle_count", 0)),
            "triangle_reduced_count_z": int(diag_z.get("reduced_triangle_count", 0)),
            "triangle_residual_count_x": int(diag_x.get("residual_triangle_count", 0)),
            "triangle_residual_count_z": int(diag_z.get("residual_triangle_count", 0)),
            "schedule_direction_x": str(diag_x.get("schedule_direction_resolved", "")),
            "schedule_direction_z": str(diag_z.get("schedule_direction_resolved", "")),
            "gauge_descent_moves_x": int(diag_x.get("gauge_descent_accepted_moves", 0)),
            "gauge_descent_moves_z": int(diag_z.get("gauge_descent_accepted_moves", 0)),
            "exception": "",
        }
    except Exception as exc:
        return {
            "decoder": str(decoder_name),
            "status": "exception_fail",
            "decode_ms": float(1000.0 * (time.perf_counter() - started)),
            "iterations_x": 0,
            "iterations_z": 0,
            "iterations_total": 0,
            "work_ms_iters": 0,
            "converged_x": 0,
            "converged_z": 0,
            "frame_converged": 0,
            "syndrome_ok_x": 0,
            "syndrome_ok_z": 0,
            "frame_syndrome_ok": 0,
            "rescue_used_x": 0,
            "rescue_used_z": 0,
            "frame_rescue_used": 0,
            "active_columns_x": 0,
            "active_columns_z": 0,
            "residual_weight_final_x": -1,
            "residual_weight_final_z": -1,
            "triangle_catalog_size_x": 0,
            "triangle_catalog_size_z": 0,
            "triangle_selected_count_x": 0,
            "triangle_selected_count_z": 0,
            "triangle_reduced_count_x": 0,
            "triangle_reduced_count_z": 0,
            "triangle_residual_count_x": 0,
            "triangle_residual_count_z": 0,
            "schedule_direction_x": "",
            "schedule_direction_z": "",
            "gauge_descent_moves_x": 0,
            "gauge_descent_moves_z": 0,
            "exception": repr(exc),
        }


def _run_shard(task: _ShardTask) -> dict[str, Any]:
    ensure_mplconfigdir()
    problem = build_split_sector_problem(backend=str(task.backend), error_rate=float(task.error_rate))
    window = DecoderWindow(
        round_radius=int(task.window_round_radius),
        max_passes=int(task.local_round_passes),
        max_iter=int(task.window_max_iter),
        separator_window_rounds=int(task.separator_window_rounds),
        separator_overlap_rounds=int(task.separator_overlap_rounds),
        separator_topk=int(task.separator_topk),
        separator_max_branches=int(task.separator_max_branches),
        separator_max_window_expansions=int(task.separator_max_window_expansions),
        separator_mean_tail=int(task.separator_mean_tail),
        separator_reliable_shell_hops=int(task.separator_reliable_shell_hops),
        separator_reliable_topk=int(task.separator_reliable_topk),
        separator_reliable_abs_mean_threshold=float(task.separator_reliable_abs_mean_threshold),
        relay_enable=bool(task.relay_enable),
        relay_trigger_residual_stall_rounds=int(task.relay_trigger_residual_stall_rounds),
        relay_trigger_rebound=int(task.relay_trigger_rebound),
        relay_gamma_stable=float(task.relay_gamma_stable),
        relay_gamma_bulk=float(task.relay_gamma_bulk),
        relay_gamma_frontier=float(task.relay_gamma_frontier),
        relay_clip_B=float(task.relay_clip_B),
        relay_frontier_shell_radius=int(task.relay_frontier_shell_radius),
        relay_flip_mode=str(task.relay_flip_mode),
        relay_flip_kappa=float(task.relay_flip_kappa),
        relay_flip_eta=float(task.relay_flip_eta),
        relay_candidate_score=str(task.relay_candidate_score),
        relay_num_legs=int(task.relay_num_legs),
        relay_leg_iters=int(task.relay_leg_iters),
        triangle_factorization=str(task.triangle_factorization),
        layered_schedule=str(task.layered_schedule),
        gauge_descent=bool(task.gauge_descent),
        gauge_descent_max_iterations=int(task.gauge_descent_max_iterations),
    )
    runner = SplitXZDecoder(
        problem,
        bp_max_iter=int(task.bp_max_iter),
        minsum_max_iter=int(task.minsum_max_iter),
        minsum_scale=float(task.minsum_scale),
        seed=int(task.seed),
    )
    rows: list[dict[str, Any]] = []
    for shot_index in range(int(task.shot_start), int(task.shot_stop)):
        frame_seed = _frame_seed(int(task.seed), int(task.p_index), int(shot_index))
        syndrome_x, syndrome_z, truth_obs_x, truth_obs_z = _sample_problem_frame(problem, seed=frame_seed)
        syndrome = SplitSectorSyndrome(X=syndrome_x, Z=syndrome_z)
        priors = SplitSectorPriors(X=problem.priors_X, Z=problem.priors_Z)
        for decoder_name in task.decoders:
            row = _run_one_decoder(
                runner=runner,
                decoder_name=str(decoder_name),
                problem=problem,
                syndrome=syndrome,
                priors=priors,
                truth_obs_x=truth_obs_x,
                truth_obs_z=truth_obs_z,
                shot_seed=frame_seed,
                task=task,
                window=window,
            )
            row.update(
                {
                    "p": float(task.error_rate),
                    "shot": int(shot_index),
                    "frame_seed": int(frame_seed),
                    "backend": str(task.backend),
                    "noisy_rounds": int(problem.metadata_X.noisy_rounds),
                    "perfect_rounds": int(problem.metadata_X.total_rounds - problem.metadata_X.noisy_rounds),
                    "detectors_x": int(problem.D_X.shape[0]),
                    "variables_x": int(problem.D_X.shape[1]),
                    "detectors_z": int(problem.D_Z.shape[0]),
                    "variables_z": int(problem.D_Z.shape[1]),
                }
            )
            rows.append(row)
    return {
        "p": float(task.error_rate),
        "p_index": int(task.p_index),
        "shot_start": int(task.shot_start),
        "shot_stop": int(task.shot_stop),
        "rows": rows,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": task.config_json,
    }


def _shard_path(run_dir: Path, *, error_rate: float, shot_start: int, shot_stop: int) -> Path:
    return run_dir / "shards" / f"p_{float_tag(error_rate)}" / f"shots_{int(shot_start):05d}_{int(shot_stop):05d}.json"


def _build_tasks(config: MatchedBenchmarkConfig) -> list[_ShardTask]:
    tasks: list[_ShardTask] = []
    shard_shots = int(config.resolved_shard_shots())
    for p_index, error_rate in enumerate(config.p_values):
        for shot_start in range(0, int(config.shots_per_point), int(shard_shots)):
            shot_stop = min(int(config.shots_per_point), int(shot_start) + int(shard_shots))
            tasks.append(
                _ShardTask(
                    backend=str(config.backend),
                    p_index=int(p_index),
                    error_rate=float(error_rate),
                    shot_start=int(shot_start),
                    shot_stop=int(shot_stop),
                    seed=int(config.seed),
                    decoders=tuple(str(name) for name in config.decoders),
                    baseline_max_iter=int(config.baseline_max_iter),
                    baseline_osd_method=str(config.baseline_osd_method),
                    baseline_osd_order=int(config.baseline_osd_order),
                    baseline_minsum_scale=float(config.baseline_minsum_scale),
                    bp_max_iter=int(config.bp_max_iter),
                    minsum_max_iter=int(config.minsum_max_iter),
                    minsum_scale=float(config.minsum_scale),
                    window_round_radius=int(config.window_round_radius),
                    window_max_iter=int(config.window_max_iter),
                    local_round_passes=int(config.local_round_passes),
                    separator_window_rounds=int(config.separator_window_rounds),
                    separator_overlap_rounds=int(config.separator_overlap_rounds),
                    separator_topk=int(config.separator_topk),
                    separator_max_branches=int(config.separator_max_branches),
                    separator_max_window_expansions=int(config.separator_max_window_expansions),
                    separator_mean_tail=int(config.separator_mean_tail),
                    separator_reliable_shell_hops=int(config.separator_reliable_shell_hops),
                    separator_reliable_topk=int(config.separator_reliable_topk),
                    separator_reliable_abs_mean_threshold=float(config.separator_reliable_abs_mean_threshold),
                    relay_enable=bool(config.relay_enable),
                    relay_trigger_residual_stall_rounds=int(config.relay_trigger_residual_stall_rounds),
                    relay_trigger_rebound=int(config.relay_trigger_rebound),
                    relay_gamma_stable=float(config.relay_gamma_stable),
                    relay_gamma_bulk=float(config.relay_gamma_bulk),
                    relay_gamma_frontier=float(config.relay_gamma_frontier),
                    relay_clip_B=float(config.relay_clip_B),
                    relay_frontier_shell_radius=int(config.relay_frontier_shell_radius),
                    relay_flip_mode=str(config.relay_flip_mode),
                    relay_flip_kappa=float(config.relay_flip_kappa),
                    relay_flip_eta=float(config.relay_flip_eta),
                    relay_candidate_score=str(config.relay_candidate_score),
                    relay_num_legs=int(config.relay_num_legs),
                    relay_leg_iters=int(config.relay_leg_iters),
                    triangle_factorization=str(config.triangle_factorization),
                    layered_schedule=str(config.layered_schedule),
                    gauge_descent=bool(config.gauge_descent),
                    gauge_descent_max_iterations=int(config.gauge_descent_max_iterations),
                    config_json=config.to_json_ready(),
                )
            )
    return tasks


def _load_shard_rows(path: Path, config_json: dict[str, Any]) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored = dict(payload.get("config", {}))
    if stored != config_json:
        raise ValueError(f"config mismatch for existing shard {path}")
    return [dict(row) for row in payload.get("rows", [])]


def _aggregate_summary(per_shot: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if per_shot.empty:
        empty = pd.DataFrame(
            columns=[
                "decoder",
                "p",
                "shots",
                "success",
                "logical_fail",
                "syndrome_fail",
                "exception_fail",
                "fail_total",
                "fer",
                "noisy_rounds",
                "fer_per_round",
                "decode_ms_mean",
                "decode_ms_p95",
                "iterations_mean",
                "iterations_p95",
                "work_ms_iters_mean",
                "work_ms_iters_p95",
                "converged_rate",
                "syndrome_ok_rate",
                "triangle_catalog_size_x",
                "triangle_catalog_size_z",
                "triangle_selected_count_x",
                "triangle_selected_count_z",
                "triangle_reduced_count_x_mean",
                "triangle_reduced_count_z_mean",
                "triangle_residual_count_x_mean",
                "triangle_residual_count_z_mean",
                "gauge_descent_moves_x_mean",
                "gauge_descent_moves_z_mean",
                "schedule_direction_x",
                "schedule_direction_z",
            ]
        )
        return empty.copy(), empty.copy()

    rows_by_point: list[dict[str, Any]] = []
    rows_overall: list[dict[str, Any]] = []
    for (decoder, p_value), group in per_shot.groupby(["decoder", "p"], sort=True):
        shots = int(len(group))
        logical_fail = int((group["status"] == "logical_fail").sum())
        syndrome_fail = int((group["status"] == "syndrome_fail").sum())
        exception_fail = int((group["status"] == "exception_fail").sum())
        fail_total = int(logical_fail + syndrome_fail + exception_fail)
        noisy_rounds = int(group["noisy_rounds"].iloc[0])
        fer = float(fail_total) / float(max(1, shots))
        rows_by_point.append(
            {
                "decoder": str(decoder),
                "p": float(p_value),
                "shots": int(shots),
                "success": int((group["status"] == "success").sum()),
                "logical_fail": int(logical_fail),
                "syndrome_fail": int(syndrome_fail),
                "exception_fail": int(exception_fail),
                "fail_total": int(fail_total),
                "fer": float(fer),
                "noisy_rounds": int(noisy_rounds),
                "fer_per_round": _frame_fer_to_per_round_exact(float(fer), int(noisy_rounds)),
                "decode_ms_mean": float(group["decode_ms"].mean()),
                "decode_ms_p95": float(group["decode_ms"].quantile(0.95)),
                "iterations_mean": float(group["iterations_total"].mean()),
                "iterations_p95": float(group["iterations_total"].quantile(0.95)),
                "work_ms_iters_mean": float(group["work_ms_iters"].mean()),
                "work_ms_iters_p95": float(group["work_ms_iters"].quantile(0.95)),
                "converged_rate": float(group["frame_converged"].mean()),
                "syndrome_ok_rate": float(group["frame_syndrome_ok"].mean()),
                "triangle_catalog_size_x": int(group["triangle_catalog_size_x"].max()),
                "triangle_catalog_size_z": int(group["triangle_catalog_size_z"].max()),
                "triangle_selected_count_x": int(group["triangle_selected_count_x"].max()),
                "triangle_selected_count_z": int(group["triangle_selected_count_z"].max()),
                "triangle_reduced_count_x_mean": float(group["triangle_reduced_count_x"].mean()),
                "triangle_reduced_count_z_mean": float(group["triangle_reduced_count_z"].mean()),
                "triangle_residual_count_x_mean": float(group["triangle_residual_count_x"].mean()),
                "triangle_residual_count_z_mean": float(group["triangle_residual_count_z"].mean()),
                "gauge_descent_moves_x_mean": float(group["gauge_descent_moves_x"].mean()),
                "gauge_descent_moves_z_mean": float(group["gauge_descent_moves_z"].mean()),
                "schedule_direction_x": str(group["schedule_direction_x"].iloc[0]),
                "schedule_direction_z": str(group["schedule_direction_z"].iloc[0]),
            }
        )
    point_df = pd.DataFrame(rows_by_point).sort_values(["p", "decoder"]).reset_index(drop=True)

    for decoder, group in per_shot.groupby("decoder", sort=True):
        shots = int(len(group))
        logical_fail = int((group["status"] == "logical_fail").sum())
        syndrome_fail = int((group["status"] == "syndrome_fail").sum())
        exception_fail = int((group["status"] == "exception_fail").sum())
        fail_total = int(logical_fail + syndrome_fail + exception_fail)
        noisy_rounds = int(group["noisy_rounds"].iloc[0])
        fer = float(fail_total) / float(max(1, shots))
        rows_overall.append(
            {
                "decoder": str(decoder),
                "points": int(group["p"].nunique()),
                "shots": int(shots),
                "success": int((group["status"] == "success").sum()),
                "logical_fail": int(logical_fail),
                "syndrome_fail": int(syndrome_fail),
                "exception_fail": int(exception_fail),
                "fail_total": int(fail_total),
                "fer": float(fer),
                "noisy_rounds": int(noisy_rounds),
                "fer_per_round": _frame_fer_to_per_round_exact(float(fer), int(noisy_rounds)),
                "decode_ms_mean": float(group["decode_ms"].mean()),
                "decode_ms_p95": float(group["decode_ms"].quantile(0.95)),
                "iterations_mean": float(group["iterations_total"].mean()),
                "iterations_p95": float(group["iterations_total"].quantile(0.95)),
                "work_ms_iters_mean": float(group["work_ms_iters"].mean()),
                "work_ms_iters_p95": float(group["work_ms_iters"].quantile(0.95)),
                "converged_rate": float(group["frame_converged"].mean()),
                "syndrome_ok_rate": float(group["frame_syndrome_ok"].mean()),
                "triangle_catalog_size_x": int(group["triangle_catalog_size_x"].max()),
                "triangle_catalog_size_z": int(group["triangle_catalog_size_z"].max()),
                "triangle_selected_count_x": int(group["triangle_selected_count_x"].max()),
                "triangle_selected_count_z": int(group["triangle_selected_count_z"].max()),
                "triangle_reduced_count_x_mean": float(group["triangle_reduced_count_x"].mean()),
                "triangle_reduced_count_z_mean": float(group["triangle_reduced_count_z"].mean()),
                "triangle_residual_count_x_mean": float(group["triangle_residual_count_x"].mean()),
                "triangle_residual_count_z_mean": float(group["triangle_residual_count_z"].mean()),
                "gauge_descent_moves_x_mean": float(group["gauge_descent_moves_x"].mean()),
                "gauge_descent_moves_z_mean": float(group["gauge_descent_moves_z"].mean()),
                "schedule_direction_x": str(group["schedule_direction_x"].iloc[0]),
                "schedule_direction_z": str(group["schedule_direction_z"].iloc[0]),
            }
        )
    overall_df = pd.DataFrame(rows_overall).sort_values(["decode_ms_mean", "decoder"]).reset_index(drop=True)
    return point_df, overall_df


def _write_partial_outputs(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    per_shot_df = pd.DataFrame(rows)
    point_df, overall_df = _aggregate_summary(per_shot_df)
    _write_df(run_dir / "per_shot.partial.csv", per_shot_df)
    _write_df(run_dir / "summary_by_point.partial.csv", point_df)
    _write_df(run_dir / "summary_overall.partial.csv", overall_df)


def _load_existing_rows(run_dir: Path, config_json: dict[str, Any], tasks: list[_ShardTask]) -> tuple[list[dict[str, Any]], list[_ShardTask]]:
    rows: list[dict[str, Any]] = []
    pending: list[_ShardTask] = []
    for task in tasks:
        shard_path = _shard_path(
            run_dir,
            error_rate=float(task.error_rate),
            shot_start=int(task.shot_start),
            shot_stop=int(task.shot_stop),
        )
        if shard_path.exists():
            try:
                rows.extend(_load_shard_rows(shard_path, config_json))
                continue
            except Exception:
                pass
        pending.append(task)
    return rows, pending


def _write_progress_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_task_manifest(run_dir: Path, tasks: list[_ShardTask]) -> None:
    rows = []
    for task in tasks:
        rows.append(
            {
                "backend": str(task.backend),
                "p": float(task.error_rate),
                "shot_start": int(task.shot_start),
                "shot_stop": int(task.shot_stop),
                "shots": int(task.shot_stop - task.shot_start),
                "shard_path": str(
                    _shard_path(
                        run_dir,
                        error_rate=float(task.error_rate),
                        shot_start=int(task.shot_start),
                        shot_stop=int(task.shot_stop),
                    )
                ),
            }
        )
    _write_csv(run_dir / "task_manifest.csv", rows)


def _verify_common_task_definition(config: MatchedBenchmarkConfig) -> dict[str, Any]:
    first_problem = build_split_sector_problem(backend=str(config.backend), error_rate=float(config.p_values[0]))
    expected = {
        "backend": str(config.backend),
        "noisy_rounds": int(first_problem.metadata_X.noisy_rounds),
        "perfect_rounds": int(first_problem.metadata_X.total_rounds - first_problem.metadata_X.noisy_rounds),
        "detectors_x": int(first_problem.D_X.shape[0]),
        "variables_x": int(first_problem.D_X.shape[1]),
        "detectors_z": int(first_problem.D_Z.shape[0]),
        "variables_z": int(first_problem.D_Z.shape[1]),
    }
    for error_rate in config.p_values[1:]:
        problem = build_split_sector_problem(backend=str(config.backend), error_rate=float(error_rate))
        current = {
            "backend": str(config.backend),
            "noisy_rounds": int(problem.metadata_X.noisy_rounds),
            "perfect_rounds": int(problem.metadata_X.total_rounds - problem.metadata_X.noisy_rounds),
            "detectors_x": int(problem.D_X.shape[0]),
            "variables_x": int(problem.D_X.shape[1]),
            "detectors_z": int(problem.D_Z.shape[0]),
            "variables_z": int(problem.D_Z.shape[1]),
        }
        if current != expected:
            raise ValueError(f"matched task definition changed across p grid: expected {expected}, got {current}")
    return expected


def estimate_runtime_range_seconds(config: MatchedBenchmarkConfig) -> tuple[float, float]:
    problem = build_split_sector_problem(backend=str(config.backend), error_rate=float(config.p_values[0]))
    syndrome_x, syndrome_z, truth_obs_x, truth_obs_z = _sample_problem_frame(problem, seed=int(config.seed))
    syndrome = SplitSectorSyndrome(X=syndrome_x, Z=syndrome_z)
    priors = SplitSectorPriors(X=problem.priors_X, Z=problem.priors_Z)
    window = DecoderWindow(
        round_radius=int(config.window_round_radius),
        max_passes=int(config.local_round_passes),
        max_iter=int(config.window_max_iter),
        separator_window_rounds=int(config.separator_window_rounds),
        separator_overlap_rounds=int(config.separator_overlap_rounds),
        separator_topk=int(config.separator_topk),
        separator_max_branches=int(config.separator_max_branches),
        separator_max_window_expansions=int(config.separator_max_window_expansions),
        separator_mean_tail=int(config.separator_mean_tail),
        separator_reliable_shell_hops=int(config.separator_reliable_shell_hops),
        separator_reliable_topk=int(config.separator_reliable_topk),
        separator_reliable_abs_mean_threshold=float(config.separator_reliable_abs_mean_threshold),
        relay_enable=bool(config.relay_enable),
        relay_trigger_residual_stall_rounds=int(config.relay_trigger_residual_stall_rounds),
        relay_trigger_rebound=int(config.relay_trigger_rebound),
        relay_gamma_stable=float(config.relay_gamma_stable),
        relay_gamma_bulk=float(config.relay_gamma_bulk),
        relay_gamma_frontier=float(config.relay_gamma_frontier),
        relay_clip_B=float(config.relay_clip_B),
        relay_frontier_shell_radius=int(config.relay_frontier_shell_radius),
        relay_flip_mode=str(config.relay_flip_mode),
        relay_flip_kappa=float(config.relay_flip_kappa),
        relay_flip_eta=float(config.relay_flip_eta),
        relay_candidate_score=str(config.relay_candidate_score),
        relay_num_legs=int(config.relay_num_legs),
        relay_leg_iters=int(config.relay_leg_iters),
        triangle_factorization=str(config.triangle_factorization),
        layered_schedule=str(config.layered_schedule),
        gauge_descent=bool(config.gauge_descent),
        gauge_descent_max_iterations=int(config.gauge_descent_max_iterations),
    )
    runner = SplitXZDecoder(
        problem,
        bp_max_iter=int(config.bp_max_iter),
        minsum_max_iter=int(config.minsum_max_iter),
        minsum_scale=float(config.minsum_scale),
        seed=int(config.seed),
    )
    timings: list[float] = []
    for decoder_name in config.decoders:
        started = time.perf_counter()
        if decoder_name == "baseline_bposd":
            _decode_frame_baseline(
                problem=problem,
                syndrome=syndrome,
                priors=priors,
                truth_obs_x=truth_obs_x,
                truth_obs_z=truth_obs_z,
                max_iter=int(config.baseline_max_iter),
                ms_scale=float(config.baseline_minsum_scale),
                osd_method=str(config.baseline_osd_method),
                osd_order=int(config.baseline_osd_order),
            )
        else:
            _decode_frame_with_runner(
                runner=runner,
                problem=problem,
                syndrome=syndrome,
                priors=priors,
                truth_obs_x=truth_obs_x,
                truth_obs_z=truth_obs_z,
                decoder_name=str(decoder_name),
                shot_seed=int(config.seed),
                bp_max_iter=int(config.bp_max_iter),
                minsum_max_iter=int(config.minsum_max_iter),
                minsum_scale=float(config.minsum_scale),
                window=window,
            )
        timings.append(1000.0 * (time.perf_counter() - started))
    frame_ms = float(sum(timings))
    total_frames = int(len(config.p_values) * int(config.shots_per_point))
    cpu_scale = max(1.0, 0.65 * float(config.resolved_cpus()))
    estimate = (frame_ms * float(total_frames)) / (1000.0 * cpu_scale)
    return float(max(0.5, 0.75 * estimate)), float(max(1.0, 1.75 * estimate))


def load_run_artifacts(run_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_root = Path(run_dir)
    config = json.loads((run_root / "config.json").read_text(encoding="utf-8"))

    def _read_csv_with_partial_fallback(stem: str) -> pd.DataFrame:
        final_path = run_root / f"{stem}.csv"
        partial_path = run_root / f"{stem}.partial.csv"
        if final_path.exists():
            return pd.read_csv(final_path)
        if partial_path.exists():
            return pd.read_csv(partial_path)
        raise FileNotFoundError(
            f"missing both {final_path} and {partial_path}; "
            "the matched benchmark run may still be in progress or incomplete"
        )

    per_shot = _read_csv_with_partial_fallback("per_shot")
    summary_by_point = _read_csv_with_partial_fallback("summary_by_point")
    summary_overall = _read_csv_with_partial_fallback("summary_overall")
    needs_refresh = (
        "noisy_rounds" not in summary_by_point.columns
        or "fer_per_round" not in summary_by_point.columns
        or "noisy_rounds" not in summary_overall.columns
        or "fer_per_round" not in summary_overall.columns
    )
    if needs_refresh and "noisy_rounds" in per_shot.columns:
        summary_by_point, summary_overall = _aggregate_summary(per_shot)
    return config, per_shot, summary_by_point, summary_overall


def run_matched_benchmark(config: MatchedBenchmarkConfig) -> Path:
    ensure_mplconfigdir()
    if not config.p_values:
        raise ValueError("p_values must be non-empty")
    run_dir = Path(config.resolved_results_dir())
    run_dir.mkdir(parents=True, exist_ok=True)
    config_json = config.to_json_ready()
    config_path = run_dir / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config_json:
            raise ValueError(f"existing run directory has a different config: {run_dir}")
    else:
        _write_json(config_path, config_json)

    task_definition = _verify_common_task_definition(config)
    _write_json(run_dir / "task_definition.json", task_definition)

    tasks = _build_tasks(config)
    _write_task_manifest(run_dir, tasks)
    rows, pending = _load_existing_rows(run_dir, config_json, tasks)
    if rows:
        _write_partial_outputs(run_dir, rows)

    total_shards = int(len(tasks))
    completed_shards = int(total_shards - len(pending))
    completed_frames = int(len({(float(row["p"]), int(row["shot"])) for row in rows}))
    progress_path = run_dir / "progress.jsonl"
    started = time.perf_counter()

    if not pending:
        per_shot_df = pd.DataFrame(rows).sort_values(["p", "shot", "decoder"]).reset_index(drop=True)
        point_df, overall_df = _aggregate_summary(per_shot_df)
        _write_df(run_dir / "per_shot.csv", per_shot_df)
        _write_df(run_dir / "summary_by_point.csv", point_df)
        _write_df(run_dir / "summary_overall.csv", overall_df)
        return run_dir

    workers = int(config.resolved_cpus())
    executor_cls = ProcessPoolExecutor
    executor_note = "process"
    try:
        pool = executor_cls(max_workers=workers)
    except (PermissionError, OSError):
        executor_cls = ThreadPoolExecutor
        executor_note = "thread"
        pool = executor_cls(max_workers=workers)

    print(
        "[start] backend={} p_points={} shots_per_point={} decoders={} workers={} executor={} pending_shards={}".format(
            str(config.backend),
            int(len(config.p_values)),
            int(config.shots_per_point),
            ",".join(config.decoders),
            int(workers),
            executor_note,
            int(len(pending)),
        ),
        flush=True,
    )

    with pool:
        future_map = {pool.submit(_run_shard, task): task for task in pending}
        for future in as_completed(future_map):
            task = future_map[future]
            payload = future.result()
            shard_rows = [dict(row) for row in payload.get("rows", [])]
            shard_path = _shard_path(
                run_dir,
                error_rate=float(task.error_rate),
                shot_start=int(task.shot_start),
                shot_stop=int(task.shot_stop),
            )
            _write_json(shard_path, payload)
            rows.extend(shard_rows)
            completed_shards += 1
            completed_frames += int(task.shot_stop - task.shot_start)
            if (
                int(completed_shards) % max(1, int(config.progress_every_shards)) == 0
                or int(completed_shards) == int(total_shards)
            ):
                elapsed = time.perf_counter() - started
                rate = float(completed_shards) / max(elapsed, 1e-9)
                eta = float(total_shards - completed_shards) / max(rate, 1e-9)
                event = {
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "completed_shards": int(completed_shards),
                    "total_shards": int(total_shards),
                    "completed_frames": int(completed_frames),
                    "total_frames": int(len(config.p_values) * int(config.shots_per_point)),
                    "elapsed_sec": float(elapsed),
                    "eta_sec": float(eta),
                    "latest_p": float(task.error_rate),
                    "latest_shot_start": int(task.shot_start),
                    "latest_shot_stop": int(task.shot_stop),
                }
                _write_progress_event(progress_path, event)
                _write_partial_outputs(run_dir, rows)
                print(
                    "[progress] shards={}/{} frames={}/{} elapsed={} eta={} latest=p={} shots={}..{}".format(
                        int(completed_shards),
                        int(total_shards),
                        int(completed_frames),
                        int(len(config.p_values) * int(config.shots_per_point)),
                        format_seconds(elapsed),
                        format_seconds(eta),
                        float(task.error_rate),
                        int(task.shot_start),
                        int(task.shot_stop),
                    ),
                    flush=True,
                )

    per_shot_df = pd.DataFrame(rows).sort_values(["p", "shot", "decoder"]).reset_index(drop=True)
    point_df, overall_df = _aggregate_summary(per_shot_df)
    _write_df(run_dir / "per_shot.csv", per_shot_df)
    _write_df(run_dir / "summary_by_point.csv", point_df)
    _write_df(run_dir / "summary_overall.csv", overall_df)
    return run_dir


def _format_markdown_table(frame: pd.DataFrame, columns: list[str], float_formats: dict[str, str]) -> str:
    if frame.empty:
        return "No rows.\n"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for _, row in frame.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append("nan")
            elif column in float_formats:
                values.append(format(float(value), float_formats[column]))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def create_plots(run_dir: Path) -> dict[str, Path]:
    ensure_mplconfigdir()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config, _, summary_by_point, summary_overall = load_run_artifacts(run_dir)
    plot_dir = Path(run_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: dict[str, Path] = {}

    fer_fig, fer_ax = plt.subplots(figsize=(7.4, 4.8), dpi=180)
    fer_table_rows: list[dict[str, Any]] = []
    for decoder in summary_by_point["decoder"].drop_duplicates().tolist():
        sub = summary_by_point.loc[summary_by_point["decoder"] == decoder].sort_values("p")
        xs = sub["p"].to_numpy(dtype=float)
        shots = sub["shots"].to_numpy(dtype=float)
        fer_raw = sub["fer"].to_numpy(dtype=float)
        fer_floor = np.minimum(0.5 / np.maximum(shots, 1.0), 1.0)
        fer_ax.plot(xs, np.maximum(fer_raw, fer_floor), marker="o", linewidth=1.8, label=str(decoder))
        fer_table_rows.extend(sub.to_dict("records"))
    fer_ax.set_xscale("log")
    fer_ax.set_yscale("log")
    fer_ax.set_xlabel("Physical location error rate p")
    fer_ax.set_ylabel("Full logical FER")
    fer_ax.set_title("Gross [[144,12,12]] matched split-sector benchmark")
    fer_ax.grid(True, which="both", alpha=0.25)
    fer_ax.legend(loc="best")
    fer_fig.tight_layout()
    fer_path = plot_dir / "fig_fer_vs_p_matched.png"
    fer_fig.savefig(fer_path)
    plt.close(fer_fig)
    plot_paths["fer_vs_p"] = fer_path
    _write_csv(plot_dir / "table_fer_vs_p_matched.csv", fer_table_rows)

    fer_per_round_fig, fer_per_round_ax = plt.subplots(figsize=(7.4, 4.8), dpi=180)
    fer_per_round_table_rows: list[dict[str, Any]] = []
    for decoder in summary_by_point["decoder"].drop_duplicates().tolist():
        sub = summary_by_point.loc[summary_by_point["decoder"] == decoder].sort_values("p")
        xs = sub["p"].to_numpy(dtype=float)
        shots = sub["shots"].to_numpy(dtype=float)
        noisy_rounds = sub["noisy_rounds"].to_numpy(dtype=int)
        fer_raw = sub["fer_per_round"].to_numpy(dtype=float)
        frame_floor = np.minimum(0.5 / np.maximum(shots, 1.0), 1.0)
        fer_floor = np.asarray(
            [_frame_fer_to_per_round_exact(float(frame_value), int(round_value)) for frame_value, round_value in zip(frame_floor, noisy_rounds)],
            dtype=float,
        )
        fer_per_round_ax.plot(xs, np.maximum(fer_raw, fer_floor), marker="o", linewidth=1.8, label=str(decoder))
        fer_per_round_table_rows.extend(sub.to_dict("records"))
    fer_per_round_ax.set_xscale("log")
    fer_per_round_ax.set_yscale("log")
    fer_per_round_ax.set_xlabel("Physical location error rate p")
    fer_per_round_ax.set_ylabel("Full logical FER per syndrome-extraction round")
    fer_per_round_ax.set_title("Gross [[144,12,12]] matched split-sector benchmark (per round)")
    fer_per_round_ax.grid(True, which="both", alpha=0.25)
    fer_per_round_ax.legend(loc="best")
    fer_per_round_fig.tight_layout()
    fer_per_round_path = plot_dir / "fig_fer_per_round_vs_p_matched.png"
    fer_per_round_fig.savefig(fer_per_round_path)
    plt.close(fer_per_round_fig)
    plot_paths["fer_per_round_vs_p"] = fer_per_round_path
    _write_csv(plot_dir / "table_fer_per_round_vs_p_matched.csv", fer_per_round_table_rows)

    runtime_fig, runtime_ax = plt.subplots(figsize=(7.6, 4.8), dpi=180)
    runtime_table_rows: list[dict[str, Any]] = []
    summary_overall = summary_overall.sort_values("decode_ms_mean").reset_index(drop=True)
    palette = ["#2f5d8a", "#48917f", "#c97f35", "#7b6aa8", "#b84a62"]
    bars = runtime_ax.bar(
        summary_overall["decoder"].tolist(),
        summary_overall["decode_ms_mean"].to_numpy(dtype=float),
        color=[palette[index % len(palette)] for index in range(len(summary_overall))],
    )
    for bar, (_, row) in zip(bars, summary_overall.iterrows()):
        runtime_ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(row["decode_ms_mean"]) * 1.03,
            f"{float(row['decode_ms_mean']):.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        runtime_table_rows.append(dict(row))
    runtime_ax.set_yscale("log")
    runtime_ax.set_ylabel("Mean decode time per full frame (ms)")
    runtime_ax.set_xlabel("Decoder")
    runtime_ax.set_title("Matched benchmark runtime by decoder")
    runtime_ax.grid(True, which="both", axis="y", alpha=0.25)
    runtime_fig.tight_layout()
    runtime_path = plot_dir / "fig_runtime_vs_decoder_matched.png"
    runtime_fig.savefig(runtime_path)
    plt.close(runtime_fig)
    plot_paths["runtime_vs_decoder"] = runtime_path
    _write_csv(plot_dir / "table_runtime_vs_decoder_matched.csv", runtime_table_rows)

    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "plots": {name: str(path) for name, path in plot_paths.items()},
    }
    _write_json(plot_dir / "plot_metadata.json", metadata)
    return plot_paths


def write_report(run_dir: Path, report_path: Path) -> Path:
    config, _, summary_by_point, summary_overall = load_run_artifacts(run_dir)
    plot_paths = create_plots(run_dir)
    task_definition_path = Path(run_dir) / "task_definition.json"
    task_definition = json.loads(task_definition_path.read_text(encoding="utf-8")) if task_definition_path.exists() else {}
    decoder_list = ", ".join(str(name) for name in config["decoders"])
    separator_window_rounds = int(config.get("separator_window_rounds", 3))
    separator_overlap_rounds = int(config.get("separator_overlap_rounds", 1))
    separator_topk = int(config.get("separator_topk", 2))
    separator_max_branches = int(config.get("separator_max_branches", 4))
    separator_max_window_expansions = int(config.get("separator_max_window_expansions", 2))
    separator_mean_tail = int(config.get("separator_mean_tail", 8))
    separator_reliable_shell_hops = int(config.get("separator_reliable_shell_hops", 1))
    separator_reliable_topk = int(config.get("separator_reliable_topk", 6))
    separator_reliable_abs_mean_threshold = float(config.get("separator_reliable_abs_mean_threshold", 6.0))
    relay_enable = bool(config.get("relay_enable", False))
    relay_trigger_residual_stall_rounds = int(config.get("relay_trigger_residual_stall_rounds", 4))
    relay_trigger_rebound = int(config.get("relay_trigger_rebound", 1))
    relay_gamma_stable = float(config.get("relay_gamma_stable", 0.15))
    relay_gamma_bulk = float(config.get("relay_gamma_bulk", 0.0))
    relay_gamma_frontier = float(config.get("relay_gamma_frontier", -0.15))
    relay_clip_B = float(config.get("relay_clip_B", 12.0))
    relay_frontier_shell_radius = int(config.get("relay_frontier_shell_radius", 1))
    relay_flip_mode = str(config.get("relay_flip_mode", "erase"))
    relay_flip_kappa = float(config.get("relay_flip_kappa", 0.25))
    relay_flip_eta = float(config.get("relay_flip_eta", 0.5))
    relay_candidate_score = str(config.get("relay_candidate_score", "instability_plus_residual"))
    relay_num_legs = int(config.get("relay_num_legs", 1))
    relay_leg_iters = int(config.get("relay_leg_iters", 8))
    triangle_factorization = str(config.get("triangle_factorization", "off"))
    layered_schedule = str(config.get("layered_schedule", "off"))
    gauge_descent = bool(config.get("gauge_descent", False))
    gauge_descent_max_iterations = int(config.get("gauge_descent_max_iterations", 128))

    best_overall_row = summary_overall.sort_values(["fer", "decode_ms_mean", "decoder"]).iloc[0]
    highest_p = float(max(float(value) for value in config["p_values"]))
    nonbaseline_high_p = summary_by_point.loc[
        (summary_by_point["p"] == highest_p) & (summary_by_point["decoder"] != "baseline_bposd")
    ].sort_values(["fer", "decode_ms_mean", "decoder"])
    best_nonbaseline_high_p = nonbaseline_high_p.iloc[0] if not nonbaseline_high_p.empty else None
    best_per_round_row = summary_overall.sort_values(["fer_per_round", "decode_ms_mean", "decoder"]).iloc[0]
    fastest_row = summary_overall.sort_values(["decode_ms_mean", "fer", "decoder"]).iloc[0]
    runtime_slowest_row = summary_overall.sort_values(["decode_ms_mean", "decoder"], ascending=[False, True]).iloc[0]
    p_list = ", ".join(f"{float(value):.4f}" for value in config["p_values"])
    noisy_rounds = int(task_definition.get("noisy_rounds", summary_by_point["noisy_rounds"].iloc[0]))
    perfect_rounds = int(task_definition.get("perfect_rounds", 1))
    detectors_x = int(task_definition.get("detectors_x", 936))
    variables_x = int(task_definition.get("variables_x", 8784))
    detectors_z = int(task_definition.get("detectors_z", 936))
    variables_z = int(task_definition.get("variables_z", 8784))
    baseline_note = (
        "The matched imported baseline here is the in-repo `ldpc.BpOsdDecoder` path on the same "
        "Stim-derived split-sector DEM tasks. The separate public Bravyi wrapper is intentionally not "
        "merged into this table because its exported `hdec_x/hdec_z` matrices (`1008 x 8857/8785`) are "
        "not the same task family as the matched `D_X/D_Z` matrices (`936 x 8784`)."
    )

    fer_caption = (
        "Figure 1. Full logical frame error rate versus physical location error rate for the first matched "
        "Gross `[[144,12,12]]` public `bravyi_depth7` split-sector benchmark. Every decoder uses the same "
        f"`memory_X` and `memory_Z` Stim-derived split-sector tasks with detector-side matrices "
        f"`D_X = {detectors_x} x {variables_x}` and `D_Z = {detectors_z} x {variables_z}`, the same p-grid `{p_list}`, the same "
        f"`{int(config['shots_per_point'])}` paired shots per point, the same deterministic frame-seed protocol "
        f"(base seed `{int(config['seed'])}`), and the same inferred `{noisy_rounds}` noisy rounds plus `{perfect_rounds}` final perfect "
        "round schedule. The x-axis is physical location error rate `p` on a logarithmic scale; the y-axis is "
        "strict full logical FER on a logarithmic scale with a finite-sample floor only for zero-failure rows. "
        f"The compared decoders are `{decoder_list}`. "
        f"The main quantitative takeaway is that `{best_overall_row['decoder']}` is the only decoder with "
        f"`{int(best_overall_row['fail_total'])}/{int(best_overall_row['shots'])}` failures over the full matched run, "
        + (
            f"and at the harder point `p={highest_p:.4f}` the best non-baseline row is "
            f"`{best_nonbaseline_high_p['decoder']}` with "
            f"`{int(best_nonbaseline_high_p['fail_total'])}/{int(best_nonbaseline_high_p['shots'])}` failures."
            if best_nonbaseline_high_p is not None
            else ""
        )
    )
    per_round_caption = (
        "Figure 2. Strict full logical frame error rate per syndrome-extraction round versus physical location error "
        "rate for the same matched Gross `[[144,12,12]]` public `bravyi_depth7` split-sector DEM benchmark as Figure 1. "
        f"The decoding matrices are `D_X = {detectors_x} x {variables_x}` and `D_Z = {detectors_z} x {variables_z}`, and the per-round "
        f"quantity is obtained by exact inversion of the observed full-frame FER over `{noisy_rounds}` noisy syndrome-extraction rounds. "
        "The x-axis is physical location error rate `p` on a logarithmic scale; the y-axis is strict full logical FER per syndrome-extraction round on a logarithmic scale, "
        "with the zero-count floor derived by applying the same exact conversion to the usual frame-level `0.5/shots` floor. "
        f"The main quantitative takeaway is that `{best_per_round_row['decoder']}` has the lowest observed per-round FER, "
        f"reaching `{float(best_per_round_row['fer_per_round']):.6g}` over the full matched run."
    )
    runtime_caption = (
        "Figure 3. Mean per-frame decode time by decoder for the same matched Gross `[[144,12,12]]` split-sector "
        "benchmark as Figures 1 and 2, aggregated over the full matched p-grid and shot set. Each bar shows the arithmetic "
        "mean end-to-end time to decode both sectors of one sampled frame under the shared task definition and seed "
        "protocol; the y-axis is runtime in milliseconds on a logarithmic scale. Runtime is secondary to FER for "
        "cross-implementation ranking, but it remains useful for workstation planning. The main quantitative takeaway "
        f"is that the fastest decoder in this run is `{fastest_row['decoder']}` at "
        f"`{float(fastest_row['decode_ms_mean']):.1f}` ms/frame, while the slowest is "
        f"`{runtime_slowest_row['decoder']}` at `{float(runtime_slowest_row['decode_ms_mean']):.1f}` ms/frame."
    )

    exact_run_cmd = config.get("command") or (
        "tools/py scripts/bench_gross_matched.py "
        f"--p-values {','.join(str(value) for value in config['p_values'])} "
        f"--shots {int(config['shots_per_point'])} "
        f"--decoders {','.join(str(value) for value in config['decoders'])} "
        f"--seed {int(config['seed'])} "
        f"--baseline-max-iter {int(config['baseline_max_iter'])} "
        f"--baseline-osd-method {shlex.quote(str(config['baseline_osd_method']))} "
        f"--baseline-osd-order {int(config['baseline_osd_order'])} "
        f"--baseline-minsum-scale {float(config['baseline_minsum_scale']):.6g} "
        f"--bp-max-iter {int(config['bp_max_iter'])} "
        f"--minsum-max-iter {int(config['minsum_max_iter'])} "
        f"--minsum-scale {float(config['minsum_scale']):.6g} "
        f"--window-round-radius {int(config['window_round_radius'])} "
        f"--window-max-iter {int(config['window_max_iter'])} "
        f"--local-round-passes {int(config['local_round_passes'])} "
        f"--separator-window-rounds {separator_window_rounds} "
        f"--separator-overlap-rounds {separator_overlap_rounds} "
        f"--separator-topk {separator_topk} "
        f"--separator-max-branches {separator_max_branches} "
        f"--separator-max-window-expansions {separator_max_window_expansions} "
        f"--separator-mean-tail {separator_mean_tail} "
        f"--separator-reliable-shell-hops {separator_reliable_shell_hops} "
        f"--separator-reliable-topk {separator_reliable_topk} "
        f"--separator-reliable-abs-mean-threshold {separator_reliable_abs_mean_threshold:.6g} "
        f"--relay-enable {int(relay_enable)} "
        f"--relay-trigger-residual-stall-rounds {relay_trigger_residual_stall_rounds} "
        f"--relay-trigger-rebound {relay_trigger_rebound} "
        f"--relay-gamma-stable {relay_gamma_stable:.6g} "
        f"--relay-gamma-bulk {relay_gamma_bulk:.6g} "
        f"--relay-gamma-frontier {relay_gamma_frontier:.6g} "
        f"--relay-clip-B {relay_clip_B:.6g} "
        f"--relay-frontier-shell-radius {relay_frontier_shell_radius} "
        f"--relay-flip-mode {relay_flip_mode} "
        f"--relay-flip-kappa {relay_flip_kappa:.6g} "
        f"--relay-flip-eta {relay_flip_eta:.6g} "
        f"--relay-candidate-score {relay_candidate_score} "
        f"--relay-num-legs {relay_num_legs} "
        f"--relay-leg-iters {relay_leg_iters} "
        f"--triangle-factorization {triangle_factorization} "
        f"--layered-schedule {layered_schedule} "
        f"--gauge-descent {int(gauge_descent)} "
        f"--gauge-descent-max-iterations {gauge_descent_max_iterations} "
        f"--results-dir {shlex.quote(str(run_dir))}"
    )
    if isinstance(exact_run_cmd, str) and not str(exact_run_cmd).startswith("tools/py "):
        exact_run_cmd = f"tools/py {exact_run_cmd}"
    plot_cmd = f"tools/py scripts/plot_gross_matched.py --run-dir {shlex.quote(str(run_dir))}"

    lines = [
        "# Gross [[144,12,12]] matched benchmark",
        "",
        "## Smoke vs matched benchmark",
        "",
        "- Earlier smoke results were useful for wiring and sanity checks only; the matched rows below are the comparable benchmark results.",
        "- This report is the first matched benchmark because every compared decoder below uses the same `bravyi_depth7` split-sector task family, p-grid, shot counts, frame-seed protocol, and `12 noisy + 1 perfect` round structure.",
        f"- {baseline_note}",
        "",
        "## Exact commands",
        "",
        "```bash",
        f"cd {REPO_ROOT}",
        f"MPLCONFIGDIR=/tmp/mplcache_grosscode {exact_run_cmd}",
        f"MPLCONFIGDIR=/tmp/mplcache_grosscode {plot_cmd}",
        "```",
        "",
        "## Exact config used",
        "",
        f"- backend: `{config['backend']}`",
        f"- p grid: `{', '.join(str(value) for value in config['p_values'])}`",
        f"- shots per point: `{int(config['shots_per_point'])}`",
        f"- base seed: `{int(config['seed'])}`",
        f"- decoders: `{', '.join(config['decoders'])}`",
        f"- detector-side matrices: `D_X = {detectors_x} x {variables_x}`, `D_Z = {detectors_z} x {variables_z}`",
        f"- syndrome-extraction rounds: `{noisy_rounds}` noisy + `{perfect_rounds}` perfect",
        f"- baseline max_iter / osd_method / osd_order / ms_scale: `{int(config['baseline_max_iter'])}` / `{config['baseline_osd_method']}` / `{int(config['baseline_osd_order'])}` / `{float(config['baseline_minsum_scale']):.3f}`",
        f"- bp max_iter: `{int(config['bp_max_iter'])}`",
        f"- min-sum max_iter / scale: `{int(config['minsum_max_iter'])}` / `{float(config['minsum_scale']):.3f}`",
        f"- window round radius / window max_iter / local_round passes: `{int(config['window_round_radius'])}` / `{int(config['window_max_iter'])}` / `{int(config['local_round_passes'])}`",
        f"- separator wavefront window / overlap / topk / branches / expansions: `{separator_window_rounds}` / `{separator_overlap_rounds}` / `{separator_topk}` / `{separator_max_branches}` / `{separator_max_window_expansions}`",
        f"- separator mean tail / reliable shell hops / reliable topk / reliable abs(mean LLR) threshold: `{separator_mean_tail}` / `{separator_reliable_shell_hops}` / `{separator_reliable_topk}` / `{separator_reliable_abs_mean_threshold:.3f}`",
        f"- relay enable / stall rounds / rebound trigger: `{int(relay_enable)}` / `{relay_trigger_residual_stall_rounds}` / `{relay_trigger_rebound}`",
        f"- relay gammas stable / bulk / frontier: `{relay_gamma_stable:.3f}` / `{relay_gamma_bulk:.3f}` / `{relay_gamma_frontier:.3f}`",
        f"- relay clip_B / shell radius / flip mode / candidate score / legs / leg iters: `{relay_clip_B:.3f}` / `{relay_frontier_shell_radius}` / `{relay_flip_mode}` / `{relay_candidate_score}` / `{relay_num_legs}` / `{relay_leg_iters}`",
        f"- relay flip kappa / eta: `{relay_flip_kappa:.3f}` / `{relay_flip_eta:.3f}`",
        f"- structure-aware flags: triangle factorization=`{triangle_factorization}`, layered schedule=`{layered_schedule}`, gauge descent=`{int(gauge_descent)}` (max iterations `{gauge_descent_max_iterations}`)",
        f"- results root: `{run_dir}`",
        "",
        "## Results table",
        "",
        _format_markdown_table(
            summary_by_point.sort_values(["p", "decoder"]).reset_index(drop=True),
            [
                "decoder",
                "p",
                "shots",
                "fail_total",
                "logical_fail",
                "syndrome_fail",
                "exception_fail",
                "fer",
                "fer_per_round",
                "decode_ms_mean",
                "work_ms_iters_mean",
                "decode_ms_p95",
                "converged_rate",
            ],
            {
                "p": ".4f",
                "fer": ".4f",
                "fer_per_round": ".5f",
                "decode_ms_mean": ".2f",
                "work_ms_iters_mean": ".2f",
                "decode_ms_p95": ".2f",
                "converged_rate": ".3f",
                "triangle_reduced_count_x_mean": ".2f",
                "triangle_reduced_count_z_mean": ".2f",
                "gauge_descent_moves_x_mean": ".2f",
                "gauge_descent_moves_z_mean": ".2f",
            },
        ).rstrip(),
        "",
        "## Structure Instrumentation",
        "",
        _format_markdown_table(
            summary_by_point.sort_values(["p", "decoder"]).reset_index(drop=True),
            [
                "decoder",
                "p",
                "triangle_catalog_size_x",
                "triangle_catalog_size_z",
                "triangle_selected_count_x",
                "triangle_selected_count_z",
                "triangle_reduced_count_x_mean",
                "triangle_reduced_count_z_mean",
                "triangle_residual_count_x_mean",
                "triangle_residual_count_z_mean",
                "gauge_descent_moves_x_mean",
                "gauge_descent_moves_z_mean",
                "schedule_direction_x",
                "schedule_direction_z",
            ],
            {
                "p": ".4f",
                "triangle_reduced_count_x_mean": ".2f",
                "triangle_reduced_count_z_mean": ".2f",
                "triangle_residual_count_x_mean": ".2f",
                "triangle_residual_count_z_mean": ".2f",
                "gauge_descent_moves_x_mean": ".2f",
                "gauge_descent_moves_z_mean": ".2f",
            },
        ).rstrip(),
        "",
        "## FER vs p",
        "",
        f"![Matched FER vs p]({plot_paths['fer_vs_p']})",
        "",
        fer_caption,
        "",
        _format_markdown_table(
            summary_by_point.sort_values(["p", "decoder"]).reset_index(drop=True),
            ["decoder", "p", "shots", "fail_total", "logical_fail", "syndrome_fail", "exception_fail", "fer", "fer_per_round"],
            {"p": ".4f", "fer": ".4f", "fer_per_round": ".5f"},
        ).rstrip(),
        "",
        "## FER per syndrome-extraction round vs p",
        "",
        f"![Matched FER per round vs p]({plot_paths['fer_per_round_vs_p']})",
        "",
        per_round_caption,
        "",
        _format_markdown_table(
            summary_by_point.sort_values(["p", "decoder"]).reset_index(drop=True),
            ["decoder", "p", "shots", "noisy_rounds", "fail_total", "fer", "fer_per_round"],
            {"p": ".4f", "fer": ".4f", "fer_per_round": ".5f"},
        ).rstrip(),
        "",
        "## Runtime vs decoder",
        "",
        f"![Matched runtime vs decoder]({plot_paths['runtime_vs_decoder']})",
        "",
        runtime_caption,
        "",
        _format_markdown_table(
            summary_overall.sort_values(["decode_ms_mean", "decoder"]).reset_index(drop=True),
            ["decoder", "shots", "fail_total", "fer", "fer_per_round", "decode_ms_mean", "decode_ms_p95", "work_ms_iters_mean", "work_ms_iters_p95"],
            {"fer": ".4f", "fer_per_round": ".5f", "decode_ms_mean": ".2f", "decode_ms_p95": ".2f", "work_ms_iters_mean": ".2f", "work_ms_iters_p95": ".2f"},
        ).rstrip(),
        "",
        "## Decoder failures and instability",
        "",
    ]

    for _, row in summary_by_point.sort_values(["p", "decoder"]).iterrows():
        notes: list[str] = []
        if int(row["syndrome_fail"]) > 0:
            notes.append(f"syndrome_fail={int(row['syndrome_fail'])}")
        if int(row["logical_fail"]) > 0:
            notes.append(f"logical_fail={int(row['logical_fail'])}")
        if int(row["exception_fail"]) > 0:
            notes.append(f"exception_fail={int(row['exception_fail'])}")
        if float(row["converged_rate"]) < 1.0:
            notes.append(f"converged_rate={float(row['converged_rate']):.3f}")
        if not notes:
            notes.append("no instability observed in this small matched run")
        lines.append(f"- `{row['decoder']}` at `p={float(row['p']):.4f}`: " + ", ".join(notes) + ".")

    lines.extend(
        [
            "",
            "## Comparability statement",
            "",
            "- Comparable within this table: all rows share the same split-sector task definition, priors, seeds, and shot allocation.",
            "- Not comparable to the separate public Bravyi wrapper yet: its exported task family is still a documented matrix mismatch, so it is excluded from winner claims here.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
