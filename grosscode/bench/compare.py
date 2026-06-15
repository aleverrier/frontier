from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from grosscode.decoders.api import DecoderWindow, SplitSectorPriors, SplitSectorSyndrome, SplitXZDecoder
from grosscode.dem.builder import SplitSectorProblem, build_split_sector_problem
from grosscode.utils.gf2 import csr_matvec_mod2
from grosscode.utils.paths import DEFAULT_RESULTS_ROOT, ensure_mplconfigdir


def _parse_decoders(raw: str) -> list[str]:
    decoders = [token.strip() for token in str(raw).split(",") if token.strip()]
    if not decoders:
        raise ValueError("expected at least one decoder")
    return decoders


def _sample_faults(priors: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(priors.size) < priors).astype(np.uint8)


def _frame_fer_to_per_round_exact(frame_fer: float, noisy_rounds: int) -> float:
    rounds = max(1, int(noisy_rounds))
    clipped = float(max(0.0, min(0.5, frame_fer)))
    return float((1.0 - math.pow(1.0 - 2.0 * clipped, 1.0 / float(rounds))) * 0.5)


def _status_from_result(
    *,
    syndrome_x: np.ndarray,
    syndrome_z: np.ndarray,
    truth_obs_x: np.ndarray,
    truth_obs_z: np.ndarray,
    result,
    problem: SplitSectorProblem,
) -> str:
    pred_syndrome_x = csr_matvec_mod2(problem.D_X, result.correction_X)
    pred_syndrome_z = csr_matvec_mod2(problem.D_Z, result.correction_Z)
    if not np.array_equal(pred_syndrome_x, syndrome_x) or not np.array_equal(pred_syndrome_z, syndrome_z):
        return "syndrome_fail"
    if not np.array_equal(result.logical_frame_action_X, truth_obs_x) or not np.array_equal(
        result.logical_frame_action_Z, truth_obs_z
    ):
        return "logical_fail"
    return "success"


def _run_one_shot(
    shot_index: int,
    *,
    problem: SplitSectorProblem,
    decoders: list[str],
    decoder_window: DecoderWindow,
    bp_max_iter: int,
    minsum_max_iter: int,
    minsum_scale: float,
    seed: int,
    scms: bool,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(int(seed) + int(shot_index))
    faults_x = _sample_faults(problem.priors_X, rng)
    faults_z = _sample_faults(problem.priors_Z, rng)
    syndrome_x = csr_matvec_mod2(problem.D_X, faults_x)
    syndrome_z = csr_matvec_mod2(problem.D_Z, faults_z)
    truth_obs_x = csr_matvec_mod2(problem.O_X, faults_x)
    truth_obs_z = csr_matvec_mod2(problem.O_Z, faults_z)
    priors = SplitSectorPriors(X=problem.priors_X, Z=problem.priors_Z)
    syndrome = SplitSectorSyndrome(X=syndrome_x, Z=syndrome_z)

    rows: list[dict[str, object]] = []
    for decoder_name in decoders:
        decoder_scms = bool(decoder_name == "self_corrected_minsum" or (scms and decoder_name != "bp"))
        runner = SplitXZDecoder(
            problem,
            bp_max_iter=int(bp_max_iter),
            minsum_max_iter=int(minsum_max_iter),
            minsum_scale=float(minsum_scale),
            seed=int(seed) + int(shot_index),
            scms=decoder_scms,
        )
        started = time.perf_counter()
        try:
            if decoder_name == "local_round":
                result = runner.decode_split_xz(syndrome, priors, window=decoder_window, decoder="local_round")
            elif decoder_name == "minsum":
                result = runner.decode_split_xz(syndrome, priors, window=decoder_window, decoder="minsum")
            elif decoder_name == "self_corrected_minsum":
                result = runner.decode_split_xz(syndrome, priors, window=decoder_window, decoder="self_corrected_minsum")
            elif decoder_name == "separator_wavefront":
                result = runner.decode_split_xz(syndrome, priors, window=decoder_window, decoder="separator_wavefront")
            elif decoder_name == "relay_minsum":
                result = runner.decode_split_xz(syndrome, priors, window=decoder_window, decoder="relay_minsum")
            else:
                result = runner.decode_split_xz(syndrome, priors, window=None, decoder="bp")
            status = _status_from_result(
                syndrome_x=syndrome_x,
                syndrome_z=syndrome_z,
                truth_obs_x=truth_obs_x,
                truth_obs_z=truth_obs_z,
                result=result,
                problem=problem,
            )
            elapsed_ms = 1000.0 * (time.perf_counter() - started)
            diag_x = dict(result.diagnostics.get("sector_X", {}))
            diag_z = dict(result.diagnostics.get("sector_Z", {}))
            rows.append(
                {
                    "shot": int(shot_index),
                    "decoder": decoder_name,
                    "status": status,
                    "decode_ms": elapsed_ms,
                    "scms": int(decoder_scms),
                    "work_ms_iters": int(diag_x.get("iterations", 0)) + int(diag_z.get("iterations", 0)),
                    "erased_edge_total": int(diag_x.get("erased_edge_total", 0)) + int(diag_z.get("erased_edge_total", 0)),
                    "syndrome_weight_x": int(syndrome_x.sum()),
                    "syndrome_weight_z": int(syndrome_z.sum()),
                    "logical_weight_x": int(truth_obs_x.sum()),
                    "logical_weight_z": int(truth_obs_z.sum()),
                }
            )
        except Exception as exc:  # pragma: no cover - exercised through benchmark smoke validation.
            rows.append(
                {
                    "shot": int(shot_index),
                    "decoder": decoder_name,
                    "status": "exception_fail",
                    "decode_ms": 1000.0 * (time.perf_counter() - started),
                    "scms": int(decoder_scms),
                    "work_ms_iters": 0,
                    "erased_edge_total": 0,
                    "error": str(exc),
                }
            )
    return rows


def _aggregate_rows(rows: list[dict[str, object]], shots: int, *, noisy_rounds: int) -> list[dict[str, object]]:
    by_decoder: dict[str, dict[str, object]] = {}
    for row in rows:
        decoder = str(row["decoder"])
        bucket = by_decoder.setdefault(
            decoder,
            {
                "decoder": decoder,
                "shots": int(shots),
                "success": 0,
                "logical_fail": 0,
                "syndrome_fail": 0,
                "exception_fail": 0,
                "decode_ms_total": 0.0,
                "work_ms_iters_total": 0.0,
                "erased_edge_total": 0.0,
                "scms": int(row.get("scms", 0)),
            },
        )
        status = str(row["status"])
        bucket[status] = int(bucket.get(status, 0)) + 1
        bucket["decode_ms_total"] = float(bucket["decode_ms_total"]) + float(row["decode_ms"])
        bucket["work_ms_iters_total"] = float(bucket["work_ms_iters_total"]) + float(row.get("work_ms_iters", 0.0))
        bucket["erased_edge_total"] = float(bucket["erased_edge_total"]) + float(row.get("erased_edge_total", 0.0))
    summary: list[dict[str, object]] = []
    for decoder in sorted(by_decoder):
        bucket = by_decoder[decoder]
        fail_total = int(bucket["logical_fail"]) + int(bucket["syndrome_fail"]) + int(bucket["exception_fail"])
        shots_float = float(max(1, int(bucket["shots"])))
        fer = fail_total / shots_float
        summary.append(
            {
                "decoder": decoder,
                "scms": int(bucket["scms"]),
                "shots": int(bucket["shots"]),
                "success": int(bucket["success"]),
                "logical_fail": int(bucket["logical_fail"]),
                "syndrome_fail": int(bucket["syndrome_fail"]),
                "exception_fail": int(bucket["exception_fail"]),
                "fail_total": fail_total,
                "fer": float(fer),
                "noisy_rounds": int(noisy_rounds),
                "fer_per_round": _frame_fer_to_per_round_exact(float(fer), int(noisy_rounds)),
                "decode_ms_mean": float(bucket["decode_ms_total"]) / shots_float,
                "work_ms_iters_mean": float(bucket["work_ms_iters_total"]) / shots_float,
                "erased_edge_total_mean": float(bucket["erased_edge_total"]) / shots_float,
            }
        )
    return summary


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_plot(
    path: Path,
    summary_rows: list[dict[str, object]],
    *,
    shots: int,
    error_rate: float,
    noisy_rounds: int,
    value_key: str,
    ylabel: str,
    title_suffix: str,
) -> None:
    ensure_mplconfigdir()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["decoder"]) for row in summary_rows]
    frame_floor = 0.5 / max(1, int(shots))
    if str(value_key) == "fer_per_round":
        floor = _frame_fer_to_per_round_exact(float(frame_floor), int(noisy_rounds))
    else:
        floor = float(frame_floor)
    values = [max(float(row[str(value_key)]), float(floor)) for row in summary_rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    palette = ["#205493", "#4aa564", "#d98324", "#7b6aa8", "#b84a62"]
    ax.bar(labels, values, color=[palette[index % len(palette)] for index in range(len(labels))])
    ax.set_yscale("log")
    ax.set_ylabel(str(ylabel))
    ax.set_xlabel("Decoder")
    ax.set_title(f"Gross split-sector smoke benchmark{title_suffix}")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_report(
    path: Path,
    *,
    figure_path: Path,
    figure_per_round_path: Path,
    summary_rows: list[dict[str, object]],
    args: argparse.Namespace,
    noisy_rounds: int,
    detectors_x: int,
    variables_x: int,
    detectors_z: int,
    variables_z: int,
) -> None:
    best = min(summary_rows, key=lambda row: (float(row["fer"]), float(row["decode_ms_mean"])))
    best_per_round = min(summary_rows, key=lambda row: (float(row["fer_per_round"]), float(row["decode_ms_mean"])))
    caption = (
        "Figure 1. Full logical FER = fail_total / shots for the split-sector gross [[144,12,12]] circuit-level "
        f"smoke benchmark on the public {args.backend} backend at physical location rate p={float(args.error_rate):.6g}. "
        f"The task decodes both memory_X and memory_Z sectors built from the public {int(noisy_rounds)} noisy rounds + 1 perfect round "
        f"split-sector DEM circuits with detector-side matrices `D_X = {int(detectors_x)} x {int(variables_x)}` and "
        f"`D_Z = {int(detectors_z)} x {int(variables_z)}`, using shared per-column DEM priors and deterministic seeds. "
        "The x-axis lists decoder families "
        "(full-block BP, windowed/factor-aware min-sum, and generalized local-round-factor), while the y-axis is full "
        f"logical FER on a logarithmic scale with an epsilon floor of 0.5/{int(args.shots)} for zero-count bars. "
        f"Main quantitative takeaway: the lowest observed smoke FER is {float(best['fer']):.3e} for {best['decoder']}, "
        f"with mean decode time {float(best['decode_ms_mean']):.3f} ms over {int(args.shots)} deterministic Monte Carlo shots."
    )
    per_round_caption = (
        "Figure 2. Exact strict logical FER per syndrome-extraction round for the same Gross split-sector DEM smoke benchmark "
        f"as Figure 1. The detector-side matrices remain `D_X = {int(detectors_x)} x {int(variables_x)}` and "
        f"`D_Z = {int(detectors_z)} x {int(variables_z)}`, and the per-round quantity is obtained by exact inversion of the "
        f"observed full-frame FER over `{int(noisy_rounds)}` noisy rounds. The x-axis lists decoder families; the y-axis is "
        "strict logical FER per syndrome-extraction round on a logarithmic scale, using the exact conversion of the "
        f"usual `0.5/{int(args.shots)}` frame-level floor. Main quantitative takeaway: the lowest observed per-round FER is "
        f"{float(best_per_round['fer_per_round']):.3e} for {best_per_round['decoder']}."
    )
    lines = [
        "# Grosscode Smoke Benchmark",
        "",
        f"- Backend: `{args.backend}`",
        f"- Error rate: `{float(args.error_rate):.6g}`",
        f"- Shots: `{int(args.shots)}`",
        f"- Syndrome-extraction rounds: `{int(noisy_rounds)}`",
        f"- Detector-side matrices: `D_X = {int(detectors_x)} x {int(variables_x)}`, `D_Z = {int(detectors_z)} x {int(variables_z)}`",
        f"- Decoders: `{args.decoders}`",
        f"- CPU workers: `{int(args.cpus)}`",
        "",
        f"![Grosscode smoke FER plot]({figure_path})",
        "",
        caption,
        "",
        f"![Grosscode smoke FER-per-round plot]({figure_per_round_path})",
        "",
        per_round_caption,
        "",
        "Companion table: `summary.csv` in the same directory, including `fer` and `fer_per_round`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(args: argparse.Namespace) -> Path:
    try:
        from joblib import Parallel, delayed
    except ModuleNotFoundError:  # pragma: no cover - depends on local Python environment.
        Parallel = None  # type: ignore[assignment]
        delayed = None  # type: ignore[assignment]

    problem = build_split_sector_problem(backend=str(args.backend), error_rate=float(args.error_rate))
    noisy_rounds = int(problem.metadata_X.noisy_rounds)
    decoders = _parse_decoders(args.decoders)
    window = DecoderWindow(
        round_radius=int(args.window_round_radius),
        max_passes=int(args.local_round_passes),
        max_iter=int(args.window_max_iter),
        separator_window_rounds=int(args.separator_window_rounds),
        separator_overlap_rounds=int(args.separator_overlap_rounds),
        separator_topk=int(args.separator_topk),
        separator_max_branches=int(args.separator_max_branches),
        separator_max_window_expansions=int(args.separator_max_window_expansions),
        separator_mean_tail=int(args.separator_mean_tail),
        separator_reliable_shell_hops=int(args.separator_reliable_shell_hops),
        separator_reliable_topk=int(args.separator_reliable_topk),
        separator_reliable_abs_mean_threshold=float(args.separator_reliable_abs_mean_threshold),
        relay_enable=bool(args.relay_enable),
        relay_trigger_residual_stall_rounds=int(args.relay_trigger_residual_stall_rounds),
        relay_trigger_rebound=int(args.relay_trigger_rebound),
        relay_gamma_stable=float(args.relay_gamma_stable),
        relay_gamma_bulk=float(args.relay_gamma_bulk),
        relay_gamma_frontier=float(args.relay_gamma_frontier),
        relay_clip_B=float(args.relay_clip_B),
        relay_frontier_shell_radius=int(args.relay_frontier_shell_radius),
        relay_flip_mode=str(args.relay_flip_mode),
        relay_flip_kappa=float(args.relay_flip_kappa),
        relay_flip_eta=float(args.relay_flip_eta),
        relay_candidate_score=str(args.relay_candidate_score),
        relay_num_legs=int(args.relay_num_legs),
        relay_leg_iters=int(args.relay_leg_iters),
    )
    out_dir = Path(args.output_dir) if args.output_dir is not None else DEFAULT_RESULTS_ROOT / (
        f"{time.strftime('%Y%m%d_%H%M%S')}_grosscode_compare"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "task_definition.json").write_text(
        json.dumps(
            {
                "backend": str(args.backend),
                "noisy_rounds": int(noisy_rounds),
                "perfect_rounds": int(problem.metadata_X.total_rounds - problem.metadata_X.noisy_rounds),
                "detectors_x": int(problem.D_X.shape[0]),
                "variables_x": int(problem.D_X.shape[1]),
                "detectors_z": int(problem.D_Z.shape[0]),
                "variables_z": int(problem.D_Z.shape[1]),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    batch_size = max(1, int(args.progress_every))
    for batch_start in range(0, int(args.shots), batch_size):
        batch_stop = min(int(args.shots), batch_start + batch_size)
        if Parallel is None or int(args.cpus) == 1:
            batch_rows = [
                _run_one_shot(
                    shot_index,
                    problem=problem,
                    decoders=decoders,
                    decoder_window=window,
                    bp_max_iter=int(args.bp_max_iter),
                    minsum_max_iter=int(args.minsum_max_iter),
                    minsum_scale=float(args.minsum_scale),
                    seed=int(args.seed),
                    scms=bool(args.scms),
                )
                for shot_index in range(batch_start, batch_stop)
            ]
        else:
            batch_rows = Parallel(n_jobs=int(args.cpus))(
                delayed(_run_one_shot)(
                    shot_index,
                    problem=problem,
                    decoders=decoders,
                    decoder_window=window,
                    bp_max_iter=int(args.bp_max_iter),
                    minsum_max_iter=int(args.minsum_max_iter),
                    minsum_scale=float(args.minsum_scale),
                    seed=int(args.seed),
                    scms=bool(args.scms),
                )
                for shot_index in range(batch_start, batch_stop)
            )
        for shard in batch_rows:
            rows.extend(shard)
        elapsed = time.perf_counter() - started
        rate = float(batch_stop) / max(elapsed, 1e-9)
        remaining = int(args.shots) - batch_stop
        eta = remaining / max(rate, 1e-9)
        print(
            f"[progress] shots={batch_stop}/{int(args.shots)} elapsed={elapsed:.2f}s rate={rate:.2f} shots/s eta={eta:.2f}s",
            flush=True,
        )

    summary_rows = _aggregate_rows(rows, int(args.shots), noisy_rounds=int(noisy_rounds))
    _write_csv(out_dir / "per_shot.csv", rows)
    _write_csv(out_dir / "summary.csv", summary_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "backend": str(args.backend),
                "error_rate": float(args.error_rate),
                "shots": int(args.shots),
                "decoders": decoders,
                "scms": bool(args.scms),
                "window_round_radius": int(args.window_round_radius),
                "local_round_passes": int(args.local_round_passes),
                "separator_window_rounds": int(args.separator_window_rounds),
                "separator_overlap_rounds": int(args.separator_overlap_rounds),
                "separator_topk": int(args.separator_topk),
                "separator_max_branches": int(args.separator_max_branches),
                "separator_max_window_expansions": int(args.separator_max_window_expansions),
                "separator_mean_tail": int(args.separator_mean_tail),
                "separator_reliable_shell_hops": int(args.separator_reliable_shell_hops),
                "separator_reliable_topk": int(args.separator_reliable_topk),
                "separator_reliable_abs_mean_threshold": float(args.separator_reliable_abs_mean_threshold),
                "relay_enable": bool(args.relay_enable),
                "relay_trigger_residual_stall_rounds": int(args.relay_trigger_residual_stall_rounds),
                "relay_trigger_rebound": int(args.relay_trigger_rebound),
                "relay_gamma_stable": float(args.relay_gamma_stable),
                "relay_gamma_bulk": float(args.relay_gamma_bulk),
                "relay_gamma_frontier": float(args.relay_gamma_frontier),
                "relay_clip_B": float(args.relay_clip_B),
                "relay_frontier_shell_radius": int(args.relay_frontier_shell_radius),
                "relay_flip_mode": str(args.relay_flip_mode),
                "relay_flip_kappa": float(args.relay_flip_kappa),
                "relay_flip_eta": float(args.relay_flip_eta),
                "relay_candidate_score": str(args.relay_candidate_score),
                "relay_num_legs": int(args.relay_num_legs),
                "relay_leg_iters": int(args.relay_leg_iters),
                "rows": summary_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if not args.no_plot:
        figure_path = out_dir / "fig_fail_rate_by_decoder_logy.png"
        figure_per_round_path = out_dir / "fig_fail_rate_per_round_by_decoder_logy.png"
        _make_plot(
            figure_path,
            summary_rows,
            shots=int(args.shots),
            error_rate=float(args.error_rate),
            noisy_rounds=int(noisy_rounds),
            value_key="fer",
            ylabel="Full logical FER",
            title_suffix="",
        )
        _make_plot(
            figure_per_round_path,
            summary_rows,
            shots=int(args.shots),
            error_rate=float(args.error_rate),
            noisy_rounds=int(noisy_rounds),
            value_key="fer_per_round",
            ylabel="Full logical FER per syndrome-extraction round",
            title_suffix=" (per round)",
        )
        _write_report(
            out_dir / "report.md",
            figure_path=figure_path,
            figure_per_round_path=figure_per_round_path,
            summary_rows=summary_rows,
            args=args,
            noisy_rounds=int(noisy_rounds),
            detectors_x=int(problem.D_X.shape[0]),
            variables_x=int(problem.D_X.shape[1]),
            detectors_z=int(problem.D_Z.shape[0]),
            variables_z=int(problem.D_Z.shape[1]),
        )
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark split-sector gross decoders on the same public circuit-level DEM tasks. "
            "The harness samples local fault variables directly from the split DEM priors with deterministic seeds."
        )
    )
    parser.add_argument("--backend", type=str, default="bravyi_depth7")
    parser.add_argument("--error-rate", type=float, default=0.004)
    parser.add_argument("--shots", type=int, default=8)
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--decoders", type=str, default="bp,minsum,local_round")
    parser.add_argument("--bp-max-iter", type=int, default=25)
    parser.add_argument("--minsum-max-iter", type=int, default=40)
    parser.add_argument("--minsum-scale", type=float, default=0.75)
    parser.add_argument("--scms", action="store_true", help="Enable Savin-style self-corrected min-sum for min-sum-based decoders in this run.")
    parser.add_argument("--window-round-radius", type=int, default=1)
    parser.add_argument("--window-max-iter", type=int, default=40)
    parser.add_argument("--local-round-passes", type=int, default=3)
    parser.add_argument("--separator-window-rounds", type=int, default=3)
    parser.add_argument("--separator-overlap-rounds", type=int, default=1)
    parser.add_argument("--separator-topk", type=int, default=2)
    parser.add_argument("--separator-max-branches", type=int, default=4)
    parser.add_argument("--separator-max-window-expansions", type=int, default=2)
    parser.add_argument("--separator-mean-tail", type=int, default=8)
    parser.add_argument("--separator-reliable-shell-hops", type=int, default=1)
    parser.add_argument("--separator-reliable-topk", type=int, default=6)
    parser.add_argument("--separator-reliable-abs-mean-threshold", type=float, default=6.0)
    parser.add_argument("--relay-enable", type=int, default=0)
    parser.add_argument("--relay-trigger-residual-stall-rounds", type=int, default=4)
    parser.add_argument("--relay-trigger-rebound", type=int, default=1)
    parser.add_argument("--relay-gamma-stable", type=float, default=0.15)
    parser.add_argument("--relay-gamma-bulk", type=float, default=0.0)
    parser.add_argument("--relay-gamma-frontier", type=float, default=-0.15)
    parser.add_argument("--relay-clip-B", type=float, default=12.0)
    parser.add_argument("--relay-frontier-shell-radius", type=int, default=1)
    parser.add_argument("--relay-flip-mode", type=str, choices=("erase", "blend", "damp"), default="erase")
    parser.add_argument("--relay-flip-kappa", type=float, default=0.25)
    parser.add_argument("--relay-flip-eta", type=float, default=0.5)
    parser.add_argument(
        "--relay-candidate-score",
        type=str,
        choices=("final_absllr", "mean_absllr", "temporal_instability", "instability_plus_residual"),
        default="instability_plus_residual",
    )
    parser.add_argument("--relay-num-legs", type=int, default=1)
    parser.add_argument("--relay-leg-iters", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.shots) < 1:
        raise ValueError("--shots must be >= 1")
    if int(args.cpus) < 1:
        raise ValueError("--cpus must be >= 1")
    if int(args.window_round_radius) < 0:
        raise ValueError("--window-round-radius must be >= 0")
    if int(args.window_max_iter) < 1:
        raise ValueError("--window-max-iter must be >= 1")
    if int(args.local_round_passes) < 1:
        raise ValueError("--local-round-passes must be >= 1")
    if int(args.progress_every) < 1:
        raise ValueError("--progress-every must be >= 1")
    out_dir = run_benchmark(args)
    print(f"[done] wrote benchmark artifacts to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
