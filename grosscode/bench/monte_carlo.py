from __future__ import annotations

import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache_betterbeam")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from grosscode.decoders.full_block import FullBlockMinSumDecoder, ImportedBpOsdDecoder
from grosscode.decoders.local_round.topk import TopKLocalRoundFactor
from grosscode.decoders.local_round.windowed_local_round import WindowedLocalRoundDecoder
from grosscode.model import DEFAULT_HX, DEFAULT_HZ, GrossSideModel, clip_prob


DEFAULT_RESULTS_BASE = Path("/Users/anthony/research/better-beam/results")
DEFAULT_DECODERS = ("baseline_bposd", "full_block_minsum", "windowed_minsum", "local_round_topk")


def format_seconds(seconds: float) -> str:
    value = max(0.0, float(seconds))
    if value < 60.0:
        return f"{value:.1f}s"
    if value < 3600.0:
        return f"{value / 60.0:.1f}m"
    return f"{value / 3600.0:.2f}h"


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=np.float64), float(q)))


def _default_chunk_shots(shots_per_point: int, nproc: int) -> int:
    if int(shots_per_point) <= 0:
        raise ValueError("shots_per_point must be positive")
    return max(1, min(int(shots_per_point), int(math.ceil(shots_per_point / max(1, nproc * 4)))))


@dataclass
class BenchmarkConfig:
    p_values: Sequence[float]
    rounds_list: Sequence[int]
    shots_per_point: int
    sides: Sequence[str] = ("x", "z")
    decoders: Sequence[str] = DEFAULT_DECODERS
    nproc: str | int = "auto"
    chunk_shots: int = 0
    base_seed: int = 12345
    p_meas_scale: float = 1.0
    hx_path: Path = DEFAULT_HX
    hz_path: Path = DEFAULT_HZ
    results_dir: Path | None = None
    baseline_max_iter: int = 30
    baseline_osd_order: int = 0
    full_block_max_iter: int = 30
    full_block_alpha: float = 0.75
    factor_max_iter: int = 20
    factor_alpha: float = 0.75
    window_size: int = 3
    sweeps: int = 3
    belief_damping: float = 0.5
    topk_interface_bits: int = 4
    progress_every_tasks: int = 1

    def resolved_nproc(self) -> int:
        if str(self.nproc) == "auto":
            return max(1, int(os.cpu_count() or 1))
        return max(1, int(self.nproc))

    def to_json_ready(self) -> Dict[str, object]:
        out = asdict(self)
        out["hx_path"] = str(self.hx_path)
        out["hz_path"] = str(self.hz_path)
        out["results_dir"] = str(self.results_dir) if self.results_dir is not None else None
        out["p_values"] = [float(x) for x in self.p_values]
        out["rounds_list"] = [int(x) for x in self.rounds_list]
        out["sides"] = [str(x) for x in self.sides]
        out["decoders"] = [str(x) for x in self.decoders]
        out["resolved_nproc"] = int(self.resolved_nproc())
        return out


def _init_aggregate_row(*, decoder: str, side: str, p: float, rounds: int) -> Dict[str, object]:
    return {
        "decoder": str(decoder),
        "side": str(side),
        "p": float(p),
        "rounds": int(rounds),
        "shots": 0,
        "fail_total": 0,
        "logical_fail": 0,
        "syndrome_fail": 0,
        "exception_fail": 0,
        "runtime_ms_samples": [],
        "work_iterations_samples": [],
        "edge_updates_samples": [],
    }


def _build_decoder_suite(model: GrossSideModel, cfg: Mapping[str, object]) -> Dict[str, object]:
    suite: Dict[str, object] = {}
    decoders = [str(x) for x in cfg["decoders"]]
    if "baseline_bposd" in decoders:
        suite["baseline_bposd"] = ImportedBpOsdDecoder(
            model=model,
            max_iter=int(cfg["baseline_max_iter"]),
            alpha=float(cfg["full_block_alpha"]),
            osd_order=int(cfg["baseline_osd_order"]),
            label="baseline_bposd",
        )
    if "full_block_minsum" in decoders:
        suite["full_block_minsum"] = FullBlockMinSumDecoder(
            model=model,
            max_iter=int(cfg["full_block_max_iter"]),
            alpha=float(cfg["full_block_alpha"]),
            label="full_block_minsum",
        )
    if "windowed_minsum" in decoders:
        factor = TopKLocalRoundFactor(
            model=model,
            max_iter=int(cfg["factor_max_iter"]),
            alpha=float(cfg["factor_alpha"]),
            topk_interface_bits=0,
        )
        suite["windowed_minsum"] = WindowedLocalRoundDecoder(
            model=model,
            factor=factor,
            window_size=int(cfg["window_size"]),
            sweeps=int(cfg["sweeps"]),
            belief_damping=float(cfg["belief_damping"]),
            label="windowed_minsum",
        )
    if "local_round_topk" in decoders:
        factor = TopKLocalRoundFactor(
            model=model,
            max_iter=int(cfg["factor_max_iter"]),
            alpha=float(cfg["factor_alpha"]),
            topk_interface_bits=int(cfg["topk_interface_bits"]),
        )
        suite["local_round_topk"] = WindowedLocalRoundDecoder(
            model=model,
            factor=factor,
            window_size=int(cfg["window_size"]),
            sweeps=int(cfg["sweeps"]),
            belief_damping=float(cfg["belief_damping"]),
            label="local_round_topk",
        )
    missing = [name for name in decoders if name not in suite]
    if missing:
        raise ValueError(f"unsupported decoders requested: {missing}")
    return suite


def _run_task(task: Mapping[str, object]) -> List[Dict[str, object]]:
    model = GrossSideModel.load(
        side=str(task["side"]),
        hx_path=Path(str(task["hx_path"])),
        hz_path=Path(str(task["hz_path"])),
    )
    suite = _build_decoder_suite(model, task)
    p = float(task["p"])
    rounds = int(task["rounds"])
    p_meas = float(task["p_meas"])
    shots = int(task["shots"])
    seed_base = int(task["seed"])
    aggregates = {
        name: _init_aggregate_row(decoder=name, side=str(task["side"]), p=p, rounds=rounds)
        for name in suite.keys()
    }
    for shot_idx in range(shots):
        shot = model.sample_shot(
            p=p,
            p_meas=p_meas,
            rounds=rounds,
            seed=seed_base + int(shot_idx),
        )
        for name, decoder in suite.items():
            result = decoder.decode(shot)
            agg = aggregates[str(name)]
            agg["shots"] = int(agg["shots"]) + 1
            if str(result.status) != "success":
                agg["fail_total"] = int(agg["fail_total"]) + 1
                key = str(result.status)
                if key in {"logical_fail", "syndrome_fail", "exception_fail"}:
                    agg[key] = int(agg[key]) + 1
            agg["runtime_ms_samples"].append(float(result.runtime_ms))
            agg["work_iterations_samples"].append(float(result.work_iterations))
            agg["edge_updates_samples"].append(float(result.edge_updates))
    return list(aggregates.values())


def _merge_task_rows(task_rows: Iterable[Dict[str, object]]) -> Dict[tuple, Dict[str, object]]:
    merged: Dict[tuple, Dict[str, object]] = {}
    for row in task_rows:
        key = (str(row["decoder"]), str(row["side"]), float(row["p"]), int(row["rounds"]))
        dst = merged.get(key)
        if dst is None:
            dst = _init_aggregate_row(
                decoder=str(row["decoder"]),
                side=str(row["side"]),
                p=float(row["p"]),
                rounds=int(row["rounds"]),
            )
            merged[key] = dst
        dst["shots"] = int(dst["shots"]) + int(row["shots"])
        dst["fail_total"] = int(dst["fail_total"]) + int(row["fail_total"])
        dst["logical_fail"] = int(dst["logical_fail"]) + int(row["logical_fail"])
        dst["syndrome_fail"] = int(dst["syndrome_fail"]) + int(row["syndrome_fail"])
        dst["exception_fail"] = int(dst["exception_fail"]) + int(row["exception_fail"])
        dst["runtime_ms_samples"].extend([float(x) for x in row["runtime_ms_samples"]])  # type: ignore[arg-type]
        dst["work_iterations_samples"].extend([float(x) for x in row["work_iterations_samples"]])  # type: ignore[arg-type]
        dst["edge_updates_samples"].extend([float(x) for x in row["edge_updates_samples"]])  # type: ignore[arg-type]
    return merged


def _finalize_summary(rows: Iterable[Dict[str, object]]) -> pd.DataFrame:
    summary_rows: List[Dict[str, object]] = []
    for row in rows:
        shots = int(row["shots"])
        fail_total = int(row["fail_total"])
        logical_fail = int(row["logical_fail"])
        syndrome_fail = int(row["syndrome_fail"])
        exception_fail = int(row["exception_fail"])
        runtime_ms_samples = [float(x) for x in row["runtime_ms_samples"]]
        work_iterations_samples = [float(x) for x in row["work_iterations_samples"]]
        edge_updates_samples = [float(x) for x in row["edge_updates_samples"]]
        fer = float(fail_total) / float(shots) if shots > 0 else float("nan")
        summary_rows.append(
            {
                "decoder": str(row["decoder"]),
                "side": str(row["side"]),
                "p": float(row["p"]),
                "rounds": int(row["rounds"]),
                "shots": int(shots),
                "fail_total": int(fail_total),
                "logical_fail": int(logical_fail),
                "syndrome_fail": int(syndrome_fail),
                "exception_fail": int(exception_fail),
                "fer": float(fer),
                "runtime_ms_mean": float(np.mean(runtime_ms_samples)) if runtime_ms_samples else float("nan"),
                "runtime_ms_p95": _quantile(runtime_ms_samples, 0.95),
                "work_iterations_mean": float(np.mean(work_iterations_samples)) if work_iterations_samples else float("nan"),
                "work_iterations_p95": _quantile(work_iterations_samples, 0.95),
                "work_iterations_p99": _quantile(work_iterations_samples, 0.99),
                "edge_updates_mean": float(np.mean(edge_updates_samples)) if edge_updates_samples else float("nan"),
                "edge_updates_p95": _quantile(edge_updates_samples, 0.95),
                "edge_updates_p99": _quantile(edge_updates_samples, 0.99),
            }
        )
    df = pd.DataFrame(summary_rows)
    if not df.empty:
        df = df.sort_values(["side", "rounds", "p", "decoder"]).reset_index(drop=True)
    return df


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def write_summary_artifacts(summary_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    try:
        summary_df.to_parquet(out_dir / "summary.parquet", index=False)
    except Exception:
        pass


def create_plots(summary_df: pd.DataFrame, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    if summary_df.empty:
        return paths
    for side in sorted(summary_df["side"].unique()):
        side_df = summary_df.loc[summary_df["side"] == side].copy()
        for rounds in sorted(side_df["rounds"].unique()):
            sub = side_df.loc[side_df["rounds"] == rounds].copy()
            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            companion_rows: List[Dict[str, object]] = []
            for decoder in sorted(sub["decoder"].unique()):
                dec_df = sub.loc[sub["decoder"] == decoder].sort_values("p")
                xs = dec_df["p"].to_numpy(dtype=float)
                ys_raw = dec_df["fer"].to_numpy(dtype=float)
                fer_floor = np.minimum(0.5 / np.maximum(dec_df["shots"].to_numpy(dtype=float), 1.0), 1.0)
                ys = np.maximum(ys_raw, fer_floor)
                ax.plot(xs, ys, marker="o", linewidth=1.8, label=str(decoder))
                for _, row in dec_df.iterrows():
                    companion_rows.append(
                        {
                            "decoder": str(decoder),
                            "p": float(row["p"]),
                            "rounds": int(rounds),
                            "shots": int(row["shots"]),
                            "fail_total": int(row["fail_total"]),
                            "logical_fail": int(row["logical_fail"]),
                            "syndrome_fail": int(row["syndrome_fail"]),
                            "exception_fail": int(row["exception_fail"]),
                            "fer": float(row["fer"]),
                            "work_iterations_mean": float(row["work_iterations_mean"]),
                            "work_iterations_p99": float(row["work_iterations_p99"]),
                        }
                    )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Physical bit-flip rate p")
            ax.set_ylabel("Full logical FER")
            ax.set_title(f"Gross repeated-syndrome benchmark, side={side}, rounds={int(rounds)}")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(loc="best")
            fig.tight_layout()
            fig_path = out_dir / f"fig_fer_vs_p_{side}_r{int(rounds):02d}.png"
            fig.savefig(fig_path, dpi=180)
            plt.close(fig)
            paths.append(fig_path)
            _write_csv(
                out_dir / f"table_fer_vs_p_{side}_r{int(rounds):02d}.csv",
                companion_rows,
                [
                    "decoder",
                    "p",
                    "rounds",
                    "shots",
                    "fail_total",
                    "logical_fail",
                    "syndrome_fail",
                    "exception_fail",
                    "fer",
                    "work_iterations_mean",
                    "work_iterations_p99",
                ],
            )

        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        companion_rows = []
        for decoder in sorted(side_df["decoder"].unique()):
            dec_df = side_df.loc[side_df["decoder"] == decoder].copy()
            x = dec_df["work_iterations_mean"].to_numpy(dtype=float)
            fer_raw = dec_df["fer"].to_numpy(dtype=float)
            fer_floor = np.minimum(0.5 / np.maximum(dec_df["shots"].to_numpy(dtype=float), 1.0), 1.0)
            y = np.maximum(fer_raw, fer_floor)
            ax.scatter(x, y, s=70, label=str(decoder))
            for _, row in dec_df.iterrows():
                ax.annotate(
                    f"r={int(row['rounds'])}, p={float(row['p']):.3g}",
                    (float(row["work_iterations_mean"]), max(float(row["fer"]), 0.5 / max(int(row["shots"]), 1))),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                )
                companion_rows.append(
                    {
                        "decoder": str(decoder),
                        "p": float(row["p"]),
                        "rounds": int(row["rounds"]),
                        "fer": float(row["fer"]),
                        "runtime_ms_mean": float(row["runtime_ms_mean"]),
                        "work_iterations_mean": float(row["work_iterations_mean"]),
                        "work_iterations_p99": float(row["work_iterations_p99"]),
                    }
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Mean BP/min-sum iterations (primary work metric)")
        ax.set_ylabel("Full logical FER")
        ax.set_title(f"Gross repeated-syndrome tradeoff, side={side}")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig_path = out_dir / f"fig_tradeoff_{side}.png"
        fig.savefig(fig_path, dpi=180)
        plt.close(fig)
        paths.append(fig_path)
        _write_csv(
            out_dir / f"table_tradeoff_{side}.csv",
            companion_rows,
            [
                "decoder",
                "p",
                "rounds",
                "fer",
                "runtime_ms_mean",
                "work_iterations_mean",
                "work_iterations_p99",
            ],
        )
    return paths


def write_run_report(summary_df: pd.DataFrame, out_dir: Path) -> Path:
    report_path = out_dir / "report.md"
    lines: List[str] = [
        "# Gross repeated-syndrome benchmark",
        "",
        "- Model: split X/Z repeated-syndrome Gross benchmark with independent data and measurement flip priors.",
        "- Reported FER: full logical FER = `fail_total / shots`.",
        "- Failure decomposition columns: `logical_fail`, `syndrome_fail`, `exception_fail`.",
        "",
    ]
    if summary_df.empty:
        lines.append("No rows were produced.")
        report_path.write_text("\n".join(lines) + "\n")
        return report_path

    for side in sorted(summary_df["side"].unique()):
        side_df = summary_df.loc[summary_df["side"] == side].copy()
        lines.append(f"## Side `{side}`")
        lines.append("")
        for rounds in sorted(side_df["rounds"].unique()):
            sub = side_df.loc[side_df["rounds"] == rounds].sort_values(["fer", "work_iterations_mean"])
            best = sub.iloc[0]
            lines.append(
                f"- Rounds `{int(rounds)}`: best observed FER is `{float(best['fer']):.6g}` "
                f"from `{best['decoder']}` at `p={float(best['p']):.6g}` with "
                f"`mean iters={float(best['work_iterations_mean']):.2f}`."
            )
        lines.append("")
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def load_summary(summary_path: Path) -> pd.DataFrame:
    return pd.read_csv(summary_path)


def run_benchmark(config: BenchmarkConfig) -> Path:
    if not config.p_values:
        raise ValueError("p_values must be non-empty")
    if not config.rounds_list:
        raise ValueError("rounds_list must be non-empty")
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(config.results_dir) if config.results_dir is not None else (DEFAULT_RESULTS_BASE / f"{ts}_grosscode_small")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis").mkdir(parents=True, exist_ok=True)
    run_config_path = out_dir / "run_config.json"
    run_config_path.write_text(json.dumps(config.to_json_ready(), indent=2))

    nproc = int(config.resolved_nproc())
    chunk_shots = int(config.chunk_shots) if int(config.chunk_shots) > 0 else _default_chunk_shots(int(config.shots_per_point), nproc)
    tasks: List[Dict[str, object]] = []
    task_index = 0
    for side in config.sides:
        for rounds in config.rounds_list:
            for p in config.p_values:
                remaining = int(config.shots_per_point)
                while remaining > 0:
                    shots = min(int(chunk_shots), int(remaining))
                    tasks.append(
                        {
                            "side": str(side),
                            "rounds": int(rounds),
                            "p": float(p),
                            "p_meas": float(clip_prob(float(p) * float(config.p_meas_scale))),
                            "shots": int(shots),
                            "seed": int(config.base_seed + task_index * 100003),
                            "hx_path": str(config.hx_path),
                            "hz_path": str(config.hz_path),
                            "decoders": [str(x) for x in config.decoders],
                            "baseline_max_iter": int(config.baseline_max_iter),
                            "baseline_osd_order": int(config.baseline_osd_order),
                            "full_block_max_iter": int(config.full_block_max_iter),
                            "full_block_alpha": float(config.full_block_alpha),
                            "factor_max_iter": int(config.factor_max_iter),
                            "factor_alpha": float(config.factor_alpha),
                            "window_size": int(config.window_size),
                            "sweeps": int(config.sweeps),
                            "belief_damping": float(config.belief_damping),
                            "topk_interface_bits": int(config.topk_interface_bits),
                        }
                    )
                    remaining -= int(shots)
                    task_index += 1
    task_rows: List[Dict[str, object]] = []
    t0 = time.perf_counter()
    total_tasks = len(tasks)
    completed = 0
    completed_shots = 0
    if nproc == 1:
        for task in tasks:
            result_rows = _run_task(task)
            task_rows.extend(result_rows)
            completed += 1
            completed_shots += sum(int(row["shots"]) for row in result_rows) // max(1, len(config.decoders))
            if completed % max(1, int(config.progress_every_tasks)) == 0 or completed == total_tasks:
                elapsed = time.perf_counter() - t0
                rate = float(completed) / max(elapsed, 1e-9)
                remaining_tasks = max(0, total_tasks - completed)
                eta = float(remaining_tasks) / max(rate, 1e-9)
                print(
                    "[progress] tasks={}/{} shots={} elapsed={} eta={}".format(
                        int(completed),
                        int(total_tasks),
                        int(completed_shots),
                        format_seconds(elapsed),
                        format_seconds(eta),
                    ),
                    flush=True,
                )
    else:
        executor_cls = ProcessPoolExecutor
        executor_note = "process"
        try:
            pool = executor_cls(max_workers=nproc)
        except (PermissionError, OSError):
            executor_cls = ThreadPoolExecutor
            executor_note = "thread"
            pool = executor_cls(max_workers=nproc)
        print(f"[start] executor={executor_note} workers={nproc} tasks={total_tasks}", flush=True)
        with pool:
            futures = [pool.submit(_run_task, task) for task in tasks]
            for future in as_completed(futures):
                result_rows = future.result()
                task_rows.extend(result_rows)
                completed += 1
                completed_shots += sum(int(row["shots"]) for row in result_rows) // max(1, len(config.decoders))
                if completed % max(1, int(config.progress_every_tasks)) == 0 or completed == total_tasks:
                    elapsed = time.perf_counter() - t0
                    rate = float(completed) / max(elapsed, 1e-9)
                    remaining_tasks = max(0, total_tasks - completed)
                    eta = float(remaining_tasks) / max(rate, 1e-9)
                    print(
                        "[progress] tasks={}/{} shots={} elapsed={} eta={}".format(
                            int(completed),
                            int(total_tasks),
                            int(completed_shots),
                            format_seconds(elapsed),
                            format_seconds(eta),
                        ),
                        flush=True,
                    )
    merged = _merge_task_rows(task_rows)
    summary_df = _finalize_summary(merged.values())
    write_summary_artifacts(summary_df, out_dir / "analysis")
    create_plots(summary_df, out_dir / "plots")
    write_run_report(summary_df, out_dir / "analysis")
    return out_dir
