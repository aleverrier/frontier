#!/usr/bin/env python3
"""Generate matched DEM sample rows for frontier replay.

Public entry point: `main`.

This module is a CLI/support module, not the preferred library API.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grosscode.dem.builder import SplitSectorProblem, build_split_sector_problem


ROW_FIELDS = (
    "scope",
    "shot",
    "source_shot",
    "source_row_identifier",
    "seed",
    "p_location",
    "truth_syndrome",
    "truth_logical",
    "truth_detector_weight",
    "truth_logical_weight",
    "sampler_recipe",
)


def _parse_scopes(raw: str) -> tuple[str, ...]:
    scopes = tuple(piece.strip() for piece in str(raw).split(",") if piece.strip())
    if not scopes:
        raise ValueError("--scopes must contain at least one scope")
    allowed = {"memory_X", "memory_Z"}
    bad = sorted(set(scopes) - allowed)
    if bad:
        raise ValueError(f"unsupported scope(s): {', '.join(bad)}")
    return scopes


def _int_from_bits(bits: np.ndarray) -> int:
    value = 0
    for index in np.flatnonzero(bits):
        value |= 1 << int(index)
    return int(value)


def _json_write(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _side_matrices(problem: SplitSectorProblem, scope: str):
    if str(scope) == "memory_X":
        return problem.D_X.tocsr(), problem.O_X.tocsr(), np.asarray(problem.priors_X, dtype=np.float64)
    if str(scope) == "memory_Z":
        return problem.D_Z.tocsr(), problem.O_Z.tocsr(), np.asarray(problem.priors_Z, dtype=np.float64)
    raise ValueError(f"unsupported scope: {scope}")


def _scope_seed(base_seed: int, scope: str) -> int:
    return int(base_seed) if str(scope) == "memory_X" else int(base_seed) + 1_000_003


def _iter_sample_rows(
    *,
    problem: SplitSectorProblem,
    scope: str,
    p_location: float,
    shot_start: int,
    shots: int,
    seed: int,
    chunk_size: int,
    progress_every_rows: int,
    written_before_scope: int,
    total_rows: int,
    started: float,
):
    detector, logical, priors = _side_matrices(problem, str(scope))
    ncols = int(detector.shape[1])
    if priors.shape != (ncols,):
        raise ValueError(f"{scope} priors shape {priors.shape} does not match matrix columns {ncols}")
    rng = np.random.default_rng(_scope_seed(int(seed), str(scope)))
    done = 0
    while done < int(shots):
        chunk = min(int(chunk_size), int(shots) - int(done))
        errors_by_shot = rng.random((chunk, ncols), dtype=np.float64) < priors[None, :]
        errors_by_col = errors_by_shot.T.astype(np.uint8, copy=False)
        syndromes = np.asarray(detector @ errors_by_col, dtype=np.uint8) & 1
        logicals = np.asarray(logical @ errors_by_col, dtype=np.uint8) & 1
        detector_weights = np.asarray(syndromes.sum(axis=0)).reshape(-1)
        logical_weights = np.asarray(logicals.sum(axis=0)).reshape(-1)
        for offset in range(chunk):
            shot = int(shot_start) + int(done) + int(offset)
            syndrome_int = _int_from_bits(syndromes[:, int(offset)])
            logical_int = _int_from_bits(logicals[:, int(offset)])
            yield {
                "scope": str(scope),
                "shot": int(shot),
                "source_shot": int(shot),
                "source_row_identifier": f"{scope}:seed{int(seed)}:shot{int(shot)}",
                "seed": int(seed),
                "p_location": float(p_location),
                "truth_syndrome": int(syndrome_int),
                "truth_logical": int(logical_int),
                "truth_detector_weight": int(detector_weights[int(offset)]),
                "truth_logical_weight": int(logical_weights[int(offset)]),
                "sampler_recipe": "independent_binary_dem_columns_v1",
            }
        done += chunk
        sampled_total = int(written_before_scope) + int(done)
        if int(progress_every_rows) > 0 and (
            sampled_total == total_rows or sampled_total % int(progress_every_rows) < chunk
        ):
            elapsed = time.perf_counter() - float(started)
            rate = sampled_total / elapsed if elapsed > 0.0 else 0.0
            eta = (total_rows - sampled_total) / rate if rate > 0.0 else float("nan")
            eta_text = f"{eta:.1f}s" if math.isfinite(float(eta)) else "unknown"
            print(
                f"[progress] sampled_rows={sampled_total}/{total_rows} elapsed={elapsed:.1f}s "
                f"rate={rate:.1f}/s eta={eta_text}",
                flush=True,
            )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DEM sample rows for frontier-replay.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""example:
  python -m tools.frontier_sample_rows --out sample_rows.csv --backend rotated_surface_d3 --p-location 0.001 --shots 4 --seed 20260615

See docs/COMMANDS.md for command details.""",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output sample_rows.csv path.")
    parser.add_argument("--metadata-out", type=Path, help="Optional metadata JSON path.")
    parser.add_argument("--backend", default="bravyi_depth7")
    parser.add_argument("--p-location", type=float, required=True)
    parser.add_argument("--shots", type=int, required=True, help="Number of shots per selected scope.")
    parser.add_argument("--shot-start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--scopes", default="memory_X,memory_Z")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--progress-every-rows", type=int, default=1000)
    parser.add_argument("--allow-existing", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if int(args.shots) <= 0:
        raise ValueError("--shots must be positive")
    if int(args.shot_start) < 0:
        raise ValueError("--shot-start must be non-negative")
    if int(args.chunk_size) <= 0:
        raise ValueError("--chunk-size must be positive")
    scopes = _parse_scopes(str(args.scopes))
    out_path = Path(args.out).expanduser().resolve()
    metadata_path = (
        Path(args.metadata_out).expanduser().resolve()
        if args.metadata_out is not None
        else out_path.with_name(out_path.stem + "_metadata.json")
    )
    if out_path.exists() and not bool(args.allow_existing):
        raise FileExistsError(f"{out_path} exists; pass --allow-existing to overwrite")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    load_started = time.perf_counter()
    problem = build_split_sector_problem(
        backend=str(args.backend),
        error_rate=float(args.p_location),
    )
    matrix_by_scope: dict[str, dict[str, object]] = {}
    for scope in scopes:
        detector, logical, priors = _side_matrices(problem, str(scope))
        metadata = problem.metadata_X if str(scope) == "memory_X" else problem.metadata_Z
        matrix_by_scope[str(scope)] = {
            "detector_matrix": f"{int(detector.shape[0])}x{int(detector.shape[1])}",
            "logical_matrix": f"{int(logical.shape[0])}x{int(logical.shape[1])}",
            "columns": int(detector.shape[1]),
            "edges": int(detector.nnz),
            "priors": int(priors.size),
            "noisy_rounds": int(metadata.noisy_rounds),
            "total_rounds": int(metadata.total_rounds),
            "stim_path": str(metadata.stim_path),
        }
    print(
        f"[setup] loaded backend={args.backend} p={float(args.p_location):.6g} "
        f"scopes={','.join(scopes)} in {time.perf_counter() - load_started:.1f}s",
        flush=True,
    )

    started = time.perf_counter()
    total_rows = int(args.shots) * len(scopes)
    written = 0
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        for scope in scopes:
            for row in _iter_sample_rows(
                problem=problem,
                scope=str(scope),
                p_location=float(args.p_location),
                shot_start=int(args.shot_start),
                shots=int(args.shots),
                seed=int(args.seed),
                chunk_size=int(args.chunk_size),
                progress_every_rows=int(args.progress_every_rows),
                written_before_scope=int(written),
                total_rows=int(total_rows),
                started=float(started),
            ):
                writer.writerow(row)
            written += int(args.shots)
            handle.flush()

    payload = {
        "backend": str(args.backend),
        "p_location": float(args.p_location),
        "shot_start": int(args.shot_start),
        "shots_per_scope": int(args.shots),
        "scopes": list(scopes),
        "seed": int(args.seed),
        "sample_rows": str(out_path),
        "sampler_recipe": "independent_binary_dem_columns_v1",
        "matrix_by_scope": matrix_by_scope,
        "elapsed_s": float(time.perf_counter() - started),
    }
    _json_write(metadata_path, payload)
    print(f"[done] sample_rows={out_path}", flush=True)
    print(f"[done] metadata={metadata_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
