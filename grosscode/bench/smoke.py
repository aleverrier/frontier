from __future__ import annotations

import argparse
from pathlib import Path

from grosscode.bench.compare import run_benchmark
from grosscode.utils.paths import DEFAULT_RESULTS_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Short smoke benchmark for the grosscode package. "
            "Runs one deterministic compare harness over bp, minsum, and local_round under the public bravyi_depth7 backend."
        )
    )
    parser.add_argument("--backend", type=str, default="bravyi_depth7")
    parser.add_argument("--error-rate", type=float, default=0.004)
    parser.add_argument("--shots", type=int, default=3)
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--scms", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_ROOT / "_smoke_grosscode_package")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compare_args = argparse.Namespace(
        backend=args.backend,
        error_rate=float(args.error_rate),
        shots=int(args.shots),
        cpus=int(args.cpus),
        seed=int(args.seed),
        decoders="bp,minsum,local_round",
        bp_max_iter=25,
        minsum_max_iter=40,
        minsum_scale=0.75,
        scms=bool(args.scms),
        window_round_radius=1,
        window_max_iter=40,
        local_round_passes=3,
        progress_every=1,
        output_dir=args.output_dir,
        no_plot=bool(args.no_plot),
    )
    out_dir = run_benchmark(compare_args)
    print(f"[done] smoke artifacts written to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
