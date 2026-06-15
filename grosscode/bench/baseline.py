from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from grosscode.circuits.backends import resolve_backend_circuit, resolve_public_baseline_script
from grosscode.utils.paths import REPO_ROOT


def build_public_baseline_command(
    *,
    backend: str,
    sectors: list[str],
    error_rate: float,
    shots: int,
    seed: int,
    cpus: int,
    decoders: str,
    results_dir: Path | None,
) -> list[str]:
    script = resolve_public_baseline_script()
    cmd = [str(REPO_ROOT / "tools" / "py"), str(script)]
    for sector in sectors:
        spec = resolve_backend_circuit(backend=backend, sector=sector, error_rate=error_rate)
        cmd.extend(["--circuit", f"gross_mem_{sector.lower()}={spec.stim_path}"])
    cmd.extend(
        [
            "--p-list",
            f"{float(error_rate):g}",
            "--rounds",
            str(int(resolve_backend_circuit(backend=backend, sector=sectors[0], error_rate=error_rate).syndrome_rounds)),
            "--shots",
            str(int(shots)),
            "--seed",
            str(int(seed)),
            "--cpus",
            str(int(cpus)),
            "--decoders",
            decoders,
        ]
    )
    if results_dir is not None:
        cmd.extend(["--results-dir", str(results_dir)])
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Thin wrapper around the public gross144_circuit_level_beam_v17_scan.py baseline. "
            "This command prints the exact reproducibility command by default and can optionally run it."
        )
    )
    parser.add_argument("--backend", type=str, default="bravyi_depth7")
    parser.add_argument("--sector", type=str, default="both", choices=["X", "Z", "both"])
    parser.add_argument("--error-rate", type=float, default=0.004)
    parser.add_argument("--shots", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--decoders", type=str, default="beam,v17")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--run", action="store_true", help="Execute the public baseline command instead of only printing it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sectors = ["X", "Z"] if args.sector == "both" else [str(args.sector)]
    cmd = build_public_baseline_command(
        backend=str(args.backend),
        sectors=sectors,
        error_rate=float(args.error_rate),
        shots=int(args.shots),
        seed=int(args.seed),
        cpus=int(args.cpus),
        decoders=str(args.decoders),
        results_dir=args.results_dir,
    )
    print(shlex.join(cmd))
    if args.run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
