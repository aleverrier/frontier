from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from tools import frontier_sample_replay
from tools import frontier_sample_rows

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_minimal_decode_example_succeeds() -> None:
    result = _run_command([sys.executable, "examples/minimal_decode.py"])

    assert "single status=ok" in result.stdout
    assert "committee status=ok" in result.stdout


def test_inspect_dem_example_succeeds() -> None:
    result = _run_command([sys.executable, "examples/inspect_dem.py"])

    assert "memory_X: detector=24x221" in result.stdout
    assert "memory_Z: detector=24x219" in result.stdout


def test_cli_help_commands_succeed() -> None:
    modules = (
        "tools.frontier_decoder",
        "tools.dem_loader",
        "tools.frontier_sample_rows",
        "tools.frontier_sample_replay",
        "tools.frontier_bb144_benchmark",
    )
    for module in modules:
        result = _run_command([sys.executable, "-m", module, "--help"])
        assert "docs/COMMANDS.md" in result.stdout


def test_tiny_rotated_surface_replay_outputs(tmp_path) -> None:
    sample_rows = tmp_path / "sample_rows.csv"
    out_dir = tmp_path / "replay"

    assert (
        frontier_sample_rows.main(
            [
                "--out",
                str(sample_rows),
                "--backend",
                "rotated_surface_d3",
                "--p-location",
                "0.001",
                "--shots",
                "2",
                "--seed",
                "20260618",
                "--progress-every-rows",
                "0",
            ]
        )
        == 0
    )
    assert (
        frontier_sample_replay.main(
            [
                "--sample-rows",
                str(sample_rows),
                "--out-dir",
                str(out_dir),
                "--code",
                "rotated_surface_d3",
                "--backend",
                "rotated_surface_d3",
                "--p-location",
                "0.001",
                "--shot-start",
                "0",
                "--shot-stop",
                "1",
                "--K",
                "16",
                "--Delta",
                "100",
                "--direction-mode",
                "fwd_bwd_committee",
                "--engine",
                "auto",
                "--column-order",
                "deadline_reorder",
                "--backward-column-order",
                "backward_deadline_reorder",
                "--cpus",
                "1",
                "--shards-per-side",
                "1",
                "--progress-every-shards",
                "1",
            ]
        )
        == 0
    )

    summary_path = out_dir / "summary_by_scope.csv"
    metadata_path = out_dir / "run_metadata.json"
    assert summary_path.exists()
    assert metadata_path.exists()

    with summary_path.open(newline="", encoding="utf-8") as handle:
        scopes = {row["scope"] for row in csv.DictReader(handle)}
    assert scopes == {"memory_X", "memory_Z", "combined"}

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "complete"
    assert metadata["code"] == "rotated_surface_d3"
