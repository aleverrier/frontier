from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from paper.plots.scripts import reproduce_plots

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "paper" / "plots" / "manifest.csv"
PLOTS_README = REPO_ROOT / "paper" / "plots" / "README.md"
DATA_MANIFEST = REPO_ROOT / "paper" / "plots" / "data" / "MANIFEST.md"
REQUIRED_SIDECAR_FIELDS = {
    "description",
    "columns",
    "source_command",
    "raw_source",
    "commit_hash",
    "release_version",
    "code_version",
    "dependency_constraints_file",
    "random_seeds",
    "sample_count",
    "decoder_settings",
    "confidence_interval_method",
    "csv_sha256",
    "caveats",
}
REQUIRED_COLUMNS = reproduce_plots.REQUIRED_COLUMNS
ALLOWED_STATUSES = {
    "reproducible",
    "data-missing",
    "script-missing",
    "external-archive-needed",
    "TODO",
}
ALLOWED_DATA_KINDS = {
    "raw",
    "derived-summary",
    "digitized",
    "synthetic-demo",
    "TODO",
}


def _run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def _manifest_rows() -> list[dict[str, str]]:
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == REQUIRED_COLUMNS
        return list(reader)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_paper_plot_manifest_schema_and_statuses_are_explicit() -> None:
    rows = _manifest_rows()

    for row in rows:
        assert row["status"] in ALLOWED_STATUSES
        assert row["data_kind"] in ALLOWED_DATA_KINDS
        assert row["status"] != "reproducible" or row["data_kind"] != "TODO"
        assert row["data_kind"] != "synthetic-demo" or "demo" in row["notes"].lower()


def test_reproducible_rows_point_to_committed_data_and_scripts(tmp_path) -> None:
    rows = _manifest_rows()
    reproducible_rows = [row for row in rows if row["status"] == "reproducible"]

    for row in reproducible_rows:
        data_path = REPO_ROOT / row["data_file"]
        script_path = REPO_ROOT / row["plotting_script"]
        assert data_path.exists(), row
        assert script_path.exists(), row
        assert row["output_file"].startswith("paper/plots/outputs/")

    if reproducible_rows:
        figure_id = reproducible_rows[0]["figure_id"]
        _run_command(
            [
                sys.executable,
                "paper/plots/scripts/reproduce_plots.py",
                "--figure",
                figure_id,
                "--out-dir",
                str(tmp_path),
                "--strict",
            ]
        )
        expected_name = Path(reproducible_rows[0]["output_file"]).name
        assert (tmp_path / expected_name).exists()
    else:
        readme = PLOTS_README.read_text(encoding="utf-8")
        assert "plot-ready summary tables" in readme
        assert "script-missing" in readme
        assert "yet reproducible" in readme


def test_script_missing_rows_point_to_committed_data_and_sidecars() -> None:
    rows = _manifest_rows()
    script_missing_rows = [row for row in rows if row["status"] == "script-missing"]
    assert script_missing_rows

    for row in script_missing_rows:
        data_path = REPO_ROOT / row["data_file"]
        sidecar_path = data_path.with_suffix(".json")
        assert data_path.exists(), row
        assert sidecar_path.exists(), row
        assert not row["plotting_script"], row
        assert row["output_file"].startswith("paper/plots/outputs/")

        with data_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            csv_columns = next(reader)

        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert REQUIRED_SIDECAR_FIELDS <= set(sidecar), row
        assert sidecar["csv_sha256"] == _sha256(data_path), row
        assert set(csv_columns) == set(sidecar["columns"]), row


def test_paper_plot_reproduction_cli_reports_missing_data_honestly() -> None:
    list_result = _run_command([sys.executable, "paper/plots/scripts/reproduce_plots.py", "--list"])
    all_result = _run_command([sys.executable, "paper/plots/scripts/reproduce_plots.py", "--all"])

    rows = _manifest_rows()
    if rows:
        assert "script-missing" in list_result.stdout
        assert "Skipping" in all_result.stdout
    else:
        assert "No paper figure rows are declared" in list_result.stdout
        assert "data-missing" in list_result.stdout
        assert "No paper plot rows selected" in all_result.stdout


def test_paper_plot_data_manifest_is_deterministic() -> None:
    generated = _run_command(
        [
            sys.executable,
            "-m",
            "tools.asset_manifest",
            "--root",
            "paper/plots/data",
            "--title",
            "Paper Plot Data Manifest",
        ]
    ).stdout
    assert DATA_MANIFEST.read_text(encoding="utf-8") == generated
