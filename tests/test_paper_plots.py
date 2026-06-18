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
DATA_MANIFEST = REPO_ROOT / "paper" / "plots" / "data" / "MANIFEST.md"
REQUIRED_COLUMNS = reproduce_plots.REQUIRED_COLUMNS
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
    "plot_reproducibility",
    "simulation_reproducibility",
    "raw_corpus",
    "source_artifact",
    "source_checkout_hash",
    "renderer",
    "output_file",
    "output_files",
}
ALLOWED_STATUSES = {
    "reproducible",
    "support-data",
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
EXPECTED_FIGURE_IDS = {
    "frontier_schematic",
    "algorithm",
    "surface_threshold",
    "color_threshold",
    "surface_memory_z_dem_mwpm",
    "bb72_dem_circuit",
    "gross_dem_circuit",
    "gross_dem_avg_retained",
    "gross_dem_avg_retained_duplicate",
    "transition_evals",
    "failure_decomposition",
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


def test_paper_plot_manifest_schema_statuses_and_expected_figures() -> None:
    rows = _manifest_rows()
    assert {row["figure_id"] for row in rows} == EXPECTED_FIGURE_IDS

    for row in rows:
        assert row["status"] in ALLOWED_STATUSES
        assert row["data_kind"] in ALLOWED_DATA_KINDS
        assert row["status"] != "reproducible" or row["data_kind"] != "TODO"
        assert row["data_kind"] != "synthetic-demo" or "demo" in row["notes"].lower()


def test_reproducible_rows_point_to_committed_data_and_scripts() -> None:
    rows = _manifest_rows()
    reproducible_rows = [row for row in rows if row["status"] == "reproducible"]
    assert reproducible_rows

    for row in reproducible_rows:
        data_path = REPO_ROOT / row["data_file"]
        script_path = REPO_ROOT / row["plotting_script"]
        assert data_path.exists(), row
        assert data_path.with_suffix(".json").exists(), row
        assert script_path.exists(), row
        assert row["output_file"].startswith("paper/plots/outputs/")


def test_support_data_rows_are_committed_but_not_standalone_outputs() -> None:
    rows = _manifest_rows()
    support_rows = [row for row in rows if row["status"] == "support-data"]
    assert support_rows

    for row in support_rows:
        data_path = REPO_ROOT / row["data_file"]
        assert data_path.exists(), row
        assert data_path.with_suffix(".json").exists(), row
        assert "not a standalone" in row["notes"]


def test_each_actual_paper_figure_has_a_reproducible_output_row() -> None:
    rows = _manifest_rows()
    reproducible_ids = {row["figure_id"] for row in rows if row["status"] == "reproducible"}
    assert EXPECTED_FIGURE_IDS <= reproducible_ids


def test_sidecars_match_csvs_and_state_plot_reproducibility() -> None:
    for row in _manifest_rows():
        if not row["data_file"]:
            continue
        data_path = REPO_ROOT / row["data_file"]
        sidecar_path = data_path.with_suffix(".json")
        with data_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            csv_columns = next(reader)

        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert REQUIRED_SIDECAR_FIELDS <= set(sidecar), row
        assert sidecar["csv_sha256"] == _sha256(data_path), row
        assert set(csv_columns) == set(sidecar["columns"]), row
        assert sidecar["plot_reproducibility"] == "committed-summary-data", row
        assert sidecar["simulation_reproducibility"] == "raw-corpus-not-committed", row
        assert sidecar["raw_corpus"]["available_in_repo"] is False, row


def test_paper_plot_reproduction_cli_lists_and_renders_all_outputs(tmp_path) -> None:
    list_result = _run_command([sys.executable, "paper/plots/scripts/reproduce_plots.py", "--list"])
    assert "reproducible" in list_result.stdout
    assert "support-data" in list_result.stdout

    _run_command(
        [
            sys.executable,
            "paper/plots/scripts/reproduce_plots.py",
            "--all",
            "--strict",
            "--out-dir",
            str(tmp_path),
        ]
    )

    rows = _manifest_rows()
    for row in rows:
        if row["status"] != "reproducible":
            continue
        expected_name = Path(row["output_file"]).name
        assert (tmp_path / expected_name).exists(), row


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
