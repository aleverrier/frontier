from __future__ import annotations

import csv
import importlib
import tomllib
from pathlib import Path

import pytest

import frontier
from frontier import progressive
from tools import dem_loader
from tools import frontier_decoder as frontier_impl
from tools import frontier_sample_rows
from tools import frontier_sample_replay as replay

REPO_ROOT = Path(__file__).resolve().parents[1]


def _factor(
    factor_id: int,
    p0: float,
    det1: int,
    log1: int,
) -> progressive.FactorTransition:
    return progressive.FactorTransition(
        factor_id=int(factor_id),
        outcomes=(
            progressive.OutcomeTransition(probability=float(p0), detector_mask=0, logical_mask=0),
            progressive.OutcomeTransition(
                probability=float(1.0 - p0),
                detector_mask=int(det1),
                logical_mask=int(log1),
            ),
        ),
        instruction_offset=int(factor_id),
        label=f"f{int(factor_id)}",
    )


def _model() -> frontier.FrontierModel:
    columns = tuple(
        progressive.columns_from_factor_transitions(
            (
                _factor(0, 0.55, 1, 1),
                _factor(1, 0.70, 1, 0),
            )
        )
    )
    return frontier.FrontierModel(
        columns=columns,
        layout=progressive.build_frontier_layout(list(columns), num_detectors=1),
        num_detectors=1,
        num_observables=1,
    )


def test_native_committee_matches_binary_adapter() -> None:
    if not frontier.native_binary_available():
        pytest.skip("native frontier extension is not built")

    model = _model()
    binary = frontier.decode_frontier_committee(model, 0, K=16, Delta=100.0, _engine="binary")
    native = frontier.decode_frontier_committee(model, 0, K=16, Delta=100.0, _engine="native_binary")

    assert native.status == binary.status
    assert native.logical_hat == binary.logical_hat
    assert native.direction == binary.direction
    assert native.log_evidence == pytest.approx(binary.log_evidence, rel=0.0, abs=1e-12)


def test_export_direction_modes_are_limited_to_selected_decoder() -> None:
    assert replay._normalize_direction_mode(None, "bidirectional_committee") == "fwd_bwd_committee"
    assert replay._normalize_direction_mode("forward", "bidirectional_committee") == "forward_only"
    assert replay._normalize_direction_mode("backward", "bidirectional_committee") == "backward_only"
    with pytest.raises(ValueError):
        replay._normalize_direction_mode("unsupported_mode", "bidirectional_committee")
    with pytest.raises(ValueError):
        replay._normalize_direction_mode("pressure_select", "bidirectional_committee")


def test_native_wrapper_exposes_only_public_decode_methods() -> None:
    if not frontier.native_binary_available():
        pytest.skip("native frontier extension is not built")

    native_model = frontier_impl._get_native_binary_model(_model())
    assert hasattr(native_model, "decode")
    assert hasattr(native_model, "decode_many")
    assert hasattr(native_model, "decode_many_select_replay")


def test_sample_rows_cli_generates_replay_loadable_rows(tmp_path) -> None:
    out = tmp_path / "sample_rows.csv"
    meta = tmp_path / "sample_rows_metadata.json"

    frontier_sample_rows.main(
        [
            "--out",
            str(out),
            "--metadata-out",
            str(meta),
            "--backend",
            "rotated_surface_d3",
            "--p-location",
            "0.001",
            "--shots",
            "2",
            "--seed",
            "1234",
            "--progress-every-rows",
            "0",
        ]
    )

    with out.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["scope"] for row in rows} == {"memory_X", "memory_Z"}
    by_scope = replay._load_sample_rows(out, scopes=("memory_X", "memory_Z"), shot_start=0, shot_stop=1)
    assert len(by_scope["memory_X"]) == 2
    assert len(by_scope["memory_Z"]) == 2
    assert meta.exists()


def test_dem_info_bad_gross_asset_override_prints_no_partial_header(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setenv("GROSSCODE_ASSET_ROOT", str(tmp_path / "missing_assets"))

    assert dem_loader.main(["--backend", "bravyi_depth7", "--p-location", "0.001"]) == 1
    captured = capsys.readouterr()
    assert "scope,detector_matrix" not in captured.out
    assert "GROSSCODE_ASSET_ROOT" in captured.err


def test_pressure_means_are_nan_without_runtime_warning() -> None:
    row = replay._summary_row(
        "memory_X",
        [
            {
                "frame_fail_type": "success",
                "failure_diagnosis": "success",
                "decode_s": 0.1,
                "transition_evals_total": 10,
                "pressure_forward": float("nan"),
                "pressure_backward": float("nan"),
                "processed_columns": 1,
                "sum_post_prune_state_count": 1,
                "noisy_rounds": 3,
            }
        ],
        sample_rows=__file__,
    )
    assert row["pressure_forward_mean"] != row["pressure_forward_mean"]
    assert row["pressure_backward_mean"] != row["pressure_backward_mean"]


def test_frontier_package_reexports_public_decoder_api() -> None:
    frontier_pkg = importlib.import_module("frontier")
    dem_pkg = importlib.import_module("frontier.dem")
    progressive_pkg = importlib.import_module("frontier.progressive")

    assert frontier_pkg.FrontierModel is frontier.FrontierModel
    assert frontier_pkg.decode_frontier is frontier.decode_frontier
    assert dem_pkg.load_dem_family is dem_loader.load_dem_family
    assert progressive_pkg.FactorTransition is progressive.FactorTransition
    progressive_impl = importlib.import_module("tools.frontier_progressive")
    assert progressive_pkg.columns_from_factor_transitions is not progressive_impl._columns_from_factor_transitions

    from frontier import FrontierModel, decode_frontier
    from frontier.dem import load_dem_family

    model = _model()
    result = decode_frontier(model, 0, K=16, Delta=100.0)
    assert isinstance(model, FrontierModel)
    assert result.status == "ok"
    assert callable(load_dem_family)


def test_console_script_modules_have_main() -> None:
    script_modules = {
        "frontier-smoke": "tools.frontier_decoder",
        "frontier-dem-info": "tools.dem_loader",
        "frontier-sample-rows": "tools.frontier_sample_rows",
        "frontier-replay": "tools.frontier_sample_replay",
        "frontier-bb144-benchmark": "tools.frontier_bb144_benchmark",
    }
    for script, module_name in script_modules.items():
        module = importlib.import_module(module_name)
        assert callable(getattr(module, "main", None)), script


def test_architecture_docs_reference_key_files() -> None:
    text = (REPO_ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    for path in (
        "frontier_native.py",
        "frontier/progressive.py",
        "native/_frontier_native.cpp",
        "tools/frontier_decoder.py",
        "tools/frontier_progressive.py",
        "tools/dem_loader.py",
        "tools/frontier_sample_rows.py",
        "tools/frontier_sample_replay.py",
        "tools/frontier_bb144_benchmark.py",
        "grosscode/dem/builder.py",
        "grosscode/circuits/backends.py",
        "tests/test_frontier_export.py",
    ):
        assert path in text


def test_commands_docs_reference_all_console_scripts() -> None:
    text = (REPO_ROOT / "docs" / "COMMANDS.md").read_text(encoding="utf-8")
    for script in (
        "frontier-smoke",
        "frontier-dem-info",
        "frontier-sample-rows",
        "frontier-replay",
        "frontier-bb144-benchmark",
    ):
        assert script in text


def test_file_scope_mentions_new_docs_and_examples() -> None:
    text = (REPO_ROOT / "docs" / "FILE_SCOPE.md").read_text(encoding="utf-8")
    for path in (
        "LICENSE",
        "CITATION.cff",
        "ACKNOWLEDGEMENTS.md",
        "CONTRIBUTING.md",
        "CHANGELOG.md",
        "NOTICE",
        "AGENTS.md",
        "constraints/README.md",
        "constraints/py314-macos-validated.txt",
        "docs/ACADEMIC_METADATA.md",
        "docs/ASSET_PROVENANCE.md",
        "docs/ASSET_MANIFEST.md",
        "docs/REPRODUCIBILITY.md",
        "docs/RELEASE.md",
        "docs/ARCHITECTURE.md",
        "docs/COMMANDS.md",
        "docs/ENVIRONMENT.md",
        "docs/LICENSING.md",
        "Makefile",
        ".github/workflows/ci.yml",
        "frontier/__init__.py",
        "frontier/decoder.py",
        "frontier/dem.py",
        "frontier/progressive.py",
        "frontier/py.typed",
        "examples/README.md",
        "examples/minimal_decode.py",
        "examples/inspect_dem.py",
        "examples/replay_rotated_surface_d3.sh",
        "tests/test_examples_and_cli.py",
        "tools/asset_manifest.py",
    ):
        assert path in text


def test_license_metadata_docs_are_present() -> None:
    license_text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    licensing_doc = (REPO_ROOT / "docs" / "LICENSING.md").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    pyproject_data = tomllib.loads(pyproject)
    file_scope = (REPO_ROOT / "docs" / "FILE_SCOPE.md").read_text(encoding="utf-8")

    assert "Apache License" in license_text
    assert "Version 2.0" in license_text
    assert "Apache License 2.0" in licensing_doc
    assert "Apache License 2.0" in readme
    assert pyproject_data["project"]["license"] == "Apache-2.0"
    assert pyproject_data["project"]["license-files"] == ["LICENSE", "NOTICE"]
    assert "License :: OSI Approved :: Apache Software License" not in pyproject
    assert "LICENSE" in file_scope


def test_academic_metadata_docs_are_present_and_linked() -> None:
    citation = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    provenance = (REPO_ROOT / "docs" / "ASSET_PROVENANCE.md").read_text(encoding="utf-8")

    assert "cff-version: 1.2.0" in citation
    assert 'license: "Apache-2.0"' in citation
    assert "CITATION.cff" in readme
    assert "docs/REPRODUCIBILITY.md" in readme
    assert "grosscode/assets/gross144" in provenance
    assert (REPO_ROOT / "CONTRIBUTING.md").exists()
    assert (REPO_ROOT / "CHANGELOG.md").exists()


def test_public_model_construction_uses_public_progressive_helper() -> None:
    test_source = (REPO_ROOT / "tests" / "test_frontier_export.py").read_text(encoding="utf-8")
    private_helper = "progressive." + "_columns_from_factor_transitions"
    assert "progressive.columns_from_factor_transitions" in test_source
    assert private_helper not in test_source


def test_public_docs_contain_no_provisional_markers() -> None:
    marker = "TO" + "DO"
    scanned_paths = [
        REPO_ROOT / "ACKNOWLEDGEMENTS.md",
        REPO_ROOT / "CITATION.cff",
        REPO_ROOT / "CONTRIBUTING.md",
        REPO_ROOT / "README.md",
        REPO_ROOT / "constraints" / "README.md",
        REPO_ROOT / "docs" / "ACADEMIC_METADATA.md",
        REPO_ROOT / "docs" / "ASSET_PROVENANCE.md",
        REPO_ROOT / "docs" / "FILE_SCOPE.md",
        REPO_ROOT / "docs" / "REPRODUCIBILITY.md",
        REPO_ROOT / "grosscode" / "circuits" / "backends.py",
    ]
    for path in scanned_paths:
        assert marker not in path.read_text(encoding="utf-8"), path
