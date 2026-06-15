from __future__ import annotations

import csv
import sys

import pytest

from tools import dem_loader
from tools import frontier_decoder as frontier
from tools import frontier_sample_rows
from tools import frontier_sample_replay as replay
from tools import steane_progressive_decoder as progressive


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
        progressive._columns_from_factor_transitions(
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

    native_model = frontier._get_native_binary_model(_model())
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
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "frontier-dem-info",
            "--backend",
            "bravyi_depth7",
            "--p-location",
            "0.001",
        ],
    )

    assert dem_loader.main() == 1
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
                "primary_transition_evals_total": 10,
                "escalation_transition_evals_total": 0,
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
