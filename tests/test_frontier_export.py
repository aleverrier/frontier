from __future__ import annotations

import pytest

from tools import frontier_decoder as frontier
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
        replay._normalize_direction_mode("stage1_nocap_stage2", "bidirectional_committee")
    with pytest.raises(ValueError):
        replay._normalize_direction_mode("pressure_select", "bidirectional_committee")


def test_native_wrapper_does_not_expose_two_stage_methods() -> None:
    if not frontier.native_binary_available():
        pytest.skip("native frontier extension is not built")

    native_model = frontier._get_native_binary_model(_model())
    assert not hasattr(native_model, "decode_overlap1_first_stage")
    assert not hasattr(native_model, "decode_many_stage1_nocap_stage2_replay")
