"""Deterministic tests for the bounded-hypergraph overlap profiler."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import frontier_overlap_profile as overlap_profile
from tools import frontier_progressive as progressive


def _column(
    index: int,
    *,
    detector_mask: int,
    logical_mask: int = 0,
    probability: float = 0.1,
) -> progressive.ProgressiveColumn:
    return progressive.ProgressiveColumn(
        family="test",
        index=int(index),
        label=f"c{int(index)}",
        instruction_offset=int(index),
        prior_probs=(1.0 - float(probability), float(probability)),
        detector_response_masks=(0, int(detector_mask)),
        logical_response_masks=(0, int(logical_mask)),
        detector_support_mask=int(detector_mask),
        detector_support_rows=tuple(overlap_profile._iter_set_bits(detector_mask)),
        original_column_index=int(index),
    )


def _toy_family() -> SimpleNamespace:
    columns = (
        _column(0, detector_mask=0b011, logical_mask=0, probability=0.08),
        _column(1, detector_mask=0b011, logical_mask=1, probability=0.13),
        _column(2, detector_mask=0b001, logical_mask=0, probability=0.19),
        _column(3, detector_mask=0b010, logical_mask=0, probability=0.23),
        _column(4, detector_mask=0b100, logical_mask=0, probability=0.29),
    )
    layout = progressive.build_frontier_layout(list(columns), num_detectors=3)
    return SimpleNamespace(columns=columns, layout=layout, logical_rows=1)


def _brute_future_loads(family: SimpleNamespace) -> tuple[int, ...]:
    result: list[int] = []
    for cut, active_mask in enumerate(family.layout.active_masks_after_column):
        result.append(
            max(
                (
                    int(column.detector_support_mask & int(active_mask)).bit_count()
                    for column in family.columns[cut + 1 :]
                ),
                default=0,
            )
        )
    return tuple(result)


def _assignment_probability(value: int, probabilities: tuple[float, ...]) -> float:
    return float(
        math.prod(
            probability if ((int(value) >> index) & 1) else 1.0 - probability
            for index, probability in enumerate(probabilities)
        )
    )


def test_chernoff_activity_and_safe_order() -> None:
    p = 0.17
    observed = overlap_profile.chernoff_activity(p, 0.5)
    assert observed == pytest.approx(2.0 * math.sqrt(p * (1.0 - p)))
    assert overlap_profile.safe_chernoff_order(score_alpha=0.8, load=2) == 0.5
    assert overlap_profile.safe_chernoff_order(score_alpha=0.8, load=4) == pytest.approx(
        0.3125
    )
    assert overlap_profile.safe_chernoff_order(score_alpha=0.8, load=6) == pytest.approx(
        5.0 / 24.0
    )


@pytest.mark.parametrize("ratio", [0.2, 1.0, 3.0])
@pytest.mark.parametrize("theta", [0.2, 0.3125, 0.5])
def test_tilted_overlap_obeys_chernoff_majorant(ratio: float, theta: float) -> None:
    probabilities = (0.08, 0.17, 0.33)
    full_mask = (1 << len(probabilities)) - 1
    overlap = sum(
        min(
            _assignment_probability(value, probabilities),
            float(ratio)
            * _assignment_probability(int(value) ^ full_mask, probabilities),
        )
        for value in range(1 << len(probabilities))
    )
    bound = float(ratio) ** float(theta) * math.prod(
        overlap_profile.chernoff_activity(probability, theta)
        for probability in probabilities
    )
    assert overlap <= bound + 2e-15


def test_weight_four_row_score_has_nonamplifying_chernoff_moment() -> None:
    # Future variables y0,y1,y2 produce all one-, two-, and three-variable
    # parities. Each future variable touches exactly four active score rows.
    probabilities = (0.19, 0.27, 0.34)
    row_masks = (0b001, 0b010, 0b100, 0b011, 0b101, 0b110, 0b111)
    alpha = 0.8
    theta = 0.3125
    q_mass: defaultdict[int, float] = defaultdict(float)
    for value in range(1 << len(probabilities)):
        q = sum(
            ((int(mask) & int(value)).bit_count() & 1) << row
            for row, mask in enumerate(row_masks)
        )
        q_mass[q] += _assignment_probability(value, probabilities)

    row_marginals: list[tuple[float, float]] = []
    for row in range(len(row_masks)):
        probability_one = sum(
            probability
            for q, probability in q_mass.items()
            if ((int(q) >> row) & 1)
        )
        row_marginals.append((1.0 - probability_one, probability_one))

    def score(q: int) -> float:
        return float(
            math.prod(
                row_marginals[row][(int(q) >> row) & 1] ** alpha
                for row in range(len(row_masks))
            )
        )

    for delta in range(1 << len(row_masks)):
        moment = sum(
            probability * (score(q ^ delta) / score(q)) ** theta
            for q, probability in q_mass.items()
        )
        assert moment <= 1.0 + 3e-15


def test_incremental_future_load_matches_bruteforce() -> None:
    family = _toy_family()
    assert overlap_profile.future_active_load_profile(family) == _brute_future_loads(family)


def test_xor_subset_partition_matches_bruteforce() -> None:
    shift_activities = ((0b01, 0.2), (0b10, 0.3), (0b11, 0.4))
    observed = overlap_profile.xor_subset_partition(
        boundary_bits=2,
        shift_activities=shift_activities,
    )
    expected = [0.0, 0.0, 0.0, 0.0]
    for subset in range(1 << len(shift_activities)):
        shift = 0
        weight = 1.0
        for index, (item_shift, activity) in enumerate(shift_activities):
            if (subset >> index) & 1:
                shift ^= int(item_shift)
                weight *= float(activity)
        expected[shift] += weight
    assert observed.tolist() == pytest.approx(expected)


def test_boundary_shift_aggregation_preserves_expected_inequality_order() -> None:
    family = _toy_family()
    polymers = overlap_profile.enumerate_low_weight_polymers(family, theta=0.5)
    assert sum(polymer.size == 1 for polymer in polymers) == 2
    assert sum(polymer.size == 2 for polymer in polymers) == 3

    profile = overlap_profile.boundary_shift_aggregation_profile(
        family,
        theta=0.5,
        rhos=(0.5, 0.75),
        K_values=(16,),
        max_boundary_bits=4,
        progress_every_cuts=0,
    )
    assert profile["polymer_universe"] == {"size_1": 2, "size_2": 3}
    for row in profile["per_cut"]:
        for values in row["rho"].values():
            assert (
                values["shift_grouped_moment"]
                <= values["exact_ungrouped_product_moment"] + 2e-15
            )
            assert (
                values["exact_ungrouped_product_moment"]
                <= values["exponential_product_moment"] + 2e-15
            )


def test_low_weight_intervals_match_direct_component_reasoning() -> None:
    family = _toy_family()
    profile = overlap_profile.low_weight_polymer_profile(
        family,
        theta=0.5,
        rhos=(0.5, 0.75),
    )

    assert profile["candidate_connected_pairs"] == 5
    assert profile["unique_visible_polymers"] == {"size_1": 2, "size_2": 3}
    counts = profile["visible_polymer_cut_counts"]
    assert counts["size_1"]["total"] == pytest.approx(3.0)
    assert counts["size_1"]["peak"] == pytest.approx(2.0)
    assert counts["size_2"]["total"] == pytest.approx(5.0)
    assert counts["size_2"]["peak"] == pytest.approx(3.0)
    assert profile["lifetime_histogram_cuts"]["size_1"] == {"1": 1, "2": 1}
    assert profile["lifetime_histogram_cuts"]["size_2"] == {"1": 2, "3": 1}
    partial_rhs = profile["lower_bound_on_product_peierls_rhs_from_sizes_1_2"]
    assert partial_rhs["0.5"]["16"]["value"] > 0.0
    assert profile["xi_partial_rho_1_diagnostic"]["sizes_1_2_lower_bound"]["peak"] > 0.0


def test_rotated_surface_cli_writes_reproducible_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    output = tmp_path / "profile.json"
    assert (
        overlap_profile.main(
            [
                "--backend",
                "rotated_surface_d3",
                "--p-location",
                "0.001",
                "--scopes",
                "memory_X",
                "--score-alpha",
                "0.8",
                "--rhos",
                "0.5,0.75",
                "--out",
                str(output),
                "--progress-every-pairs",
                "0",
                "--aggregate-boundary-shifts",
                "--progress-every-cuts",
                "0",
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["configuration"]["backend"] == "rotated_surface_d3"
    assert payload["configuration"]["K_values"] == [16, 512, 1024, 8192]
    assert len(payload["profiles"]) == 1
    profile = payload["profiles"][0]
    assert profile["matrix"]["detector_rows"] == 24
    assert profile["matrix"]["columns"] == 221
    assert profile["detector_column_weight"]["maximum"] == 4
    assert profile["future_active_score_load"]["maximum"] <= 4
    assert profile["low_weight_open_polymers"]["unique_visible_polymers"]["size_1"] > 0
    aggregation = profile["low_weight_open_polymers"]["boundary_shift_aggregation"]
    assert aggregation["max_boundary_bits"] <= 12
    grouped = aggregation["cap_rhs_from_sizes_1_2"]["0.75"]["shift_grouped"]["512"]
    ungrouped = aggregation["cap_rhs_from_sizes_1_2"]["0.75"][
        "exact_ungrouped_product"
    ]["512"]
    assert grouped["value"] <= ungrouped["value"]
    optimized = aggregation["rho_optimization"]["minimum_cap_rhs_by_K"][
        "shift_grouped"
    ]["1024"]
    assert 0.0 < optimized["rho"] < 1.0
    assert optimized["value"] > 1.0
    critical = aggregation["rho_optimization"][
        "critical_K_for_partial_rhs_below_one"
    ]["shift_grouped"]
    assert critical["first_integer_strictly_above"] > 1024
