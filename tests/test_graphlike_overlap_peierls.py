"""Finite exhaustive checks for docs/GRAPHLIKE_OVERLAP_PEIERLS.md."""

from __future__ import annotations

import math
from collections import defaultdict

import pytest


def _parity(mask: int, value: int) -> int:
    return (int(mask) & int(value)).bit_count() & 1


def _assignment_probability(value: int, probabilities: tuple[float, ...]) -> float:
    result = 1.0
    for index, probability in enumerate(probabilities):
        result *= probability if ((int(value) >> index) & 1) else 1.0 - probability
    return float(result)


def _pushforward_mass(
    probabilities: tuple[float, ...],
    *,
    completed_masks: tuple[int, ...],
    boundary_masks: tuple[int, ...],
) -> dict[tuple[tuple[int, ...], tuple[int, ...]], float]:
    mass: defaultdict[tuple[tuple[int, ...], tuple[int, ...]], float] = defaultdict(
        float
    )
    for value in range(1 << len(probabilities)):
        completed = tuple(_parity(mask, value) for mask in completed_masks)
        boundary = tuple(_parity(mask, value) for mask in boundary_masks)
        mass[(completed, boundary)] += _assignment_probability(value, probabilities)
    return dict(mass)


def _quotient_overlap(
    mass: dict[tuple[tuple[int, ...], tuple[int, ...]], float],
    *,
    boundary_shift: tuple[int, ...],
    ratio: float,
) -> float:
    total = 0.0
    for (completed, boundary), left in mass.items():
        shifted = tuple(
            int(bit) ^ int(delta)
            for bit, delta in zip(boundary, boundary_shift, strict=True)
        )
        right = float(mass.get((completed, shifted), 0.0))
        total += min(float(left), float(ratio) * right)
    return float(total)


def _local_overlap(
    probabilities: tuple[float, ...],
    *,
    support: tuple[int, ...],
    ratio: float,
) -> float:
    local_probabilities = tuple(probabilities[index] for index in support)
    full_mask = (1 << len(local_probabilities)) - 1
    return float(
        sum(
            min(
                _assignment_probability(value, local_probabilities),
                float(ratio)
                * _assignment_probability(value ^ full_mask, local_probabilities),
            )
            for value in range(1 << len(local_probabilities))
        )
    )


def _hellinger_weight(
    probabilities: tuple[float, ...], support: tuple[int, ...]
) -> float:
    return float(
        math.prod(
            2.0 * math.sqrt(probabilities[index] * (1.0 - probabilities[index]))
            for index in support
        )
    )


def test_exact_overlap_does_not_multiply_over_components() -> None:
    p = 0.1
    singleton = _local_overlap((p,), support=(0,), ratio=1.0)
    pair = _local_overlap((p, p), support=(0, 1), ratio=1.0)
    beta = 2.0 * math.sqrt(p * (1.0 - p))

    assert singleton == pytest.approx(2.0 * p)
    assert pair == pytest.approx(2.0 * p)
    assert pair > singleton**2
    assert pair <= beta**2


@pytest.mark.parametrize("ratio", [0.37, 1.0, 2.4])
def test_quotient_lift_handles_alternative_and_invisible_components(
    ratio: float,
) -> None:
    # Completed parity x0+x1 connects {0,1}.  Boundary x0+x2 can be shifted
    # either by the visible component {0,1} or by the visible singleton {2}.
    # Coordinate x3 is completely invisible and is summed out by the quotient.
    probabilities = (0.13, 0.29, 0.37, 0.41)
    mass = _pushforward_mass(
        probabilities,
        completed_masks=(0b0011,),
        boundary_masks=(0b0101,),
    )
    observed = _quotient_overlap(mass, boundary_shift=(1,), ratio=ratio)
    family_supports = ((0, 1), (2,))
    exact_family_bound = sum(
        _local_overlap(probabilities, support=support, ratio=ratio)
        for support in family_supports
    )
    hellinger_family_bound = math.sqrt(ratio) * sum(
        _hellinger_weight(probabilities, support) for support in family_supports
    )

    assert observed <= exact_family_bound + 2e-15
    assert exact_family_bound <= hellinger_family_bound + 2e-15


def test_graphlike_row_score_has_nonamplifying_half_moment() -> None:
    # Future variables y0,y1 produce q=(y0,y0+y1,y1).  Each future column
    # touches exactly two rows, so alpha<=1 satisfies the Finner half-load.
    probabilities = (0.19, 0.31)
    row_masks = (0b01, 0b11, 0b10)
    alpha = 0.8
    q_mass: defaultdict[int, float] = defaultdict(float)
    for value in range(4):
        q = sum(_parity(mask, value) << index for index, mask in enumerate(row_masks))
        q_mass[q] += _assignment_probability(value, probabilities)

    row_marginals: list[tuple[float, float]] = []
    for row in range(3):
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
                for row in range(3)
            )
        )

    for delta in range(8):
        half_moment = sum(
            probability * math.sqrt(score(q ^ delta) / score(q))
            for q, probability in q_mass.items()
        )
        assert half_moment <= 1.0 + 2e-15


def test_fractional_exact_family_and_product_bounds() -> None:
    probabilities = (0.13, 0.29, 0.37, 0.41)
    mass = _pushforward_mass(
        probabilities,
        completed_masks=(0b0011,),
        boundary_masks=(0b0101,),
    )
    omega = _quotient_overlap(mass, boundary_shift=(1,), ratio=1.0)
    family_supports = ((0, 1), (2,))
    exact_activities = [
        _local_overlap(probabilities, support=support, ratio=1.0)
        for support in family_supports
    ]
    product_activities = [
        _hellinger_weight(probabilities, support) for support in family_supports
    ]

    for rho in (0.25, 0.5, 0.8):
        exact_fractional_bound = sum(activity**rho for activity in exact_activities)
        product_peierls_bound = (
            math.prod(1.0 + activity**rho for activity in product_activities) - 1.0
        )
        assert omega**rho <= exact_fractional_bound + 2e-15
        assert exact_fractional_bound <= product_peierls_bound + 2e-15
