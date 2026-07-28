"""Finite exhaustive checks for docs/COMPARISON_SPECTRUM_THEOREM.md."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict

import pytest


def _parity(mask: int, value: int) -> int:
    return (int(mask) & int(value)).bit_count() & 1


def _xor_boundary(lhs: tuple[int, int], rhs: tuple[int, int]) -> tuple[int, int]:
    return int(lhs[0]) ^ int(rhs[0]), int(lhs[1]) ^ int(rhs[1])


def _tiny_cut() -> dict[str, object]:
    # Prefix variables are bits 0,1,2 and suffix variables are bits 3,4.
    # One completed row, three active rows, and one unopened row are present.
    prefix_masks = {
        "completed": (0b011,),
        "active": (0b001, 0b010, 0b100),
        "logical": 0b100,
    }
    suffix_masks = {
        "active": (0b01, 0b10, 0b11),
        "unopened": (0b11,),
    }
    prefix_probabilities = (0.13, 0.29, 0.41)
    suffix_probabilities = (0.17, 0.37)
    score = {
        0b000: 1.00,
        0b001: 0.61,
        0b010: 1.73,
        0b011: 0.83,
        0b100: 1.29,
        0b101: 0.47,
        0b110: 2.11,
        0b111: 0.72,
    }
    return {
        "prefix_masks": prefix_masks,
        "suffix_masks": suffix_masks,
        "prefix_probabilities": prefix_probabilities,
        "suffix_probabilities": suffix_probabilities,
        "score": score,
    }


def _assignment_probability(value: int, probabilities: tuple[float, ...]) -> float:
    result = 1.0
    for index, probability in enumerate(probabilities):
        result *= probability if ((int(value) >> index) & 1) else (1.0 - probability)
    return float(result)


def _prefix_tables(
    model: dict[str, object],
) -> tuple[
    dict[tuple[int, tuple[int, int]], float],
    tuple[tuple[int, int], ...],
]:
    prefix_masks = model["prefix_masks"]
    probabilities = model["prefix_probabilities"]
    assert isinstance(prefix_masks, dict)
    assert isinstance(probabilities, tuple)
    completed_mask = int(prefix_masks["completed"][0])
    active_masks = tuple(int(mask) for mask in prefix_masks["active"])
    logical_mask = int(prefix_masks["logical"])
    masses: defaultdict[tuple[int, tuple[int, int]], float] = defaultdict(float)
    boundary_by_prefix: dict[int, tuple[int, int]] = {}
    completed_by_prefix: dict[int, int] = {}

    for prefix in range(1 << len(probabilities)):
        completed = _parity(completed_mask, prefix)
        active = sum(_parity(mask, prefix) << index for index, mask in enumerate(active_masks))
        logical = _parity(logical_mask, prefix)
        boundary = (int(active), int(logical))
        boundary_by_prefix[prefix] = boundary
        completed_by_prefix[prefix] = completed
        masses[(completed, boundary)] += _assignment_probability(prefix, probabilities)

    shifts = {
        _xor_boundary(boundary_by_prefix[left], boundary_by_prefix[right])
        for left in boundary_by_prefix
        for right in boundary_by_prefix
        if completed_by_prefix[left] == completed_by_prefix[right]
    }
    return dict(masses), tuple(sorted(shifts))


def _suffix_table(
    model: dict[str, object],
) -> tuple[tuple[int, int, float], ...]:
    suffix_masks = model["suffix_masks"]
    probabilities = model["suffix_probabilities"]
    assert isinstance(suffix_masks, dict)
    assert isinstance(probabilities, tuple)
    active_masks = tuple(int(mask) for mask in suffix_masks["active"])
    unopened_mask = int(suffix_masks["unopened"][0])
    rows: list[tuple[int, int, float]] = []
    for suffix in range(1 << len(probabilities)):
        active = sum(_parity(mask, suffix) << index for index, mask in enumerate(active_masks))
        unopened = _parity(unopened_mask, suffix)
        rows.append((active, unopened, _assignment_probability(suffix, probabilities)))
    return tuple(rows)


def _suffix_completion_mass(
    suffix_rows: tuple[tuple[int, int, float], ...],
    *,
    syndrome_active: int,
    syndrome_unopened: int,
    boundary_active: int,
) -> float:
    residual = int(syndrome_active) ^ int(boundary_active)
    return float(
        sum(
            probability
            for active, unopened, probability in suffix_rows
            if int(active) == residual and int(unopened) == int(syndrome_unopened)
        )
    )


def _omega(
    *,
    prefix_mass: dict[tuple[int, tuple[int, int]], float],
    suffix_rows: tuple[tuple[int, int, float], ...],
    score: dict[int, float],
    shifts: tuple[tuple[int, int], ...],
) -> dict[tuple[int, int], float]:
    result: dict[tuple[int, int], float] = {}
    boundaries = {boundary for _, boundary in prefix_mass}
    for shift in shifts:
        if shift == (0, 0):
            continue
        delta = int(shift[0])
        total = 0.0
        for completed in (0, 1):
            for boundary in boundaries:
                shifted = _xor_boundary(boundary, shift)
                left = float(prefix_mass.get((completed, boundary), 0.0))
                right = float(prefix_mass.get((completed, shifted), 0.0))
                for suffix_active, _suffix_unopened, probability in suffix_rows:
                    ratio = float(score[int(suffix_active) ^ delta]) / float(
                        score[int(suffix_active)]
                    )
                    total += float(probability) * min(left, right * ratio)
        result[shift] = float(total)
    return result


def _syndrome_envelope(
    *,
    prefix_mass: dict[tuple[int, tuple[int, int]], float],
    suffix_rows: tuple[tuple[int, int, float], ...],
    score: dict[int, float],
    shift: tuple[int, int],
) -> float:
    boundaries = {boundary for _, boundary in prefix_mass}
    total = 0.0
    for completed, syndrome_active, syndrome_unopened in itertools.product(
        (0, 1), range(8), (0, 1)
    ):
        for boundary in boundaries:
            boundary_active = int(boundary[0])
            z = _suffix_completion_mass(
                suffix_rows,
                syndrome_active=syndrome_active,
                syndrome_unopened=syndrome_unopened,
                boundary_active=boundary_active,
            )
            shifted = _xor_boundary(boundary, shift)
            residual = int(syndrome_active) ^ boundary_active
            ratio = float(score[residual ^ int(shift[0])]) / float(score[residual])
            total += z * min(
                float(prefix_mass.get((completed, boundary), 0.0)),
                float(prefix_mass.get((completed, shifted), 0.0)) * ratio,
            )
    return float(total)


def _attenuation(
    *,
    completed: int,
    syndrome_active: int,
    syndrome_unopened: int,
    boundary: tuple[int, int],
) -> float:
    token = (
        17 * int(completed)
        + 11 * int(syndrome_active)
        + 7 * int(syndrome_unopened)
        + 5 * int(boundary[0])
        + 3 * int(boundary[1])
    )
    return 0.22 + 0.13 * float(token % 6)


def _actual_losses(
    *,
    prefix_mass: dict[tuple[int, tuple[int, int]], float],
    suffix_rows: tuple[tuple[int, int, float], ...],
    score: dict[int, float],
    K: int,
    Delta: float,
) -> tuple[float, float]:
    boundaries = sorted({boundary for _, boundary in prefix_mass})
    cap_loss = 0.0
    gap_loss = 0.0
    for completed, syndrome_active, syndrome_unopened in itertools.product(
        (0, 1), range(8), (0, 1)
    ):
        surviving: dict[tuple[int, int], float] = {}
        exact_suffix: dict[tuple[int, int], float] = {}
        rank_rows: list[tuple[float, float, tuple[int, int]]] = []
        for boundary in boundaries:
            unpruned = float(prefix_mass.get((completed, boundary), 0.0))
            if unpruned == 0.0:
                continue
            mass = unpruned * _attenuation(
                completed=completed,
                syndrome_active=syndrome_active,
                syndrome_unopened=syndrome_unopened,
                boundary=boundary,
            )
            residual = int(syndrome_active) ^ int(boundary[0])
            h = mass * float(score[residual])
            surviving[boundary] = mass
            exact_suffix[boundary] = _suffix_completion_mass(
                suffix_rows,
                syndrome_active=syndrome_active,
                syndrome_unopened=syndrome_unopened,
                boundary_active=int(boundary[0]),
            )
            rank_rows.append((h, mass, boundary))

        best = max(h for h, _mass, _boundary in rank_rows)
        gap_survivors: list[tuple[float, float, tuple[int, int]]] = []
        for h, mass, boundary in rank_rows:
            exact_mass = mass * exact_suffix[boundary]
            if h < math.exp(-float(Delta)) * best:
                gap_loss += exact_mass
            else:
                gap_survivors.append((h, mass, boundary))
        gap_survivors.sort(reverse=True)
        for _h, mass, boundary in gap_survivors[int(K) :]:
            cap_loss += mass * exact_suffix[boundary]
    return float(cap_loss), float(gap_loss)


def _gap_spectrum(
    *,
    prefix_mass: dict[tuple[int, tuple[int, int]], float],
    suffix_rows: tuple[tuple[int, int, float], ...],
    score: dict[int, float],
    shifts: tuple[tuple[int, int], ...],
    Delta: float,
) -> dict[tuple[int, int], float]:
    boundaries = {boundary for _, boundary in prefix_mass}
    result: dict[tuple[int, int], float] = {}
    for shift in shifts:
        if shift == (0, 0):
            continue
        delta = int(shift[0])
        total = 0.0
        for completed in (0, 1):
            for boundary in boundaries:
                shifted = _xor_boundary(boundary, shift)
                left = float(prefix_mass.get((completed, boundary), 0.0))
                right = float(prefix_mass.get((completed, shifted), 0.0))
                for suffix_active, _suffix_unopened, probability in suffix_rows:
                    ratio = float(score[int(suffix_active) ^ delta]) / float(
                        score[int(suffix_active)]
                    )
                    total += float(probability) * min(
                        left,
                        math.exp(-float(Delta)) * right * ratio,
                    )
        result[shift] = float(total)
    return result


def test_exact_syndrome_fiber_identity() -> None:
    model = _tiny_cut()
    prefix_mass, shifts = _prefix_tables(model)
    suffix_rows = _suffix_table(model)
    score = model["score"]
    assert isinstance(score, dict)
    spectrum = _omega(
        prefix_mass=prefix_mass,
        suffix_rows=suffix_rows,
        score=score,
        shifts=shifts,
    )
    for shift, expected in spectrum.items():
        observed = _syndrome_envelope(
            prefix_mass=prefix_mass,
            suffix_rows=suffix_rows,
            score=score,
            shift=shift,
        )
        assert observed == pytest.approx(expected, rel=2e-14, abs=2e-14)


@pytest.mark.parametrize(("K", "Delta"), [(1, 100.0), (2, 100.0), (2, 0.9)])
def test_recursive_cap_and_gap_bounds(K: int, Delta: float) -> None:
    model = _tiny_cut()
    prefix_mass, shifts = _prefix_tables(model)
    suffix_rows = _suffix_table(model)
    score = model["score"]
    assert isinstance(score, dict)
    spectrum = _omega(
        prefix_mass=prefix_mass,
        suffix_rows=suffix_rows,
        score=score,
        shifts=shifts,
    )
    cap_loss, gap_loss = _actual_losses(
        prefix_mass=prefix_mass,
        suffix_rows=suffix_rows,
        score=score,
        K=K,
        Delta=Delta,
    )

    ordered = sorted(spectrum.values(), reverse=True)
    trimmed = min(sum(ordered[q:]) / float(K - q) for q in range(K))
    assert cap_loss <= trimmed + 2e-14

    for rho in (0.25, 0.5, 0.8):
        quasi_norm = sum(value**rho for value in ordered) ** (1.0 / rho)
        fractional_bound = quasi_norm * float(K) ** (-1.0 / rho)
        assert trimmed <= fractional_bound + 2e-14

    gap_spectrum = _gap_spectrum(
        prefix_mass=prefix_mass,
        suffix_rows=suffix_rows,
        score=score,
        shifts=shifts,
        Delta=Delta,
    )
    assert gap_loss <= sum(gap_spectrum.values()) + 2e-14
