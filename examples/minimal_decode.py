#!/usr/bin/env python3
"""Minimal two-factor frontier decode example."""

from __future__ import annotations

from frontier import FrontierModel, decode_frontier, decode_frontier_committee
from frontier.progressive import (
    FactorTransition,
    OutcomeTransition,
    build_frontier_layout,
    columns_from_factor_transitions,
)


def _factor(factor_id: int, p0: float, det1: int, log1: int) -> FactorTransition:
    return FactorTransition(
        factor_id=int(factor_id),
        outcomes=(
            OutcomeTransition(probability=float(p0), detector_mask=0, logical_mask=0),
            OutcomeTransition(
                probability=float(1.0 - p0),
                detector_mask=int(det1),
                logical_mask=int(log1),
            ),
        ),
        instruction_offset=int(factor_id),
        label=f"f{int(factor_id)}",
    )


def build_model() -> FrontierModel:
    columns = tuple(
        columns_from_factor_transitions(
            (
                _factor(0, 0.55, 1, 1),
                _factor(1, 0.70, 1, 0),
            )
        )
    )
    return FrontierModel(
        columns=columns,
        layout=build_frontier_layout(list(columns), num_detectors=1),
        num_detectors=1,
        num_observables=1,
    )


def main() -> int:
    model = build_model()
    single = decode_frontier(model, 0, K=16, Delta=100.0)
    committee = decode_frontier_committee(model, 1, K=16, Delta=100.0)
    print(f"single status={single.status} logical_hat={single.logical_hat} engine={single.engine}")
    print(
        "committee "
        f"status={committee.status} logical_hat={committee.logical_hat} "
        f"direction={committee.direction} engine={committee.engine}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
