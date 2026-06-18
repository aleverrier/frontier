"""Public frontier model-construction primitives.

This module provides stable names for light users and examples while the
implementation remains in `tools.frontier_progressive`.
"""

from __future__ import annotations

from typing import Sequence

from tools import frontier_progressive as _progressive

ProgressiveColumn = _progressive.ProgressiveColumn
ProgressiveFrontierLayout = _progressive.ProgressiveFrontierLayout
FactorTransition = _progressive.FactorTransition
OutcomeTransition = _progressive.OutcomeTransition
build_frontier_layout = _progressive.build_frontier_layout
optimize_column_order = _progressive.optimize_column_order


def columns_from_factor_transitions(
    factor_transitions: Sequence[FactorTransition],
) -> list[ProgressiveColumn]:
    """Build frontier columns from factor transitions."""

    return _progressive._columns_from_factor_transitions(factor_transitions)


__all__ = [
    "ProgressiveColumn",
    "ProgressiveFrontierLayout",
    "FactorTransition",
    "OutcomeTransition",
    "build_frontier_layout",
    "optimize_column_order",
    "columns_from_factor_transitions",
]
