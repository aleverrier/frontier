from __future__ import annotations

from pathlib import Path

from grosscode.circuits.backends import ResolvedBackendCircuit, ScheduleResolutionError


def resolve_depth8_candidate_circuit(
    *,
    sector: str,
    error_rate: float = 0.004,
    syndrome_rounds: int = 12,
    qtanner_root: str | Path | None = None,
) -> ResolvedBackendCircuit:
    raise ScheduleResolutionError(
        "backend='depth8_candidate' remains a documented placeholder. "
        "The public schedule is not reconstructible in this repo without guessing, so this entry point exists "
        "only to fail loudly until a public schedule source is available."
    )


__all__ = ["ResolvedBackendCircuit", "ScheduleResolutionError", "resolve_depth8_candidate_circuit"]
