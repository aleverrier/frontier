from __future__ import annotations

from pathlib import Path

from grosscode.circuits.backends import ResolvedBackendCircuit, resolve_backend_circuit


def resolve_bravyi_depth7_circuit(
    *,
    sector: str,
    error_rate: float = 0.004,
    syndrome_rounds: int = 12,
    qtanner_root: str | Path | None = None,
) -> ResolvedBackendCircuit:
    return resolve_backend_circuit(
        backend="bravyi_depth7",
        sector=sector,
        error_rate=error_rate,
        syndrome_rounds=syndrome_rounds,
        qtanner_root=qtanner_root,
    )


__all__ = ["ResolvedBackendCircuit", "resolve_bravyi_depth7_circuit"]
