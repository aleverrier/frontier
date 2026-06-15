from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from grosscode.dem.builder import build_split_sector_problem, load_dem_side_with_metadata_from_stim
from grosscode.utils.gf2 import dense_mod2 as dense_mod2_legacy


@dataclass(frozen=True)
class DenseDemSide:
    matrix: np.ndarray
    observables: np.ndarray | None
    priors: np.ndarray
    metadata: Any
    matrix_label: str


def _normalize_sector(sector: str) -> str:
    text = str(sector).strip().upper().replace("MEMORY_", "")
    if text not in {"X", "Z"}:
        raise ValueError(f"sector must resolve to X or Z, got {sector!r}")
    return text


def load_split_sector_side(
    *,
    backend: str = "bravyi_depth7",
    error_rate: float = 0.004,
    sector: str = "X",
    qtanner_root: str | None = None,
) -> DenseDemSide:
    resolved = _normalize_sector(sector)
    problem = build_split_sector_problem(backend=backend, error_rate=error_rate, qtanner_root=qtanner_root)
    if resolved == "X":
        return DenseDemSide(
            matrix=dense_mod2_legacy(problem.D_X.toarray()),
            observables=dense_mod2_legacy(problem.O_X.toarray()),
            priors=np.asarray(problem.priors_X, dtype=np.float64).reshape(-1),
            metadata=problem.metadata_X,
            matrix_label=f"{backend}: D_X = {problem.D_X.shape[0]} x {problem.D_X.shape[1]}",
        )
    return DenseDemSide(
        matrix=dense_mod2_legacy(problem.D_Z.toarray()),
        observables=dense_mod2_legacy(problem.O_Z.toarray()),
        priors=np.asarray(problem.priors_Z, dtype=np.float64).reshape(-1),
        metadata=problem.metadata_Z,
        matrix_label=f"{backend}: D_Z = {problem.D_Z.shape[0]} x {problem.D_Z.shape[1]}",
    )


def load_dem_side_from_stim_path(
    *,
    stim_path: str | Path,
    backend: str,
    sector: str,
    error_rate: float,
    noisy_rounds: int,
    perfect_rounds: int = 1,
) -> DenseDemSide:
    side = load_dem_side_with_metadata_from_stim(
        stim_path=stim_path,
        backend=backend,
        sector=sector,
        error_rate=error_rate,
        noisy_rounds=noisy_rounds,
        perfect_rounds=perfect_rounds,
    )
    return DenseDemSide(
        matrix=dense_mod2_legacy(side.check_matrix.toarray()),
        observables=dense_mod2_legacy(side.observables_matrix.toarray()),
        priors=np.asarray(side.priors, dtype=np.float64).reshape(-1),
        metadata=side.metadata,
        matrix_label=f"{backend}: {Path(stim_path).name} ({side.check_matrix.shape[0]} x {side.check_matrix.shape[1]})",
    )
