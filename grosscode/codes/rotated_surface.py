from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import stim

from grosscode.codes.css import CSSCode, derive_css_code_from_checks


_BACKEND_RE = re.compile(
    r"^(?P<family>rotated_surface|surface)_d(?P<distance>\d+)(?:_r(?P<rounds>\d+))?$"
)


@dataclass(frozen=True)
class RotatedSurfaceBackendSpec:
    backend: str
    family: str
    distance: int
    syndrome_rounds: int


def is_rotated_surface_backend(backend: str) -> bool:
    return _BACKEND_RE.match(str(backend)) is not None


def get_rotated_surface_backend_spec(backend: str) -> RotatedSurfaceBackendSpec:
    text = str(backend)
    match = _BACKEND_RE.match(text)
    if match is None:
        raise ValueError(f"unsupported rotated-surface backend: {backend}")
    distance = int(match.group("distance"))
    rounds_text = match.group("rounds")
    rounds = int(rounds_text) if rounds_text is not None else int(distance)
    if distance < 3 or distance % 2 == 0:
        raise ValueError("rotated surface backend requires odd distance >= 3")
    if rounds <= 0:
        raise ValueError("rotated surface backend requires rounds > 0")
    return RotatedSurfaceBackendSpec(
        backend=text,
        family=str(match.group("family")),
        distance=int(distance),
        syndrome_rounds=int(rounds),
    )


def _set_pcm_row(n: int, pcm: np.ndarray, row_idx: int, i: int, j: int) -> None:
    i1, j1 = (i + 1) % n, (j + 1) % n
    pcm[row_idx][i * n + j] = 1
    pcm[row_idx][i1 * n + j1] = 1
    pcm[row_idx][i1 * n + j] = 1
    pcm[row_idx][i * n + j1] = 1


def _rotated_surface_checks(distance: int) -> tuple[np.ndarray, np.ndarray]:
    n2 = int(distance) * int(distance)
    m = (n2 - 1) // 2
    hx = np.zeros((m, n2), dtype=np.uint8)
    hz = np.zeros((m, n2), dtype=np.uint8)
    x_idx = 0
    z_idx = 0

    for i in range(int(distance) - 1):
        for j in range(int(distance) - 1):
            if (i + j) % 2 == 0:
                _set_pcm_row(int(distance), hz, z_idx, i, j)
                z_idx += 1
            else:
                _set_pcm_row(int(distance), hx, x_idx, i, j)
                x_idx += 1

    for j in range(int(distance) - 1):
        if j % 2 == 0:
            hx[x_idx][j] = 1
            hx[x_idx][j + 1] = 1
        else:
            hx[x_idx][(int(distance) - 1) * int(distance) + j] = 1
            hx[x_idx][(int(distance) - 1) * int(distance) + (j + 1)] = 1
        x_idx += 1

    for i in range(int(distance) - 1):
        if i % 2 == 0:
            hz[z_idx][i * int(distance) + (int(distance) - 1)] = 1
            hz[z_idx][(i + 1) * int(distance) + (int(distance) - 1)] = 1
        else:
            hz[z_idx][i * int(distance)] = 1
            hz[z_idx][(i + 1) * int(distance)] = 1
        z_idx += 1

    return hx, hz


@lru_cache(maxsize=16)
def _load_rotated_surface_code_cached(distance: int) -> CSSCode:
    hx, hz = _rotated_surface_checks(int(distance))
    return derive_css_code_from_checks(
        name=f"rotated surface [[{int(distance) * int(distance)},{1},{int(distance)}]]",
        hx=sp.csr_matrix(hx, dtype=np.uint8),
        hz=sp.csr_matrix(hz, dtype=np.uint8),
        metadata={
            "source": "local rotated-surface constructor matching Stim rotated_memory circuits",
            "distance_hint": int(distance),
        },
    )


def load_rotated_surface_code(*, backend: str) -> CSSCode:
    spec = get_rotated_surface_backend_spec(str(backend))
    return _load_rotated_surface_code_cached(int(spec.distance))


def build_rotated_surface_circuit_text(
    *,
    backend: str,
    sector: str,
    error_rate: float,
    syndrome_rounds: int | None = None,
) -> str:
    spec = get_rotated_surface_backend_spec(str(backend))
    rounds = int(spec.syndrome_rounds if syndrome_rounds is None else syndrome_rounds)
    sector_norm = str(sector).upper()
    if sector_norm not in {"X", "Z"}:
        raise ValueError("sector must be 'X' or 'Z'")
    task = "surface_code:rotated_memory_x" if sector_norm == "X" else "surface_code:rotated_memory_z"
    circuit = stim.Circuit.generated(
        task,
        distance=int(spec.distance),
        rounds=int(rounds),
        after_clifford_depolarization=float(error_rate),
        before_round_data_depolarization=float(error_rate),
        before_measure_flip_probability=float(error_rate),
        after_reset_flip_probability=float(error_rate),
    )
    return str(circuit)


def generated_rotated_surface_stim_path(
    *,
    backend: str,
    sector: str,
    error_rate: float,
    syndrome_rounds: int | None = None,
) -> Path:
    spec = get_rotated_surface_backend_spec(str(backend))
    rounds = int(spec.syndrome_rounds if syndrome_rounds is None else syndrome_rounds)
    cache_root = Path(tempfile.gettempdir()) / "better_beam_backend_stim"
    cache_root.mkdir(parents=True, exist_ok=True)
    rate_tag = str(float(error_rate)).replace("-", "m").replace(".", "p")
    name = f"{spec.backend},memory_{str(sector).upper()},error_rate={rate_tag},syndrome_rounds={int(rounds)}.stim"
    return cache_root / name


def ensure_rotated_surface_stim_file(
    *,
    backend: str,
    sector: str,
    error_rate: float,
    syndrome_rounds: int | None = None,
) -> Path:
    path = generated_rotated_surface_stim_path(
        backend=str(backend),
        sector=str(sector),
        error_rate=float(error_rate),
        syndrome_rounds=syndrome_rounds,
    )
    if path.exists():
        return path
    text = build_rotated_surface_circuit_text(
        backend=str(backend),
        sector=str(sector),
        error_rate=float(error_rate),
        syndrome_rounds=syndrome_rounds,
    )
    path.write_text(text, encoding="utf-8")
    return path


__all__ = [
    "RotatedSurfaceBackendSpec",
    "build_rotated_surface_circuit_text",
    "ensure_rotated_surface_stim_file",
    "generated_rotated_surface_stim_path",
    "get_rotated_surface_backend_spec",
    "is_rotated_surface_backend",
    "load_rotated_surface_code",
]
