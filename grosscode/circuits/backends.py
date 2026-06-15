from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from grosscode.codes.bivariate_bicycle import (
    BB_90_8_10_BACKEND,
    ensure_bivariate_bicycle_stim_file,
    get_bivariate_bicycle_backend_spec,
    is_bivariate_bicycle_backend,
)
from grosscode.codes.generalized_bicycle import (
    ensure_generalized_bicycle_stim_file,
    get_generalized_bicycle_backend_spec,
    is_generalized_bicycle_backend,
)
from grosscode.codes.rotated_surface import (
    ensure_rotated_surface_stim_file,
    get_rotated_surface_backend_spec,
    is_rotated_surface_backend,
)
from grosscode.utils.paths import resolve_qtanner_root


_PUBLIC_STIM_RE = re.compile(
    r"^BB\[\[144,12,12\]\],memory_(?P<sector>[XZ]),error_rate=(?P<rate>[^,]+),syndrome_rounds=(?P<rounds>\d+)\.stim$"
)


class ScheduleResolutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class AvailableCircuitRate:
    sector: str
    error_rate: float
    syndrome_rounds: int
    stim_path: Path


@dataclass(frozen=True)
class ResolvedBackendCircuit:
    backend: str
    sector: str
    error_rate: float
    syndrome_rounds: int
    stim_path: Path
    noisy_rounds: int
    perfect_rounds: int
    schedule_notes: tuple[str, ...]


def _stim_root(qtanner_root: str | Path | None = None) -> Path:
    root = resolve_qtanner_root(qtanner_root)
    stim_root = root / "third_party" / "BeamSearchDecoder" / "StimCircuit"
    if not stim_root.exists():
        raise FileNotFoundError(f"public Stim circuit directory missing: {stim_root}")
    return stim_root


def list_available_public_circuit_rates(qtanner_root: str | Path | None = None) -> list[AvailableCircuitRate]:
    out: list[AvailableCircuitRate] = []
    for path in sorted(_stim_root(qtanner_root).iterdir()):
        match = _PUBLIC_STIM_RE.match(path.name)
        if match is None:
            continue
        out.append(
            AvailableCircuitRate(
                sector=match.group("sector"),
                error_rate=float(match.group("rate")),
                syndrome_rounds=int(match.group("rounds")),
                stim_path=path,
            )
        )
    return out


def resolve_backend_circuit(
    *,
    backend: str,
    sector: str,
    error_rate: float = 0.004,
    initial_data_error_rate: float | None = None,
    syndrome_rounds: int = 12,
    qtanner_root: str | Path | None = None,
) -> ResolvedBackendCircuit:
    sector_norm = sector.upper()
    if sector_norm not in {"X", "Z"}:
        raise ValueError("sector must be 'X' or 'Z'")
    if initial_data_error_rate is not None and not is_bivariate_bicycle_backend(backend):
        raise ValueError("--initial-data-error-rate is only supported for bivariate-bicycle backends")
    if backend == "depth8_candidate":
        raise ScheduleResolutionError(
            "backend='depth8_candidate' is not publicly reconstructible in this repo yet. "
            "The interface exists, but schedule details remain TODO by policy."
        )
    if is_rotated_surface_backend(backend):
        spec = get_rotated_surface_backend_spec(backend)
        rounds = int(spec.syndrome_rounds if int(syndrome_rounds) == 12 else syndrome_rounds)
        stim_path = ensure_rotated_surface_stim_file(
            backend=str(backend),
            sector=sector_norm,
            error_rate=float(error_rate),
            syndrome_rounds=int(rounds),
        )
        return ResolvedBackendCircuit(
            backend=str(backend),
            sector=sector_norm,
            error_rate=float(error_rate),
            syndrome_rounds=int(rounds),
            stim_path=stim_path,
            noisy_rounds=int(rounds),
            perfect_rounds=0,
            schedule_notes=(
                f"Rotated surface-code memory circuit generated locally from Stim with distance {int(spec.distance)} and rounds {int(rounds)}.",
                "The detector-side DEM is built directly from the generated Stim circuit and uses the rotated_memory_x / rotated_memory_z task family.",
            ),
        )
    if is_bivariate_bicycle_backend(backend):
        spec = get_bivariate_bicycle_backend_spec(backend)
        rounds = int(spec.syndrome_rounds if int(syndrome_rounds) == 12 else syndrome_rounds)
        stim_path = ensure_bivariate_bicycle_stim_file(
            backend=str(backend),
            sector=sector_norm,
            error_rate=float(error_rate),
            initial_data_error_rate=initial_data_error_rate,
            syndrome_rounds=int(rounds),
        )
        pinit_note = (
            ""
            if initial_data_error_rate is None
            else (
                f" Initial data-qubit preparation errors are replacement-set to "
                f"p_init={float(initial_data_error_rate):.6g}; all other location errors use "
                f"p={float(error_rate):.6g}."
            )
        )
        return ResolvedBackendCircuit(
            backend=str(backend),
            sector=sector_norm,
            error_rate=float(error_rate),
            syndrome_rounds=int(rounds),
            stim_path=stim_path,
            noisy_rounds=int(rounds),
            perfect_rounds=1,
            schedule_notes=(
                f"Non-default BB [[{int(spec.n)},{int(spec.k)},{int(spec.distance)}]] circuit generated locally from the SlidingWindowDecoder constructor.{pinit_note}",
                "The detector-side DEM is derived from that locally generated Stim circuit, not from the accepted public Gross split-sector benchmark.",
            ),
        )
    if is_generalized_bicycle_backend(backend):
        spec = get_generalized_bicycle_backend_spec(backend)
        rounds = int(spec.syndrome_rounds if int(syndrome_rounds) == 12 else syndrome_rounds)
        stim_path = ensure_generalized_bicycle_stim_file(
            backend=str(backend),
            sector=sector_norm,
            error_rate=float(error_rate),
            syndrome_rounds=int(rounds),
        )
        return ResolvedBackendCircuit(
            backend=str(spec.backend),
            sector=sector_norm,
            error_rate=float(error_rate),
            syndrome_rounds=int(rounds),
            stim_path=stim_path,
            noisy_rounds=int(rounds),
            perfect_rounds=1,
            schedule_notes=(
                f"Non-default {spec.label} sector circuit generated locally from the arXiv:2604.19481v1 Table X three-ring schedule.",
                "The generated X/Z detector-side DEMs are sector-only Pauli circuits: X uses the scheduled CX layers, Z uses the scheduled CZ layers, with one final perfect stabilizer round.",
                "DEM extraction keeps Stim error mechanisms undecomposed for this backend because the scheduled two-qubit Pauli faults can produce large detector hyperedges.",
                "Loss, leakage, beacon gadgets, and the complementary stabilizer sector are not included in this local DEM-preparation model.",
            ),
        )
    if backend != "bravyi_depth7":
        raise ValueError(f"unsupported backend: {backend}")

    candidates = [
        item
        for item in list_available_public_circuit_rates(qtanner_root)
        if item.sector == sector_norm and item.syndrome_rounds == int(syndrome_rounds)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"no public gross stim circuits found for sector={sector_norm} rounds={int(syndrome_rounds)}"
        )
    for item in candidates:
        if abs(float(item.error_rate) - float(error_rate)) < 1e-12:
            return ResolvedBackendCircuit(
                backend=backend,
                sector=sector_norm,
                error_rate=float(item.error_rate),
                syndrome_rounds=int(item.syndrome_rounds),
                stim_path=item.stim_path,
                noisy_rounds=int(item.syndrome_rounds),
                perfect_rounds=1,
                schedule_notes=(
                    "Mapped to the public gross memory_X/memory_Z Stim circuits shipped with qtanner-ssf.",
                    "Detector coordinates are absent in the public files, so detector rounds are inferred from an even split of rows across 12 noisy rounds plus 1 final perfect round.",
                ),
            )
    available = ", ".join(str(item.error_rate) for item in candidates)
    raise FileNotFoundError(
        f"public bravyi_depth7 circuit for sector={sector_norm} error_rate={error_rate} not found. "
        f"Available rates: {available}"
    )
