from __future__ import annotations

import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import stim  # type: ignore

from grosscode.codes.css import CSSCode, derive_css_code_from_checks


Q102_BACKEND = "q102_gb_102_22_9"


@dataclass(frozen=True)
class ScheduleTerm:
    family: str
    index: int
    transpose: bool = False


@dataclass(frozen=True)
class GeneralizedBicycleBackendSpec:
    backend: str
    aliases: tuple[str, ...]
    label: str
    n: int
    k: int
    distance: int
    ell: int
    a_pows: tuple[int, ...]
    b_pows: tuple[int, ...]
    syndrome_rounds: int
    schedule: tuple[tuple[ScheduleTerm, ScheduleTerm], ...]
    source_note: str


_Q102_SCHEDULE: tuple[tuple[ScheduleTerm, ScheduleTerm], ...] = (
    (ScheduleTerm("B", 1), ScheduleTerm("B", 4, transpose=True)),
    (ScheduleTerm("A", 1), ScheduleTerm("A", 2, transpose=True)),
    (ScheduleTerm("A", 3), ScheduleTerm("A", 4, transpose=True)),
    (ScheduleTerm("B", 2), ScheduleTerm("B", 3, transpose=True)),
    (ScheduleTerm("B", 3), ScheduleTerm("B", 2, transpose=True)),
    (ScheduleTerm("A", 4), ScheduleTerm("A", 3, transpose=True)),
    (ScheduleTerm("A", 2), ScheduleTerm("A", 1, transpose=True)),
    (ScheduleTerm("B", 4), ScheduleTerm("B", 1, transpose=True)),
)


_GB_BACKEND_SPECS: dict[str, GeneralizedBicycleBackendSpec] = {
    Q102_BACKEND: GeneralizedBicycleBackendSpec(
        backend=Q102_BACKEND,
        aliases=("q102", "gb_q102", "gb_102_22_9"),
        label="Q102 GB [[102,22,9]]",
        n=102,
        k=22,
        distance=9,
        ell=51,
        a_pows=(22, 26, 37, 50),
        b_pows=(19, 28, 29, 35),
        syndrome_rounds=9,
        schedule=_Q102_SCHEDULE,
        source_note=(
            "Walking-cat paper arXiv:2604.19481v1 Appendix C/Table XXX and "
            "Section IX/Table X: GB8 ell=51, A=x^22+x^26+x^37+x^50, "
            "B=x^19+x^28+x^29+x^35, 8-layer Q102 schedule."
        ),
    ),
}

_GB_ALIAS_TO_BACKEND: dict[str, str] = {
    str(spec.backend): str(spec.backend)
    for spec in _GB_BACKEND_SPECS.values()
} | {
    str(alias): str(spec.backend)
    for spec in _GB_BACKEND_SPECS.values()
    for alias in spec.aliases
}


def is_generalized_bicycle_backend(backend: str) -> bool:
    return str(backend) in _GB_ALIAS_TO_BACKEND


def get_generalized_bicycle_backend_spec(backend: str) -> GeneralizedBicycleBackendSpec:
    key = str(backend)
    if key not in _GB_ALIAS_TO_BACKEND:
        raise ValueError(f"unsupported generalized-bicycle backend: {backend}")
    return _GB_BACKEND_SPECS[_GB_ALIAS_TO_BACKEND[key]]


def _rate_tag(value: float) -> str:
    return str(float(value)).replace("-", "m").replace(".", "p")


def _circulant_power(ell: int, power: int) -> sp.csr_matrix:
    rows = np.arange(int(ell), dtype=np.int32)
    cols = (rows + int(power)) % int(ell)
    data = np.ones(int(ell), dtype=np.uint8)
    return sp.csr_matrix((data, (rows, cols)), shape=(int(ell), int(ell)), dtype=np.uint8)


def _sum_terms(ell: int, powers: tuple[int, ...]) -> sp.csr_matrix:
    out = sp.csr_matrix((int(ell), int(ell)), dtype=np.uint8)
    for power in powers:
        out = out + _circulant_power(int(ell), int(power))
    out.data %= 2
    out.eliminate_zeros()
    return out.tocsr()


def _term_matrices(spec: GeneralizedBicycleBackendSpec) -> tuple[dict[int, sp.csr_matrix], dict[int, sp.csr_matrix]]:
    a_terms = {index + 1: _circulant_power(int(spec.ell), int(power)) for index, power in enumerate(spec.a_pows)}
    b_terms = {index + 1: _circulant_power(int(spec.ell), int(power)) for index, power in enumerate(spec.b_pows)}
    return a_terms, b_terms


@lru_cache(maxsize=8)
def _load_code_cached(backend: str) -> CSSCode:
    spec = get_generalized_bicycle_backend_spec(backend)
    a_matrix = _sum_terms(int(spec.ell), tuple(spec.a_pows))
    b_matrix = _sum_terms(int(spec.ell), tuple(spec.b_pows))
    hx = sp.hstack([a_matrix, b_matrix], format="csr", dtype=np.uint8)
    hz = sp.hstack([b_matrix.T, a_matrix.T], format="csr", dtype=np.uint8)
    return derive_css_code_from_checks(
        name=str(spec.label),
        hx=hx,
        hz=hz,
        metadata={
            "backend": str(spec.backend),
            "family": "generalized_bicycle",
            "ell": int(spec.ell),
            "a_pows": tuple(int(value) for value in spec.a_pows),
            "b_pows": tuple(int(value) for value in spec.b_pows),
            "distance_hint": int(spec.distance),
            "source": str(spec.source_note),
        },
    )


def load_generalized_bicycle_code(*, backend: str) -> CSSCode:
    return _load_code_cached(str(get_generalized_bicycle_backend_spec(backend).backend))


def _row_targets(matrix: sp.csr_matrix) -> np.ndarray:
    coo = matrix.tocoo()
    order = np.argsort(coo.row, kind="stable")
    rows = np.asarray(coo.row[order], dtype=np.int32)
    cols = np.asarray(coo.col[order], dtype=np.int32)
    expected = np.arange(int(matrix.shape[0]), dtype=np.int32)
    if rows.shape != expected.shape or not np.array_equal(rows, expected):
        raise ValueError("expected a permutation matrix with exactly one target per row")
    return cols


def _matrix_for_term(
    term: ScheduleTerm,
    *,
    a_terms: dict[int, sp.csr_matrix],
    b_terms: dict[int, sp.csr_matrix],
) -> sp.csr_matrix:
    family = str(term.family).upper()
    if family == "A":
        matrix = a_terms[int(term.index)]
    elif family == "B":
        matrix = b_terms[int(term.index)]
    else:
        raise ValueError(f"invalid schedule family: {term.family!r}")
    return matrix.T.tocsr() if bool(term.transpose) else matrix


def build_generalized_bicycle_circuit(
    *,
    backend: str,
    sector: str,
    error_rate: float,
    syndrome_rounds: int | None = None,
) -> stim.Circuit:
    spec = get_generalized_bicycle_backend_spec(backend)
    code = load_generalized_bicycle_code(backend=str(spec.backend))
    rounds = int(spec.syndrome_rounds if syndrome_rounds is None else syndrome_rounds)
    if rounds <= 0:
        raise ValueError("syndrome_rounds must be positive")
    sector_norm = str(sector).upper()
    if sector_norm not in {"X", "Z"}:
        raise ValueError("sector must be 'X' or 'Z'")

    half = int(spec.ell)
    n = int(spec.n)
    if n != 2 * half:
        raise ValueError("this builder expects a GB code with n = 2 * ell")

    check_offset = 0
    left_data_offset = half
    right_data_offset = half + half

    p = float(error_rate)
    a_terms, b_terms = _term_matrices(spec)
    schedule_targets: list[tuple[str, np.ndarray]] = []
    for x_term, z_term in spec.schedule:
        if sector_norm == "X":
            term = x_term
            data_half = "left" if str(term.family).upper() == "A" else "right"
        else:
            term = z_term
            data_half = "right" if str(term.family).upper() == "A" else "left"
        schedule_targets.append(
            (data_half, _row_targets(_matrix_for_term(term, a_terms=a_terms, b_terms=b_terms)))
        )

    detector_initial = stim.Circuit()
    for i in range(half):
        detector_initial.append("DETECTOR", [stim.target_rec(-half + i)])

    detector_repeat = stim.Circuit()
    for i in range(half):
        detector_repeat.append("DETECTOR", [stim.target_rec(-half + i), stim.target_rec(-2 * half + i)])

    def append_sector_detectors(circuit: stim.Circuit, *, repeat: bool) -> None:
        circuit += detector_repeat if repeat else detector_initial

    def append_sec(circuit: stim.Circuit, *, repeat: bool) -> None:
        if repeat:
            for i in range(half):
                circuit.append("Z_ERROR", [check_offset + i], p)

        for data_half, targets in schedule_targets:
            data_offset = left_data_offset if data_half == "left" else right_data_offset
            for i in range(half):
                pair = [check_offset + i, data_offset + int(targets[i])]
                circuit.append("CNOT" if sector_norm == "X" else "CZ", pair)
                circuit.append("DEPOLARIZE2", pair, p)
            circuit.append("TICK")

        for i in range(half):
            circuit.append("Z_ERROR", [check_offset + i], p)
            circuit.append("MRX", [check_offset + i])
        append_sector_detectors(circuit, repeat=repeat)
        circuit.append("TICK")

    circuit = stim.Circuit()
    for i in range(half):
        circuit.append("RX", [check_offset + i])
        circuit.append("Z_ERROR", [check_offset + i], p)
    for i in range(n):
        circuit.append("R" if sector_norm == "Z" else "RX", [left_data_offset + i])
        circuit.append("X_ERROR" if sector_norm == "Z" else "Z_ERROR", [left_data_offset + i], p)
    circuit.append("TICK")

    append_sec(circuit, repeat=False)
    repeat_sec = stim.Circuit()
    append_sec(repeat_sec, repeat=True)
    if rounds > 1:
        circuit += (rounds - 1) * repeat_sec

    for i in range(n):
        circuit.append("M" if sector_norm == "Z" else "MX", [left_data_offset + i])

    pcm = code.HZ if sector_norm == "Z" else code.HX
    logical_pcm = code.LZ if sector_norm == "Z" else code.LX
    data_offset = -n
    latest_check_offset = -n - half
    for check_index, row in enumerate(pcm.toarray().astype(np.uint8, copy=False)):
        targets = [stim.target_rec(data_offset + int(ind)) for ind in np.flatnonzero(row)]
        targets.append(stim.target_rec(latest_check_offset + int(check_index)))
        circuit.append("DETECTOR", targets)

    for logical_index, row in enumerate(np.asarray(logical_pcm, dtype=np.uint8)):
        targets = [stim.target_rec(data_offset + int(ind)) for ind in np.flatnonzero(row)]
        circuit.append("OBSERVABLE_INCLUDE", targets, float(logical_index))

    return circuit


def build_generalized_bicycle_circuit_text(
    *,
    backend: str,
    sector: str,
    error_rate: float,
    syndrome_rounds: int | None = None,
) -> str:
    return str(
        build_generalized_bicycle_circuit(
            backend=str(backend),
            sector=str(sector),
            error_rate=float(error_rate),
            syndrome_rounds=syndrome_rounds,
        )
    )


def generated_generalized_bicycle_stim_path(
    *,
    backend: str,
    sector: str,
    error_rate: float,
    syndrome_rounds: int | None = None,
) -> Path:
    spec = get_generalized_bicycle_backend_spec(backend)
    rounds = int(spec.syndrome_rounds if syndrome_rounds is None else syndrome_rounds)
    cache_root = Path(tempfile.gettempdir()) / "better_beam_backend_stim"
    cache_root.mkdir(parents=True, exist_ok=True)
    name = (
        f"{spec.backend},memory_{str(sector).upper()},error_rate={_rate_tag(float(error_rate))},"
        f"sector_only_v2,syndrome_rounds={int(rounds)}.stim"
    )
    return cache_root / name


def ensure_generalized_bicycle_stim_file(
    *,
    backend: str,
    sector: str,
    error_rate: float,
    syndrome_rounds: int | None = None,
) -> Path:
    path = generated_generalized_bicycle_stim_path(
        backend=str(backend),
        sector=str(sector),
        error_rate=float(error_rate),
        syndrome_rounds=syndrome_rounds,
    )
    if path.exists():
        return path
    path.write_text(
        build_generalized_bicycle_circuit_text(
            backend=str(backend),
            sector=str(sector),
            error_rate=float(error_rate),
            syndrome_rounds=syndrome_rounds,
        ),
        encoding="utf-8",
    )
    return path


__all__ = [
    "GeneralizedBicycleBackendSpec",
    "Q102_BACKEND",
    "ScheduleTerm",
    "build_generalized_bicycle_circuit",
    "build_generalized_bicycle_circuit_text",
    "ensure_generalized_bicycle_stim_file",
    "generated_generalized_bicycle_stim_path",
    "get_generalized_bicycle_backend_spec",
    "is_generalized_bicycle_backend",
    "load_generalized_bicycle_code",
]
