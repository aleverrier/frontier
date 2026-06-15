from __future__ import annotations

import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import scipy.sparse as sp

from grosscode.codes.css import CSSCode, derive_css_code_from_checks
from grosscode.utils.paths import REPO_ROOT
from sliding_window_baseline import DEFAULT_UPSTREAM_REPO_DIR, ensure_upstream_ready, prepend_repo_to_syspath


BB_72_12_6_BACKEND = "bb_72_12_6"
BB_90_8_10_BACKEND = "bb_90_8_10"


@dataclass(frozen=True)
class BivariateBicycleBackendSpec:
    backend: str
    n: int
    k: int
    distance: int
    l: int
    m: int
    a_x_pows: tuple[int, ...]
    a_y_pows: tuple[int, ...]
    b_x_pows: tuple[int, ...]
    b_y_pows: tuple[int, ...]
    syndrome_rounds: int


_BB_BACKEND_SPECS: dict[str, BivariateBicycleBackendSpec] = {
    BB_72_12_6_BACKEND: BivariateBicycleBackendSpec(
        backend=BB_72_12_6_BACKEND,
        n=72,
        k=12,
        distance=6,
        l=6,
        m=6,
        a_x_pows=(3,),
        a_y_pows=(1, 2),
        b_x_pows=(1, 2),
        b_y_pows=(3,),
        syndrome_rounds=6,
    ),
    BB_90_8_10_BACKEND: BivariateBicycleBackendSpec(
        backend=BB_90_8_10_BACKEND,
        n=90,
        k=8,
        distance=10,
        l=15,
        m=3,
        a_x_pows=(9,),
        a_y_pows=(1, 2),
        b_x_pows=(2, 7),
        b_y_pows=(0,),
        syndrome_rounds=10,
    ),
}


def is_bivariate_bicycle_backend(backend: str) -> bool:
    return str(backend) in _BB_BACKEND_SPECS


def get_bivariate_bicycle_backend_spec(backend: str) -> BivariateBicycleBackendSpec:
    key = str(backend)
    if key not in _BB_BACKEND_SPECS:
        raise ValueError(f"unsupported bivariate-bicycle backend: {backend}")
    return _BB_BACKEND_SPECS[key]


def _python_bin(python_bin: str | Path | None = None) -> str:
    if python_bin is not None:
        return str(Path(python_bin))
    candidate = REPO_ROOT / "tools" / "py"
    if candidate.exists():
        return str(candidate)
    return "python3"


def _rate_tag(value: float) -> str:
    return str(float(value)).replace("-", "m").replace(".", "p")


def _format_stim_probability(value: float) -> str:
    return f"{float(value):.17g}"


def _replace_initial_data_error_rate(
    circuit_text: str,
    *,
    spec: BivariateBicycleBackendSpec,
    sector: str,
    initial_data_error_rate: float,
) -> str:
    """Replace only the initial data-preparation error rate in the generated circuit."""
    data_start = int(spec.l * spec.m)
    data_stop = int(data_start + spec.n)
    expected_error = "Z_ERROR" if str(sector).upper() == "X" else "X_ERROR"
    new_prob = _format_stim_probability(float(initial_data_error_rate))
    replaced = 0
    out: list[str] = []
    in_initial_block = True
    for line in circuit_text.splitlines():
        stripped = line.strip()
        if in_initial_block and stripped == "TICK":
            in_initial_block = False
        if in_initial_block and stripped.startswith(f"{expected_error}("):
            parts = stripped.split()
            if len(parts) == 2:
                try:
                    qubit = int(parts[1])
                except ValueError:
                    qubit = -1
                if data_start <= qubit < data_stop:
                    indent = line[: len(line) - len(line.lstrip())]
                    line = f"{indent}{expected_error}({new_prob}) {qubit}"
                    replaced += 1
        out.append(line)
    if replaced != int(spec.n):
        raise ValueError(
            "failed to replace exactly the initial data-qubit error layer for "
            f"{spec.backend} memory_{str(sector).upper()}: replaced {replaced}, expected {int(spec.n)}"
        )
    return "\n".join(out) + ("\n" if circuit_text.endswith("\n") else "")


@lru_cache(maxsize=8)
def _load_repo_modules(repo_dir_text: str, python_bin_text: str) -> tuple[object, object]:
    repo_dir = Path(repo_dir_text)
    ensure_upstream_ready(repo_dir, python_bin=python_bin_text)
    prepend_repo_to_syspath(repo_dir)
    from src.build_circuit import build_circuit  # type: ignore
    from src.codes_q import create_bivariate_bicycle_codes  # type: ignore

    return create_bivariate_bicycle_codes, build_circuit


@lru_cache(maxsize=8)
def _load_code_cached(backend: str, repo_dir_text: str, python_bin_text: str) -> CSSCode:
    spec = get_bivariate_bicycle_backend_spec(backend)
    create_code, _ = _load_repo_modules(repo_dir_text, python_bin_text)
    code, _, _ = create_code(
        int(spec.l),
        int(spec.m),
        list(spec.a_x_pows),
        list(spec.a_y_pows),
        list(spec.b_x_pows),
        list(spec.b_y_pows),
    )
    return derive_css_code_from_checks(
        name=f"BB [[{int(spec.n)},{int(spec.k)},{int(spec.distance)}]]",
        hx=sp.csr_matrix(code.hx, dtype="uint8"),
        hz=sp.csr_matrix(code.hz, dtype="uint8"),
        metadata={
            "source": "local SlidingWindowDecoder bivariate-bicycle constructor",
            "backend": str(spec.backend),
            "distance_hint": int(spec.distance),
            "upstream_repo_dir": str(repo_dir_text),
        },
    )


def load_bivariate_bicycle_code(
    *,
    backend: str,
    upstream_repo_dir: str | Path | None = None,
    python_bin: str | Path | None = None,
) -> CSSCode:
    repo_dir = Path(upstream_repo_dir) if upstream_repo_dir is not None else DEFAULT_UPSTREAM_REPO_DIR
    return _load_code_cached(str(backend), str(repo_dir.resolve()), _python_bin(python_bin))


def build_bivariate_bicycle_circuit_text(
    *,
    backend: str,
    sector: str,
    error_rate: float,
    initial_data_error_rate: float | None = None,
    syndrome_rounds: int | None = None,
    upstream_repo_dir: str | Path | None = None,
    python_bin: str | Path | None = None,
) -> str:
    spec = get_bivariate_bicycle_backend_spec(backend)
    rounds = int(spec.syndrome_rounds if syndrome_rounds is None else syndrome_rounds)
    repo_dir = Path(upstream_repo_dir) if upstream_repo_dir is not None else DEFAULT_UPSTREAM_REPO_DIR
    create_code, build_circuit = _load_repo_modules(str(repo_dir.resolve()), _python_bin(python_bin))
    code, a_list, b_list = create_code(
        int(spec.l),
        int(spec.m),
        list(spec.a_x_pows),
        list(spec.a_y_pows),
        list(spec.b_x_pows),
        list(spec.b_y_pows),
    )
    circuit = build_circuit(
        code,
        a_list,
        b_list,
        float(error_rate),
        int(rounds),
        z_basis=str(sector).upper() == "Z",
    )
    text = str(circuit)
    if initial_data_error_rate is not None:
        text = _replace_initial_data_error_rate(
            text,
            spec=spec,
            sector=str(sector),
            initial_data_error_rate=float(initial_data_error_rate),
        )
    return text


def generated_bivariate_bicycle_stim_path(
    *,
    backend: str,
    sector: str,
    error_rate: float,
    initial_data_error_rate: float | None = None,
    syndrome_rounds: int | None = None,
) -> Path:
    spec = get_bivariate_bicycle_backend_spec(backend)
    rounds = int(spec.syndrome_rounds if syndrome_rounds is None else syndrome_rounds)
    cache_root = Path(tempfile.gettempdir()) / "better_beam_backend_stim"
    cache_root.mkdir(parents=True, exist_ok=True)
    rate_tag = _rate_tag(float(error_rate))
    pinit_tag = "" if initial_data_error_rate is None else f",initial_data_error_rate={_rate_tag(float(initial_data_error_rate))}"
    name = (
        f"{backend},memory_{str(sector).upper()},error_rate={rate_tag}"
        f"{pinit_tag},syndrome_rounds={int(rounds)}.stim"
    )
    return cache_root / name


def ensure_bivariate_bicycle_stim_file(
    *,
    backend: str,
    sector: str,
    error_rate: float,
    initial_data_error_rate: float | None = None,
    syndrome_rounds: int | None = None,
    upstream_repo_dir: str | Path | None = None,
    python_bin: str | Path | None = None,
) -> Path:
    path = generated_bivariate_bicycle_stim_path(
        backend=str(backend),
        sector=str(sector),
        error_rate=float(error_rate),
        initial_data_error_rate=initial_data_error_rate,
        syndrome_rounds=syndrome_rounds,
    )
    if path.exists():
        return path
    text = build_bivariate_bicycle_circuit_text(
        backend=str(backend),
        sector=str(sector),
        error_rate=float(error_rate),
        initial_data_error_rate=initial_data_error_rate,
        syndrome_rounds=syndrome_rounds,
        upstream_repo_dir=upstream_repo_dir,
        python_bin=python_bin,
    )
    path.write_text(text, encoding="utf-8")
    return path


__all__ = [
    "BB_72_12_6_BACKEND",
    "BB_90_8_10_BACKEND",
    "BivariateBicycleBackendSpec",
    "build_bivariate_bicycle_circuit_text",
    "ensure_bivariate_bicycle_stim_file",
    "generated_bivariate_bicycle_stim_path",
    "get_bivariate_bicycle_backend_spec",
    "is_bivariate_bicycle_backend",
    "load_bivariate_bicycle_code",
]
