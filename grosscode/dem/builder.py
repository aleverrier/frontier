from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from grosscode.codes.bivariate_bicycle import is_bivariate_bicycle_backend, load_bivariate_bicycle_code
from grosscode.codes.generalized_bicycle import is_generalized_bicycle_backend, load_generalized_bicycle_code
from grosscode.codes.gross144 import load_gross144_code
from grosscode.codes.rotated_surface import is_rotated_surface_backend, load_rotated_surface_code
from grosscode.circuits.backends import ResolvedBackendCircuit, resolve_backend_circuit
from grosscode.utils.gf2 import binary_csr_mod2


@dataclass(frozen=True)
class VariableGroup:
    label: str
    round_start: int
    round_stop: int
    ordered_start: int
    ordered_stop: int
    count: int
    fault_class: str


@dataclass(frozen=True)
class SplitSectorMetadata:
    backend: str
    sector: str
    error_rate: float
    noisy_rounds: int
    total_rounds: int
    detectors_per_round: int
    detector_round_index: np.ndarray
    detector_round_slices: tuple[tuple[int, int, int], ...]
    column_round_start: np.ndarray
    column_round_stop: np.ndarray
    ordered_column_index: np.ndarray
    variable_groups: tuple[VariableGroup, ...]
    local_fault_class_counts: dict[str, int]
    schedule_assumptions: tuple[str, ...]
    stim_path: str


@dataclass(frozen=True)
class SplitSectorProblem:
    HX: sp.csr_matrix
    HZ: sp.csr_matrix
    LX: np.ndarray
    LZ: np.ndarray
    D_X: sp.csr_matrix
    D_Z: sp.csr_matrix
    O_X: sp.csr_matrix
    O_Z: sp.csr_matrix
    priors_X: np.ndarray
    priors_Z: np.ndarray
    metadata_X: SplitSectorMetadata
    metadata_Z: SplitSectorMetadata


@dataclass(frozen=True)
class LoadedDemSideWithMetadata:
    check_matrix: sp.csr_matrix
    observables_matrix: sp.csr_matrix
    priors: np.ndarray
    metadata: SplitSectorMetadata
    stim_path: Path


def _fault_class_for_span(round_start: int, round_stop: int, total_rounds: int) -> str:
    if round_start == round_stop == total_rounds - 1:
        return "final_round_local"
    if round_start == round_stop:
        return "within_round"
    span = int(round_stop) - int(round_start) + 1
    if round_stop == round_start + 1 and round_stop == total_rounds - 1:
        return "bridge_to_final_round"
    if round_stop == round_start + 1:
        return "bridge_consecutive_rounds"
    return f"span_{int(span)}_rounds"


def _build_metadata(spec: ResolvedBackendCircuit, check_matrix: sp.csr_matrix) -> SplitSectorMetadata:
    n_detectors, n_columns = map(int, check_matrix.shape)
    total_rounds = int(spec.noisy_rounds + spec.perfect_rounds)
    if total_rounds <= 0:
        raise ValueError("total rounds must be positive")
    if n_detectors % total_rounds != 0:
        raise ValueError(
            f"cannot infer detector round partition for {spec.stim_path}: {n_detectors} detectors over {total_rounds} rounds"
        )
    detectors_per_round = n_detectors // total_rounds
    detector_round_index = np.repeat(np.arange(total_rounds, dtype=np.int16), detectors_per_round)
    detector_round_slices = tuple(
        (round_index, round_index * detectors_per_round, (round_index + 1) * detectors_per_round)
        for round_index in range(total_rounds)
    )

    csc = check_matrix.tocsc()
    round_start = np.empty(n_columns, dtype=np.int16)
    round_stop = np.empty(n_columns, dtype=np.int16)
    for col in range(n_columns):
        begin = int(csc.indptr[col])
        end = int(csc.indptr[col + 1])
        rows = csc.indices[begin:end]
        if rows.size == 0:
            raise ValueError(f"column {col} has no detector support; schedule metadata is incomplete")
        col_rounds = np.asarray(rows // detectors_per_round, dtype=np.int16)
        round_start[col] = int(col_rounds.min())
        round_stop[col] = int(col_rounds.max())

    variable_groups: list[VariableGroup] = []
    ordered_parts: list[np.ndarray] = []
    class_counts: dict[str, int] = {}
    cursor = 0
    unique_spans = sorted({(int(a), int(b)) for a, b in zip(round_start.tolist(), round_stop.tolist())})
    for span_start, span_stop in unique_spans:
        columns = np.flatnonzero((round_start == span_start) & (round_stop == span_stop)).astype(np.int32)
        ordered_parts.append(columns)
        fault_class = _fault_class_for_span(span_start, span_stop, total_rounds)
        class_counts[fault_class] = class_counts.get(fault_class, 0) + int(columns.size)
        variable_groups.append(
            VariableGroup(
                label=f"r{span_start}_to_r{span_stop}",
                round_start=span_start,
                round_stop=span_stop,
                ordered_start=cursor,
                ordered_stop=cursor + int(columns.size),
                count=int(columns.size),
                fault_class=fault_class,
            )
        )
        cursor += int(columns.size)
    ordered_column_index = (
        np.concatenate(ordered_parts).astype(np.int32, copy=False) if ordered_parts else np.zeros(0, dtype=np.int32)
    )

    assumptions = tuple(str(note) for note in spec.schedule_notes) + (
        f"Rows are split evenly into {total_rounds} detector rounds by the maintained metadata builder.",
        "Fault-column round spans are inferred from min/max detector-row support only.",
        "Columns spanning more than adjacent rounds are retained and labeled by their inferred span length.",
    )
    return SplitSectorMetadata(
        backend=spec.backend,
        sector=spec.sector,
        error_rate=float(spec.error_rate),
        noisy_rounds=int(spec.noisy_rounds),
        total_rounds=total_rounds,
        detectors_per_round=detectors_per_round,
        detector_round_index=detector_round_index,
        detector_round_slices=detector_round_slices,
        column_round_start=round_start,
        column_round_stop=round_stop,
        ordered_column_index=ordered_column_index,
        variable_groups=tuple(variable_groups),
        local_fault_class_counts=class_counts,
        schedule_assumptions=assumptions,
        stim_path=str(spec.stim_path),
    )


def _load_dem_matrices(
    stim_path: Path,
    *,
    decompose_errors: bool = True,
) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    import stim  # type: ignore
    from ldpc.ckt_noise.dem_matrices import detector_error_model_to_check_matrices  # type: ignore

    circuit = stim.Circuit.from_file(str(stim_path))
    dem = circuit.detector_error_model(
        decompose_errors=bool(decompose_errors),
        ignore_decomposition_failures=True,
    )
    matrices = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True)
    check_matrix = binary_csr_mod2(matrices.check_matrix.tocsr())
    observables_matrix = binary_csr_mod2(matrices.observables_matrix.tocsr())
    priors = np.asarray(matrices.priors, dtype=np.float64).reshape(-1)
    return check_matrix, observables_matrix, priors


def load_dem_side_with_metadata_from_stim(
    *,
    stim_path: str | Path,
    backend: str,
    sector: str,
    error_rate: float,
    noisy_rounds: int,
    perfect_rounds: int = 1,
) -> LoadedDemSideWithMetadata:
    path = Path(stim_path).expanduser().resolve()
    sector_text = str(sector).strip().upper().replace("MEMORY_", "")
    if sector_text not in {"X", "Z"}:
        raise ValueError(f"sector must resolve to 'X' or 'Z', got {sector!r}")
    check_matrix, observables_matrix, priors = _load_dem_matrices(path)
    spec = ResolvedBackendCircuit(
        backend=str(backend),
        sector=sector_text,
        error_rate=float(error_rate),
        syndrome_rounds=int(noisy_rounds),
        stim_path=path,
        noisy_rounds=int(noisy_rounds),
        perfect_rounds=int(perfect_rounds),
        schedule_notes=(
            f"External Stim circuit loaded directly from `{path}`.",
            f"Detector-side DEM metadata is attached under backend label `{backend}` for sector `{sector_text}`.",
        ),
    )
    metadata = _build_metadata(spec, check_matrix)
    return LoadedDemSideWithMetadata(
        check_matrix=check_matrix,
        observables_matrix=observables_matrix,
        priors=priors,
        metadata=metadata,
        stim_path=path,
    )


def _load_code_for_backend(root_text: str | None, backend: str):
    if is_bivariate_bicycle_backend(backend):
        return load_bivariate_bicycle_code(backend=str(backend))
    if is_generalized_bicycle_backend(backend):
        return load_generalized_bicycle_code(backend=str(backend))
    if is_rotated_surface_backend(backend):
        return load_rotated_surface_code(backend=str(backend))
    return load_gross144_code(root_text)


@lru_cache(maxsize=8)
def _build_cached(
    root_text: str | None,
    backend: str,
    error_rate: float,
    initial_data_error_rate: float | None,
) -> SplitSectorProblem:
    code = _load_code_for_backend(root_text, backend)
    spec_x = resolve_backend_circuit(
        backend=backend,
        sector="X",
        error_rate=error_rate,
        initial_data_error_rate=initial_data_error_rate,
        qtanner_root=root_text,
    )
    spec_z = resolve_backend_circuit(
        backend=backend,
        sector="Z",
        error_rate=error_rate,
        initial_data_error_rate=initial_data_error_rate,
        qtanner_root=root_text,
    )
    decompose_errors = not is_generalized_bicycle_backend(backend)
    d_x, o_x, priors_x = _load_dem_matrices(spec_x.stim_path, decompose_errors=decompose_errors)
    d_z, o_z, priors_z = _load_dem_matrices(spec_z.stim_path, decompose_errors=decompose_errors)
    metadata_x = _build_metadata(spec_x, d_x)
    metadata_z = _build_metadata(spec_z, d_z)
    return SplitSectorProblem(
        HX=code.HX,
        HZ=code.HZ,
        LX=code.LX,
        LZ=code.LZ,
        D_X=d_x,
        D_Z=d_z,
        O_X=o_x,
        O_Z=o_z,
        priors_X=priors_x,
        priors_Z=priors_z,
        metadata_X=metadata_x,
        metadata_Z=metadata_z,
    )


def build_split_sector_problem(
    *,
    backend: str = "bravyi_depth7",
    error_rate: float = 0.004,
    initial_data_error_rate: float | None = None,
    qtanner_root: str | Path | None = None,
) -> SplitSectorProblem:
    root_text = None if qtanner_root is None else str(Path(qtanner_root))
    return _build_cached(
        root_text,
        str(backend),
        float(error_rate),
        None if initial_data_error_rate is None else float(initial_data_error_rate),
    )
