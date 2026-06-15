from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import scipy.sparse as sp

from grosscode.dem.builder import SplitSectorMetadata, SplitSectorProblem


SectorName = Literal["X", "Z"]


def _sector_views(
    problem: SplitSectorProblem,
    sector: SectorName,
) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray, SplitSectorMetadata]:
    if sector == "X":
        return problem.D_X, problem.O_X, problem.priors_X, problem.metadata_X
    if sector == "Z":
        return problem.D_Z, problem.O_Z, problem.priors_Z, problem.metadata_Z
    raise ValueError(f"unsupported sector: {sector}")


def _histogram(values: np.ndarray) -> dict[str, int]:
    if values.size == 0:
        return {}
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(key)): int(count) for key, count in zip(unique.tolist(), counts.tolist(), strict=True)}


def _fault_class_by_column(metadata: SplitSectorMetadata) -> list[str]:
    span_to_class = {
        (int(group.round_start), int(group.round_stop)): str(group.fault_class)
        for group in metadata.variable_groups
    }
    out: list[str] = []
    for start, stop in zip(metadata.column_round_start.tolist(), metadata.column_round_stop.tolist(), strict=True):
        out.append(span_to_class[(int(start), int(stop))])
    return out


def build_sector_block_metadata(
    problem: SplitSectorProblem,
    *,
    sector: SectorName,
) -> dict[str, object]:
    matrix, observables, priors, metadata = _sector_views(problem, sector)
    row_counts = np.bincount(metadata.detector_round_index, minlength=metadata.total_rounds).astype(np.int32, copy=False)
    round_span = (metadata.column_round_stop - metadata.column_round_start).astype(np.int16, copy=False)
    time_support_length = (round_span + 1).astype(np.int16, copy=False)
    fault_class_by_column = _fault_class_by_column(metadata)
    final_round = int(metadata.total_rounds - 1)
    touches_final = np.flatnonzero(metadata.column_round_stop == final_round).astype(np.int32, copy=False)
    bridge_to_final = np.flatnonzero(
        (metadata.column_round_start == final_round - 1) & (metadata.column_round_stop == final_round)
    ).astype(np.int32, copy=False)
    final_local = np.flatnonzero(
        (metadata.column_round_start == final_round) & (metadata.column_round_stop == final_round)
    ).astype(np.int32, copy=False)

    return {
        "backend": str(metadata.backend),
        "sector": str(metadata.sector),
        "error_rate": float(metadata.error_rate),
        "stim_path": str(metadata.stim_path),
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "logical_matrix_shape": [int(observables.shape[0]), int(observables.shape[1])],
        "priors_length": int(priors.shape[0]),
        "noisy_rounds": int(metadata.noisy_rounds),
        "total_rounds": int(metadata.total_rounds),
        "detectors_per_round": int(metadata.detectors_per_round),
        "detector_rows_per_round": [int(x) for x in row_counts.tolist()],
        "round_boundaries": [
            {
                "round": int(round_index),
                "row_start": int(row_start),
                "row_stop": int(row_stop),
                "count": int(row_stop - row_start),
            }
            for round_index, row_start, row_stop in metadata.detector_round_slices
        ],
        "column_round_start": [int(x) for x in metadata.column_round_start.tolist()],
        "column_round_stop": [int(x) for x in metadata.column_round_stop.tolist()],
        "column_fault_class": fault_class_by_column,
        "time_support_length": [int(x) for x in time_support_length.tolist()],
        "ordered_column_index": [int(x) for x in metadata.ordered_column_index.tolist()],
        "column_group_boundaries": [
            {
                "label": str(group.label),
                "round_start": int(group.round_start),
                "round_stop": int(group.round_stop),
                "ordered_start": int(group.ordered_start),
                "ordered_stop": int(group.ordered_stop),
                "count": int(group.count),
                "fault_class": str(group.fault_class),
            }
            for group in metadata.variable_groups
        ],
        "columns_by_fault_class": {str(key): int(value) for key, value in sorted(metadata.local_fault_class_counts.items())},
        "columns_by_time_support_length": _histogram(time_support_length),
        "adjacency_histogram": _histogram(round_span),
        "touches_final_round_columns": [int(x) for x in touches_final.tolist()],
        "bridge_to_final_round_columns": [int(x) for x in bridge_to_final.tolist()],
        "final_round_local_columns": [int(x) for x in final_local.tolist()],
        "schedule_assumptions": [str(item) for item in metadata.schedule_assumptions],
    }


def build_problem_block_metadata(problem: SplitSectorProblem) -> dict[str, object]:
    x_summary = build_sector_block_metadata(problem, sector="X")
    z_summary = build_sector_block_metadata(problem, sector="Z")
    return {
        "backend": str(problem.metadata_X.backend),
        "error_rate": float(problem.metadata_X.error_rate),
        "codepath": {
            "entrypoint": "grosscode.dem.builder.build_split_sector_problem",
            "gross_code_loader": "grosscode.codes.gross144.load_gross144_code",
            "backend_resolver": "grosscode.circuits.backends.resolve_backend_circuit",
            "stim_dem_conversion": (
                "stim.Circuit.from_file -> circuit.detector_error_model("
                "decompose_errors=True, ignore_decomposition_failures=True) -> "
                "ldpc.ckt_noise.dem_matrices.detector_error_model_to_check_matrices("
                "allow_undecomposed_hyperedges=True)"
            ),
            "round_inference": "grosscode.dem.builder._build_metadata",
            "notes": [
                "This export describes the public split-sector Stim DEM path.",
                "It is not the same matrix family as the upstream decoder_setup.py HdecX/HdecZ effective model export.",
            ],
        },
        "columns_by_physical_sector": {
            "X": int(problem.D_X.shape[1]),
            "Z": int(problem.D_Z.shape[1]),
        },
        "logical_matrix_dimensions": {
            "X": [int(problem.O_X.shape[0]), int(problem.O_X.shape[1])],
            "Z": [int(problem.O_Z.shape[0]), int(problem.O_Z.shape[1])],
        },
        "sectors": {
            "X": x_summary,
            "Z": z_summary,
        },
    }


def format_problem_block_report(problem: SplitSectorProblem) -> str:
    payload = build_problem_block_metadata(problem)
    lines = [
        f"backend={payload['backend']} error_rate={payload['error_rate']}",
        "columns by physical sector:",
        f"  X={payload['columns_by_physical_sector']['X']} Z={payload['columns_by_physical_sector']['Z']}",
    ]
    for sector in ("X", "Z"):
        info = payload["sectors"][sector]
        lines.extend(
            [
                f"{sector} sector:",
                "  detector rows per round: "
                + ", ".join(str(count) for count in info["detector_rows_per_round"]),
                f"  number of rounds inferred: {info['total_rounds']}",
                "  columns by fault class: "
                + ", ".join(f"{key}={value}" for key, value in info["columns_by_fault_class"].items()),
                "  columns by physical sector: "
                + f"{sector}={info['matrix_shape'][1]}",
                "  columns by time support length: "
                + ", ".join(f"{key}={value}" for key, value in info["columns_by_time_support_length"].items()),
                "  adjacency histogram: "
                + ", ".join(f"{key}={value}" for key, value in info["adjacency_histogram"].items()),
                f"  logical matrix dimensions: {info['logical_matrix_shape'][0]} x {info['logical_matrix_shape'][1]}",
            ]
        )
    return "\n".join(lines) + "\n"


def write_problem_block_metadata(
    problem: SplitSectorProblem,
    output_path: str | Path,
) -> Path:
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_problem_block_metadata(problem)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
