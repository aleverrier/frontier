#!/usr/bin/env python3
"""Minimal DEM-to-frontier loader for BB/Gross and surface-code backends.

Public entry points: `LoadedProgressiveFamily`, `SUPPORTED_COLUMN_ORDERS`,
`load_dem_family`, `build_backward_deadline_ordered_family`, and `main`.

This module is both support/library code and the implementation of
`frontier-dem-info`.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grosscode.codes.generalized_bicycle import get_generalized_bicycle_backend_spec, is_generalized_bicycle_backend
from grosscode.codes.rotated_surface import is_rotated_surface_backend
from grosscode.dem.builder import build_split_sector_problem, load_dem_side_with_metadata_from_stim
from tools import frontier_progressive as progressive


SUPPORTED_COLUMN_ORDERS = (
    "deadline_reorder",
    "fwd_deadline",
    "time_order",
    "backward_deadline_reorder",
    "bwd_deadline",
)


@dataclass(frozen=True, slots=True)
class LoadedProgressiveFamily:
    backend: str
    family_key: str
    scope: str
    scope_label: str
    benchmark_title: str
    benchmark_description: str
    benchmark_source_note: str
    detector_symbol: str
    logical_symbol: str
    metadata_symbol: str
    priors_symbol: str
    column_order_name: str
    column_order_source: str
    model_label: str
    decode_label: str
    columns: tuple[progressive.ProgressiveColumn, ...]
    layout: progressive.ProgressiveFrontierLayout
    matrix_rows: int
    matrix_cols: int
    logical_rows: int
    edge_count: int
    noisy_rounds: int
    total_rounds: int
    correction_state_mode: str = "none"
    correction_state_bits: int = 0


def _benchmark_descriptor(backend: str) -> tuple[str, str]:
    backend_text = str(backend)
    if backend_text == "bravyi_depth7":
        return (
            "Gross split-sector DEM",
            "accepted public split-sector detector-side DEM benchmark",
        )
    if is_generalized_bicycle_backend(backend_text):
        spec = get_generalized_bicycle_backend_spec(backend_text)
        return (
            f"{spec.label} split-sector DEM",
            f"{spec.label} detector-side DEM benchmark built locally from the configured schedule",
        )
    if is_rotated_surface_backend(backend_text) or backend_text.startswith("surface_"):
        return (
            f"Rotated surface-code DEM ({backend_text})",
            f"rotated surface-code detector-side DEM benchmark for backend `{backend_text}`",
        )
    return (
        f"Detector-side DEM ({backend_text})",
        f"detector-side DEM benchmark for backend `{backend_text}`",
    )


def _external_benchmark_descriptor(*, backend: str, benchmark_label: str, stim_path: Path) -> tuple[str, str, str]:
    label = str(benchmark_label).strip() or f"External Stim DEM ({stim_path.name})"
    description = f"external detector-side DEM benchmark loaded from Stim circuit `{stim_path}` under backend label `{backend}`"
    return label, description, f"{description}."


def _bitmask_from_indices(indices: Sequence[int]) -> int:
    out = 0
    for index in indices:
        out |= 1 << int(index)
    return int(out)


def _log_probs_from_probs(values: Sequence[float]) -> tuple[float, ...]:
    return tuple(float("-inf") if float(value) <= 0.0 else float(math.log(float(value))) for value in values)


def _ordered_columns_by_mode(
    columns_time_ordered: Sequence[progressive.ProgressiveColumn],
    *,
    num_detectors: int,
    column_order: str,
) -> tuple[list[progressive.ProgressiveColumn], tuple[int, ...], str, str]:
    order_key = str(column_order).strip().lower()
    if order_key in {"backward_deadline_reorder", "bwd_deadline"}:
        raise ValueError("backward deadline order is derived with build_backward_deadline_ordered_family")
    if order_key == "time_order":
        ordering = tuple(range(len(columns_time_ordered)))
        return (
            [replace(columns_time_ordered[int(index)], index=int(index)) for index in ordering],
            ordering,
            "metadata time order",
            "metadata.ordered_column_index",
        )
    if order_key in {"deadline_reorder", "fwd_deadline"}:
        reordered_columns, ordering = progressive.optimize_column_order(
            list(columns_time_ordered),
            num_detectors=int(num_detectors),
        )
        reordered = [
            replace(reordered_columns[int(target_index)], index=int(target_index))
            for target_index in range(len(reordered_columns))
        ]
        return (
            reordered,
            tuple(int(index) for index in ordering),
            "deadline_reorder",
            "metadata.ordered_column_index followed by optimize_column_order",
        )
    raise ValueError(
        f"unsupported column_order {column_order!r}; supported minimal orders are {list(SUPPORTED_COLUMN_ORDERS)}"
    )


def load_dem_family(
    *,
    backend: str,
    p_location: float,
    scope: str,
    column_order: str = "deadline_reorder",
    stim_path: Path | None = None,
    external_benchmark_label: str | None = None,
    external_noisy_rounds: int | None = None,
    external_perfect_rounds: int = 1,
) -> LoadedProgressiveFamily:
    """Load one detector-side DEM sector as frontier columns.

    This is intentionally limited to the public decoder path: binary frontier
    columns, time/deadline ordering, and the derived backward deadline order.
    """

    order_key = str(column_order).strip().lower()
    if order_key in {"backward_deadline_reorder", "bwd_deadline"}:
        base = load_dem_family(
            backend=str(backend),
            p_location=float(p_location),
            scope=str(scope),
            column_order="deadline_reorder",
            stim_path=stim_path,
            external_benchmark_label=external_benchmark_label,
            external_noisy_rounds=external_noisy_rounds,
            external_perfect_rounds=int(external_perfect_rounds),
        )
        return build_backward_deadline_ordered_family(base_family=base)

    problem = None
    loaded_side = None
    if stim_path is not None:
        if external_noisy_rounds is None or int(external_noisy_rounds) <= 0:
            raise ValueError("external Stim loading requires external_noisy_rounds > 0")
        loaded_side = load_dem_side_with_metadata_from_stim(
            stim_path=Path(stim_path),
            backend=str(backend),
            sector=("X" if str(scope) == "memory_X" else "Z"),
            error_rate=float(p_location),
            noisy_rounds=int(external_noisy_rounds),
            perfect_rounds=int(external_perfect_rounds),
        )
        benchmark_title, benchmark_description, benchmark_source_note = _external_benchmark_descriptor(
            backend=str(backend),
            benchmark_label=(
                str(external_benchmark_label)
                if external_benchmark_label is not None and str(external_benchmark_label).strip()
                else f"{str(backend)} external Stim DEM"
            ),
            stim_path=Path(stim_path),
        )
    else:
        problem = build_split_sector_problem(
            backend=str(backend),
            error_rate=float(p_location),
        )
        benchmark_title, benchmark_description = _benchmark_descriptor(str(backend))
        benchmark_source_note = (
            f"{benchmark_description} from `grosscode.dem.builder.build_split_sector_problem(backend={backend!r}, "
            f"error_rate={float(p_location):.6g})`."
        )

    scope_key = str(scope)
    if scope_key == "memory_X":
        if stim_path is not None:
            assert loaded_side is not None
            detector = loaded_side.check_matrix.tocsc()
            logical = loaded_side.observables_matrix.tocsc()
            priors = np.asarray(loaded_side.priors, dtype=np.float64).reshape(-1)
            metadata = loaded_side.metadata
        else:
            assert problem is not None
            detector = problem.D_X.tocsc()
            logical = problem.O_X.tocsc()
            priors = np.asarray(problem.priors_X, dtype=np.float64).reshape(-1)
            metadata = problem.metadata_X
        family_key = "binary_dem_x"
        scope_label = "X side"
        detector_symbol = "D_X"
        logical_symbol = "O_X"
        metadata_symbol = "metadata_X"
        priors_symbol = "priors_X"
    elif scope_key == "memory_Z":
        if stim_path is not None:
            assert loaded_side is not None
            detector = loaded_side.check_matrix.tocsc()
            logical = loaded_side.observables_matrix.tocsc()
            priors = np.asarray(loaded_side.priors, dtype=np.float64).reshape(-1)
            metadata = loaded_side.metadata
        else:
            assert problem is not None
            detector = problem.D_Z.tocsc()
            logical = problem.O_Z.tocsc()
            priors = np.asarray(problem.priors_Z, dtype=np.float64).reshape(-1)
            metadata = problem.metadata_Z
        family_key = "binary_dem_z"
        scope_label = "Z side"
        detector_symbol = "D_Z"
        logical_symbol = "O_Z"
        metadata_symbol = "metadata_Z"
        priors_symbol = "priors_Z"
    else:
        raise ValueError("scope must be one of memory_X or memory_Z")

    ordered_columns = np.asarray(metadata.ordered_column_index, dtype=np.int32)
    columns_time_ordered: list[progressive.ProgressiveColumn] = []
    for order_index, source_column in enumerate(ordered_columns.tolist()):
        q = float(priors[int(source_column)])
        det_start = int(detector.indptr[int(source_column)])
        det_stop = int(detector.indptr[int(source_column) + 1])
        det_rows = detector.indices[det_start:det_stop].astype(np.int32, copy=False)
        log_start = int(logical.indptr[int(source_column)])
        log_stop = int(logical.indptr[int(source_column) + 1])
        log_rows = logical.indices[log_start:log_stop].astype(np.int32, copy=False)
        detector_mask = _bitmask_from_indices(det_rows.tolist())
        logical_mask = _bitmask_from_indices(log_rows.tolist())
        prior_probs = (float(1.0 - q), float(q))
        round_start = int(metadata.column_round_start[int(source_column)])
        round_stop = int(metadata.column_round_stop[int(source_column)])
        columns_time_ordered.append(
            progressive.ProgressiveColumn(
                family=str(family_key),
                index=int(order_index),
                label=f"r{round_start}_to_r{round_stop}",
                instruction_offset=int(order_index),
                prior_probs=prior_probs,
                detector_response_masks=(0, int(detector_mask)),
                logical_response_masks=(0, int(logical_mask)),
                detector_support_mask=int(detector_mask),
                prior_log_probs=_log_probs_from_probs(prior_probs),
                detector_support_rows=tuple(int(value) for value in det_rows.tolist()),
                correction_response_masks=None,
                original_column_index=int(source_column),
            )
        )

    columns, _ordering, column_order_name, column_order_source = _ordered_columns_by_mode(
        columns_time_ordered,
        num_detectors=int(detector.shape[0]),
        column_order=str(column_order),
    )
    layout = progressive.build_frontier_layout(columns, num_detectors=int(detector.shape[0]))
    return LoadedProgressiveFamily(
        backend=str(backend),
        family_key=str(family_key),
        scope=str(scope_key),
        scope_label=str(scope_label),
        benchmark_title=str(benchmark_title),
        benchmark_description=str(benchmark_description),
        benchmark_source_note=str(benchmark_source_note),
        detector_symbol=str(detector_symbol),
        logical_symbol=str(logical_symbol),
        metadata_symbol=str(metadata_symbol),
        priors_symbol=str(priors_symbol),
        column_order_name=str(column_order_name),
        column_order_source=str(column_order_source).replace("metadata.", f"{metadata_symbol}."),
        model_label=f"{benchmark_description}, {scope_label.lower()} only",
        decode_label=f"binary DEM progressive ({scope_label.lower()})",
        columns=tuple(columns),
        layout=layout,
        matrix_rows=int(detector.shape[0]),
        matrix_cols=int(detector.shape[1]),
        logical_rows=int(logical.shape[0]),
        edge_count=int(detector.nnz),
        noisy_rounds=int(metadata.noisy_rounds),
        total_rounds=int(metadata.total_rounds),
    )


def build_backward_deadline_ordered_family(*, base_family: LoadedProgressiveFamily) -> LoadedProgressiveFamily:
    reversed_columns = progressive._reverse_progressive_columns(base_family.columns)
    reordered_columns, _ordering = progressive.optimize_column_order(
        list(reversed_columns),
        num_detectors=int(base_family.matrix_rows),
    )
    reordered = tuple(
        replace(reordered_columns[int(target_index)], index=int(target_index))
        for target_index in range(len(reordered_columns))
    )
    backward_layout = progressive.build_frontier_layout(
        list(reordered),
        num_detectors=int(base_family.matrix_rows),
    )
    return replace(
        base_family,
        columns=reordered,
        layout=backward_layout,
        column_order_name=f"backward deadline reorder (anchor={str(base_family.column_order_name)})",
        column_order_source=(
            f"reverse of {str(base_family.column_order_name)}, then optimize_column_order on the reversed family"
        ),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load supported DEM matrices and print frontier dimensions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""example:
  python -m tools.dem_loader --backend rotated_surface_d3 --p-location 0.001 --column-order deadline_reorder

See docs/COMMANDS.md for command details.""",
    )
    parser.add_argument("--backend", default="bravyi_depth7")
    parser.add_argument("--p-location", type=float, default=0.004)
    parser.add_argument(
        "--scope",
        action="append",
        choices=("memory_X", "memory_Z"),
        help="Scope to inspect. Defaults to both memory_X and memory_Z.",
    )
    parser.add_argument("--column-order", choices=SUPPORTED_COLUMN_ORDERS, default="deadline_reorder")
    parser.add_argument("--stim-path", type=Path)
    parser.add_argument("--external-benchmark-label")
    parser.add_argument("--external-noisy-rounds", type=int)
    parser.add_argument("--external-perfect-rounds", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    scopes = tuple(args.scope) if args.scope else ("memory_X", "memory_Z")
    rows: list[tuple[str, LoadedProgressiveFamily]] = []
    try:
        for scope in scopes:
            rows.append(
                (
                    str(scope),
                    load_dem_family(
                        backend=str(args.backend),
                        p_location=float(args.p_location),
                        scope=str(scope),
                        column_order=str(args.column_order),
                        stim_path=args.stim_path,
                        external_benchmark_label=args.external_benchmark_label,
                        external_noisy_rounds=args.external_noisy_rounds,
                        external_perfect_rounds=int(args.external_perfect_rounds),
                    ),
                )
            )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("scope,detector_matrix,logical_matrix,columns,edges,noisy_rounds,total_rounds,column_order")
    for scope, family in rows:
        print(
            ",".join(
                (
                    str(scope),
                    f"{int(family.matrix_rows)}x{int(family.matrix_cols)}",
                    f"{int(family.logical_rows)}x{int(family.matrix_cols)}",
                    str(len(family.columns)),
                    str(int(family.edge_count)),
                    str(int(family.noisy_rounds)),
                    str(int(family.total_rounds)),
                    str(family.column_order_name),
                )
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
