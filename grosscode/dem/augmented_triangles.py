from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.sparse as sp

from grosscode.dem.builder import SplitSectorMetadata
from grosscode.dem.triangles import (
    ExactTriangleRelation,
    TriangleColumnMetadata,
    _column_metadata,
    _combined_signature_by_column,
)
from grosscode.utils.gf2 import binary_csr_mod2


AugmentedTriangleRelationKind = Literal[
    "same_round_split",
    "adjacent_round_mixed",
    "same_round_other",
    "adjacent_round_other",
    "nonlocal_other",
]

_AUGMENTED_TRIANGLE_CACHE: dict[tuple[object, ...], tuple[ExactTriangleRelation, ...]] = {}


def triangle_kind_priority(kind: str) -> int:
    order = {
        "adjacent_round_mixed": 0,
        "same_round_split": 1,
        "adjacent_round_other": 2,
        "same_round_other": 3,
        "nonlocal_other": 4,
    }
    return int(order.get(str(kind), 99))


def classify_augmented_triangle(
    column_meta: tuple[TriangleColumnMetadata, TriangleColumnMetadata, TriangleColumnMetadata],
) -> tuple[AugmentedTriangleRelationKind, int, int]:
    metas = tuple(column_meta)
    round_lo = min(int(meta.round_start) for meta in metas)
    round_hi = max(int(meta.round_stop) for meta in metas)
    bridge_count = sum(1 for meta in metas if int(meta.round_stop) > int(meta.round_start))
    within_count = 3 - int(bridge_count)
    fault_classes = tuple(sorted(str(meta.fault_class) for meta in metas))
    detector_weights = tuple(sorted(int(meta.detector_weight) for meta in metas))

    if (
        int(round_lo) == int(round_hi)
        and fault_classes == ("within_round", "within_round", "within_round")
        and detector_weights == (3, 3, 6)
    ):
        return ("same_round_split", int(round_lo), int(round_hi))
    if int(round_hi) == int(round_lo) + 1 and int(bridge_count) == 2 and int(within_count) == 1:
        return ("adjacent_round_mixed", int(round_lo), int(round_hi))
    if int(round_hi) == int(round_lo):
        return ("same_round_other", int(round_lo), int(round_hi))
    if int(round_hi) == int(round_lo) + 1:
        return ("adjacent_round_other", int(round_lo), int(round_hi))
    return ("nonlocal_other", int(round_lo), int(round_hi))


def _cache_key(
    *,
    matrix: sp.csr_matrix,
    observables: sp.csr_matrix,
    metadata: SplitSectorMetadata,
    sector: str,
) -> tuple[object, ...]:
    return (
        str(sector),
        str(metadata.stim_path),
        int(metadata.total_rounds),
        int(matrix.shape[0]),
        int(matrix.shape[1]),
        int(observables.shape[0]),
        int(observables.shape[1]),
        tuple(sorted((str(k), int(v)) for k, v in metadata.local_fault_class_counts.items())),
    )


def catalog_exact_augmented_triangles(
    *,
    matrix: sp.csr_matrix,
    observables: sp.csr_matrix,
    metadata: SplitSectorMetadata,
    sector: str,
) -> tuple[ExactTriangleRelation, ...]:
    key = _cache_key(matrix=matrix, observables=observables, metadata=metadata, sector=str(sector))
    cached = _AUGMENTED_TRIANGLE_CACHE.get(key)
    if cached is not None:
        return cached

    matrix = binary_csr_mod2(matrix).tocsr()
    observables = binary_csr_mod2(observables).tocsr()
    column_meta = _column_metadata(matrix=matrix, observables=observables, metadata=metadata, sector=str(sector))
    signatures = [int(value) for value in _combined_signature_by_column(matrix, observables)]
    signature_to_indices: dict[int, list[int]] = {}
    for col, signature in enumerate(signatures):
        signature_to_indices.setdefault(int(signature), []).append(int(col))

    relations: list[ExactTriangleRelation] = []
    for i in range(int(matrix.shape[1])):
        sig_i = int(signatures[i])
        for j in range(i + 1, int(matrix.shape[1])):
            target_signature = int(sig_i ^ int(signatures[j]))
            for k in signature_to_indices.get(int(target_signature), ()):
                if int(k) <= int(j):
                    continue
                ordered = (int(i), int(j), int(k))
                metas = (column_meta[ordered[0]], column_meta[ordered[1]], column_meta[ordered[2]])
                kind, round_lo, round_hi = classify_augmented_triangle(metas)
                relations.append(
                    ExactTriangleRelation(
                        sector=str(sector),
                        round_lo=int(round_lo),
                        round_hi=int(round_hi),
                        relation_kind=str(kind),  # type: ignore[arg-type]
                        columns=ordered,
                        column_metadata=metas,
                    )
                )
    relations.sort(
        key=lambda item: (
            int(item.round_lo),
            int(item.round_hi),
            triangle_kind_priority(str(item.relation_kind)),
            tuple(int(col) for col in item.columns),
        )
    )
    result = tuple(relations)
    _AUGMENTED_TRIANGLE_CACHE[key] = result
    return result


def relation_support_array(
    relations: tuple[ExactTriangleRelation, ...] | list[ExactTriangleRelation],
) -> np.ndarray:
    return np.asarray([tuple(int(col) for col in relation.columns) for relation in relations], dtype=np.int32)
