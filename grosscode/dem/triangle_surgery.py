from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from grosscode.decoders.structure_aware import SectorStructureModel, build_sector_structure_model
from grosscode.dem.builder import SplitSectorMetadata
from grosscode.dem.triangles import ExactTriangleRelation
from grosscode.utils.gf2 import binary_csr_mod2


@dataclass(frozen=True)
class HardTriangleSurgeryVariant:
    sector: str
    matrix: sp.csr_matrix
    observables: sp.csr_matrix
    priors: np.ndarray
    structure_model: SectorStructureModel
    transformed_relations: tuple[ExactTriangleRelation, ...]
    auxiliary_check_count: int
    composite_columns: np.ndarray


def _strip_columns(matrix: sp.spmatrix, columns: np.ndarray) -> sp.csr_matrix:
    out = binary_csr_mod2(matrix).tocsc(copy=True)
    unique_columns = np.unique(np.asarray(columns, dtype=np.int32))
    for col in unique_columns.tolist():
        start = int(out.indptr[int(col)])
        stop = int(out.indptr[int(col) + 1])
        out.data[start:stop] = 0
    return binary_csr_mod2(out.tocsr())


def build_hard_triangle_surgery_variant(
    *,
    matrix: sp.spmatrix,
    observables: sp.spmatrix,
    priors: np.ndarray,
    metadata: SplitSectorMetadata,
    sector: str,
) -> HardTriangleSurgeryVariant:
    structure_model = build_sector_structure_model(
        matrix=binary_csr_mod2(matrix),
        observables=binary_csr_mod2(observables),
        metadata=metadata,
        sector=str(sector),
    )
    relations = tuple(structure_model.selection.selected_relations)
    composite_columns = np.asarray([int(relation.columns[2]) for relation in relations], dtype=np.int32)
    stripped_matrix = _strip_columns(matrix, composite_columns)
    stripped_observables = _strip_columns(observables, composite_columns)

    if relations:
        aux_rows = np.repeat(np.arange(len(relations), dtype=np.int32), 3)
        aux_cols = np.asarray([int(col) for relation in relations for col in relation.columns], dtype=np.int32)
        aux_data = np.ones(int(aux_rows.size), dtype=np.uint8)
        aux_matrix = sp.coo_matrix(
            (aux_data, (aux_rows, aux_cols)),
            shape=(len(relations), int(stripped_matrix.shape[1])),
            dtype=np.uint8,
        ).tocsr()
    else:
        aux_matrix = sp.csr_matrix((0, int(stripped_matrix.shape[1])), dtype=np.uint8)

    transformed_matrix = binary_csr_mod2(sp.vstack([stripped_matrix, aux_matrix], format="csr"))
    return HardTriangleSurgeryVariant(
        sector=str(sector),
        matrix=transformed_matrix,
        observables=stripped_observables,
        priors=np.asarray(priors, dtype=np.float64).copy(),
        structure_model=structure_model,
        transformed_relations=relations,
        auxiliary_check_count=int(aux_matrix.shape[0]),
        composite_columns=composite_columns,
    )
