from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from grosscode.dem.augmented_triangles import relation_support_array, triangle_kind_priority
from grosscode.dem.triangles import ExactTriangleRelation
from grosscode.utils.gf2 import csr_matvec_mod2, dense_mod2, rank_dense_mod2


TriangleBasisStrategy = str


@dataclass(frozen=True)
class TriangleBasisArtifact:
    sector: str
    strategy: TriangleBasisStrategy
    detector_shape: tuple[int, int]
    logical_shape: tuple[int, int]
    augmented_rank: int
    kernel_dimension: int
    triangle_count: int
    triangle_supports: np.ndarray
    triangle_kinds: tuple[str, ...]
    basis_indices: np.ndarray
    basis_supports: np.ndarray
    catalog_participation_counts: np.ndarray
    basis_participation_counts: np.ndarray

    @property
    def basis_rank(self) -> int:
        return int(self.basis_indices.size)

    @property
    def n_cols(self) -> int:
        return int(self.detector_shape[1])

    @property
    def factor_edge_count(self) -> int:
        return int(self.basis_supports.size)

    @property
    def max_factor_degree(self) -> int:
        if int(self.basis_participation_counts.size) == 0:
            return 0
        return int(np.max(self.basis_participation_counts))

    def degree_histogram(self) -> dict[int, int]:
        if int(self.basis_participation_counts.size) == 0:
            return {}
        values, counts = np.unique(np.asarray(self.basis_participation_counts, dtype=np.int32), return_counts=True)
        return {int(v): int(c) for v, c in zip(values.tolist(), counts.tolist())}

    def summary_row(self) -> dict[str, object]:
        degrees = np.asarray(self.basis_participation_counts, dtype=np.int32)
        return {
            "strategy": str(self.strategy),
            "triangle_count": int(self.triangle_count),
            "basis_rank": int(self.basis_rank),
            "kernel_dimension": int(self.kernel_dimension),
            "max_basis_degree": int(np.max(degrees)) if degrees.size else 0,
            "mean_basis_degree": float(np.mean(degrees, dtype=np.float64)) if degrees.size else 0.0,
            "p90_basis_degree": float(np.quantile(degrees, 0.9)) if degrees.size else 0.0,
            "p99_basis_degree": float(np.quantile(degrees, 0.99)) if degrees.size else 0.0,
        }


def _support_to_mask(support: tuple[int, int, int] | np.ndarray) -> int:
    cols = tuple(int(col) for col in np.asarray(support, dtype=np.int64).reshape(-1).tolist())
    mask = 0
    for col in cols:
        mask ^= 1 << int(col)
    return int(mask)


def _insert_mask(mask: int, basis_by_pivot: dict[int, int]) -> bool:
    current = int(mask)
    while int(current) != 0:
        pivot = int(current.bit_length() - 1)
        existing = basis_by_pivot.get(int(pivot))
        if existing is None:
            basis_by_pivot[int(pivot)] = int(current)
            return True
        current ^= int(existing)
    return False


def rank_of_supports(supports: np.ndarray) -> int:
    basis_by_pivot: dict[int, int] = {}
    for support in np.asarray(supports, dtype=np.int32):
        _insert_mask(_support_to_mask(support), basis_by_pivot)
    return int(len(basis_by_pivot))


def compute_participation_counts(supports: np.ndarray, n_cols: int) -> np.ndarray:
    counts = np.zeros(int(n_cols), dtype=np.int32)
    for support in np.asarray(supports, dtype=np.int32):
        counts[np.asarray(support, dtype=np.int32)] += 1
    return counts


def _triangle_order(
    *,
    relations: tuple[ExactTriangleRelation, ...] | list[ExactTriangleRelation],
    strategy: TriangleBasisStrategy,
    catalog_participation_counts: np.ndarray,
) -> list[int]:
    relation_list = list(relations)
    if str(strategy) == "naive":
        return list(range(len(relation_list)))
    if str(strategy) != "low_participation":
        raise ValueError(f"unknown triangle-basis strategy: {strategy}")
    return sorted(
        range(len(relation_list)),
        key=lambda idx: (
            max(int(catalog_participation_counts[int(col)]) for col in relation_list[idx].columns),
            sum(int(catalog_participation_counts[int(col)]) for col in relation_list[idx].columns),
            triangle_kind_priority(str(relation_list[idx].relation_kind)),
            tuple(int(col) for col in relation_list[idx].columns),
        ),
    )


def build_triangle_basis_artifact(
    *,
    relations: tuple[ExactTriangleRelation, ...] | list[ExactTriangleRelation],
    detector_shape: tuple[int, int],
    logical_shape: tuple[int, int],
    augmented_rank: int,
    kernel_dimension: int,
    strategy: TriangleBasisStrategy,
    progress_every: int = 0,
) -> TriangleBasisArtifact:
    relation_tuple = tuple(relations)
    triangle_supports = relation_support_array(relation_tuple)
    triangle_count = int(triangle_supports.shape[0])
    if int(kernel_dimension) <= 0:
        raise ValueError("kernel_dimension must be positive")
    if int(triangle_count) < int(kernel_dimension):
        raise ValueError("triangle catalog is too small to span the requested kernel dimension")

    catalog_participation = compute_participation_counts(triangle_supports, int(detector_shape[1]))
    order = _triangle_order(
        relations=relation_tuple,
        strategy=str(strategy),
        catalog_participation_counts=catalog_participation,
    )
    basis_by_pivot: dict[int, int] = {}
    selected_indices: list[int] = []
    for position, rel_idx in enumerate(order, start=1):
        if _insert_mask(_support_to_mask(triangle_supports[int(rel_idx)]), basis_by_pivot):
            selected_indices.append(int(rel_idx))
            if len(selected_indices) >= int(kernel_dimension):
                break
        if int(progress_every) > 0 and position % int(progress_every) == 0:
            print(
                f"[triangle-basis] strategy={strategy} scanned={position}/{triangle_count} "
                f"rank={len(selected_indices)}/{kernel_dimension}",
                flush=True,
            )
    if int(len(selected_indices)) != int(kernel_dimension):
        raise RuntimeError(
            f"triangle basis selection failed for strategy={strategy}: rank={len(selected_indices)} "
            f"expected={kernel_dimension}"
        )
    basis_indices = np.asarray(selected_indices, dtype=np.int32)
    basis_supports = np.asarray(triangle_supports[basis_indices], dtype=np.int32)
    basis_participation = compute_participation_counts(basis_supports, int(detector_shape[1]))
    return TriangleBasisArtifact(
        sector=str(relation_tuple[0].sector if relation_tuple else "unknown"),
        strategy=str(strategy),
        detector_shape=(int(detector_shape[0]), int(detector_shape[1])),
        logical_shape=(int(logical_shape[0]), int(logical_shape[1])),
        augmented_rank=int(augmented_rank),
        kernel_dimension=int(kernel_dimension),
        triangle_count=int(triangle_count),
        triangle_supports=np.asarray(triangle_supports, dtype=np.int32),
        triangle_kinds=tuple(str(relation.relation_kind) for relation in relation_tuple),
        basis_indices=basis_indices,
        basis_supports=basis_supports,
        catalog_participation_counts=np.asarray(catalog_participation, dtype=np.int32),
        basis_participation_counts=np.asarray(basis_participation, dtype=np.int32),
    )


def verify_triangle_basis_artifact(
    *,
    artifact: TriangleBasisArtifact,
    detector_matrix: sp.csr_matrix,
    logical_matrix: sp.csr_matrix,
) -> dict[str, object]:
    det = detector_matrix.tocsr()
    log = logical_matrix.tocsr()
    selected_kernel_ok = True
    for support in np.asarray(artifact.basis_supports, dtype=np.int32):
        vec = np.zeros(int(det.shape[1]), dtype=np.uint8)
        vec[np.asarray(support, dtype=np.int32)] = 1
        if int(np.count_nonzero(csr_matvec_mod2(det, vec))) != 0 or int(np.count_nonzero(csr_matvec_mod2(log, vec))) != 0:
            selected_kernel_ok = False
            break
    basis_rank = rank_of_supports(np.asarray(artifact.basis_supports, dtype=np.int32))
    span_rank = rank_of_supports(np.asarray(artifact.triangle_supports, dtype=np.int32))
    return {
        "selected_kernel_ok": bool(selected_kernel_ok),
        "basis_rank": int(basis_rank),
        "span_rank": int(span_rank),
        "kernel_dimension": int(artifact.kernel_dimension),
        "basis_rank_ok": bool(int(basis_rank) == int(artifact.kernel_dimension)),
        "span_rank_ok": bool(int(span_rank) == int(artifact.kernel_dimension)),
    }


def save_triangle_basis_artifact(path: str | Path, artifact: TriangleBasisArtifact) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(
        {
            "sector": str(artifact.sector),
            "strategy": str(artifact.strategy),
            "detector_shape": [int(artifact.detector_shape[0]), int(artifact.detector_shape[1])],
            "logical_shape": [int(artifact.logical_shape[0]), int(artifact.logical_shape[1])],
            "augmented_rank": int(artifact.augmented_rank),
            "kernel_dimension": int(artifact.kernel_dimension),
            "triangle_count": int(artifact.triangle_count),
            "triangle_kinds": [str(kind) for kind in artifact.triangle_kinds],
        },
        sort_keys=True,
    )
    np.savez_compressed(
        out_path,
        triangle_supports=np.asarray(artifact.triangle_supports, dtype=np.int32),
        basis_indices=np.asarray(artifact.basis_indices, dtype=np.int32),
        basis_supports=np.asarray(artifact.basis_supports, dtype=np.int32),
        catalog_participation_counts=np.asarray(artifact.catalog_participation_counts, dtype=np.int32),
        basis_participation_counts=np.asarray(artifact.basis_participation_counts, dtype=np.int32),
        metadata_json=np.asarray(metadata_json),
    )
    return out_path


def load_triangle_basis_artifact(path: str | Path) -> TriangleBasisArtifact:
    raw = np.load(Path(path), allow_pickle=False)
    metadata = json.loads(str(np.asarray(raw["metadata_json"]).item()))
    return TriangleBasisArtifact(
        sector=str(metadata["sector"]),
        strategy=str(metadata["strategy"]),
        detector_shape=(int(metadata["detector_shape"][0]), int(metadata["detector_shape"][1])),
        logical_shape=(int(metadata["logical_shape"][0]), int(metadata["logical_shape"][1])),
        augmented_rank=int(metadata["augmented_rank"]),
        kernel_dimension=int(metadata["kernel_dimension"]),
        triangle_count=int(metadata["triangle_count"]),
        triangle_supports=np.asarray(raw["triangle_supports"], dtype=np.int32),
        triangle_kinds=tuple(str(item) for item in metadata["triangle_kinds"]),
        basis_indices=np.asarray(raw["basis_indices"], dtype=np.int32),
        basis_supports=np.asarray(raw["basis_supports"], dtype=np.int32),
        catalog_participation_counts=np.asarray(raw["catalog_participation_counts"], dtype=np.int32),
        basis_participation_counts=np.asarray(raw["basis_participation_counts"], dtype=np.int32),
    )


def augmented_nullity(detector_matrix: sp.csr_matrix, logical_matrix: sp.csr_matrix) -> tuple[int, int]:
    augmented = sp.vstack([detector_matrix.tocsr(), logical_matrix.tocsr()], format="csr")
    rank = rank_dense_mod2(np.asarray(augmented.toarray(), dtype=np.uint8))
    return int(rank), int(augmented.shape[1] - rank)


def basis_toggle_preserves_augmented_target(
    *,
    detector_matrix: sp.csr_matrix,
    logical_matrix: sp.csr_matrix,
    support: np.ndarray,
    reference: np.ndarray,
) -> bool:
    ref = dense_mod2(reference).reshape(-1)
    support_vec = np.zeros(int(detector_matrix.shape[1]), dtype=np.uint8)
    support_vec[np.asarray(support, dtype=np.int32)] = 1
    toggled = ref ^ support_vec
    return bool(
        np.array_equal(csr_matvec_mod2(detector_matrix, toggled), csr_matvec_mod2(detector_matrix, ref))
        and np.array_equal(csr_matvec_mod2(logical_matrix, toggled), csr_matvec_mod2(logical_matrix, ref))
    )
