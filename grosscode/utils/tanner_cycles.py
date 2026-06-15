from __future__ import annotations

from collections import deque

import numpy as np
import scipy.sparse as sp

from grosscode.utils.gf2 import binary_csr_mod2


def bipartite_girth(matrix: sp.spmatrix) -> int | None:
    binary = binary_csr_mod2(matrix).tocsr()
    m, n = binary.shape
    row_neighbors = [
        binary.indices[binary.indptr[row] : binary.indptr[row + 1]].astype(np.int32, copy=False)
        for row in range(m)
    ]
    binary_csc = binary.tocsc()
    col_neighbors = [
        binary_csc.indices[binary_csc.indptr[col] : binary_csc.indptr[col + 1]].astype(np.int32, copy=False)
        for col in range(n)
    ]

    def _neighbors(node: int):
        if node < m:
            return (m + int(col) for col in row_neighbors[node])
        return (int(row) for row in col_neighbors[node - m])

    best: int | None = None
    for start in range(m):
        dist = {start: 0}
        parent = {start: -1}
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            depth = dist[node]
            if best is not None and 2 * depth + 2 >= best:
                continue
            for nxt in _neighbors(node):
                if nxt not in dist:
                    dist[nxt] = depth + 1
                    parent[nxt] = node
                    queue.append(nxt)
                    continue
                if parent[node] == nxt:
                    continue
                cycle_len = dist[node] + dist[nxt] + 1
                if cycle_len >= 4 and cycle_len % 2 == 0:
                    if best is None or cycle_len < best:
                        best = cycle_len
        if best == 4:
            break
    return best


def count_four_cycles(matrix: sp.spmatrix) -> int:
    binary = binary_csr_mod2(matrix).tocsr()
    overlap = (binary @ binary.T).tocsr()
    overlap.setdiag(0)
    overlap.eliminate_zeros()
    upper = sp.triu(overlap, k=1).tocoo()
    return int(np.sum(upper.data * (upper.data - 1) // 2, dtype=np.int64))


def count_six_cycles(matrix: sp.spmatrix) -> int:
    binary = binary_csr_mod2(matrix).tocsr()
    overlap = (binary @ binary.T).tocsr()
    overlap.setdiag(0)
    overlap.eliminate_zeros()
    upper = sp.triu(overlap, k=1).tocoo()

    pair_weight: dict[tuple[int, int], int] = {}
    neighbors: list[set[int]] = [set() for _ in range(int(binary.shape[0]))]
    for row_a, row_b, shared in zip(upper.row.tolist(), upper.col.tolist(), upper.data.tolist(), strict=True):
        key = (int(row_a), int(row_b))
        pair_weight[key] = int(shared)
        neighbors[int(row_a)].add(int(row_b))
        neighbors[int(row_b)].add(int(row_a))

    binary_csc = binary.tocsc()
    triple_common: dict[tuple[int, int, int], int] = {}
    for col in range(int(binary.shape[1])):
        rows = np.asarray(binary_csc.indices[binary_csc.indptr[col] : binary_csc.indptr[col + 1]], dtype=np.int32)
        if int(rows.size) < 3:
            continue
        rows = np.sort(rows)
        for idx_a in range(int(rows.size) - 2):
            row_a = int(rows[idx_a])
            for idx_b in range(idx_a + 1, int(rows.size) - 1):
                row_b = int(rows[idx_b])
                for idx_c in range(idx_b + 1, int(rows.size)):
                    row_c = int(rows[idx_c])
                    key = (row_a, row_b, row_c)
                    triple_common[key] = triple_common.get(key, 0) + 1

    total = 0
    for row_a in range(int(binary.shape[0])):
        for row_b in sorted(int(value) for value in neighbors[row_a] if int(value) > row_a):
            common_neighbors = neighbors[row_a].intersection(neighbors[row_b])
            for row_c in sorted(int(value) for value in common_neighbors if int(value) > row_b):
                ab = pair_weight[(int(row_a), int(row_b))]
                ac = pair_weight[(int(row_a), int(row_c))]
                bc = pair_weight[(int(row_b), int(row_c))]
                common_triplet = int(triple_common.get((int(row_a), int(row_b), int(row_c)), 0))
                contribution = int(ab * ac * bc - common_triplet * (ab + ac + bc) + 2 * common_triplet)
                if contribution < 0:
                    raise ValueError(
                        "negative 6-cycle contribution encountered; row-triple overlap bookkeeping is inconsistent"
                    )
                total += int(contribution)
    return int(total)


def small_cycle_summary(matrix: sp.spmatrix) -> dict[str, int | None]:
    return {
        "girth_bipartite": bipartite_girth(matrix),
        "four_cycle_count_exact": count_four_cycles(matrix),
        "six_cycle_count_exact": count_six_cycles(matrix),
    }
