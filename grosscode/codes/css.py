from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import scipy.sparse as sp

from grosscode.utils.gf2 import (
    binary_csr_mod2,
    dense_mod2,
    invert_dense_mod2,
    nullspace_basis_dense,
    rank_dense_mod2,
    select_independent_rows_mod2,
)


@dataclass(frozen=True)
class CSSCode:
    name: str
    HX: sp.csr_matrix
    HZ: sp.csr_matrix
    LX: np.ndarray
    LZ: np.ndarray
    n: int
    k: int
    metadata: Mapping[str, object]


def derive_css_code_from_checks(
    *,
    name: str,
    hx: sp.spmatrix,
    hz: sp.spmatrix,
    metadata: Mapping[str, object] | None = None,
) -> CSSCode:
    hx_csr = binary_csr_mod2(hx).tocsr()
    hz_csr = binary_csr_mod2(hz).tocsr()
    if hx_csr.shape[1] != hz_csr.shape[1]:
        raise ValueError("HX and HZ must act on the same number of qubits")
    hx_dense = dense_mod2(hx_csr.toarray())
    hz_dense = dense_mod2(hz_csr.toarray())
    overlap = dense_mod2(hx_dense @ hz_dense.T)
    if np.any(overlap):
        raise ValueError("HX and HZ do not commute over GF(2)")

    n = int(hx_csr.shape[1])
    k = int(n - rank_dense_mod2(hx_dense) - rank_dense_mod2(hz_dense))
    if k < 0:
        raise ValueError("derived negative encoded dimension")

    if k == 0:
        lx = np.zeros((0, n), dtype=np.uint8)
        lz = np.zeros((0, n), dtype=np.uint8)
    else:
        x_kernel = nullspace_basis_dense(hz_dense)
        z_kernel = nullspace_basis_dense(hx_dense)
        lx = select_independent_rows_mod2(x_kernel, seed_rows=hx_dense, max_rows=k)
        lz = select_independent_rows_mod2(z_kernel, seed_rows=hz_dense, max_rows=k)
        if lx.shape[0] != k or lz.shape[0] != k:
            raise ValueError("failed to derive full logical operator basis")
        symplectic = dense_mod2(lz @ lx.T)
        if rank_dense_mod2(symplectic) != k:
            raise ValueError("derived logical bases are not full-rank under CSS pairing")
        lz = dense_mod2(invert_dense_mod2(symplectic) @ lz)

    return CSSCode(
        name=name,
        HX=hx_csr,
        HZ=hz_csr,
        LX=lx.astype(np.uint8, copy=False),
        LZ=lz.astype(np.uint8, copy=False),
        n=n,
        k=k,
        metadata=dict(metadata or {}),
    )

