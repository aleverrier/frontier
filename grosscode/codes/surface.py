from __future__ import annotations

from functools import lru_cache

import numpy as np
import scipy.sparse as sp

from grosscode.codes.css import CSSCode, derive_css_code_from_checks


def _repetition_check(distance: int) -> sp.csr_matrix:
    d = int(distance)
    rows = np.repeat(np.arange(d - 1, dtype=np.int32), 2)
    cols = np.empty(2 * (d - 1), dtype=np.int32)
    cols[0::2] = np.arange(d - 1, dtype=np.int32)
    cols[1::2] = np.arange(1, d, dtype=np.int32)
    data = np.ones(2 * (d - 1), dtype=np.uint8)
    return sp.csr_matrix((data, (rows, cols)), shape=(d - 1, d), dtype=np.uint8)


def standard_surface_checks(distance: int) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Return the CSS checks for the standard planar surface code.

    This is the hypergraph product of two length-``distance`` repetition-code
    checks, with parameters ``[[d^2 + (d-1)^2, 1, d]]``.
    """

    d = int(distance)
    if d < 3:
        raise ValueError("standard surface code requires distance >= 3")
    h = _repetition_check(d)
    m, n = h.shape
    hx = sp.hstack(
        [
            sp.kron(h, sp.identity(n, dtype=np.uint8, format="csr"), format="csr"),
            sp.kron(sp.identity(m, dtype=np.uint8, format="csr"), h.T, format="csr"),
        ],
        format="csr",
        dtype=np.uint8,
    )
    hz = sp.hstack(
        [
            sp.kron(sp.identity(n, dtype=np.uint8, format="csr"), h, format="csr"),
            sp.kron(h.T, sp.identity(m, dtype=np.uint8, format="csr"), format="csr"),
        ],
        format="csr",
        dtype=np.uint8,
    )
    return hx, hz


@lru_cache(maxsize=16)
def _load_standard_surface_code_cached(distance: int) -> CSSCode:
    d = int(distance)
    hx, hz = standard_surface_checks(d)
    n = d * d + (d - 1) * (d - 1)
    return derive_css_code_from_checks(
        name=f"standard planar surface [[{n},{1},{d}]]",
        hx=hx,
        hz=hz,
        metadata={
            "source": "local standard planar surface constructor as HGP(rep_d, rep_d)",
            "distance_hint": d,
            "surface_family": "standard",
        },
    )


def load_standard_surface_code(*, distance: int) -> CSSCode:
    d = int(distance)
    if d < 3 or d % 2 == 0:
        raise ValueError("standard surface code loader currently expects odd distance >= 3")
    return _load_standard_surface_code_cached(d)


__all__ = [
    "load_standard_surface_code",
    "standard_surface_checks",
]
