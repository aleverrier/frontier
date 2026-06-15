from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from scipy.io import mmread

from grosscode.codes.css import CSSCode, derive_css_code_from_checks
from grosscode.utils.paths import resolve_gross_asset_root


def _matrix_path(root: Path, which: str) -> Path:
    path = root / "gross_code" / f"{which}_Gross_144_12_12.mtx"
    if not path.exists():
        raise FileNotFoundError(f"public gross-code matrix missing: {path}")
    return path


@lru_cache(maxsize=4)
def _load_cached(root_text: str) -> CSSCode:
    root = Path(root_text)
    hx_path = _matrix_path(root, "HX")
    hz_path = _matrix_path(root, "HZ")
    hx = mmread(str(hx_path)).tocsr()
    hz = mmread(str(hz_path)).tocsr()
    return derive_css_code_from_checks(
        name="gross [[144,12,12]]",
        hx=hx,
        hz=hz,
        metadata={
            "source": "public Gross benchmark matrices",
            "hx_path": str(hx_path),
            "hz_path": str(hz_path),
            "distance_hint": 12,
        },
    )


def load_gross144_code(asset_root: str | Path | None = None) -> CSSCode:
    root = resolve_gross_asset_root(asset_root)
    return _load_cached(str(root))
