from __future__ import annotations

from pathlib import Path

from grosscode.codes.css import CSSCode
from grosscode.codes.gross144 import load_gross144_code


def load_gross_144_12_12(asset_root: str | Path | None = None) -> CSSCode:
    """Compatibility wrapper for the Gross [[144,12,12]] code loader."""
    return load_gross144_code(asset_root=asset_root)


__all__ = ["CSSCode", "load_gross_144_12_12"]
