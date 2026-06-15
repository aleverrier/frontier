from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
BUNDLED_GROSS_ASSET_ROOT = REPO_ROOT / "grosscode" / "assets" / "gross144"


def resolve_cache_root(root: str | Path | None = None, *, app_name: str = "frontier") -> Path:
    if root is not None:
        path = Path(root).expanduser()
    else:
        override = os.environ.get("FRONTIER_CACHE_DIR")
        if override:
            path = Path(override).expanduser()
        else:
            xdg_root = os.environ.get("XDG_CACHE_HOME")
            if xdg_root:
                path = Path(xdg_root).expanduser() / str(app_name)
            elif sys.platform == "darwin":
                path = Path.home() / "Library" / "Caches" / str(app_name)
            else:
                path = Path.home() / ".cache" / str(app_name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_gross_asset_root(root: str | Path | None = None) -> Path:
    configured_root = root or os.environ.get("GROSSCODE_ASSET_ROOT")
    path = Path(configured_root).expanduser() if configured_root is not None else BUNDLED_GROSS_ASSET_ROOT
    if not path.exists():
        if configured_root is None:
            raise FileNotFoundError(
                "Gross-code matrix/circuit assets are not available. The bundled asset directory "
                f"is missing: {path}. Restore the bundled files or set GROSSCODE_ASSET_ROOT to a "
                "custom asset root."
            )
        raise FileNotFoundError(
            f"Gross-code matrix/circuit asset root not found: {path}. "
            "Set GROSSCODE_ASSET_ROOT to a directory containing gross_code/ and stim_circuits/."
        )
    return path


def ensure_mplconfigdir() -> Path:
    mpl_dir = Path(tempfile.gettempdir()) / "frontier_mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    return mpl_dir
