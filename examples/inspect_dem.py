#!/usr/bin/env python3
"""Inspect the small rotated-surface DEM family."""

from __future__ import annotations

from frontier.dem import load_dem_family


def main() -> int:
    for scope in ("memory_X", "memory_Z"):
        family = load_dem_family(
            backend="rotated_surface_d3",
            p_location=0.001,
            scope=scope,
            column_order="deadline_reorder",
        )
        print(
            f"{scope}: detector={family.matrix_rows}x{family.matrix_cols} "
            f"logical={family.logical_rows}x{family.matrix_cols} "
            f"columns={len(family.columns)} order={family.column_order_name} "
            f"source={family.column_order_source}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
