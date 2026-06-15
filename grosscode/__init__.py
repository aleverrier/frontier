"""Gross-code DEM helpers vendored for the Frontier export.

This package intentionally keeps its top-level import small. Import concrete
submodules such as ``grosscode.dem.builder`` directly when needed.
"""

from grosscode.dem.builder import SplitSectorProblem, build_split_sector_problem

__all__ = [
    "SplitSectorProblem",
    "build_split_sector_problem",
]
