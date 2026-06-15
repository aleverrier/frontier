# Frontier File Scope

This repo is intentionally scoped to the C++-accelerated frontier decoder and
the BB/Gross plus surface-code matrix builders needed to exercise it. Files not
serving that surface were removed.

| File | Why it remains |
| --- | --- |
| `README.md` | Public install, smoke-test, replay, benchmark, matrix, and reproduction instructions. |
| `docs/WORKLOG.md` | Agent-readable change log for repo maintenance. |
| `docs/WORKLOG.tex` | Human-readable TeX change log matching `docs/WORKLOG.md`. |
| `docs/FILE_SCOPE.md` | This audit of the retained file set. |
| `pyproject.toml` | Package metadata, dependencies, console scripts, and pytest config. |
| `setup.py` | Native C++ extension build definition for `_frontier_native`. |
| `frontier_native.py` | Python wrapper around the compiled native extension. |
| `native/_frontier_native.cpp` | C++ frontier engine used by the public decoder path. |
| `tools/__init__.py` | Lightweight package marker for CLI/support modules. |
| `tools/frontier_decoder.py` | Public Python frontier API and `frontier-smoke` CLI. |
| `tools/frontier_sample_replay.py` | Matched sample replay CLI for BB144/Gross and related DEM rows. |
| `tools/frontier_bb144_benchmark.py` | Focused BB144/Gross native timing probe. |
| `tools/gross144_dem_x_progressive_report.py` | Fresh side-level BB144/Gross Monte Carlo/report CLI used by the README reproduction path. |
| `tools/frontierk_prune_blocks.py` | Small support type used by `tools/steane_progressive_decoder.py`. |
| `tools/steane_progressive_decoder.py` | Frontier recurrence implementation and ordering utilities used by the public wrapper and reports. |
| `tests/test_frontier_export.py` | Regression coverage for the exported frontier wrapper/replay behavior. |
| `grosscode/__init__.py` | Small top-level export for split-sector DEM construction. |
| `grosscode/core.py` | Shared sparse-matrix and probability helpers used by retained DEM helpers. |
| `grosscode/codes/__init__.py` | Public code-builder exports. |
| `grosscode/codes/css.py` | Generic CSS-code container and validation helpers. |
| `grosscode/codes/surface.py` | Planar surface-code CSS check construction. |
| `grosscode/codes/rotated_surface.py` | Rotated surface-code CSS and Stim memory-circuit construction. |
| `grosscode/codes/gross144.py` | Accepted Gross/BB144 CSS matrix loader. |
| `grosscode/codes/gross_144_12_12.py` | Compatibility wrapper for loading the Gross `[[144,12,12]]` code. |
| `grosscode/codes/bivariate_bicycle.py` | BB-code backend generation through the sliding-window upstream constructor. |
| `grosscode/codes/generalized_bicycle.py` | Generalized-bicycle backend generation for supported BB-style benchmarks. |
| `grosscode/circuits/__init__.py` | Public circuit-resolution exports. |
| `grosscode/circuits/backends.py` | Backend-to-Stim circuit resolution for Gross, BB, generalized-bicycle, and rotated-surface families. |
| `grosscode/dem/__init__.py` | Public split-sector DEM builder exports. |
| `grosscode/dem/builder.py` | Detector-side DEM matrix builder used by replay/report/benchmark paths. |
| `grosscode/dem/stim_fault_pipeline.py` | Stim sampling and correction-map utilities used by the BB144/Gross report path. |
| `grosscode/dem/triangles.py` | Local triangle relation cataloging used for retained BB144/Gross ordering/report options. |
| `grosscode/utils/__init__.py` | Public utility exports. |
| `grosscode/utils/gf2.py` | GF(2) sparse/dense linear algebra used by code and DEM builders. |
| `grosscode/utils/paths.py` | Repo/cache path and optional public Gross asset resolution. |
| `sliding_window_baseline.py` | Minimal bridge used to generate bivariate-bicycle Stim circuits. |

Removed categories:

- `grosscode/decoders/**`: legacy BP, min-sum, windowed, local-round,
  triangle-quotient, and structure-aware decoder families.
- `grosscode/bench/**`: older comparison and matched-benchmark harnesses that
  exercised the removed decoder families.
- `grosscode/polar_dem/**`: independent polar-transform DEM experiments.
- Legacy model/baseline/nonbinary helpers and triangle-basis/projected-location
  tooling that supported those removed experiments.
