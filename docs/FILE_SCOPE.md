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
| `tools/dem_loader.py` | Minimal DEM-to-frontier loader and `frontier-dem-info` CLI for BB/Gross and surface-code detector matrices. |
| `tools/frontier_sample_rows.py` | DEM sample-row generator for `frontier-replay`, covering BB/Gross and surface-code detector matrices. |
| `tools/frontier_sample_replay.py` | Matched sample replay CLI for BB144/Gross and related DEM rows. |
| `tools/frontier_bb144_benchmark.py` | Focused BB144/Gross native timing probe over explicit sample rows. |
| `tools/frontier_progressive.py` | Minimal frontier column/layout/order helpers used by the public wrapper and DEM loader. |
| `tests/test_frontier_export.py` | Regression coverage for the exported frontier wrapper/replay behavior. |
| `grosscode/__init__.py` | Small top-level export for split-sector DEM construction. |
| `grosscode/core.py` | Shared sparse-matrix and probability helpers used by retained DEM helpers. |
| `grosscode/codes/__init__.py` | Public code-builder exports. |
| `grosscode/codes/css.py` | Generic CSS-code container and validation helpers. |
| `grosscode/codes/surface.py` | Planar surface-code CSS check construction. |
| `grosscode/codes/rotated_surface.py` | Rotated surface-code CSS and Stim memory-circuit construction. |
| `grosscode/codes/gross144.py` | Accepted Gross/BB144 CSS matrix loader. |
| `grosscode/codes/gross_144_12_12.py` | Compatibility wrapper for loading the Gross `[[144,12,12]]` code. |
| `grosscode/codes/generalized_bicycle.py` | Generalized-bicycle backend generation for supported BB-style benchmarks. |
| `grosscode/assets/gross144/dem/*` | Bundled BB144/Gross `p=0.001` split-sector DEM detector/logical sparse matrices and priors. |
| `grosscode/assets/gross144/gross_code/*` | Bundled Gross `[[144,12,12]]` CSS parity-check matrices required by the default benchmark. |
| `grosscode/assets/gross144/stim_circuits/*` | Bundled BB144/Gross public memory X/Z Stim circuits for supported `bravyi_depth7` rates. |
| `grosscode/circuits/__init__.py` | Public circuit-resolution exports. |
| `grosscode/circuits/backends.py` | Backend-to-Stim circuit resolution for Gross, generalized-bicycle, and rotated-surface families. |
| `grosscode/dem/__init__.py` | Public split-sector DEM builder exports. |
| `grosscode/dem/builder.py` | Detector-side DEM matrix builder used by replay/report/benchmark paths. |
| `grosscode/utils/__init__.py` | Public utility exports. |
| `grosscode/utils/gf2.py` | GF(2) sparse/dense linear algebra used by code and DEM builders. |
| `grosscode/utils/paths.py` | Repo/cache path and bundled/custom Gross asset resolution. |

Removed categories:

- `grosscode/decoders/**`: legacy BP, min-sum, windowed, local-round,
  triangle-quotient, and structure-aware decoder families.
- `grosscode/bench/**`: older comparison and matched-benchmark harnesses that
  exercised the removed decoder families.
- `grosscode/polar_dem/**`: independent polar-transform DEM experiments.
- Legacy model/baseline/nonbinary helpers and triangle-basis/projected-location
  tooling that supported those removed experiments.
- Archived Gross report scripts, triangle-relation order experiments, and
  Stim fault-analysis helpers that are not required by `frontier-dem-info`,
  `frontier-replay`, or `frontier-bb144-benchmark`.
- Non-self-contained external BB circuit generation. The retained BB path is
  the bundled BB144/Gross split-sector DEM benchmark and its bundled Stim/CSS
  assets.
