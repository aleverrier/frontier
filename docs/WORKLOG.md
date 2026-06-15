# Frontier Worklog

## 2026-06-15

- Created the initial `frontier` export.
- Selected repo shape 3 and decoder mode 3: native C++ decoder plus DEM replay/benchmark CLI, with forward `deadline_reorder` and backward `backward_deadline_reorder`.
- Public modes are limited to `forward_only`, `backward_only`, and `fwd_bwd_committee`.
- Validation completed:
  - `python setup.py build_ext --inplace`
  - `python -m py_compile frontier_native.py tools/dem_loader.py tools/frontier_decoder.py tools/frontier_sample_replay.py tools/frontier_bb144_benchmark.py tools/steane_progressive_decoder.py tests/test_frontier_export.py`
  - `python -m pytest -q` (`3 passed`)
  - `python -m tools.frontier_decoder --K 16 --Delta 100 --shots 3`

## Open Items

- Pushed to GitHub: `git@github.com:aleverrier/frontier.git`, branch `main`; first published commit was `3b45933`.
- Continue to keep the standalone export minimal: BB/Gross and surface-code matrix builders, DEM loading, the frontier decoder, replay, benchmark, and tests.

## 2026-06-15 Naming and Matrix Docs Cleanup

- Renamed the public exported modules and extension to `frontier_*`, including `_frontier_native`, `frontier_native.py`, `tools/frontier_decoder.py`, `tools/frontier_sample_replay.py`, and `tools/frontier_bb144_benchmark.py`.
- Renamed public Python API types/functions to `Frontier*` / `decode_frontier*`.
- Added README matrix-availability notes: static matrix files are not checked in for generated surface-code DEMs; in-repo constructors cover surface-code checks and generated rotated-surface DEMs, while the accepted Gross split-sector DEM builder uses the bundled Gross-code matrix/circuit assets.

## 2026-06-15 Public Path Cleanup

- Removed private checkout examples and hardcoded local filesystem defaults from README text, worklogs, and exported Python helpers.
- Changed optional Gross/Tanner asset handling to require explicit public asset configuration instead of silently falling back to a local developer path.
- Renamed temporary cache defaults to `frontier` names.

## 2026-06-15 Repo Scope Cleanup

- Audited the standalone export file-by-file for the intended public surface: frontier on BB/Gross and surface-code detector-side matrices.
- Removed the legacy `grosscode/decoders/**` package, including full BP/min-sum, windowed BP/min-sum, local-round, triangle-quotient, triangle small-set-flip, and structure-aware decoder families.
- Removed old research-only support trees: `grosscode/bench/**`, `grosscode/polar_dem/**`, projected-location/min-sum helpers, triangle-basis/reference-recovery helpers, Tanner redundant extraction, nonbinary CNOT/quaternary BP helpers, and legacy model/baseline modules.
- Kept the frontier/native path, BB/Gross/generalized-bicycle/rotated-surface/surface matrix builders, and split-sector DEM builder.
- Added `docs/FILE_SCOPE.md` as the file-by-file retained-scope audit.

## 2026-06-15 BB144/Gross Reproducibility Docs

- Added README instructions for reproducing BB144/Gross split-sector DEM results at a chosen `p = --p-location`.
- Documented matrix inspection via `frontier-dem-info` and exact matched full-frame replay via `frontier-replay` plus a saved `sample_rows.csv`.
- Clarified that exact published full-frame rows require the same matched sample-row corpus; large sample corpora are not checked into this repo.
- Made `frontier-bb144-benchmark --help` side-effect-free by deferring Matplotlib cache directory creation until after argument parsing.

## 2026-06-15 Minimal Repo Tightening

- Replaced the archived BB144/Gross report driver with `tools/dem_loader.py`, a small loader and `frontier-dem-info` CLI for supported BB/Gross and surface-code detector matrices.
- Removed archived report, triangle-ordering, and Stim fault-analysis helper files that were not required by the public decoder path.
- Removed physically dead native C++ code for non-public staged decoding and removed staged replay CSV fields.
- Made `frontier-bb144-benchmark` require an explicit `--sample-rows` file instead of defaulting to a local `results/...` path.
- Renamed the retained prune-block helper to `tools/frontier_prune_blocks.py`.
- Validation completed after tightening:
  - `PYTHONPYCACHEPREFIX=/tmp/frontier_pycache python -m py_compile frontier_native.py tools/dem_loader.py tools/frontier_decoder.py tools/frontier_sample_replay.py tools/frontier_bb144_benchmark.py tools/steane_progressive_decoder.py tools/frontier_prune_blocks.py tests/test_frontier_export.py`
  - `python setup.py build_ext --inplace`
  - `python -m pytest -q -p no:cacheprovider` (`3 passed`)
  - `python -m tools.frontier_decoder --K 16 --Delta 100 --shots 3`
  - `python -m tools.dem_loader --backend rotated_surface_d3 --p-location 0.001 --column-order deadline_reorder`
  - `python -m tools.dem_loader --help`
  - `python -m tools.frontier_bb144_benchmark --help`

## 2026-06-15 Fresh-Clone Reproducibility Fixes

- Audited a clean clone of `git@github.com:aleverrier/frontier.git` as a new user.
- Fixed the clean Python 3.14 install instructions by installing `setuptools` and `wheel` before `python setup.py build_ext --inplace`.
- Replaced public Gross asset configuration with bundled `grosscode/assets/gross144` files plus an optional `GROSSCODE_ASSET_ROOT` override, and removed the old internal asset-root environment wording from code and docs.
- Bundled the Gross `[[144,12,12]]` CSS matrices, BB144/Gross memory X/Z Stim circuits, and materialized `bravyi_depth7`, `p=0.001` split-sector DEM sparse matrices/priors.
- Added `frontier-sample-rows`, a public CLI that generates `sample_rows.csv` from the same detector-side DEM matrices and priors used by `frontier-replay`.
- Documented the complete fresh BB144/Gross DEM reproduction flow at `p=0.001`, `Delta=12`, `K=512`, and `fwd_bwd_committee`: generate 10k matched sample rows, then replay them with the native engine.
- Made `frontier-dem-info` load all requested matrices before printing its CSV header, so missing assets no longer produce a partial CSV.
- Removed replay summary warnings when pressure diagnostics are disabled.
- Validation completed:
  - `PYTHONPYCACHEPREFIX=/tmp/frontier_pycache /tmp/frontier-fresh-SDHYq1/frontier/.venv/bin/python -m py_compile ...`
  - `PYTHONPYCACHEPREFIX=/tmp/frontier_pycache /tmp/frontier-fresh-SDHYq1/frontier/.venv/bin/python -m pytest -q -p no:cacheprovider` (`6 passed`)
  - `python setup.py build_ext --inplace` using the fresh-clone venv interpreter
  - `python -m tools.frontier_decoder --K 16 --Delta 100 --shots 3`
  - `python -m tools.dem_loader --backend rotated_surface_d3 --p-location 0.001 --column-order deadline_reorder`
  - `GROSSCODE_ASSET_ROOT=/tmp/missing python -m tools.dem_loader --backend bravyi_depth7 --p-location 0.001 --column-order deadline_reorder` now prints a one-line missing-assets error and no CSV header
  - `python -m tools.dem_loader --backend bravyi_depth7 --p-location 0.001 --column-order deadline_reorder` reports `936x8784` detector matrices for both memory sectors from bundled assets
  - generated and replayed sample rows for `rotated_surface_d3` and a 10-shot BB144/Gross `p=0.001`, `Delta=12`, `K=512`, `fwd_bwd_committee` native replay
