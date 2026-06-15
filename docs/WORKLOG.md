# Frontier Worklog

## 2026-06-15

- Created the initial `frontier` export from the `better-beam` decoder code.
- Selected repo shape 3 and decoder mode 3: native C++ decoder plus DEM replay/benchmark CLI, with forward `deadline_reorder` and backward `backward_deadline_reorder`.
- Public modes are limited to `forward_only`, `backward_only`, and `fwd_bwd_committee`.
- The two-stage decoder is intentionally not exposed in the public CLI or native Python wrapper.
- Validation completed:
  - `/Users/anthony/research/better-beam/tools/py setup.py build_ext --inplace`
  - `/Users/anthony/research/better-beam/tools/py -m py_compile frontier_native.py tools/frontier_decoder.py tools/frontier_sample_replay.py tools/frontier_bb144_benchmark.py tools/gross144_dem_x_progressive_report.py tools/steane_progressive_decoder.py tests/test_frontier_export.py`
  - `/Users/anthony/research/better-beam/tools/py -m pytest -q` (`3 passed`)
  - `/Users/anthony/research/better-beam/tools/py -m tools.frontier_decoder --K 16 --Delta 100 --shots 3`

## Open Items

- Pushed to GitHub: `git@github.com:aleverrier/frontier.git`, branch `main`; first published commit was `3b45933`.
- Decide whether to later refactor the native C++ file to physically delete the unused internal stage1/stage2 implementation code.

## 2026-06-15 Naming and Matrix Docs Cleanup

- Renamed the public exported modules and extension to `frontier_*`, including `_frontier_native`, `frontier_native.py`, `tools/frontier_decoder.py`, `tools/frontier_sample_replay.py`, and `tools/frontier_bb144_benchmark.py`.
- Renamed public Python API types/functions to `Frontier*` / `decode_frontier*`.
- Added README matrix-availability notes: static matrix files are not checked in; in-repo constructors cover surface-code checks and generated rotated-surface DEMs, while the accepted Gross split-sector DEM builder needs a public `qtanner-ssf` checkout via `GROSSCODE_QTANNER_ROOT` or `QTANNER_ROOT`.

## 2026-06-15 BB144/Gross Reproducibility Docs

- Added README instructions for reproducing BB144/Gross split-sector DEM results at a chosen `p = --p-location`.
- Documented both workflows: fresh side-level Monte Carlo via `tools.gross144_dem_x_progressive_report`, and exact matched full-frame replay via `frontier-replay` plus a saved `sample_rows.csv`.
- Clarified that exact published full-frame rows require the same matched sample-row corpus; large sample corpora are not checked into this repo.
- Made `frontier-bb144-benchmark --help` side-effect-free by deferring Matplotlib cache directory creation until after argument parsing.
