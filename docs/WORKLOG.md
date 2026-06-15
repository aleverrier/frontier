# Frontier Worklog

## 2026-06-15

- Created the initial `frontier` export from the `better-beam` decoder code.
- Selected repo shape 3 and decoder mode 3: native C++ decoder plus DEM replay/benchmark CLI, with forward `deadline_reorder` and backward `backward_deadline_reorder`.
- Public modes are limited to `forward_only`, `backward_only`, and `fwd_bwd_committee`.
- The two-stage decoder is intentionally not exposed in the public CLI or native Python wrapper.
- Validation completed:
  - `/Users/anthony/research/better-beam/tools/py setup.py build_ext --inplace`
  - `/Users/anthony/research/better-beam/tools/py -m py_compile frontier_fast_native.py tools/frontier_fast_decoder.py tools/frontier_fast_sample_replay.py tools/frontier_fast_bb144_benchmark.py tools/gross144_dem_x_progressive_report.py tools/steane_progressive_decoder.py tests/test_frontier_export.py`
  - `/Users/anthony/research/better-beam/tools/py -m pytest -q` (`3 passed`)
  - `/Users/anthony/research/better-beam/tools/py -m tools.frontier_fast_decoder --K 16 --Delta 100 --shots 3`

## Open Items

- Pushed to GitHub: `git@github.com:aleverrier/frontier.git`, branch `main`; first published commit was `3b45933`.
- Decide whether to later refactor the native C++ file to physically delete the unused internal stage1/stage2 implementation code.
