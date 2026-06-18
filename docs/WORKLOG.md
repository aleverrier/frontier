# Frontier Worklog

## 2026-06-18 Metadata Finalization

- Replaced provisional metadata with declared current status: software authors
  Anthony Leverrier and Ruediger Urbanke, no declared ORCIDs, no assigned DOI,
  no declared funding metadata, and no separate asset-license statement beyond
  the repository `NOTICE`.
- Added `constraints/py314-macos-validated.txt` from the MacBook Python 3.14.2
  validation environment and removed the provisional constraints note.
- Removed provisional markers from public docs and the depth-8 backend error
  message, then added a regression test that checks public docs stay free of
  those markers.

## 2026-06-18 Academic Release Metadata Pass

- Added public research-software metadata and release hygiene files:
  `CITATION.cff`, `ACKNOWLEDGEMENTS.md`, `CONTRIBUTING.md`, `CHANGELOG.md`,
  `docs/ACADEMIC_METADATA.md`, `docs/ASSET_PROVENANCE.md`,
  `docs/REPRODUCIBILITY.md`, `docs/RELEASE.md`, and `constraints/`.
- Added `tools/asset_manifest.py` and generated `docs/ASSET_MANIFEST.md` with
  SHA256 checksums for all bundled Gross/BB144 assets.
- Updated README navigation, preferred public API guidance, architecture docs,
  file-scope audit, licensing notes, Makefile, CI, and tests for the
  publication-readiness checklist.
- Modernized `pyproject.toml` to the current PyPA license expression form:
  `license = "Apache-2.0"` plus `license-files = ["LICENSE", "NOTICE"]`.
- Added exact dependency constraints for the validated MacBook Python 3.14.2
  environment in `constraints/py314-macos-validated.txt`.
- Validation completed in a temporary Python 3.14 virtualenv:
  - `python -m pip install -U pip setuptools wheel`
  - `python -m pip install -e .`
  - `python setup.py build_ext --inplace`
  - `python -m pytest -q` (`18 passed`)
  - `python -m tools.frontier_decoder --K 16 --Delta 100 --shots 3`
  - `python -m tools.dem_loader --backend rotated_surface_d3 --p-location 0.001 --column-order deadline_reorder`
  - `python -m tools.dem_loader --backend bravyi_depth7 --p-location 0.001 --column-order deadline_reorder`
  - `python examples/minimal_decode.py`
  - `python examples/inspect_dem.py`
  - `PYTHON_BIN=$PYTHON_BIN bash examples/replay_rotated_surface_d3.sh`
  - regenerated-manifest diff check against `python -m tools.asset_manifest`
  - private-path grep over README/docs/AGENTS/CONTRIBUTING/CHANGELOG/CITATION/pyproject/tests
  - `git diff --check`

## 2026-06-18 Apache-2.0 Licensing

- Added the root `LICENSE` file with the standard Apache License 2.0 text and
  a root `NOTICE` file for repository attribution plus bundled-asset provenance
  notes.
- Updated `pyproject.toml` with `license = { file = "LICENSE" }` and the
  Apache Software License classifier.
- Replaced the former license-status note in `docs/LICENSING.md` with Apache-2.0
  scope guidance, third-party licensing notes, and the rationale for not using
  CC BY-NC-SA as a software-code license.
- Updated README navigation and the README `License` section, plus
  `docs/ARCHITECTURE.md`, `AGENTS.md`, and `docs/FILE_SCOPE.md`, so humans and
  agents preserve Apache-2.0 and third-party notices in future changes.
- Added a lightweight regression test that checks the license text, docs,
  README, package metadata, and file-scope audit.

## 2026-06-18 Navigation Polish and CLI Testability

- Removed developer-machine absolute paths from validation notes and kept
  interpreter references generic as `$PYTHON_BIN`.
- Added `frontier.progressive` as a public wrapper for small model-construction
  primitives, then updated `examples/minimal_decode.py` to use only public
  `frontier.*` imports.
- Removed `sys.path` manipulation from Python examples; they now model
  installed editable-package usage.
- Normalized all CLI modules around `main(argv=None)` and `_parse_args(argv)`,
  including `tools.dem_loader`, `tools.frontier_decoder`,
  `tools.frontier_sample_replay`, and `tools.frontier_bb144_benchmark`.
- Corrected `docs/ENVIRONMENT.md` to match `resolve_cache_root`, fixed the
  rotated-surface replay command label in `docs/COMMANDS.md`, and documented
  platform/compiler expectations in `docs/ARCHITECTURE.md`.
- Added `tests/test_examples_and_cli.py` for example subprocess checks, CLI
  help checks, and a tiny rotated-surface replay output smoke test.
- Extended CI to exercise installed console scripts, examples, and help output.
- Validation completed with `$PYTHON_BIN` as the active project interpreter:
  - `$PYTHON_BIN -m pip install -e .`
  - `$PYTHON_BIN setup.py build_ext --inplace`
  - `$PYTHON_BIN -m pytest -q` (`15 passed`)
  - `$PYTHON_BIN -m tools.frontier_decoder --K 16 --Delta 100 --shots 3`
  - `$PYTHON_BIN -m tools.dem_loader --backend rotated_surface_d3 --p-location 0.001 --column-order deadline_reorder`
  - `$PYTHON_BIN -m tools.dem_loader --backend bravyi_depth7 --p-location 0.001 --column-order deadline_reorder`
  - `$PYTHON_BIN examples/minimal_decode.py`
  - `$PYTHON_BIN examples/inspect_dem.py`
  - `PYTHON_BIN=$PYTHON_BIN bash examples/replay_rotated_surface_d3.sh`

## 2026-06-18 Navigation and Public API Orientation

- Added `docs/ARCHITECTURE.md`, `docs/COMMANDS.md`, `docs/ENVIRONMENT.md`, `docs/LICENSING.md`, `AGENTS.md`, a `Makefile`, GitHub Actions CI, and tiny runnable examples under `examples/`.
- Added the lightweight `frontier/` package (`frontier.__init__`, `frontier.decoder`, `frontier.dem`, and `py.typed`) as stable public re-exports while keeping existing `tools.*` imports and console-script implementations intact.
- Updated README navigation and `docs/FILE_SCOPE.md` so humans and agents can find the architecture guide, command index, environment variables, examples, retained-file audit, and licensing notes quickly.
- Added module docstrings, argparse help epilogues, and section headers to the retained long `tools/` modules without changing decoder math, native extension naming, or public reproduction command semantics.
- Added lightweight tests for public package imports, console-script module entry points, architecture docs, command docs, and file-scope coverage.
- Validation completed with a project-local Python interpreter because this
  shell had no active `python` alias and the system `python3` lacked
  `setuptools`:
  - `$PYTHON_BIN setup.py build_ext --inplace`
  - `$PYTHON_BIN -m pytest -q` (`11 passed`)
  - `$PYTHON_BIN -m tools.frontier_decoder --K 16 --Delta 100 --shots 3`
  - `$PYTHON_BIN -m tools.dem_loader --backend rotated_surface_d3 --p-location 0.001 --column-order deadline_reorder`
  - `$PYTHON_BIN -m tools.dem_loader --backend bravyi_depth7 --p-location 0.001 --column-order deadline_reorder`
  - `$PYTHON_BIN examples/minimal_decode.py`
  - `$PYTHON_BIN examples/inspect_dem.py`
  - `PYTHON_BIN=$PYTHON_BIN bash examples/replay_rotated_surface_d3.sh`

## 2026-06-15

- Created the initial `frontier` export.
- Selected repo shape 3 and decoder mode 3: native C++ decoder plus DEM replay/benchmark CLI, with forward `deadline_reorder` and backward `backward_deadline_reorder`.
- Public modes are limited to `forward_only`, `backward_only`, and `fwd_bwd_committee`.
- Validation completed:
  - `python setup.py build_ext --inplace`
  - `python -m py_compile frontier_native.py tools/dem_loader.py tools/frontier_decoder.py tools/frontier_sample_replay.py tools/frontier_bb144_benchmark.py tools/frontier_progressive.py tests/test_frontier_export.py`
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
- Kept the frontier/native path, BB144/Gross/generalized-bicycle/rotated-surface/surface matrix builders, and split-sector DEM builder.
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
- Validation completed after tightening:
  - `python -m py_compile frontier_native.py tools/dem_loader.py tools/frontier_decoder.py tools/frontier_sample_replay.py tools/frontier_bb144_benchmark.py tools/frontier_progressive.py tests/test_frontier_export.py` with bytecode redirected outside the repo
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
- Corrected the fresh-clone install command to start with `python3 -m venv .venv`, since this macOS host does not provide a `python` executable before the venv exists.
- Updated the matrix section to state explicitly that Gross/BB144 static assets are bundled while other supported matrix families are built or generated.
- Added `frontier-sample-rows`, a public CLI that generates `sample_rows.csv` from the same detector-side DEM matrices and priors used by `frontier-replay`.
- Documented the complete fresh BB144/Gross DEM reproduction flow at `p=0.001`, `Delta=12`, `K=512`, and `fwd_bwd_committee`: generate 10k matched sample rows, then replay them with the native engine.
- Made `frontier-dem-info` load all requested matrices before printing its CSV header, so missing assets no longer produce a partial CSV.
- Removed replay summary warnings when pressure diagnostics are disabled.
- Validation completed:
  - `python -m py_compile ...` from a fresh-clone virtual environment, with bytecode redirected outside the repo
  - `python -m pytest -q -p no:cacheprovider` from that fresh-clone virtual environment (`6 passed`)
  - `python setup.py build_ext --inplace` using the fresh-clone venv interpreter
  - `python -m tools.frontier_decoder --K 16 --Delta 100 --shots 3`
  - `python -m tools.dem_loader --backend rotated_surface_d3 --p-location 0.001 --column-order deadline_reorder`
  - a bad `GROSSCODE_ASSET_ROOT` override now prints a one-line missing-assets error and no CSV header
  - `python -m tools.dem_loader --backend bravyi_depth7 --p-location 0.001 --column-order deadline_reorder` reports `936x8784` detector matrices for both memory sectors from bundled assets
  - generated and replayed sample rows for `rotated_surface_d3` and a 10-shot BB144/Gross `p=0.001`, `Delta=12`, `K=512`, `fwd_bwd_committee` native replay

## 2026-06-15 BB144 Reproduction Success Criteria

- Added a README section explaining how a fresh user can tell that the BB144/Gross `p=0.001`, `Delta=12`, `K=512`, `fwd_bwd_committee` reproduction worked.
- Documented the expected `frontier-dem-info` matrix output: `memory_X` and `memory_Z` detector matrices are `936x8784`, logical matrices are `12x8784`, with 12 noisy rounds and `deadline_reorder`.
- Documented the expected generated workload shape: 20,000 side rows for 10,000 paired full-frame shots, 40 replay shard tasks when `--shards-per-side 20` is used, and a complete native replay recorded in `run_metadata.json`.
- Clarified that `summary_by_scope.csv` contains `memory_X`, `memory_Z`, and `combined`, and that the `combined` row is the strict full logical FER over paired side rows.
- Avoided claiming a zero FER estimate: at `p=0.001`, 10k shots is a smoke-scale reproducibility sample, and no observed failures should be reported as below the resolution of that sample rather than as evidence that the FER is zero.
- Avoided quoting a wall-clock target in the README. Wall-clock timing is machine-dependent; the README now asks timing reports to include machine/Python/workers/native availability/batch size and points to transition-evaluation counts for more machine-independent comparison.

## 2026-06-15 Minimal Public Cleanup

- Removed the non-self-contained external BB circuit-generation bridge because it depended on a checkout that is not part of this repo.
- Removed the remaining replay re-decode CLI flags, CSV fields, metadata fields, and report text. The retained replay modes are forward-only, backward-only, and forward/backward committee with the requested `K` and `Delta`.
- Replaced the old large progressive helper with `tools/frontier_progressive.py`, which contains only the frontier column/layout/order helpers required by the public decoder and DEM loader.
- Removed an obsolete sample-row option that only applied to the deleted external constructor path.
- Cleaned docs and worklogs to avoid exact local temporary clone paths while keeping the reproducibility commands and validation notes useful.
