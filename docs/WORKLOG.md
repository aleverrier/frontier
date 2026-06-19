# Frontier Worklog

## 2026-06-19 Public arXiv Metadata Update

- Replaced the temporary paper-citation placeholder with the public arXiv
  record for Anthony Leverrier and Rüdiger Urbanke, "Approximating optimal
  decoding of quantum LDPC codes with narrow frontiers," `arXiv:2606.20513`
  [quant-ph], submitted 2026-06-18.
- Updated `CITATION.cff`, `README.md`, `docs/ACADEMIC_METADATA.md`,
  `docs/RELEASE.md`, `CHANGELOG.md`, and `codemeta.json` so the software
  citation remains distinct from the associated paper citation.
- Updated `tests/test_frontier_export.py` to assert the public arXiv metadata
  and reject the old placeholder wording.

## 2026-06-18 BB144 p=0.002 Quick DEM Replay

- Cloned `git@github.com:aleverrier/frontier.git` into
  `/Users/anthony/research/tests/frontier` for a fresh local reproducibility
  check.
- Created a Python 3.14.2 virtualenv from `constraints/py314-macos-validated.txt`,
  installed the editable package, and rebuilt `_frontier_native`.
- Validation before the run:
  - `.venv/bin/python -m tools.dem_loader --backend bravyi_depth7 --p-location 0.002 --column-order deadline_reorder`
    reported `memory_X` and `memory_Z` detector matrices `936x8784`, logical
    matrices `12x8784`, 12 noisy rounds, and `deadline_reorder`.
  - `.venv/bin/python -m tools.frontier_decoder --K 16 --Delta 100 --shots 3`
    passed with 3/3 successful smoke shots.
- Generated 1000 matched BB144/Gross DEM sample shots per memory sector at
  `p_location=0.002` with seed `20260618`. Local ignored artifact:
  `results/bb144_p0p002_1000_sample_rows.csv`.
- Replayed the sample with native Frontier, `K=1024`, `Delta=10`,
  `score_alpha=0.8`, `fwd_bwd_committee`, `cpus=1`, `shards_per_side=10`,
  and `native_batch_size=64`. Local ignored result root:
  `results/bb144_p0p002_frontier_replay_k1024_Delta10_alpha0p8_1000`.
- Result summary from `summary_by_scope.csv`: `memory_X` had `0/1000`
  failures, `memory_Z` had `0/1000` failures, and the paired `combined` row had
  `0/1000` failures. This should be read as below the resolution of the
  1000-shot smoke sample, not as evidence that the FER is zero.
- Combined mean decode time was `0.0403428542s`; combined mean total transition
  evaluations were `661768.62`; replay wall time was `51.5s`; the native engine
  was available and used.

## 2026-06-18 Paper Placeholder, LLM Acknowledgement, And Figure Deduplication

- Added an explicit paper-citation placeholder in `CITATION.cff`, `README.md`,
  and `docs/ACADEMIC_METADATA.md`: the Frontier decoder paper arXiv identifier
  is pending and should replace the placeholder as soon as the public arXiv
  record exists.  No fake DOI or arXiv identifier was added.
- Added a large-language-model acknowledgement to `ACKNOWLEDGEMENTS.md` and
  `docs/ACADEMIC_METADATA.md`, matching the paper statement that OpenAI Codex
  assisted with programming, documentation, and testing.
- Removed the duplicate Gross/BB144 FER-vs-average-retained manifest row
  `gross_dem_avg_retained_duplicate`; the figure is now represented only as the
  right panel of `gross_dem_circuit`, so `reproduce_plots.py --all --strict`
  does not render the same PNG twice.
- Updated `paper/plots/README.md`, `docs/FILE_SCOPE.md`, and regression tests so
  reproducible figure outputs must be unique and the post-deduplication Figure
  9/10 numbering is documented.

## 2026-06-18 Build Metadata Classifier Repair

- Removed the stale `License :: OSI Approved :: Apache Software License`
  classifier from `pyproject.toml`. The project already declares
  `license = "Apache-2.0"` and `license-files = ["LICENSE", "NOTICE"]`, and
  current `setuptools` rejects legacy license classifiers when validating the
  PEP 639 license-expression form.
- This fixes the fresh-clone build blocker observed by
  `python setup.py build_ext --inplace`, which previously failed before C++
  compilation with `setuptools.errors.InvalidConfigError: License classifiers
  have been superseded by license expressions`.
- Validation with `/Users/anthony/research/better-beam/tools/py` completed:
  native build, `pytest -q` (`27 passed`; cache-write warning only),
  `tools.frontier_decoder --K 16 --Delta 100 --shots 3`, both DEM loader checks
  for `rotated_surface_d3` and `bravyi_depth7` at `p_location=0.001`, both
  Python examples, paper plot reproduction with `--all --strict`, and
  `git diff --check`.
- The Gross/BB144 DEM info check reported the expected accepted split-sector
  dimensions: `D_X=D_Z=936x8784`, `O_X=O_Z=12x8784`, with 12 noisy rounds.

## 2026-06-18 Paper Plot Renderer Completion

- Added shared plot helpers and committed Matplotlib renderers for all current
  paper figures listed in `paper/plots/manifest.csv`.
- Updated the manifest so actual figure rows are `reproducible` from committed
  summary tables. The transition percentile table is now `support-data` because
  it is consumed by the transition-evaluation renderer and is not a standalone
  plot output.
- Regenerated `paper/plots/data/MANIFEST.md` after normalizing JSON sidecars
  with explicit plot-vs-simulation reproducibility fields, raw-corpus absence,
  renderer paths, and output paths.
- Updated README, paper plot docs, reproducibility docs, contribution notes,
  file-scope audit, academic metadata authority order, and plot/metadata
  regression tests.
- Refreshed manuscript-source metadata against the available
  `frontier_decoder2.tex` file, sha256
  `288da4629eddc7038f38f3ae2948d358b57a018544eb6f2591a7aebe5f8e5380`, and the
  available rendered PDF candidate `Frontier_decoder-2.pdf`, sha256
  `1406a80c7448f6964634da42d4f520b0cf03f97b60899034ac2aff1219cb29c5`. A
  literal `frontier_decoder2(2).tex` file was not present.
- Kept the PyPA-compatible `license = "Apache-2.0"` and `license-files`
  metadata without the legacy Apache classifier; current setuptools rejects
  license classifiers when validating the license-expression form.
- Validation completed with `$PYTHON_BIN` as the active project interpreter:
  editable install, native build, `pytest -q` (`27 passed`), plot/metadata
  subset (`23 passed`), frontier smoke, both DEM loader checks with
  `p_location=0.001` and `deadline_reorder`, both Python examples, plot
  `--list`, plot `--all --strict`, asset-manifest diffs, private-path grep,
  `ruff check .`, and `git diff --check`. The bare replay shell command is
  environment-dependent on this machine because the unactivated system
  `python3` lacks `scipy`; `PYTHON_BIN=$PYTHON_BIN bash
  examples/replay_rotated_surface_d3.sh` passed.

## 2026-06-18 Paper Plot Summary Tables

- Treated `frontier_decoder2.tex` as the current paper source for the figure
  inventory, with sha256
  `d1abc814aab7ec6e8bacfab0af31b95d7f84b4a01170b648c2ceefacd5ae153e`
  at the time of the initial table import; the renderer-completion entry above
  records the later available manuscript hash refresh.
- Added compact plot-ready CSV tables and JSON sidecars for every current
  figure or panel, including the generated schematic, BB72 algorithm recap
  state table, surface/color code-capacity panels, surface memory-Z DEM
  comparison, BB72/Gross detector-side DEM panels, Gross transition-evaluation
  tail/percentile data, and the Gross p=0.002 failure-decomposition table.
- Left all manifest rows as `script-missing`, not `reproducible`, because the
  committed repo still lacks figure-specific renderers for the published PNGs.
- Normalized local and scratch source paths in imported CSVs to stable source
  labels and avoided copying raw per-shot corpora.
- Updated paper-plot docs and tests so `script-missing` rows must have committed
  CSV data, same-stem sidecars, and matching checksums while the CLI continues
  to skip unreproducible rows honestly.

## 2026-06-18 Paper Plot Reproduction Scaffold

- Audited the checkout for paper figure lists and plot-ready summary tables;
  none were present, so no paper plot data or numeric figure values were added.
- Added an honest paper plot scaffold under `paper/plots/`: schema-only
  `manifest.csv`, missing-data documentation, ignored generated outputs, and
  `reproduce_plots.py` commands that list or skip missing rows without
  fabricating data.
- Extended `tools/asset_manifest.py` with optional `--root` and `--title`
  arguments while preserving the default Gross/BB144 asset manifest output, then
  added `paper/plots/data/MANIFEST.md` for the current retained paper-data
  directory.
- Added `tests/test_paper_plots.py` so the manifest schema, missing-data
  honesty, plot reproduction entry point, and data-manifest checksums are
  enforced.
- Added `codemeta.json`, `AUTHORS.md`, `SECURITY.md`, and
  `constraints/py312-ubuntu-ci.TODO.md` using only declared repository facts,
  and updated README, AGENTS, CONTRIBUTING, reproducibility, release, metadata,
  file-scope, and typecheck docs accordingly.

## 2026-06-18 Metadata Finalization

- Replaced provisional metadata with declared current status: software authors
  Anthony Leverrier and Rüdiger Urbanke, no declared ORCIDs, no assigned DOI,
  Plan France 2030 project ANR-22-PETQ-0006 funding, COSMIQ/Inria sabbatical
  hospitality acknowledgement, and no separate asset-license statement beyond
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
