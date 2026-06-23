# Frontier File Scope

This repo is intentionally scoped to the C++-accelerated frontier decoder and
the BB/Gross plus surface-code matrix builders needed to exercise it. Files not
serving that surface were removed.

| File | Why it remains |
| --- | --- |
| `README.md` | Public install, smoke-test, replay, benchmark, matrix, and reproduction instructions. |
| `llms.txt` | Compact plain-text project summary for LLM agents and search/indexing tools. |
| `AGENTS.md` | Operational setup, validation, and style checklist for coding agents. |
| `CITATION.cff` | Citation metadata for citable software releases. |
| `codemeta.json` | CodeMeta software metadata derived from declared package, citation, and repository facts. |
| `AUTHORS.md` | Software authorship list matching `CITATION.cff` plus contributor-list policy. |
| `ACKNOWLEDGEMENTS.md` | Funding, institutional, and upstream-software acknowledgement status. |
| `CONTRIBUTING.md` | Human-facing contribution, validation, compatibility, reproducibility, and asset-change rules. |
| `SECURITY.md` | Minimal security-reporting policy for public research software. |
| `CHANGELOG.md` | Release-level change log; distinct from the internal worklog files. |
| `LICENSE` | Apache License 2.0 text for repository code and documentation unless a file states otherwise. |
| `NOTICE` | Repository attribution and bundled-asset provenance note without relicensing third-party material. |
| `constraints/README.md` | Explains reproducibility constraints versus package requirements. |
| `constraints/py314-macos-validated.txt` | Exact dependency constraints captured from the MacBook Python 3.14.2 validation environment. |
| `constraints/py312-ubuntu-ci.TODO.md` | Placeholder instructions for truthfully capturing exact Ubuntu Python 3.12 CI constraints. |
| `docs/WORKLOG.md` | Agent-readable change log for repo maintenance. |
| `docs/WORKLOG.tex` | Human-readable TeX change log matching `docs/WORKLOG.md`. |
| `docs/FRONTIER_CARD.md` | Short human/agent-readable decoder card with classification labels, inputs, outputs, use cases, limitations, minimal commands, and citation pointers. |
| `docs/DECODER_POSITIONING.md` | Short qualitative guide for positioning Frontier relative to BP+OSD, beam/representative search, tensor-network decoding, and variable elimination. |
| `docs/ACADEMIC_METADATA.md` | Declared citation, funding, DOI, and provenance metadata status. |
| `docs/ASSET_PROVENANCE.md` | Bundled Gross/BB144 asset provenance table and declared asset-license status. |
| `docs/ASSET_MANIFEST.md` | Generated SHA256 checksum manifest for bundled Gross/BB144 assets. |
| `docs/REPRODUCIBILITY.md` | Smoke and publication-grade reproducibility requirements. |
| `docs/BENCHMARK_SCHEMA.md` | CSV/JSON schema for benchmark result summaries, with explicit no-fabrication and raw-corpus caveats. |
| `docs/RELEASE.md` | Release, tag, archive, DOI, and validation checklist for citable releases. |
| `docs/VISIBILITY_RELEASE_CHECKLIST.md` | Release discoverability, citation-hygiene, release-note, archive, and announcement checklist. |
| `docs/FILE_SCOPE.md` | This audit of the retained file set. |
| `docs/ARCHITECTURE.md` | Human/agent orientation guide for workflows, module ownership, public APIs, native dispatch, and safe changes. |
| `docs/COMMANDS.md` | Console-script command index with minimal commands, outputs, and common failure modes. |
| `docs/ENVIRONMENT.md` | Public environment-variable documentation plus internal native debug toggles. |
| `docs/LICENSING.md` | Apache-2.0 scope, third-party licensing notes, and vendoring-notice guidance. |
| `paper/README.md` | Top-level paper-specific reproduction area index. |
| `paper/plots/README.md` | Paper-plot reproduction status, manifest contract, commands, required data columns, and provenance caveats. |
| `paper/plots/manifest.csv` | Figure/panel-to-data/script/output manifest for the recorded `frontier_decoder2.tex` figure inventory; actual figure rows are `reproducible` and the transition percentile table is `support-data`. |
| `paper/plots/data/*.csv` | Minimal plot-ready summary, schematic, and compact figure-state tables for current paper figures; raw per-shot corpora are intentionally excluded. |
| `paper/plots/data/*.json` | Same-stem sidecar metadata, checksums, source labels, decoder settings, and caveats for each paper plot CSV. |
| `paper/plots/data/README.md` | Exact current paper plot table columns plus required JSON sidecar schema. |
| `paper/plots/data/MANIFEST.md` | Deterministic checksum manifest for retained paper-plot data files. |
| `paper/plots/scripts/reproduce_plots.py` | Paper-plot reproduction entry point that lists manifest rows and dispatches reproducible rows to figure-specific renderers. |
| `paper/plots/scripts/plot_utils.py` | Shared CSV, sidecar, manifest, output, axis, and Matplotlib helper utilities for paper plot renderers. |
| `paper/plots/scripts/plot_frontier_schematic.py` | Renderer for the Figure 1 frontier active-boundary schematic from committed element coordinates. |
| `paper/plots/scripts/plot_algorithm_recap.py` | Renderer for the Figure 2 four-panel ordered-frontier recursion state-plane recap. |
| `paper/plots/scripts/plot_surface_threshold.py` | Renderer for the Figure 3 rotated-surface code-capacity FER and retained-list panels. |
| `paper/plots/scripts/plot_color_threshold.py` | Renderer for the Figure 4 hexagonal color-code FER and retained-state panels. |
| `paper/plots/scripts/plot_surface_memory_z_dem_mwpm.py` | Renderer for the Figure 5 rotated-surface memory-Z DEM Frontier versus MWPM comparison. |
| `paper/plots/scripts/plot_bb72_dem.py` | Renderer for the Figure 6 BB72 detector-side DEM panels. |
| `paper/plots/scripts/plot_gross_dem.py` | Renderer for the Figure 7 and Figure 8 Gross/BB144 detector-side DEM panels. |
| `paper/plots/scripts/plot_transition_evals.py` | Renderer for the Figure 9 Gross/BB144 transition-evaluation tail curve using percentile support data. |
| `paper/plots/scripts/plot_failure_decomposition.py` | Renderer for the Figure 10 Gross/BB144 failure-decomposition figure. |
| `paper/plots/outputs/.gitignore` | Keeps generated local plot outputs out of git unless reference images are intentionally added elsewhere. |
| `Makefile` | Standard local shortcuts for native build, tests, smoke, DEM info, and cleanup. |
| `.github/workflows/ci.yml` | Lightweight GitHub Actions validation for install, native build, tests, smoke, and DEM info. |
| `pyproject.toml` | Package metadata, dependencies, console scripts, and pytest config. |
| `setup.py` | Native C++ extension build definition for `_frontier_native`. |
| `frontier/__init__.py` | Public top-level import surface re-exporting stable decoder and DEM helpers. |
| `frontier/decoder.py` | Stable decoder API re-exports backed by `tools.frontier_decoder`. |
| `frontier/dem.py` | Stable DEM loader API re-exports backed by `tools.dem_loader`. |
| `frontier/progressive.py` | Stable model-construction API re-exports backed by `tools.frontier_progressive`. |
| `frontier/py.typed` | Marker declaring the public `frontier` package as typed. |
| `frontier_native.py` | Python wrapper around the compiled native extension. |
| `native/_frontier_native.cpp` | C++ frontier engine used by the public decoder path. |
| `tools/__init__.py` | Lightweight package marker for CLI/support modules. |
| `tools/frontier_decoder.py` | Public Python frontier API and `frontier-smoke` CLI. |
| `tools/dem_loader.py` | Minimal DEM-to-frontier loader and `frontier-dem-info` CLI for BB/Gross and surface-code detector matrices. |
| `tools/frontier_sample_rows.py` | DEM sample-row generator for `frontier-replay`, covering BB/Gross and surface-code detector matrices. |
| `tools/frontier_sample_replay.py` | Matched sample replay CLI for BB144/Gross and related DEM rows. |
| `tools/frontier_bb144_benchmark.py` | Focused BB144/Gross native timing probe over explicit sample rows. |
| `tools/frontier_progressive.py` | Minimal frontier column/layout/order helpers used by the public wrapper and DEM loader. |
| `tools/asset_manifest.py` | Deterministic checksum manifest generator for bundled Gross/BB144 assets. |
| `examples/README.md` | Short guide to runnable examples. |
| `examples/minimal_decode.py` | Tiny public-API decode example matching the smoke model. |
| `examples/inspect_dem.py` | Minimal DEM loader example for `rotated_surface_d3`. |
| `examples/replay_rotated_surface_d3.sh` | Tiny temp-directory sample-row and replay workflow. |
| `tests/test_frontier_export.py` | Regression coverage for the exported frontier wrapper/replay behavior. |
| `tests/test_examples_and_cli.py` | Subprocess smoke coverage for examples, CLI help, and tiny rotated-surface replay outputs. |
| `tests/test_paper_plots.py` | Paper-plot manifest, support-data, renderer dispatch, output reproduction, sidecar, and data-checksum regression tests. |
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
