# Frontier Architecture

## What This Repository Does

This repository is a standalone export of the C++-accelerated frontier decoder
path for BB/Gross and surface-code detector-side DEM matrices. It keeps the
decoder wrapper, the `_frontier_native` C++ extension, matrix/DEM builders, and
small replay and benchmark CLIs needed to reproduce the public workflows.

It is not a general archive of all historical BP/min-sum, polar DEM, or research
benchmark harnesses. `docs/FILE_SCOPE.md` is the retained-file audit.

## Main Workflows

1. Install the package and build the native extension with
   `python setup.py build_ext --inplace`.
2. Run the smoke test with `frontier-smoke` or
   `python -m tools.frontier_decoder`.
3. Inspect supported DEM matrices with `frontier-dem-info`.
4. Generate matched detector-side sample rows with `frontier-sample-rows`.
5. Replay or benchmark sample rows with `frontier-replay` and
   `frontier-bb144-benchmark`.

See `docs/COMMANDS.md` for command details and `README.md` for the complete
BB144/Gross reproduction commands.

## Module Map

| Path | Role |
| --- | --- |
| `frontier_native.py` | Import wrapper around the compiled `_frontier_native` extension. It keeps native availability checks and class aliases in one place. |
| `frontier/progressive.py` | Public re-export surface for small frontier model-construction primitives used by examples and light API users. |
| `native/_frontier_native.cpp` | C++ frontier engine and Python bindings. This is the hot native implementation; keep the extension name `_frontier_native`. |
| `tools/frontier_decoder.py` | Public decoder dataclasses, dispatch API, Python reference decoder, native adapters, committee selection, and `frontier-smoke` CLI. |
| `tools/frontier_progressive.py` | Frontier column, layout, ordering, scoring, and payload helpers shared by the decoder and DEM loader. |
| `tools/dem_loader.py` | Detector-side DEM-to-frontier loader plus `frontier-dem-info` CLI for BB/Gross and surface-code backends. |
| `tools/frontier_sample_rows.py` | Matched DEM sample-row generator used by replay workflows. |
| `tools/frontier_sample_replay.py` | Replay CLI, CSV schema handling, shard execution, native batch paths, summaries, and reports. |
| `tools/frontier_bb144_benchmark.py` | Focused BB144/Gross native timing probe over explicit sample rows. |
| `grosscode/dem/builder.py` | Split-sector detector-side DEM construction and bundled DEM snapshot loading. |
| `grosscode/circuits/backends.py` | Backend-to-Stim circuit resolution for Gross, generalized-bicycle, and rotated-surface families. |
| `grosscode/codes/*` | CSS code builders and loaders for Gross/BB144, generalized bicycle, rotated-surface, and standard surface-code checks. |
| `grosscode/utils/*` | Repo path, cache path, asset-root, and GF(2) helper utilities. |
| `tests/test_frontier_export.py` | Lightweight regression tests for public imports, CLI modules, docs, sample rows, native wrapper behavior, and replay summaries. |

`tools/` currently contains both console-script modules and compatibility
implementation modules. New public imports should prefer the `frontier` package
when possible, while old `tools.*` imports remain supported.

## Public API vs Internal Helpers

User-facing command APIs are the console scripts from `pyproject.toml`:

- `frontier-smoke`
- `frontier-dem-info`
- `frontier-sample-rows`
- `frontier-replay`
- `frontier-bb144-benchmark`

### Preferred Public API

- `frontier.FrontierModel`
- `frontier.FrontierResult`
- `frontier.FrontierStats`
- `frontier.FrontierCommitteeMember`
- `frontier.decode_frontier`
- `frontier.decode_frontier_committee`
- `frontier.dem.load_dem_family`
- `frontier.dem.build_backward_deadline_ordered_family`
- `frontier.progressive.FactorTransition`
- `frontier.progressive.OutcomeTransition`
- `frontier.progressive.columns_from_factor_transitions`
- `frontier.progressive.build_frontier_layout`

New examples should use `frontier.*` unless they intentionally demonstrate a
lower-level matrix builder such as `grosscode.dem.builder.build_split_sector_problem`,
`grosscode.codes.surface.standard_surface_checks`, or
`grosscode.codes.rotated_surface.load_rotated_surface_code`.

### Compatibility/Internal Implementation Modules

- `tools.frontier_decoder`
- `tools.dem_loader`
- `tools.frontier_progressive`
- other `tools.*`

`tools.*` remains supported for console scripts and backward compatibility, but
it is not the preferred import surface for new examples. Underscore-prefixed
functions are internal implementation hooks and may change. Tests may exercise
selected underscore helpers when they pin behavior at a boundary, but external
users should not build workflows on those names.

## Native Engine Dispatch

`decode_frontier(..., _engine="auto")` follows this high-level ladder:

1. Prefer the native binary engine when `_frontier_native` is built and the
   model is compatible with the binary payload path.
2. Use the binary/Python adapter path when the model is binary-compatible but
   the native engine is unavailable.
3. Use the native choice engine when available and compatible with a non-binary
   model.
4. Fall back to the pure Python reference decoder.

`frontier_lite` and `maxlog_int` metric modes require native-binary
compatibility. They intentionally fail rather than silently falling back to a
different mathematical path.

## Data and Assets

Bundled assets live under `grosscode/assets/gross144`:

- Gross `[[144,12,12]]` CSS matrices.
- BB144/Gross memory X/Z Stim circuits for the public rates documented in
  `README.md`.
- A prebuilt `bravyi_depth7`, `p=0.001` detector-side DEM snapshot.

Important environment variables are documented in `docs/ENVIRONMENT.md`:

- `GROSSCODE_ASSET_ROOT` overrides the bundled Gross/BB144 asset root.
- `FRONTIER_CACHE_DIR` controls generated cache placement.
- Generated replay/benchmark outputs should stay under local result paths such
  as `results/` and should not be committed.

Repository code and documentation are Apache-2.0 unless otherwise marked. Do
not add third-party code or assets without preserving upstream license notices.
If vendoring code from MIT, BSD, Apache, or similar projects, preserve headers
and record the provenance in `NOTICE` or a third-party notice file.

## Platform and Compiler Support

- CI validates Ubuntu with Python 3.11 and 3.12.
- CI includes a minimal macOS Python 3.12 smoke job.
- The native extension expects a C++17 compiler compatible with the current
  source and `setup.py` flags.
- Windows/MSVC is not currently advertised as supported. Treat it as a porting
  target only after explicit implementation and CI coverage.

## How to Make Safe Changes

- Avoid changing decoder tie-breaking without explicit tests.
- Keep CSV schemas stable unless a migration is documented.
- Update `README.md` and `docs/FILE_SCOPE.md` when adding or removing files.
- Update tests when changing public CLI or API behavior.
- Preserve bundled Gross/BB144 assets and the `_frontier_native` extension name.
- Preserve Apache-2.0 and third-party license notices.
- Run the validation commands in `AGENTS.md` before landing decoder or workflow
  changes.
