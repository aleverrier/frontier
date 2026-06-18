# Contributing

This repository is a focused public export of the C++-accelerated frontier
decoder. Keep changes compatible with the documented command-line tools and the
preferred `frontier.*` Python API.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e .
python setup.py build_ext --inplace
```

For a pinned reproducibility environment, see `constraints/README.md`.

## Validation

Run the core checks before submitting decoder, packaging, asset, or workflow
changes:

```bash
python setup.py build_ext --inplace
python -m pytest -q
python -m tools.frontier_decoder --K 16 --Delta 100 --shots 3
python -m tools.dem_loader --backend rotated_surface_d3 --p-location 0.001 --column-order deadline_reorder
python -m tools.dem_loader --backend bravyi_depth7 --p-location 0.001 --column-order deadline_reorder
```

Optional local checks:

```bash
make lint
make typecheck
```

## Coding Style

- Prefer small, explicit functions with type annotations for new Python code.
- Keep examples and new user-facing code on the `frontier.*` public API.
- Keep `tools.*` imports working for existing console scripts and backward
  compatibility.
- Do not change decoder mathematics, pruning/scoring/tie-breaking, native
  dispatch, benchmark semantics, or bundled data without a dedicated review and
  tests.
- Preserve the `_frontier_native` extension name.

## Tests And Benchmarks

- `make test` / `python -m pytest -q` is the required lightweight test gate.
- Benchmark or result changes must document the exact command, commit hash,
  sample corpus, matrix family, decoder settings, worker count, and native
  availability.
- Do not commit large generated sample corpora or result directories.

## Bundled Assets

Bundled assets live under `grosscode/assets/gross144`. Before adding or
modifying any asset:

- confirm its provenance and license status;
- update `docs/ASSET_PROVENANCE.md`;
- regenerate or update `docs/ASSET_MANIFEST.md`;
- update `docs/FILE_SCOPE.md` if retained-file scope changes.

Use:

```bash
python -m tools.asset_manifest > docs/ASSET_MANIFEST.md
```

## Licensing And Citation Metadata

- Preserve Apache-2.0 repository licensing and third-party notices.
- Do not add SPDX headers to provenance-ambiguous third-party assets.
- Substantial contributions should update `CITATION.cff`,
  `ACKNOWLEDGEMENTS.md`, and `docs/ACADEMIC_METADATA.md` when author,
  funding, DOI, or preferred-citation facts change.

## Issues

When reporting an issue, include the command, expected behavior, observed
behavior, Python version, operating system/compiler, whether
`frontier_native_available` is true, and the relevant sample-corpus or DEM
identifier if the issue involves decoding output.

For agent-specific maintenance notes, see `AGENTS.md`.
