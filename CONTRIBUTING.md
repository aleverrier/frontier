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

`make typecheck` is an advisory public-API check scoped to the `frontier`
package. It is not a strict type gate for the full retained research-code
surface.

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

## Paper Plot Data

Paper plot reproduction is tracked under `paper/plots/`. Any change that adds
or changes a paper plot must update:

- `paper/plots/manifest.csv`
- the minimal CSV data file and same-stem JSON sidecar under `paper/plots/data/`
- `paper/plots/data/MANIFEST.md`
- `paper/plots/README.md`
- `tests/test_paper_plots.py`, if the reproduction contract changes

Do not fabricate plot values, paper figure references, archive identifiers, or
statistical caveats. Do not mark a row `reproducible` unless the committed data
and script regenerate the listed output. Use `support-data` for committed
companion tables that are consumed by another renderer but are not standalone
figure outputs. If data are present but the renderer is not, use
`script-missing`. If data are not present, use `data-missing` or
`external-archive-needed`.

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

## Interaction Norms

Keep issues and reviews professional, specific, and focused on the software,
data, documentation, and reproducibility questions. This repository does not
currently maintain a separate code-of-conduct governance document.

## Issues

When reporting an issue, include the command, expected behavior, observed
behavior, Python version, operating system/compiler, whether
`frontier_native_available` is true, and the relevant sample-corpus or DEM
identifier if the issue involves decoding output.

For agent-specific maintenance notes, see `AGENTS.md`.
