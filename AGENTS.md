# AGENTS.md

Operational notes for humans and coding agents working in this repository.

Human-facing contribution expectations live in `CONTRIBUTING.md`; release and
archival steps live in `docs/RELEASE.md`.

Metadata authority order is documented in `docs/ACADEMIC_METADATA.md`. Use
`pyproject.toml` for package version, `CITATION.cff` for citation metadata,
`CHANGELOG.md` for release notes, `docs/WORKLOG.md` for maintenance notes,
`docs/ASSET_MANIFEST.md` for bundled asset checksums, and
`paper/plots/manifest.csv` for paper-plot data status.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e .
python setup.py build_ext --inplace
```

## Validation

```bash
python setup.py build_ext --inplace
python -m pytest -q
python -m tools.frontier_decoder --K 16 --Delta 100 --shots 3
python -m tools.dem_loader --backend rotated_surface_d3 --p-location 0.001 --column-order deadline_reorder
python -m tools.dem_loader --backend bravyi_depth7 --p-location 0.001 --column-order deadline_reorder
```

## Platform Notes

See `docs/ARCHITECTURE.md` for supported/tested platform assumptions. Do not
promise Windows/MSVC support unless it has been explicitly ported and tested.

## Style Expectations

- Keep public CLI arguments backward-compatible unless a migration is documented.
- Do not add hidden local paths or developer-machine defaults.
- Do not commit large generated result files; use `results/` locally.
- Keep bundled Gross/BB144 assets under `grosscode/assets/gross144`.
- Document new retained files in `docs/FILE_SCOPE.md`.
- Preserve Apache-2.0 headers/notices and any third-party notices.
- Do not add files with incompatible or unclear licensing.
- Do not fabricate paper plot data, benchmark values, figure lists, paper
  references, DOIs, ORCIDs, grant numbers, or asset provenance.
- Do not edit plot image files manually.
- Do not change plot data without updating provenance, sidecar metadata,
  checksums, reproduction docs, and tests.
- Do not mark a paper plot as `reproducible` unless the committed data and
  script regenerate the listed output file.
- If figure data are present but no committed renderer exists, mark the row as
  `script-missing`. If figure data are missing, mark the manifest row as
  `data-missing` or `external-archive-needed`; if the paper figure list is not
  present, keep the manifest schema-only and document the missing list in
  `paper/plots/README.md`.
- Record vendored third-party code/assets in `NOTICE` or `docs/LICENSING.md`.
- Update `CITATION.cff`, `ACKNOWLEDGEMENTS.md`, and
  `docs/ACADEMIC_METADATA.md` when citation, funding, DOI, or author facts
  change.
- If a file has a different license, keep that license notice with the file.
- Prefer small modules with explicit responsibilities.
- Use type annotations for new Python functions.

## Before Committing

- Rebuild the native extension if decoder or binding code changed.
- Run the validation commands above.
- Update `README.md`, `docs/FILE_SCOPE.md`, `docs/WORKLOG.md`, and
  `docs/WORKLOG.tex` when behavior, scope, or files change.
- Update `paper/plots/manifest.csv`, `paper/plots/data/MANIFEST.md`, and
  `paper/plots/README.md` when paper plot data or scripts change.
- Confirm generated caches, build products, and large results are not staged.
- Preserve the `_frontier_native` extension name and the existing reproduction workflows.
