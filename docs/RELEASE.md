# Release Process

Use this checklist for citable public releases. For discoverability, citation
hygiene, release-note, archive, and announcement checks, also complete
[`docs/VISIBILITY_RELEASE_CHECKLIST.md`](VISIBILITY_RELEASE_CHECKLIST.md).

## Pre-Release Checklist

- Confirm the working tree is clean.
- Update the version in `pyproject.toml`.
- Update `CHANGELOG.md`.
- Ensure `CITATION.cff` version, date, and related-paper citation metadata are
  current.
- Review `ACKNOWLEDGEMENTS.md` and `docs/ACADEMIC_METADATA.md` for release
  consistency.
- Confirm asset provenance and license status in `docs/ASSET_PROVENANCE.md`.
- Generate or update checksums with:

```bash
python -m tools.asset_manifest > docs/ASSET_MANIFEST.md
python -m tools.asset_manifest --root paper/plots/data --title "Paper Plot Data Manifest" > paper/plots/data/MANIFEST.md
```

- Confirm `paper/plots/manifest.csv` is honest: reproducible rows must have
  committed data and scripts; committed companion tables that are not
  standalone outputs must be `support-data`; rows with committed data but no
  renderer must remain `script-missing`; missing data must remain marked
  `data-missing` or `external-archive-needed`.

- Run validation:

```bash
python -m pip install -U pip setuptools wheel
python -m pip install -e .
python setup.py build_ext --inplace
python -m pytest -q
python -m tools.frontier_decoder --K 16 --Delta 100 --shots 3
python -m tools.dem_loader --backend rotated_surface_d3 --p-location 0.001 --column-order deadline_reorder
python -m tools.dem_loader --backend bravyi_depth7 --p-location 0.001 --column-order deadline_reorder
python examples/minimal_decode.py
python examples/inspect_dem.py
bash examples/replay_rotated_surface_d3.sh
python paper/plots/scripts/reproduce_plots.py --all --strict --out-dir /tmp/frontier-paper-plots
```

## Tag And Archive

```bash
git tag -a vX.Y.Z -m "frontier vX.Y.Z"
git push origin vX.Y.Z
```

Archive the tagged release on Zenodo or an institutional repository.

## After Public Identifier Assignment

- Add the software DOI to `CITATION.cff`.
- Keep the related paper metadata synchronized with the current public arXiv
  record.
- Add preferred citation metadata if the paper/software citation target is
  chosen.
- Link the DOI or updated publication identifier from `README.md` if
  appropriate.

Do not claim a software DOI or publication identifier in repository metadata
before it exists.
