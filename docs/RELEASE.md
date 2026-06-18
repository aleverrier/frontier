# Release Process

Use this checklist for citable public releases.

## Pre-Release Checklist

- Confirm the working tree is clean.
- Update the version in `pyproject.toml`.
- Update `CHANGELOG.md`.
- Ensure `CITATION.cff` version and date are current.
- Confirm `ACKNOWLEDGEMENTS.md` and `docs/ACADEMIC_METADATA.md` do not contain
  release-blocking placeholders.
- Confirm asset provenance and license status in `docs/ASSET_PROVENANCE.md`.
- Generate or update checksums with:

```bash
python -m tools.asset_manifest > docs/ASSET_MANIFEST.md
```

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
```

## Tag And Archive

```bash
git tag -a vX.Y.Z -m "frontier vX.Y.Z"
git push origin vX.Y.Z
```

Archive the tagged release on Zenodo or an institutional repository.

## After DOI Assignment

- Add the software DOI to `CITATION.cff`.
- Add preferred citation metadata if the paper/software citation target is
  chosen.
- Link the DOI from `README.md` if appropriate.

Do not claim a DOI in repository metadata before one exists.
