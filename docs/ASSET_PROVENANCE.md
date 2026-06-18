# Asset Provenance

This document describes bundled research assets under
`grosscode/assets/gross144`. It records current repository facts without
inferring upstream DOI, institution, or standalone asset-license claims.

| Path or glob | Asset type | Role in workflow | Source/provenance | Generation method | License/provenance status | Expected dimensions or rates | Checksum status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `grosscode/assets/gross144/gross_code/*.mtx` | Gross/BB144 CSS parity-check matrices | Default Gross `[[144,12,12]]` matrix inputs for split-sector DEM construction | Bundled in this repository as part of the maintained Gross/BB144 asset set. No separate upstream DOI/source statement is included. | Bundled matrix files loaded by `grosscode.codes.gross144`. | Repository `NOTICE` records that bundled assets are not relicensed; no separate asset license statement is included. | Gross `[[144,12,12]]` CSS matrices. | Listed in `docs/ASSET_MANIFEST.md`. |
| `grosscode/assets/gross144/stim_circuits/*.stim` | Stim memory circuits | Public BB144/Gross memory X/Z circuit inputs for DEM construction at supported rates | Bundled in this repository as part of the maintained Gross/BB144 asset set. No separate upstream DOI/source statement is included. | Bundled Stim files resolved by `grosscode.circuits.backends`. | Repository `NOTICE` records that bundled assets are not relicensed; no separate asset license statement is included. | `memory_X` and `memory_Z` at rates `0.0005`, `0.001`, `0.002`, `0.003`, `0.004`, `0.005`, `0.006`; 12 syndrome rounds. | Listed in `docs/ASSET_MANIFEST.md`. |
| `grosscode/assets/gross144/dem/*.npz` | Sparse detector/logical matrices | Materialized `bravyi_depth7`, `p=0.001` split-sector DEM snapshot used by default smoke/replay paths | Bundled in this repository as part of the maintained Gross/BB144 asset set. No separate upstream DOI/source statement is included. | Derived detector-side snapshot consumed by `grosscode.dem.builder.build_split_sector_problem`. | Repository `NOTICE` records that bundled assets are not relicensed; no separate asset license statement is included. | Detector matrices `936x8784`; logical matrices `12x8784`. | Listed in `docs/ASSET_MANIFEST.md`. |
| `grosscode/assets/gross144/dem/*.npy` | Prior-probability arrays | Priors paired with the materialized DEM snapshot | Bundled in this repository as part of the maintained Gross/BB144 asset set. No separate upstream DOI/source statement is included. | Derived from the same `bravyi_depth7`, `p=0.001` split-sector DEM snapshot. | Repository `NOTICE` records that bundled assets are not relicensed; no separate asset license statement is included. | One prior per DEM column; expected column count `8784`. | Listed in `docs/ASSET_MANIFEST.md`. |
| `grosscode/assets/gross144/dem/*.json` | DEM metadata JSON | Snapshot metadata used to report matrix dimensions and noisy rounds | Bundled in this repository as part of the maintained Gross/BB144 asset set. No separate upstream DOI/source statement is included. | Written alongside the materialized DEM snapshot. | Repository `NOTICE` records that bundled assets are not relicensed; no separate asset license statement is included. | Expected detector `936x8784`, logical `12x8784`, 12 noisy rounds. | Listed in `docs/ASSET_MANIFEST.md`. |

Generate or refresh the checksum manifest with:

```bash
python -m tools.asset_manifest > docs/ASSET_MANIFEST.md
```

Do not treat Apache-2.0 repository licensing as a relicensing statement for
provenance-ambiguous third-party research assets.
