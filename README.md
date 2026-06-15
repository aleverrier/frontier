# frontier

Frontier decoder export.

This repository contains the working C++-accelerated frontier decoder path:

- native C++ binary frontier engine (`_frontier_native`)
- forward-only, backward-only, and forward/backward committee decoding
- `deadline_reorder` for the forward pass
- `backward_deadline_reorder` for the backward pass
- Gross/BB and surface-code DEM inspection, replay, and benchmark CLIs

## Scope

This repo is scoped to running the frontier decoder on BB/Gross and surface-code
detector-side matrices. It intentionally ships only:

- frontier Python wrappers and the C++ native extension
- BB/Gross, generalized/bivariate bicycle, rotated-surface, and planar
  surface-code matrix/DEM builders
- DEM inspection, replay, smoke, and BB144/Gross benchmark CLIs
- small tests and reproducibility notes

Legacy BP/min-sum decoder families, triangle-quotient decoders, polar DEM
experiments, and old research benchmark harnesses are not part of this export.
See `docs/FILE_SCOPE.md` for the file-by-file audit.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e .
python setup.py build_ext --inplace
```

## Smoke Test

```bash
python -m pytest -q
frontier-smoke --K 16 --Delta 100 --shots 3
```

## DEM Matrices

Use `frontier-dem-info` to build a supported detector-side matrix family and
print the dimensions used by the decoder:

```bash
frontier-dem-info \
  --backend bravyi_depth7 \
  --p-location 0.004 \
  --column-order deadline_reorder
```

For the accepted BB144/Gross split-sector DEM benchmark, the expected dimensions
are `D_X = D_Z = 936 x 8784`, `O_X = O_Z = 12 x 8784`, with 12 noisy
syndrome-extraction rounds.

This repo bundles the Gross `[[144,12,12]]` CSS matrices, the BB144/Gross
memory X/Z Stim circuits for the exact rates listed below, and a materialized
split-sector DEM snapshot for `bravyi_depth7`, `p=0.001`. The bundled
`p=0.001` DEM files live under `grosscode/assets/gross144/dem/`:

- `bravyi_depth7_p0p001_memory_X_detector.npz`
- `bravyi_depth7_p0p001_memory_X_logical.npz`
- `bravyi_depth7_p0p001_memory_X_priors.npy`
- `bravyi_depth7_p0p001_memory_Z_detector.npz`
- `bravyi_depth7_p0p001_memory_Z_logical.npz`
- `bravyi_depth7_p0p001_memory_Z_priors.npy`

The bundled BB144/Gross Stim circuit rates are exactly `0.0005`, `0.001`,
`0.002`, `0.003`, `0.004`, `0.005`, and `0.006`, for both `memory_X` and
`memory_Z`.

For custom Gross assets, set `GROSSCODE_ASSET_ROOT` to a directory containing:

- `gross_code/HX_Gross_144_12_12.mtx`
- `gross_code/HZ_Gross_144_12_12.mtx`
- `stim_circuits/BB[[144,12,12]],memory_X,error_rate=...,syndrome_rounds=12.stim`
- `stim_circuits/BB[[144,12,12]],memory_Z,error_rate=...,syndrome_rounds=12.stim`

## DEM Replay

Use `frontier-sample-rows` to generate independent detector-side DEM sample
rows from the same matrices and priors used by the decoder:

```bash
frontier-sample-rows \
  --out sample_rows.csv \
  --backend bravyi_depth7 \
  --p-location 0.001 \
  --shots 1000 \
  --seed 20260615
```

Then use `frontier-replay` on those matched sample rows:

```bash
frontier-replay \
  --sample-rows sample_rows.csv \
  --out-dir results/frontier_replay \
  --code bb144 \
  --backend bravyi_depth7 \
  --p-location 0.001 \
  --shot-start 0 \
  --shot-stop 999 \
  --K 512 \
  --Delta 12 \
  --direction-mode fwd_bwd_committee \
  --engine native_binary \
  --column-order deadline_reorder \
  --backward-column-order backward_deadline_reorder \
  --cpus 1 \
  --progress-every-shards 1
```

For CPU-saturated runs on macOS, launch from Terminal and set
`FRONTIER_NATIVE_BATCH_THREADS` to the number of native worker threads you
want the extension to use.

## Benchmark

```bash
frontier-bb144-benchmark \
  --sample-rows sample_rows.csv \
  --backend bravyi_depth7 \
  --p-location 0.001 \
  --column-order deadline_reorder \
  --K 512 \
  --Delta 12 \
  --payload replay
```

The benchmark path reports the accepted Gross split-sector DEM dimensions:
`D_X = D_Z = 936 x 8784`, `O_X = O_Z = 12 x 8784`, with 12 noisy rounds.

## Reproducing BB144/Gross DEM Results

For the accepted BB144/Gross split-sector DEM benchmark, `p` is passed as
`--p-location`. Use `--backend bravyi_depth7` unless you intentionally want a
non-default circuit family. The accepted detector-side matrices are
`D_X = D_Z = 936 x 8784`, `O_X = O_Z = 12 x 8784`, with 12 noisy
syndrome-extraction rounds.

The default Gross benchmark works from the bundled files. Set
`GROSSCODE_ASSET_ROOT` only when intentionally overriding those bundled assets
with a custom asset root.

To reproduce a published full-frame row exactly, use the same matched
`sample_rows.csv` that was used for that row and pass the intended probability
as `--p-location`. This repo does not check in large sample corpora. The CSV
must contain both `memory_X` and `memory_Z` rows for the requested shot ids and
must include at least:
`scope`, `shot`, `seed`, `truth_syndrome`, and `truth_logical`.

To produce a fresh reproducible 10k-shot BB144/Gross DEM sample at `p=0.001`
and decode it with `Delta=12`, `K=512`, and the forward/backward committee:

```bash
frontier-sample-rows \
  --out results/bb144_p0p001_sample_rows.csv \
  --backend bravyi_depth7 \
  --p-location 0.001 \
  --shots 10000 \
  --seed 20260615 \
  --progress-every-rows 1000

frontier-replay \
  --sample-rows results/bb144_p0p001_sample_rows.csv \
  --out-dir results/bb144_p0p001_frontier_replay_k512_Delta12 \
  --code bb144 \
  --backend bravyi_depth7 \
  --p-location 0.001 \
  --shot-start 0 \
  --shot-stop 9999 \
  --K 512 \
  --Delta 12 \
  --direction-mode fwd_bwd_committee \
  --engine native_binary \
  --column-order deadline_reorder \
  --backward-column-order backward_deadline_reorder \
  --cpus 10 \
  --shards-per-side 20 \
  --native-batch-size 64 \
  --progress-every-shards 1
```

Replay writes `summary_by_scope.csv`, `per_shot_rows.csv`,
`combined_per_shot_rows.csv`, `run_metadata.json`, and `report.md`. The
`combined` row in
`summary_by_scope.csv` is the full logical frame error rate over paired
`memory_X`/`memory_Z` shots.

## Matrices

The repo contains matrix builders, not checked-in static `.dem`, `.mtx`, `.npy`,
or `.npz` matrix files.

- Gross split-sector detector-side DEM:
  `grosscode.dem.builder.build_split_sector_problem(...)` returns `D_X`, `D_Z`,
  `O_X`, `O_Z`, priors, and metadata. For `backend="bravyi_depth7"` and
  `p=0.001`, this loads the bundled materialized DEM snapshot. For other
  bundled public rates, it builds the DEM from the bundled Stim circuits.
- Rotated-surface code-capacity checks:
  `grosscode.codes.rotated_surface.load_rotated_surface_code(...)` constructs
  `HX/HZ` in repo, and rotated-surface DEMs are generated from Stim
  `rotated_memory_x/z` circuits for backends such as `rotated_surface_d5`.
- Standard planar surface-code checks:
  `grosscode.codes.surface.standard_surface_checks(distance)` returns the CSS
  `HX/HZ` sparse matrices for the standard planar surface code.

Minimal examples:

```python
from grosscode.codes.surface import standard_surface_checks
from grosscode.dem.builder import build_split_sector_problem

hx, hz = standard_surface_checks(5)
print(hx.shape, hz.shape)

problem = build_split_sector_problem(backend="bravyi_depth7", error_rate=0.004)
print(problem.D_X.shape, problem.D_Z.shape)
```
