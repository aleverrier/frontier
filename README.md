# frontier

Frontier decoder export.

This repository contains the working C++-accelerated frontier decoder path
selected from the `better-beam` research tree:

- native C++ binary frontier engine (`_frontier_native`)
- forward-only, backward-only, and forward/backward committee decoding
- `deadline_reorder` for the forward pass
- `backward_deadline_reorder` for the backward pass
- Gross split-sector DEM replay and benchmark CLIs

The two-stage decoder is intentionally not part of the public CLI or Python
API exposed by this export.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
python setup.py build_ext --inplace
```

## Smoke Test

```bash
python -m pytest -q
frontier-smoke --K 16 --Delta 100 --shots 3
```

## DEM Replay

Use `frontier-replay` for matched sample rows:

```bash
frontier-replay \
  --sample-rows path/to/sample_rows.csv \
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
  --backend bravyi_depth7 \
  --p-location 0.001 \
  --column-order deadline_reorder \
  --K 512 \
  --Delta 12 \
  --payload replay
```

The benchmark path reports the accepted Gross split-sector DEM dimensions:
`D_X = D_Z = 936 x 8784`, `O_X = O_Z = 12 x 8784`, with 12 noisy rounds.

## Matrices

The repo contains matrix builders, not checked-in static `.dem`, `.mtx`, `.npy`,
or `.npz` matrix files.

- Gross split-sector detector-side DEM:
  `grosscode.dem.builder.build_split_sector_problem(...)` returns `D_X`, `D_Z`,
  `O_X`, `O_Z`, priors, and metadata. For `backend="bravyi_depth7"`, this uses
  the public Gross Stim circuits and `HX/HZ` matrices from `qtanner-ssf`; set
  `GROSSCODE_QTANNER_ROOT` or `QTANNER_ROOT` to that checkout.
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
