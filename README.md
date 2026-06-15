# frontier

Fast FrontierFast decoder export.

This repository contains the working C++-accelerated FrontierFast decoder path
selected from the `better-beam` research tree:

- native C++ binary FrontierFast engine (`_frontier_fast_native`)
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
frontier-smoke --K 16 --delta 100 --shots 3
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
`FRONTIERFAST_NATIVE_BATCH_THREADS` to the number of native worker threads you
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
