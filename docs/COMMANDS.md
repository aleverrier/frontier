# Frontier Command Index

This page is a quick command reference. `README.md` remains the source for the
full BB144/Gross reproduction workflow.

## `frontier-smoke`

- Purpose: run the tiny two-factor frontier smoke benchmark.
- Module: `tools/frontier_decoder.py`.
- Minimal command:

```bash
frontier-smoke --K 16 --Delta 100 --shots 3
```

- Output files: none.
- Common failures:
  - Import errors usually mean the editable install was not activated.
  - Native fallback is acceptable for this smoke test, but decoder internals
    should still be validated after rebuilding `_frontier_native`.

## `frontier-dem-info`

- Purpose: load supported detector-side DEM matrix families and print decoder
  dimensions.
- Module: `tools/dem_loader.py`.
- Minimal command:

```bash
frontier-dem-info --backend rotated_surface_d3 --p-location 0.001 --column-order deadline_reorder
```

- Output files: none; writes a CSV table to stdout.
- Common failures:
  - Missing Gross/BB144 assets: restore bundled files or set
    `GROSSCODE_ASSET_ROOT` to a directory with `gross_code/` and
    `stim_circuits/`.
  - Unsupported column order: use one of the values printed by `--help`.

## `frontier-sample-rows`

- Purpose: generate independent detector-side DEM sample rows for
  `frontier-replay`.
- Module: `tools/frontier_sample_rows.py`.
- Minimal command:

```bash
frontier-sample-rows --out sample_rows.csv --backend rotated_surface_d3 --p-location 0.001 --shots 4 --seed 20260615
```

- Output files:
  - `sample_rows.csv`
  - `sample_rows_metadata.json` by default, unless `--metadata-out` is set
- Common failures:
  - Existing output file without `--allow-existing`.
  - Large `--shots` values can create large CSVs; keep generated corpora out of
    commits.

## `frontier-replay`

- Purpose: decode matched sample rows, write per-shot rows, summaries, metadata,
  and a short report.
- Module: `tools/frontier_sample_replay.py`.
- Minimal command:

```bash
frontier-replay \
  --sample-rows sample_rows.csv \
  --out-dir results/frontier_replay \
  --code rotated_surface_d3 \
  --backend rotated_surface_d3 \
  --p-location 0.001 \
  --shot-start 0 \
  --shot-stop 3 \
  --K 16 \
  --Delta 100 \
  --direction-mode fwd_bwd_committee \
  --engine auto \
  --column-order deadline_reorder \
  --backward-column-order backward_deadline_reorder \
  --cpus 1 \
  --progress-every-shards 1
```

- Output files:
  - `run_metadata.json`
  - `per_shot_rows.csv`
  - `combined_per_shot_rows.csv` when both memory sectors are present
  - `summary_by_scope.csv`
  - `report.md`
- Common failures:
  - Missing requested sample rows for the shot range.
  - Non-empty output directory without `--allow-existing`.
  - Native engine unavailable: rerun `python setup.py build_ext --inplace`.

## `frontier-bb144-benchmark`

- Purpose: run a focused BB144/Gross native timing probe over explicit sample
  rows.
- Module: `tools/frontier_bb144_benchmark.py`.
- Minimal command:

```bash
frontier-bb144-benchmark \
  --sample-rows sample_rows.csv \
  --backend bravyi_depth7 \
  --p-location 0.001 \
  --column-order deadline_reorder \
  --K 512 \
  --Delta 12 \
  --rows-per-scope 10 \
  --repeats 3 \
  --payload replay
```

- Output files: none; writes timing and work metrics to stdout.
- Common failures:
  - Too few rows per requested scope in `--sample-rows`.
  - Native extension not built.
  - Running this without an explicit sample corpus is unsupported by design.

## BB144/Gross Happy-Path Mini Workflow

For full details and interpretation guidance, use the README reproduction
section. The compact flow is:

```bash
frontier-dem-info \
  --backend bravyi_depth7 \
  --p-location 0.001 \
  --column-order deadline_reorder

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

The expected Gross split-sector DEM dimensions are `D_X = D_Z = 936 x 8784`
and `O_X = O_Z = 12 x 8784`, with 12 noisy syndrome-extraction rounds.
