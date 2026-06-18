# Paper Plot Reproduction

This directory defines the repository contract for reproducing paper figures.
It intentionally does not contain fabricated plot values.

## Current Status

The current paper inventory is taken from `frontier_decoder2.tex`, sha256
`d1abc814aab7ec6e8bacfab0af31b95d7f84b4a01170b648c2ceefacd5ae153e`.

This checkout now contains minimal plot-ready summary tables and JSON sidecars
for every figure or figure panel in that TeX source. The manifest rows are not
yet reproducible: they are still marked `script-missing`, not `reproducible`,
because this repository does not yet contain figure-specific renderers that
regenerate the published PNGs. The CLI therefore lists the rows and skips them
honestly.

No raw per-shot corpora are committed here. The committed tables are compact
summaries or compact figure-input state tables.

## Current Figure Tables

- `frontier_schematic`: explanatory generated schematic; data file
  `paper/plots/data/fig_frontier_schematic_elements.csv`.
- `algorithm`: four-panel BB72 algorithm recap; data file
  `paper/plots/data/fig_algorithm_recap_states.csv`.
- `surface_threshold`: rotated-surface code-capacity FER and retained-list
  panels; data file `paper/plots/data/fig_surface_threshold.csv`.
- `color_threshold`: color-code FER and retained-list panels; data files
  `paper/plots/data/fig_color_threshold_fer.csv` and
  `paper/plots/data/fig_color_threshold_retained_states.csv`.
- `surface_memory_z_dem_mwpm`: surface memory-Z detector-side DEM comparison;
  data file `paper/plots/data/fig_surface_memory_z_dem_mwpm.csv`.
- `bb72_dem_circuit`: BB72 detector-side DEM FER-vs-p and retained-list sweep;
  data files `paper/plots/data/fig_bb72_dem_fer_vs_p.csv` and
  `paper/plots/data/fig_bb72_dem_fer_vs_mean_states.csv`.
- `gross_dem_circuit`: Gross/BB144 detector-side DEM FER-vs-p and p=0.001
  retained-list sweep; data files `paper/plots/data/fig_gross_dem_fer_vs_p.csv`
  and `paper/plots/data/fig_gross_dem_fer_vs_avg_retained.csv`.
- `gross_dem_avg_retained`: Gross/BB144 p=0.001 average-vs-peak retained-list
  table; data file `paper/plots/data/fig_gross_dem_avg_vs_peak_retained.csv`.
- `gross_dem_avg_retained_duplicate`: the attached TeX reuses the
  `fig:gross_dem_avg_retained` label for a standalone FER-vs-average-retained
  figure; the manifest assigns a distinct id and reuses
  `paper/plots/data/fig_gross_dem_fer_vs_avg_retained.csv`.
- `transition_evals`: Gross/BB144 transition-evaluation tail curve plus
  percentile guides; data files `paper/plots/data/fig_transition_evals_tail.csv`
  and `paper/plots/data/fig_transition_evals_percentiles.csv`.
- `failure_decomposition`: Gross/BB144 p=0.002 failure-decomposition table;
  data file `paper/plots/data/fig_failure_decomposition.csv`.

## Setup

Use the normal project setup:

```bash
python -m pip install -e .
python setup.py build_ext --inplace
```

For exact environments, install with a validated constraints file:

```bash
python -m pip install -e . -c constraints/<validated-environment>.txt
```

## Commands

List declared figures and table status:

```bash
python paper/plots/scripts/reproduce_plots.py --list
```

Attempt all reproducible figures:

```bash
python paper/plots/scripts/reproduce_plots.py --all
```

Write outputs to a custom directory:

```bash
python paper/plots/scripts/reproduce_plots.py --all --out-dir /tmp/frontier-paper-plots
```

Reproduce one figure after a verified manifest row, data file, and renderer are
added:

```bash
python paper/plots/scripts/reproduce_plots.py --figure <figure_id>
```

Generated files are written to `paper/plots/outputs/` by default. That
directory is ignored by git except for its `.gitignore` file.

## Manifest Contract

`manifest.csv` maps each paper figure or panel to the minimal data and script
needed to recreate it. Required columns are:

- `figure_id`
- `panel_id`
- `title`
- `paper_reference`
- `data_file`
- `plotting_script`
- `output_file`
- `data_kind`
- `data_source`
- `generation_command`
- `environment`
- `status`
- `notes`

Allowed `data_kind` values are `raw`, `derived-summary`, `digitized`,
`synthetic-demo`, and `TODO`. Allowed `status` values are `reproducible`,
`data-missing`, `script-missing`, `external-archive-needed`, and `TODO`.

Only rows with `status == reproducible` may be plotted by default. Rows with
`script-missing` have committed data but no committed renderer. Rows with
`data-missing` or `external-archive-needed` must explain the missing columns or
archive requirement in `notes`.

## Required Data Columns

For each current committed table, the exact required columns are the CSV
headers recorded in `paper/plots/data/README.md` and mirrored in the same-stem
JSON sidecar. Future paper plot-ready CSVs should keep the smallest faithful
summary table needed to regenerate the published panel. Do not commit raw
per-shot corpora when summary data are sufficient.

Every plot-ready CSV should include these normalized columns when applicable,
or explain in the sidecar why a column is not applicable:

- `figure_id`: manifest figure identifier.
- `panel_id`: manifest panel identifier, empty for single-panel figures.
- `series_id`: curve, decoder, code, or experiment label.
- `x_value`: numeric horizontal-axis value.
- `x_label`: horizontal-axis semantic name, for example `p_location`.
- `x_unit`: horizontal-axis unit, or empty for dimensionless values.
- `y_value`: numeric plotted value.
- `y_label`: vertical-axis semantic name, for example
  `logical_frame_error_rate`.
- `y_unit`: vertical-axis unit, or empty for dimensionless values.
- `n_samples`: sample count used for the plotted estimate, if applicable.
- `n_failures`: failure count used for the plotted estimate, if applicable.
- `ci_low`: lower confidence interval endpoint, if plotted or reported.
- `ci_high`: upper confidence interval endpoint, if plotted or reported.
- `ci_method`: confidence interval method, if applicable.
- `code`: code family, for example `bb144` or `rotated_surface_d3`.
- `backend`: circuit or DEM backend.
- `scope`: `memory_X`, `memory_Z`, `combined`, or another declared scope.
- `p_location`: physical error probability when relevant.
- `K`: frontier beam cap when relevant.
- `Delta`: frontier pruning threshold when relevant.
- `direction_mode`: decoder direction mode when relevant.
- `engine`: requested decoder engine when relevant.
- `seed`: random seed or seed family when relevant.
- `commit`: source commit or release version that generated the summary.
- `environment`: constraints file or environment note.
- `source_run_id`: immutable run, archive, or corpus identifier.
- `notes`: caveats needed to interpret the row.

For timing or work plots, replace sample/failure columns only when they are not
meaningful, and add explicit columns such as `decode_seconds`,
`transition_evals`, or percentile columns with units in the sidecar.

Each CSV must have a JSON sidecar with the same stem. The sidecar must include
the fields listed in `paper/plots/data/README.md`.

## Data Provenance Caveats

- Do not edit plot image files manually.
- Do not mark a row `reproducible` unless committed data and a committed script
  regenerate the listed output file.
- Do not change plot data without updating its JSON sidecar and
  `paper/plots/data/MANIFEST.md`.
- Do not add digitized data unless the source and digitization procedure are
  explicitly documented.
- Do not add `synthetic-demo` rows for paper figures.
- If raw corpora are too large for git, archive them externally and commit only
  minimal derived summary tables plus sidecar metadata and checksums.

Regenerate the data checksum manifest with:

```bash
python -m tools.asset_manifest --root paper/plots/data --title "Paper Plot Data Manifest" > paper/plots/data/MANIFEST.md
```

## Expected Files

Expected committed files for the current scaffold are:

- `paper/plots/manifest.csv`
- `paper/plots/data/*.csv`
- `paper/plots/data/*.json`
- `paper/plots/data/README.md`
- `paper/plots/data/MANIFEST.md`
- `paper/plots/scripts/reproduce_plots.py`
- `paper/plots/outputs/.gitignore`
