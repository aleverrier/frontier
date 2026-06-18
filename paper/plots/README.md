# Paper Plot Reproduction

This directory contains the manifest, compact plot-ready data, and Matplotlib
renderers needed to regenerate the paper plot PNGs from committed summary
tables. It intentionally does not contain raw per-shot corpora or full run
output trees.

## Current Status

The plot inventory currently targets the available `frontier_decoder2.tex`
manuscript source, sha256
`288da4629eddc7038f38f3ae2948d358b57a018544eb6f2591a7aebe5f8e5380`.
The newest available rendered PDF candidate was `Frontier_decoder-2.pdf`,
sha256 `1406a80c7448f6964634da42d4f520b0cf03f97b60899034ac2aff1219cb29c5`.
A literal `frontier_decoder2(2).tex` file was not present in the working
environment during this cleanup pass.

Every current paper figure has a committed renderer and is
plot-reproducible from committed summary data. This is not the same as
simulation reproducibility: the raw sample corpora and full publication-scale
run outputs are not committed here.

## Commands

List declared figures and table status:

```bash
python paper/plots/scripts/reproduce_plots.py --list
```

Regenerate all plot-reproducible outputs:

```bash
python paper/plots/scripts/reproduce_plots.py --all --strict
```

Write outputs to a custom directory:

```bash
python paper/plots/scripts/reproduce_plots.py --all --strict --out-dir /tmp/frontier-paper-plots
```

Regenerate one figure ID:

```bash
python paper/plots/scripts/reproduce_plots.py --figure <figure_id> --strict
```

Generated files are written to `paper/plots/outputs/` by default. That
directory is ignored by git except for its `.gitignore` file.

## Figure Inventory

| Figure ID | Paper figure | Data file(s) | Renderer | Command | Output file(s) | Plot reproducible? | Simulation reproducible from repo? | Raw-corpus archive status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `frontier_schematic` | Figure 1 | `paper/plots/data/fig_frontier_schematic_elements.csv` | `paper/plots/scripts/plot_frontier_schematic.py` | `python paper/plots/scripts/reproduce_plots.py --figure frontier_schematic --strict` | `paper/plots/outputs/fig_frontier_decoder_schematic.png` | yes, from committed schematic elements | not applicable; explanatory schematic | no raw corpus |
| `algorithm` | Figure 2 | `paper/plots/data/fig_algorithm_recap_states.csv` | `paper/plots/scripts/plot_algorithm_recap.py` | `python paper/plots/scripts/reproduce_plots.py --figure algorithm --strict` | `paper/plots/outputs/fig_algorithm_recap_step1_retained.png`; `paper/plots/outputs/fig_algorithm_recap_step2_evolved.png`; `paper/plots/outputs/fig_algorithm_recap_step3_merged.png`; `paper/plots/outputs/fig_algorithm_recap_step4_pruned.png` | yes, from committed compact state table | no; source run state table only | raw/full run not committed |
| `surface_threshold` | Figure 3 | `paper/plots/data/fig_surface_threshold.csv` | `paper/plots/scripts/plot_surface_threshold.py` | `python paper/plots/scripts/reproduce_plots.py --figure surface_threshold --strict` | `paper/plots/outputs/fig_surface_threshold_log_paper.png`; `paper/plots/outputs/fig_surface_average_retained_list_size_paper.png` | yes, from committed summary table | no | raw sample corpora not committed |
| `color_threshold` | Figure 4 | `paper/plots/data/fig_color_threshold_fer.csv`; `paper/plots/data/fig_color_threshold_retained_states.csv` | `paper/plots/scripts/plot_color_threshold.py` | `python paper/plots/scripts/reproduce_plots.py --figure color_threshold --strict` | `paper/plots/outputs/fig_color_code_frontierfast_fer_vs_p_selected_d9_d13_d17_d21.png`; `paper/plots/outputs/fig_color_code_rung0_mean_retained_states_paper.png` | yes, from committed summary tables | no | raw sample corpora not committed |
| `surface_memory_z_dem_mwpm` | Figure 5 | `paper/plots/data/fig_surface_memory_z_dem_mwpm.csv` | `paper/plots/scripts/plot_surface_memory_z_dem_mwpm.py` | `python paper/plots/scripts/reproduce_plots.py --figure surface_memory_z_dem_mwpm --strict` | `paper/plots/outputs/fig_surface_memory_z_dem_frontier_vs_mwpm.png` | yes, from committed summary table | no | raw sample corpora not committed |
| `bb72_dem_circuit` | Figure 6 | `paper/plots/data/fig_bb72_dem_fer_vs_p.csv`; `paper/plots/data/fig_bb72_dem_fer_vs_mean_states.csv` | `paper/plots/scripts/plot_bb72_dem.py` | `python paper/plots/scripts/reproduce_plots.py --figure bb72_dem_circuit --strict` | `paper/plots/outputs/fig_bb72_dem_fer_vs_p_paper.png`; `paper/plots/outputs/fig_bb72_dem_fer_vs_mean_states_paper.png` | yes, from committed summary tables | no | raw sample corpora not committed |
| `gross_dem_circuit` | Figure 7 | `paper/plots/data/fig_gross_dem_fer_vs_p.csv`; `paper/plots/data/fig_gross_dem_fer_vs_avg_retained.csv` | `paper/plots/scripts/plot_gross_dem.py` | `python paper/plots/scripts/reproduce_plots.py --figure gross_dem_circuit --strict` | `paper/plots/outputs/fig_bb144_recent_frontier_fer_vs_p_paper.png`; `paper/plots/outputs/fig_bb144_p001_fer_vs_avg_retained_list_size_paper.png` | yes, from committed summary tables | no | raw sample corpora not committed |
| `gross_dem_avg_retained` | Figure 8 | `paper/plots/data/fig_gross_dem_avg_vs_peak_retained.csv` | `paper/plots/scripts/plot_gross_dem.py` | `python paper/plots/scripts/reproduce_plots.py --figure gross_dem_avg_retained --strict` | `paper/plots/outputs/fig_bb144_p001_avg_vs_peak_retained_list_size_paper.png` | yes, from committed summary table | no | raw sample corpora not committed |
| `gross_dem_avg_retained_duplicate` | Figure 9 | `paper/plots/data/fig_gross_dem_fer_vs_avg_retained.csv` | `paper/plots/scripts/plot_gross_dem.py` | `python paper/plots/scripts/reproduce_plots.py --figure gross_dem_avg_retained_duplicate --strict` | `paper/plots/outputs/fig_bb144_p001_fer_vs_avg_retained_list_size_paper.png` | yes, from committed summary table | no | raw sample corpora not committed |
| `transition_evals` | Figure 10 | `paper/plots/data/fig_transition_evals_tail.csv`; support data `paper/plots/data/fig_transition_evals_percentiles.csv` | `paper/plots/scripts/plot_transition_evals.py` | `python paper/plots/scripts/reproduce_plots.py --figure transition_evals --strict` | `paper/plots/outputs/fig_bb144_p001_transition_eval_hist_paper.png` | yes, from committed tail table plus committed percentile guide table | no | raw timing rows not committed |
| `failure_decomposition` | Figure 11 | `paper/plots/data/fig_failure_decomposition.csv` | `paper/plots/scripts/plot_failure_decomposition.py` | `python paper/plots/scripts/reproduce_plots.py --figure failure_decomposition --strict` | `paper/plots/outputs/fig_bb144_p002_failure_decomposition_avg_axis.png` | yes, from committed diagnostic summary table | no | raw sample corpora not committed |

The paper source reused the TeX label `fig:gross_dem_avg_retained` for two
different Gross/BB144 retained-list figures. The manifest preserves this fact
in `paper_reference` while assigning distinct manifest IDs:
`gross_dem_avg_retained` for average-vs-peak retained list size and
`gross_dem_avg_retained_duplicate` for the standalone FER-vs-average-retained
figure.

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
`support-data`, `data-missing`, `script-missing`, `external-archive-needed`,
and `TODO`.

- `reproducible`: committed data plus a committed renderer regenerate the
  listed output.
- `support-data`: committed data used by another renderer; skipped by `--all`.
- `script-missing`: committed data exists but no renderer exists.
- `data-missing`: no committed data exists.
- `external-archive-needed`: raw or summary data must be supplied externally.
- `TODO`: only for clearly unfinished manifest rows.

## Data And Provenance Caveats

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

Expected committed files for the current plot reproduction contract are:

- `paper/plots/manifest.csv`
- `paper/plots/data/*.csv`
- `paper/plots/data/*.json`
- `paper/plots/data/README.md`
- `paper/plots/data/MANIFEST.md`
- `paper/plots/scripts/reproduce_plots.py`
- `paper/plots/scripts/plot_utils.py`
- `paper/plots/scripts/plot_*.py`
- `paper/plots/outputs/.gitignore`
