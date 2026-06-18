# Paper Plot Data

This directory contains minimal plot-ready summary tables for the recorded paper
source `frontier_decoder2.tex`, sha256
`288da4629eddc7038f38f3ae2948d358b57a018544eb6f2591a7aebe5f8e5380`.
The available rendered PDF candidate is `Frontier_decoder-2.pdf`, sha256
`1406a80c7448f6964634da42d4f520b0cf03f97b60899034ac2aff1219cb29c5`.

The files are deliberately compact. They are summary tables, generated schematic inputs, or compact figure-state tables. They are not raw per-shot corpora.

The current checkout includes Matplotlib renderers for every paper figure listed
in `paper/plots/manifest.csv`. The table
`fig_transition_evals_percentiles.csv` is marked `support-data`: it is consumed
by the transition-evaluation renderer but is not a standalone figure output.

`MANIFEST.md` is generated with:

```bash
python -m tools.asset_manifest --root paper/plots/data --title "Paper Plot Data Manifest" > paper/plots/data/MANIFEST.md
```

## Current Tables And Columns

Every CSV has a same-stem JSON sidecar. The exact columns needed for each current figure table are:

- `paper/plots/data/fig_algorithm_recap_states.csv` (194 data rows):
  `stage, state_key, parent_state_key, local_state, merged_from_count, rank, kept, prefix_mass_P, future_score_F, score_P_plus_alpha_F, detector_mask, logical_mask, next_column_index, next_boundary_column_index, alpha, Delta, K, best_score_after_next_column, cutoff_score_after_next_column, closure_rejected`

- `paper/plots/data/fig_bb72_dem_fer_vs_mean_states.csv` (11 data rows):
  `Delta, K, p_location, trials, fail_total, fer, ci_low, ci_high, fer_per_round, per_round_ci_low, per_round_ci_high, logical_fail, syndrome_fail, exception_fail, truth_missing_terminal, truth_present_but_not_selected, bad_ranking, mean_decode_s, mean_transition_evals, mean_post_states_per_decoder, mean_post_states_total, mean_post_states_x, mean_post_states_z, matrix_rows, matrix_cols, logical_rows, noisy_rounds, result_dir, plot_group, plot_x, plot_y, decoder, mean_work, work_metric, mean_decode_ms, source, source_decoder`

- `paper/plots/data/fig_bb72_dem_fer_vs_p.csv` (15 data rows):
  `decoder, corpus, p, frames, fail, fer, fer_lo_1sigma, fer_hi_1sigma, fer_per_round, fer_per_round_lo_1sigma, fer_per_round_hi_1sigma, logical_fail, syndrome_fail, exception_fail, mean_ms, p95_ms, decode_label, notes, source_decoder, decoder_label, ci_low95, ci_high95, per_round_ci_low95, per_round_ci_high95`

- `paper/plots/data/fig_color_threshold_fer.csv` (68 data rows):
  `circuit, distance, p, shots, rung0_fail_total, rung0_logical_fail, rung0_syndrome_fail, rung0_exception_fail, rung0_success, rung0_fer, rung0_wilson95_lo, rung0_wilson95_hi, final_fail_total, final_fer, rescued_net_fail_delta, committee_disagreed, escalated, primary_forward, primary_backward, matrix_rows, matrix_cols, K, Delta, escalation_K, escalation_Delta, plot_group, plot_x, plot_y, plot_ci_low, plot_ci_high`

- `paper/plots/data/fig_color_threshold_retained_states.csv` (68 data rows):
  `circuit, distance, rounds, p, source_p, stim_path, noise_model, matrix_rows, matrix_cols, logical_rows, matrix_nnz, frontier_max_active_detectors, shots, fail_total, logical_fail, syndrome_fail, exception_fail, fer, fer_per_round, wilson95_lo, wilson95_hi, K, Delta, selected_K, selected_Delta, escalation_K, escalation_Delta, score_alpha, decoder_mode, engine_requested, engines_seen, selected_forward, selected_backward, committee_disagreed, escalated, escalation_fraction, mean_decode_s, p95_decode_s, p99_decode_s, mean_transition_evals_total, mean_primary_transition_evals_total, mean_escalation_transition_evals_total, mean_selected_transition_evals, mean_pre_prune_state_count, mean_post_prune_state_count, max_pre_prune_state_count_max, max_post_prune_state_count_max, mean_sum_pre_prune_state_count, mean_sum_post_prune_state_count, elapsed_s, seed, plot_group, plot_x, plot_y`

- `paper/plots/data/fig_failure_decomposition.csv` (15 data rows):
  `Delta, K, p_location, source_trials, scan_trials, avg_retained_list_size, mean_peak_retained_list_size, fail_total, fer, empty_terminal_frontier, logical_support_loss, terminal_ranking_failure, syndrome_fail, truth_missing_terminal, bad_ranking, empty_terminal_frontier_rate, logical_support_loss_rate, terminal_ranking_failure_rate, no_path_rate, truth_missing_rate, bad_ranking_rate, logical_fail, exception_fail, scan_result_dir`

- `paper/plots/data/fig_frontier_schematic_elements.csv` (45 data rows):
  `element_type, element_id, source, target, x, y, matrix_row, matrix_column, processed_count, is_processed, is_active, is_logical, is_closed, value, notes`

- `paper/plots/data/fig_gross_dem_avg_vs_peak_retained.csv` (8 data rows):
  `decoder, Delta, K, status, trials, fail_total, fer, ci_low, ci_high, logical_fail, syndrome_fail, exception_fail, x_avg_retained_list_size_per_side_column, y_mean_peak_retained_list_size_per_side_trace, scan_mean_frame_peak_list_size, scan_transition_evals_total_mean, scan_trials, source, point_label`

- `paper/plots/data/fig_gross_dem_fer_vs_avg_retained.csv` (10 data rows):
  `decoder, Delta, K, status, trials, fail_total, fer, ci_low, ci_high, fer_per_round, per_round_ci_low, per_round_ci_high, logical_fail, syndrome_fail, exception_fail, truth_missing_terminal, truth_present_but_not_selected, bad_ranking, x_mean_list_size_per_side, x_coordinate_source, series, x_avg_retained_list_size_per_side_column, scan_transition_evals_total_mean, scan_mean_peak_side_list_size, scan_mean_frame_peak_list_size, scan_fail_total, scan_trials, plot_group, plot_x, source_decoder`

- `paper/plots/data/fig_gross_dem_fer_vs_p.csv` (12 data rows):
  `decoder, p_location, fail_total, trials, fer, fer_low95, fer_high95, fer_per_round, fer_per_round_low95, fer_per_round_high95, config, status, source, matrix, source_decoder, decoder_label`

- `paper/plots/data/fig_surface_memory_z_dem_mwpm.csv` (8 data rows):
  `distance, decoder, status, shots, fail_total, fer, fer_lo, fer_hi, fer_per_round, fer_per_round_lo, fer_per_round_hi, matrix_rows, matrix_cols, avg_retained_list_size, avg_retained_list_size_stderr, retained_estimate_samples, frontier_K, frontier_Delta`

- `paper/plots/data/fig_surface_threshold.csv` (76 data rows):
  `distance, p, K, Delta, source, status, shots, fail_total, logical_fail, syndrome_fail, exception_fail, fer, fer_lo95, fer_hi95, mean_states_total, joint_matrix_rows, joint_matrix_cols`

- `paper/plots/data/fig_transition_evals_percentiles.csv` (7 data rows):
  `metric, transition_evals`

- `paper/plots/data/fig_transition_evals_tail.csv` (62 data rows):
  `bin_left, bin_right, count, tail_count_ge_bin_left, tail_fraction_ge_bin_left, tail_ci_low95, tail_ci_high95`

## JSON Sidecar Schema

Each CSV must have a same-stem `.json` sidecar containing:

- `description`
- `columns` with units and definitions for every CSV column
- `source_command` or `raw_source`
- `commit_hash` or `release_version`
- `code_version`
- `dependency_constraints_file`
- `random_seeds`, if relevant
- `sample_count`, if relevant
- `decoder_settings`, including `K`, `Delta`, `score_alpha`, `metric_mode`, and `int_metric_scale` when relevant
- `confidence_interval_method`, if plotted
- `csv_sha256`
- `caveats`
- `plot_reproducibility`
- `simulation_reproducibility`
- `raw_corpus`
- `source_artifact`
- `source_checkout_hash`
- `renderer`
- `output_file`
- `output_files`, when one CSV feeds multiple manifest outputs

If a required field is unknown, leave it null or empty and explain the limitation in `caveats`; do not fill it with a guess.

## Provenance Notes

- Source paths from local or scratch machines were normalized to stable labels
  such as `better-beam:` and `scratch-better-beam-results:`. These labels are
  local/source artifact handles, not public DOIs or archive identifiers.
- Rows in `paper/plots/manifest.csv` are `reproducible` only when a committed
  renderer regenerates the listed PNG from committed summary tables.
- Offline diagnostic columns such as truth-missing or bad-ranking failure classes use ground-truth labels for analysis only and must not be used as online decoder decisions.
- Raw sample corpora and full publication-scale run outputs are not committed
  here; the sidecars record plot reproducibility from summary data separately
  from simulation reproducibility.
