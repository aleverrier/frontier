# Benchmark result schema

This document defines the preferred CSV and JSON schema for benchmark result
summaries in this repository. It is a documentation contract only: do not add
or infer benchmark rows unless an existing script, committed summary table, or
archived source corpus supports the values.

For CSV exports, every result file should include the required columns below in
its header. For JSON exports, use a list of objects with the same required keys,
or an object containing a `results` list whose entries use these keys. Unknown
or intentionally unmeasured values should be left blank in CSV or set to `null`
in JSON, with the reason documented in `source_data`, a sidecar metadata file,
or the surrounding report. Do not fill missing fields with guessed values.

| Column | Meaning |
| --- | --- |
| `code_family` | Code or benchmark family, for example BB144/Gross, rotated surface, or planar surface. |
| `code_parameters` | Human-readable code parameters, such as distance, block length, logical count, or benchmark identifier. |
| `noise_model` | Noise or sampling model used for the rows, including memory sector or side when relevant. |
| `circuit_rounds` | Number of noisy syndrome-extraction rounds, or blank/null for non-circuit or code-capacity studies. |
| `matrix_shape_detector` | Detector/parity-check matrix dimensions used by the decoder, formatted as `rows x columns`. |
| `matrix_shape_logical` | Logical-observable matrix dimensions, formatted as `rows x columns`, or blank/null when not applicable. |
| `decoder` | Decoder family or implementation name. |
| `decoder_version_or_commit` | Git commit, release tag, or external decoder version used for the row. |
| `decoder_parameters` | Serialized decoder settings sufficient to identify the configuration. |
| `column_order` | Column or fault-variable ordering used by the decoder. |
| `physical_error_rate` | Physical error probability or location probability used for sampling. |
| `shots` | Number of full logical frames or benchmark trials represented by the row. |
| `logical_failures` | Number of full logical frame failures. This should not be a syndrome-only failure count. |
| `logical_failure_rate` | `logical_failures / shots` for the full logical frame definition used by the benchmark. |
| `confidence_interval_method` | Statistical interval method, for example Wilson, bootstrap, exact binomial, or blank/null if absent. |
| `ci_low` | Lower confidence interval endpoint for `logical_failure_rate`, or blank/null if absent. |
| `ci_high` | Upper confidence interval endpoint for `logical_failure_rate`, or blank/null if absent. |
| `average_frontier_size` | Mean retained frontier size, averaged over the same scope as the result row, or blank/null if absent. |
| `peak_frontier_size` | Maximum retained frontier size observed for the row, or blank/null if absent. |
| `median_transition_count` | Median transition count or transition-evaluation count per trial, or blank/null if absent. |
| `p95_transition_count` | 95th percentile transition count or transition-evaluation count per trial, or blank/null if absent. |
| `p99_transition_count` | 99th percentile transition count or transition-evaluation count per trial, or blank/null if absent. |
| `walltime_seconds` | End-to-end wall time for the row or run segment, in seconds, or blank/null if unavailable. |
| `hardware_summary` | Short machine, CPU/GPU, worker-count, and native-extension summary needed to interpret timing. |
| `source_script` | Script, console command, or renderer that produced the row. |
| `source_data` | Input sample corpus, committed summary table, sidecar, DOI, archive identifier, or local path policy. |
| `paper_figure` | Paper figure/panel identifier using this row, or blank/null if not tied to a figure. |

## CSV header

```csv
code_family,code_parameters,noise_model,circuit_rounds,matrix_shape_detector,matrix_shape_logical,decoder,decoder_version_or_commit,decoder_parameters,column_order,physical_error_rate,shots,logical_failures,logical_failure_rate,confidence_interval_method,ci_low,ci_high,average_frontier_size,peak_frontier_size,median_transition_count,p95_transition_count,p99_transition_count,walltime_seconds,hardware_summary,source_script,source_data,paper_figure
```

## JSON shape

```json
{
  "schema": "frontier-benchmark-result-v1",
  "results": []
}
```

Each object in `results` must include all required keys listed above. Additional
keys are allowed when a specific benchmark needs more audit detail, such as
failure decomposition or per-round metrics.

## Paper plot data note

Existing `paper/plots` data are compact summary and reproduction artifacts for
rendering documented figures. They are not publication-scale raw simulation
corpora unless a specific file or sidecar explicitly says so. When raw corpora
or full publication-scale run outputs are absent, keep that absence explicit
instead of backfilling inferred benchmark rows.
