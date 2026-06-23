# Reproducibility

This repository separates fast smoke checks from publication-grade
reproducibility. Smoke checks verify that the export is installed and exercises
the expected matrix/decoder paths; publication claims require archived inputs,
exact environment constraints, and enough samples for the reported statistical
resolution.

## Tier 1: Smoke And Sanity Reproducibility

```bash
python3 -m venv .venv
source .venv/bin/activate
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

When exact constraints are available, install with:

```bash
python -m pip install -e . -c constraints/<validated-environment>.txt
```

The committed `constraints/py314-macos-validated.txt` file was captured from
the MacBook Python 3.14.2 validation environment. It is one available validated
environment, not a default for all users. For cluster or Linux reproduction,
use the capture pattern in `constraints/README.md` from the validated target
environment. The Ubuntu Python 3.12 CI constraints are not pinned in this
checkout; see `constraints/py312-ubuntu-ci.TODO.md`.

The `bravyi_depth7`, `p=0.001` Gross/BB144 DEM smoke path should report
detector matrices `936x8784` and logical matrices `12x8784` for both
`memory_X` and `memory_Z`.

The 10k-shot BB144/Gross `p=0.001` replay command in `README.md` is
smoke-scale. If it observes no failures, report the result as below the
resolution of that 10k-shot sample, not as evidence that FER is zero.

## Tier 2: Publication-Grade Reproducibility

Publication-grade reproduction should record:

- exact git commit hash or release tag;
- archived release DOI once available;
- exact Python version;
- exact dependency constraints;
- compiler, platform, and native-extension build status;
- sample-corpus DOI, path, or immutable identifier;
- random seeds and sampling method;
- command lines;
- output files archived with the result;
- expected summary columns and schemas, using `docs/BENCHMARK_SCHEMA.md` for
  benchmark result CSV/JSON summaries;
- confidence intervals and statistical interpretation for FER estimates.

Large publication sample corpora are not checked into this repository. Archive
them separately and cite their immutable identifier in reports.

## Paper Plot Reproduction

Paper plot reproduction lives in `paper/plots/`.

```bash
python paper/plots/scripts/reproduce_plots.py --list
python paper/plots/scripts/reproduce_plots.py --all --strict --out-dir /tmp/frontier-paper-plots
```

The manifest `paper/plots/manifest.csv` maps each figure or panel to the
minimal plot-ready data file, plotting script, output path, data kind, source,
generation command, environment, status, and caveats. `paper/plots/data/` is for
small summary CSV files and same-stem JSON sidecars, not raw per-shot corpora.

This checkout contains compact plot-ready summary tables, JSON sidecars, and
Matplotlib renderers for every current paper figure in the recorded
`frontier_decoder2.tex` inventory. Rows marked `reproducible` regenerate their
listed PNG output from committed summary data. Rows marked `support-data` are
committed companion inputs used by another renderer and are skipped by `--all`.

Raw paper sample corpora and full publication-scale run outputs are not
committed here. A plot being reproducible from committed summary tables does
not mean that the underlying simulation is reproducible from this repo alone.
Future rows with missing tables must use `data-missing` or
`external-archive-needed`; rows with tables but no renderer must use
`script-missing`.

Regenerate paper-data checksums with:

```bash
python -m tools.asset_manifest --root paper/plots/data --title "Paper Plot Data Manifest" > paper/plots/data/MANIFEST.md
```

## Reporting Results

Every reported decoder result should include the fields in
`docs/BENCHMARK_SCHEMA.md` when it is exported as a benchmark CSV/JSON summary.
At minimum, narrative reports should include:

- commit hash or release tag;
- machine/CPU;
- Python version;
- operating system and compiler;
- dependency lock or constraints file;
- `frontier_native_available`;
- worker count;
- `native_batch_size`;
- `K`, `Delta`, `score_alpha`, `metric_mode`, and `int_metric_scale`;
- matrix family, side/scope, and dimensions;
- sample-corpus identifier;
- number of shots;
- failure decomposition when available;
- confidence intervals when estimating FER.
