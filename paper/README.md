# Paper Materials

This directory is reserved for paper-specific reproduction material that belongs
with the public software release.

- `plots/README.md`: paper-plot reproduction status, commands, and data policy.
- `plots/manifest.csv`: figure-to-data/script/output manifest.
- `plots/data/`: minimal plot-ready summary tables and sidecar metadata when
  paper data are available.
- `plots/scripts/`: plotting and manifest-listing entry points.
- `plots/outputs/`: local generated plot outputs, ignored by git by default.

The current checkout includes a figure manifest plus compact plot-ready tables
and sidecars for the recorded `frontier_decoder2.tex` inventory. It also
includes Matplotlib renderers for every current paper figure, so
`python paper/plots/scripts/reproduce_plots.py --all --strict` regenerates the
listed local PNG outputs from committed summary data. It does not include raw
per-shot paper sample corpora.
