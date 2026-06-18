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
and sidecars for `frontier_decoder2.tex`. It does not include raw per-shot paper
sample corpora, and no row is marked `reproducible` until a committed renderer
can regenerate the listed output from the committed data.
