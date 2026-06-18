# Examples

These examples are intentionally tiny and safe to run after an editable install.

- `minimal_decode.py` builds the same two-factor model used by the smoke test
  and calls `decode_frontier` plus `decode_frontier_committee`.
- `inspect_dem.py` loads the `rotated_surface_d3` DEM family and prints matrix
  dimensions and column-order metadata.
- `replay_rotated_surface_d3.sh` creates a tiny sample-row CSV in a temporary
  directory, replays it with small `K`/`Delta`, and prints the output location.

Run them after installing the package and building the native extension:

```bash
python setup.py build_ext --inplace
python examples/minimal_decode.py
python examples/inspect_dem.py
bash examples/replay_rotated_surface_d3.sh
```

If your shell does not have a `python` alias, run the shell example with
`PYTHON_BIN=/path/to/python bash examples/replay_rotated_surface_d3.sh`.
