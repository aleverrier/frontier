# Environment Variables

## Stable Public Variables

| Variable | Purpose |
| --- | --- |
| `GROSSCODE_ASSET_ROOT` | Optional override for bundled Gross/BB144 assets. The directory must contain `gross_code/` and `stim_circuits/`. Leave unset to use `grosscode/assets/gross144`. |
| `FRONTIER_CACHE_DIR` | Optional cache root for generated files used by `grosscode.utils.paths`. Defaults to a temporary cache directory. |
| `MPLCONFIGDIR` | Matplotlib cache/config directory. Replay and benchmark tools set a temporary default when needed to avoid writing to a user-global config path. |
| `FRONTIER_NATIVE_BATCH_THREADS` | Native extension worker-thread count for batch decode paths. Set this for CPU-saturated replay/benchmark runs. |
| `FRONTIER_NATIVE_PROFILE` | Enables native profiling output when set to a truthy value supported by the native extension. Use only for debugging or optimization. |
| `FRONTIER_SAMPLE_REPLAY_DISABLE_FLAT_NATIVE_REPLAY` | Disables the flat native replay fast path in `tools/frontier_sample_replay.py` for debugging. |

## Internal Native Debugging Toggles

The C++ extension also contains native debugging and optimization toggles. These
are not stable public API and should be used only while profiling or bisecting
native behavior:

- `FRONTIER_NATIVE_DISABLE_BATCH_WORKSPACE_REUSE`
- `FRONTIER_NATIVE_DISABLE_ONE_PASS_PRUNE`
- `FRONTIER_NATIVE_ONE_PASS_PRUNE_MIN`
- `FRONTIER_NATIVE_ENABLE_FINAL_PRUNE_SORT`
- `FRONTIER_NATIVE_DISABLE_FINAL_PRUNE_SORT`
- `FRONTIER_NATIVE_ENABLE_CLOSE_EMPTY_SPLIT_MERGE`
- `FRONTIER_NATIVE_DISABLE_CLOSE_EMPTY_SPLIT_MERGE`
- `FRONTIER_NATIVE_DISABLE_COMPACT_CLOSE_EMPTY_SPLIT_MERGE`
- `FRONTIER_NATIVE_DISABLE_NO_MERGE_TRANSITION`
- `FRONTIER_NATIVE_DISABLE_SINGLE_PARENT_STEP`
- `FRONTIER_NATIVE_DISABLE_SMALL_STATE_STEP`
- `FRONTIER_NATIVE_DISABLE_SMALL_PATTERN_TABLE`

Python-side native dispatch debugging also checks:

- `FRONTIER_NATIVE_PHASE_TIMING`
- `FRONTIER_NATIVE_FORCE_FULL_KEY`
- `FRONTIER_NATIVE_DISABLE_SMALL_PATTERN_TABLE`

Do not rely on these internal names in published workflows.
