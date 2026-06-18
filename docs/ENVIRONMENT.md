# Environment Variables

## Stable Public Variables

| Variable | Where Used | Expected Values | Purpose |
| --- | --- | --- | --- |
| `GROSSCODE_ASSET_ROOT` | `grosscode.utils.paths.resolve_gross_asset_root` | Path to an asset directory containing `gross_code/` and `stim_circuits/` | Optional override for bundled Gross/BB144 assets. Leave unset to use `grosscode/assets/gross144`. |
| `FRONTIER_CACHE_DIR` | `grosscode.utils.paths.resolve_cache_root` | Path to a writable cache directory | Optional cache root for generated files. If unset, the cache root is `$XDG_CACHE_HOME/frontier` when `XDG_CACHE_HOME` is set, `~/Library/Caches/frontier` on macOS, and `~/.cache/frontier` otherwise. |
| `MPLCONFIGDIR` | `grosscode.utils.paths.ensure_mplconfigdir`, `tools.frontier_sample_replay`, `tools.frontier_bb144_benchmark` | Path to a writable Matplotlib config/cache directory | Replay and benchmark tools set a temporary default when needed to avoid writing to a user-global config path. |
| `FRONTIER_NATIVE_BATCH_THREADS` | `native/_frontier_native.cpp` | Positive integer thread count | Native extension worker-thread count for batch decode paths. Set this for CPU-saturated replay/benchmark runs. |

## Debugging Variables

These variables are useful for local debugging or profiling but are not stable
workflow API.

| Variable | Where Used | Expected Values | Purpose |
| --- | --- | --- | --- |
| `FRONTIER_NATIVE_PROFILE` | `native/_frontier_native.cpp` | Truthy/falsey toggle as parsed by the native extension | Enables native profiling output. |
| `FRONTIER_SAMPLE_REPLAY_DISABLE_FLAT_NATIVE_REPLAY` | `tools.frontier_sample_replay` | `1`, `true`, `on`, or `yes` to disable | Disables the flat native replay fast path for comparison/debugging. |

## Internal Native Debugging Toggles

The C++ extension also contains native debugging and optimization toggles. These
are not stable public API and should be used only while profiling or bisecting
native behavior:

| Variable | Where Used | Expected Values | Purpose |
| --- | --- | --- | --- |
| `FRONTIER_NATIVE_DISABLE_BATCH_WORKSPACE_REUSE` | `native/_frontier_native.cpp` | `1`, `true`, `on`, or `yes` | Disable native batch workspace reuse. |
| `FRONTIER_NATIVE_DISABLE_ONE_PASS_PRUNE` | `native/_frontier_native.cpp` | `1`, `true`, `on`, or `yes` | Disable the one-pass prune optimization. |
| `FRONTIER_NATIVE_ONE_PASS_PRUNE_MIN` | `native/_frontier_native.cpp` | Positive integer | Override the minimum candidate count for one-pass pruning. |
| `FRONTIER_NATIVE_ENABLE_FINAL_PRUNE_SORT` | `native/_frontier_native.cpp` | `1`, `true`, `on`, or `yes` | Force-enable final prune sorting. |
| `FRONTIER_NATIVE_DISABLE_FINAL_PRUNE_SORT` | `native/_frontier_native.cpp` | `1`, `true`, `on`, or `yes` | Disable final prune sorting when not force-enabled. |
| `FRONTIER_NATIVE_ENABLE_CLOSE_EMPTY_SPLIT_MERGE` | `native/_frontier_native.cpp` | `1`, `true`, `on`, or `yes` | Enable close-empty split merge behavior. |
| `FRONTIER_NATIVE_DISABLE_CLOSE_EMPTY_SPLIT_MERGE` | `native/_frontier_native.cpp` | `1`, `true`, `on`, or `yes` | Disable close-empty split merge after it is enabled. |
| `FRONTIER_NATIVE_DISABLE_COMPACT_CLOSE_EMPTY_SPLIT_MERGE` | `native/_frontier_native.cpp` | `1`, `true`, `on`, or `yes` | Disable compact close-empty split merge behavior. |
| `FRONTIER_NATIVE_DISABLE_NO_MERGE_TRANSITION` | `native/_frontier_native.cpp` | `1`, `true`, `on`, or `yes` | Disable the no-merge transition optimization. |
| `FRONTIER_NATIVE_DISABLE_SINGLE_PARENT_STEP` | `native/_frontier_native.cpp` | `1`, `true`, `on`, or `yes` | Disable the single-parent native step optimization. |
| `FRONTIER_NATIVE_DISABLE_SMALL_STATE_STEP` | `native/_frontier_native.cpp` | `1`, `true`, `on`, or `yes` | Disable the small-state native step optimization. |
| `FRONTIER_NATIVE_DISABLE_SMALL_PATTERN_TABLE` | `native/_frontier_native.cpp`, `tools.frontier_decoder` | `1`, `true`, `on`, or `yes` | Disable the small-pattern table path and include that choice in Python native-model cache keys. |
| `FRONTIER_NATIVE_PHASE_TIMING` | `tools.frontier_decoder` | `1`, `true`, `on`, or `yes` | Request native phase timing collection in Python-created native model specs. |
| `FRONTIER_NATIVE_FORCE_FULL_KEY` | `tools.frontier_decoder` | `1`, `true`, `on`, or `yes` | Include full-key mode in Python native-model cache/spec construction. |

Do not rely on these internal names in published workflows.
