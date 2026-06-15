# Polar DEM Framework

This package implements a correctness-first experimental framework for the question:

- does a detector error model become easier to decode after the binary Arikan transform over GF(2)?
- does the gap statistic `g_max` predict the SCL list size needed for near-MAP decoding?

## Conventions

- All vectors are column vectors.
- For `N = 2^m`, the transform is the binary Arikan matrix `A_N = F^{⊗ m}` over GF(2) with
  `F = [[1, 0], [1, 1]]`.
- The implementation does **not** substitute the real Walsh-Hadamard matrix or the usual row-vector polar encoder formula.
- The fast transform in [arikan.py](arikan.py) applies exactly this GF(2) map, and because `A_N^{-1} = A_N`, the same routine inverts it.
- Dynamic frozen constraints are derived from `Q = M_det A_N` by right-to-left Gaussian elimination so each pivot is the rightmost `1` in its row.
- Structural scans on realistic DEMs are kept separate from actual decoding experiments:
  - scans only inspect `Q`, the frozen/free profile, and `g_max`;
  - decoding benchmarks are run only on standalone exact instances where the syndrome constraints are complete.

## Layout

- [gf2.py](gf2.py): dense GF(2) linear algebra and right-pivot elimination
- [arikan.py](arikan.py): Arikan transform and within-window orderings
- [dynamic_frozen.py](dynamic_frozen.py): affine frozen-rule extraction and `g_max`
- [sc_posterior.py](sc_posterior.py): exact SC posterior evaluator for the specified column-vector transform plus Monte Carlo reliability estimation
- [scl_decoder.py](scl_decoder.py): SCL with affine frozen bits
- [exact_map.py](exact_map.py): exhaustive fault-MAP / logical-MAP on tiny standalone instances
- [experiments_small.py](experiments_small.py): exact small-instance benchmark
- [experiments_large_scan.py](experiments_large_scan.py): large DEM structural window scan
- [plotting.py](plotting.py): saved plots
- [adapters.py](adapters.py): thin bridge to the repo's DEM loaders

## Reproduction scripts

- small exact benchmark:

```bash
python -m tools.polar_dem_small_benchmark --results-dir results/_polar_dem_small
```

- large structural scan on the maintained Gross split-sector detector-side DEM:

```bash
python -m tools.polar_dem_large_scan --results-dir results/_polar_dem_large
```
