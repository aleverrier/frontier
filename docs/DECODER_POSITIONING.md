# Positioning Frontier among QLDPC decoders

Frontier is best described as a pruned dynamic-programming decoder over an
ordered sequence of fault variables. It is not BP+OSD, beam search, a
Tesseract-like representative search, or a tensor-network decoder, although it
has points of contact with all of them. The key distinction is the retained
object: Frontier keeps boundary states keyed by active residual syndrome and
logical label, accumulates probability mass for equivalent prefixes, and then
prunes that retained frontier according to explicit `K` and `Delta` parameters.
Comparisons should therefore use the same code family, matrix or DEM, noise
model, sample corpus, and reporting convention.

| Decoder family | Retained object | Uses probability mass? | Degeneracy treatment | Typical bottleneck |
| --- | --- | --- | --- | --- |
| BP/min-sum | Tanner-graph messages and local beliefs | Approximately, through local beliefs or costs rather than a global sum | Mostly implicit through local marginalization | Iteration count, trapping sets, convergence failures |
| BP+OSD/BP+LSD | BP reliabilities plus an ordered or localized postprocessing list | Usually through reliabilities or costs, not by summing every equivalent correction | Partly handled by postprocessing, but often representative- or list-oriented unless explicitly grouped | OSD order, list/local solve size, Gaussian elimination or local-rank work |
| Representative search/beam | Candidate error representatives, including Tesseract-like or beam-ranked paths | Usually no; the beam ranks representatives rather than total coset mass | Equivalent representatives compete unless the implementation deduplicates or groups them | Beam width, branching factor, ranking heuristic, missed representatives |
| Tensor network/variable elimination | Tensors, separators, or eliminated-variable tables | Yes, when contractions sum over eliminated variables; exact if untruncated | Explicit summation over internal variables can account for degeneracy | Treewidth, separator/table size, contraction order, truncation error |
| Frontier | Boundary states keyed by active residual syndrome and logical label | Yes; merged prefixes accumulate mass before pruning | Equivalent prefixes with the same active residual and logical label are merged until pruning | Frontier size, `K`/`Delta`, column ordering, active-boundary width, transition count |

## Why the retained object matters

The retained object determines what uncertainty remains available to the
decoder. BP/min-sum keeps local message information, so it can be fast but may
lose global coset structure. Representative search and beam-style decoders keep
promising corrections, so they can miss cases where many individually modest
corrections combine into a high-mass logical class. Tensor-network and variable
elimination methods keep exact or truncated factor tables, so they can represent
probability mass directly but become expensive when separator width grows.
Frontier keeps a boundary-state table for an ordered factorization: when two
prefixes induce the same active residual syndrome and logical label, they are
merged and their scores are combined. This makes the ordering, retained frontier
size, and pruning policy central parts of the decoder rather than incidental
implementation details.

## Recommended comparison baselines

Use baselines that answer different questions, and label each one precisely:

- BP or min-sum as a fast iterative baseline, with schedule, damping, scaling,
  and iteration count reported.
- BP+OSD or BP+LSD as a strong sparse-matrix postprocessing baseline, with OSD
  order or LSD locality parameters reported.
- Representative search, beam search, or Tesseract-like search as a
  most-likely-representative family, with beam width, ranking score, and any
  deduplication rule reported.
- Tensor-network or variable-elimination decoding for small or structured
  instances where exact or controlled-truncation references are feasible.
- Domain-standard baselines such as MWPM for surface-code detector-error-model
  studies, when the comparison matrix and noise model make that baseline
  meaningful.

Do not mix code-capacity parity-check matrices, detector-side DEM matrices, and
effective models under one label. If a benchmark uses a non-default matrix
family or a surrogate model, say so in the command, table, caption, and
interpretation.

## What to report in benchmarks

At minimum, report:

- logical failure rate;
- confidence intervals;
- average and peak frontier size;
- median/p95/p99 runtime or transition count;
- pruning parameters `K` and `Delta`;
- column ordering;
- code family and DEM/noise model.

Also report enough setup detail for reproduction: decoder version or commit,
sample-corpus generation method, number of trials, random seeds when applicable,
logical-success convention, and whether results were measured on code-capacity,
phenomenological, detector-side DEM, or another explicitly named model.
