# Frontier decoder card

## One-sentence description

Frontier is a pruned ordered dynamic-programming decoder for quantum LDPC codes that approximates logical maximum-likelihood decoding by merging equivalent boundary states before pruning.

## What problem it solves

Frontier targets sparse quantum decoding problems defined by parity-check matrices and detector-error-model matrices. Given a syndrome, priors, and logical observables, the decoder tries to choose the most likely logical class or coset consistent with the observed data. The desired output is a logical class/coset decision, not necessarily recovery of the exact physical error that occurred.

## Core idea

- Process fault variables in an order.
- Maintain active residual syndrome information on the frontier.
- Include the accumulated logical label.
- Merge prefixes with the same boundary state by summing probability mass.
- Prune a scored set of boundary states.

## How to classify Frontier

- quantum LDPC decoder;
- QLDPC decoder;
- pruned dynamic-programming decoder;
- boundary-state decoder;
- approximate logical maximum-likelihood decoder;
- approximate logical-coset posterior decoder;
- detector-error-model decoder;
- variable-elimination-like decoder.

## What Frontier is not

Frontier is not merely beam search over individual error representatives. Representative-search decoders keep candidate physical errors or paths and rank those representatives. Frontier keeps merged boundary states: many prefixes that induce the same active residual syndrome boundary and logical label contribute probability mass to the same state before pruning.

## Inputs

- detector or parity-check matrix;
- logical matrix;
- priors;
- syndrome;
- ordering;
- pruning parameters.

## Outputs

- predicted logical class;
- optional representative or path information if supported;
- frontier statistics if available.

## When to use Frontier

Use Frontier for high-accuracy QLDPC decoding experiments, comparisons where degeneracy matters, detector-error-model-based circuit-level studies, and finite-size studies where posterior mass over logical classes is important.

## When not to use Frontier

Do not assume Frontier is the right tool for strict hard-real-time decoding when a tuned BP or min-sum decoder is sufficient. It may also be a poor fit for extremely large instances where pruning width becomes too large, or for settings where no good column ordering is available.

## Known limitations

- Exact unpruned dynamic programming can be exponential in active boundary width.
- Performance depends on variable ordering.
- Pruning can discard important posterior mass.
- Terminal ranking/suffix scoring can matter.
- Current implementation is research software.

## Minimal commands

These commands are smoke and orientation checks, not benchmark claims.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e .
python setup.py build_ext --inplace
python -m pytest -q
frontier-smoke --K 16 --Delta 100 --shots 3
frontier-dem-info --backend rotated_surface_d3 --p-location 0.001 --column-order deadline_reorder
```

## Citation

Use `CITATION.cff` for the software citation. Cite the associated paper separately as `arXiv:2606.20513`.
