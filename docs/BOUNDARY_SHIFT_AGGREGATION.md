# Boundary-Shift Aggregation of Low-Weight Frontier Polymers

Status: exact deterministic size-1/2 quotient-aggregation diagnostic,
2026-07-28.

This note follows `BOUNDED_HYPERGRAPH_OVERLAP.md`. That note showed that the
bounded-hypergraph theorem applies to the retained detector error models, but
that the independent-polymer product relaxation is numerically vacuous.
Here we preserve one piece of structure that the product bound discarded:
different polymer families that induce the same relative boundary shift are
aggregated before the fractional power is taken.

For surface d3, this recovers up to 34 orders of magnitude at a fixed
fractional exponent. After optimizing the exponent, however, the exact
size-1/2 contribution remains above one at \(K=1024\) and \(K=2048\).
Boundary-shift aggregation alone therefore does not yield a practical
certificate. The next bound must also enforce compatibility or move closer
to the exact comparison spectrum.

This is an exact calculation within the stated truncated polymer universe.
It performs no decoding or Monte Carlo.

## 1. Three nested low-weight relaxations

Fix a cut \(t\), Chernoff order \(\theta\), and let
\(\mathcal P_t^{\le2}\) be the visible size-1 and size-2 open-prefix
polymers. Write
\[
w_\gamma
=
\prod_{j\in\gamma}a_{\theta,j},
\qquad
s_\gamma=D_t\mathbf1_\gamma
\tag{1}
\]
for the polymer activity and its active-detector-plus-logical boundary
shift.

Drop compatibility temporarily, but preserve the XOR boundary shift. Define
the exact subset partition
\[
Z_t^{\le2}(\xi)
=
\sum_{\substack{
\mathcal F\subseteq\mathcal P_t^{\le2}\\
\bigoplus_{\gamma\in\mathcal F}s_\gamma=\xi}}
\prod_{\gamma\in\mathcal F}w_\gamma.
\tag{2}
\]
The empty family contributes one to \(Z_t^{\le2}(0)\).

For \(0<\rho<1\), compare:
\[
\begin{aligned}
G_{t,\rho}^{\le2}
&=
\sum_{\xi\ne0}Z_t^{\le2}(\xi)^\rho,
\\
U_{t,\rho}^{\le2}
&=
\prod_{\gamma\in\mathcal P_t^{\le2}}
(1+w_\gamma^\rho)-1,
\\
E_{t,\rho}^{\le2}
&=
\exp\left(
\sum_{\gamma\in\mathcal P_t^{\le2}}w_\gamma^\rho
\right)-1.
\end{aligned}
\tag{3}
\]
These obey
\[
\boxed{
G_{t,\rho}^{\le2}
\le
U_{t,\rho}^{\le2}
\le
E_{t,\rho}^{\le2}.
}
\tag{4}
\]

For the first inequality, group nonempty families by their XOR shift and use
\((\sum_i x_i)^\rho\le\sum_i x_i^\rho\), then discard the nonempty
zero-shift families. The second inequality is
\(\log(1+x)\le x\).

The previous note used \(E\). The middle expression \(U\) removes only the
exponential relaxation. The new expression \(G\) additionally preserves
boundary-shift aggregation and excludes all nonempty families whose total
shift vanishes.

## 2. Relation to the rigorous full bound

Let \(Z_t(\xi)\) be (2) with polymers of every size. Dropping compatibility
from the bounded-hypergraph pointwise theorem gives
\[
\omega_t(\xi)\le Z_t(\xi).
\tag{5}
\]
Consequently,
\[
D^{\rm cap}
\le
K^{-1/\rho}
\sum_t
\left[
\sum_{\xi\ne0}Z_t(\xi)^\rho
\right]^{1/\rho}.
\tag{6}
\]

The computed \(Z_t^{\le2}\) is a nonnegative partial contribution:
adding a polymer of shift \(s\) updates
\[
Z'(\xi)=Z(\xi)+wZ(\xi+s)\ge Z(\xi).
\tag{7}
\]
Thus the reported size-1/2 expression is a lower bound on the right-hand
side of (6), not a lower bound on actual cap loss. If this partial expression
already exceeds one, the full compatibility-dropped grouped certificate is
necessarily vacuous.

Compatibility is still not enforced in (2). Subsets may contain overlapping
or adjacent polymers that could not be distinct connected components of one
prefix difference. The calculation isolates boundary aggregation from that
remaining source of overcount.

## 3. Exact XOR dynamic program

At a surface-code cut, encode the active detector bits followed by the
logical bits. The retained surface d3 models have at most 11 active detectors
and one logical bit, so the quotient space has at most
\[
2^{11+1}=4096
\tag{8}
\]
labels.

Initialize
\[
Z^{(0)}(0)=1,\qquad Z^{(0)}(\xi\ne0)=0.
\]
For polymer \(m\), activity \(w_m\), and shift \(s_m\), update all quotient
labels by
\[
\boxed{
Z^{(m+1)}(\xi)
=
Z^{(m)}(\xi)
+
w_m Z^{(m)}(\xi+s_m).
}
\tag{9}
\]
This is the exact subset recurrence for (2). It costs
\(O(|\mathcal P_t^{\le2}|2^{|\Gamma_t|+k})\) per cut and is practical for
surface d3. The implementation uses log-domain integration for the final
cap expression, so small fractional exponents do not overflow.

## 4. Fixed-\(\rho\) reduction

The profile uses the same setting as the bounded-hypergraph audit:

```text
backend = rotated_surface_d3
p_location = 0.001
column_order = deadline_reorder
score_alpha = 0.8
theta = 5/16
```

The table reports the ratio between the old exponential expression and the
new shift-grouped expression after summing the rooted cut terms. This ratio
is independent of \(K\).

| Scope | \(\rho=0.50\) | \(\rho=0.75\) | \(\rho=0.99\) |
| --- | ---: | ---: | ---: |
| memory X | \(2.13\times10^{34}\) | \(3.03\times10^4\) | 1.215 |
| memory Z | \(1.18\times10^{28}\) | \(7.81\times10^3\) | 1.207 |

Aggregation is therefore extremely valuable at small \(\rho\). Near
\(\rho=1\), concavity is weak and grouping buys only about 20%.

For example, at \(\rho=0.5\) and \(K=1024\), the surface-X partial
right-hand side falls from \(1.20\times10^{35}\) to \(5.63\), while
surface Z falls from \(6.12\times10^{28}\) to \(5.19\). This is an enormous
reduction, but it remains above one.

## 5. Optimizing the fractional exponent

The artifact evaluates a deterministic grid
\[
\rho=0.05,0.06,\ldots,0.99
\tag{10}
\]
and reports the smallest value on that grid for each \(K\).

| \(K\) | X minimum | X \(\rho\) | Z minimum | Z \(\rho\) |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 285.558 | 0.99 | 269.676 | 0.99 |
| 512 | 8.617 | 0.99 | 8.137 | 0.99 |
| 1024 | 4.183 | 0.85 | 3.859 | 0.80 |
| 2048 | 1.395 | 0.45 | 1.299 | 0.50 |
| 4096 | 0.0715 | 0.05 | 0.116 | 0.05 |
| 8192 | \(6.82\times10^{-8}\) | 0.05 | \(1.11\times10^{-7}\) | 0.05 |

The grid-estimated continuous \(K\) threshold where the partial grouped
right-hand side first falls below one is

| Scope | Threshold \(K\) | Optimizing \(\rho\) |
| --- | ---: | ---: |
| memory X | 2346.45 | 0.37 |
| memory Z | 2317.66 | 0.45 |

These are grid diagnostics, not a proof of the continuous optimum in
\(\rho\). The margin at \(K=1024\) is nevertheless large: every tested
\(\rho\) leaves the partial expression at least 3.86. Higher polymers can
only increase it.

The \(K=4096\) row should not be interpreted as a useful pruning guarantee.
There are at most 4096 active-plus-logical boundary labels at any surface d3
cut, so a literal \(K=4096\) cap cannot discard a boundary state. The
nonzero fractional bound merely shows that the fractional corollary does not
exploit this finite-support endpoint sharply.

## 6. Conclusion

Boundary aggregation addresses a real and sometimes enormous loss in the
product-Peierls argument. It approximately halves the grid-estimated
low-weight threshold—from 5235 to 2346 for X and from 4914 to 2318 for Z.
But it does not make the certificate useful at \(K=1024\), and the size-1/2
partial expression is still above one at \(K=2048\).

Therefore:

1. counting higher polymers remains counterproductive;
2. boundary aggregation alone is insufficient;
3. the next relaxation to attack is compatibility;
4. an alternative is direct computation of dominant exact
   comparison-spectrum overlaps \(\omega_t(\xi)\) on selected surface cuts.

The most useful next audit is the conflict structure of the live size-1/2
polymers at the cuts dominating the \(K=1024\) grouped expression. It will
determine whether exact compatibility can be enforced by component
factorization or bounded-treewidth elimination, or whether a direct
comparison-spectrum calculation is the better route.

## 7. Reproduction

```bash
frontier-overlap-profile \
  --backend rotated_surface_d3 \
  --p-location 0.001 \
  --column-order deadline_reorder \
  --score-alpha 0.8 \
  --K-values 16,512,1024,2048,4096,8192 \
  --aggregate-boundary-shifts \
  --max-boundary-bits 16 \
  --out \
    docs/data/overlap_profiles/rotated_surface_d3_p0p001_boundary_aggregated.json
```

The two-scope calculation took approximately 3.51 seconds on one CPU and
printed cut counts, elapsed time, and ETA every 25 cuts. The committed
artifact SHA256 is

```text
d4a5dc3a7d3aed11fe1e8e26e41cde7ed037e9e6ee7c0850ad0dabe9de08ccfb
```

The profiler's regression tests compare the XOR dynamic program with brute
subset enumeration on a finite toy model and verify
\(G\le U\le E\) cut by cut.
