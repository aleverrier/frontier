# Bounded-Hypergraph Overlap--Peierls Extension

Status: rigorous finite-model theorem, exact structural profile, and
real-matrix conclusion, 2026-07-28.

This note extends `GRAPHLIKE_OVERLAP_PEIERLS.md` from detector-column weight
two to arbitrary bounded detector-column weight. It also tests the resulting
product-Peierls certificate on the retained rotated-surface distance-3 and
BB144/Gross detector error models.

The mathematical extension succeeds. For score exponent
\(\alpha=0.8\), the exact ordered-matrix loads certify Chernoff orders
\(\theta=5/16\) for the surface model and \(\theta=5/24\) for BB144/Gross.
The numerical certificate does not succeed: exact enumeration of only the
visible size-1 and size-2 open-prefix polymers already makes the
product-Peierls right-hand side larger than one at the practical caps tested.
This is a rigorous diagnosis of that **bound**, not evidence that Frontier's
actual cap loss is large.

No decoder run or Monte Carlo simulation is used below. The committed JSON
files are deterministic functions of the retained matrices, order, physical
location rate, and score exponent.

## 1. General Chernoff overlap activity

Use the ordered prefix quotient, compatible visible open-prefix polymers,
coupling lift, and notation of `GRAPHLIKE_OVERLAP_PEIERLS.md`. For a
processed Bernoulli variable \(j\), probability \(p_j\), and
\(0\le\theta\le1\), define
\[
a_{\theta,j}
=
(1-p_j)^{1-\theta}p_j^\theta
+
p_j^{1-\theta}(1-p_j)^\theta.
\tag{1}
\]
At \(\theta=1/2\), this is the earlier Hellinger activity
\(\beta_j=2\sqrt{p_j(1-p_j)}\).

For a set \(S\) and tilt \(r>0\), recall
\[
\kappa_{t,r}(S)
=
\sum_{x\in\mathbb F_2^S}
\min\{P_{t,S}(x),rP_{t,S}(x+\mathbf1_S)\}.
\tag{2}
\]

**Lemma 1 (tilted Chernoff majorant).** For every \(S\), \(r>0\), and
\(0\le\theta\le1\),
\[
\boxed{
\kappa_{t,r}(S)
\le
r^\theta\prod_{j\in S}a_{\theta,j}.
}
\tag{3}
\]

**Proof.** Apply
\[
\min\{a,rb\}\le a^{1-\theta}(rb)^\theta
\]
term by term in (2). The remaining sum factorizes over the product marginal,
and its one-coordinate factor is exactly (1). \(\square\)

Define the score moment
\[
R_{t,\theta}(\delta)
=
\mathbb E_{Q_t}
\left[
\frac{g_t(Q_t+\delta)}{g_t(Q_t)}
\right]^\theta.
\tag{4}
\]
For the row-factorized score
\[
g_t(q)
=
\prod_{i\in\Gamma_t}
\rho_{i,t}(q_i)^{\alpha_{i,t}},
\qquad
\rho_{i,t}(b)=\Pr[(Q_t)_i=b],
\tag{5}
\]
Finner's hypergraph Hölder inequality gives the sufficient condition
\[
\boxed{
\theta
\max_{j\in J_t}
\sum_{\substack{
i\in\Gamma_t:\ \delta_i=1\\
H_{ij}=1}}
\alpha_{i,t}
\le1
\quad\Longrightarrow\quad
R_{t,\theta}(\delta)\le1.
}
\tag{6}
\]
Indeed, give active factor \(i\) exponent
\((\theta\alpha_{i,t})^{-1}\). Its powered expectation is one, and (6) is
the fractional-cover condition on every independent future variable.

## 2. Bounded-hypergraph pointwise theorem

Let
\[
b_t
=
\max_{j\in J_t}
|\operatorname{supp}H_j\cap\Gamma_t|
\tag{7}
\]
be the exact future-active detector load at cut \(t\). If all active score
exponents equal a scalar \(\alpha\), choose
\[
\theta_t
=
\min\left\{\frac12,\frac{1}{\alpha b_t}\right\},
\tag{8}
\]
with \(\theta_t=1/2\) when \(b_t=0\) or \(\alpha=0\). The cap at \(1/2\)
selects the most favorable symmetric Chernoff order available without
violating (6) when the Bernoulli probabilities are below \(1/2\).

For a visible polymer \(\gamma\), define
\[
w_{t,\theta_t}(\gamma)
=
\prod_{j\in\gamma}a_{\theta_t,j}.
\tag{9}
\]

**Theorem 2 (bounded-hypergraph pointwise domination).** Suppose the score
has form (5), and choose \(0<\theta_t\le1\) such that
\[
\theta_t
\max_{j\in J_t}
\sum_{\substack{i\in\Gamma_t\\H_{ij}=1}}
\alpha_{i,t}
\le1.
\tag{10}
\]
Then, for every nonzero relative boundary shift \(\xi\),
\[
\boxed{
\omega_t(\xi)
\le
\sum_{\substack{
\mathcal F\ {\rm compatible\ visible}\\
D_t\mathcal F=\xi}}
\prod_{\gamma\in\mathcal F}
w_{t,\theta_t}(\gamma).
}
\tag{11}
\]

**Proof.** The quotient subcoupling theorem in the companion note bounds
each family by one tilted product overlap on its union
\(S_{\mathcal F}\). Lemma 1 bounds that overlap by the score moment in (4)
times \(\prod_{j\in S_{\mathcal F}}a_{\theta_t,j}\). Condition (10) implies
that the score moment is at most one by (6). Compatible components are
disjoint, so the product over the union factorizes into (9). \(\square\)

The graphlike theorem is the special case \(b_t\le2\),
\(\alpha_{i,t}\le1\), and \(\theta_t=1/2\).

For \(0<\rho<1\), put
\[
\Xi_{t,\rho,\theta_t}
=
\sum_{\gamma\in\mathcal P_t^{\rm vis}}
w_{t,\theta_t}(\gamma)^\rho.
\tag{12}
\]
The same subadditivity and compatibility-dropping argument as in the
graphlike note gives
\[
\sum_{\xi\ne0}\omega_t(\xi)^\rho
\le e^{\Xi_{t,\rho,\theta_t}}-1,
\tag{13}
\]
and hence
\[
\boxed{
D^{\rm cap}
\le
K^{-1/\rho}
\sum_t
\left(e^{\Xi_{t,\rho,\theta_t}}-1\right)^{1/\rho}.
}
\tag{14}
\]
For a score gap \(\Delta\), Lemma 1 contributes
\(e^{-\theta_t\Delta}\), so
\[
D_t^{\rm gap}
\le
e^{-\theta_t\Delta}
\left(e^{\Xi_{t,1,\theta_t}}-1\right).
\tag{15}
\]
These are bounds on exact posterior mass removed by pruning. They are not
bounds on truth-present terminal ranking error.

## 3. Exact size-1 and size-2 lifetime profiler

The deterministic CLI `frontier-overlap-profile` computes (7) at every cut
and enumerates every visible open-prefix polymer of sizes one and two. Number
columns from \(0\) to \(N-1\), and let \(\ell_i\) be the last column touching
detector row \(i\). A cut \(t\) means immediately after column \(t\).

For a singleton column \(j\) with detector support \(B_j\), the exact
visibility interval is
\[
j\le t<\min_{i\in B_j}\ell_i.
\tag{16}
\]
A logical-only singleton remains visible through the final cut.

For a pair \(j<k\), write
\[
C=B_j\cap B_k,\qquad B=B_j\mathbin\triangle B_k.
\]
It can be connected only if \(C\ne\varnothing\). Its exact connected,
completed-parity-compatible interval is
\[
\min_{i\in C}\ell_i
\le t<
\min_{i\in B}\ell_i.
\tag{17}
\]
When \(B=\varnothing\), the upper endpoint is \(N\), provided the two
columns have different logical masks. Equations (16)--(17) follow directly:
a shared row must have completed to connect the pair, while every
odd-parity row must still be active. The implementation uses interval
differences, so every lifetime contributes to every cut exactly once.

Let \(\Xi^{\le2}_{t,\rho,\theta}\) denote (12) restricted to these
polymers. Since all terms are nonnegative,
\[
B^{\le2}_\rho(K)
=
K^{-1/\rho}
\sum_t
\left(e^{\Xi^{\le2}_{t,\rho,\theta}}-1\right)^{1/\rho}
\tag{18}
\]
is a **lower bound on the right-hand side of (14)**. It is not a lower bound
on cap loss. Thus \(B^{\le2}_\rho(K)>1\) proves only that this particular
product-Peierls certificate is vacuous before any higher polymers are added.

## 4. Retained-matrix structural audit

The profile uses `p_location=0.001`, `deadline_reorder`,
\(\alpha=0.8\), both memory sectors, and the exact per-column Bernoulli
probabilities produced by the retained DEM loader.

| Backend/scope | Detector matrix | Column weights | Max row degree | Max active width | Max \(b_t\) | Safe \(\theta\) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| surface d3 X | \(24\times221\) | \(1:24,2:84,3:76,4:37\) | 47 | 11 | 4 | \(5/16\) |
| surface d3 Z | \(24\times219\) | \(1:24,2:82,3:76,4:37\) | 48 | 11 | 4 | \(5/16\) |
| BB144 X | \(936\times8784\) | \(2:864,3:5328,4:864,5:864,6:864\) | 35 | 134 | 6 | \(5/24\) |
| BB144 Z | \(936\times8784\) | \(2:864,3:5328,4:864,5:864,6:864\) | 35 | 134 | 6 | \(5/24\) |

The retained DEMs are therefore not graphlike. The measured cutwise load
also reaches the full maximum column weight:

- surface X has \(b_t=4\) on 169 of 221 cuts and surface Z on 154 of 219;
- BB144 X has \(b_t=6\) on 8318 of 8784 cuts and BB144 Z on 8719 of 8784.

Consequently, replacing full column weight by exact future-active load does
not improve the globally safe Chernoff order for these orderings.

## 5. Exact low-weight results

| Backend/scope | Unique size 1 | Unique size 2 | Polymer-cut incidences | Peak live count | \(\sum_t\Xi^{\le2}_{t,1,\theta}\) | Peak \(\Xi^{\le2}_{t,1,\theta}\) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| surface d3 X | 197 | 1514 | 55,642 | 563 | 557.108 | 4.734 |
| surface d3 Z | 195 | 1405 | 57,722 | 477 | 561.531 | 4.377 |
| BB144 X | 7872 | 57,195 | 11,377,694 | 1885 | 677,410.997 | 111.738 |
| BB144 Z | 7872 | 47,757 | 8,759,425 | 1364 | 534,743.023 | 83.744 |

The most favorable sampled fractional exponent is \(\rho=0.99\). Even there,
the size-1/2 lower bound (18) is:

| Backend/scope | \(K=16\) | \(K=512\) | \(K=1024\) | \(K=8192\) |
| --- | ---: | ---: | ---: | ---: |
| surface d3 X | 346.924 | 10.468 | 5.198 | 0.636 |
| surface d3 Z | 325.405 | 9.819 | 4.875 | 0.597 |
| BB144 X | \(1.49\times10^{51}\) | \(4.49\times10^{49}\) | \(2.23\times10^{49}\) | \(2.73\times10^{48}\) |
| BB144 Z | \(4.35\times10^{38}\) | \(1.31\times10^{37}\) | \(6.52\times10^{36}\) | \(7.98\times10^{35}\) |

The \(\rho=0.5\) and \(0.75\) values are still larger. For example, BB144 X
at \(K=512\) has base-10 logarithms \(390.85\) and \(128.14\),
respectively. Higher polymers can only increase (18).

The outcome is therefore decisive:

1. the bounded-hypergraph extension closes the theorem's applicability gap
   for the retained matrices;
2. the raw product-Peierls relaxation is far too loose to certify practical
   BB144/Gross cap sizes;
3. counting larger polymers cannot repair this relaxation;
4. the next mathematical target is to preserve boundary-shift aggregation
   and quotient cancellation, or to measure the exact comparison spectrum,
   before returning to terminal logical-sector ranking.

This conclusion does not contradict successful decoder runs. Equation (14)
drops both compatibility and all cancellations between many polymers that
map to the same boundary shift.

The first target in item 4 is now tested in
`BOUNDARY_SHIFT_AGGREGATION.md`. Exact XOR grouping recovers up to 34 orders
of magnitude at fixed \(\rho\), but the optimized size-1/2 expression remains
above one at \(K=1024\) and \(K=2048\). Boundary aggregation alone is
therefore insufficient; compatibility or the exact comparison spectrum must
be retained next.

## 6. Reproduction and artifacts

From an editable installation:

```bash
frontier-overlap-profile \
  --backend rotated_surface_d3 \
  --p-location 0.001 \
  --column-order deadline_reorder \
  --score-alpha 0.8 \
  --out docs/data/overlap_profiles/rotated_surface_d3_p0p001.json

frontier-overlap-profile \
  --backend bravyi_depth7 \
  --p-location 0.001 \
  --column-order deadline_reorder \
  --score-alpha 0.8 \
  --out docs/data/overlap_profiles/bravyi_depth7_p0p001.json
```

The same commands can be run as
`python -m tools.frontier_overlap_profile`. The surface profile took
approximately \(0.14\) seconds and the BB144/Gross profile approximately
\(2.28\) seconds on one CPU in the recorded run. Pair enumeration prints
elapsed time and ETA every 250,000 candidate pairs by default.

Committed artifact checksums:

```text
0f6ce11b3ee55a68283ac40b0bf4caff7e312fa0a56ff3e5c5ca3e981d159678  rotated_surface_d3_p0p001.json
e0b51b2782f7978bc5026d190aea32fa9dd62312bb9ad701d525abc21357bc25  bravyi_depth7_p0p001.json
```

The JSON records the configuration, exact input-family checksum for each
scope, structural histograms, lifetime histograms, partial activities, and
log-stable values of (18). It contains no sampled outcomes or claimed FER.

`tests/test_frontier_overlap_profile.py` exhaustively checks the tilted
Chernoff majorant on a small product law, Finner non-amplification for a
correlated weight-four future hypergraph, incremental load accounting against
brute force, the exact singleton/pair intervals on a toy family, and the
surface CLI output.
