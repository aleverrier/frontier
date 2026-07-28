# Adaptive Kernel-Spectrum Certificate for Frontier Cap Pruning

Status: rigorous finite-model majorant and deterministic surface-d3
calculation, 2026-07-28.

This note completes the finite surface-code investigation begun in
`BOUNDARY_SHIFT_AGGREGATION.md`. The earlier size-1/2 product bound was
vacuous because it discarded boundary aggregation, polymer compatibility,
shift-specific score loads, and the dominant-head structure of the exact cap
theorem. Retaining all four gives a non-vacuous, all-size result:

\[
\boxed{
\sum_t D^{\rm cap}_t
\le 0.2519\quad\text{for surface-d3 memory X at }K=1024,
}
\]
\[
\boxed{
\sum_t D^{\rm cap}_t
\le 0.2767\quad\text{for surface-d3 memory Z at }K=1024.
}
\]

Here \(D^{\rm cap}_t\) is the syndrome-averaged exact posterior mass first
removed by the top-\(K\) cap at cut \(t\). The bound remains valid after
arbitrary earlier recursive pruning. It is not an FER estimate, and it does
not include score-gap pruning.

The calculation is deterministic. It performs no decoding and no Monte
Carlo.

## 1. Why the previous partial bound stopped at \(K\approx1500\)

At a fixed cut, a live size-one polymer has empty completed-check signature.
A live size-two polymer consists of two processed columns with the same
nonempty completed-check signature. Consequently:

1. all live singletons are mutually compatible;
2. at most one pair may be chosen from each signature class;
3. pair classes are compatible exactly when their completed-row signatures
   are disjoint.

The dominant surface cuts contain 392--563 live size-1/2 polymers, but only
20--33 pair-signature classes. Some individual classes contain 120--378
nominal pair choices, of which at most one is legal.

An exact row-mask set-packing dynamic program, combined with a
Walsh-Hadamard transform for boundary XOR, enforces this compatibility.
Compatibility plus boundary aggregation lowers the optimized size-1/2
fractional threshold to approximately

\[
K=1509\quad\text{for X},\qquad K=1455\quad\text{for Z}.
\]

At \(K=1024\), however, the partial right-hand sides are still 1.819 and
1.750. The next decisive loss is not compatibility. It is the use of one
globally worst Chernoff order for every boundary shift.

## 2. Shift-specific Finner order

For active-detector shift \(\delta\), define its exact future score load

\[
\lambda_t(\delta)
=
\max_{j\in J_t}
\sum_{\substack{
i\in\Gamma_t\\
\delta_i=1,\ H_{ij}=1}}
\alpha_i.
\tag{1}
\]

The Finner condition in `BOUNDED_HYPERGRAPH_OVERLAP.md` is pointwise in the
relative boundary shift. Therefore one may choose

\[
\boxed{
\theta_t^\star(\delta)
=
\min\left\{
\frac12,\frac{1}{\lambda_t(\delta)}
\right\},
}
\tag{2}
\]

with \(\theta_t^\star(0)=1/2\). For scalar score exponent
\(\alpha=0.8\) and the retained surface model, only three values occur:

\[
\theta^\star\in
\left\{
\frac5{16},\frac5{12},\frac12
\right\}.
\tag{3}
\]

The global bounded-hypergraph profile used \(5/16\) for every shift because
some future column touches four affected score rows. Equation (2) permits
\(1/2\) whenever the particular shift has load at most two. This reduces the
size-1/2 grouped fractional right-hand side at \(K=1024\) from

\[
1.819\ \longrightarrow\ 0.254\quad\text{for X},
\]
\[
1.750\ \longrightarrow\ 0.236\quad\text{for Z}.
\]

This is still a partial diagnostic because it omits connected components of
size three and above.

## 3. All-size completed-kernel majorant

For a fixed \(\theta\), define

\[
a_{\theta,j}
=
(1-p_j)^{1-\theta}p_j^\theta
+
p_j^{1-\theta}(1-p_j)^\theta.
\tag{4}
\]

At cut \(t\), define the weighted completed-kernel spectrum

\[
\boxed{
\mathcal Z_{t,\theta}(\xi)
=
\sum_{\substack{
v\in\mathbb F_2^{I_t}\\
H_{C_t,I_t}v=0\\
D_tv=\xi}}
\prod_{j:v_j=1}a_{\theta,j}.
}
\tag{5}
\]

This sum contains vectors of every Hamming weight.

### Lemma 1: visible-family injection

Every compatible family \(\mathcal F\) of visible open-prefix polymers maps
to the union vector

\[
v_{\mathcal F}
=
\bigoplus_{\gamma\in\mathcal F}\mathbf1_\gamma.
\tag{6}
\]

The map is injective. Given its union, the family is recovered as the
connected components in the completed-check interaction graph. Moreover,

\[
H_{C_t,I_t}v_{\mathcal F}=0,\qquad
D_tv_{\mathcal F}=D_t\mathcal F,
\tag{7}
\]

and disjointness of compatible components gives

\[
\prod_{\gamma\in\mathcal F}
\prod_{j\in\gamma}a_{\theta,j}
=
\prod_{j:(v_{\mathcal F})_j=1}a_{\theta,j}.
\tag{8}
\]

Therefore the compatible visible-family partition is bounded pointwise by
\(\mathcal Z_{t,\theta}\). Equation (5) can additionally count vectors with
invisible connected components, so it is a majorant rather than an equality.

Combining Lemma 1 with the shift-specific bounded-hypergraph theorem gives

\[
\boxed{
\omega_t(\xi)
\le
z_t(\xi)
:=
\mathcal Z_{t,\theta_t^\star(\delta_t\xi)}(\xi).
}
\tag{9}
\]

This is the key all-size pointwise bound.

## 4. Exact active-frontier recurrence

Equation (5) does not require polymer enumeration. Maintain a weighted state
on the active detector residual \(r\) and logical prefix \(\ell\). Before
closing the rows whose last touch is column \(j\), update

\[
P_j(r,\ell)
\leftarrow
P_{j-1}(r,\ell)
+
a_{\theta,j}
P_{j-1}(r+H_j,\ell+L_j).
\tag{10}
\]

Then:

1. discard states whose residual is nonzero on a row closing at \(j\);
2. remove the closed row coordinates;
3. record the remaining active-plus-logical spectrum.

This recurrence exactly enumerates every vector in (5). Its state count is
at most

\[
2^{|\Gamma_t|+k}.
\tag{11}
\]

For the retained surface-d3 models,
\(|\Gamma_t|\le11\) and \(k=1\), so each fixed-\(\theta\) recurrence has at
most 4096 states. Three recurrences provide the three values in (3), after
which each shift selects its permitted spectrum entry using (2).

The recurrence is an exact proof computation, not the practical pruned
decoder.

## 5. Use the exact trimmed head

Let

\[
z_{t,(1)}\ge z_{t,(2)}\ge\cdots
\]

be the decreasing rearrangement of the nonzero-shift majorant (9). The exact
recursive cap theorem and pointwise monotonicity give

\[
\boxed{
D_t^{\rm cap}
\le
\inf_{0\le q<K}
\frac{\sum_{j>q}z_{t,(j)}}{K-q}.
}
\tag{12}
\]

This is much sharper here than replacing the spectrum by one fractional
moment. It also captures the exact finite-support endpoint: a cut with fewer
than \(K\) nonzero relative shifts cannot remove a state through the cap.

For comparison, the optimized fractional corollary applied to the same
all-size \(z_t\) is still vacuous at \(K=1024\):

\[
1.927\quad\text{for X},\qquad
2.068\quad\text{for Z}.
\tag{13}
\]

The exact trimmed-head calculation gives 0.2519 and 0.2767. The useful
certificate therefore comes from the ordered spectrum itself, not merely
from its best sampled \(\ell^\rho\) quasi-norm.

## 6. Numerical result

The retained configuration is

```text
backend = rotated_surface_d3
p_location = 0.001
column_order = deadline_reorder
score_alpha = 0.8
global theta = 5/16
adaptive theta = {5/16, 5/12, 1/2}
```

The table compares successive calculations. “Fractional” means the best
value on the deterministic grid
\(\rho=0.05,0.06,\ldots,0.99\). “Trimmed” means (12).

| Majorant | Scope | \(K=512\) | \(K=1024\) | \(K=2048\) |
| --- | --- | ---: | ---: | ---: |
| size-1/2, compatibility + shift, fractional | X | 4.110 | 1.819 | 0.543 |
| size-1/2, compatibility + shift, fractional | Z | 4.118 | 1.750 | 0.506 |
| size-1/2, adaptive \(\theta\), fractional | X | 0.711 | 0.254 | 0.0558 |
| size-1/2, adaptive \(\theta\), fractional | Z | 0.693 | 0.236 | 0.0500 |
| all-size kernel, adaptive \(\theta\), fractional | X | 5.852 | 1.927 | 0.361 |
| all-size kernel, adaptive \(\theta\), fractional | Z | 6.330 | 2.068 | 0.404 |
| all-size kernel, adaptive \(\theta\), trimmed | X | 1.787 | **0.2519** | 0.01524 |
| all-size kernel, adaptive \(\theta\), trimmed | Z | 1.840 | **0.2767** | 0.01586 |

At \(K=4096\), (12) is exactly zero because there are at most 4096 boundary
states, hence at most 4095 nonzero relative shifts.

For \(K=1024\), only 78 X cuts and 77 Z cuts have formal boundary support
large enough for the cap to act. The leading X contributions occur around
cuts 153--156; the leading Z contributions occur around cuts 140--155. The
largest individual cut contribution is approximately 0.0207 for X and
0.0179 for Z.

The exact adaptive size-1/2 trimmed totals are 0.0505 for X and 0.0400 for
Z. The difference to the all-kernel values measures the combined effect of
higher-weight vectors and the intentional overcount of invisible connected
components. It is not an estimate of actual cap loss.

## 7. What is now understood

The surface result requires four distinct mechanisms:

1. **Compatibility.** Hundreds of nominal low-weight polymers reduce to a
   constrained set-packing problem on a small completed-row universe.
2. **Boundary aggregation.** Families with the same XOR shift must be
   combined before the fractional power or ordered-spectrum calculation.
3. **Shift-specific score load.** Most shifts do not incur the globally
   worst Finner load and therefore have much smaller Chernoff activities.
4. **Head trimming and finite support.** The cap responds to the tail after
   up to \(K-1\) dominant shifts are removed, not to the raw total activity.

Worst-case width remains relevant to the cost of an exact proof computation,
but it does not determine cap-induced error. The relevant object is the
ordered, score-calibrated comparison spectrum.

The result is finite-model and family-specific. It does not yet give a
scalable BB144/Gross certificate because that model has up to 134 active
detectors per split sector, far beyond dense exact spectrum storage.

## 8. Cap loss is not terminal Bayes error

The public Gross \(p=0.001,\Delta=20,K=16384\) summary reports three logical
failures in \(10^8\) trials, with zero truth-missing cases and three
truth-present-but-not-selected cases. In the replay code, “truth present”
means that the sampled logical sector has nonzero retained terminal mass.
“Not selected” means another retained sector has greater mass.

This classification proves that those sampled failures were not empty
logical-support failures. It does **not** prove that pruning caused the wrong
ranking. Exact logical ML also fails on samples whose true sector is not the
posterior maximum.

To separate algorithmic distortion from irreducible Bayes error, one needs
either:

1. an exact logical-ML comparator on the same samples; or
2. exact sector survival fractions
   \(a_\ell(s)=\widehat Z_\ell(s)/Z_\ell(s)\).

The decisive condition remains

\[
\max_{\ell\ne\ell_*}
\log\frac{a_\ell(s)}{a_{\ell_*}(s)}
<
\min_{\ell\ne\ell_*}
\log\frac{Z_{\ell_*}(s)}{Z_\ell(s)}.
\tag{14}
\]

Thus the full problem splits cleanly:

| Question | Controlled object | Present status |
| --- | --- | --- |
| Can the cap remove exact posterior mass? | trimmed \(z_t(\xi)\) spectrum | non-vacuous all-size surface certificate |
| Can the score gap remove exact posterior mass? | exact gap-overlap spectrum | theorem proved; no comparable surface profile yet |
| Can pruning change the logical-ML sector? | sectorwise survival bias | exact identity, no quantitative bound |
| Can the sampled truth lose under exact ML? | exact logical posterior gap | intrinsic Bayes event, not a pruning error |

## 9. Reproduction

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

The two-scope calculation uses one CPU, prints cut progress and ETA, and
is reproducible byte-for-byte. The recorded runs took 27.53 and 27.63
seconds on the development machine. The committed artifact
contains every per-cut structural summary, the size-1/2 comparisons, the
three fixed-\(\theta\) kernel state counts, the adaptive fractional moments,
and the exact trimmed bounds.

Its SHA256 is

```text
44134ed72038d29292efd092ba9e35091d1ff7508a704caf64f1d43f4e819f1e
```

The implementation is checked against:

1. brute-force XOR subset enumeration;
2. a finite signature-class compatibility example;
3. brute-force kernel enumeration on the toy cut;
4. the inequalities
   \[
   G_{\rm adaptive}^{\le2}
   \le G_{\rm fixed}^{\le2},
   \qquad
   G_{\rm adaptive}^{\le2}
   \le Z_{\rm adaptive}^{\rm kernel};
   \]
5. the finite-support and dominant-head endpoints of (12).
