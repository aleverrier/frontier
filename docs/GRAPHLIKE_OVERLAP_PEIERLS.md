# Graphlike Overlap--Peierls Reduction for Frontier

Status: rigorous finite-model theorem and proof draft, 2026-07-28.

This note completes the graphlike family step left open in
`COMPARISON_SPECTRUM_THEOREM.md`.  It gives:

1. an exact coupling lift from quotient boundary overlaps back to prefix
   assignments;
2. an exact family bound using one overlap on the **union** of the visible
   connected components;
3. a counterexample showing that exact componentwise overlap weights do not
   multiply;
4. a graphlike Hellinger majorant that does multiply and therefore proves the
   required pointwise polymer domination;
5. explicit fractional cap, score-gap, connected-set, and integrated-lifetime
   corollaries.

The correction in item 3 matters.  The low-weight exact overlap
\(\Omega_m(p)\) remains useful for finite-size family counting, but it cannot
be assigned independently to connected components.  The rigorous
componentwise activity is instead the Hellinger weight
\[
\beta_j=2\sqrt{p_j(1-p_j)}.
\]
Consequently, the exact family theorem retains the favorable
\(\Omega_m(p)=O(p^{\lceil m/2\rceil})\) behavior for a fixed union of size
\(m\), while the unconditional product theorem has the weaker but fully
factorized \(\prod_j\beta_j\) activity.

As in the companion note, all statements concern posterior mass removed by
gap/cap pruning.  They do not turn raw frontier width into an error bound and
do not by themselves explain truth-present terminal ranking failures.

## 1. Ordered prefix quotient

Fix a cut \(t\).  Let \(I_t\) be the processed variables and write
\[
P_t(u)=\Pr[X_{I_t}=u],
\qquad u\in\mathbb F_2^{I_t},
\]
for their product law.  Define
\[
F_t(u)=\bigl(c(u),b(u)\bigr)
=
\left(
H_{C_t,I_t}u,\,
H_{\Gamma_t,I_t}u,\,
L_{I_t}u
\right).
\tag{1}
\]
The pushforward mass is
\[
A_t(y)=\sum_{u:F_t(u)=y}P_t(u),
\qquad y=(c,b).
\tag{2}
\]

For a nonzero relative boundary shift
\[
\xi\in G_t=D_t(\ker H_{C_t,I_t}),
\qquad
D_tv=(H_{\Gamma_t,I_t}v,L_{I_t}v),
\tag{3}
\]
let \(\delta_t\xi\) denote its active-detector component.  At fixed future
active parity \(q\), put
\[
r_t(q,\xi)
=
\frac{g_t(q+\delta_t\xi)}{g_t(q)}>0
\tag{4}
\]
and define the fixed-\(q\) quotient overlap
\[
O_{t,q}(\xi)
=
\sum_y
\min\left\{
A_t(y),\,
r_t(q,\xi)A_t(y+\xi)
\right\}.
\tag{5}
\]
Here \(y+\xi=(c,b+\xi)\): the completed syndrome is unchanged.
The comparison spectrum from the companion theorem is exactly
\[
\omega_t(\xi)=\mathbb E_{Q_t}O_{t,Q_t}(\xi).
\tag{6}
\]

## 2. Completed-check interaction graph

Let \(\mathcal G_t\) be the graph whose vertices are the processed variables
\(I_t\).  Two distinct variables are adjacent when they both occur in some
completed detector row in \(C_t\).

For a subset \(\gamma\subseteq I_t\), identify \(\gamma\) with its indicator
vector \(\mathbf1_\gamma\).  Call \(\gamma\) an **open-prefix polymer** when

- \(\gamma\ne\varnothing\);
- \(\gamma\) is connected in \(\mathcal G_t\);
- \(H_{C_t,I_t}\mathbf1_\gamma=0\).

It is **visible** when
\[
D_t\mathbf1_\gamma\ne0.
\tag{7}
\]
Two polymers are compatible when no graph edge joins them.  For a compatible
family \(\mathcal F\), define
\[
S_{\mathcal F}=\bigcup_{\gamma\in\mathcal F}\gamma,
\qquad
D_t\mathcal F
=
\bigoplus_{\gamma\in\mathcal F}D_t\mathbf1_\gamma.
\tag{8}
\]

**Lemma 1 (componentwise completed parity).**  If
\[
v\in\ker H_{C_t,I_t},
\tag{9}
\]
then every connected component \(\gamma\) of
\(\operatorname{supp}v\) in \(\mathcal G_t\) is an open-prefix polymer.

**Proof.**  A completed row cannot meet two different connected components:
any two support variables in that row would be adjacent.  The total parity of
the row on \(v\) is zero, so its parity on the unique component it meets is
also zero.  This holds for every completed row.  \(\square\)

The visible components of such a \(v\) form a nonempty compatible family
whenever \(D_tv\ne0\).  Invisible components may be present, but their
boundary shifts vanish.

## 3. Coupling lift and exact family theorem

For \(S\subseteq I_t\) and \(r>0\), define the tilted local product overlap
\[
\kappa_{t,r}(S)
=
\sum_{x\in\mathbb F_2^S}
\min\left\{
P_{t,S}(x),\,
rP_{t,S}(x+\mathbf1_S)
\right\},
\tag{10}
\]
where \(P_{t,S}\) is the product marginal on \(S\).

For a visible compatible family, put
\[
\overline\kappa_t(\mathcal F)
=
\mathbb E_{Q_t}
\kappa_{t,r_t(Q_t,D_t\mathcal F)}(S_{\mathcal F}).
\tag{11}
\]
Only the active component of \(D_t\mathcal F\) enters the ratio in (11).

**Theorem 2 (quotient coupling lift).**  For every nonzero
\(\xi\in G_t\),
\[
\boxed{
\omega_t(\xi)
\le
\sum_{\substack{
\mathcal F\ {\rm compatible\ visible}\\
D_t\mathcal F=\xi}}
\overline\kappa_t(\mathcal F).
}
\tag{12}
\]

**Proof.**  Fix \(q\), abbreviate \(r=r_t(q,\xi)\), and let
\[
\nu_y=\min\{A_t(y),rA_t(y+\xi)\}.
\tag{13}
\]
For nonzero fibers define a flow between prefix assignments by
\[
\lambda_y(u,u')
=
\nu_y
\frac{P_t(u)}{A_t(y)}
\frac{P_t(u')}{A_t(y+\xi)},
\qquad
F_t(u)=y,\quad F_t(u')=y+\xi.
\tag{14}
\]
Zero fibers contribute no flow.  The total flow is \(O_{t,q}(\xi)\).  Its
first marginal is at most \(P_t\), because \(\nu_y\le A_t(y)\), and its
second marginal is at most \(rP_t\), because
\(\nu_y\le rA_t(y+\xi)\).

Every pair carrying flow has difference
\[
v=u+u'\in\ker H_{C_t,I_t},
\qquad D_tv=\xi.
\tag{15}
\]
By Lemma 1, the visible connected components of
\(\operatorname{supp}v\) form a compatible family \(\mathcal F\) with
\(D_t\mathcal F=\xi\).  Partition the flow according to this family,
ignoring any invisible components of \(v\).

Fix one family and write \(S=S_{\mathcal F}\).  On every pair in its flow
class,
\[
u'_S=u_S+\mathbf1_S.
\tag{16}
\]
Projecting the first marginal to \(S\) bounds the flow starting at \(x\) by
\(P_{t,S}(x)\).  Projecting the second marginal bounds the same flow by
\(rP_{t,S}(x+\mathbf1_S)\).  Hence the total flow in this class is at most
\(\kappa_{t,r}(S)\).  Sum over the families and then average over \(Q_t\).
\(\square\)

The proof permits arbitrary quotient merging and arbitrary invisible kernel
components.  No injective choice of a representative of \(\xi\) is required.

## 4. Why exact component activities do not multiply

Let two processed Bernoulli-\(p\) variables have no completed constraints,
let the boundary map be the identity, let \(g_t\equiv1\), and consider the
shift that flips both variables.  The interaction graph has two compatible
visible singleton polymers.

For \(0<p<1/2\), the exact product-law overlap is
\[
\Omega_2(p)=2p.
\tag{17}
\]
The product of the two singleton overlaps would instead be
\[
\Omega_1(p)^2=(2p)^2=4p^2<2p.
\tag{18}
\]
Thus the proposed bound
\[
\omega_t(\xi)
\stackrel{\rm false}{\le}
\sum_{\mathcal F:D_t\mathcal F=\xi}
\prod_{\gamma\in\mathcal F}\Omega_{|\gamma|}(p)
\tag{19}
\]
already fails in an unquotiented two-bit model.

The coupling lift identifies the correct exact object: one overlap on the
whole family support.  In the homogeneous, unscored case,
\[
\boxed{
\overline\kappa_t(\mathcal F)
=
\Omega_{|S_{\mathcal F}|}(p),
}
\tag{20}
\]
where, for \(J\sim{\rm Bin}(|S_{\mathcal F}|,p)\),
\[
\Omega_m(p)
=
2\Pr[J>m/2]+\Pr[J=m/2].
\tag{21}
\]

This failure of componentwise multiplication is a correlation phenomenon in
the overlap coupling, not a failure of connected-defect decomposition.

## 5. Exact finite-size fractional family bound

Assume \(p_j=p\) on the processed variables and \(g_t\equiv1\).  Let
\(M_t(m)\) count compatible visible families \(\mathcal F\) such that
\[
D_t\mathcal F\ne0,
\qquad |S_{\mathcal F}|=m.
\tag{22}
\]
For \(0<\rho<1\), Theorem 2 and
\((\sum_i a_i)^\rho\le\sum_i a_i^\rho\) give
\[
\boxed{
\sum_{\xi\ne0}\omega_t(\xi)^\rho
\le
\sum_{m\ge1}M_t(m)\Omega_m(p)^\rho.
}
\tag{23}
\]
Combining this with the sharp cap theorem from the companion note yields
\[
\boxed{
D_t^{\rm cap}
\le
K^{-1/\rho}
\left[
\sum_{m\ge1}M_t(m)\Omega_m(p)^\rho
\right]^{1/\rho}.
}
\tag{24}
\]

Equation (24) is the strongest result here when the compatible-family counts
can be computed or bounded directly.  It keeps the exact low-weight
\(p\)-scale overlap.  It is not a single-polymer cluster expansion because
\(\Omega_{m_1+\cdots+m_k}\) does not factor over the components.

## 6. Hellinger factorization

Let processed variable \(j\) have Bernoulli probability \(p_j\) and define
\[
\beta_j=2\sqrt{p_j(1-p_j)}.
\tag{25}
\]

**Lemma 3 (tilted Hellinger majorant).**  For every \(S\subseteq I_t\) and
\(r>0\),
\[
\boxed{
\kappa_{t,r}(S)
\le
\sqrt r\prod_{j\in S}\beta_j.
}
\tag{26}
\]

**Proof.**  Use \(\min\{a,rb\}\le\sqrt r\sqrt{ab}\) in (10).  The remaining
Hellinger coefficient factorizes:
\[
\sum_x\sqrt{P_{t,S}(x)P_{t,S}(x+\mathbf1_S)}
=
\prod_{j\in S}2\sqrt{p_j(1-p_j)}.
\]
\(\square\)

Write
\[
R_{t,1/2}(\delta)
=
\mathbb E_{Q_t}
\left[
\frac{g_t(Q_t+\delta)}{g_t(Q_t)}
\right]^{1/2}.
\tag{27}
\]
Equations (11) and (26) imply
\[
\overline\kappa_t(\mathcal F)
\le
R_{t,1/2}(\delta_tD_t\mathcal F)
\prod_{j\in S_{\mathcal F}}\beta_j.
\tag{28}
\]

For the row-factorized score
\[
g_t(q)
=
\prod_{i\in\Gamma_t}
\rho_{i,t}(q_i)^{\alpha_{i,t}},
\qquad
\rho_{i,t}(b)=\Pr[(Q_t)_i=b],
\tag{29}
\]
Finner's hypergraph Hölder inequality gives the sufficient condition
\[
\frac12
\max_{j\in J_t}
\sum_{\substack{
i:(\delta)_i=1\\
H_{ij}=1}}
\alpha_{i,t}
\le1
\quad\Longrightarrow\quad
R_{t,1/2}(\delta)\le1.
\tag{30}
\]

## 7. Complete graphlike pointwise theorem

Call the detector matrix graphlike when every detector column has weight at
most two.  Logical incidences do not count toward this detector-column
weight.

**Theorem 4 (graphlike pointwise polymer domination).**  Suppose:

1. every detector column has weight at most two;
2. the score has the row-factorized form (29);
3. \(\alpha_{i,t}\le1\) for every active row.

For each visible polymer define
\[
w_t(\gamma)=\prod_{j\in\gamma}\beta_j.
\tag{31}
\]
Then, for every nonzero \(\xi\in G_t\),
\[
\boxed{
\omega_t(\xi)
\le
\sum_{\substack{
\mathcal F\ {\rm compatible\ visible}\\
D_t\mathcal F=\xi}}
\prod_{\gamma\in\mathcal F}w_t(\gamma).
}
\tag{32}
\]

**Proof.**  A future detector column meets at most two active rows.  With
\(\alpha_{i,t}\le1\), the left side of (30) is at most
\(\tfrac12(2)=1\), so \(R_{t,1/2}(\delta)\le1\) for every shift
\(\delta\).  Apply (28), then use disjointness of compatible components:
\[
\prod_{j\in S_{\mathcal F}}\beta_j
=
\prod_{\gamma\in\mathcal F}w_t(\gamma).
\]
Insert this in Theorem 2.  \(\square\)

This proves the pointwise assumption that was left conditional in equation
(27) of `COMPARISON_SPECTRUM_THEOREM.md`.  It does so with Hellinger polymer
activities, not with the false exact-overlap product in (19).

## 8. Fractional Peierls, gap, and excess-risk bounds

For \(0<\rho<1\), define
\[
\Xi_{t,\rho}
=
\sum_{\gamma\in\mathcal P_t^{\rm vis}}
w_t(\gamma)^\rho.
\tag{33}
\]
Subadditivity, followed by dropping compatibility constraints, gives
\[
\boxed{
\sum_{\xi\ne0}\omega_t(\xi)^\rho
\le
\prod_{\gamma\in\mathcal P_t^{\rm vis}}
\left(1+w_t(\gamma)^\rho\right)-1
\le
e^{\Xi_{t,\rho}}-1.
}
\tag{34}
\]
Therefore
\[
\boxed{
D^{\rm cap}
\le
K^{-1/\rho}
\sum_t
\left(e^{\Xi_{t,\rho}}-1\right)^{1/\rho}.
}
\tag{35}
\]

The same lift applies to the exact gap activity after replacing \(r\) by
\(e^{-\Delta}r\).  Lemma 3 contributes \(e^{-\Delta/2}\), so
\[
\boxed{
D_t^{\rm gap}
\le
e^{-\Delta/2}
\left(e^{\Xi_{t,1}}-1\right).
}
\tag{36}
\]
Combining (35), (36), and the exact excess-Bayes-risk argument gives
\[
\boxed{
R_{\rm Frontier}-R_{\rm logical\ ML}
\le
\sum_t
\left[
e^{-\Delta/2}\left(e^{\Xi_{t,1}}-1\right)
+
K^{-1/\rho}
\left(e^{\Xi_{t,\rho}}-1\right)^{1/\rho}
\right].
}
\tag{37}
\]

These are finite-model inequalities.  Their usefulness on a code family
depends on controlling both polymer counts and the number of cuts for which a
polymer remains visible.

## 9. A self-contained connected-set count

Let \(d_t\) be the maximum number of processed variables incident to a
completed detector, with \(d_t=1\) by convention when there are no completed
incidences.  If detector columns have weight at most two, then the maximum
degree of \(\mathcal G_t\) is at most
\[
\Delta_t=2(d_t-1).
\tag{38}
\]
Indeed, a processed variable has at most two completed-detector endpoints,
and each endpoint contributes at most \(d_t-1\) neighboring variables.

In any graph of maximum degree \(\Delta_t\), the number of connected
\(m\)-vertex sets containing a fixed root is at most
\[
\Delta_t^{2(m-1)}.
\tag{39}
\]
To see this, choose a canonical rooted spanning tree of the induced connected
set and encode it by its depth-first traversal, a walk of length
\(2(m-1)\).  The visited vertex set recovers the connected set, and each walk
step has at most \(\Delta_t\) choices.

Let \(n_t=|I_t|\) and
\[
\overline\beta_t=\max_{j\in I_t}\beta_j.
\tag{40}
\]
Dropping the completed-parity and visibility restrictions, (39) gives
\[
\Xi_{t,\rho}
\le
n_t\sum_{m\ge1}
\Delta_t^{2(m-1)}
\overline\beta_t^{\rho m}.
\tag{41}
\]
Hence, whenever
\[
\Delta_t^2\overline\beta_t^\rho<1,
\tag{42}
\]
\[
\boxed{
\Xi_{t,\rho}
\le
\frac{n_t\overline\beta_t^\rho}
{1-\Delta_t^2\overline\beta_t^\rho}.
}
\tag{43}
\]
For \(\Delta_t=0\), interpret the \(m=1\) factor in (41) as one; (43)
still holds.

This count is intentionally coarse.  It counts all connected sets rather than
parity-compatible visible polymers and carries an extensive \(n_t\) factor.
Parity, ordering, and finite visibility can only improve it.

## 10. Integrated ordering lifetime

The pointwise theorem alone does not remove the sum over cuts.  Define
\[
\mathcal L_\rho=\sum_t\Xi_{t,\rho},
\qquad
x_\rho=\sup_t\Xi_{t,\rho}.
\tag{44}
\]
If \(0<x_\rho<\infty\), then
\[
\left(e^x-1\right)^{1/\rho}
\le
e^{x_\rho/\rho}
x_\rho^{1/\rho-1}x,
\qquad 0\le x\le x_\rho.
\tag{45}
\]
Thus (35) implies the explicit lifetime form
\[
\boxed{
D^{\rm cap}
\le
K^{-1/\rho}
e^{x_\rho/\rho}
x_\rho^{1/\rho-1}
\mathcal L_\rho.
}
\tag{46}
\]

Equation (46) isolates what an ordering theorem or measurement must control:
the time-integrated activity \(\mathcal L_\rho\), not the worst-case raw
boundary width.  A sharper family result should count only parity-compatible
open polymers and charge each by its actual visible lifetime.

## 11. What is now proved and what remains

The graphlike overlap--Peierls step is complete at the following level.

- Quotient merging and invisible kernel components are handled by an explicit
  subcoupling lift.
- Exact overlap is retained on the union of each compatible visible family.
- Exact componentwise overlap multiplication is disproved.
- Hellinger weights restore componentwise multiplication.
- Detector-column weight two and score exponents at most one make the
  Hellinger score moment non-amplifying by Finner.
- The resulting pointwise theorem implies explicit fractional cap, score-gap,
  connected-count, and integrated-lifetime bounds.

The remaining improvements are family- and ordering-specific:

1. replace the coarse connected-set count by parity-compatible open-defect
   counts;
2. measure or prove short visible lifetimes under the chosen ordering;
3. use the exact family bound (23) for finite-size low-weight improvements;
4. analyze terminal logical ranking separately through charged/projective
   sector distortion.

In particular, the overlap improvement does not automatically improve the
asymptotic single-polymer surface tension after unconditional productization:
\(\beta(p)\sim2\sqrt p\).  Its \(p\)-scale advantage survives in the exact
nonmultiplicative family activity \(\Omega_m(p)\), which is valuable when
\(M_t(m)\) can be controlled directly.

## 12. Deterministic verification and references

`tests/test_graphlike_overlap_peierls.py` checks:

- the two-singleton counterexample;
- the exact union-overlap and Hellinger inequalities;
- quotient merging with two alternative visible representatives and an
  invisible kernel coordinate;
- Finner non-amplification in a correlated three-row, graphlike future model;
- the resulting fractional family bound.

The test is an exhaustive finite algebra check, not a numerical simulation.

The hypergraph Hölder step is the probability inequality introduced in:

- Helmut Finner, "A Generalization of Hölder's Inequality and Some Probability
  Inequalities," *The Annals of Probability* 20(4), 1893--1901 (1992),
  <https://doi.org/10.1214/aop/1176989534>.

For related connected-subset/percolation machinery in quantum LDPC decoding,
see:

- Omar Fawzi, Antoine Grospellier, and Anthony Leverrier, "Efficient decoding
  of random errors for quantum expander codes," arXiv:1711.08351 (2017),
  <https://arxiv.org/abs/1711.08351>.
