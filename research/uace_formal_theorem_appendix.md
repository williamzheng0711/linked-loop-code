# Formal Appendix Draft: Finite-Length UACE Bounds for the Current LLC Decoder

Status: paper-facing theorem/proof draft for the current no-SIC UACE
implementation at the repo/TCom finite instance.

This appendix is intentionally narrower than a full LLC achievability theorem.
It states what is currently theorem-level for the implemented decoder and what
is a validated finite-instance prediction.

The consolidated validation artifact is
`research/uace_publish_validation_certificate.md`, with dashboard figure
`research/figures/uace_publish_validation_dashboard.png`.

## A. Model

There are \(K\) active users.  User \(i\) transmits an LLC codeword

\[
X_i=(X_{i,0},\ldots,X_{i,L-1}),\qquad X_{i,\ell}\in[2^J].
\]

Each section is independently erased with probability \(p_e\).  In section
\(\ell\), the UACE output is the set of distinct non-erased symbols,

\[
Y_\ell=\{X_{i,\ell}: E_{i,\ell}=0,\ i=1,\ldots,K\}.
\]

For a tagged user, write the erasure mask as

\[
E=(E_0,\ldots,E_{L-1})\in\{0,1\}^L,
\qquad
\Pr[E=e]=p_e^{|e|}(1-p_e)^{L-|e|}.
\]

The finite instance considered here is

\[
B=128,\qquad J=16,\qquad L=16,\qquad M=3,\qquad m_\ell=p_\ell=8.
\]

The decoder is the current no-SIC repository decoder, including its fixed root
schedule, rotated scan order, erasure-slot guards, first-visible-saver solve,
row order, and final parity check.

## B. Exact Schedule Term

Define the collision-free schedule rejection indicator

\[
S_\phi(e)
=
\mathbf 1\{
\text{the current phase-}\phi\text{ root/recovery schedule rejects mask }e
\}.
\]

Then the exact erasure-only drop probability of phase \(\phi\) is

\[
P_{\mathrm{sch},\phi}
=
\sum_{e\in\{0,1\}^L}
S_\phi(e)p_e^{|e|}(1-p_e)^{L-|e|}.
\]

For the current \(L=16,M=3\) decoder,

\[
P_{\mathrm{sch,I}}
=
\Pr[|E|\ge1],
\]

\[
P_{\mathrm{sch,II}}
=
\Pr[|E|\ge 2],
\]

and

\[
P_{\mathrm{sch,III}}
=
\Pr[|E|\ge3]+33p_e^2(1-p_e)^{14}.
\]

Proof sketch.  Phase I accepts only the zero-erasure mask.  Phase II carries at
most one erased section, and the two phase-II roots cover all one-erasure
masks, so the rejected set is exactly \(\{|E|\ge2\}\).  Phase III carries at
most two erased sections.  Exhaustive finite-state enumeration of the \(2^{16}\)
masks under the implemented root schedule shows that all masks of weight at
least three are rejected, and exactly 33 of the \(\binom{16}{2}=120\)
two-erasure masks are rejected.  The schedule automaton was validated against
the repository decoder in the collision-free \(K=1\) setting with zero
actual/schedule mismatches for all masks of weights 0, 1, 2, and 3.

This term should be read as a \(K\)-independent erasure floor, not as the full
multi-user PDP.  Let \(\mathcal S_\phi=\{S_\phi(E)=1\}\), and let
\(\mathcal I_{\phi,K}\) be the event that the implemented schedule would accept
the tagged mask but the other \(K-1\) users create a preempting path, a returned
switch, or a hallucinated message.  Then

\[
P_{\mathrm{PDP},\phi}(K)
=
\Pr[\mathcal S_\phi]
+
\Pr[\mathcal S_\phi^c\cap\mathcal I_{\phi,K}]
\le
P_{\mathrm{sch},\phi}+P_{\mathrm{path},\phi}(K).
\]

The first term is the same erasure-only baseline one would see in a
collision-free \(K=1\) experiment.  The second term is where finite-\(K\)
A-channel occupancy, path switching, parity collisions, and hallucinations
enter the bound.

## C. Exact Occupancy Scale

Let \(Q=2^J\).  The exact expected A-list size in one section is

\[
\lambda_A
=
Q\left[
1-\left(1-\frac{1-p_e}{Q}\right)^K
\right].
\]

For a tagged non-erased section, the exact probability that at least one other
user occupies the tagged user's symbol is

\[
\rho_A
=
1-\left(1-\frac{1-p_e}{Q}\right)^{K-1}.
\]

The one-section occupancy PMF can be computed recursively.  If \(P_i(s)\) is
the probability that \(s\) symbols are occupied after \(i\) users, then

\[
P_{i+1}(s)
\mathrel{+}=
P_i(s)
\left[
p_e+(1-p_e)\frac{s}{Q}
\right],
\]

\[
P_{i+1}(s+1)
\mathrel{+}=
P_i(s)
(1-p_e)\frac{Q-s}{Q}.
\]

Under the independent-section random-symbol abstraction,

\[
\mathbb E\prod_{\ell\in T}|Y_\ell|
=
\lambda_A^{|T|}.
\]

## D. Rank-Corrected False-Survivor Moment

For an accepted erasure mask in decoder attempt \(a\), the parity constraints
for a false candidate can be written as

\[
A x_{\mathrm{erased}}+Bz_{\mathrm{known}}=0.
\]

For uniform known symbols, the probability that some erased assignment satisfies
these equations is

\[
2^{-e},
\qquad
e=\operatorname{rank}([A\ B])-\operatorname{rank}(A).
\]

Let \(A_a(w,e)\) denote the exact number of accepted erasure masks of weight
\(w\) and exponent \(e\) for attempt \(a\).  Then the false full-message first
moment is bounded by

\[
\mathbb E N_{\mathrm{false},a}
\le
\sum_w\sum_e
A_a(w,e)\lambda_A^{L-w}2^{-e}.
\]

Consequently,

\[
P_{\mathrm{hallucination}}
\le
\frac{1}{K}\sum_{a\in\mathcal A_\phi}
\sum_w\sum_e
A_a(w,e)\lambda_A^{L-w}2^{-e}
\]

by Markov's inequality.  This is theorem-shaped under the independent-section
random-symbol abstraction, but it is not the final dependency-aware
path-shape enumeration.

For the repo profile, the accepted-mask exponent profile is

| erased sections \(w\) | exponent \(e\) |
|---:|---:|
| 0 | 128 |
| 1 | 112 |
| 2 | 96 |

In particular, Phase I only accepts \(w=0,e=128\), hence

\[
P_{\mathrm{false,I}}(K)
\le
\frac{1}{K}\lambda_A^{16}2^{-128}.
\]

## E. Ordered Pair-Preemption Bound

Consider phase-III accepted two-erasure masks.  For an attempt \(a\), tagged
mask \(m\), and two-color identity profile \(\pi\), color 0 denotes the tagged
user and color 1 denotes one alternate user.  The ordered decoder equations are
homogeneous GF(2) equations

\[
H_{a,m,\pi}X=0.
\]

A false two-user path preempts the true tagged path in row order if, at the
first selected section where the alternate symbol differs from the tagged
symbol, the alternate symbol is smaller.  This preemption event is a disjoint
union over first-difference section/bit events \(d\) of affine systems

\[
H_{a,m,\pi}X=0,\qquad
R_{a,m,\pi,d}X=r_{a,m,\pi,d}.
\]

Therefore each disjunct has exact probability

\[
2^{-\operatorname{rank}([H_{a,m,\pi};R_{a,m,\pi,d}])}.
\]

Let \(s_{a,m,\pi,d}\) be the number of alternate-selected section symbols that
must be non-erased up to the first difference.  Define

\[
q_{a,m}(p_e)
\le
\sum_{\pi}\sum_d
(1-p_e)^{s_{a,m,\pi,d}}
2^{-\operatorname{rank}([H_{a,m,\pi};R_{a,m,\pi,d}])}.
\]

The conservative \(K\)-user pair-preemption lift is

\[
P_{\mathrm{preempt}}
\le
\sum_{a,m}
p_e^{|m|}(1-p_e)^{L-|m|}
c_{a,m}
\min\{1,(K-1)q_{a,m}(p_e)\}.
\]

For the current finite instance, exact two-color enumeration over phase-III
accepted two-erasure gap representatives gives

| quantity | value |
|---|---:|
| profiles checked | 106483 |
| minimum preempt rank | 18 |
| multiplicity-weighted erasure-weighted pair bound at \(p_e=0.1\) | \(1.04348\times10^{-3}\) |

After tagged-mask weighting and \(K\)-user lifting,

| \(K\) | \(p_e=0.1\) | \(p_e=0.2\) | \(p_e=0.3\) |
|---:|---:|---:|---:|
| 30 | \(6.923\times10^{-5}\) | \(4.732\times10^{-5}\) | \(1.437\times10^{-5}\) |
| 40 | \(9.310\times10^{-5}\) | \(6.364\times10^{-5}\) | \(1.932\times10^{-5}\) |

## F. Composite Finite-Instance Predictor

For no-SIC phase III in the tested \(K=30,40\) regime, the current composite
predictor is

\[
\widehat{\mathrm{PDP}}_{\mathrm{III}}
=
P_{\mathrm{sch,III}}+P_{\mathrm{preempt}},
\]

\[
\widehat{\mathrm{PHP}}_{\mathrm{III}}
\le
P_{\mathrm{preempt}}+P_{\mathrm{hallucination}}.
\]

Numerically,

| \(K\) | \(p_e\) | schedule UE | pair preempt | hallucination PHP | PDP prediction | PHP bound |
|---:|---:|---:|---:|---:|---:|---:|
| 30 | 0.100 | 0.286244 | \(6.923\times10^{-5}\) | \(6.933\times10^{-9}\) | 0.286313 | \(6.924\times10^{-5}\) |
| 30 | 0.200 | 0.706210 | \(4.732\times10^{-5}\) | \(1.333\times10^{-9}\) | 0.706258 | \(4.732\times10^{-5}\) |
| 30 | 0.300 | 0.920784 | \(1.437\times10^{-5}\) | \(2.057\times10^{-10}\) | 0.920798 | \(1.437\times10^{-5}\) |
| 40 | 0.100 | 0.286244 | \(9.310\times10^{-5}\) | \(2.916\times10^{-7}\) | 0.286337 | \(9.339\times10^{-5}\) |
| 40 | 0.200 | 0.706210 | \(6.364\times10^{-5}\) | \(5.607\times10^{-8}\) | 0.706274 | \(6.370\times10^{-5}\) |
| 40 | 0.300 | 0.920784 | \(1.932\times10^{-5}\) | \(8.651\times10^{-9}\) | 0.920803 | \(1.933\times10^{-5}\) |

The predictor is schedule-dominant: the correction terms are below the current
targeted simulation resolution.

## G. Validation Resolution

The leading schedule term can be validated at high empirical resolution because
it depends only on the tagged user's erasure mask.  A schedule-only Monte Carlo
with \(K\) users per frame and 25000 frames per \(K\) gives:

| \(K\) | \(p_e\) | sampled users | exact \(P_{\mathrm{sch,III}}\) | empirical | \(z\)-score |
|---:|---:|---:|---:|---:|---:|
| 30 | 0.100 | 750000 | 0.286244 | 0.286333 | 0.17 |
| 30 | 0.200 | 750000 | 0.706210 | 0.706413 | 0.39 |
| 30 | 0.300 | 750000 | 0.920784 | 0.921087 | 0.97 |
| 40 | 0.100 | 1000000 | 0.286244 | 0.286021 | -0.49 |
| 40 | 0.200 | 1000000 | 0.706210 | 0.705432 | -1.71 |
| 40 | 0.300 | 1000000 | 0.920784 | 0.920563 | -0.82 |

Thus the dominant PDP term is predictive at the resolution ordinary Monte
Carlo can see.

The targeted K-user probes condition on schedule-success users and test whether
the first-valid path from the true root is preempted by a wrong path.  Zero
observed wrong paths must be interpreted at finite resolution.

If \(n\) completed checks produce zero wrong paths, the one-sided \(95\%\)
zero-event upper bound is

\[
u_n=1-0.05^{1/n}.
\]

Because the targeted probe is conditioned on schedule success, the comparable
conditional pair-preemption scale is

\[
r_{\mathrm{cond}}
=
\frac{P_{\mathrm{preempt}}}{1-P_{\mathrm{sch,III}}}.
\]

To make a zero-event simulation upper bound fall below a target rate \(r\),
the required number of completed checks is

\[
n_{\mathrm{req}}(r)
=
\left\lceil
\frac{\log 0.05}{\log(1-r)}
\right\rceil .
\]

For \(K=40,p_e=0.1\), \(n=59\), so

\[
u_{59}=0.049508,
\qquad
(1-P_{\mathrm{sch,III}})u_{59}=0.035336.
\]

The analytic pair-preemption term \(9.310\times10^{-5}\) is only
\(0.2635\%\) of this unconditional empirical resolution.  Thus zero observed
wrong paths is consistent with the theory but does not empirically prove a
\(10^{-4}\)-scale event rate.

Across the tested phase-III \(K=30,40\) points, the completed targeted checks
are below the zero-event resolution required to observe the analytic pair term:

| \(K\) | \(p_e\) | \(r_{\mathrm{cond}}\) | completed checks | \(n_{\mathrm{req}}(r_{\mathrm{cond}})\) |
|---:|---:|---:|---:|---:|
| 30 | 0.100 | \(9.699\times10^{-5}\) | 44 | 30885 |
| 30 | 0.200 | \(1.610\times10^{-4}\) | 13 | 18598 |
| 30 | 0.300 | \(1.813\times10^{-4}\) | 16 | 16513 |
| 40 | 0.100 | \(1.304\times10^{-4}\) | 59 | 22966 |
| 40 | 0.200 | \(2.166\times10^{-4}\) | 18 | 13829 |
| 40 | 0.300 | \(2.439\times10^{-4}\) | 10 | 12282 |

The PHP/root probes have an even larger resolution gap.  For example, the
rank-corrected first-moment PHP bound at \(K=40,p_e=0.1\) is
\(2.916\times10^{-7}\), which would require about \(1.03\times10^7\)
user-equivalent zero-event checks before a 95% empirical upper bound reached
the analytic scale.  The root probes therefore serve as gross implementation
stress tests; the publishable PHP statement must be analytic rather than
Monte-Carlo-certified.

The PHP component audit gives a sharper decomposition:

| \(K\) | \(p_e\) | pair-preempt | pure hallucination | composite PHP bound | composite-scale zero-event checks |
|---:|---:|---:|---:|---:|---:|
| 30 | 0.100 | \(6.923\times10^{-5}\) | \(6.933\times10^{-9}\) | \(6.924\times10^{-5}\) | 43267 |
| 30 | 0.200 | \(4.732\times10^{-5}\) | \(1.333\times10^{-9}\) | \(4.732\times10^{-5}\) | 63305 |
| 30 | 0.300 | \(1.437\times10^{-5}\) | \(2.057\times10^{-10}\) | \(1.437\times10^{-5}\) | 208467 |
| 40 | 0.100 | \(9.310\times10^{-5}\) | \(2.916\times10^{-7}\) | \(9.339\times10^{-5}\) | 32076 |
| 40 | 0.200 | \(6.364\times10^{-5}\) | \(5.607\times10^{-8}\) | \(6.370\times10^{-5}\) | 47031 |
| 40 | 0.300 | \(1.932\times10^{-5}\) | \(8.651\times10^{-9}\) | \(1.933\times10^{-5}\) | 154988 |

Thus in the \(K=30,40\) regime the composite PHP bound is dominated by the
exact pair-preemption term.  The pure-hallucination first moment is dominated
by the symmetric phase-III root0/root6, weight-2, rank-exponent-96 accepted
shapes; it is not the visible PHP bottleneck in this finite instance.

Direct two-user pair-decoder probes give the closest executable check of the
pair-preemption correction.  They condition on accepted tagged two-erasure
masks, add one alternate user, and run the actual first-valid-path search.  The
full gap-representative \(1000\)-trial probe at \(p_e=0.1\) observed zero
preemptions in \(13\) represented mask rows, but its \(K=40\) projected
\(95\%\) union upper is \(0.040299\), while the exact affine-rank correction is
\(9.310\times10^{-5}\).  Thus this direct probe is \(432.9\) times too coarse
to resolve the analytic correction.  A deeper root10-only \(20000\)-trial probe
also observed zero preemptions and gives a local \(K=40\) upper
\(1.336\times10^{-5}\), but the root10 mask contributes only
\(6.889\times10^{-7}\) in the raw analytic projection.  These probes rule out
large implementation-level pair failures; they do not replace the exact
finite-instance affine enumeration.

## H. Multi-Alternate Diagnostics

Three-color profiles use the tagged user plus two alternate users.  For a
two-erasure path with 14 known sections, the number of canonical three-color
restricted-growth profiles is

\[
S(14,3)=788970.
\]

The chunk-aware diagnostics at \(K=40,p_e=0.1\) are generated by
`research/uace_multicolor_chunk_aggregate.py`, with a machine-readable
continuation manifest in `research/uace_multicolor_manifest.json`.  The
aggregate removes duplicate profile intervals, such as the root10 \(0\)--\(1000\)
window contained in the root10 \(0\)--\(5000\) report.

| component | profile windows | max coverage/mask | \(K=40,p_e=0.1\) extra |
|---|---:|---:|---:|
| gap representatives | 12 windows of \(0\)--\(1000\) | 0.001267 | small part of aggregate |
| root0/root6 \((1,4)\) | each has \(1000\)--\(6000\) added | 0.007605 | \(5.511\times10^{-13}\) each |
| root10 \((6,10)\) | \(0\)--\(5000\), \(5000\)--\(10000\), \(10000\)--\(15000\) | 0.019012 | dominant part of aggregate |
| overlap-removed 3-color aggregate | 17 included windows | 0.019012 | \(2.790\times10^{-7}\) |

Compared with the exact two-color \(K=40,p_e=0.1\) term
\(9.310\times10^{-5}\), the overlap-removed three-color finite-window
aggregate has ratio

\[
\frac{2.790\times10^{-7}}{9.310\times10^{-5}}
=2.997\times10^{-3}.
\]

For context, the older profile-window probes were:

| probe | coverage/mask | extra | ratio to two-color exact |
|---|---:|---:|---:|
| 3-color, 5000 profiles/mask over 13 gap representatives | 0.006337 | \(1.883\times10^{-7}\) | 0.002022 |
| 3-color, root10 first 50000 profiles | 0.063374 | \(2.773\times10^{-7}\) | 0.002979 |
| 3-color, root10 offset 50000--60000 | 0.012675 | \(2.874\times10^{-12}\) | \(3.087\times10^{-8}\) |

The latest root10 \(10000\)--\(15000\) chunk contributes only
\(7.535\times10^{-12}\) at \(K=40,p_e=0.1\), giving further evidence that the
large root10 contribution is concentrated in the earliest profile windows.
The two added symmetric root0/root6 chunks have even smaller lifted
contribution, \(5.511\times10^{-13}\) each.

These are evidence that multi-alternate paths are not the visible error driver
in the tested regime.  They are not a theorem-level full three-color bound.
The next mathematical target is a profile transfer matrix or equivalent DP
that sums all multi-color identity profiles without profile-order truncation.

## I. Remaining Non-Theorem Components

The following remain outside the current theorem-level claim:

- dependency-aware pure-hallucination enumeration for schedule-failed roots;
- full multi-color profile DP;
- full phase-III root-sweep validation without node caps;
- SIC, including false deletion and propagation.
