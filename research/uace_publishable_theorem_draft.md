# UACE LLC Finite-Length Bound: Publishable Theorem Draft

Status: consolidated no-SIC UACE theorem draft for the current repo/TCom
finite instance.

This note collects the parts of the analysis that are now sufficiently precise
to be written as a paper theorem.  It deliberately separates theorem-level
finite sums from simulation evidence and from remaining open terms.

For a more paper-facing theorem/proof layout, see
`research/uace_formal_theorem_appendix.md`.

For a consolidated validation certificate and dashboard figure, see
`research/uace_publish_validation_certificate.md` and
`research/figures/uace_publish_validation_dashboard.png`.

## 1. Channel And Decoder Model

There are \(K\) active users.  Each user transmits an LLC codeword with
\(L\) sections and section alphabet size

\[
Q=2^J.
\]

Each section is independently erased with probability \(p_e\).  The UACE
output in section \(\ell\) is the set of distinct non-erased section symbols.
For a tagged user, write its erasure mask as

\[
E=(E_0,\ldots,E_{L-1})\in\{0,1\}^L,
\qquad
\Pr[E=e]=p_e^{|e|}(1-p_e)^{L-|e|}.
\]

The repo/TCom benchmark considered here is

\[
B=128,\qquad J=16,\qquad L=16,\qquad M=3,\qquad m_\ell=p_\ell=8.
\]

The decoder is the current no-SIC phase decoder in the repository, including
its fixed root schedule, scan order, erasure-slot guard, first-visible-saver
linear solve, and final parity check.

## 2. Exact Schedule Term

For phase \(\phi\), define

\[
S_\phi(e)
=
\mathbf 1\{\text{the current phase-}\phi\text{ root/recovery schedule rejects mask }e\}
\]

under a collision-free abstraction.  Then the exact erasure-only drop term is

\[
P_{\mathrm{sch},\phi}
=
\sum_{e\in\{0,1\}^L}
S_\phi(e)p_e^{|e|}(1-p_e)^{L-|e|}.
\]

For the current \(L=16,M=3\) implementation,

\[
P_{\mathrm{sch,I}}
=
\Pr[|E|\ge1],
\]

\[
P_{\mathrm{false,I}}(K)
\le
\frac{1}{K}\lambda_A^{16}2^{-128},
\]

where \(\lambda_A\) is the exact one-section occupancy scale defined in
Section 3.

\[
P_{\mathrm{sch,II}}
=
\Pr[|E|\ge2],
\]

and

\[
P_{\mathrm{sch,III}}
=
\Pr[|E|\ge3]+33p_e^2(1-p_e)^{14}.
\]

Proof sketch: enumerate all masks through the same finite-state schedule
automaton used by the implementation.  Phase I accepts exactly the zero-erasure
mask.  Phase II carries at most one erasure, so the bad set is exactly
\(|E|\ge2\).  Phase III carries at most two erasures; the schedule accepts 87
of the 120 two-erasure masks and rejects 33, giving the closed form above.

It is important that \(P_{\mathrm{sch},\phi}\) is not the full \(K\)-user PDP.
It is the \(K\)-independent erasure-geometry floor obtained after turning off
A-channel collisions and all false paths.  If \(\mathcal S_\phi\) denotes the
schedule-rejection event for the tagged user and \(\mathcal I_{\phi,K}\)
denotes the event that the schedule would accept the tagged user but the
\(K-1\) other users create a preempting path, a returned switch, or a
hallucinated message, then

\[
P_{\mathrm{PDP},\phi}(K)
=
\Pr[\mathcal S_\phi]
+
\Pr[\mathcal S_\phi^c\cap\mathcal I_{\phi,K}]
\le
P_{\mathrm{sch},\phi}+P_{\mathrm{path},\phi}(K).
\]

Thus \(P_{\mathrm{sch},\phi}\) is essentially the \(K=1\) erasure-only
baseline, while \(P_{\mathrm{path},\phi}(K)\) contains the multi-user
interference.  The empirical claim below is only that, for the tested
\(J=16,K=30,40\) no-SIC phase-III regime, the visible PDP scale is dominated
by \(P_{\mathrm{sch},\phi}\); the theorem still carries explicit
\(K\)-dependent correction terms.

## 3. Exact Occupancy Terms

The exact expected A-list size in a section is

\[
\lambda_A
=
Q\left[
1-\left(1-\frac{1-p_e}{Q}\right)^K
\right].
\]

For a tagged non-erased section, the exact collision probability is

\[
\rho_A
=
1-\left(1-\frac{1-p_e}{Q}\right)^{K-1}.
\]

The circular \(M\)-collision-run event is computed by exact mask enumeration
or an equivalent transfer matrix.  In the benchmark \(K=100,J=16,L=16,M=3\),
this term is only \(4.007\times10^{-8}\) at \(p_e=0.1\), far below the schedule
term.

## 4. Rank-Corrected False-Survivor First Moment

For an accepted erasure mask in attempt \(a\), the false-path parity equations
can be written as

\[
A x_{\mathrm{erased}}+Bz_{\mathrm{known}}=0.
\]

For uniform known symbols, the probability that the erased variables can be
chosen to satisfy the equations is

\[
2^{-e},
\qquad
e=\operatorname{rank}([A\ B])-\operatorname{rank}(A).
\]

Let \(A_a(w,e)\) be the exact count of accepted masks of weight \(w\) in
attempt \(a\) with exponent \(e\).  Then

\[
\mathbb E N_{\mathrm{false},a}
\le
\sum_w\sum_e A_a(w,e)\lambda_A^{L-w}2^{-e}.
\]

For the current \(L=16,M=3,J=16\) profile, the accepted-mask exponent profile is

| weight \(w\) | exponent \(e\) | naive exponent \(8(L-w)\) |
|---:|---:|---:|
| 0 | 128 | 128 |
| 1 | 112 | 120 |
| 2 | 96 | 112 |

Phase I only accepts the \(w=0,e=128\) profile, so its finite-\(K\)
false-message/PHP correction is

\[
P_{\mathrm{false,I}}(K)
\le
\frac{1}{K}\lambda_A^{16}2^{-128}.
\]

This first moment is useful as a broad PHP control, but it is too loose for
phase-III two-erasure path switches because it counts many dependent identity
profiles separately.

## 5. Ordered Pair-Preemption Bound

The dominant loose class is phase-III accepted two-erasure masks.  For a fixed
attempt \(a\), accepted tagged mask \(m\), and two-color identity profile
\(\pi\), color 0 denotes the tagged user and color 1 an alternate user.

The current ordered decoder induces homogeneous GF(2) equations

\[
H_{a,m,\pi}X=0
\]

in the two users' information bits \(X\).  A profile preempts the true tagged
path in row order if, at the first selected section where the alternate symbol
differs from the tagged symbol, the alternate symbol is smaller.  This event is
a disjoint union over first-difference section/bit pairs \(d\) of affine systems

\[
H_{a,m,\pi}X=0,\qquad
R_{a,m,\pi,d}X=r_{a,m,\pi,d}.
\]

Thus each disjunct has exact probability

\[
2^{-\operatorname{rank}([H_{a,m,\pi};R_{a,m,\pi,d}])}.
\]

Let \(s_{a,m,\pi,d}\) be the number of alternate-selected section symbols that
must be present up to the first-difference event.  Define the erasure-weighted
pair bound

\[
q_{a,m}(p_e)
\le
\sum_{\pi}
\sum_d
(1-p_e)^{s_{a,m,\pi,d}}
2^{-\operatorname{rank}([H_{a,m,\pi};R_{a,m,\pi,d}])}.
\]

Lifting from one alternate user to \(K-1\) alternate users gives the conservative
finite-length bound

\[
P_{\mathrm{preempt}}
\le
\sum_{a,m}
p_e^{|m|}(1-p_e)^{L-|m|}
c_{a,m}
\min\{1,(K-1)q_{a,m}(p_e)\},
\]

where \(c_{a,m}\) is the multiplicity of the represented mask class.

For the repo/TCom finite instance, exact enumeration over all two-color
profiles for phase-III accepted two-erasure gap representatives gives:

| quantity | value |
|---|---:|
| two-color profiles checked | 106483 |
| preempt-feasible profiles | 87651 |
| minimum parity-valid rank | 16 |
| minimum preempt rank | 18 |
| multiplicity-weighted raw pair bound | \(1.15942\times10^{-3}\) |
| multiplicity-weighted erasure-weighted pair bound at \(p_e=0.1\) | \(1.04348\times10^{-3}\) |

After tagged-mask weighting and \(K\)-user lifting:

| \(K\) | \(p_e=0.1\) | \(p_e=0.2\) | \(p_e=0.3\) |
|---:|---:|---:|---:|
| 30 | \(6.923\times10^{-5}\) | \(4.732\times10^{-5}\) | \(1.437\times10^{-5}\) |
| 40 | \(9.310\times10^{-5}\) | \(6.364\times10^{-5}\) | \(1.932\times10^{-5}\) |
| 100 | \(2.363\times10^{-4}\) | \(1.615\times10^{-4}\) | \(4.904\times10^{-5}\) |

This is the current sharp finite-instance path-preemption term.

## 6. Phase-III Predictor

For the tested K=30/K=40 phase-III regime, use

\[
\widehat{\mathrm{PDP}}_{\mathrm{III}}
=
P_{\mathrm{sch,III}}+P_{\mathrm{preempt}},
\qquad
\widehat{\mathrm{PHP}}_{\mathrm{III}}
\le
P_{\mathrm{preempt}}+P_{\mathrm{hallucination}},
\]

where \(P_{\mathrm{hallucination}}\) is currently controlled by the
rank-corrected first moment and empirical root probes.

The composite finite-instance predictor is:

| \(K\) | \(p_e\) | schedule UE | exact pair preempt | hallucination PHP bound | PDP prediction | PHP bound | targeted wrong path / checked |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 0.100 | 0.286244 | \(6.923\times10^{-5}\) | \(6.933\times10^{-9}\) | 0.286313 | \(6.924\times10^{-5}\) | 0 / 44 |
| 30 | 0.200 | 0.706210 | \(4.732\times10^{-5}\) | \(1.333\times10^{-9}\) | 0.706258 | \(4.732\times10^{-5}\) | 0 / 13 |
| 30 | 0.300 | 0.920784 | \(1.437\times10^{-5}\) | \(2.057\times10^{-10}\) | 0.920798 | \(1.437\times10^{-5}\) | 0 / 16 |
| 40 | 0.100 | 0.286244 | \(9.310\times10^{-5}\) | \(2.916\times10^{-7}\) | 0.286337 | \(9.339\times10^{-5}\) | 0 / 59 |
| 40 | 0.200 | 0.706210 | \(6.364\times10^{-5}\) | \(5.607\times10^{-8}\) | 0.706274 | \(6.370\times10^{-5}\) | 0 / 18 |
| 40 | 0.300 | 0.920784 | \(1.932\times10^{-5}\) | \(8.651\times10^{-9}\) | 0.920803 | \(1.933\times10^{-5}\) | 0 / 13 |

The PHP component audit shows that this PHP bound is pair-dominated in the
tested regime.  For example, at \(K=40,p_e=0.1\),

\[
P_{\mathrm{pair\text{-}preempt}}=9.310\times10^{-5},
\qquad
P_{\mathrm{hallucination}}=2.916\times10^{-7},
\]

so the pair term is about \(319\) times larger than the pure hallucination
first moment.  Across all \(K=30,40\), \(p_e\in\{0.1,0.2,0.3\}\), the pure
hallucination first moment is dominated by the symmetric root0/root6,
two-erasure, rank-exponent-96 accepted shapes.  A zero-event experiment would
still need \(32076\) user-equivalent checks to resolve the composite
\(K=40,p_e=0.1\) PHP scale, and \(1.03\times10^7\) checks to resolve the pure
hallucination scale alone.

The leading schedule term is directly visible in ordinary Monte Carlo.  A
schedule-only erasure-mask simulation with 25000 frames per \(K\) gives:

| \(K\) | \(p_e\) | sampled users | exact \(P_{\mathrm{sch,III}}\) | empirical | \(z\)-score |
|---:|---:|---:|---:|---:|---:|
| 30 | 0.100 | 750000 | 0.286244 | 0.286333 | 0.17 |
| 30 | 0.200 | 750000 | 0.706210 | 0.706413 | 0.39 |
| 30 | 0.300 | 750000 | 0.920784 | 0.921087 | 0.97 |
| 40 | 0.100 | 1000000 | 0.286244 | 0.286021 | -0.49 |
| 40 | 0.200 | 1000000 | 0.706210 | 0.705432 | -1.71 |
| 40 | 0.300 | 1000000 | 0.920784 | 0.920563 | -0.82 |

The extra path term is below the sampling resolution of the completed targeted
simulations, and the observed wrong-path count is zero in every completed row.
The \(K=40,p_e=0.1\) row now aggregates two completed targeted trials: the
sampled schedule-failure rate is \(21/80=0.2625\), while every one of the 59
schedule-success users recovered through its true first-valid path.

The direct two-user pair-decoder probe is a more focused implementation check
of the pair-preemption term.  It conditions on represented accepted two-erasure
masks, inserts one alternate user, and runs the actual first-valid-path search.
For the full gap-representative \(p_e=0.1\) probe, \(13\) represented mask rows
with \(1000\) trials each produced zero preemptions.  The corresponding
\(K=40\) projected \(95\%\) union upper is \(0.040299\), whereas the exact
affine-rank pair correction is \(9.310\times10^{-5}\), so the probe is still
\(432.9\) times too coarse to resolve the analytic correction.  A deeper
root10-only \(20000\)-trial probe also observed zero preemptions and reduces
the local \(K=40\) upper to \(1.336\times10^{-5}\), but root10's raw analytic
contribution is only \(6.889\times10^{-7}\).  Hence direct decoder probes are
implementation sanity checks; the correction itself is an exact finite-instance
enumeration.

The targeted zero-event resolution is still much coarser than the analytic
pair-preemption term.  For example, with \(59\) completed checks at
\(K=40,p_e=0.1\), the one-sided zero-event \(95\%\) conditional upper bound is
\[
1-0.05^{1/59}=0.049508,
\]
which corresponds to an unconditional upper scale
\[
(1-P_{\mathrm{sch,III}})0.049508=0.035336.
\]
The analytic pair term \(9.310\times10^{-5}\) is only \(0.2635\%\) of this
empirical resolution.  Thus zero observed wrong paths is consistent with the
theory but is not, by itself, a tight empirical upper bound.

Equivalently, if

\[
r_{\mathrm{cond}}
=
\frac{P_{\mathrm{preempt}}}{1-P_{\mathrm{sch,III}}},
\]

then a zero-event experiment needs

\[
n_{\mathrm{req}}
=
\left\lceil \frac{\log 0.05}{\log(1-r_{\mathrm{cond}})}\right\rceil
\]

completed schedule-success checks before its \(95\%\) upper bound has the same
scale as the analytic pair term.  The current K=30/K=40 targeted checks are far
below that threshold:

| \(K\) | \(p_e\) | \(r_{\mathrm{cond}}\) | completed checks | \(n_{\mathrm{req}}\) |
|---:|---:|---:|---:|---:|
| 30 | 0.100 | \(9.699\times10^{-5}\) | 44 | 30885 |
| 30 | 0.200 | \(1.610\times10^{-4}\) | 13 | 18598 |
| 30 | 0.300 | \(1.813\times10^{-4}\) | 16 | 16513 |
| 40 | 0.100 | \(1.304\times10^{-4}\) | 59 | 22966 |
| 40 | 0.200 | \(2.166\times10^{-4}\) | 18 | 13829 |
| 40 | 0.300 | \(2.439\times10^{-4}\) | 10 | 12282 |

## 7. Full-Decoder Phase-II Predictive Check

The phase-III full root sweep is runtime-limited, but phase II is fully
executable in the \(K=30,40\) regime.  It gives a clean visible-scale check of
the same decomposition:

\[
P_{\mathrm{PDP},\mathrm{II}}(K)
\le
P_{\mathrm{sch,II}}+P_{\mathrm{path,II}}(K).
\]

For phase II, the current schedule term is \(P_{\mathrm{sch,II}}=\Pr[|E|\ge2]\),
and the first-moment path/PHP correction is below ordinary Monte Carlo
resolution.  Full no-SIC playground decoder runs give:

| \(K\) | \(p_e\) | trials | users | predicted PDP | empirical PDP | sampled schedule fail | \(z\) vs schedule | PHP bound | empirical PHP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 0.100 | 5 | 150 | 0.485272 | 0.433333 | 0.433333 | -1.27 | \(3.026\times10^{-13}\) | 0 |
| 30 | 0.200 | 3 | 90 | 0.859263 | 0.866667 | 0.866667 | 0.20 | \(5.172\times10^{-14}\) | 0 |
| 30 | 0.300 | 3 | 90 | 0.973888 | 0.988889 | 0.988889 | 0.89 | \(6.981\times10^{-15}\) | 0 |
| 40 | 0.100 | 3 | 120 | 0.485272 | 0.475000 | 0.475000 | -0.23 | \(1.696\times10^{-11}\) | 0 |
| 40 | 0.200 | 3 | 120 | 0.859263 | 0.883333 | 0.883333 | 0.76 | \(2.900\times10^{-12}\) | 0 |
| 40 | 0.300 | 3 | 120 | 0.973888 | 0.958333 | 0.958333 | -1.07 | \(3.915\times10^{-13}\) | 0 |

The important diagnostic is not that each finite-sample PDP equals the
closed-form expectation exactly.  Rather, empirical PDP equals the sampled
schedule-failure rate in every completed run, empirical PHP is zero in every
completed run, and all deviations from \(P_{\mathrm{sch,II}}\) are within
\(1.3\) binomial standard errors.  This is the directly observed regime where
the finite-instance theory has predictive power at the visible PDP/PHP scale.
The corresponding figure is
`research/figures/uace_k30_k40_validation.png`.

The machine-checkable summary is `research/uace_predictive_validation_gate.md`.
Its current readout is three PASS rows, six WARN rows, and one GAP row.  The
PASS rows cover phase-II full-decoder predictivity, phase-III schedule-term
resolution, and analytic PHP control.  The WARN rows cover targeted
schedule-success checks, cap-limited phase-III fast-wrapper checks, localized
root-search runtime-tail profiling, true-path ordering diagnostics,
pre-true-path preemption verification, and finite-window three-color
diagnostics.
The localized runtime-tail profile shows that root0/root6 attempts contain
many attempt-schedule users but still abort under the current cap, while false
final-valid paths remain absent.  The true-path ordering diagnostic follows
schedule-success true prefixes over \(K=30,40\) and
\(p_e\in\{0.1,0.2,0.3\}\): all 174 profiled true paths are final-valid, but the
true continuation can have many prior siblings in the current row/DFS order,
with max mean prior siblings 74.83.  The GAP row is full phase-III root-sweep
validation without node caps.

The pre-true-path verifier is the closest executable check of the theorem
event: it searches only sibling subtrees that occur before the tagged user's
true continuation.  Over the \(K=30,40\), \(p_e\in\{0.1,0.2,0.3\}\) seed-6310
grid, it completed all 65 checks with zero aborts and observed zero wrong
preemptions.

## 8. Multi-Alternate And Hallucination Checks

Three-color profiles use the tagged user plus two alternate users.  Full
enumeration has

\[
S(14,3)=788970
\]

canonical profiles per mask, so the present evidence is a truncation/chunk
probe.  The newest aggregate is chunk-aware: it removes duplicate profile
intervals before summing the finite-window union contribution, and
`research/uace_multicolor_manifest.json` records the next chunks.

| probe | included windows | max coverage/mask | \(K\) | \(p_e=0.1\) extra | ratio to 2-color exact |
|---|---:|---:|---:|---:|---:|
| overlap-removed 3-color chunk aggregate | 17 | 0.019012 | 40 | \(2.790\times10^{-7}\) | 0.002997 |

For context, the earlier unaggregated probes were:

| probe | profiles checked | coverage/mask | \(K\) | \(p_e=0.1\) extra | ratio to 2-color exact |
|---|---:|---:|---:|---:|---:|
| 3-color, 5000 profiles/mask over 13 gap representatives | 65000 | 0.006337 | 40 | \(1.883\times10^{-7}\) | 0.002022 |
| 3-color, root10 only, 50000 profiles | 50000 | 0.063374 | 40 | \(2.773\times10^{-7}\) | 0.002979 |
| 3-color, root10 offset 50000--60000 | 10000 | 0.012675 | 40 | \(2.874\times10^{-12}\) | \(3.087\times10^{-8}\) |

These are more than two orders below the two-color term for
\(K=40,p_e=0.1\).  The symmetric root0/root6 \((1,4)\)
\(1000\)--\(6000\) chunks each contribute only \(5.511\times10^{-13}\) at
\(K=40,p_e=0.1\) after erasure and ordered-user lifting, despite adding
\(5000\) profiles per root.  The root10 \(5000\)--\(10000\) chunk contributes
\(9.276\times10^{-8}\), smaller than the first \(0\)--\(5000\) chunk's
\(1.834\times10^{-7}\).  The \(10000\)--\(15000\) chunk drops to
\(7.535\times10^{-12}\), and the older \(50000\)--\(60000\) window is much
smaller still.  This supports the diagnostic claim that three-color paths are
not the visible driver in the tested regime.  It is still not a full 3-color
theorem; a final proof should replace truncation by a profile transfer matrix
or another dependency-aware DP.

Root-level hallucination probes give:

| \(K\) | \(p_e\) | sampled roots | valid outputs | true outputs | hallucinations | aborted roots |
|---:|---:|---:|---:|---:|---:|---:|
| 30 | 0.100 | 15 | 7 | 7 | 0 | 0 |
| 30 | 0.200 | 15 | 6 | 6 | 0 | 0 |
| 40 | 0.100 | 15 | 4 | 4 | 0 | 4 |
| 40 | 0.200 | 15 | 0 | 0 | 0 | 6 |
| 30 | 0.300 | 50 | 0 | 0 | 0 | 20 |
| 40 | 0.300 | 50 | 0 | 0 | 0 | 20 |

No hallucinated message has been observed in completed root searches.  Aborted
empty root searches remain the main empirical limitation.  These root probes
are useful gross stress tests, but they are not large enough to resolve the
analytic PHP scale.  At \(K=40,p_e=0.1\), for example, the first-moment PHP
bound is \(2.916\times10^{-7}\), requiring about \(1.03\times10^7\)
user-equivalent zero-event checks before a \(95\%\) empirical upper bound
could reach that scale.

## 9. What Is Theorem-Level Now

The following components are theorem-level finite computations for the current
repo/TCom instance:

- exact schedule term \(P_{\mathrm{sch},\phi}\);
- exact A-list occupancy and collision probabilities;
- exact accepted-mask rank profile \(e=\operatorname{rank}([A\ B])-\operatorname{rank}(A)\);
- exact two-color affine-rank pair-preemption union bound;
- rank-corrected first-moment Markov control for PHP/hallucination under the
  independent-section random-symbol abstraction;
- composite finite-instance predictor
  \(\widehat{\mathrm{PDP}}=P_{\mathrm{sch}}+P_{\mathrm{preempt}}\),
  \(\widehat{\mathrm{PHP}}\le P_{\mathrm{preempt}}+P_{\mathrm{hallucination}}\).

The following components are still evidence or theorem targets:

- dependency-aware pure-hallucination/path-shape enumeration for
  schedule-failed roots;
- full multi-color identity-profile DP;
- full root-sweep phase-III decoder validation without node caps;
- SIC, which can both help and propagate false deletions.

The predictive-validation gate should be treated as the operational boundary of
the current claim: PASS items are supported at the finite-instance level, WARN
items are useful evidence with explicit limitations, and GAP items must not be
claimed as solved.
