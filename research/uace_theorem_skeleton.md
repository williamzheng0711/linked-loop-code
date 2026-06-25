# UACE LLC Bound: Theorem Skeleton

Status: working theorem draft for the current no-SIC UACE decoder.

## Model

Let $K$ users transmit linked-loop codewords of $L$ sections over the UACE.  Each section uses alphabet size

$$
Q=2^J,
$$

and is erased independently with probability $p_e$.  The A-channel output in a section is the set of distinct non-erased symbols.

For a tagged user, let

$$
E=(E_0,\ldots,E_{L-1})\in\{0,1\}^L
$$

be its erasure mask.  For any bad-mask set $\mathcal{B}$,

$$
P(\mathcal{B})
=
\sum_{e\in\mathcal{B}}
p_e^{|e|}(1-p_e)^{L-|e|}.
$$

## Proposition 1: Exact Current-Schedule Term

For a fixed current decoder phase $\phi$, define

$$
S_\phi(e)=
\mathbf{1}\{
\text{the current root/recovery schedule rejects erasure mask }e
\}
$$

under a collision-free abstraction.  Then the erasure-only drop term is exactly

$$
P_{\mathrm{schedule},\phi}
=
\sum_{e\in\{0,1\}^L}
S_\phi(e)p_e^{|e|}(1-p_e)^{L-|e|}.
$$

For $L=16,M=3$ in the current repo profile,

$$
P_{\mathrm{schedule,II}}
=
\Pr[|E|\ge2],
$$

and

$$
P_{\mathrm{schedule,III}}
=
\Pr[|E|\ge3]+33p_e^2(1-p_e)^{14}.
$$

## Proposition 2: Exact A-List Occupancy Scale

Let $S_\ell$ be the A-list size in one section.  Then

$$
\lambda_A
:=
\mathbb{E}[S_\ell]
=
Q\left[
1-\left(1-\frac{1-p_e}{Q}\right)^K
\right].
$$

For a tagged non-erased symbol, the exact collision probability is

$$
\rho_A
=
1-\left(1-\frac{1-p_e}{Q}\right)^{K-1}.
$$

The full one-section PMF can be written as a stable occupancy recursion.  With $P_i(s)=\Pr[S=s]$ after processing $i$ users,

$$
P_{i+1}(s)
\mathrel{+}=
P_i(s)
\left[
p_e+(1-p_e)\frac{s}{Q}
\right],
$$

and

$$
P_{i+1}(s+1)
\mathrel{+}=
P_i(s)
(1-p_e)\frac{Q-s}{Q}.
$$

Under the independent-section random-symbol abstraction,

$$
\mathbb{E}\prod_{\ell\in T}S_\ell
=
\lambda_A^{|T|}.
$$

Thus the product factor $\lambda_A^{L-w}$ in the false-survivor first moment is an exact product moment in this abstraction, not a Jensen approximation to $\mathbb{E}[S^{L-w}]$.

## Proposition 3: Accepted-Shape False-Survivor First Moment

For a current decoder attempt $a$, define

$$
A_a(w,e)
=
\#\{
\text{erasure masks of weight }w\text{ accepted by attempt }a
\text{ with parity-rank exponent }e
\}.
$$

The count $A_a(w,e)$ is finite and exact: it is computed by the same schedule automaton used for $P_{\mathrm{schedule}}$, including root rotation, erasure-slot constraints, sequential recovery, dirty-saver rejection, and local rank checks, followed by an exact GF(2) rank computation for the erased-info feasibility system.

For an erasure mask, the observed parity constraints for a false path have the form

$$
A x_{\mathrm{erased}}+Bz_{\mathrm{known}}=0.
$$

For uniform known symbols, the probability that some erased-info assignment satisfies these equations is

$$
2^{-e},
\qquad
e=\operatorname{rank}([A\ B])-\operatorname{rank}(A).
$$

Assume false path parity syndromes are uniform under this rank profile.  The first moment of false full-message survivors in attempt $a$ satisfies

$$
\mathbb{E}N_{\mathrm{false},a}
\le
\sum_{w=0}^{L}
\sum_e
A_a(w,e)\lambda_A^{L-w}2^{-e}.
$$

Summing over attempts up to phase $\phi$ gives

$$
\mathbb{E}N_{\mathrm{false},\phi}
\le
\sum_{a\in\mathcal{A}_\phi}
\sum_{w=0}^{L}
\sum_e
A_a(w,e)\lambda_A^{L-w}2^{-e}.
$$

## Corollary: PDP/PHP Prediction

For no-SIC UACE, the current working bound is

$$
\mathrm{PDP}_\phi
\le
P_{\mathrm{schedule},\phi}
+
\frac{\mathbb{E}N_{\mathrm{false},\phi}}{K},
$$

and by Markov's inequality,

$$
\mathrm{PHP}_\phi
\le
\frac{\mathbb{E}N_{\mathrm{false},\phi}}{K}.
$$

The division by $K$ converts the total false full-message expectation into a per-active-user hallucination probability scale.

## Current Numerical Regime

For $L=16,M=3,J=16,r=8$, the accepted shape profiles are:

| attempt | $A_a(w)$ | total |
|---|---:|---:|
| phase-I root 0 | $w=0:1$ | 1 |
| phase-II root 0 | $w=0:1,\ w=1:15$ | 16 |
| phase-II root 8 | $w=0:1,\ w=1:1$ | 2 |
| phase-III root 0 | $w=0:1,\ w=1:15,\ w=2:75$ | 91 |
| phase-III root 6 | $w=0:1,\ w=1:15,\ w=2:75$ | 91 |
| phase-III root 10 | $w=0:1,\ w=1:2,\ w=2:1$ | 4 |

The exact rank exponent profile for accepted masks in this repo profile is:

| weight $w$ | rank exponent $e$ | naive exponent $8(L-w)$ |
|---:|---:|---:|
| 0 | 128 | 128 |
| 1 | 112 | 120 |
| 2 | 96 | 112 |

For example, at $K=40,p_e=0.3$,

$$
P_{\mathrm{schedule,III}}=0.920784,
\qquad
\frac{\mathbb{E}N_{\mathrm{false,III}}}{K}
=8.651\times10^{-9}.
$$

This still predicts schedule-dominant PDP and empirically invisible PHP for K=40, matching the completed targeted path-interference probes.  The same rank-corrected first moment is no longer negligible for K=100 in phase III at low erasure probability, so the theorem should not claim uniform smallness over all K up to 100.

The contribution breakdown is highly concentrated.  For $K=40,p_e=0.3$, the two dominant terms are:

| contributor | shapes | PHP contribution | share |
|---|---:|---:|---:|
| phase-III root 6, $w=2,e=96$ | 75 | $4.296\times10^{-9}$ | 0.497 |
| phase-III root 0, $w=2,e=96$ | 75 | $4.296\times10^{-9}$ | 0.497 |

Thus the next dependency-aware refinement should start with two-erasure accepted shapes in the root-0/root-6 phase-III attempts.

The first identity-profile refinement confirms that this is the right place to look, but also shows why a final-parity-only theorem is still too loose.  If a candidate two-erasure path is allowed to switch between exactly two true users, the exact final-parity exponent can be as small as

$$
e_{\mathrm{2color}}=16,
$$

far below the independent-symbol exponent $e=96$.  Enumerating all accepted two-erasure masks gives the following necessary-condition component scales:

| $K$ | $p_e$ | two-color final-parity profile scale per user |
|---:|---:|---:|
| 30 | 0.100 | $2.141\times10^{-1}$ |
| 30 | 0.300 | $6.348\times10^{-3}$ |
| 40 | 0.100 | $2.880\times10^{-1}$ |
| 40 | 0.300 | $8.537\times10^{-3}$ |

An order-aware symbolic executor was then used to replay the current sequential decoder on representative accepted two-erasure masks.  It replaces existential erased variables by the actual linear forms recovered from saver parities and includes the intermediate full-saver consistency checks.  On the gap-representative inventory, this did not reject any two-color profiles and left the minimum exponent at 16.  Thus sequential recovery equations alone are not the missing protection.

However, the current targeted decoder probes over $K\in\{30,40\}$ and $p_e\in\{0.1,0.2,0.3\}$ found no wrong-path preemption among 160 completed schedule-success path checks.  A separate true-path ordering diagnostic over the same K/erasure grid found that all 174 profiled schedule-success true paths are final-valid, while the true continuation can have many prior siblings in the current row/DFS order.  A pre-true-path verifier then searched only sibling subtrees before the true continuation and observed zero wrong preemptions among 65 completed checks, with zero aborts.  Therefore the final theorem must not count every final-parity-consistent identity profile as a decoder error.  It must refine the identity-profile state by the first-valid-path ordering used by the implementation, and it must account for dependency/coalescence among many profiles that correspond to overlapping path events.

A pair-level Monte Carlo diagnostic gives the first quantitative evidence for that coalescence.  For each gap-representative accepted two-erasure mask, it fixed one tagged user and one alternate user, exhaustively enumerated all $2^{14}-1=8191$ canonical two-color profiles, and counted the event that at least one valid false profile appears before the true path in row order.  Across $2600$ checked pair events, the observed preemption count was 0.  The zero-event 95% upper bound is

$$
p_{\mathrm{pair\ preempt}}
\le
1-0.05^{1/2600}
=1.152\times10^{-3}.
$$

This was then strengthened by a direct two-user decoder probe.  For each accepted two-erasure gap representative, the probe fixes the tagged user's erased sections, samples one alternate LLC user and that alternate user's erasures, constructs the actual two-row UACE list, and runs the current first-valid-path decoder from the tagged root.  It therefore tests the theorem object directly rather than counting profiles.

The direct pair probe found:

| setting | pair decoder runs | preemptions | path failures | aborted |
|---|---:|---:|---:|---:|
| $p_e\in\{0.1,0.2,0.3\}$, 200 trials/mask/pe | 7800 | 0 | 0 | 0 |
| $p_e=0.1$, 1000 trials/mask | 13000 | 0 | 0 | 0 |

For the $p_e=0.1$ strengthened run, the projected extra term is:

| $K$ | schedule UE | empirical extra | 95% binomial upper | 95% union upper |
|---:|---:|---:|---:|---:|
| 30 | 0.286244 | 0 | 0.028744 | 0.029966 |
| 40 | 0.286244 | 0 | 0.038090 | 0.040299 |

This is not yet a theorem, but it converts the remaining problem into a precise object.  For an attempt $a$ and tagged erased mask $m$, define

$$
q_{a,m}
=
\Pr[
\text{one alternate user makes the current first-valid decoder output a non-tagged message}
\mid E_{\mathrm{tag}}=m,a
].
$$

The conservative lift to $K$ users is the union form

$$
P_{\mathrm{preempt}}
\le
\sum_{a,m}
p_e^{|m|}(1-p_e)^{L-|m|}
c_{a,m}
\min\{1,(K-1)q_{a,m}\},
$$

where $c_{a,m}$ is the mask multiplicity represented by a gap class.  A predictive independence-scale approximation replaces the last factor by

$$
1-(1-q_{a,m})^{K-1}.
$$

The next theorem should upper-bound $q_{a,m}$ analytically, using row-order and profile coalescence rather than individual-profile first moments.

A first finite-instance analytic upper bound is now available for the two-user term.  For every fixed two-color identity profile, the ordered decoder equations are homogeneous GF(2) equations.  Row-order preemption can be written as a finite disjoint union of affine GF(2) systems:

$$
\begin{aligned}
&\text{all earlier alternate-selected symbols equal the tagged symbols},\\
&\text{at the first differing selected section, the prefix bits agree and }x_{\rm alt,b}=0,\ x_{\rm tag,b}=1.
\end{aligned}
$$

Thus each first-difference event has exact probability $2^{-r}$, where $r$ is the rank of the corresponding affine system.  Summing these disjoint first-difference events over all two-color profiles gives a rigorous pair-level union bound.  For the phase-III gap-representative accepted two-erasure masks:

| quantity | value |
|---|---:|
| two-color profiles checked | 106483 |
| preempt-feasible profiles | 87651 |
| minimum parity-valid rank | 16 |
| minimum preempt rank | 18 |
| multiplicity-weighted raw pair union bound | $1.15942\times10^{-3}$ |
| multiplicity-weighted erasure-weighted pair bound, $p_e=0.1$ | $1.04348\times10^{-3}$ |

After multiplying by the tagged two-erasure mask probability and lifting to $K$ users, the erasure-weighted conservative union extra term is:

| $K$ | $p_e=0.1$ | $p_e=0.2$ | $p_e=0.3$ |
|---:|---:|---:|---:|
| 30 | $6.923\times10^{-5}$ | $4.732\times10^{-5}$ | $1.437\times10^{-5}$ |
| 40 | $9.310\times10^{-5}$ | $6.364\times10^{-5}$ | $1.932\times10^{-5}$ |
| 100 | $2.363\times10^{-4}$ | $1.615\times10^{-4}$ | $4.904\times10^{-5}$ |

This is the current best publishable-shaped path-preemption term for UACE/no-SIC phase III in the repo/TCom finite instance.  It is still a union bound over two-color profiles, so it does not yet cover pure hallucination or multi-alternate coalescence exactly, but it has predictive scale and agrees with the zero-preemption simulations.

A first multi-alternate probe checks 3-color identity profiles, i.e. paths using the tagged user plus two alternate users.  Full enumeration has $S(14,3)=788970$ profiles per mask, so the present calculation is a truncation probe:

| probe | profiles checked | $K$ | $p_e=0.1$ erasure-weighted extra |
|---|---:|---:|---:|
| 3-color, 5000 profiles/mask over 13 gap reps | 65000 | 40 | $1.883\times10^{-7}$ |
| 3-color, root10 only, 50000 profiles | 50000 | 40 | $2.773\times10^{-7}$ |

These values are far below the two-color term.  The theorem version should compress this profile family by a transfer matrix rather than by raw enumeration.

For $L=16,M=3$, those 75 two-erasure shapes have circular gap profile:

| circular gaps | count |
|---|---:|
| $(3,13)$ | 12 |
| $(4,12)$ | 14 |
| $(5,11)$ | 14 |
| $(6,10)$ | 14 |
| $(7,9)$ | 14 |
| $(8,8)$ | 7 |

Equivalently, the accepted two-erasure class begins at circular separation 3.  Gap 1 and gap 2 masks are rejected by the current schedule because local saver parities are mutually contaminated.

The additional targeted phase-III probes now include $K=30,40$ and $p_e=0.1,0.2,0.3$.  All completed schedule-success checks have wrong path count 0 and targeted PHP 0.  Separate hallucination root probes sampled schedule-failed/effective roots at $K=30,40$ and found no hallucinated messages among completed searches.  Some hard phase-III root-0/root-6 empty searches hit the node cap, so the root hallucination probes remain a sanity check rather than a proof.

## Remaining Gap

The first-moment theorem controls total false survivors but still treats false parity syndromes as uniform and does not classify dependencies by switch/return structure.  A publish-ready next theorem should replace or refine Proposition 3 with a transfer-matrix/path-shape DP that tracks:

- path switch and return as identity profiles;
- pair-level first-preempt event under first-valid-path ordering;
- dependency/coalescence among many identity profiles;
- pure hallucination;
- parity-rank dependence across sections;
- schedule-failed roots that can still output false messages.
