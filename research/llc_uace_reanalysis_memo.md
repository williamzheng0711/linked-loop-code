# LLC UACE Re-Analysis Research Memo

Date: 2026-06-21

Status note: this early memo is superseded by
`research/llc_uace_bound_report.md`, which contains the updated phase-III
schedule automaton, validated mask counts, figures, and corrected numerical
tables.  Some phase-III numbers below are from the pre-validation proxy.

This memo implements the first-round deliverable in the LLC UACE re-analysis plan.  Its goal is not to modify the simulator, but to pin down why the problem is still worth studying and how to replace the loose TCom-style bounds with a tighter, event-informative analysis.

## Executive Verdict

The LLC is still meaningful, but the right claim is narrow:

- It is a serious erasure-resilient outer code for unsourced A/B-channel abstractions.
- It should be compared most directly with tree-code, t-tree/list-recoverable-code, and outer LDPC style disambiguation schemes under the same outer-channel model.
- It should not be positioned as a universal replacement for modern full URA physical-layer schemes such as fading/MIMO URA, ODMA, sparse IDMA, blind detection, or integrated sensing/random access. Those systems solve a larger end-to-end channel problem and often use the A-channel only as an abstraction after inner decoding.

The most promising new contribution is a structural finite-length analysis of LLC path ambiguity under erasures.  The TCom analysis correctly identifies important mechanisms, but it compresses them into very crude union-bound events.  A better analysis should separately quantify:

- exact A-channel occupancy and collision multiplicity;
- recoverability of erasure patterns under the LLC memory/rank constraints;
- false parity-consistent paths, grouped by path-switch structure and parity-rank exponent;
- practical-decoder losses caused by root choice, greedy path selection, and SIC deletion.

The first theorem target should be UACE without SIC for an intrinsic multi-root path-list decoder.  The repo/published decoder should then be treated as a practical implementation whose additional failure modes are measured and bounded separately.

## Baseline Sources

Local source material:

- EUSIPCO 2023 local PDF: `Coding for the Unsourced A-Channel With Erasures: The Linked Loop Code`.
- ICASSP 2024 local PDF: `Coding for the Unsourced B-Channel With Erasures: Enhancing the Linked Loop Code`.
- TCom 2025 local PDF: `Linked-Loop Codes for the Unsourced A- and B-Channels With Erasures`.

External anchors:

- LLC UACE arXiv version: https://arxiv.org/abs/2312.02160
- LLC UBCE/eLLC arXiv version: https://arxiv.org/abs/2406.08767
- TCom 2025 DOI record: https://doi.org/10.1109/TCOMM.2025.3604325
- Finite-blocklength A-channel bounds: https://arxiv.org/abs/2210.01951
- CCS + AMP/BP line: https://arxiv.org/abs/2010.04364
- List-recoverable CCS/t-tree line: https://arxiv.org/abs/2201.07695
- URA survey map: https://arxiv.org/abs/2409.14911

## What The Existing Papers Establish

The EUSIPCO paper establishes the LLC as a first code specifically designed for UACE.  The important conceptual move is to use tail-biting parity links so that a missing section can be recovered from nearby parity constraints, unlike a standard tree code whose parity flow is essentially one directional.

The ICASSP paper extends the model to UBCE/eLLC.  The multiset output preserves multiplicity, and this makes SIC much more natural than in UACE.  This helps motivate LLC as a family of outer codes, but it should not be the first target for a tighter UACE theorem.

The TCom paper gives the fullest public LLC description.  It defines:

- UACE and UBCE with i.i.d. erasures;
- LLC sections `v(l) = w(l).p(l)`, with `m(l) + p(l) = J`;
- memory `M`, where parity section `p(l)` depends on the previous `M` information sections in a tail-biting way;
- phase-I decoding for zero-erasure codewords;
- phase-II and phase-II-e decoding with one or more `NaN` placeholders;
- PDP and PHP as the main error metrics.

The repo matches this structure.  In `playground/general_lib.py`, `phase1_decoder` searches parity-consistent paths over the A/B-channel output lists.  `phase2plus_decoder` rotates the root, allows up to `d` missing sections, and calls `Path_goes_section_l` from `playground/general_utils.py`.  The current launcher settings use `B = 128`, `J = 16`, `L in {15,16}`, and `M in {2,3}`.

## Why The TCom Bounds Are Loose

The current TCom Section IV is useful as a taxonomy of events, but not as a predictive finite-length theory.

Main looseness points:

- Collision probability is reduced to powers of `K 2^{-J}`.  This misses the exact A-channel occupancy law and the multiplicity of collisions at a section.
- The run-length analysis uses a union bound for `M` consecutive collision/erasure events.  It ignores overlap, circular tail-biting, and root restarts.
- Hallucination is treated as roughly "two runs of collisions."  A false LLC path is actually a parity-consistent sequence with a rank exponent depending on its switch pattern.
- Phase-II errors are reduced to parity collision times erasure.  This misses the rank condition for reconstructing a lost section from whichever saver sections remain available.
- Unrecoverable erasure patterns are not proved in the paper; the stated lemma is too opaque for reuse.
- SIC is mixed into the theory as an additive correction, but SIC creates a coupled stochastic process: true sections can be removed by hallucinated messages, and multiplicity information differs sharply between UACE and UBCE.

There is also a decoder-model issue.  For an exhaustive path-list decoder without SIC, a no-erasure true path is not dropped merely because an alternative path exists; the true path remains in the list.  Drops in phase I are mainly a practical-decoder phenomenon caused by greedy path choice, list truncation, SIC, or root handling.  Hallucinations, by contrast, are intrinsic: any false parity-consistent full path is a false output unless filtered later.

This suggests splitting theory into two layers:

1. Intrinsic LLC path-list theory: exhaustive, multi-root, no SIC.
2. Practical decoder theory: root schedule, `Paths[0]` selection, finite list management, and SIC/error propagation.

## Survey Positioning

The LLC remains worth studying if the model is stated honestly.

Closest competitors:

- Tree code / CCS outer code: the original CCS outer disambiguation strategy.  Strong baseline for unsourced A-channel style list stitching, but not designed around section erasures.
- Outer LDPC / BP disambiguation: can be adapted to A-channel list disambiguation and may offer better soft iterative behavior, but its erasure-specific finite-length story is different.
- t-tree / list-recoverable code: can recover multiple missing sections in principle, and is therefore the most relevant "maybe better code" family.  It typically pays in decoding complexity and different parameter optimization.
- Finite-blocklength A-channel bounds: useful as a converse/achievability benchmark for the underlying A-channel abstraction, but not a direct LLC decoder analysis.

Broader but less direct competitors:

- AMP/BP CCS, fading/MIMO URA, sparse IDMA, ODMA, blind detection, and integrated sensing/random access schemes.  These should be surveyed as the larger URA context, not as one-to-one alternatives to LLC outer coding.

Practical conclusion: a paper or note titled around "tight finite-length analysis of LLC over UACE" is meaningful.  A paper claiming LLC is the best modern URA code would be poorly framed.

## Proposed Model

Primary channel: UACE.

Parameters:

- `K`: number of active users.
- `L`: number of sections.
- `J`: bits per section symbol, so `Q = 2^J`.
- `M`: LLC memory/window size.
- `p_e`: per-section erasure probability.
- `m_l`: information bits in section `l`.
- `p_l = J - m_l`: parity bits in section `l`.

Repo/TCom instantiation:

- `B = 128`
- `J = 16`
- `L = 16`, with `m_l = 8`, `p_l = 8`
- or `L = 15`, with alternating `m_l in {8,9}`
- `M in {2,3}`

Decoder for first theorem:

- Multi-root exhaustive path-list decoder.
- No SIC.
- No arbitrary selection of the first surviving path.
- A candidate path is output only if it satisfies every available parity constraint and all erased sections can be reconstructed by the LLC rank rule.

Practical decoder comparison:

- Use the existing repo behavior as a second-layer target:
  - phase I starts from section 0;
  - phase II rotates to selected roots such as 0 and 8;
  - `phase2plus_decoder` keeps `Paths[0]` when multiple paths survive;
  - SIC deletes sections of decoded paths when enabled.

## Tight-Bound Ingredients

### 1. Exact A-Channel Occupancy

The first correction is to replace the approximation `K 2^{-J}` with exact occupancy terms.

Let `Q = 2^J`.  For a tagged user's section symbol, conditioned on the tagged symbol not being erased, the number of other non-erased users colliding with it is

```text
C_l ~ Binomial(K - 1, (1 - p_e) / Q).
```

Therefore

```text
rho_A(K,J,p_e) = P[C_l >= 1]
               = 1 - (1 - (1 - p_e)/Q)^(K - 1).
```

Conditioned on exactly `n` non-erased competitors,

```text
rho_A(n,J) = 1 - (1 - 1/Q)^n.
```

The observed A-list cardinality in a section obeys the standard occupancy law:

```text
E[|Y_l|] = Q * (1 - (1 - (1 - p_e)/Q)^K).
```

For false-path enumeration, the multiplicity distribution is more informative than only `rho_A`:

```text
P[C_l = c] = binom(K - 1, c)
             ((1 - p_e)/Q)^c
             (1 - (1 - p_e)/Q)^(K - 1 - c).
```

This directly improves both PDP/PHP estimates because a section with two or more colliders gives more branch choices than a section with exactly one collider.

### 2. Exact Circular Run Probabilities

The TCom proof uses a union bound for a run of `M` bad events.  Because LLC is tail-biting, the exact object is a circular Bernoulli sequence.

For a generic bad-section probability `rho`, define `R_circ(L,M,rho)` as the probability that a length-`L` circular binary sequence contains at least `M` consecutive bad positions.

A finite-state transfer matrix computes it exactly:

- States are `(M-1)`-bit suffixes.
- Appending a new bit has weight `rho` for bad and `1-rho` for good.
- Transitions that create `M` consecutive bad bits are forbidden.
- Taking `trace(T^L)` sums valid circular sequences, including wrap-around consistency.

Then

```text
R_circ(L,M,rho) = 1 - trace(T^L).
```

This replaces the loose bound

```text
(L - M) rho^M
```

and should be used for collision runs, erasure runs, and mixed collision/erasure runs.

For the repo regimes `L <= 16`, brute-force enumeration over all `2^L` section patterns is even simpler and exact.

### 3. Rank-Based Erasure Recoverability

The recoverability condition should be expressed by rank, not only by "erasures are separated by `M`."

For a lost section `l`, its candidate saver sections are

```text
S_l = {l+1, l+2, ..., l+M} mod L.
```

For a given erasure set `E`, the available saver set is initially

```text
A_l(E) = S_l \ E.
```

A lost section `l` is linearly recoverable if the concatenated transfer matrix from `w(l)` into the parity portions of available saver sections has rank at least `m_l`:

```text
rank( concat_{s in A_l(E)} G_{l,s} ) >= m_l.
```

For multiple erasures, recovery can be iterative.  Once a lost section is recovered, it can help evaluate later parity equations.  The exact intrinsic unrecoverable indicator is:

```text
Recover(E) = fixed point of rank-peeling over erased sections.
UE(E) = 1 if not all erased sections are recovered.
```

Thus the exact unrecoverable probability is

```text
P_UE = sum_{E subset [L]} p_e^|E| (1-p_e)^(L-|E|) * UE(E).
```

For `L = 15` or `16`, this sum is tiny: at most `65536` patterns.  This is a much stronger replacement for the omitted Lemma 6 proof and can also expose when the current repo special cases for `lostSection == 1` are necessary.

### 4. False-Path Enumeration By Parity Rank

The cleanest PHP analysis is through the expected number of false valid paths.

Let a candidate path be

```text
a = (a_0, a_1, ..., a_{L-1}),
```

where `a_l` indexes an observed symbol in section `l`.  It is false if it is not one of the transmitted payload paths.

For a fixed path-shape `a`, define `r(a)` as the rank of the independent parity constraints that are genuinely tested against wrong information fragments.  Then the random-coding probability that this false path passes all checks is approximately/exactly

```text
P[path a is parity-consistent] = 2^{-r(a)}.
```

The structural bound is

```text
P_PHP <= E[ N_false ]
      = E[ sum_{a in Y_0 x ... x Y_{L-1}, a false} 2^{-r(a)} ].
```

A sharper Poisson/Chen-Stein style approximation is

```text
P_PHP approx 1 - E[ exp(-Nbar_false(Y)) ],
```

where

```text
Nbar_false(Y) = sum_{a false in product_l Y_l} 2^{-r(a)}.
```

This formulation is substantially more informative than "two collision runs" because it tells which path-switch structures dominate PHP.

For theory, paths can be grouped by equality pattern rather than enumerated over all `K^L` identities.  The state only needs enough information to know:

- whether the current path is on the tagged true payload;
- which of the previous `M` decider sections use the same underlying payload identity;
- whether the path has left and later returned to the tagged payload;
- how many independent parity bits have been imposed.

This leads naturally to a dynamic program over local path states.

### 5. PDP For The Intrinsic Decoder

For the intrinsic exhaustive decoder without SIC:

- A zero-erasure true path is not dropped by a competing path; it remains a valid output.
- Therefore phase-I PDP for zero-erasure users is zero, except for root/model/path-list constraints not present in the intrinsic decoder.
- PDP comes mainly from:
  - section erasure pattern is unrecoverable;
  - lost sections are reconstructable in principle but parity collision causes a wrong reconstruction;
  - practical truncation or filtering removes the true path.

For a clean first theorem, state:

```text
P_PDP <= P_UE + P_wrong_reconstruction + P_root_failure
```

with `P_root_failure = 0` for a true multi-root decoder and nonzero for the repo/published root schedule.

For one erased section `l`, if a subset `A_l` of saver sections is used and the rank condition holds, wrong reconstruction requires a parity-consistent wrong vector.  The natural exponent is the rank surplus:

```text
P_wrong_reconstruction(l | A_l)
   <= 2^{m_l - rank(concat_{s in A_l} G_{l,s})}
```

and equals zero when the linear solution is unique over the known correct neighboring sections.  In practice, wrong reconstruction is driven by collided/incorrect saver symbols, so the above should be multiplied by the probability that the saver path itself has switched.

### 6. Practical Decoder Penalties

The repo decoder adds several non-intrinsic error sources:

- fixed root schedule rather than all roots;
- rotating roots only in a few hard-coded phase calls;
- selecting `Paths[0]` when multiple surviving paths exist;
- deleting decoded sections under SIC;
- using A-channel lists after multiplicity removal, which makes UACE SIC fragile.

A useful decomposition is:

```text
P_PDP(repo) <= P_PDP(intrinsic)
             + P_missed_root
             + P_wrong_first_path
             + P_SIC_deletes_true_section
             + P_list_or_runtime_truncation.
```

Similarly,

```text
P_PHP(repo) <= P_PHP(intrinsic)
             + P_SIC_propagated_hallucination.
```

These are not merely implementation details.  They explain why a bound for the published pseudo-code may fail to predict the actual repo curves.

## Candidate First Theorem

A publishable first theorem should avoid overpromising and focus on the intrinsic decoder.

Suggested statement:

```text
Theorem A (UACE LLC intrinsic finite-length bound).
Consider an LLC ensemble with parameters (K,L,J,M,{m_l},{p_l}) over the UACE
with i.i.d. erasure probability p_e.  Assume full-rank local generator blocks and
use a multi-root exhaustive path-list decoder without SIC.  Then

P_PDP <= P_UE + P_switched_reconstruction,

where P_UE is the exact rank-peeling erasure-pattern probability

P_UE = sum_{E subset [L]} p_e^|E| (1-p_e)^(L-|E|) UE(E),

and P_switched_reconstruction is bounded by a finite-state path-switch DP using
the exact A-channel occupancy law.  Moreover

P_PHP <= E[ sum_{false paths a} 2^{-r(a)} ],

where r(a) is the parity-rank exponent induced by the switch pattern of a.
For the homogeneous case m_l=m, p_l=p, this expectation reduces to a transfer
matrix over the last M path identity states.
```

This theorem is intentionally modular.  It is useful because each term is computable for `L <= 16` and interpretable:

- `P_UE`: pure erasure geometry/rank problem.
- `P_switched_reconstruction`: erasure plus collision problem.
- `P_PHP`: false complete codeword problem.

## Immediate Validation Plan

Use the existing repo settings:

```text
K = 100
L = 16
J = 16
M = 3
p_e in {0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2}
channel_type = A
SIC = 0 for first theorem checks
```

Compare four quantities:

- empirical PDP/PHP from the current simulator;
- original TCom bounds;
- exact occupancy plus circular-run bound;
- structural rank/path bound.

Sanity tests:

- `p_e = 0`: erasure unrecoverability vanishes; only false-path/PHP effects remain.
- `K = 1`: A-channel collision disappears; PDP should equal only unrecoverable erasure behavior.
- large `J`: collision and false-path terms vanish exponentially.
- `M = 1`: path-switch/run analysis should collapse to a simple first-order Markov/circular-run case.

Instrumentation needed later:

- number of symbols per section after UACE multiplicity removal;
- per-section collision multiplicities for each tagged user;
- number of surviving paths per root before final check;
- whether each output path is true or false;
- erasure set for every dropped user;
- rank-peeling result for every erasure set;
- whether a drop was caused by wrong root, no surviving true path, `Paths[0]`, or SIC deletion.

## Current Numerical Snapshot

The first analytical calculator is now available at:

```text
research/uace_bound_explorer.py
```

For the TCom-style UACE regime, it generates:

```text
research/uace_bound_tcom_regime.md
```

The first empirical probe is also available at:

```text
research/uace_empirical_probe.py
```

It reuses the existing playground encoder/decoder, but wraps the run with structured metrics: PDP, PHP, number decoded, correct outputs, false positives, and empirical erasure-count classes.

Initial findings for `K=100`, `L=16`, `J=16`, `M=3`:

- The exact tagged collision probability is only about `1.2e-3` to `1.5e-3` over `p_e in [0,0.2]`.
- The exact circular probability of `M=3` consecutive tagged collisions is about `3e-8` to `5e-8`, so phase-I collision-run events cannot explain the main PDP trend.
- A one-erasure decoder has a large erasure-count floor: `P[#erasures >= 2]` is about `0.485` at `p_e=0.1`.
- The TCom-style geometric unrecoverable event is smaller, about `0.231` at `p_e=0.1`.
- The repo-matrix rank-peeling unrecoverable probability is much smaller, about `0.0166` at `p_e=0.1`.

This strongly suggests that a simulation-correlated bound should decompose PDP into:

```text
PDP approx rank-peeling UE + practical decoder penalty + small collision/path term.
```

If the measured simulation PDP is close to `P[#erasures >= 2]`, the current practical decoder is mostly behaving like a one-erasure decoder.  If it is closer to the rank-peeling curve, then phase-II-e is exploiting the LLC's true multi-erasure recovery potential.  The next required step is therefore instrumentation, not more algebra in isolation.

Pilot empirical result:

```text
research/uace_empirical_probe_K30_pe010.md
```

For `K=30`, `L=16`, `M=3`, `p_e=0.1`, phase-II decoding without SIC gave PDP `0.5333` and PHP `0` in one trial.  In that same trial, the empirical fraction of users with two or more erased sections was also `0.5333`.  This is consistent with the first proxy: phase-II-only PDP is dominated by the `#erasures >= 2` event.

A direct `K=100`, `p_e=0.1`, phase-II interactive pilot was interrupted after more than two minutes in `phase1_decoder`.  The bottleneck is the naive accept-all path expansion before enough parity checks are available.  Full TCom-regime empirical overlays should therefore be run either offline or after adding lighter instrumentation/pruning.

The first bound-vs-empirical overlay tool is:

```text
research/uace_trend_overlay.py
```

The first erasure-only practical schedule classifier is:

```text
research/uace_schedule_bound.py
```

Small-regime outputs:

```text
research/uace_trend_overlay_K6.md
research/uace_trend_overlay_K10.md
research/uace_trend_overlay_K6_phase3_pe010.md
research/uace_trend_overlay_K6_phase3_pe010_trials3.md
research/uace_schedule_bound_phase2.md
research/uace_schedule_bound_phase3.md
```

Observed trend:

- For phase-II decoding, the empirical PDP follows the empirical `#erasures >= 2` fraction in the `K=6`, `K=10`, and `K=30` pilots.  This supports the interpretation that phase-II-only decoding is primarily a one-erasure decoder in practice.
- For a phase-III pilot at `K=6`, `p_e=0.1`, the empirical `#erasures >= 2` fraction was `0.3333`, but PDP dropped to `0.1667`.  This is the first direct sign that the existing phase-II-e style decoder can recover some multi-erasure patterns.
- After adding per-trial erasure-mask diagnostics, the stronger phase-III pilot with three trials at `K=6`, `p_e=0.1` gave empirical PDP `0.1111`, empirical practical-schedule fail `0.1111`, and empirical ideal-rank fail `0`.  In this sample, every observed drop is explained by the current root/schedule erasure-only classifier.
- PHP was zero in these small pilots, consistent with the analytical result that collision/path hallucination terms are extremely small for `J=16`.

The next theoretical target is therefore sharper:

```text
PDP_phaseII    ~= P[#erasures >= 2] + tiny path term,
PDP_phaseII-e  ~= P[unrecoverable under actual root/schedule/rank rules]
                  + tiny path term.
```

For `L=16`, `M=3`, the practical schedule classifier gives:

- phase-II practical schedule UE equals `P[#erasures >= 2]`, exactly matching the one-erasure-decoder interpretation;
- phase-III practical schedule UE is `0.2405` at `p_e=0.1`, close to the TCom geometric UE `0.2311`;
- ideal rank-peeling UE is only `0.0166` at `p_e=0.1`.

This separates the research problem into two useful bounds:

```text
current decoder bound:  PDP <= schedule_UE + collision/path extras,
ideal LLC bound:       PDP <= rank_peeling_UE + collision/path extras.
```

The current decoder bound is the one that should track the existing simulation curves.  The ideal LLC bound quantifies the headroom available if the root schedule and sequential NaN handling are improved.

The theorem-style draft and compact summary are now available at:

```text
research/uace_bound_theorem_draft.md
research/uace_bound_summary.md
```

The draft states the phase-II closed form, the current phase/root schedule mask-sum bound, and the ideal rank-peeling bound in proof-ready notation.  The summary table keeps the three key curves side by side:

```text
phase-II schedule, phase-III schedule, ideal rank-peeling.
```

## Recommended Next Steps

1. Instrument the current simulator without changing decoding behavior.
   - Log path counts, root choices, erasure sets, and true/false outputs.
   - Keep this separate from the core decoder to avoid changing experiment semantics.
   - Compare empirical PDP/PHP against `research/uace_bound_tcom_regime.md`.

2. Extend the exact erasure-pattern analyzer beyond the first rank-peeling version.
   - Input: `L`, `M`, `messageLens`, `parityLens`, `Gis`.
   - Output: exact `P_UE(p_e)`, pattern classes, and sample bad masks.

3. Extend the occupancy/run-bound calculator.
   - Input: `K`, `J`, `L`, `M`, `p_e`.
   - Output: exact `rho_A`, collision multiplicity moments, and `R_circ`.

4. Develop the false-path DP.
   - Start with homogeneous `L=16`, `m=p=8`, `M=3`.
   - State should track only the last `M` identity/equality classes and the accumulated parity-rank exponent.
   - Validate by brute force for tiny `K,L,J`.

5. Decide paper direction.
   - If `P_UE + rank/path PHP` tracks simulation within one order of magnitude, this is a strong theory note.
   - If practical-decoder penalties dominate, the better paper may be "analysis-informed decoder redesign" rather than just a tighter bound.

## Open Technical Risks

- Independence of parity checks is not automatic for every switch pattern because LLC constraints overlap through shared information sections.  The rank-exponent method is designed to handle this, but it must be proved carefully.
- The A-channel removes multiplicity.  A collision can hide an erasure or create a path switch; these are coupled when conditioning on the observed list.
- SIC over UACE is inherently dangerous because removing a hallucinated path removes symbols that may belong to true users.  UBCE handles this better because multiplicity is available.
- The repo has special-case recovery logic for `lostSection == 1` under some `(L,M)` settings.  The rank-peeling analyzer should tell whether these are mathematical necessities or implementation patches.

## Bottom Line

The research program is meaningful.  The best path is to make the first result a clean, exact/DP-aided finite-length analysis for UACE LLC without SIC, then add practical decoder penalties.  This will produce a tighter and more informative theory than the TCom Section IV union bounds while keeping the claims aligned with the actual contribution of LLC.
