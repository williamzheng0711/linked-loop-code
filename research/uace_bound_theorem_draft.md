# UACE LLC Bound Draft

Date: 2026-06-21

Status note: this draft is superseded by
`research/llc_uace_bound_report.md`.  The final report uses the corrected
current-decoder schedule automaton; the older phase-III numerical values below
were produced before matching the automaton to `Path_goes_entry_k`.

This note turns the current numerical exploration into theorem-shaped statements.  The goal is to produce a bound that tracks the existing simulation behavior before trying to prove an idealized, decoder-independent LLC bound.

## 1. Setup

Consider the UACE with `K` active users, `L` sections, alphabet size `Q = 2^J`, and per-section erasure probability `p_e`.  For a tagged user, define the erasure mask

```text
E = (E_0, ..., E_{L-1}) in {0,1}^L,
```

where `E_l = 1` means that section `l` is erased.  Let

```text
Pr(E = e) = p_e^{|e|} (1-p_e)^{L-|e|}.
```

For a set of bad masks `B subset {0,1}^L`, write

```text
P(B) = sum_{e in B} p_e^{|e|} (1-p_e)^{L-|e|}.
```

The current repo/TCom regime is usually

```text
B = 128, J = 16, L in {15,16}, M in {2,3}.
```

This note focuses on `L = 16`, `M = 3`, no SIC, UACE.

## 2. Collision Scale

Conditioned on the tagged user's section not being erased, the number of other non-erased users colliding with the tagged section symbol is

```text
C_l ~ Binomial(K-1, (1-p_e)/2^J).
```

Thus

```text
rho_A(K,J,p_e) = Pr(C_l >= 1)
               = 1 - (1 - (1-p_e)/2^J)^{K-1}.
```

For the TCom benchmark `K=100`, `J=16`, this is about `10^{-3}`, and the exact circular probability of three consecutive tagged collisions is about `10^{-8}`.  Therefore the leading PDP term in the current simulation regime is not a collision-run term.  Collision and false-path terms should be retained in the final theorem, but only as small additive corrections:

```text
PDP <= erasure_schedule_term + O(R_circ(L,M,rho_A)) + PHP-coupled terms.
```

## 3. Phase-II Current Decoder Bound

Phase-II decoding in the current schedule allows at most one `NaN`/erased section per decoded path.  It is run from root 0 and root 8 in the current `L=16` experiments.

Under the collision-free abstraction, phase-II succeeds exactly when the tagged user's erasure mask has weight 0 or 1.  Therefore the erasure-only PDP term is

```text
B_II = {e : |e| >= 2}
P_schedule,II = P(B_II)
              = 1 - (1-p_e)^L - L p_e (1-p_e)^{L-1}.
```

The current-decoder bound for phase-II is

```text
PDP_II <= P_schedule,II + P_path,II,
```

where `P_path,II` contains symbol-collision, parity-collision, false-path, and hallucination side effects.  In the current `J=16` regime, the numerical evidence indicates that `P_path,II` is tiny relative to `P_schedule,II`.

Empirical support:

- In the `K=6` and `K=10` overlays, empirical PDP equals the sampled `#erasures >= 2` fraction for phase-II.
- In the `K=30`, `p_e=0.1` pilot, empirical PDP and sampled `#erasures >= 2` were both `0.5333`.

This is already a simulation-correlated theoretical bound: it explains the phase-II trend and gives the correct dominant term.

## 4. Phase-II-e / Phase-III Schedule Bound

The current phase-III schedule adds attempts with two allowed `NaN` positions and roots 0, 6, and 10.  Its recoverability is not the ideal LLC recoverability; it is constrained by:

- the selected root set;
- the order in which sections are scanned after root rotation;
- the rule that a new `NaN` can be inserted only when previous carried `NaN`s are already known;
- the available saver sections already visible in the prefix;
- the rank of the concatenated local generator blocks.

Define `S_phase(e)` to be 1 if the current phase schedule fails on erasure mask `e` in the collision-free abstraction, and 0 otherwise.  The practical schedule bound is

```text
P_schedule,phase = sum_e S_phase(e) p_e^{|e|} (1-p_e)^{L-|e|}.
```

For phase-III, the current-decoder bound is

```text
PDP_III <= P_schedule,III + P_path,III.
```

For `L=16`, `M=3`, matrix seed 0, `research/uace_schedule_bound.py` gives

```text
p_e      P_schedule,III
0.025    0.012555
0.050    0.058787
0.075    0.138025
0.100    0.240490
0.150    0.468681
0.200    0.671026
```

At `p_e=0.1`, this is close to the TCom geometric unrecoverable term (`0.231148`) and much smaller than the phase-II one-erasure floor (`0.485272`).

Empirical support:

- In the `K=6`, `p_e=0.1`, phase-III three-trial overlay, empirical PDP was `0.111111`.
- On the same sampled masks, empirical schedule-fail fraction was also `0.111111`.
- Empirical ideal-rank fail fraction was `0`.

Thus, in this probe, the current decoder's drops are fully explained by the practical schedule classifier.  No visible collision or hallucination contribution appears.

## 5. Ideal LLC Rank-Peeling Bound

The idealized LLC bound removes the current root/order limitations.  It asks whether an erasure mask is recoverable by any iterative rank-peeling process.

For a lost section `l`, define saver sections

```text
S_l = {l+1, ..., l+M} mod L.
```

Given currently known information sections, the lost `w_l` is recoverable when the concatenated transfer matrix into available saver parity sections has rank at least `m_l`:

```text
rank(concat_{s in available S_l} G_{l,s}) >= m_l.
```

Let `R(e) = 1` if iterative rank-peeling fails on mask `e`, and 0 otherwise.  Then

```text
P_rank = sum_e R(e) p_e^{|e|} (1-p_e)^{L-|e|}.
```

For `L=16`, `M=3`, seed 0:

```text
p_e      P_rank
0.025    0.000255
0.050    0.002070
0.075    0.007024
0.100    0.016618
0.150    0.054677
0.200    0.122841
```

This curve should not be expected to match the current decoder simulation.  It measures the LLC's latent erasure-recovery potential under a more complete root/order strategy.

## 6. Two-Bound Interpretation

The useful split is:

```text
current decoder:
PDP_current <= P_schedule,current + P_path,current

ideal LLC:
PDP_ideal <= P_rank + P_path,ideal
```

where `P_path` includes collision-driven wrong path switches, parity collisions, and hallucinations.  For `J=16`, `K <= 100`, the existing calculations show that the first collision-run contribution is very small, so the leading behavior is erasure-schedule dominated.

This gives a theory that is informative in two ways:

- It tracks current simulation trends through `P_schedule,current`.
- It identifies improvement headroom through the gap `P_schedule,current - P_rank`.

At `p_e=0.1`, `L=16`, `M=3`:

```text
phase-II schedule bound:  0.485272
phase-III schedule bound: 0.240490
ideal rank-peeling bound: 0.016618
```

That gap says the existing phase-III schedule improves dramatically over phase-II, but still leaves substantial decoder-design headroom.

## 7. Proof Roadmap

1. Prove the exact A-channel occupancy lemma:

```text
C_l ~ Binomial(K-1, (1-p_e)/2^J).
```

2. Prove the phase-II closed form:

```text
P_schedule,II = Pr(|E| >= 2).
```

This follows directly from the phase-II single-NaN rule and the existence of at least one non-erased root for all masks of weight 0 or 1.

3. Define the schedule automaton for phase-III:

- state: root, current prefix, carried erasures, recovered erasures;
- transition: reveal next section, optionally carry a `NaN`, update recovered set by rank tests;
- acceptance: all erased sections recovered by the end.

Then the exact schedule bound is a finite mask sum over masks rejected by this automaton.

4. Define the ideal rank-peeling automaton:

- state: recovered set;
- transition: add any erased section whose available saver matrix has rank at least `m_l`;
- acceptance: all erased sections recovered.

5. Add path/collision terms:

```text
PDP <= erasure term + Pr(false/wrong path affects tagged message).
```

For the current numerical regime, the collision term can first be bounded by exact circular-run probabilities using `rho_A`; later it can be sharpened by false-path parity-rank enumeration.

## 8. Immediate Next Experiment

The most useful next simulation is not a larger blind sweep.  It is a targeted validation of the schedule bound:

- sample many erasure masks directly for `L=16`, `M=3`, `p_e in {0.05,0.1,0.15}`;
- compute `schedule_fail_emp` and `rank_fail_emp` without running the expensive path decoder;
- run the full decoder only for masks where schedule and rank disagree, to isolate practical root/order failures.

This will test tightness while avoiding the `K^M` early path-expansion bottleneck.
