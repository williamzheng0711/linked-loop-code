# UACE Full Phase-III Runtime Note

This note records an attempted full no-SIC phase-III validation run:

```text
python3 research/uace_fast_empirical_probe.py \
  --K 30 --L 16 --M 3 --pe 0.1 --phase 3 --trials 1 --seed 6310 \
  --max-nodes-per-root 5000000 \
  --output research/uace_fast_probe_K30_phase3_pe010_seed6310_cap5m.md
```

The run was manually interrupted after several minutes without producing a
completed trial.  The stack was inside `first_valid_path_for_root`, specifically
inside `general_utils.Path_goes_section_l`, while generating child paths for a
full root-sweep attempt.

Interpretation:

- This is not evidence of a PDP/PHP mismatch.
- It confirms that full phase-III root-sweep validation has a severe runtime
  tail even with the fast first-valid-path wrapper.
- The current validation strategy should therefore remain:
  1. exact schedule enumeration for the erasure-only PDP term;
  2. exact affine GF(2) pair-preemption enumeration for the K-dependent path
     correction;
  3. targeted K-user probes conditioned on schedule-success users;
  4. root-level hallucination probes and rank/first-moment bounds for PHP.

