# UACE Hallucination Root Probe

- `K = 30`
- `L = 16`
- `M = 3`
- `pe = 0.2`
- `phase = 3`
- trials: `1`
- roots per attempt: `3` (`0` means all effective roots)
- max nodes per root: `1000000`

Theory:

- schedule UE: `0.706210`
- first-moment PHP bound: `1.333e-09`

Aggregate:

- sampled roots: `15`
- valid outputs found: `6`
- hallucinations found: `0`
- aborted roots: `0`

| seed | attempt | sampled roots | valid outputs | true outputs | hallucinations | aborted roots | node visits | seconds |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 5320 | phase-II root 0 | 3 | 2 | 2 | 0 | 0 | 27458 | 0.72 |
| 5320 | phase-II root 8 | 3 | 0 | 0 | 0 | 0 | 1673 | 0.01 |
| 5320 | phase-III root 0 | 3 | 2 | 2 | 0 | 0 | 824355 | 17.90 |
| 5320 | phase-III root 6 | 3 | 2 | 2 | 0 | 0 | 1258482 | 27.26 |
| 5320 | phase-III root 10 | 3 | 0 | 0 | 0 | 0 | 2638 | 0.02 |

Interpretation:

- This is an empirical PHP stress test, not a full PDP decoder run.
- A hallucination is counted only when a final-valid decoded message is not in the transmitted user set.
- Aborted roots are runtime inconclusive; they are not counted as hallucinations.

