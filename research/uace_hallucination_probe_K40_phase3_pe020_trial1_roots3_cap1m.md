# UACE Hallucination Root Probe

- `K = 40`
- `L = 16`
- `M = 3`
- `pe = 0.2`
- `phase = 3`
- trials: `1`
- roots per attempt: `3` (`0` means all effective roots)
- max nodes per root: `1000000`

Theory:

- schedule UE: `0.706210`
- first-moment PHP bound: `5.607e-08`

Aggregate:

- sampled roots: `15`
- valid outputs found: `0`
- hallucinations found: `0`
- aborted roots: `6`

| seed | attempt | sampled roots | valid outputs | true outputs | hallucinations | aborted roots | node visits | seconds |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 5420 | phase-II root 0 | 3 | 0 | 0 | 0 | 0 | 155676 | 3.20 |
| 5420 | phase-II root 8 | 3 | 0 | 0 | 0 | 0 | 3697 | 0.03 |
| 5420 | phase-III root 0 | 3 | 0 | 0 | 0 | 3 | 3000003 | 63.14 |
| 5420 | phase-III root 6 | 3 | 0 | 0 | 0 | 3 | 3000003 | 62.10 |
| 5420 | phase-III root 10 | 3 | 0 | 0 | 0 | 0 | 3508 | 0.03 |

Interpretation:

- This is an empirical PHP stress test, not a full PDP decoder run.
- A hallucination is counted only when a final-valid decoded message is not in the transmitted user set.
- Aborted roots are runtime inconclusive; they are not counted as hallucinations.

