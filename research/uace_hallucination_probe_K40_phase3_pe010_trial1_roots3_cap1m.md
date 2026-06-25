# UACE Hallucination Root Probe

- `K = 40`
- `L = 16`
- `M = 3`
- `pe = 0.1`
- `phase = 3`
- trials: `1`
- roots per attempt: `3` (`0` means all effective roots)
- max nodes per root: `1000000`

Theory:

- schedule UE: `0.286244`
- first-moment PHP bound: `2.916e-07`

Aggregate:

- sampled roots: `15`
- valid outputs found: `4`
- hallucinations found: `0`
- aborted roots: `4`

| seed | attempt | sampled roots | valid outputs | true outputs | hallucinations | aborted roots | node visits | seconds |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 8400 | phase-II root 0 | 3 | 1 | 1 | 0 | 0 | 203035 | 4.11 |
| 8400 | phase-II root 8 | 3 | 1 | 1 | 0 | 0 | 3558 | 0.03 |
| 8400 | phase-III root 0 | 3 | 2 | 2 | 0 | 1 | 2465657 | 50.66 |
| 8400 | phase-III root 6 | 3 | 0 | 0 | 0 | 3 | 3000003 | 62.39 |
| 8400 | phase-III root 10 | 3 | 0 | 0 | 0 | 0 | 4785 | 0.04 |

Interpretation:

- This is an empirical PHP stress test, not a full PDP decoder run.
- A hallucination is counted only when a final-valid decoded message is not in the transmitted user set.
- Aborted roots are runtime inconclusive; they are not counted as hallucinations.

