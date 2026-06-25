# UACE Hallucination Root Probe

- `K = 40`
- `L = 16`
- `M = 3`
- `pe = 0.3`
- `phase = 3`
- trials: `2`
- roots per attempt: `5` (`0` means all effective roots)
- max nodes per root: `50000`

Theory:

- schedule UE: `0.920784`
- first-moment PHP bound: `8.651e-09`

Aggregate:

- sampled roots: `50`
- valid outputs found: `0`
- hallucinations found: `0`
- aborted roots: `20`

| seed | attempt | sampled roots | valid outputs | true outputs | hallucinations | aborted roots | node visits | seconds |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 6400 | phase-II root 0 | 5 | 0 | 0 | 0 | 0 | 151107 | 3.26 |
| 6400 | phase-II root 8 | 5 | 0 | 0 | 0 | 0 | 5550 | 0.04 |
| 6400 | phase-III root 0 | 5 | 0 | 0 | 0 | 5 | 250005 | 5.42 |
| 6400 | phase-III root 6 | 5 | 0 | 0 | 0 | 5 | 250005 | 5.41 |
| 6400 | phase-III root 10 | 5 | 0 | 0 | 0 | 0 | 4460 | 0.04 |
| 6401 | phase-II root 0 | 5 | 0 | 0 | 0 | 0 | 195070 | 4.13 |
| 6401 | phase-II root 8 | 5 | 0 | 0 | 0 | 0 | 4536 | 0.03 |
| 6401 | phase-III root 0 | 5 | 0 | 0 | 0 | 5 | 250005 | 5.51 |
| 6401 | phase-III root 6 | 5 | 0 | 0 | 0 | 5 | 250005 | 5.51 |
| 6401 | phase-III root 10 | 5 | 0 | 0 | 0 | 0 | 4640 | 0.05 |

Interpretation:

- This is an empirical PHP stress test, not a full PDP decoder run.
- A hallucination is counted only when a final-valid decoded message is not in the transmitted user set.
- Aborted roots are runtime inconclusive; they are not counted as hallucinations.

