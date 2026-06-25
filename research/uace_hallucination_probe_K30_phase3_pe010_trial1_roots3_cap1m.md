# UACE Hallucination Root Probe

- `K = 30`
- `L = 16`
- `M = 3`
- `pe = 0.1`
- `phase = 3`
- trials: `1`
- roots per attempt: `3` (`0` means all effective roots)
- max nodes per root: `1000000`

Theory:

- schedule UE: `0.286244`
- first-moment PHP bound: `6.933e-09`

Aggregate:

- sampled roots: `15`
- valid outputs found: `7`
- hallucinations found: `0`
- aborted roots: `0`

| seed | attempt | sampled roots | valid outputs | true outputs | hallucinations | aborted roots | node visits | seconds |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 8300 | phase-II root 0 | 3 | 3 | 3 | 0 | 0 | 31768 | 0.64 |
| 8300 | phase-II root 8 | 3 | 1 | 1 | 0 | 0 | 2180 | 0.02 |
| 8300 | phase-III root 0 | 3 | 1 | 1 | 0 | 0 | 2072679 | 42.95 |
| 8300 | phase-III root 6 | 3 | 2 | 2 | 0 | 0 | 1734676 | 36.64 |
| 8300 | phase-III root 10 | 3 | 0 | 0 | 0 | 0 | 2603 | 0.02 |

Interpretation:

- This is an empirical PHP stress test, not a full PDP decoder run.
- A hallucination is counted only when a final-valid decoded message is not in the transmitted user set.
- Aborted roots are runtime inconclusive; they are not counted as hallucinations.

