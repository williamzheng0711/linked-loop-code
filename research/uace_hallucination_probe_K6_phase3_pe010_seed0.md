# UACE Hallucination Root Probe

- `K = 6`
- `L = 16`
- `M = 3`
- `pe = 0.1`
- `phase = 3`
- trials: `1`
- roots per attempt: `0` (`0` means all effective roots)
- max nodes per root: `200000`

Theory:

- schedule UE: `0.286244`
- first-moment PHP bound: `8.743e-23`

Aggregate:

- sampled roots: `29`
- valid outputs found: `18`
- hallucinations found: `0`
- aborted roots: `0`

| seed | attempt | sampled roots | valid outputs | true outputs | hallucinations | aborted roots | node visits | seconds |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | phase-II root 0 | 6 | 4 | 4 | 0 | 0 | 1747 | 0.03 |
| 0 | phase-II root 8 | 6 | 2 | 2 | 0 | 0 | 302 | 0.01 |
| 0 | phase-III root 0 | 6 | 5 | 5 | 0 | 0 | 9371 | 0.19 |
| 0 | phase-III root 6 | 6 | 5 | 5 | 0 | 0 | 12258 | 0.21 |
| 0 | phase-III root 10 | 5 | 2 | 2 | 0 | 0 | 207 | 0.01 |

Interpretation:

- This is an empirical PHP stress test, not a full PDP decoder run.
- A hallucination is counted only when a final-valid decoded message is not in the transmitted user set.
- Aborted roots are runtime inconclusive; they are not counted as hallucinations.

