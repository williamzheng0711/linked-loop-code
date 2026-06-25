# UACE Hallucination Root Probe

- `K = 30`
- `L = 16`
- `M = 3`
- `pe = 0.3`
- `phase = 3`
- trials: `2`
- roots per attempt: `5` (`0` means all effective roots)
- max nodes per root: `50000`

Theory:

- schedule UE: `0.920784`
- first-moment PHP bound: `2.057e-10`

Aggregate:

- sampled roots: `50`
- valid outputs found: `0`
- hallucinations found: `0`
- aborted roots: `20`

| seed | attempt | sampled roots | valid outputs | true outputs | hallucinations | aborted roots | node visits | seconds |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 6300 | phase-II root 0 | 5 | 0 | 0 | 0 | 0 | 66295 | 1.42 |
| 6300 | phase-II root 8 | 5 | 0 | 0 | 0 | 0 | 2617 | 0.02 |
| 6300 | phase-III root 0 | 5 | 0 | 0 | 0 | 5 | 250005 | 5.51 |
| 6300 | phase-III root 6 | 5 | 0 | 0 | 0 | 5 | 250005 | 5.47 |
| 6300 | phase-III root 10 | 5 | 0 | 0 | 0 | 0 | 3204 | 0.02 |
| 6301 | phase-II root 0 | 5 | 0 | 0 | 0 | 0 | 78953 | 1.68 |
| 6301 | phase-II root 8 | 5 | 0 | 0 | 0 | 0 | 2629 | 0.02 |
| 6301 | phase-III root 0 | 5 | 0 | 0 | 0 | 5 | 250005 | 5.53 |
| 6301 | phase-III root 6 | 5 | 0 | 0 | 0 | 5 | 250005 | 5.59 |
| 6301 | phase-III root 10 | 5 | 0 | 0 | 0 | 0 | 2601 | 0.02 |

Interpretation:

- This is an empirical PHP stress test, not a full PDP decoder run.
- A hallucination is counted only when a final-valid decoded message is not in the transmitted user set.
- Aborted roots are runtime inconclusive; they are not counted as hallucinations.

