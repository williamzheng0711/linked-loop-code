# UACE Empirical Probe

- `K = 50`
- `L = 16`
- `J = 16`
- `M = 3`
- `p_e = 0.1`
- `phase = 1`
- `SIC = 0`
- trials: `2`

| metric | mean | std |
|---|---:|---:|
| pdp | 0.840000 | 0.040000 |
| php | 0.000000 | 0.000000 |
| decoded | 8.000000 | 2.000000 |
| correct | 8.000000 | 2.000000 |
| false_positive | 0.000000 | 0.000000 |
| p0_erasure_emp | 0.160000 | 0.040000 |
| p1_erasure_emp | 0.370000 | 0.010000 |
| pge2_erasure_emp | 0.470000 | 0.050000 |
| schedule_fail_emp | 0.840000 | 0.040000 |
| rank_fail_emp | 0.020000 | 0.020000 |

Per-trial rows:

| seed | PDP | PHP | decoded | correct | false positive | P0 era | P1 era | P>=2 era | schedule fail | rank fail |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8050 | 0.880000 | 0.000000 | 6 | 6 | 0 | 0.120000 | 0.360000 | 0.520000 | 0.880000 | 0.000000 |
| 8051 | 0.800000 | 0.000000 | 10 | 10 | 0 | 0.200000 | 0.380000 | 0.420000 | 0.800000 | 0.040000 |
