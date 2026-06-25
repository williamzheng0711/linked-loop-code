# Fast UACE Empirical Probe

- `K = 40`
- `L = 16`
- `M = 3`
- `pe = 0.1`
- `phase = 3`
- trials: `1`
- max nodes per root: `10000`

| metric | mean |
|---|---:|
| pdp | 0.650000 |
| php | 0.000000 |
| decoded | 14.000000 |
| correct | 14.000000 |
| false_positive | 0.000000 |
| schedule_fail_emp | 0.200000 |
| rank_fail_emp | 0.000000 |
| seconds | 23.503724 |
| node_visits | 1195290.000000 |
| aborted_roots | 108.000000 |

Per-trial rows:

| seed | PDP | PHP | decoded | correct | false positive | schedule fail | rank fail | seconds | node visits | aborted roots |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6310 | 0.650000 | 0.000000 | 14 | 14 | 0 | 0.200000 | 0.000000 | 23.50 | 1195290 | 108 |

Note: this fast wrapper is exact only when `aborted_roots = 0`.  If roots abort, the PDP is an upper-biased drop estimate because those roots are treated as undecoded.

