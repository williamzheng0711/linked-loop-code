# Fast UACE Empirical Probe

- `K = 40`
- `L = 16`
- `M = 3`
- `pe = 0.1`
- `phase = 3`
- trials: `1`
- max nodes per root: `50000`

| metric | mean |
|---|---:|
| pdp | 0.500000 |
| php | 0.000000 |
| decoded | 20.000000 |
| correct | 20.000000 |
| false_positive | 0.000000 |
| schedule_fail_emp | 0.200000 |
| rank_fail_emp | 0.000000 |
| seconds | 107.482529 |
| node_visits | 5287097.000000 |
| aborted_roots | 97.000000 |

Per-trial rows:

| seed | PDP | PHP | decoded | correct | false positive | schedule fail | rank fail | seconds | node visits | aborted roots |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6310 | 0.500000 | 0.000000 | 20 | 20 | 0 | 0.200000 | 0.000000 | 107.48 | 5287097 | 97 |

Note: this fast wrapper is exact only when `aborted_roots = 0`.  If roots abort, the PDP is an upper-biased drop estimate because those roots are treated as undecoded.

