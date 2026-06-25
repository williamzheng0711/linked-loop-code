# Fast UACE Empirical Probe

- `K = 30`
- `L = 16`
- `M = 3`
- `pe = 0.1`
- `phase = 3`
- trials: `1`
- max nodes per root: `10000`

| metric | mean |
|---|---:|
| pdp | 0.666667 |
| php | 0.000000 |
| decoded | 10.000000 |
| correct | 10.000000 |
| false_positive | 0.000000 |
| schedule_fail_emp | 0.366667 |
| rank_fail_emp | 0.000000 |
| seconds | 15.024725 |
| node_visits | 770607.000000 |
| aborted_roots | 70.000000 |

Per-trial rows:

| seed | PDP | PHP | decoded | correct | false positive | schedule fail | rank fail | seconds | node visits | aborted roots |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6310 | 0.666667 | 0.000000 | 10 | 10 | 0 | 0.366667 | 0.000000 | 15.02 | 770607 | 70 |

Note: this fast wrapper is exact only when `aborted_roots = 0`.  If roots abort, the PDP is an upper-biased drop estimate because those roots are treated as undecoded.

