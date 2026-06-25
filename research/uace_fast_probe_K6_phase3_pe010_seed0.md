# Fast UACE Empirical Probe

- `K = 6`
- `L = 16`
- `M = 3`
- `pe = 0.1`
- `phase = 3`
- trials: `1`
- max nodes per root: `200000`

| metric | mean |
|---|---:|
| pdp | 0.166667 |
| php | 0.000000 |
| decoded | 5.000000 |
| correct | 5.000000 |
| false_positive | 0.000000 |
| schedule_fail_emp | 0.166667 |
| rank_fail_emp | 0.000000 |
| seconds | 0.459044 |
| node_visits | 23885.000000 |
| aborted_roots | 0.000000 |

Per-trial rows:

| seed | PDP | PHP | decoded | correct | false positive | schedule fail | rank fail | seconds | node visits | aborted roots |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.166667 | 0.000000 | 5 | 5 | 0 | 0.166667 | 0.000000 | 0.46 | 23885 | 0 |

Note: this fast wrapper is exact only when `aborted_roots = 0`.  If roots abort, the PDP is an upper-biased drop estimate because those roots are treated as undecoded.

