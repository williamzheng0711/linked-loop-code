# Fast UACE Empirical Probe

- `K = 30`
- `L = 16`
- `M = 3`
- `pe = 0.1`
- `phase = 3`
- trials: `1`
- max nodes per root: `100000`

| metric | mean |
|---|---:|
| pdp | 0.566667 |
| php | 0.000000 |
| decoded | 13.000000 |
| correct | 13.000000 |
| false_positive | 0.000000 |
| schedule_fail_emp | 0.366667 |
| rank_fail_emp | 0.000000 |
| seconds | 114.043713 |
| node_visits | 5522095.000000 |
| aborted_roots | 47.000000 |

Per-trial rows:

| seed | PDP | PHP | decoded | correct | false positive | schedule fail | rank fail | seconds | node visits | aborted roots |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6310 | 0.566667 | 0.000000 | 13 | 13 | 0 | 0.366667 | 0.000000 | 114.04 | 5522095 | 47 |

Note: this fast wrapper is exact only when `aborted_roots = 0`.  If roots abort, the PDP is an upper-biased drop estimate because those roots are treated as undecoded.

