# Fast UACE Empirical Probe

- `K = 30`
- `L = 16`
- `M = 3`
- `pe = 0.1`
- `phase = 3`
- trials: `1`
- max nodes per root: `50000`

| metric | mean |
|---|---:|
| pdp | 0.566667 |
| php | 0.000000 |
| decoded | 13.000000 |
| correct | 13.000000 |
| false_positive | 0.000000 |
| schedule_fail_emp | 0.366667 |
| rank_fail_emp | 0.000000 |
| seconds | 62.626408 |
| node_visits | 3078229.000000 |
| aborted_roots | 50.000000 |

Per-trial rows:

| seed | PDP | PHP | decoded | correct | false positive | schedule fail | rank fail | seconds | node visits | aborted roots |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6310 | 0.566667 | 0.000000 | 13 | 13 | 0 | 0.366667 | 0.000000 | 62.63 | 3078229 | 50 |

Note: this fast wrapper is exact only when `aborted_roots = 0`.  If roots abort, the PDP is an upper-biased drop estimate because those roots are treated as undecoded.

