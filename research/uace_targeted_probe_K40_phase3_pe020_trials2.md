# UACE Targeted Path Probe

- `K = 40`
- `L = 16`
- `M = 3`
- `pe = 0.2`
- `phase = 3`
- trials: `2`
- max nodes per schedule-success user: `5000000`

Theory:

- schedule UE: `0.706210`
- first-moment PHP bound: `5.607e-08`

| metric | mean |
|---|---:|
| schedule_fail_emp | 0.775000 |
| rank_fail_emp | 0.187500 |
| schedule_success_users | 9.000000 |
| checked_users | 9.000000 |
| path_fail_users | 0.000000 |
| wrong_path_users | 0.000000 |
| aborted_users | 0.000000 |
| targeted_php | 0.000000 |
| node_visits | 3947501.000000 |
| seconds | 81.787036 |

Per-trial rows:

| seed | schedule fail | schedule-success users | checked | path fail | wrong path | aborted | targeted PHP | node visits | seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7600 | 0.775000 | 9 | 9 | 0 | 0 | 0 | 0.000000 | 1615441 | 33.30 |
| 7601 | 0.775000 | 9 | 9 | 0 | 0 | 0 | 0.000000 | 6279561 | 130.27 |

Interpretation:

- `path_fail_users` among schedule-success users is the directly observed extra path-interference term in this targeted probe.
- `wrong_path_users` counts preemption by a parity-consistent path whose decoded message is not the tagged user's message.
- The probe is exact for the targeted users only when `aborted = 0`; otherwise it is conservative for PDP.

