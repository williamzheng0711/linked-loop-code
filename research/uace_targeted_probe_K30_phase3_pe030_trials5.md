# UACE Targeted Path Probe

- `K = 30`
- `L = 16`
- `M = 3`
- `pe = 0.3`
- `phase = 3`
- trials: `5`
- max nodes per schedule-success user: `500000`

Theory:

- schedule UE: `0.920784`
- first-moment PHP bound: `4.412e-15`

| metric | mean |
|---|---:|
| schedule_fail_emp | 0.893333 |
| rank_fail_emp | 0.286667 |
| schedule_success_users | 3.200000 |
| checked_users | 3.200000 |
| path_fail_users | 0.000000 |
| wrong_path_users | 0.000000 |
| aborted_users | 0.000000 |
| targeted_php | 0.000000 |
| node_visits | 334411.400000 |
| seconds | 7.557755 |

Per-trial rows:

| seed | schedule fail | schedule-success users | checked | path fail | wrong path | aborted | targeted PHP | node visits | seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5300 | 0.933333 | 2 | 2 | 0 | 0 | 0 | 0.000000 | 263295 | 5.94 |
| 5301 | 0.833333 | 5 | 5 | 0 | 0 | 0 | 0.000000 | 357371 | 7.97 |
| 5302 | 0.900000 | 3 | 3 | 0 | 0 | 0 | 0.000000 | 358703 | 8.11 |
| 5303 | 0.900000 | 3 | 3 | 0 | 0 | 0 | 0.000000 | 355690 | 7.87 |
| 5304 | 0.900000 | 3 | 3 | 0 | 0 | 0 | 0.000000 | 336998 | 7.90 |

Interpretation:

- `path_fail_users` among schedule-success users is the directly observed extra path-interference term in this targeted probe.
- `wrong_path_users` counts preemption by a parity-consistent path whose decoded message is not the tagged user's message.
- The probe is exact for the targeted users only when `aborted = 0`; otherwise it is conservative for PDP.

