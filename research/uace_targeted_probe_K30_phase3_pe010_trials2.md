# UACE Targeted Path Probe

- `K = 30`
- `L = 16`
- `M = 3`
- `pe = 0.1`
- `phase = 3`
- trials: `2`
- max nodes per schedule-success user: `2000000`

Theory:

- schedule UE: `0.286244`
- first-moment PHP bound: `6.933e-09`

| metric | mean |
|---|---:|
| schedule_fail_emp | 0.266667 |
| rank_fail_emp | 0.000000 |
| schedule_success_users | 22.000000 |
| checked_users | 22.000000 |
| path_fail_users | 0.000000 |
| wrong_path_users | 0.000000 |
| aborted_users | 0.000000 |
| targeted_php | 0.000000 |
| node_visits | 4274241.000000 |
| seconds | 94.059140 |

Per-trial rows:

| seed | schedule fail | schedule-success users | checked | path fail | wrong path | aborted | targeted PHP | node visits | seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5310 | 0.266667 | 22 | 22 | 0 | 0 | 0 | 0.000000 | 4250466 | 91.52 |
| 5311 | 0.266667 | 22 | 22 | 0 | 0 | 0 | 0.000000 | 4298016 | 96.60 |

Interpretation:

- `path_fail_users` among schedule-success users is the directly observed extra path-interference term in this targeted probe.
- `wrong_path_users` counts preemption by a parity-consistent path whose decoded message is not the tagged user's message.
- The probe is exact for the targeted users only when `aborted = 0`; otherwise it is conservative for PDP.

