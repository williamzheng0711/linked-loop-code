# UACE Targeted Path Probe

- `K = 40`
- `L = 16`
- `M = 3`
- `pe = 0.1`
- `phase = 3`
- trials: `1`
- max nodes per schedule-success user: `5000000`

Theory:

- schedule UE: `0.286244`
- first-moment PHP bound: `2.916e-07`

| metric | mean |
|---|---:|
| schedule_fail_emp | 0.125000 |
| rank_fail_emp | 0.000000 |
| schedule_success_users | 35.000000 |
| checked_users | 35.000000 |
| path_fail_users | 0.000000 |
| wrong_path_users | 0.000000 |
| aborted_users | 0.000000 |
| targeted_php | 0.000000 |
| node_visits | 12623219.000000 |
| seconds | 255.638090 |

Per-trial rows:

| seed | schedule fail | schedule-success users | checked | path fail | wrong path | aborted | targeted PHP | node visits | seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5410 | 0.125000 | 35 | 35 | 0 | 0 | 0 | 0.000000 | 12623219 | 255.64 |

Interpretation:

- `path_fail_users` among schedule-success users is the directly observed extra path-interference term in this targeted probe.
- `wrong_path_users` counts preemption by a parity-consistent path whose decoded message is not the tagged user's message.
- The probe is exact for the targeted users only when `aborted = 0`; otherwise it is conservative for PDP.

