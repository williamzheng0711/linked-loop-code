# UACE Targeted Path Probe

- `K = 40`
- `L = 16`
- `M = 3`
- `pe = 0.3`
- `phase = 3`
- trials: `1`
- max nodes per schedule-success user: `5000000`

Theory:

- schedule UE: `0.920784`
- first-moment PHP bound: `1.860e-13`

| metric | mean |
|---|---:|
| schedule_fail_emp | 0.875000 |
| rank_fail_emp | 0.250000 |
| schedule_success_users | 5.000000 |
| checked_users | 5.000000 |
| path_fail_users | 0.000000 |
| wrong_path_users | 0.000000 |
| aborted_users | 0.000000 |
| targeted_php | 0.000000 |
| node_visits | 1621943.000000 |
| seconds | 35.355101 |

Per-trial rows:

| seed | schedule fail | schedule-success users | checked | path fail | wrong path | aborted | targeted PHP | node visits | seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5403 | 0.875000 | 5 | 5 | 0 | 0 | 0 | 0.000000 | 1621943 | 35.36 |

Interpretation:

- `path_fail_users` among schedule-success users is the directly observed extra path-interference term in this targeted probe.
- `wrong_path_users` counts preemption by a parity-consistent path whose decoded message is not the tagged user's message.
- The probe is exact for the targeted users only when `aborted = 0`; otherwise it is conservative for PDP.

