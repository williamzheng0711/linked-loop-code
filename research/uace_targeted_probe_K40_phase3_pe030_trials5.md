# UACE Targeted Path Probe

- `K = 40`
- `L = 16`
- `M = 3`
- `pe = 0.3`
- `phase = 3`
- trials: `5`
- max nodes per schedule-success user: `500000`

Theory:

- schedule UE: `0.920784`
- first-moment PHP bound: `1.860e-13`

| metric | mean |
|---|---:|
| schedule_fail_emp | 0.935000 |
| rank_fail_emp | 0.290000 |
| schedule_success_users | 2.600000 |
| checked_users | 2.600000 |
| path_fail_users | 0.600000 |
| wrong_path_users | 0.000000 |
| aborted_users | 0.600000 |
| targeted_php | 0.000000 |
| node_visits | 496550.600000 |
| seconds | 11.018491 |

Per-trial rows:

| seed | schedule fail | schedule-success users | checked | path fail | wrong path | aborted | targeted PHP | node visits | seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5400 | 0.925000 | 3 | 3 | 0 | 0 | 0 | 0.000000 | 473470 | 10.52 |
| 5401 | 0.925000 | 3 | 3 | 1 | 0 | 1 | 0.000000 | 531046 | 11.70 |
| 5402 | 0.975000 | 1 | 1 | 0 | 0 | 0 | 0.000000 | 267374 | 5.86 |
| 5403 | 0.875000 | 5 | 5 | 2 | 0 | 2 | 0.000000 | 1041978 | 23.34 |
| 5404 | 0.975000 | 1 | 1 | 0 | 0 | 0 | 0.000000 | 168885 | 3.67 |

Interpretation:

- `path_fail_users` among schedule-success users is the directly observed extra path-interference term in this targeted probe.
- `wrong_path_users` counts preemption by a parity-consistent path whose decoded message is not the tagged user's message.
- The probe is exact for the targeted users only when `aborted = 0`; otherwise it is conservative for PDP.

