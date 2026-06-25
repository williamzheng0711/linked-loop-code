# UACE Targeted Path Probe

- `K = 30`
- `L = 16`
- `M = 3`
- `pe = 0.2`
- `phase = 3`
- trials: `2`
- max nodes per schedule-success user: `5000000`

Theory:

- schedule UE: `0.706210`
- first-moment PHP bound: `2.863e-14`

| metric | mean |
|---|---:|
| schedule_fail_emp | 0.783333 |
| rank_fail_emp | 0.116667 |
| schedule_success_users | 6.500000 |
| checked_users | 6.500000 |
| path_fail_users | 0.000000 |
| wrong_path_users | 0.000000 |
| aborted_users | 0.000000 |
| targeted_php | 0.000000 |
| node_visits | 1072714.000000 |
| seconds | 23.463697 |

Per-trial rows:

| seed | schedule fail | schedule-success users | checked | path fail | wrong path | aborted | targeted PHP | node visits | seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5200 | 0.866667 | 4 | 4 | 0 | 0 | 0 | 0.000000 | 766660 | 16.50 |
| 5201 | 0.700000 | 9 | 9 | 0 | 0 | 0 | 0.000000 | 1378768 | 30.43 |

Interpretation:

- `path_fail_users` among schedule-success users is the directly observed extra path-interference term in this targeted probe.
- `wrong_path_users` counts preemption by a parity-consistent path whose decoded message is not the tagged user's message.
- The probe is exact for the targeted users only when `aborted = 0`; otherwise it is conservative for PDP.

