#!/usr/bin/env python3
"""Machine-check the current LLC/UACE predictive-validation evidence.

The output is intentionally conservative.  A PASS means the referenced
finite-instance evidence supports that narrow claim.  A WARN marks evidence
that is useful but not theorem-level.  A GAP marks a still-open requirement for
the full user goal, such as phase-III full root-sweep validation without caps.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from uace_bound_explorer import format_probability
from uace_composite_predictor import composite_rows
from uace_php_audit import audit_rows
from uace_validation_summary import (
    DEFAULT_FILES,
    DEFAULT_FAST_FILES,
    DEFAULT_PRETRUE_FILES,
    DEFAULT_ROOT_PROFILE_FILES,
    DEFAULT_SCHEDULE_MC_FILES,
    DEFAULT_TARGETED_FILES,
    DEFAULT_TRUE_PATH_ORDER_FILES,
    aggregate_targeted,
    binomial_se,
    parse_overlay,
    parse_fast_full,
    parse_schedule_mc,
    parse_root_profile,
    parse_targeted,
    parse_true_path_order,
    parse_pretrue_preemption,
)


@dataclass(frozen=True)
class GateRow:
    status: str
    claim: str
    evidence: str
    detail: str


@dataclass(frozen=True)
class EmpiricalProbeRow:
    k: int
    pe: float
    trials: int
    pdp: float
    php: float
    schedule_fail: float


def status(ok: bool, warn: bool = False) -> str:
    if ok:
        return "PASS"
    if warn:
        return "WARN"
    return "GAP"


def parse_multicolor_summary(path: Path) -> tuple[int, float, float] | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"\| `research/uace_multicolor_chunk_aggregate\.md` \| 3 \| (\d+) \| ([0-9.eE+-]+) \| 40 \| 0\.100 \| ([0-9.eE+-]+) \| ([0-9.eE+-]+) \|",
        text,
    )
    if not match:
        return None
    return int(match.group(1)), float(match.group(2)), float(match.group(4))


def parse_empirical_probe(path: Path) -> EmpiricalProbeRow:
    text = path.read_text(encoding="utf-8")

    def scalar(pattern: str, label: str) -> str:
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"could not parse {label} from {path}")
        return match.group(1)

    k = int(scalar(r"- `K = ([0-9]+)`", "K"))
    pe = float(scalar(r"- `p_e = ([0-9.]+)`", "p_e"))
    trials = int(scalar(r"- trials: `([0-9]+)`", "trials"))
    pdp = float(scalar(r"\| pdp \| ([0-9.]+) \|", "pdp"))
    php = float(scalar(r"\| php \| ([0-9.]+) \|", "php"))
    schedule_fail = float(
        scalar(r"\| schedule_fail_emp \| ([0-9.]+) \|", "schedule_fail_emp")
    )
    return EmpiricalProbeRow(k=k, pe=pe, trials=trials, pdp=pdp, php=php, schedule_fail=schedule_fail)


def parse_k_sweep_summary(path: Path) -> tuple[int, float, float, bool] | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")

    def match_float(pattern: str) -> float:
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"could not parse {pattern} from {path}")
        return float(match.group(1))

    def match_int(pattern: str) -> int:
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"could not parse {pattern} from {path}")
        return int(match.group(1))

    rows = match_int(r"- rows: `([0-9]+)`")
    max_gap = match_float(r"- max abs\(PDP - sampled schedule fail\): `([0-9.eE+-]+)`")
    max_z = match_float(r"- max abs\(z vs closed-form theory\): `([0-9.eE+-]+)`")
    php_zero_match = re.search(r"- empirical PHP zero throughout: `(True|False)`", text)
    if not php_zero_match:
        raise ValueError(f"could not parse PHP-zero readout from {path}")
    return rows, max_gap, max_z, php_zero_match.group(1) == "True"


def build_rows(args: argparse.Namespace) -> list[GateRow]:
    rows: list[GateRow] = []

    phase1_rows = [
        parse_empirical_probe(Path(item))
        for item in args.phase1_files
        if Path(item).exists()
    ]
    expected_phase1 = {(30, 0.1), (30, 0.2), (30, 0.3), (40, 0.1), (40, 0.2), (40, 0.3)}
    found_phase1 = {(row.k, round(row.pe, 1)) for row in phase1_rows}
    phase1_complete = expected_phase1 <= found_phase1
    phase1_max_schedule_gap = max(
        (abs(row.pdp - row.schedule_fail) for row in phase1_rows),
        default=float("inf"),
    )
    phase1_max_abs_z = 0.0
    for row in phase1_rows:
        users = row.k * row.trials
        p_sch = 1.0 - ((1.0 - row.pe) ** args.L)
        se = binomial_se(p_sch, users)
        if se > 0:
            phase1_max_abs_z = max(phase1_max_abs_z, abs((row.pdp - p_sch) / se))
    phase1_php_zero = all(row.php == 0 for row in phase1_rows)
    rows.append(
        GateRow(
            status=status(
                phase1_complete
                and phase1_max_schedule_gap < 1e-12
                and phase1_max_abs_z <= args.z_gate
                and phase1_php_zero
            ),
            claim="Phase-I full decoder is zero-erasure schedule-predictive for K=30/K=40",
            evidence=f"{len(phase1_rows)} full playground decoder rows",
            detail=(
                f"coverage={len(found_phase1)}/6, max abs(PDP-sampled schedule)={phase1_max_schedule_gap:.3e}, "
                f"max abs(z)={phase1_max_abs_z:.2f}, empirical PHP zero={phase1_php_zero}"
            ),
        )
    )

    k_sweep = parse_k_sweep_summary(Path(args.phase12_k_sweep_file))
    if k_sweep is None:
        rows.append(
            GateRow(
                status="GAP",
                claim="Phase-I/II schedule predictivity generalizes across tested K",
                evidence=args.phase12_k_sweep_file,
                detail="summary not found",
            )
        )
    else:
        sweep_rows, sweep_gap, sweep_z, sweep_php_zero = k_sweep
        rows.append(
            GateRow(
                status=status(
                    sweep_rows >= 11
                    and sweep_gap < 1e-12
                    and sweep_z <= args.z_gate
                    and sweep_php_zero
                ),
                claim="Phase-I/II schedule predictivity generalizes across tested K",
                evidence=args.phase12_k_sweep_file,
                detail=(
                    f"rows={sweep_rows}, max abs(PDP-sampled schedule)={sweep_gap:.3e}, "
                    f"max abs(z)={sweep_z:.2f}, empirical PHP zero={sweep_php_zero}"
                ),
            )
        )

    full_rows = [
        parse_overlay(Path(item))
        for item in args.full_decoder_files
        if Path(item).exists()
    ]
    expected_full = {(30, 0.1), (30, 0.2), (30, 0.3), (40, 0.1), (40, 0.2), (40, 0.3)}
    found_full = {(row.k, round(row.pe, 1)) for row in full_rows}
    phase2_complete = expected_full <= found_full
    max_schedule_gap = max(
        (abs(row.empirical_pdp - row.empirical_schedule_fail) for row in full_rows),
        default=float("inf"),
    )
    max_abs_z = 0.0
    for row in full_rows:
        users = row.k * row.trials
        se = binomial_se(row.schedule_ue, users)
        if se > 0:
            max_abs_z = max(max_abs_z, abs((row.empirical_pdp - row.schedule_ue) / se))
    php_zero = all(row.empirical_php == 0 for row in full_rows)
    rows.append(
        GateRow(
            status=status(phase2_complete and max_schedule_gap < 1e-12 and max_abs_z <= args.z_gate and php_zero),
            claim="Phase-II full decoder is schedule-predictive for K=30/K=40",
            evidence=f"{len(full_rows)} full playground decoder rows",
            detail=(
                f"coverage={len(found_full)}/6, max abs(PDP-sampled schedule)={max_schedule_gap:.3e}, "
                f"max abs(z)={max_abs_z:.2f}, empirical PHP zero={php_zero}"
            ),
        )
    )

    schedule_rows = [
        row
        for path in args.schedule_mc_files
        if Path(path).exists()
        for row in parse_schedule_mc(Path(path))
    ]
    expected_schedule = expected_full
    found_schedule = {(row.k, round(row.pe, 1)) for row in schedule_rows}
    max_schedule_z = max((abs(row.z_score) for row in schedule_rows), default=float("inf"))
    max_schedule_hw = max((row.half_width_95 for row in schedule_rows), default=float("inf"))
    rows.append(
        GateRow(
            status=status(expected_schedule <= found_schedule and max_schedule_z <= args.z_gate),
            claim="Phase-III dominant schedule term is empirically resolved",
            evidence=f"{sum(row.users for row in schedule_rows)} erasure-mask samples",
            detail=f"coverage={len(found_schedule)}/6, max abs(z)={max_schedule_z:.2f}, max 95% half-width={format_probability(max_schedule_hw)}",
        )
    )

    targeted_rows = [
        row
        for path in args.targeted_files
        if Path(path).exists()
        for row in parse_targeted(Path(path))
    ]
    targeted = aggregate_targeted(targeted_rows)
    wrong_paths = sum(row.wrong_path_users for row in targeted)
    targeted_php = sum(row.targeted_php_upper for row in targeted)
    aborted = sum(row.aborted_users for row in targeted)
    completed = sum(row.checked_users - row.aborted_users for row in targeted)
    rows.append(
        GateRow(
            status=status(wrong_paths == 0 and targeted_php == 0 and aborted == 0, warn=(wrong_paths == 0 and targeted_php == 0)),
            claim="Phase-III schedule-success users show no visible path preemption",
            evidence=f"{completed} completed targeted schedule-success checks",
            detail=f"wrong paths={wrong_paths}, targeted PHP={format_probability(targeted_php)}, aborted={aborted}",
        )
    )

    audit = audit_rows(args)
    composite_needed = max((row.composite_needed for row in audit), default=0)
    pair_dominates = all(row.pair_preempt >= row.hallucination_php for row in audit)
    rows.append(
        GateRow(
            status=status(pair_dominates and composite_needed > completed, warn=True),
            claim="Phase-III PHP is analytically controlled but below direct-MC resolution",
            evidence="pair-preemption exact enumeration plus hallucination first moment",
            detail=f"pair dominates hallucination={pair_dominates}, max checks needed={composite_needed}, completed targeted checks={completed}",
        )
    )

    fast_rows = [
        parse_fast_full(Path(item))
        for item in args.fast_files
        if Path(item).exists()
    ]
    if fast_rows:
        false_positives = sum(row.false_positive for row in fast_rows)
        php_visible = any(row.php > 0 for row in fast_rows)
        exact_zero_abort = [row for row in fast_rows if row.aborted_roots == 0]
        capped_abort = [row for row in fast_rows if row.aborted_roots > 0]
        capped_by_k: dict[int, int] = {}
        max_cap_by_k: dict[int, int] = {}
        for row in fast_rows:
            max_cap_by_k[row.k] = max(max_cap_by_k.get(row.k, 0), row.max_nodes_per_root)
            if row.aborted_roots > 0:
                capped_by_k[row.k] = capped_by_k.get(row.k, 0) + 1
        capped_detail = ", ".join(f"{k}:{v}" for k, v in sorted(capped_by_k.items()))
        cap_detail = ", ".join(f"{k}:{v}" for k, v in sorted(max_cap_by_k.items()))
        rows.append(
            GateRow(
                status=status(false_positives == 0 and not php_visible and not capped_abort, warn=(false_positives == 0 and not php_visible)),
                claim="Phase-III full-wrapper cap sweep shows no visible PHP",
                evidence=f"{len(fast_rows)} fast-wrapper rows",
                detail=(
                    f"zero-abort sanity rows={len(exact_zero_abort)}, cap-limited rows by K={capped_detail}, "
                    f"false positives={false_positives}, max cap/root by K={cap_detail}"
                ),
            )
        )

    root_profile_rows = [
        row
        for item in args.root_profile_files
        if Path(item).exists()
        for row in parse_root_profile(Path(item))
    ]
    if root_profile_rows:
        total_false_paths = sum(row.false_paths for row in root_profile_rows)
        by_attempt: dict[str, int] = {}
        for row in root_profile_rows:
            by_attempt[row.attempt] = by_attempt.get(row.attempt, 0) + row.aborted_roots
        dominant_attempt, dominant_aborts = max(by_attempt.items(), key=lambda item: item[1])
        max_cap = max(row.max_nodes_per_root for row in root_profile_rows)
        max_cap_rows = [row for row in root_profile_rows if row.max_nodes_per_root == max_cap]
        unresolved_sched: dict[str, int] = {}
        for row in max_cap_rows:
            unresolved_sched[row.attempt] = unresolved_sched.get(row.attempt, 0) + max(row.attempt_sched_users - row.true_paths, 0)
        dominant_unresolved_attempt, dominant_unresolved = max(
            unresolved_sched.items(),
            key=lambda item: item[1],
        )
        rows.append(
            GateRow(
                status=status(total_false_paths == 0 and dominant_aborts == 0, warn=(total_false_paths == 0)),
                claim="Phase-III root-search runtime tail is localized",
                evidence=f"{len(root_profile_rows)} attempt-profile rows",
                detail=(
                    f"false paths={total_false_paths}, dominant aborted attempt={dominant_attempt} ({dominant_aborts} aborts), "
                    f"at max cap unresolved attempt-schedule users dominated by {dominant_unresolved_attempt} ({dominant_unresolved})"
                ),
            )
        )

    true_path_order_rows = [
        row
        for item in args.true_path_order_files
        if Path(item).exists()
        for row in parse_true_path_order(Path(item))
    ]
    if true_path_order_rows:
        total_users = sum(row.users for row in true_path_order_rows)
        total_final_valid = sum(row.final_valid for row in true_path_order_rows)
        max_children = max(row.max_children for row in true_path_order_rows)
        max_mean_prior = max(row.mean_prior_siblings for row in true_path_order_rows)
        max_log10_work = max(row.max_log10_prefix_work for row in true_path_order_rows)
        covered_k = sorted({row.k for row in true_path_order_rows})
        true_paths_complete = total_users == total_final_valid and len(covered_k) >= 2
        rows.append(
            GateRow(
                status=("WARN" if true_paths_complete else "GAP"),
                claim="Phase-III true paths exist but are delayed by child ordering",
                evidence=f"{len(true_path_order_rows)} true-path ordering rows",
                detail=(
                    f"K covered={covered_k}, final-valid true paths={total_final_valid}/{total_users}, "
                    f"max children={max_children}, max mean prior siblings={max_mean_prior:.2f}, max log10 prefix work={max_log10_work:.2f}"
                ),
            )
        )

    pretrue_rows = [
        row
        for item in args.pretrue_files
        if Path(item).exists()
        for row in parse_pretrue_preemption(Path(item))
    ]
    if pretrue_rows:
        total_checked = sum(row.checked_users for row in pretrue_rows)
        total_completed = sum(row.completed_users for row in pretrue_rows)
        total_wrong = sum(row.wrong_preemptions for row in pretrue_rows)
        total_abort = sum(row.aborted_users for row in pretrue_rows)
        covered = sorted({(row.k, round(row.pe, 1)) for row in pretrue_rows})
        rows.append(
            GateRow(
                status=("WARN" if total_wrong == 0 and total_completed > 0 else "GAP"),
                claim="Pre-true-path verifier finds no wrong preemption before true paths",
                evidence=f"{len(pretrue_rows)} pre-true verifier rows",
                detail=(
                    f"coverage={covered}, checked={total_checked}, completed={total_completed}, "
                    f"wrong preemptions={total_wrong}, aborted={total_abort}"
                ),
            )
        )

    multicolor = parse_multicolor_summary(Path(args.multicolor_summary))
    if multicolor is None:
        rows.append(
            GateRow(
                status="GAP",
                claim="Multi-color path diagnostics are reproducible",
                evidence=args.multicolor_summary,
                detail="could not parse aggregate row",
            )
        )
    else:
        included, coverage, ratio = multicolor
        rows.append(
            GateRow(
                status=("WARN" if included >= 17 and ratio < 1e-2 and coverage < 1.0 else status(included >= 17 and ratio < 1e-2)),
                claim="Three-color finite-window contribution is not visible at K=40, pe=0.1",
                evidence=args.multicolor_summary,
                detail=f"included windows={included}, max coverage={coverage:.6f}, ratio to two-color={ratio:.6f}",
            )
        )

    full_phase3_file = Path(args.full_phase3_file)
    runtime_note = Path(args.runtime_note)
    rows.append(
        GateRow(
            status="GAP",
            claim="Full phase-III root-sweep decoder validation without caps",
            evidence=str(full_phase3_file if full_phase3_file.exists() else runtime_note),
            detail=(
                "not completed; current evidence uses exact finite enumeration, "
                "large-sample schedule MC, targeted probes, and runtime-tail note"
            ),
        )
    )

    return rows


def build_report(args: argparse.Namespace) -> str:
    rows = build_rows(args)
    lines = [
        "# UACE Predictive Validation Gate",
        "",
        "This report is generated by `research/uace_predictive_validation_gate.py`.",
        "",
        "| status | claim | evidence | detail |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row.status} | {row.claim} | {row.evidence} | {row.detail} |")

    passes = sum(row.status == "PASS" for row in rows)
    warnings = sum(row.status == "WARN" for row in rows)
    gaps = sum(row.status == "GAP" for row in rows)
    lines.extend(
        [
            "",
            "## Readout",
            "",
            f"- PASS rows: `{passes}`",
            f"- WARN rows: `{warnings}`",
            f"- GAP rows: `{gaps}`",
            "",
            "A WARN row is usable evidence with an explicit limitation.  A GAP row is not a failure of the current bound; it is a remaining requirement before claiming a complete phase-III theorem for the implemented full root-sweep decoder.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--M", type=int, default=3)
    parser.add_argument("--J", type=int, default=16)
    parser.add_argument("--R", type=int, default=8)
    parser.add_argument("--phase", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--Ks", type=int, nargs="+", default=[30, 40])
    parser.add_argument("--pes", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    parser.add_argument("--pair-file", type=Path, default=Path("research/uace_pair_preemption_exact.md"))
    parser.add_argument("--phase12-k-sweep-file", default="research/uace_phase12_k_sweep_validation.md")
    parser.add_argument(
        "--phase1-files",
        nargs="+",
        default=[
            "research/uace_empirical_phase1_K30_pe010_trials5.md",
            "research/uace_empirical_phase1_K30_pe020_trials3.md",
            "research/uace_empirical_phase1_K30_pe030_trials3.md",
            "research/uace_empirical_phase1_K40_pe010_trials3.md",
            "research/uace_empirical_phase1_K40_pe020_trials3.md",
            "research/uace_empirical_phase1_K40_pe030_trials3.md",
        ],
    )
    parser.add_argument("--full-decoder-files", nargs="+", default=list(DEFAULT_FILES))
    parser.add_argument("--schedule-mc-files", nargs="+", default=list(DEFAULT_SCHEDULE_MC_FILES))
    parser.add_argument("--targeted-files", nargs="+", default=list(DEFAULT_TARGETED_FILES))
    parser.add_argument("--fast-files", nargs="+", default=list(DEFAULT_FAST_FILES))
    parser.add_argument("--root-profile-files", nargs="+", default=list(DEFAULT_ROOT_PROFILE_FILES))
    parser.add_argument("--true-path-order-files", nargs="+", default=list(DEFAULT_TRUE_PATH_ORDER_FILES))
    parser.add_argument("--pretrue-files", nargs="+", default=list(DEFAULT_PRETRUE_FILES))
    parser.add_argument("--multicolor-summary", default="research/uace_multicolor_summary.md")
    parser.add_argument("--full-phase3-file", default="research/uace_fast_probe_K30_phase3_pe010_seed6310_cap5m.md")
    parser.add_argument("--runtime-note", default="research/uace_fast_full_runtime_note.md")
    parser.add_argument("--z-gate", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=Path("research/uace_predictive_validation_gate.md"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    content = build_report(args)
    if args.output:
        args.output.write_text(content + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
