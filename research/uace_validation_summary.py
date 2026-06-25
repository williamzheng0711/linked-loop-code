#!/usr/bin/env python3
"""Consolidate K=30/K=40 LLC/UACE validation runs."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

from uace_bound_explorer import format_probability
from uace_composite_predictor import composite_rows
from uace_interference_bound import interference_moments


DEFAULT_FILES = (
    "research/uace_trend_overlay_K30_phase2_pe010_trials5.md",
    "research/uace_trend_overlay_K30_phase2_pe020_trials3.md",
    "research/uace_trend_overlay_K30_phase2_pe030_trials3.md",
    "research/uace_trend_overlay_K40_phase2_pe010_trials3.md",
    "research/uace_trend_overlay_K40_phase2_pe020_trials3.md",
    "research/uace_trend_overlay_K40_phase2_pe030_trials3.md",
)

DEFAULT_TARGETED_FILES = (
    "research/uace_targeted_probe_K30_phase3_pe010_trials2.md",
    "research/uace_targeted_probe_K30_phase3_pe020_trials2.md",
    "research/uace_targeted_probe_K30_phase3_pe030_trials5.md",
    "research/uace_targeted_probe_K40_phase3_pe010_trial1_cap5m.md",
    "research/uace_targeted_probe_K40_phase3_pe010_seed5411_cap5m.md",
    "research/uace_targeted_probe_K40_phase3_pe020_trials2.md",
    "research/uace_targeted_probe_K40_phase3_pe030_trials5.md",
)

DEFAULT_PAIR_FILE = "research/uace_pair_preemption_exact.md"

DEFAULT_FAST_FILES = (
    "research/uace_fast_probe_K6_phase3_pe010_seed0.md",
    "research/uace_fast_probe_K30_phase3_pe010_seed6310_cap10k.md",
    "research/uace_fast_probe_K30_phase3_pe010_seed6310_cap50k.md",
    "research/uace_fast_probe_K30_phase3_pe010_seed6310_cap100k.md",
    "research/uace_fast_probe_K40_phase3_pe010_seed6310_cap10k.md",
    "research/uace_fast_probe_K40_phase3_pe010_seed6310_cap50k.md",
)

DEFAULT_ROOT_PROFILE_FILES = (
    "research/uace_phase3_root_profile_K30_pe010_seed6310_cap10k.md",
    "research/uace_phase3_root_profile_K30_pe010_seed6310_cap50k.md",
    "research/uace_phase3_root_profile_K40_pe010_seed6310_cap10k.md",
    "research/uace_phase3_root_profile_K40_pe010_seed6310_cap50k.md",
)

DEFAULT_TRUE_PATH_ORDER_FILES = (
    "research/uace_true_path_order_profile_K30_pe010_seed6310.md",
    "research/uace_true_path_order_profile_K30_pe020_seed6310.md",
    "research/uace_true_path_order_profile_K30_pe030_seed6310.md",
    "research/uace_true_path_order_profile_K40_pe010_seed6310.md",
    "research/uace_true_path_order_profile_K40_pe020_seed6310.md",
    "research/uace_true_path_order_profile_K40_pe030_seed6310.md",
)

DEFAULT_PRETRUE_FILES = (
    "research/uace_pretrue_preemption_probe_K30_pe010_seed6310_cap2m.md",
    "research/uace_pretrue_preemption_probe_K30_pe020_seed6310_cap500k.md",
    "research/uace_pretrue_preemption_probe_K30_pe030_seed6310_cap500k.md",
    "research/uace_pretrue_preemption_probe_K40_pe010_seed6310_merged_cap5m.md",
    "research/uace_pretrue_preemption_probe_K40_pe020_seed6310_cap2m.md",
    "research/uace_pretrue_preemption_probe_K40_pe030_seed6310_cap2m.md",
)

DEFAULT_HALLUCINATION_FILES = (
    "research/uace_hallucination_probe_K30_phase3_pe010_trial1_roots3_cap1m.md",
    "research/uace_hallucination_probe_K30_phase3_pe020_trial1_roots3_cap1m.md",
    "research/uace_hallucination_probe_K30_phase3_pe030_trials2_roots5_cap50k.md",
    "research/uace_hallucination_probe_K40_phase3_pe010_trial1_roots3_cap1m.md",
    "research/uace_hallucination_probe_K40_phase3_pe020_trial1_roots3_cap1m.md",
    "research/uace_hallucination_probe_K40_phase3_pe030_trials2_roots5_cap50k.md",
)

DEFAULT_SCHEDULE_MC_FILES = (
    "research/uace_schedule_monte_carlo.md",
)

DEFAULT_PAIR_VALIDATION_FILE = "research/uace_pair_validation_summary.md"
DEFAULT_PHP_AUDIT_FILE = "research/uace_php_audit.md"


@dataclass(frozen=True)
class EmpiricalRow:
    source: Path
    k: int
    phase: int
    trials: int
    pe: float
    schedule_ue: float
    rank_ue: float
    empirical_pdp: float
    empirical_php: float
    empirical_schedule_fail: float
    seconds: float


@dataclass(frozen=True)
class TargetedTrialRow:
    source: Path
    k: int
    phase: int
    pe: float
    schedule_ue: float
    seed: int
    schedule_fail: float
    schedule_success_users: int
    checked_users: int
    path_fail_users: int
    wrong_path_users: int
    aborted_users: int
    targeted_php: float
    seconds: float


@dataclass(frozen=True)
class TargetedAggregate:
    k: int
    phase: int
    pe: float
    schedule_ue: float
    trials: int
    users: int
    schedule_fail_emp: float
    checked_users: int
    path_fail_users: int
    wrong_path_users: int
    aborted_users: int
    targeted_php_upper: float
    seconds: float
    sources: tuple[Path, ...]


@dataclass(frozen=True)
class FastFullRow:
    source: Path
    k: int
    phase: int
    trials: int
    pe: float
    max_nodes_per_root: int
    pdp: float
    php: float
    decoded: int
    correct: int
    schedule_fail_emp: float
    rank_fail_emp: float
    false_positive: int
    aborted_roots: int
    node_visits: int
    seconds: float


@dataclass(frozen=True)
class RootProfileRow:
    source: Path
    k: int
    phase: int
    pe: float
    seed: int
    max_nodes_per_root: int
    schedule_fail_emp: float
    rank_fail_emp: float
    attempt: str
    chosen_root: int
    d: int
    effective_roots: int
    roots_with_users: int
    candidate_users: int
    attempt_sched_users: int
    phase_sched_users: int
    found_paths: int
    true_paths: int
    false_paths: int
    aborted_roots: int
    node_visits: int
    max_root_visits: int
    seconds: float


@dataclass(frozen=True)
class TruePathOrderRow:
    source: Path
    k: int
    l: int
    m: int
    pe: float
    seed: int
    attempt: str
    users: int
    final_valid: int
    max_children: int
    max_true_index: int
    mean_prior_siblings: float
    max_prior_siblings: int
    max_log10_prefix_work: float


@dataclass(frozen=True)
class PretruePreemptionRow:
    source: Path
    k: int
    phase: int
    pe: float
    max_nodes_per_user: int
    seed: int
    schedule_fail: float
    schedule_success_users: int
    phase_i_users: int
    checked_users: int
    completed_users: int
    wrong_preemptions: int
    early_correct: int
    true_missing: int
    true_invalid: int
    aborted_users: int
    node_visits: int
    seconds: float


@dataclass(frozen=True)
class HallucinationRow:
    source: Path
    k: int
    phase: int
    trials: int
    pe: float
    roots_per_attempt: int
    sampled_roots: int
    valid_outputs: int
    true_outputs: int
    hallucinations: int
    aborted_roots: int
    php_bound: float


@dataclass(frozen=True)
class ScheduleMcRow:
    source: Path
    k: int
    phase: int
    pe: float
    users: int
    exact: float
    empirical: float
    z_score: float
    half_width_95: float
    failures: int


@dataclass(frozen=True)
class PhpAuditSummaryRow:
    k: int
    pe: float
    pair_preempt: float
    pure_hallucination: float
    php_bound: float
    pair_over_hallucination: float
    dominant_component: str
    composite_checks: int


def extract_scalar(pattern: str, text: str, name: str) -> int:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"missing {name}")
    return int(match.group(1))


def extract_float(pattern: str, text: str, name: str) -> float:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"missing {name}")
    return float(match.group(1))


def parse_overlay(path: Path) -> EmpiricalRow:
    text = path.read_text(encoding="utf-8")
    k = extract_scalar(r"`K = (\d+)`", text, "K")
    phase = extract_scalar(r"decoder phase: `(\d+)`", text, "phase")
    trials = extract_scalar(r"empirical trials per `p_e`: `(\d+)`", text, "trials")

    rows = [line for line in text.splitlines() if line.startswith("| ") and not line.startswith("| pe ")]
    data_rows = [line for line in rows if re.match(r"\| [0-9.]+ \|", line)]
    if len(data_rows) != 1:
        raise ValueError(f"expected one data row in {path}, found {len(data_rows)}")
    parts = [item.strip() for item in data_rows[0].strip("|").split("|")]
    return EmpiricalRow(
        source=path,
        k=k,
        phase=phase,
        trials=trials,
        pe=float(parts[0]),
        schedule_ue=float(parts[2]),
        rank_ue=float(parts[4]),
        empirical_pdp=float(parts[5]),
        empirical_php=float(parts[6]),
        empirical_schedule_fail=float(parts[8]),
        seconds=float(parts[10]),
    )


def parse_targeted(path: Path) -> list[TargetedTrialRow]:
    text = path.read_text(encoding="utf-8")
    k = extract_scalar(r"`K = (\d+)`", text, "K")
    phase = extract_scalar(r"`phase = (\d+)`", text, "phase")
    pe = extract_float(r"`pe = ([0-9.]+)`", text, "pe")
    schedule_ue = extract_float(r"schedule UE: `([0-9.eE+-]+)`", text, "schedule UE")

    rows: list[TargetedTrialRow] = []
    for line in text.splitlines():
        if not re.match(r"\| \d+ \|", line):
            continue
        parts = [item.strip() for item in line.strip("|").split("|")]
        if len(parts) != 10:
            continue
        rows.append(
            TargetedTrialRow(
                source=path,
                k=k,
                phase=phase,
                pe=pe,
                schedule_ue=schedule_ue,
                seed=int(parts[0]),
                schedule_fail=float(parts[1]),
                schedule_success_users=int(parts[2]),
                checked_users=int(parts[3]),
                path_fail_users=int(parts[4]),
                wrong_path_users=int(parts[5]),
                aborted_users=int(parts[6]),
                targeted_php=float(parts[7]),
                seconds=float(parts[9]),
            )
        )
    if not rows:
        raise ValueError(f"no targeted rows parsed from {path}")
    return rows


def aggregate_targeted(rows: list[TargetedTrialRow]) -> list[TargetedAggregate]:
    grouped: dict[tuple[int, int, float], list[TargetedTrialRow]] = {}
    for row in rows:
        grouped.setdefault((row.k, row.phase, row.pe), []).append(row)

    out: list[TargetedAggregate] = []
    for (k, phase, pe), bucket in sorted(grouped.items()):
        users = k * len(bucket)
        schedule_fail_users = sum(int(round(item.schedule_fail * k)) for item in bucket)
        false_messages_upper = sum(int(round(item.targeted_php * k)) for item in bucket)
        checked = sum(item.checked_users for item in bucket)
        out.append(
            TargetedAggregate(
                k=k,
                phase=phase,
                pe=pe,
                schedule_ue=bucket[0].schedule_ue,
                trials=len(bucket),
                users=users,
                schedule_fail_emp=schedule_fail_users / users if users else float("nan"),
                checked_users=checked,
                path_fail_users=sum(item.path_fail_users for item in bucket),
                wrong_path_users=sum(item.wrong_path_users for item in bucket),
                aborted_users=sum(item.aborted_users for item in bucket),
                targeted_php_upper=false_messages_upper / users if users else float("nan"),
                seconds=sum(item.seconds for item in bucket),
                sources=tuple(sorted({item.source for item in bucket})),
            )
        )
    return out


def parse_pair_extras(path: Path) -> dict[tuple[int, float], float]:
    text = path.read_text(encoding="utf-8")
    extras: dict[tuple[int, float], float] = {}
    for line in text.splitlines():
        if not re.match(r"\| \d+ \| [0-9.]+ \|", line):
            continue
        parts = [item.strip() for item in line.strip("|").split("|")]
        if len(parts) == 5:
            extras[(int(parts[0]), float(parts[1]))] = float(parts[4])
    return extras


def parse_fast_full(path: Path) -> FastFullRow:
    text = path.read_text(encoding="utf-8")
    k = extract_scalar(r"`K = (\d+)`", text, "K")
    phase = extract_scalar(r"`phase = (\d+)`", text, "phase")
    trials = extract_scalar(r"trials: `(\d+)`", text, "trials")
    pe = extract_float(r"`pe = ([0-9.]+)`", text, "pe")
    max_nodes_per_root = extract_scalar(r"max nodes per root: `(\d+)`", text, "max nodes per root")

    data_rows = [
        line
        for line in text.splitlines()
        if re.match(r"\| \d+ \| [0-9.]+ \|", line)
    ]
    if len(data_rows) != trials:
        raise ValueError(f"expected {trials} fast rows in {path}, found {len(data_rows)}")
    pdp = php = schedule_fail = rank_fail = seconds = 0.0
    false_positive = 0
    decoded = 0
    correct = 0
    aborted_roots = 0
    node_visits = 0
    for line in data_rows:
        parts = [item.strip() for item in line.strip("|").split("|")]
        pdp += float(parts[1])
        php += float(parts[2])
        decoded += int(parts[3])
        correct += int(parts[4])
        false_positive += int(parts[5])
        schedule_fail += float(parts[6])
        rank_fail += float(parts[7])
        seconds += float(parts[8])
        node_visits += int(parts[9])
        aborted_roots += int(parts[10])
    return FastFullRow(
        source=path,
        k=k,
        phase=phase,
        trials=trials,
        pe=pe,
        max_nodes_per_root=max_nodes_per_root,
        pdp=pdp / trials,
        php=php / trials,
        decoded=decoded,
        correct=correct,
        schedule_fail_emp=schedule_fail / trials,
        rank_fail_emp=rank_fail / trials,
        false_positive=false_positive,
        aborted_roots=aborted_roots,
        node_visits=node_visits,
        seconds=seconds,
    )


def parse_root_profile(path: Path) -> list[RootProfileRow]:
    text = path.read_text(encoding="utf-8")
    k = extract_scalar(r"`K = (\d+)`", text, "K")
    phase = extract_scalar(r"phase: `(\d+)`", text, "phase")
    seed = extract_scalar(r"seed: `(\d+)`", text, "seed")
    max_nodes_per_root = extract_scalar(r"max nodes per root: `(\d+)`", text, "max nodes per root")
    pe = extract_float(r"`pe = ([0-9.]+)`", text, "pe")
    schedule_fail_emp = extract_float(r"sampled schedule fail: `([0-9.]+)`", text, "sampled schedule fail")
    rank_fail_emp = extract_float(r"sampled rank fail: `([0-9.]+)`", text, "sampled rank fail")
    rows: list[RootProfileRow] = []
    for line in text.splitlines():
        if not line.startswith("| phase-"):
            continue
        parts = [item.strip() for item in line.strip("|").split("|")]
        if len(parts) == 11:
            roots_with_users = candidate_users = attempt_sched_users = phase_sched_users = 0
            found_idx = 4
        elif len(parts) == 15:
            roots_with_users = int(parts[4])
            candidate_users = int(parts[5])
            attempt_sched_users = int(parts[6])
            phase_sched_users = int(parts[7])
            found_idx = 8
        else:
            continue
        rows.append(
            RootProfileRow(
                source=path,
                k=k,
                phase=phase,
                pe=pe,
                seed=seed,
                max_nodes_per_root=max_nodes_per_root,
                schedule_fail_emp=schedule_fail_emp,
                rank_fail_emp=rank_fail_emp,
                attempt=parts[0],
                chosen_root=int(parts[1]),
                d=int(parts[2]),
                effective_roots=int(parts[3]),
                roots_with_users=roots_with_users,
                candidate_users=candidate_users,
                attempt_sched_users=attempt_sched_users,
                phase_sched_users=phase_sched_users,
                found_paths=int(parts[found_idx]),
                true_paths=int(parts[found_idx + 1]),
                false_paths=int(parts[found_idx + 2]),
                aborted_roots=int(parts[found_idx + 3]),
                node_visits=int(parts[found_idx + 4]),
                max_root_visits=int(parts[found_idx + 5]),
                seconds=float(parts[found_idx + 6]),
            )
        )
    if not rows:
        raise ValueError(f"no root profile rows parsed from {path}")
    return rows


def parse_true_path_order(path: Path) -> list[TruePathOrderRow]:
    text = path.read_text(encoding="utf-8")
    k = extract_scalar(r"`K = (\d+)`", text, "K")
    l = extract_scalar(r"`L = (\d+)`", text, "L")
    m = extract_scalar(r"`M = (\d+)`", text, "M")
    seed = extract_scalar(r"seed: `(\d+)`", text, "seed")
    pe = extract_float(r"`pe = ([0-9.]+)`", text, "pe")

    rows: list[TruePathOrderRow] = []
    in_attempt_table = False
    for line in text.splitlines():
        if line.startswith("## Per-Attempt Summary"):
            in_attempt_table = True
            continue
        if in_attempt_table and line.startswith("## Per-Section Summary"):
            break
        if not in_attempt_table or not line.startswith("| phase-"):
            continue
        parts = [item.strip() for item in line.strip("|").split("|")]
        if len(parts) != 8:
            continue
        rows.append(
            TruePathOrderRow(
                source=path,
                k=k,
                l=l,
                m=m,
                pe=pe,
                seed=seed,
                attempt=parts[0],
                users=int(parts[1]),
                final_valid=int(parts[2]),
                max_children=int(parts[3]),
                max_true_index=int(parts[4]),
                mean_prior_siblings=float(parts[5]),
                max_prior_siblings=int(parts[6]),
                max_log10_prefix_work=float(parts[7]),
            )
        )
    if not rows:
        raise ValueError(f"no true-path ordering rows parsed from {path}")
    return rows


def parse_pretrue_preemption(path: Path) -> list[PretruePreemptionRow]:
    text = path.read_text(encoding="utf-8")
    k = extract_scalar(r"`K = (\d+)`", text, "K")
    phase = extract_scalar(r"`phase = (\d+)`", text, "phase")
    max_nodes_match = re.search(r"max nodes per checked user: `(\d+)`", text)
    max_nodes = int(max_nodes_match.group(1)) if max_nodes_match else -1
    pe = extract_float(r"`pe = ([0-9.]+)`", text, "pe")
    rows: list[PretruePreemptionRow] = []
    for line in text.splitlines():
        if not re.match(r"\| \d+ \| [0-9.]+ \|", line):
            continue
        parts = [item.strip() for item in line.strip("|").split("|")]
        if len(parts) != 13:
            continue
        rows.append(
            PretruePreemptionRow(
                source=path,
                k=k,
                phase=phase,
                pe=pe,
                max_nodes_per_user=max_nodes,
                seed=int(parts[0]),
                schedule_fail=float(parts[1]),
                schedule_success_users=int(parts[2]),
                phase_i_users=int(parts[3]),
                checked_users=int(parts[4]),
                completed_users=int(parts[5]),
                wrong_preemptions=int(parts[6]),
                early_correct=int(parts[7]),
                true_missing=int(parts[8]),
                true_invalid=int(parts[9]),
                aborted_users=int(parts[10]),
                node_visits=int(parts[11]),
                seconds=float(parts[12]),
            )
        )
    if not rows:
        raise ValueError(f"no pre-true preemption rows parsed from {path}")
    return rows


def parse_hallucination(path: Path) -> HallucinationRow:
    text = path.read_text(encoding="utf-8")
    true_outputs = 0
    for line in text.splitlines():
        if not re.match(r"\| \d+ \| [^|]+ \|", line):
            continue
        parts = [item.strip() for item in line.strip("|").split("|")]
        if len(parts) == 9:
            true_outputs += int(parts[4])

    return HallucinationRow(
        source=path,
        k=extract_scalar(r"`K = (\d+)`", text, "K"),
        phase=extract_scalar(r"`phase = (\d+)`", text, "phase"),
        trials=extract_scalar(r"trials: `(\d+)`", text, "trials"),
        pe=extract_float(r"`pe = ([0-9.]+)`", text, "pe"),
        roots_per_attempt=extract_scalar(r"roots per attempt: `(\d+)`", text, "roots per attempt"),
        sampled_roots=extract_scalar(r"sampled roots: `(\d+)`", text, "sampled roots"),
        valid_outputs=extract_scalar(r"valid outputs found: `(\d+)`", text, "valid outputs"),
        true_outputs=true_outputs,
        hallucinations=extract_scalar(r"hallucinations found: `(\d+)`", text, "hallucinations"),
        aborted_roots=extract_scalar(r"aborted roots: `(\d+)`", text, "aborted roots"),
        php_bound=extract_float(r"first-moment PHP bound: `([0-9.eE+-]+)`", text, "PHP bound"),
    )


def parse_schedule_mc(path: Path) -> list[ScheduleMcRow]:
    text = path.read_text(encoding="utf-8")
    phase = extract_scalar(r"phase: `(\d+)`", text, "phase")
    rows: list[ScheduleMcRow] = []
    for line in text.splitlines():
        if not re.match(r"\| \d+ \| [0-9.]+ \|", line):
            continue
        parts = [item.strip() for item in line.strip("|").split("|")]
        if len(parts) != 8:
            continue
        rows.append(
            ScheduleMcRow(
                source=path,
                k=int(parts[0]),
                phase=phase,
                pe=float(parts[1]),
                users=int(parts[2]),
                exact=float(parts[3]),
                empirical=float(parts[4]),
                z_score=float(parts[5]),
                half_width_95=float(parts[6]),
                failures=int(parts[7]),
            )
        )
    if not rows:
        raise ValueError(f"no schedule Monte Carlo rows parsed from {path}")
    return rows


def parse_php_audit(path: Path) -> list[PhpAuditSummaryRow]:
    text = path.read_text(encoding="utf-8")
    rows: list[PhpAuditSummaryRow] = []
    in_component_table = False
    for line in text.splitlines():
        if line.startswith("## Component Table"):
            in_component_table = True
            continue
        if line.startswith("## Zero-Event"):
            break
        if not in_component_table or not re.match(r"\| \d+ \| [0-9.]+ \|", line):
            continue
        parts = [item.strip() for item in line.strip("|").split("|")]
        if len(parts) != 9:
            continue
        rows.append(
            PhpAuditSummaryRow(
                k=int(parts[0]),
                pe=float(parts[1]),
                pair_preempt=float(parts[2]),
                pure_hallucination=float(parts[3]),
                php_bound=float(parts[4]),
                pair_over_hallucination=float(parts[5]),
                dominant_component=parts[6],
                composite_checks=int(parts[8]),
            )
        )
    return rows


def binomial_se(p: float, n: int) -> float:
    if n <= 0:
        return float("nan")
    return math.sqrt(max(0.0, p * (1.0 - p)) / n)


def zero_event_upper(n: int, alpha: float = 0.05) -> float:
    """One-sided Clopper-Pearson upper bound after zero events in n trials."""
    if n <= 0:
        return float("nan")
    return 1.0 - alpha ** (1.0 / n)


def zero_event_required_trials(rate: float, alpha: float = 0.05) -> int | None:
    """Trials needed for a zero-event upper bound to fall below ``rate``."""
    if not (0.0 < rate < 1.0):
        return None
    return math.ceil(math.log(alpha) / math.log(1.0 - rate))


def interference_php_bound(row: EmpiricalRow, args: argparse.Namespace) -> float:
    moments = interference_moments(
        k=row.k,
        length=args.L,
        j=args.J,
        memory=args.M,
        phase=row.phase,
        pe=row.pe,
        parity_bits_per_section=args.R,
    )
    return min(1.0, sum(item.expected_false_survivors for item in moments) / row.k)


def build_report(args: argparse.Namespace) -> str:
    rows = [parse_overlay(Path(item)) for item in args.files]
    targeted_paths = [Path(item) for item in args.targeted_files if Path(item).exists()]
    targeted_rows = [row for path in targeted_paths for row in parse_targeted(path)]
    targeted = aggregate_targeted(targeted_rows) if targeted_rows else []
    pair_extras = parse_pair_extras(Path(args.pair_file)) if Path(args.pair_file).exists() else {}
    fast_paths = [Path(item) for item in args.fast_files if Path(item).exists()]
    fast_rows = [parse_fast_full(path) for path in fast_paths]
    root_profile_paths = [Path(item) for item in args.root_profile_files if Path(item).exists()]
    root_profile_rows = [row for path in root_profile_paths for row in parse_root_profile(path)]
    true_path_order_paths = [Path(item) for item in args.true_path_order_files if Path(item).exists()]
    true_path_order_rows = [row for path in true_path_order_paths for row in parse_true_path_order(path)]
    pretrue_paths = [Path(item) for item in args.pretrue_files if Path(item).exists()]
    pretrue_rows = [row for path in pretrue_paths for row in parse_pretrue_preemption(path)]
    hallucination_paths = [Path(item) for item in args.hallucination_files if Path(item).exists()]
    hallucination_rows = [parse_hallucination(path) for path in hallucination_paths]
    schedule_mc_paths = [Path(item) for item in args.schedule_mc_files if Path(item).exists()]
    schedule_mc_rows = [row for path in schedule_mc_paths for row in parse_schedule_mc(path)]
    pair_validation_path = Path(args.pair_validation_file)
    php_audit_path = Path(args.php_audit_file)
    php_audit_rows = parse_php_audit(php_audit_path) if php_audit_path.exists() else []
    composite = composite_rows(args)
    lines = [
        "# K=30/K=40 UACE Prediction Validation",
        "",
        "This report is generated by `research/uace_validation_summary.py`.",
        "",
        "For a compact publish-facing certificate and dashboard figure, see `research/uace_publish_validation_certificate.md` and `research/figures/uace_publish_validation_dashboard.png`.",
        "",
        "Model under test:",
        "",
        "$$",
        "\\widehat{\\mathrm{PDP}}",
        "=",
        "P_{\\mathrm{schedule}} + \\frac{\\mathbb{E}N_{\\mathrm{false}}}{K},",
        "\\qquad",
        "\\widehat{\\mathrm{PHP}}",
        "\\le",
        "\\frac{\\mathbb{E}N_{\\mathrm{false}}}{K}.",
        "$$",
        "",
        "Here $P_{\\mathrm{schedule}}$ is the exact erasure-only current-decoder schedule term.  The second term is the K-dependent first-moment false-survivor bound from `research/uace_interference_bound.py`.",
        "",
        "| K | phase | pe | trials | users | PDP pred | empirical PDP | empirical schedule fail | z vs schedule | PHP bound | empirical PHP | seconds | source |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        users = row.k * row.trials
        php_bound = interference_php_bound(row, args)
        pred = min(1.0, row.schedule_ue + php_bound)
        se = binomial_se(row.schedule_ue, users)
        z = (row.empirical_pdp - row.schedule_ue) / se if se > 0 else 0.0
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.k),
                    str(row.phase),
                    f"{row.pe:.3f}",
                    str(row.trials),
                    str(users),
                    format_probability(pred),
                    format_probability(row.empirical_pdp),
                    format_probability(row.empirical_schedule_fail),
                    f"{z:.2f}",
                    format_probability(php_bound),
                    format_probability(row.empirical_php),
                    f"{row.seconds:.2f}",
                    f"`{row.source.name}`",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "Readout:",
            "",
            "- In every completed K=30/K=40 full-decoder run, empirical PDP equals the sampled schedule-failure rate.",
            "- Empirical PHP is zero in all completed runs, consistent with the very small first-moment PHP bounds.",
            "- Deviations from the closed-form schedule expectation are explained by finite user/sample counts; the `z vs schedule` column is measured in binomial standard errors using `K * trials` users.",
            "- These runs validate phase-II/no-SIC behavior.  Phase-III K=30/K=40 full-decoder validation is still limited by runtime and remains a separate requirement before claiming a final publish-ready theorem.",
            "",
        ]
    )

    if targeted:
        lines.extend(
            [
                "## Phase-III Composite Predictor",
                "",
                "This is the finite-instance predictor used for the targeted phase-III comparison.  PDP uses schedule plus exact pair-preemption; PHP is bounded by exact pair-preemption plus the rank-corrected first-moment hallucination term.",
                "",
                "| K | pe | schedule UE | exact pair-preempt | hallucination PHP bound | PDP prediction | PHP bound |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in composite:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.k),
                        f"{row.pe:.3f}",
                        format_probability(row.schedule),
                        format_probability(row.pair_preempt),
                        format_probability(row.hallucination_php),
                        format_probability(row.pdp_pred),
                        format_probability(row.php_bound),
                    ]
                )
                + " |"
            )
        lines.append("")

        if php_audit_rows:
            lines.extend(
                [
                    "## PHP Component Audit",
                    "",
                    f"Detailed PHP component accounting is generated in `{php_audit_path}`.",
                    "",
                    "| K | pe | pair-preempt | pure hallucination | composite PHP bound | pair / hallucination | dominant pure-hallucination component | zero-event checks for composite scale |",
                    "|---:|---:|---:|---:|---:|---:|---|---:|",
                ]
            )
            for row in php_audit_rows:
                if row.k not in args.Ks or row.pe not in args.pes:
                    continue
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(row.k),
                            f"{row.pe:.3f}",
                            format_probability(row.pair_preempt),
                            format_probability(row.pure_hallucination),
                            format_probability(row.php_bound),
                            format_probability(row.pair_over_hallucination),
                            row.dominant_component,
                            str(row.composite_checks),
                        ]
                    )
                    + " |"
                )
            lines.extend(
                [
                    "",
                    "Readout:",
                    "",
                    "- The composite PHP bound is dominated by exact pair-preemption in all K=30/K=40 phase-III rows.",
                    "- Pure hallucination is much smaller and is dominated by root 0/root 6 two-erasure accepted shapes with rank exponent 96.",
                    "- Direct PHP Monte Carlo would need tens of thousands of user-equivalent zero-event checks even for the composite scale, and far more for pure hallucination.",
                    "",
                ]
            )

        if schedule_mc_rows:
            lines.extend(
                [
                    "## Large-Sample Schedule-Only Check",
                    "",
                    "This high-resolution check samples only erasure masks and evaluates the exact current-decoder schedule automaton.  It is not a full decoder simulation, but it validates the dominant term in the K=30/K=40 phase-III PDP predictor.",
                    "",
                    "| K | pe | sampled users | exact schedule UE | empirical schedule UE | z vs exact | 95% half-width |",
                    "|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in sorted(schedule_mc_rows, key=lambda item: (item.k, item.pe)):
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(row.k),
                            f"{row.pe:.3f}",
                            str(row.users),
                            format_probability(row.exact),
                            format_probability(row.empirical),
                            f"{row.z_score:.2f}",
                            format_probability(row.half_width_95),
                        ]
                    )
                    + " |"
                )
            lines.extend(
                [
                    "",
                    "Readout:",
                    "",
                    "- The empirical erasure-schedule rates are within ordinary binomial fluctuation of the exact finite-mask expression.",
                    "- This is the part of PDP that ordinary Monte Carlo can resolve at feasible sample sizes; the analytic path/PHP corrections are much smaller.",
                    "",
                ]
            )

        lines.extend(
            [
                "## Phase-III Targeted K-User Validation",
                "",
                "For phase III, full root-sweep decoding is runtime-limited.  The targeted probe conditions on schedule-success users and checks the expensive part directly: whether the first parity-valid path from the user's true root is lost to a wrong multi-user path.  This validates the K-dependent path-interference correction on top of the K-independent schedule term.",
                "",
                "| K | pe | trials | users | schedule UE | empirical schedule fail | z vs schedule | exact pair extra | PDP prediction | checked | wrong path | aborted | targeted PHP | seconds |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in targeted:
            extra = pair_extras.get((row.k, row.pe), 0.0)
            pred = min(1.0, row.schedule_ue + extra)
            se = binomial_se(row.schedule_ue, row.users)
            z = (row.schedule_fail_emp - row.schedule_ue) / se if se > 0 else 0.0
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.k),
                        f"{row.pe:.3f}",
                        str(row.trials),
                        str(row.users),
                        format_probability(row.schedule_ue),
                        format_probability(row.schedule_fail_emp),
                        f"{z:.2f}",
                        format_probability(extra),
                        format_probability(pred),
                        str(row.checked_users),
                        str(row.wrong_path_users),
                        str(row.aborted_users),
                        format_probability(row.targeted_php_upper),
                        f"{row.seconds:.2f}",
                    ]
                )
                + " |"
            )

        lines.extend(
            [
                "",
                "Readout:",
                "",
                "- The schedule column is K-independent because it is the single tagged user's erasure-geometry failure probability.",
                "- The exact pair extra is K-dependent; it is parsed from the affine GF(2) pair-preemption enumeration in `research/uace_pair_preemption_exact.md`.",
                "- Zero wrong paths in the completed targeted checks is consistent with the predicted pair extra below `1e-4` for K=30/K=40.",
                "- Aborted rows are runtime-inconclusive for PDP, but they did not produce hallucinated/wrong decoded messages.",
                "",
                "Targeted sources:",
                "",
            ]
        )
        for path in targeted_paths:
            lines.append(f"- `{path}`")
        lines.append("")

        composite_by_key = {(row.k, row.pe): row for row in composite}
        lines.extend(
            [
                "## Targeted Zero-Event Resolution",
                "",
                "The table below compares the analytic pair-preemption scale with the finite targeted-simulation resolution.  `95% conditional upper` is the one-sided zero-event upper bound among completed schedule-success path searches.  `95% unconditional upper` multiplies this by the closed-form schedule-success probability, so it is comparable to the unconditional pair-preemption term.",
                "",
                "| K | pe | completed checks | wrong paths | exact pair-preempt | 95% conditional upper | 95% unconditional upper | theory / unconditional upper |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in targeted:
            completed = row.checked_users - row.aborted_users
            upper_cond = zero_event_upper(completed)
            composite_row = composite_by_key.get((row.k, row.pe))
            pair_preempt = composite_row.pair_preempt if composite_row else pair_extras.get((row.k, row.pe), 0.0)
            schedule = composite_row.schedule if composite_row else row.schedule_ue
            upper_uncond = (1.0 - schedule) * upper_cond if completed > 0 else float("nan")
            ratio = pair_preempt / upper_uncond if upper_uncond and upper_uncond > 0 else float("nan")
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.k),
                        f"{row.pe:.3f}",
                        str(completed),
                        str(row.wrong_path_users),
                        format_probability(pair_preempt),
                        format_probability(upper_cond),
                        format_probability(upper_uncond),
                        format_probability(ratio),
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "Readout:",
                "",
                "- The analytic pair-preemption terms are well below the zero-event resolution of the completed targeted simulations.",
                "- Therefore zero observed wrong paths is consistent with the theory but is not by itself a tight empirical upper bound.",
                "",
                "## Targeted Sample Requirements",
                "",
                "The pair-preemption term is unconditional.  Since the targeted probe only checks schedule-success users, the comparable conditional rate is",
                "",
                "$$",
                "r_{\\mathrm{cond}}",
                "=",
                "\\frac{P_{\\mathrm{pair\\text{-}preempt}}}{1-P_{\\mathrm{sch,III}}}.",
                "$$",
                "",
                "The required checks column is the number of completed schedule-success path checks needed for a zero-event 95% upper bound to fall below this conditional theory scale.",
                "",
                "| K | pe | conditional pair scale | completed checks | required checks | completed / required |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in targeted:
            completed = row.checked_users - row.aborted_users
            composite_row = composite_by_key.get((row.k, row.pe))
            pair_preempt = composite_row.pair_preempt if composite_row else pair_extras.get((row.k, row.pe), 0.0)
            schedule = composite_row.schedule if composite_row else row.schedule_ue
            cond_rate = pair_preempt / (1.0 - schedule) if schedule < 1.0 else float("nan")
            required = zero_event_required_trials(cond_rate)
            progress = completed / required if required else float("nan")
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.k),
                        f"{row.pe:.3f}",
                        format_probability(cond_rate),
                        str(completed),
                        str(required) if required is not None else "n/a",
                        format_probability(progress),
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "Readout:",
                "",
                "- Direct Monte Carlo would need thousands to tens of thousands of completed schedule-success checks at each point before a zero-event result has the same scale as the analytic pair-preemption term.",
                "- The current targeted runs therefore validate the absence of gross path-interference failures, while the `1e-4` correction itself is primarily an analytic finite-instance enumeration.",
                "",
            ]
        )

    if fast_rows:
        lines.extend(
            [
                "## Full Phase-III Fast-Wrapper Checks",
                "",
                "These rows run the full no-SIC phase-III decoder through the fast first-valid-path wrapper.  A row is exact only when `aborted roots = 0`; otherwise PDP is conservative because an aborted root is treated as undecoded.",
                "",
                "| K | pe | trials | cap/root | PDP upper | PHP | schedule fail | rank fail | false positives | aborted roots | node visits | seconds | source |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in fast_rows:
            if row.phase != 3:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.k),
                        f"{row.pe:.3f}",
                        str(row.trials),
                        str(row.max_nodes_per_root),
                        format_probability(row.pdp),
                        format_probability(row.php),
                        format_probability(row.schedule_fail_emp),
                        format_probability(row.rank_fail_emp),
                        str(row.false_positive),
                        str(row.aborted_roots),
                        str(row.node_visits),
                        f"{row.seconds:.2f}",
                        f"`{row.source}`",
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "Readout:",
                "",
                "- Exact zero-abort rows directly check whether full-decoder PDP tracks sampled schedule failure and whether PHP is visible.",
                "- Positive-abort rows are cap-limited PDP upper bounds: excess over sampled schedule failure should not be interpreted as a decoding-theory error unless the row has zero aborts.",
                "- K=30/K=40 full phase-III rows remain opportunistic because runtime grows quickly with empty root searches.",
                "",
            ]
        )

    if root_profile_rows:
        lines.extend(
            [
                "## Phase-III Root-Search Runtime Profile",
                "",
                "These rows profile K=30 and K=40, \(p_e=0.1\), seed-6310 frames attempt by attempt.  They explain the cap-limited full-wrapper behavior: completed paths are true, false paths are absent, and the runtime tail is concentrated in specific root attempts.",
                "",
                "| K | cap/root | attempt | effective roots | attempt-schedule users | phase-schedule users | found paths | true paths | false paths | aborted roots | node visits | max root visits | seconds | source |",
                "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in root_profile_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.k),
                        str(row.max_nodes_per_root),
                        row.attempt,
                        str(row.effective_roots),
                        str(row.attempt_sched_users),
                        str(row.phase_sched_users),
                        str(row.found_paths),
                        str(row.true_paths),
                        str(row.false_paths),
                        str(row.aborted_roots),
                        str(row.node_visits),
                        str(row.max_root_visits),
                        f"{row.seconds:.2f}",
                        f"`{row.source.name}`",
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "Readout:",
                "",
                "- At cap/root \(=50000\), phase-II root 8 and phase-III root 10 are inexpensive for both K values, while phase-III root 0/root 6 still account for most remaining aborts and node visits.",
                "- No false final-valid paths are observed in the profiled attempts; the phase-III excess PDP in capped full-wrapper rows is therefore a runtime-abort upper-bound effect, not observed PHP.",
                "",
            ]
        )

    if true_path_order_rows:
        lines.extend(
            [
                "## Phase-III True-Path Ordering Profile",
                "",
                "These rows condition on attempt-schedule-success users and follow only the true prefix.  They do not enumerate every false subtree; instead they measure where the true continuation appears in the current decoder's child order.",
                "",
                "| K | pe | attempt | users | final-valid true paths | max children | max true index | mean prior siblings | max prior siblings | max log10 prefix work | source |",
                "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in sorted(true_path_order_rows, key=lambda item: (item.k, item.pe, item.attempt)):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.k),
                        f"{row.pe:.3f}",
                        row.attempt,
                        str(row.users),
                        str(row.final_valid),
                        str(row.max_children),
                        str(row.max_true_index),
                        f"{row.mean_prior_siblings:.2f}",
                        str(row.max_prior_siblings),
                        f"{row.max_log10_prefix_work:.2f}",
                        f"`{row.source.name}`",
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "Readout:",
                "",
                "- All profiled attempt-schedule-success users have final-valid true paths, so the sampled failures are not missing intrinsic LLC paths.",
                "- Increasing K from 30 to 40 widens the child lists and pushes the true continuation later in the current row/DFS order; this explains the root-search runtime tail without creating visible PHP.",
                "- This is an ordering/runtime diagnostic, not a replacement for the analytic pair-preemption bound.",
                "",
            ]
        )

    if pretrue_rows:
        lines.extend(
            [
                "## Pre-True-Path Preemption Probe",
                "",
                "This probe follows each schedule-success true prefix and searches only earlier sibling subtrees in the current decoder order.  It asks the exact targeted question: whether a final-valid wrong path appears before the tagged user's true path.",
                "",
                "| K | pe | checked | completed | wrong preemptions | early correct | true missing | true invalid | aborted | node visits | seconds | source |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in sorted(pretrue_rows, key=lambda item: (item.k, item.pe, item.seed)):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.k),
                        f"{row.pe:.3f}",
                        str(row.checked_users),
                        str(row.completed_users),
                        str(row.wrong_preemptions),
                        str(row.early_correct),
                        str(row.true_missing),
                        str(row.true_invalid),
                        str(row.aborted_users),
                        str(row.node_visits),
                        f"{row.seconds:.2f}",
                        f"`{row.source.name}`",
                    ]
                )
                + " |"
            )
        total_completed = sum(row.completed_users for row in pretrue_rows)
        total_wrong = sum(row.wrong_preemptions for row in pretrue_rows)
        total_abort = sum(row.aborted_users for row in pretrue_rows)
        lines.extend(
            [
                "",
                "Readout:",
                "",
                f"- Completed pre-true checks: `{total_completed}`; wrong preemptions: `{total_wrong}`; aborted checks: `{total_abort}`.",
                "- This is closer to the theorem event than a full root sweep, because branches after the true path cannot change the tagged user's first-valid outcome.",
                "",
            ]
        )

    if pair_extras:
        lines.extend(
            [
                "## Exact Phase-III Pair Bound",
                "",
                "The finite-instance affine checker enumerates all two-color profiles for the accepted two-erasure phase-III gap representatives.  The erasure-weighted conservative union extras are:",
                "",
                "| K | pe | erasure-weighted pair extra |",
                "|---:|---:|---:|",
            ]
        )
        for (k, pe), extra in sorted(pair_extras.items()):
            if k in (30, 40):
                lines.append(f"| {k} | {pe:.3f} | {format_probability(extra)} |")
        lines.append("")

    if pair_validation_path.exists():
        lines.extend(
            [
                "## Direct Pair-Decoder Probe Resolution",
                "",
                f"Detailed pair-level validation is summarized in `{pair_validation_path}`.  The direct probe conditions on represented accepted two-erasure masks, inserts one alternate user, and runs the actual first-valid-path decoder.",
                "",
                "| probe | scope | observed preemptions | empirical resolution | readout |",
                "|---|---|---:|---:|---|",
                "| `uace_pair_decoder_probe_pe010_trials1000.md` | 13 gap-representative rows at `p_e=0.1` | 0 | K=40 95% union upper `0.040299` | too coarse for exact `9.310e-05` correction |",
                "| `uace_pair_decoder_probe_root10_pe010_trials20000.md` | root10 mask `(6,10)` only | 0 | K=40 95% union upper `1.336e-05` | local sanity check; root10 raw analytic contribution is `6.889e-07` |",
                "",
                "Readout:",
                "",
                "- These direct probes rule out gross implementation-level pair failures.",
                "- They do not empirically resolve the exact pair correction; the correction remains an analytic finite-instance affine-rank enumeration.",
                "",
            ]
        )

    if hallucination_rows:
        lines.extend(
            [
                "## PHP/Hallucination Root Probes",
                "",
                "These probes sample effective roots and count a hallucination only when a final-valid decoded message is outside the transmitted set.  They are PHP stress tests, not full PDP simulations.",
                "",
                "| K | pe | trials | roots/attempt | sampled roots | valid outputs | true outputs | hallucinations | aborted roots | first-moment PHP bound | source |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in sorted(hallucination_rows, key=lambda item: (item.k, item.pe)):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.k),
                        f"{row.pe:.3f}",
                        str(row.trials),
                        str(row.roots_per_attempt),
                        str(row.sampled_roots),
                        str(row.valid_outputs),
                        str(row.true_outputs),
                        str(row.hallucinations),
                        str(row.aborted_roots),
                        format_probability(row.php_bound),
                        f"`{row.source}`",
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "Readout:",
                "",
                "- No hallucinated full message has been observed in completed root searches.",
                "- Aborted roots remain inconclusive and should be covered analytically by rank/path-shape bounds rather than counted as successes.",
                "",
                "## Hallucination Zero-Event Visibility",
                "",
                "The table below gives the root-probe zero-event visibility.  The completed-root upper bound is not a direct per-user PHP estimate, because sampled roots are not independent transmitted-user decoding attempts.  It is included to show the empirical scale of the stress test.  The final column asks how many user-equivalent zero-event checks would be needed before a 95% upper bound could fall below the analytic first-moment PHP scale.",
                "",
                "| K | pe | completed roots | hallucinations | completed-root 95% upper | first-moment PHP bound | user-equivalent checks needed |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(hallucination_rows, key=lambda item: (item.k, item.pe)):
            completed = row.sampled_roots - row.aborted_roots
            upper = zero_event_upper(completed)
            required = zero_event_required_trials(row.php_bound)
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.k),
                        f"{row.pe:.3f}",
                        str(completed),
                        str(row.hallucinations),
                        format_probability(upper),
                        format_probability(row.php_bound),
                        str(required) if required is not None else "n/a",
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "Readout:",
                "",
                "- The completed root probes are many orders of magnitude too small to empirically resolve the analytic PHP bounds.",
                "- Their role is to catch large implementation-level hallucination modes; the publishable PHP statement must remain analytic.",
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
    parser.add_argument("--files", nargs="+", default=list(DEFAULT_FILES))
    parser.add_argument("--targeted-files", nargs="+", default=list(DEFAULT_TARGETED_FILES))
    parser.add_argument("--pair-file", type=Path, default=Path(DEFAULT_PAIR_FILE))
    parser.add_argument("--fast-files", nargs="+", default=list(DEFAULT_FAST_FILES))
    parser.add_argument("--root-profile-files", nargs="+", default=list(DEFAULT_ROOT_PROFILE_FILES))
    parser.add_argument("--true-path-order-files", nargs="+", default=list(DEFAULT_TRUE_PATH_ORDER_FILES))
    parser.add_argument("--pretrue-files", nargs="+", default=list(DEFAULT_PRETRUE_FILES))
    parser.add_argument("--hallucination-files", nargs="+", default=list(DEFAULT_HALLUCINATION_FILES))
    parser.add_argument("--schedule-mc-files", nargs="+", default=list(DEFAULT_SCHEDULE_MC_FILES))
    parser.add_argument("--pair-validation-file", default=DEFAULT_PAIR_VALIDATION_FILE)
    parser.add_argument("--php-audit-file", default=DEFAULT_PHP_AUDIT_FILE)
    parser.add_argument("--output", type=Path, default=Path("research/uace_k30_k40_validation_summary.md"))
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
