#!/usr/bin/env python3
"""Create a publish-facing LLC/UACE validation certificate and dashboard."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "llc_uace_mplconfig")
)

import matplotlib.pyplot as plt
import numpy as np

from uace_bound_explorer import format_probability
from uace_composite_predictor import composite_rows
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
    interference_php_bound,
    parse_overlay,
    parse_fast_full,
    parse_root_profile,
    parse_schedule_mc,
    parse_targeted,
    parse_true_path_order,
    parse_pretrue_preemption,
)
from uace_php_audit import audit_rows


@dataclass(frozen=True)
class CertificateInputs:
    full_decoder: list
    composite: list
    schedule_mc: list
    targeted: list
    php: list
    true_path_order: list
    fast_full: list
    root_profile: list
    pretrue: list


def collect_inputs(args: argparse.Namespace) -> CertificateInputs:
    full_decoder = [
        parse_overlay(Path(path))
        for path in args.full_decoder_files
        if Path(path).exists()
    ]
    schedule_mc = [
        row
        for path in args.schedule_mc_files
        if Path(path).exists()
        for row in parse_schedule_mc(Path(path))
    ]
    targeted_rows = [
        row
        for path in args.targeted_files
        if Path(path).exists()
        for row in parse_targeted(Path(path))
    ]
    return CertificateInputs(
        full_decoder=full_decoder,
        composite=composite_rows(args),
        schedule_mc=schedule_mc,
        targeted=aggregate_targeted(targeted_rows),
        php=audit_rows(args),
        true_path_order=[
            row
            for path in args.true_path_order_files
            if Path(path).exists()
            for row in parse_true_path_order(Path(path))
        ],
        fast_full=[
            parse_fast_full(Path(path))
            for path in args.fast_files
            if Path(path).exists()
        ],
        root_profile=[
            row
            for path in args.root_profile_files
            if Path(path).exists()
            for row in parse_root_profile(Path(path))
        ],
        pretrue=[
            row
            for path in args.pretrue_files
            if Path(path).exists()
            for row in parse_pretrue_preemption(Path(path))
        ],
    )


def key(k: int, pe: float) -> tuple[int, float]:
    return (k, round(pe, 12))


def make_dashboard(args: argparse.Namespace, data: CertificateInputs) -> None:
    fig_path = args.figure
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    comp = {(row.k, row.pe): row for row in data.composite}
    sched = {(row.k, row.pe): row for row in data.schedule_mc}
    php = {(row.k, row.pe): row for row in data.php}
    targeted = {(row.k, row.pe): row for row in data.targeted}
    ordered_keys = [(k, pe) for k in args.Ks for pe in args.pes]
    labels = [f"K={k}\n$p_e$={pe:.1f}" for k, pe in ordered_keys]
    x = np.arange(len(ordered_keys))

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11.0, 9.2),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.35, 1.15, 1.15]},
    )

    ax = axes[0]
    schedule = np.array([comp[(k, pe)].schedule for k, pe in ordered_keys])
    pdp = np.array([comp[(k, pe)].pdp_pred for k, pe in ordered_keys])
    schedule_emp = np.array([sched[(k, pe)].empirical for k, pe in ordered_keys])
    schedule_hw = np.array([sched[(k, pe)].half_width_95 for k, pe in ordered_keys])
    targeted_emp = np.array([targeted[(k, pe)].schedule_fail_emp for k, pe in ordered_keys])

    ax.plot(x, schedule, "o-", label="exact schedule term", linewidth=2.0, color="#1f4e79")
    ax.plot(x, pdp, "s--", label="composite PDP prediction", linewidth=1.7, color="#c55a11")
    ax.errorbar(
        x,
        schedule_emp,
        yerr=schedule_hw,
        fmt="^",
        color="#548235",
        label="large-sample schedule MC",
        capsize=3,
    )
    ax.scatter(x, targeted_emp, marker="x", s=54, color="#7030a0", label="targeted K-user sampled masks")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("PDP / schedule failure")
    ax.set_title("Phase-III K=30/K=40 PDP prediction is schedule-dominant")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=8, loc="upper left")

    ax = axes[1]
    pair = np.array([php[(k, pe)].pair_preempt for k, pe in ordered_keys])
    pure = np.array([php[(k, pe)].hallucination_php for k, pe in ordered_keys])
    composite_php = np.array([php[(k, pe)].php_bound for k, pe in ordered_keys])
    ax.semilogy(x, pair, "o-", label="exact pair-preempt", linewidth=2.0, color="#c55a11")
    ax.semilogy(x, pure, "s-", label="pure hallucination first moment", linewidth=2.0, color="#5b9bd5")
    ax.semilogy(x, composite_php, "^--", label="composite PHP bound", linewidth=1.6, color="#7030a0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("probability")
    ax.set_title("PHP bound components: pair-preemption dominates pure hallucination")
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.legend(ncol=3, fontsize=8, loc="upper right")

    ax = axes[2]
    required = np.array([php[(k, pe)].composite_needed for k, pe in ordered_keys], dtype=float)
    completed = []
    for k, pe in ordered_keys:
        row = targeted[(k, pe)]
        completed.append(max(row.checked_users - row.aborted_users, 1))
    completed = np.array(completed, dtype=float)
    width = 0.32
    ax.bar(x - width / 2, completed, width=width, label="completed targeted checks", color="#70ad47")
    ax.bar(x + width / 2, required, width=width, label="checks for 95% upper at PHP scale", color="#a64d79")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("checks")
    ax.set_title("Rare-event empirical resolution remains the limiting factor")
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")

    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def build_certificate(args: argparse.Namespace, data: CertificateInputs) -> str:
    comp = {(row.k, row.pe): row for row in data.composite}
    sched = {(row.k, row.pe): row for row in data.schedule_mc}
    php = {(row.k, row.pe): row for row in data.php}
    targeted = {(row.k, row.pe): row for row in data.targeted}
    ordered_keys = [(k, pe) for k in args.Ks for pe in args.pes]

    lines = [
        "# UACE Publish-Level Validation Certificate",
        "",
        "This certificate is generated by `research/uace_publish_validation_certificate.py`.",
        "",
        f"Dashboard figure: `{args.figure}`",
        "",
        "Predictive validation gate: `research/uace_predictive_validation_gate.md`",
        "",
        "## Claims",
        "",
        "| claim | evidence | status |",
        "|---|---|---|",
        "| Phase-I full decoder follows the zero-erasure finite-instance predictor | K=30/K=40 playground decoder runs at multiple erasure rates; validation gate row | validated at visible scale |",
        "| Phase-I/II schedule predictivity generalizes across tested K | K-sweep at pe=0.1 with 11 rows, zero PDP-schedule gap, and zero empirical PHP | validated at visible scale |",
        "| Phase-III PDP is schedule-dominant for K=30/K=40 | Exact schedule term plus large-sample erasure-mask Monte Carlo | validated at visible scale |",
        "| Phase-II full decoder follows the finite-instance predictor | K=30/K=40 playground decoder runs at multiple erasure rates | validated at visible scale |",
        "| Pair-preemption is the main finite correction | Exact two-color affine-rank enumeration, direct pair-decoder probes with zero preemptions | analytic correction with implementation sanity checks |",
        "| No observed pre-true wrong path | pre-true-path verifier over K=30/K=40 grid | direct theorem-event sanity check with aborts noted |",
        "| Full phase-III root sweep has a runtime tail | capped full-wrapper rows, root-profile rows, true-path ordering profiles | localized implementation limitation |",
        "| Pure hallucination is not the visible PHP bottleneck | Rank-corrected first moment, PHP component audit | analytic bound; empirically below resolution |",
        "| Ordinary Monte Carlo cannot resolve PHP correction directly | zero-event sample requirement tables | quantified limitation |",
        "| Full phase-III root-sweep theorem without caps | validation gate GAP row | remaining theorem target |",
        "| Full multi-color theorem | 3-color chunk aggregate with 17 included windows plus machine-readable manifest | remaining theorem target |",
        "",
        "## Full Decoder Phase-II Check",
        "",
        "| K | pe | trials | users | PDP prediction | empirical PDP | sampled schedule | z-score | PHP bound | empirical PHP |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in data.full_decoder:
        users = row.k * row.trials
        php_bound = interference_php_bound(row, args)
        pred = min(1.0, row.schedule_ue + php_bound)
        se = binomial_se(row.schedule_ue, users)
        z_score = (row.empirical_pdp - row.schedule_ue) / se if se > 0 else 0.0
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.k),
                    f"{row.pe:.3f}",
                    str(row.trials),
                    str(users),
                    format_probability(pred),
                    format_probability(row.empirical_pdp),
                    format_probability(row.empirical_schedule_fail),
                    f"{z_score:.2f}",
                    format_probability(php_bound),
                    format_probability(row.empirical_php),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Readout: empirical PDP equals the sampled schedule-failure rate in every completed phase-II full-decoder run, and empirical PHP is zero throughout.",
            "",
            "## Composite Predictor And Visible Validation",
            "",
            "| K | pe | PDP prediction | schedule MC | MC z-score | targeted sampled schedule | PHP bound | completed checks | checks needed at PHP scale |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for k, pe in ordered_keys:
        comp_row = comp[(k, pe)]
        sched_row = sched[(k, pe)]
        php_row = php[(k, pe)]
        targ_row = targeted[(k, pe)]
        completed = targ_row.checked_users - targ_row.aborted_users
        lines.append(
            "| "
            + " | ".join(
                [
                    str(k),
                    f"{pe:.3f}",
                    format_probability(comp_row.pdp_pred),
                    format_probability(sched_row.empirical),
                    f"{sched_row.z_score:.2f}",
                    format_probability(targ_row.schedule_fail_emp),
                    format_probability(php_row.php_bound),
                    str(completed),
                    str(php_row.composite_needed),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## PHP Component Audit",
            "",
            "| K | pe | pair-preempt | pure hallucination | pair / hallucination | dominant pure-hallucination component |",
            "|---:|---:|---:|---:|---:|---|",
        ]
    )
    for k, pe in ordered_keys:
        row = php[(k, pe)]
        ratio = row.pair_preempt / row.hallucination_php if row.hallucination_php > 0 else float("inf")
        component = row.dominant_attempt
        if row.dominant_attempt == "phase-III root 0" and row.dominant_weight == 2:
            component = "phase-III root 0/root 6 (tie)"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(k),
                    f"{pe:.3f}",
                    format_probability(row.pair_preempt),
                    format_probability(row.hallucination_php),
                    format_probability(ratio),
                    f"{component}, w={row.dominant_weight}, e={row.dominant_exponent}",
                ]
            )
            + " |"
        )

    if data.true_path_order:
        lines.extend(
            [
                "",
                "## True-Path Ordering Diagnostic",
                "",
                "This diagnostic follows only schedule-success true prefixes.  It verifies that the true path exists and is final-valid, while measuring how many earlier siblings the current first-valid DFS may inspect before reaching it.",
                "",
                "| K | pe | attempt | users | final-valid true paths | mean prior siblings | max prior siblings | max log10 prefix work |",
                "|---:|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(data.true_path_order, key=lambda item: (item.k, item.pe, item.attempt)):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.k),
                        f"{row.pe:.3f}",
                        row.attempt,
                        str(row.users),
                        str(row.final_valid),
                        f"{row.mean_prior_siblings:.2f}",
                        str(row.max_prior_siblings),
                        f"{row.max_log10_prefix_work:.2f}",
                    ]
                )
                + " |"
        )

    if data.pretrue:
        lines.extend(
            [
                "",
                "## Pre-True-Path Preemption",
                "",
                "This verifier searches only sibling subtrees that precede the tagged user's true path in the current decoder order.  A wrong preemption here is exactly a final-valid wrong path before the true path.",
                "",
                "| K | pe | checked | completed | wrong preemptions | aborted | node visits |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(data.pretrue, key=lambda item: (item.k, item.pe)):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.k),
                        f"{row.pe:.3f}",
                        str(row.checked_users),
                        str(row.completed_users),
                        str(row.wrong_preemptions),
                        str(row.aborted_users),
                        str(row.node_visits),
                    ]
                )
                + " |"
            )

    if data.fast_full:
        lines.extend(
            [
                "",
                "## Full-Wrapper Runtime Tail",
                "",
                "These rows run the full no-SIC phase-III fast wrapper.  Positive abort counts make PDP a conservative upper bound, but PHP and false positives are still direct diagnostics.",
                "",
                "| K | pe | cap/root | PDP upper | PHP | schedule fail | correct decoded | false positives | aborted roots |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(data.fast_full, key=lambda item: (item.k, item.pe, item.max_nodes_per_root)):
            if row.phase != 3:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.k),
                        f"{row.pe:.3f}",
                        str(row.max_nodes_per_root),
                        format_probability(row.pdp),
                        format_probability(row.php),
                        format_probability(row.schedule_fail_emp),
                        str(row.correct),
                        str(row.false_positive),
                        str(row.aborted_roots),
                    ]
                )
                + " |"
            )

    if data.root_profile:
        max_cap = max(row.max_nodes_per_root for row in data.root_profile)
        max_cap_rows = [row for row in data.root_profile if row.max_nodes_per_root == max_cap]
        lines.extend(
            [
                "",
                "## Root-Profile Localization",
                "",
                f"The table keeps only the largest profiled cap/root, `{max_cap}`, and shows where the remaining aborts live.",
                "",
                "| K | attempt | attempt-schedule users | true paths found | false paths | aborted roots | node visits |",
                "|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(max_cap_rows, key=lambda item: (item.k, item.attempt)):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.k),
                        row.attempt,
                        str(row.attempt_sched_users),
                        str(row.true_paths),
                        str(row.false_paths),
                        str(row.aborted_roots),
                        str(row.node_visits),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "For \(K=30,40\), the theory is predictive for the visible PDP scale: the exact schedule term agrees with high-resolution erasure-mask Monte Carlo, and the composite PDP differs from the schedule term by less than the plotted sampling error.  PHP is analytically controlled but remains below feasible ordinary Monte Carlo resolution.  The true-path ordering and pre-true-path diagnostics explain the current cap-limited full phase-III root sweep as an implementation/search-tail issue rather than observed PHP.  The current publishable claim should therefore be a finite-instance theorem plus validation certificate, not a claim that rare PHP events have been directly measured or that full root-sweep validation without caps is finished.",
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
    parser.add_argument("--full-decoder-files", nargs="+", default=list(DEFAULT_FILES))
    parser.add_argument("--schedule-mc-files", nargs="+", default=list(DEFAULT_SCHEDULE_MC_FILES))
    parser.add_argument("--targeted-files", nargs="+", default=list(DEFAULT_TARGETED_FILES))
    parser.add_argument("--true-path-order-files", nargs="+", default=list(DEFAULT_TRUE_PATH_ORDER_FILES))
    parser.add_argument("--pretrue-files", nargs="+", default=list(DEFAULT_PRETRUE_FILES))
    parser.add_argument("--fast-files", nargs="+", default=list(DEFAULT_FAST_FILES))
    parser.add_argument("--root-profile-files", nargs="+", default=list(DEFAULT_ROOT_PROFILE_FILES))
    parser.add_argument("--figure", type=Path, default=Path("research/figures/uace_publish_validation_dashboard.png"))
    parser.add_argument("--output", type=Path, default=Path("research/uace_publish_validation_certificate.md"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = collect_inputs(args)
    make_dashboard(args, data)
    content = build_certificate(args, data)
    args.output.write_text(content + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
