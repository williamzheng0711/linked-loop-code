#!/usr/bin/env python3
"""Generate K=30/K=40 validation figure for the Chinese report."""

from __future__ import annotations

import re
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "llc_uace_mplconfig")
)

import matplotlib.pyplot as plt
import numpy as np

from uace_validation_summary import DEFAULT_FILES, interference_php_bound, parse_overlay


class Args:
    L = 16
    M = 3
    J = 16
    R = 8


PHASE1_FILES = (
    "research/uace_empirical_phase1_K30_pe010_trials5.md",
    "research/uace_empirical_phase1_K30_pe020_trials3.md",
    "research/uace_empirical_phase1_K30_pe030_trials3.md",
    "research/uace_empirical_phase1_K40_pe010_trials3.md",
    "research/uace_empirical_phase1_K40_pe020_trials3.md",
    "research/uace_empirical_phase1_K40_pe030_trials3.md",
)


@dataclass(frozen=True)
class Phase1Row:
    k: int
    pe: float
    trials: int
    pred: float
    empirical_pdp: float
    empirical_schedule_fail: float
    empirical_php: float


def parse_phase1(path: Path) -> Phase1Row:
    text = path.read_text(encoding="utf-8")

    def scalar(pattern: str, label: str) -> str:
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"could not parse {label} from {path}")
        return match.group(1)

    k = int(scalar(r"- `K = ([0-9]+)`", "K"))
    pe = float(scalar(r"- `p_e = ([0-9.]+)`", "p_e"))
    trials = int(scalar(r"- trials: `([0-9]+)`", "trials"))
    empirical_pdp = float(scalar(r"\| pdp \| ([0-9.]+) \|", "pdp"))
    empirical_php = float(scalar(r"\| php \| ([0-9.]+) \|", "php"))
    schedule = float(scalar(r"\| schedule_fail_emp \| ([0-9.]+) \|", "schedule"))
    pred = 1.0 - ((1.0 - pe) ** Args.L)
    return Phase1Row(k, pe, trials, pred, empirical_pdp, schedule, empirical_php)


def plot_phase2(rows: list) -> None:
    args = Args()

    labels = [f"K={row.k}\npe={row.pe:.1f}\nT={row.trials}" for row in rows]
    x = np.arange(len(rows))
    width = 0.25

    pred = np.array([min(1.0, row.schedule_ue + interference_php_bound(row, args)) for row in rows])
    empirical = np.array([row.empirical_pdp for row in rows])
    sampled_schedule = np.array([row.empirical_schedule_fail for row in rows])
    php_bound = np.array([interference_php_bound(row, args) for row in rows])
    empirical_php = np.array([row.empirical_php for row in rows])

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        gridspec_kw={"height_ratios": [2.2, 1.2]},
        constrained_layout=True,
    )

    ax0.bar(x - width, pred, width, label="theory PDP", color="#2f5597")
    ax0.bar(x, empirical, width, label="empirical PDP", color="#c55a11")
    ax0.bar(x + width, sampled_schedule, width, label="sampled schedule fail", color="#70ad47")
    ax0.set_ylim(0.0, 1.08)
    ax0.set_ylabel("PDP")
    ax0.set_title("K=30 / K=40 phase-II UACE validation")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)
    ax0.grid(axis="y", alpha=0.25)
    ax0.legend(ncol=3, loc="upper left")

    floor = 1e-16
    ax1.semilogy(x, np.maximum(php_bound, floor), "o-", label="PHP first-moment bound", color="#7030a0")
    ax1.semilogy(x, np.maximum(empirical_php, floor), "s", label="empirical PHP", color="#c00000")
    ax1.set_ylim(5e-17, 1e-10)
    ax1.set_ylabel("PHP")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(axis="y", alpha=0.25, which="both")
    ax1.legend(loc="upper left")

    output = Path("research/figures/uace_k30_k40_validation.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"wrote {output}")


def plot_phase1_phase2(phase1_rows: list[Phase1Row], phase2_rows: list) -> None:
    args = Args()
    phase1_rows = sorted(phase1_rows, key=lambda item: (item.k, item.pe))
    phase2_rows = sorted(phase2_rows, key=lambda item: (item.k, item.pe))
    labels = [f"K={row.k}\npe={row.pe:.1f}" for row in phase1_rows]
    x = np.arange(len(labels))
    width = 0.25

    p1_pred = np.array([row.pred for row in phase1_rows])
    p1_emp = np.array([row.empirical_pdp for row in phase1_rows])
    p1_sched = np.array([row.empirical_schedule_fail for row in phase1_rows])
    p1_php = np.array([row.empirical_php for row in phase1_rows])

    p2_pred = np.array(
        [min(1.0, row.schedule_ue + interference_php_bound(row, args)) for row in phase2_rows]
    )
    p2_emp = np.array([row.empirical_pdp for row in phase2_rows])
    p2_sched = np.array([row.empirical_schedule_fail for row in phase2_rows])
    p2_php = np.array([row.empirical_php for row in phase2_rows])
    p2_php_bound = np.array([interference_php_bound(row, args) for row in phase2_rows])

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.0, 7.2),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.3, 1.0]},
    )

    for ax, title, pred, emp, sched in (
        (axes[0, 0], "Phase I: zero-erasure schedule", p1_pred, p1_emp, p1_sched),
        (axes[0, 1], "Phase II: one-erasure schedule", p2_pred, p2_emp, p2_sched),
    ):
        ax.bar(x - width, pred, width, label="closed-form / theory PDP", color="#2f5597")
        ax.bar(x, emp, width, label="empirical PDP", color="#c55a11")
        ax.bar(x + width, sched, width, label="sampled schedule fail", color="#70ad47")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, 1.08)
        ax.set_ylabel("PDP")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")

    floor = 1e-16
    axes[1, 0].semilogy(x, np.maximum(p1_php, floor), "s", color="#c00000", label="empirical PHP")
    axes[1, 0].set_ylim(5e-17, 1e-10)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels)
    axes[1, 0].set_ylabel("PHP")
    axes[1, 0].grid(axis="y", alpha=0.25, which="both")
    axes[1, 0].legend(fontsize=8, loc="upper left")

    axes[1, 1].semilogy(
        x,
        np.maximum(p2_php_bound, floor),
        "o-",
        color="#7030a0",
        label="PHP first-moment bound",
    )
    axes[1, 1].semilogy(x, np.maximum(p2_php, floor), "s", color="#c00000", label="empirical PHP")
    axes[1, 1].set_ylim(5e-17, 1e-10)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels)
    axes[1, 1].set_ylabel("PHP")
    axes[1, 1].grid(axis="y", alpha=0.25, which="both")
    axes[1, 1].legend(fontsize=8, loc="upper left")

    fig.suptitle("Phase I / II K=30 and K=40 UACE finite-instance validation", fontsize=14)
    output = Path("research/figures/uace_phase1_phase2_validation.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190)
    print(f"wrote {output}")


def main() -> int:
    phase2_rows = [parse_overlay(Path(item)) for item in DEFAULT_FILES]
    phase1_rows = [parse_phase1(Path(item)) for item in PHASE1_FILES]
    plot_phase2(phase2_rows)
    plot_phase1_phase2(phase1_rows, phase2_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
