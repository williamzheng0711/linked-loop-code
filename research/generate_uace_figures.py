#!/usr/bin/env python3
"""Generate figures for the LLC UACE bound report."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "llc_uace_mplconfig")
)

import matplotlib.pyplot as plt
import numpy as np

from uace_bound_explorer import (
    bad_mask_probability,
    circular_run_probability,
    erasure_count_tail,
    geometric_unrecoverable_probability,
    rank_peeling_bad_masks,
    tagged_collision_probability,
)
from uace_schedule_bound import schedule_bad_masks


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"


def parse_overlay(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = None
    rows: list[list[float]] = []
    for line in lines:
        if line.startswith("| pe |"):
            header = [item.strip() for item in line.strip("|").split("|")]
            continue
        if header and line.startswith("| ") and not line.startswith("|---"):
            cells = [item.strip() for item in line.strip("|").split("|")]
            if len(cells) != len(header):
                continue
            rows.append([float(cell) if cell != "n/a" else np.nan for cell in cells])
    if header is None:
        raise ValueError(f"could not parse overlay table: {path}")
    data = np.array(rows, dtype=float)
    return data[:, 0], {name: data[:, idx] for idx, name in enumerate(header)}


def parse_validation(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    in_table = False
    rows = []
    for line in lines:
        if line.startswith("| weight |"):
            in_table = True
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table and line.startswith("| "):
            cells = [item.strip() for item in line.strip("|").split("|")]
            if len(cells) != 5:
                break
            rows.append([int(cell) for cell in cells])
    data = np.array(rows, dtype=float)
    weight = data[:, 0]
    masks = data[:, 1]
    return weight, data[:, 2] / masks, data[:, 3] / masks, data[:, 4] / masks


def plot_bound_curves() -> None:
    length = 16
    memory = 3
    j = 16
    k = 100
    pes = np.linspace(0.0, 0.2, 81)
    rank_bad = rank_peeling_bad_masks(length, memory, 0)
    phase3_bad = schedule_bad_masks(length, memory, 3, 0)

    phase1 = np.array([1.0 - ((1.0 - pe) ** length) for pe in pes])
    phase2 = np.array([erasure_count_tail(length, pe, 2) for pe in pes])
    phase3 = np.array([bad_mask_probability(phase3_bad, length, pe) for pe in pes])
    rank = np.array([bad_mask_probability(rank_bad, length, pe) for pe in pes])
    geom = np.array([geometric_unrecoverable_probability(length, memory, pe) for pe in pes])
    coll = np.array(
        [
            circular_run_probability(
                length, memory, tagged_collision_probability(k, j, pe)
            )
            for pe in pes
        ]
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(pes, phase1, label="phase-I zero-erasure schedule", linewidth=1.9, linestyle="-.")
    ax.plot(pes, phase2, label="phase-II current schedule", linewidth=2.2)
    ax.plot(pes, phase3, label="phase-III current schedule", linewidth=2.2)
    ax.plot(pes, geom, label="TCom geometric UE", linewidth=1.8, linestyle="--")
    ax.plot(pes, rank, label="ideal rank-peeling", linewidth=2.0)
    ax.plot(pes, coll, label="3-collision run, K=100", linewidth=1.4, linestyle=":")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_xlabel(r"section erasure probability $p_e$")
    ax.set_ylabel("probability")
    ax.set_title(r"UACE erasure and collision terms, $L=16,M=3,J=16$")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "uace_bound_curves.png", dpi=220)
    plt.close(fig)


def plot_empirical_overlay() -> None:
    pe2, phase2 = parse_overlay(ROOT / "uace_trend_overlay_K6_phase2_trials5.md")
    pe3, phase3 = parse_overlay(ROOT / "uace_trend_overlay_K6_phase3_trials5.md")

    length = 16
    memory = 3
    pes = np.linspace(0.0, 0.2, 81)
    phase3_bad = schedule_bad_masks(length, memory, 3, 0)
    phase2_curve = np.array([erasure_count_tail(length, pe, 2) for pe in pes])
    phase3_curve = np.array([bad_mask_probability(phase3_bad, length, pe) for pe in pes])

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(pes, phase2_curve, color="#1f77b4", linewidth=2.2, label="phase-II schedule bound")
    ax.plot(pes, phase3_curve, color="#d62728", linewidth=2.2, label="phase-III schedule bound")
    ax.scatter(pe2, phase2["empirical PDP"], color="#1f77b4", marker="o", s=54, label="phase-II PDP, K=6")
    ax.scatter(pe3, phase3["empirical PDP"], color="#d62728", marker="s", s=54, label="phase-III PDP, K=6")
    ax.scatter(pe2, phase2["empirical schedule fail"], color="#1f77b4", marker="x", s=48, label="phase-II sampled masks")
    ax.scatter(pe3, phase3["empirical schedule fail"], color="#d62728", marker="x", s=48, label="phase-III sampled masks")
    ax.set_xlabel(r"section erasure probability $p_e$")
    ax.set_ylabel("PDP / erasure-mask failure rate")
    ax.set_title("Full-decoder pilots track sampled erasure-schedule failures")
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0, 0.85)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "uace_empirical_overlay.png", dpi=220)
    plt.close(fig)


def plot_validation() -> None:
    weight, actual, schedule, rank = parse_validation(ROOT / "uace_mask_validation_phase3_w3.md")
    width = 0.24
    x = np.arange(len(weight))
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(x - width, actual, width=width, label="actual decoder")
    ax.bar(x, schedule, width=width, label="schedule automaton")
    ax.bar(x + width, rank, width=width, label="ideal rank-peeling")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(item)) for item in weight])
    ax.set_xlabel("number of erased sections")
    ax.set_ylabel("success fraction over masks")
    ax.set_title("Mask-level validation, phase III, K=1")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "uace_mask_validation.png", dpi=220)
    plt.close(fig)


def main() -> int:
    FIG_DIR.mkdir(exist_ok=True)
    plot_bound_curves()
    plot_empirical_overlay()
    plot_validation()
    print(f"wrote figures to {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
