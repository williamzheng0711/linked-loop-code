#!/usr/bin/env python3
"""Generate a Phase-III runtime-tail diagnostic figure."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "llc_uace_mplconfig")
)

import matplotlib.pyplot as plt
import numpy as np

from uace_validation_summary import (
    DEFAULT_FAST_FILES,
    DEFAULT_PRETRUE_FILES,
    DEFAULT_ROOT_PROFILE_FILES,
    DEFAULT_TRUE_PATH_ORDER_FILES,
    parse_fast_full,
    parse_pretrue_preemption,
    parse_root_profile,
    parse_true_path_order,
)


def main() -> int:
    fast_rows = [
        parse_fast_full(Path(path))
        for path in DEFAULT_FAST_FILES
        if Path(path).exists()
    ]
    root_rows = [
        row
        for path in DEFAULT_ROOT_PROFILE_FILES
        if Path(path).exists()
        for row in parse_root_profile(Path(path))
    ]
    order_rows = [
        row
        for path in DEFAULT_TRUE_PATH_ORDER_FILES
        if Path(path).exists()
        for row in parse_true_path_order(Path(path))
    ]
    pretrue_rows = [
        row
        for path in DEFAULT_PRETRUE_FILES
        if Path(path).exists()
        for row in parse_pretrue_preemption(Path(path))
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4), constrained_layout=True)

    # Panel A: capped full-wrapper trend.
    ax = axes[0, 0]
    for k, color in [(30, "#2f5597"), (40, "#c55a11")]:
        rows = sorted(
            [row for row in fast_rows if row.k == k and row.phase == 3 and abs(row.pe - 0.1) < 1e-12],
            key=lambda item: item.max_nodes_per_root,
        )
        if not rows:
            continue
        caps = np.array([row.max_nodes_per_root for row in rows], dtype=float)
        pdp = np.array([row.pdp for row in rows])
        sched = np.array([row.schedule_fail_emp for row in rows])
        aborted = np.array([row.aborted_roots for row in rows])
        ax.plot(caps, pdp, "o-", color=color, label=f"K={k} PDP upper")
        ax.plot(caps, sched, "--", color=color, alpha=0.55, label=f"K={k} sampled schedule")
        for x, y, abort in zip(caps, pdp, aborted):
            ax.annotate(f"a={int(abort)}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_ylim(0.0, 0.75)
    ax.set_xlabel("node cap per root")
    ax.set_ylabel("PDP / schedule fail")
    ax.set_title("Capped full-wrapper: excess PDP tracks aborts")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")

    # Panel B: root-localized aborts at largest profiled cap.
    ax = axes[0, 1]
    max_cap_by_k: dict[int, int] = {}
    for row in root_rows:
        if row.phase == 3 and abs(row.pe - 0.1) < 1e-12:
            max_cap_by_k[row.k] = max(max_cap_by_k.get(row.k, 0), row.max_nodes_per_root)
    selected = [
        row
        for row in root_rows
        if row.phase == 3
        and abs(row.pe - 0.1) < 1e-12
        and row.max_nodes_per_root == max_cap_by_k.get(row.k)
        and row.attempt.startswith("phase-III")
    ]
    selected = sorted(selected, key=lambda item: (item.k, item.chosen_root))
    labels = [f"K={row.k}\nr{row.chosen_root}" for row in selected]
    x = np.arange(len(selected))
    width = 0.36
    ax.bar(x - width / 2, [row.true_paths for row in selected], width, label="true paths found", color="#70ad47")
    ax.bar(x + width / 2, [row.aborted_roots for row in selected], width, label="aborted roots", color="#c00000")
    ax.plot(x, [row.false_paths for row in selected], "kx", label="false paths")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("count")
    ax.set_title("Runtime tail localizes to phase-III root 0 / root 6")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")

    # Panel C: true path exists, but is delayed in the child order.
    ax = axes[1, 0]
    attempts = ["phase-III root 0", "phase-III root 6", "phase-III root 10"]
    offsets = {"phase-III root 0": -0.18, "phase-III root 6": 0.0, "phase-III root 10": 0.18}
    colors = {"phase-III root 0": "#2f5597", "phase-III root 6": "#c55a11", "phase-III root 10": "#70ad47"}
    for idx, attempt in enumerate(attempts):
        rows = [row for row in order_rows if row.attempt == attempt]
        xs = np.full(len(rows), idx + offsets[attempt])
        ys = [row.mean_prior_siblings for row in rows]
        ax.scatter(xs, ys, color=colors[attempt], s=42, label=attempt.replace("phase-III ", ""))
        if rows:
            ax.plot([idx - 0.26, idx + 0.26], [max(ys), max(ys)], color=colors[attempt], linewidth=2)
    total_users = sum(row.users for row in order_rows)
    total_valid = sum(row.final_valid for row in order_rows)
    ax.text(0.02, 0.94, f"final-valid true paths: {total_valid}/{total_users}", transform=ax.transAxes, fontsize=9)
    ax.set_xticks(range(len(attempts)))
    ax.set_xticklabels([item.replace("phase-III ", "") for item in attempts])
    ax.set_ylabel("mean prior siblings before true continuation")
    ax.set_title("True paths exist, but current DFS may see many siblings first")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")

    # Panel D: pre-true verifier checks the theorem event directly.
    ax = axes[1, 1]
    pretrue_rows = sorted(pretrue_rows, key=lambda item: (item.k, item.pe))
    labels = [f"K={row.k}\npe={row.pe:.1f}" for row in pretrue_rows]
    x = np.arange(len(pretrue_rows))
    ax.bar(x, [row.completed_users for row in pretrue_rows], color="#70ad47", label="completed checks")
    ax.bar(x, [row.aborted_users for row in pretrue_rows], bottom=[row.completed_users for row in pretrue_rows], color="#a5a5a5", label="aborted checks")
    ax.plot(x, [row.wrong_preemptions for row in pretrue_rows], "rx", markersize=8, label="wrong preemptions")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("users / events")
    ax.set_title("Pre-true-path verifier: no wrong preemption observed")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Phase-III validation gap is a localized runtime/search-tail issue", fontsize=14)
    out = Path("research/figures/uace_phase3_runtime_tail.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=190)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
