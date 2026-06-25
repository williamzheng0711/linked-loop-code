#!/usr/bin/env python3
"""Hallucination-focused root probe for LLC/UACE.

The full phase-III decoder can be slow because it tries many roots whose search
trees contain no valid output.  This probe samples roots from each current
decoder attempt and asks a narrower question: when a first final-valid path is
found, is the decoded full message a true transmitted message or a
hallucination?

It is a runtime-feasible empirical sanity check for the PHP side of the
first-moment bound.  It is not a full PDP benchmark.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from uace_empirical_probe import rows_as_set, uace_channel
from uace_fast_empirical_probe import first_valid_path_for_root
from uace_interference_bound import interference_moments
from uace_schedule_bound import bad_mask_probability, schedule_bad_masks
from uace_targeted_path_probe import build_attempt_context, bit_row_key, unrotate_decoded


def load_playground():
    repo_root = Path(__file__).resolve().parents[1]
    playground = repo_root / "playground"
    sys.path.insert(0, str(playground))
    import abch_utils  # type: ignore
    import general_lib  # type: ignore
    import general_utils as gu  # type: ignore
    import linkedloop as LLC  # type: ignore
    import static_repo  # type: ignore

    return static_repo, general_lib, gu, LLC, abch_utils


@dataclass(frozen=True)
class ProbeRow:
    seed: int
    attempt: str
    sampled_roots: int
    valid_outputs: int
    true_outputs: int
    hallucinations: int
    aborted_roots: int
    node_visits: int
    seconds: float


def attempt_specs(phase: int):
    specs = []
    if phase >= 2:
        specs.extend(
            [
                ("phase-II root 0", 1, None),
                ("phase-II root 8", 1, [8]),
            ]
        )
    if phase >= 3:
        specs.extend(
            [
                ("phase-III root 0", 2, None),
                ("phase-III root 6", 2, [6]),
                ("phase-III root 10", 2, [6, 10]),
            ]
        )
    return specs


def run_trial(args: argparse.Namespace, seed: int) -> list[ProbeRow]:
    static_repo, general_lib, gu, LLC, abch_utils = load_playground()
    import uace_fast_empirical_probe as fast_probe

    fast_probe.gu_global = gu
    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)

    message_lens, parity_lens = static_repo.get_allocation(args.L)
    gis, columns_index, sub_g_invs = static_repo.get_G_info(
        args.L, args.M, message_lens, parity_lens, seed=seed
    )
    gijs = static_repo.partition_Gs(args.L, args.M, parity_lens, gis)
    tx_bits = rng.integers(0, 2, size=(args.K, static_repo.B), dtype=int)
    tx_cdwds = general_lib.encode(
        tx_bits,
        args.K,
        args.L,
        args.L * static_repo.J,
        args.M,
        message_lens,
        parity_lens,
        gijs,
    )
    tx_symbols = abch_utils.binary_to_symbol(tx_cdwds, args.L, args.K)
    rx_symbols, _erasure_counts, _erasure_masks = uace_channel(tx_symbols, args.pe, rng)
    grand_list = abch_utils.symbol_to_binary(args.K, args.L, rx_symbols)
    true_set = rows_as_set(tx_bits)

    rows = []
    for name, d, chosen_roots in attempt_specs(args.phase):
        start = time.time()
        ctx = build_attempt_context(
            static_repo,
            gu,
            args.L,
            args.M,
            chosen_roots,
            grand_list,
            message_lens,
            parity_lens,
            gis,
            columns_index,
            sub_g_invs,
        )
        valid_roots = list(ctx["caches"][0][0])
        if args.roots_per_attempt > 0 and len(valid_roots) > args.roots_per_attempt:
            valid_roots = list(rng.choice(valid_roots, size=args.roots_per_attempt, replace=False))

        valid_outputs = 0
        true_outputs = 0
        hallucinations = 0
        aborted = 0
        node_visits = 0

        for root in valid_roots:
            path, visits, did_abort = first_valid_path_for_root(
                root=int(root),
                d=d,
                grand=ctx["grand"],
                K=args.K,
                L=args.L,
                M=args.M,
                message_lens=ctx["message_lens"],
                parity_lens=ctx["parity_lens"],
                gis=ctx["gis"],
                gijs=ctx["gijs"],
                columns_index=ctx["columns_index"],
                sub_g_invs=ctx["sub_g_invs"],
                erasure_slot=ctx["erasure_slot"],
                caches=ctx["caches"],
                LLC=LLC,
                max_nodes=args.max_nodes_per_root,
            )
            node_visits += visits
            if did_abort:
                aborted += 1
                continue
            if path is None:
                continue
            decoded = gu.output_message_oop(ctx["grand"], [path], args.L, static_repo.J)
            decoded = unrotate_decoded(decoded, ctx["chosen_root"], ctx["message_lens"], args.L)
            valid_outputs += 1
            if bit_row_key(decoded[0]) in true_set:
                true_outputs += 1
            else:
                hallucinations += 1

        rows.append(
            ProbeRow(
                seed=seed,
                attempt=name,
                sampled_roots=len(valid_roots),
                valid_outputs=valid_outputs,
                true_outputs=true_outputs,
                hallucinations=hallucinations,
                aborted_roots=aborted,
                node_visits=node_visits,
                seconds=time.time() - start,
            )
        )
    return rows


def build_report(args: argparse.Namespace, rows: list[ProbeRow]) -> str:
    schedule_bad = schedule_bad_masks(args.L, args.M, args.phase, args.seed)
    schedule_ue = bad_mask_probability(schedule_bad, args.L, args.pe)
    moments = interference_moments(
        k=args.K,
        length=args.L,
        j=16,
        memory=args.M,
        phase=args.phase,
        pe=args.pe,
        parity_bits_per_section=8,
        seed=args.seed,
    )
    php_bound = sum(item.expected_false_survivors for item in moments) / args.K

    total_roots = sum(row.sampled_roots for row in rows)
    total_hallucinations = sum(row.hallucinations for row in rows)
    total_valid = sum(row.valid_outputs for row in rows)
    total_aborted = sum(row.aborted_roots for row in rows)

    lines = [
        "# UACE Hallucination Root Probe",
        "",
        f"- `K = {args.K}`",
        f"- `L = {args.L}`",
        f"- `M = {args.M}`",
        f"- `pe = {args.pe}`",
        f"- `phase = {args.phase}`",
        f"- trials: `{args.trials}`",
        f"- roots per attempt: `{args.roots_per_attempt}` (`0` means all effective roots)",
        f"- max nodes per root: `{args.max_nodes_per_root}`",
        "",
        "Theory:",
        "",
        f"- schedule UE: `{schedule_ue:.6f}`",
        f"- first-moment PHP bound: `{php_bound:.3e}`",
        "",
        "Aggregate:",
        "",
        f"- sampled roots: `{total_roots}`",
        f"- valid outputs found: `{total_valid}`",
        f"- hallucinations found: `{total_hallucinations}`",
        f"- aborted roots: `{total_aborted}`",
        "",
        "| seed | attempt | sampled roots | valid outputs | true outputs | hallucinations | aborted roots | node visits | seconds |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.seed} | {row.attempt} | {row.sampled_roots} | {row.valid_outputs} | "
            f"{row.true_outputs} | {row.hallucinations} | {row.aborted_roots} | "
            f"{row.node_visits} | {row.seconds:.2f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- This is an empirical PHP stress test, not a full PDP decoder run.",
            "- A hallucination is counted only when a final-valid decoded message is not in the transmitted user set.",
            "- Aborted roots are runtime inconclusive; they are not counted as hallucinations.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--K", type=int, default=40)
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--M", type=int, default=3)
    parser.add_argument("--pe", type=float, default=0.3)
    parser.add_argument("--phase", type=int, default=3, choices=(2, 3))
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--roots-per-attempt", type=int, default=0)
    parser.add_argument("--max-nodes-per-root", type=int, default=500_000)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[ProbeRow] = []
    for idx in range(args.trials):
        rows.extend(run_trial(args, args.seed + idx))
    content = build_report(args, rows)
    if args.output:
        args.output.write_text(content + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
