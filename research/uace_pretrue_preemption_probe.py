#!/usr/bin/env python3
"""Check whether a wrong path appears before the true path.

For a schedule-success tagged user, the current decoder's first-valid DFS is
wrong only if a final-valid wrong path appears before the tagged user's true
path in the decoder child order.  This probe follows the true prefix and
searches only the earlier sibling subtrees.  It preserves the current
row/DFS order while avoiding branches that occur after the true path.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from uace_bound_explorer import bad_mask_probability, format_probability
from uace_schedule_bound import schedule_bad_masks, schedule_succeeds
from uace_targeted_path_probe import (
    bit_row_key,
    build_attempt_context,
    first_success_attempt,
    load_playground,
    unrotate_decoded,
)


@dataclass(frozen=True)
class TrialRow:
    seed: int
    schedule_fail: float
    schedule_success_users: int
    phase_i_users: int
    checked_users: int
    wrong_preemptions: int
    early_correct: int
    true_missing: int
    true_invalid: int
    aborted_users: int
    node_visits: int
    seconds: float


@dataclass(frozen=True)
class UserResult:
    seed: int
    user: int
    attempt: str
    status: str
    visits: int


def section_row_lookup(grand: np.ndarray, length: int, j: int) -> list[dict[bytes, int]]:
    lookups: list[dict[bytes, int]] = []
    for section in range(length):
        current: dict[bytes, int] = {}
        for row in np.flatnonzero(grand[:, section * j] != -1):
            current[bit_row_key(grand[row, section * j : (section + 1) * j])] = int(row)
        lookups.append(current)
    return lookups


def first_valid_from_prefix(
    *,
    start_section: int,
    start_path,
    ctx: dict,
    gu,
    LLC,
    static_repo,
    args: argparse.Namespace,
    remaining_budget: int,
) -> tuple[object | None, int, bool]:
    valid_ks_by_section, parity_cache, deciders_cache, avail_savers_cache, parity_lookup_by_section, solve_cache = ctx["caches"]
    stack: list[tuple[int, object]] = [(start_section, start_path)]
    visits = 0
    while stack:
        section, path = stack.pop()
        visits += 1
        if visits > remaining_budget:
            return None, visits, True
        if section == args.L:
            if gu.final_parity_check_oop(
                path,
                ctx["grand"],
                ctx["message_lens"],
                ctx["parity_lens"],
                args.L,
                ctx["gijs"],
                args.M,
                parity_cache=parity_cache,
                deciders_cache=deciders_cache,
            ):
                return path, visits, False
            continue

        children = gu.Path_goes_section_l(
            section,
            path,
            ctx["d"],
            ctx["grand"],
            args.K,
            ctx["message_lens"],
            ctx["parity_lens"],
            args.L,
            args.M,
            ctx["gis"],
            ctx["gijs"],
            ctx["columns_index"],
            ctx["sub_g_invs"],
            ctx["erasure_slot"],
            valid_ks_by_section=valid_ks_by_section,
            parity_cache=parity_cache,
            deciders_cache=deciders_cache,
            avail_savers_cache=avail_savers_cache,
            parity_lookup_by_section=parity_lookup_by_section,
            solve_cache=solve_cache,
        )
        for child in reversed(children):
            stack.append((section + 1, child))
    return None, visits, False


def decoded_key(path, *, ctx: dict, gu, static_repo, args: argparse.Namespace) -> bytes:
    decoded = gu.output_message_oop(ctx["grand"], [path], args.L, static_repo.J)
    decoded = unrotate_decoded(decoded, ctx["chosen_root"], ctx["message_lens"], args.L)
    return bit_row_key(decoded[0])


def verify_user(
    *,
    user: int,
    attempt: tuple[str, int, int, set[int]],
    tx_bits: np.ndarray,
    tx_cdwds: np.ndarray,
    erasure_mask: int,
    ctx: dict,
    gu,
    LLC,
    static_repo,
    args: argparse.Namespace,
) -> tuple[str, int]:
    name, root, _d, _erasure_slot = attempt
    if ctx["d"] == 0:
        return "phase_i", 0

    lookups = ctx["section_row_lookup"]
    root_bits = tx_cdwds[user, root * static_repo.J : (root + 1) * static_repo.J]
    root_row = lookups[0].get(bit_row_key(root_bits))
    if root_row is None:
        return "true_missing", 0

    true_key = bit_row_key(tx_bits[user])
    path = LLC.GLinkedLoop([root_row], ctx["message_lens"])
    visits_total = 0

    for section in range(1, args.L):
        valid_ks_by_section, parity_cache, deciders_cache, avail_savers_cache, parity_lookup_by_section, solve_cache = ctx["caches"]
        children = gu.Path_goes_section_l(
            section,
            path,
            ctx["d"],
            ctx["grand"],
            args.K,
            ctx["message_lens"],
            ctx["parity_lens"],
            args.L,
            args.M,
            ctx["gis"],
            ctx["gijs"],
            ctx["columns_index"],
            ctx["sub_g_invs"],
            ctx["erasure_slot"],
            valid_ks_by_section=valid_ks_by_section,
            parity_cache=parity_cache,
            deciders_cache=deciders_cache,
            avail_savers_cache=avail_savers_cache,
            parity_lookup_by_section=parity_lookup_by_section,
            solve_cache=solve_cache,
        )

        original_section = (root + section) % args.L
        erased = bool((erasure_mask >> original_section) & 1)
        if erased:
            true_child = -1
        else:
            bits = tx_cdwds[user, original_section * static_repo.J : (original_section + 1) * static_repo.J]
            true_child = lookups[section].get(bit_row_key(bits))
        true_index = -1
        for idx, child in enumerate(children):
            if child.get_path()[-1] == true_child:
                true_index = idx
                break
        if true_index < 0:
            return "true_missing", visits_total

        for sibling in children[:true_index]:
            remaining = args.max_nodes_per_user - visits_total
            if remaining <= 0:
                return "aborted", visits_total
            found, visits, aborted = first_valid_from_prefix(
                start_section=section + 1,
                start_path=sibling,
                ctx=ctx,
                gu=gu,
                LLC=LLC,
                static_repo=static_repo,
                args=args,
                remaining_budget=remaining,
            )
            visits_total += visits
            if aborted:
                return "aborted", visits_total
            if found is not None:
                if decoded_key(found, ctx=ctx, gu=gu, static_repo=static_repo, args=args) == true_key:
                    return "early_correct", visits_total
                return "wrong_preemption", visits_total

        path = children[true_index]

    if not gu.final_parity_check_oop(
        path,
        ctx["grand"],
        ctx["message_lens"],
        ctx["parity_lens"],
        args.L,
        ctx["gijs"],
        args.M,
        parity_cache=ctx["caches"][1],
        deciders_cache=ctx["caches"][2],
    ):
        return "true_invalid", visits_total
    return "no_preemption", visits_total


def run_trial(args: argparse.Namespace, seed: int) -> tuple[TrialRow, list[UserResult]]:
    static_repo, general_lib, gu, LLC, abch_utils = load_playground()
    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)
    start = time.time()

    message_lens, parity_lens = static_repo.get_allocation(args.L)
    gis, columns_index, sub_g_invs = static_repo.get_G_info(args.L, args.M, message_lens, parity_lens, seed=seed)
    gijs = static_repo.partition_Gs(args.L, args.M, parity_lens, gis)

    tx_bits = rng.integers(0, 2, size=(args.K, static_repo.B), dtype=int)
    tx_cdwds = general_lib.encode(tx_bits, args.K, args.L, args.L * static_repo.J, args.M, message_lens, parity_lens, gijs)
    tx_symbols = abch_utils.binary_to_symbol(tx_cdwds, args.L, args.K)
    rx_symbols, _erasure_counts, erasure_masks = __import__("uace_empirical_probe").uace_channel(tx_symbols, args.pe, rng)
    grand_list = abch_utils.symbol_to_binary(args.K, args.L, rx_symbols)

    contexts: dict[str, dict] = {}
    for name, _root, d, _erasure_slot in __import__("uace_schedule_bound").schedule_attempts(args.L, args.phase):
        if d == 0:
            continue
        chosen_roots = None
        if name == "phase-II root 8":
            chosen_roots = [8]
        elif name == "phase-III root 6":
            chosen_roots = [6]
        elif name == "phase-III root 10":
            chosen_roots = [6, 10]
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
        ctx["name"] = name
        ctx["d"] = d
        ctx["section_row_lookup"] = section_row_lookup(ctx["grand"], args.L, static_repo.J)
        contexts[name] = ctx

    schedule_success = [
        schedule_succeeds(
            int(mask),
            length=args.L,
            memory=args.M,
            phase=args.phase,
            message_lens=message_lens,
            gijs=gijs,
        )
        for mask in erasure_masks
    ]

    counts = {
        "phase_i": 0,
        "no_preemption": 0,
        "wrong_preemption": 0,
        "early_correct": 0,
        "true_missing": 0,
        "true_invalid": 0,
        "aborted": 0,
    }
    node_visits = 0
    checked = 0
    user_results: list[UserResult] = []
    only_users = set(args.only_users or [])
    for user, ok in enumerate(schedule_success):
        if only_users and user not in only_users:
            continue
        if not ok:
            continue
        attempt = first_success_attempt(
            int(erasure_masks[user]),
            length=args.L,
            memory=args.M,
            phase=args.phase,
            message_lens=message_lens,
            gijs=gijs,
        )
        if attempt is None:
            counts["true_missing"] += 1
            user_results.append(UserResult(seed, user, "none", "true_missing", 0))
            continue
        name, _root, d, _erasure_slot = attempt
        if d == 0:
            counts["phase_i"] += 1
            user_results.append(UserResult(seed, user, name, "phase_i", 0))
            continue
        checked += 1
        status, visits = verify_user(
            user=user,
            attempt=attempt,
            tx_bits=tx_bits,
            tx_cdwds=tx_cdwds,
            erasure_mask=int(erasure_masks[user]),
            ctx=contexts[name],
            gu=gu,
            LLC=LLC,
            static_repo=static_repo,
            args=args,
        )
        counts[status] += 1
        node_visits += visits
        user_results.append(UserResult(seed, user, name, status, visits))

    return (
        TrialRow(
            seed=seed,
            schedule_fail=float(1.0 - np.mean(schedule_success)),
            schedule_success_users=int(sum(schedule_success)),
            phase_i_users=counts["phase_i"],
            checked_users=checked,
            wrong_preemptions=counts["wrong_preemption"],
            early_correct=counts["early_correct"],
            true_missing=counts["true_missing"],
            true_invalid=counts["true_invalid"],
            aborted_users=counts["aborted"],
            node_visits=node_visits,
            seconds=time.time() - start,
        ),
        user_results,
    )


def build_report(args: argparse.Namespace, rows: list[TrialRow], user_rows: list[UserResult]) -> str:
    schedule_bad = schedule_bad_masks(args.L, args.M, args.phase, args.seed)
    schedule_ue = bad_mask_probability(schedule_bad, args.L, args.pe)
    total_checked = sum(row.checked_users for row in rows)
    total_completed = sum(row.checked_users - row.aborted_users for row in rows)
    total_wrong = sum(row.wrong_preemptions for row in rows)
    lines = [
        "# UACE Pre-True-Path Preemption Probe",
        "",
        "This report is generated by `research/uace_pretrue_preemption_probe.py`.",
        "",
        f"- `K = {args.K}`",
        f"- `L = {args.L}`",
        f"- `M = {args.M}`",
        f"- `pe = {args.pe}`",
        f"- `phase = {args.phase}`",
        f"- trials: `{args.trials}`",
        f"- max nodes per checked user: `{args.max_nodes_per_user}`",
        f"- only users: `{','.join(str(item) for item in args.only_users) if args.only_users else 'all'}`",
        f"- schedule UE: `{format_probability(schedule_ue)}`",
        "",
        "Summary:",
        "",
        f"- checked path users: `{total_checked}`",
        f"- completed path users: `{total_completed}`",
        f"- wrong preemptions: `{total_wrong}`",
        "",
        "| seed | schedule fail | schedule-success users | phase-I users | checked | completed | wrong preemptions | early correct | true missing | true invalid | aborted | node visits | seconds |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        completed = row.checked_users - row.aborted_users
        lines.append(
            f"| {row.seed} | {row.schedule_fail:.6f} | {row.schedule_success_users} | "
            f"{row.phase_i_users} | {row.checked_users} | {completed} | {row.wrong_preemptions} | "
            f"{row.early_correct} | {row.true_missing} | {row.true_invalid} | {row.aborted_users} | "
            f"{row.node_visits} | {row.seconds:.2f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- A `wrong preemption` is a final-valid decoded message before the tagged user's true path and different from the tagged message.",
            "- `early correct` means the first earlier final-valid path already decodes to the tagged message, so it is not a PDP/PHP event.",
            "- Aborted users are inconclusive under the node cap and should not be counted as successes.",
            "",
        ]
    )
    if user_rows:
        lines.extend(
            [
                "User rows:",
                "",
                "| seed | user | attempt | status | node visits |",
                "|---:|---:|---|---|---:|",
            ]
        )
        for row in user_rows:
            lines.append(f"| {row.seed} | {row.user} | {row.attempt} | {row.status} | {row.visits} |")
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--K", type=int, default=30)
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--M", type=int, default=3)
    parser.add_argument("--pe", type=float, default=0.1)
    parser.add_argument("--phase", type=int, default=3, choices=(2, 3))
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6310)
    parser.add_argument("--max-nodes-per-user", type=int, default=500_000)
    parser.add_argument("--only-users", type=int, nargs="*", default=[])
    parser.add_argument("--output", type=Path, default=Path("research/uace_pretrue_preemption_probe.md"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trial_rows: list[TrialRow] = []
    user_rows: list[UserResult] = []
    for idx in range(args.trials):
        trial, users = run_trial(args, args.seed + idx)
        trial_rows.append(trial)
        user_rows.extend(users)
    content = build_report(args, trial_rows, user_rows)
    if args.output:
        args.output.write_text(content + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
