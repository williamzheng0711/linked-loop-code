#!/usr/bin/env python3
"""Profile phase-III fast-wrapper root searches attempt by attempt.

This script is a diagnostic companion to `uace_fast_empirical_probe.py`.  It
does not claim to be an exact full-decoder validation when roots abort.  Its
purpose is to locate the runtime tail: which phase/root attempts consume nodes,
which attempts abort under a cap, and whether any completed searches produce
false messages.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from uace_bound_explorer import format_probability, rank_peeling_succeeds
from uace_empirical_probe import rows_as_set, uace_channel
import uace_fast_empirical_probe as fast_probe
from uace_fast_empirical_probe import (
    first_valid_path_for_root,
    load_playground,
    rotate_decoder_state,
)
from uace_schedule_bound import schedule_succeeds
from uace_schedule_bound import attempt_succeeds


@dataclass(frozen=True)
class AttemptProfile:
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


def unrotate_decoded(decoded: np.ndarray, chosen_root: int, message_lens: np.ndarray, length: int) -> np.ndarray:
    if decoded.size == 0:
        return decoded
    width = int(sum(message_lens))
    out = decoded.copy()
    out[:, range(width)] = out[:, np.mod(np.arange(width) + sum(message_lens[0 : length - chosen_root]), width)]
    return out


def attempt_specs(phase: int) -> list[tuple[str, int, list[int] | None]]:
    specs: list[tuple[str, int, list[int] | None]] = []
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


def profile_attempt(
    *,
    name: str,
    d: int,
    chosen_roots: list[int] | None,
    static_repo,
    gu,
    LLC,
    grand_list: np.ndarray,
    true_set: set[tuple[int, ...]],
    args: argparse.Namespace,
    message_lens: np.ndarray,
    parity_lens: np.ndarray,
    gis: np.ndarray,
    columns_index: np.ndarray,
    sub_g_invs: np.ndarray,
    tx_symbols: np.ndarray,
    erasure_masks: np.ndarray,
    gijs_original: dict,
) -> AttemptProfile:
    start = time.time()
    fast_probe.gu_global = gu
    fast_probe.L_GLOBAL = args.L
    (
        chosen_root,
        erasure_slot,
        grand,
        msg_lens,
        par_lens,
        rot_gis,
        rot_columns,
        rot_invs,
        gijs,
    ) = rotate_decoder_state(
        static_repo,
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
    deciders_cache, avail_savers_cache = gu.build_decoder_lookup(args.L, args.M)
    solve_cache = gu.build_solve_cache(args.L, args.M, rot_columns, rot_invs, rot_gis)
    valid_ks_by_section = [np.flatnonzero(grand[:, section * static_repo.J] != -1) for section in range(args.L)]
    parity_lookup_by_section = gu.build_parity_lookup_by_section(grand, args.L, msg_lens, valid_ks_by_section)
    parity_cache = gu.build_parity_cache(grand, args.L, args.M, msg_lens, gijs)
    caches = (
        valid_ks_by_section,
        parity_cache,
        deciders_cache,
        avail_savers_cache,
        parity_lookup_by_section,
        solve_cache,
    )

    effective_roots = [int(root) for root in range(args.K) if grand[root, 0] != -1]
    root_to_users: dict[int, list[int]] = {}
    for user in range(args.K):
        if (int(erasure_masks[user]) >> chosen_root) & 1:
            continue
        root_to_users.setdefault(int(tx_symbols[user, chosen_root]), []).append(user)

    roots_with_users = candidate_users = attempt_sched_users = phase_sched_users = 0
    found_paths = true_paths = false_paths = aborted_roots = node_visits = max_root_visits = 0
    for root in effective_roots:
        root_symbol_bits = np.asarray(grand[root, : static_repo.J], dtype=int)
        root_symbol = int(root_symbol_bits @ (1 << np.arange(static_repo.J - 1, -1, -1)))
        candidate_user_ids = root_to_users.get(root_symbol, [])
        if candidate_user_ids:
            roots_with_users += 1
            candidate_users += len(candidate_user_ids)
            for user in candidate_user_ids:
                user_mask = int(erasure_masks[user])
                if attempt_succeeds(
                    user_mask,
                    length=args.L,
                    memory=args.M,
                    root=chosen_root,
                    d=d,
                    erasure_slot=set(erasure_slot),
                    message_lens=message_lens,
                    gijs=gijs_original,
                ):
                    attempt_sched_users += 1
                if schedule_succeeds(
                    user_mask,
                    length=args.L,
                    memory=args.M,
                    phase=args.phase,
                    message_lens=message_lens,
                    gijs=gijs_original,
                ):
                    phase_sched_users += 1
        path, visits, aborted = first_valid_path_for_root(
            root=root,
            d=d,
            grand=grand,
            K=args.K,
            L=args.L,
            M=args.M,
            message_lens=msg_lens,
            parity_lens=par_lens,
            gis=rot_gis,
            gijs=gijs,
            columns_index=rot_columns,
            sub_g_invs=rot_invs,
            erasure_slot=erasure_slot,
            caches=caches,
            LLC=LLC,
            max_nodes=args.max_nodes_per_root,
        )
        node_visits += visits
        max_root_visits = max(max_root_visits, visits)
        aborted_roots += int(aborted)
        if path is None:
            continue
        found_paths += 1
        recovered = gu.output_message_oop(grand, [path], args.L, static_repo.J)
        recovered = unrotate_decoded(recovered, chosen_root, msg_lens, args.L)
        recovered_key = np.asarray(recovered[0], dtype=np.uint8).tobytes()
        if recovered_key in true_set:
            true_paths += 1
        else:
            false_paths += 1

    return AttemptProfile(
        attempt=name,
        chosen_root=chosen_root,
        d=d,
        effective_roots=len(effective_roots),
        roots_with_users=roots_with_users,
        candidate_users=candidate_users,
        attempt_sched_users=attempt_sched_users,
        phase_sched_users=phase_sched_users,
        found_paths=found_paths,
        true_paths=true_paths,
        false_paths=false_paths,
        aborted_roots=aborted_roots,
        node_visits=node_visits,
        max_root_visits=max_root_visits,
        seconds=time.time() - start,
    )


def build_report(args: argparse.Namespace) -> str:
    static_repo, general_lib, gu, LLC, abch_utils = load_playground()
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    message_lens, parity_lens = static_repo.get_allocation(args.L)
    gis, columns_index, sub_g_invs = static_repo.get_G_info(
        args.L,
        args.M,
        message_lens,
        parity_lens,
        seed=args.seed,
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
    rx_symbols, _erasure_counts, erasure_masks = uace_channel(tx_symbols, args.pe, rng)
    grand_list = abch_utils.symbol_to_binary(args.K, args.L, rx_symbols)
    true_set = rows_as_set(tx_bits)

    schedule_fail_emp = float(
        np.mean(
            [
                not schedule_succeeds(
                    int(mask),
                    length=args.L,
                    memory=args.M,
                    phase=args.phase,
                    message_lens=message_lens,
                    gijs=gijs,
                )
                for mask in erasure_masks
            ]
        )
    )
    rank_fail_emp = float(
        np.mean(
            [
                not rank_peeling_succeeds(int(mask), args.L, args.M, message_lens, gijs)
                for mask in erasure_masks
            ]
        )
    )

    profiles = [
        profile_attempt(
            name=name,
            d=d,
            chosen_roots=chosen_roots,
            static_repo=static_repo,
            gu=gu,
            LLC=LLC,
            grand_list=grand_list,
            true_set=true_set,
            args=args,
            message_lens=message_lens,
            parity_lens=parity_lens,
            gis=gis,
            columns_index=columns_index,
            sub_g_invs=sub_g_invs,
            tx_symbols=tx_symbols,
            erasure_masks=erasure_masks,
            gijs_original=gijs,
        )
        for name, d, chosen_roots in attempt_specs(args.phase)
    ]

    lines = [
        "# UACE Phase-III Root-Search Profile",
        "",
        "This report is generated by `research/uace_phase3_root_profile.py`.",
        "",
        f"- `K = {args.K}`",
        f"- `L = {args.L}`",
        f"- `M = {args.M}`",
        f"- `pe = {args.pe}`",
        f"- phase: `{args.phase}`",
        f"- seed: `{args.seed}`",
        f"- max nodes per root: `{args.max_nodes_per_root}`",
        f"- sampled schedule fail: `{schedule_fail_emp:.6f}`",
        f"- sampled rank fail: `{rank_fail_emp:.6f}`",
        "",
        "| attempt | chosen root | d | effective roots | roots with users | candidate users | attempt-schedule users | phase-schedule users | found paths | true paths | false paths | aborted roots | node visits | max root visits | seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in profiles:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.attempt,
                    str(row.chosen_root),
                    str(row.d),
                    str(row.effective_roots),
                    str(row.roots_with_users),
                    str(row.candidate_users),
                    str(row.attempt_sched_users),
                    str(row.phase_sched_users),
                    str(row.found_paths),
                    str(row.true_paths),
                    str(row.false_paths),
                    str(row.aborted_roots),
                    str(row.node_visits),
                    str(row.max_root_visits),
                    f"{row.seconds:.2f}",
                ]
            )
            + " |"
        )

    total_false = sum(row.false_paths for row in profiles)
    total_aborted = sum(row.aborted_roots for row in profiles)
    total_visits = sum(row.node_visits for row in profiles)
    lines.extend(
        [
            "",
            "## Readout",
            "",
            f"- false paths observed across profiled attempts: `{total_false}`",
            f"- aborted roots across profiled attempts: `{total_aborted}`",
            f"- total node visits: `{total_visits}`",
            "",
            "Positive aborted-root counts mean this is a capped runtime profile, not an exact full-decoder PDP/PHP run.  False paths are still useful PHP diagnostics because any completed false final-valid path would appear in the `false paths` column.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--K", type=int, default=30)
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--M", type=int, default=3)
    parser.add_argument("--pe", type=float, default=0.1)
    parser.add_argument("--phase", type=int, default=3, choices=(2, 3))
    parser.add_argument("--seed", type=int, default=6310)
    parser.add_argument("--max-nodes-per-root", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=Path("research/uace_phase3_root_profile.md"))
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
