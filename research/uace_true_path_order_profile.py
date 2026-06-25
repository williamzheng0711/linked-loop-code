#!/usr/bin/env python3
"""Profile where the true path sits in the decoder child order.

For a schedule-success user, the true path exists.  Full phase-III root sweep
can still be slow because the current first-valid DFS may place the true child
after many parity-consistent false children.  This script follows only the
true prefix and records, at each section, how many children the decoder would
generate and which child is the true continuation.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from uace_bound_explorer import format_probability
from uace_empirical_probe import uace_channel
from uace_fast_empirical_probe import load_playground, rotate_decoder_state
from uace_schedule_bound import attempt_succeeds


@dataclass(frozen=True)
class StepRow:
    attempt: str
    user: int
    section: int
    erased: bool
    children: int
    true_index: int
    prior_siblings: int


@dataclass(frozen=True)
class UserRow:
    attempt: str
    user: int
    root_row: int
    erasure_weight: int
    final_valid: bool
    max_children: int
    max_true_index: int
    total_prior_siblings: int
    first_delayed_section: int
    log10_prefix_work: float


def symbol_from_bits(bits: np.ndarray) -> int:
    return int(np.asarray(bits, dtype=int) @ (1 << np.arange(bits.shape[0] - 1, -1, -1)))


def build_symbol_row_lookup(grand: np.ndarray, length: int, j: int) -> list[dict[int, int]]:
    lookups: list[dict[int, int]] = []
    for section in range(length):
        current: dict[int, int] = {}
        for row in np.flatnonzero(grand[:, section * j] != -1):
            symbol = symbol_from_bits(grand[row, section * j : (section + 1) * j])
            current[symbol] = int(row)
        lookups.append(current)
    return lookups


def attempt_specs(length: int) -> list[tuple[str, int, int, list[int] | None, set[int]]]:
    return [
        ("phase-III root 0", 0, 2, None, set()),
        ("phase-III root 6", 6 % length, 2, [6], {(-6) % length}),
        ("phase-III root 10", 10 % length, 2, [6, 10], {(-6) % length, (-10) % length}),
    ]


def profile_attempt(
    *,
    name: str,
    root: int,
    d: int,
    chosen_roots: list[int] | None,
    erasure_slot: set[int],
    args: argparse.Namespace,
    static_repo,
    gu,
    LLC,
    tx_symbols: np.ndarray,
    erasure_masks: np.ndarray,
    grand_list: np.ndarray,
    message_lens: np.ndarray,
    parity_lens: np.ndarray,
    gis: np.ndarray,
    columns_index: np.ndarray,
    sub_g_invs: np.ndarray,
    gijs_original: dict,
) -> tuple[list[UserRow], list[StepRow]]:
    (
        chosen_root,
        _rot_erasure_slot,
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
    assert chosen_root == root

    deciders_cache, avail_savers_cache = gu.build_decoder_lookup(args.L, args.M)
    solve_cache = gu.build_solve_cache(args.L, args.M, rot_columns, rot_invs, rot_gis)
    valid_ks_by_section = [np.flatnonzero(grand[:, section * static_repo.J] != -1) for section in range(args.L)]
    parity_lookup_by_section = gu.build_parity_lookup_by_section(grand, args.L, msg_lens, valid_ks_by_section)
    parity_cache = gu.build_parity_cache(grand, args.L, args.M, msg_lens, gijs)
    symbol_rows = build_symbol_row_lookup(grand, args.L, static_repo.J)

    user_rows: list[UserRow] = []
    step_rows: list[StepRow] = []
    for user in range(args.K):
        mask = int(erasure_masks[user])
        if (mask >> root) & 1:
            continue
        if not attempt_succeeds(
            mask,
            length=args.L,
            memory=args.M,
            root=root,
            d=d,
            erasure_slot=erasure_slot,
            message_lens=message_lens,
            gijs=gijs_original,
        ):
            continue

        root_symbol = int(tx_symbols[user, root])
        root_row = symbol_rows[0].get(root_symbol)
        if root_row is None:
            continue

        path = LLC.GLinkedLoop([root_row], msg_lens)
        max_children = 0
        max_true_index = 0
        total_prior = 0
        first_delayed = -1
        log10_prefix_work = 0.0
        missing_true = False

        for section in range(1, args.L):
            original_section = (root + section) % args.L
            erased = bool((mask >> original_section) & 1)
            true_child = -1 if erased else symbol_rows[section].get(int(tx_symbols[user, original_section]))
            children = gu.Path_goes_section_l(
                section,
                path,
                d,
                grand,
                args.K,
                msg_lens,
                par_lens,
                args.L,
                args.M,
                rot_gis,
                gijs,
                rot_columns,
                rot_invs,
                list(erasure_slot),
                valid_ks_by_section=valid_ks_by_section,
                parity_cache=parity_cache,
                deciders_cache=deciders_cache,
                avail_savers_cache=avail_savers_cache,
                parity_lookup_by_section=parity_lookup_by_section,
                solve_cache=solve_cache,
            )
            true_index = -1
            for idx, child in enumerate(children):
                if child.get_path()[-1] == true_child:
                    true_index = idx
                    break
            if true_index < 0:
                missing_true = True
                break
            prior = true_index
            max_children = max(max_children, len(children))
            max_true_index = max(max_true_index, true_index)
            total_prior += prior
            if prior > 0 and first_delayed < 0:
                first_delayed = section
            log10_prefix_work += math.log10(max(1, len(children)))
            step_rows.append(
                StepRow(
                    attempt=name,
                    user=user,
                    section=section,
                    erased=erased,
                    children=len(children),
                    true_index=true_index,
                    prior_siblings=prior,
                )
            )
            path = children[true_index]

        final_valid = False
        if not missing_true:
            final_valid = bool(
                gu.final_parity_check_oop(
                    path,
                    grand,
                    msg_lens,
                    par_lens,
                    args.L,
                    gijs,
                    args.M,
                    parity_cache=parity_cache,
                    deciders_cache=deciders_cache,
                )
            )
        user_rows.append(
            UserRow(
                attempt=name,
                user=user,
                root_row=root_row,
                erasure_weight=int(mask.bit_count()),
                final_valid=final_valid,
                max_children=max_children,
                max_true_index=max_true_index,
                total_prior_siblings=total_prior,
                first_delayed_section=first_delayed,
                log10_prefix_work=log10_prefix_work,
            )
        )

    return user_rows, step_rows


def summarize_user_rows(user_rows: list[UserRow]) -> list[str]:
    lines = [
        "| attempt | users | final valid | max children | max true index | mean prior siblings | max prior siblings | max log10 prefix work |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for attempt in sorted({row.attempt for row in user_rows}):
        rows = [row for row in user_rows if row.attempt == attempt]
        total_prior = [row.total_prior_siblings for row in rows]
        lines.append(
            "| "
            + " | ".join(
                [
                    attempt,
                    str(len(rows)),
                    str(sum(row.final_valid for row in rows)),
                    str(max((row.max_children for row in rows), default=0)),
                    str(max((row.max_true_index for row in rows), default=0)),
                    f"{(sum(total_prior) / len(total_prior)) if total_prior else 0:.2f}",
                    str(max(total_prior, default=0)),
                    f"{max((row.log10_prefix_work for row in rows), default=0.0):.2f}",
                ]
            )
            + " |"
        )
    return lines


def summarize_step_rows(step_rows: list[StepRow]) -> list[str]:
    lines = [
        "| attempt | section | samples | mean children | max children | mean true index | max true index | delayed fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    keys = sorted({(row.attempt, row.section) for row in step_rows})
    for attempt, section in keys:
        rows = [row for row in step_rows if row.attempt == attempt and row.section == section]
        delayed = sum(row.true_index > 0 for row in rows) / len(rows)
        lines.append(
            "| "
            + " | ".join(
                [
                    attempt,
                    str(section),
                    str(len(rows)),
                    f"{sum(row.children for row in rows) / len(rows):.2f}",
                    str(max(row.children for row in rows)),
                    f"{sum(row.true_index for row in rows) / len(rows):.2f}",
                    str(max(row.true_index for row in rows)),
                    format_probability(delayed),
                ]
            )
            + " |"
        )
    return lines


def build_report(args: argparse.Namespace) -> str:
    static_repo, general_lib, gu, LLC, abch_utils = load_playground()
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    message_lens, parity_lens = static_repo.get_allocation(args.L)
    gis, columns_index, sub_g_invs = static_repo.get_G_info(args.L, args.M, message_lens, parity_lens, seed=args.seed)
    gijs = static_repo.partition_Gs(args.L, args.M, parity_lens, gis)

    tx_bits = rng.integers(0, 2, size=(args.K, static_repo.B), dtype=int)
    tx_cdwds = general_lib.encode(tx_bits, args.K, args.L, args.L * static_repo.J, args.M, message_lens, parity_lens, gijs)
    tx_symbols = abch_utils.binary_to_symbol(tx_cdwds, args.L, args.K)
    rx_symbols, _erasure_counts, erasure_masks = uace_channel(tx_symbols, args.pe, rng)
    grand_list = abch_utils.symbol_to_binary(args.K, args.L, rx_symbols)

    all_user_rows: list[UserRow] = []
    all_step_rows: list[StepRow] = []
    for name, root, d, chosen_roots, erasure_slot in attempt_specs(args.L):
        if args.only_attempt and name != args.only_attempt:
            continue
        user_rows, step_rows = profile_attempt(
            name=name,
            root=root,
            d=d,
            chosen_roots=chosen_roots,
            erasure_slot=erasure_slot,
            args=args,
            static_repo=static_repo,
            gu=gu,
            LLC=LLC,
            tx_symbols=tx_symbols,
            erasure_masks=erasure_masks,
            grand_list=grand_list,
            message_lens=message_lens,
            parity_lens=parity_lens,
            gis=gis,
            columns_index=columns_index,
            sub_g_invs=sub_g_invs,
            gijs_original=gijs,
        )
        all_user_rows.extend(user_rows)
        all_step_rows.extend(step_rows)

    lines = [
        "# UACE True-Path Order Profile",
        "",
        "This report is generated by `research/uace_true_path_order_profile.py`.",
        "",
        f"- `K = {args.K}`",
        f"- `L = {args.L}`",
        f"- `M = {args.M}`",
        f"- `pe = {args.pe}`",
        f"- seed: `{args.seed}`",
        f"- only attempt: `{args.only_attempt or 'all phase-III attempts'}`",
        "",
        "## Per-Attempt Summary",
        "",
    ]
    lines.extend(summarize_user_rows(all_user_rows))
    lines.extend(["", "## Per-Section Summary", ""])
    lines.extend(summarize_step_rows(all_step_rows))
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- `true index` is zero-based in the child list returned by the current decoder expansion.  A positive value means the DFS explores at least that many sibling branches before the true continuation.",
            "- `log10 prefix work` is the sum of log10 child counts along the true prefix.  It is not an exact runtime, but it exposes where true paths are buried under broad child lists.",
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
    parser.add_argument("--seed", type=int, default=6310)
    parser.add_argument("--only-attempt", default="")
    parser.add_argument("--output", type=Path, default=Path("research/uace_true_path_order_profile.md"))
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
