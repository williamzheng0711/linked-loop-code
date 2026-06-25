#!/usr/bin/env python3
"""Erasure-only recoverability bound for the current LLC phase/root schedule.

This script sits between the ideal rank-peeling bound and full simulation.  It
classifies erasure masks under an abstraction of the actual playground
phase2plus root schedule, assuming no symbol collisions and no false paths.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from uace_bound_explorer import (
    bad_mask_probability,
    bernoulli_mask_probability,
    cantor_pairing,
    format_probability,
    gf2_rank,
    geometric_unrecoverable_probability,
    load_repo_code_matrices,
    popcount,
    rank_peeling_bad_masks,
    saver_sections,
    who_decides_p_sec,
)


DEFAULT_PES = (0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2)


def bit_is_set(mask: int, idx: int) -> bool:
    return ((mask >> idx) & 1) == 1


def rotate_mask(mask: int, length: int, root: int) -> int:
    result = 0
    for rotated in range(length):
        original = (root + rotated) % length
        if bit_is_set(mask, original):
            result |= 1 << rotated
    return result


def original_section(root: int, rotated_section: int, length: int) -> int:
    return (root + rotated_section) % length


def saver_available_by_decoder(saver: int, current_section: int, length: int, memory: int) -> bool:
    return all(((saver - offset) % length) <= current_section for offset in range(memory + 1))


def correctly_recoverable_by_decoder_step(
    *,
    root: int,
    length: int,
    memory: int,
    erased_rot: set[int],
    recovered_rot: set[int],
    carried_erasures: set[int],
    current_section: int,
    message_lens: np.ndarray,
    gijs: dict,
) -> set[int]:
    """Model the recovery step actually attempted by `Path_goes_entry_k`.

    The current implementation does not run arbitrary peeling after every
    section.  When recovery is triggered, it loops over carried but unsolved
    erasures and solves each from saver parities already visible to the current
    scan position.  This helper keeps only recoveries that are correct under a
    collision-free erasure abstraction.
    """
    for lost_rot in sorted(carried_erasures - recovered_rot):
        actual_avail_savers = [
            saver_rot
            for saver_rot in saver_sections(length, lost_rot, memory)
            if saver_available_by_decoder(saver_rot, current_section, length, memory)
        ]
        if not actual_avail_savers:
            continue

        def saver_block_is_clean(saver_rot: int) -> bool:
            if saver_rot in erased_rot:
                return False
            deciders_rot = who_decides_p_sec(length, saver_rot, memory)
            return all(
                decider == lost_rot or decider not in carried_erasures
                for decider in deciders_rot
            )

        # In the L=16,M=3 repository profile, `columns_index` is always
        # 0..7, so `solveInfoBack` uses the first visible saver block even if
        # a later visible block would be cleaner.  This intentionally models
        # the current decoder, not the ideal LLC peeling rule.
        saver_rot = actual_avail_savers[0]
        if not saver_block_is_clean(saver_rot):
            continue

        # When all M saver blocks are visible, the implementation verifies the
        # recovered answer against the full concatenated vector.  Any dirty
        # saver block in that full vector makes the current implementation
        # reject, even if the selected first block alone would determine the
        # erased information.
        if len(actual_avail_savers) == memory and not all(
            saver_block_is_clean(item) for item in actual_avail_savers
        ):
            continue

        lost_orig = original_section(root, lost_rot, length)
        saver_orig = original_section(root, saver_rot, length)
        transfer = np.array(gijs[cantor_pairing(lost_orig, saver_orig)], dtype=np.uint8)
        if gf2_rank(transfer) >= int(message_lens[lost_orig]):
            recovered_rot.add(lost_rot)
    return recovered_rot


def attempt_succeeds(
    mask: int,
    *,
    length: int,
    memory: int,
    root: int,
    d: int,
    erasure_slot: set[int],
    message_lens: np.ndarray,
    gijs: dict,
) -> bool:
    rotated_mask = rotate_mask(mask, length, root)
    erased_rot = {idx for idx in range(length) if bit_is_set(rotated_mask, idx)}
    if 0 in erased_rot:
        return False
    if not erased_rot:
        return True

    recovered_rot: set[int] = set()
    carried_erasures: set[int] = set()

    for section in range(1, length):
        if section in erased_rot:
            can_add_erasure = (
                len(carried_erasures) < d
                and (d - len(erasure_slot) > 0 or section in erasure_slot)
                and carried_erasures.issubset(recovered_rot)
            )
            if not can_add_erasure:
                return False
            carried_erasures.add(section)
            if section == length - 1:
                recovered_rot = correctly_recoverable_by_decoder_step(
                    root=root,
                    length=length,
                    memory=memory,
                    erased_rot=erased_rot,
                    recovered_rot=recovered_rot,
                    carried_erasures=carried_erasures,
                    current_section=section,
                    message_lens=message_lens,
                    gijs=gijs,
                )
            continue

        if section >= memory:
            can_decide_current_parity = all(
                decider not in carried_erasures or decider in recovered_rot
                for decider in who_decides_p_sec(length, section, memory)
            )
            if not can_decide_current_parity or section == length - 1:
                recovered_rot = correctly_recoverable_by_decoder_step(
                    root=root,
                    length=length,
                    memory=memory,
                    erased_rot=erased_rot,
                    recovered_rot=recovered_rot,
                    carried_erasures=carried_erasures,
                    current_section=section,
                    message_lens=message_lens,
                    gijs=gijs,
                )

    return erased_rot.issubset(recovered_rot)


def schedule_attempts(length: int, phase: int) -> list[tuple[str, int, int, set[int]]]:
    attempts: list[tuple[str, int, int, set[int]]] = [("phase-I root 0", 0, 0, set())]
    if phase >= 2:
        attempts.extend(
            [
                ("phase-II root 0", 0, 1, set()),
                ("phase-II root 8", 8 % length, 1, {(-8) % length}),
            ]
        )
    if phase >= 3:
        attempts.extend(
            [
                ("phase-III root 0", 0, 2, set()),
                ("phase-III root 6", 6 % length, 2, {(-6) % length}),
                ("phase-III root 10", 10 % length, 2, {(-6) % length, (-10) % length}),
            ]
        )
    return attempts


def schedule_succeeds(
    mask: int,
    *,
    length: int,
    memory: int,
    phase: int,
    message_lens: np.ndarray,
    gijs: dict,
) -> bool:
    for _name, root, d, erasure_slot in schedule_attempts(length, phase):
        if d == 0:
            if mask == 0:
                return True
            continue
        if attempt_succeeds(
            mask,
            length=length,
            memory=memory,
            root=root,
            d=d,
            erasure_slot=erasure_slot,
            message_lens=message_lens,
            gijs=gijs,
        ):
            return True
    return False


def schedule_bad_masks(length: int, memory: int, phase: int, seed: int) -> list[int]:
    message_lens, _parity_lens, gijs = load_repo_code_matrices(length, memory, seed)
    bad = []
    for mask in range(1 << length):
        if not schedule_succeeds(
            mask,
            length=length,
            memory=memory,
            phase=phase,
            message_lens=message_lens,
            gijs=gijs,
        ):
            bad.append(mask)
    return bad


def mask_to_sections(mask: int, length: int) -> tuple[int, ...]:
    return tuple(idx for idx in range(length) if bit_is_set(mask, idx))


def summarize_bad_masks(bad_masks: list[int], length: int, limit: int = 10) -> str:
    by_weight: dict[int, int] = {}
    for mask in bad_masks:
        weight = popcount(mask)
        by_weight[weight] = by_weight.get(weight, 0) + 1

    examples = []
    for mask in sorted(bad_masks, key=lambda item: (popcount(item), item))[:limit]:
        examples.append(str(mask_to_sections(mask, length)))

    parts = [
        "bad mask counts by erasure weight: "
        + ", ".join(f"{weight}:{count}" for weight, count in sorted(by_weight.items()))
    ]
    if examples:
        parts.append("smallest examples: " + "; ".join(examples))
    return "\n".join(parts)


def build_report(args: argparse.Namespace) -> str:
    practical_bad = schedule_bad_masks(args.L, args.M, args.phase, args.seed)
    ideal_bad = rank_peeling_bad_masks(args.L, args.M, args.seed)

    lines = [
        "# UACE Erasure-Only Schedule Bound",
        "",
        "This report is generated by `research/uace_schedule_bound.py`.",
        "",
        "Regime:",
        "",
        f"- `L = {args.L}`",
        f"- `M = {args.M}`",
        f"- phase schedule up to: `{args.phase}`",
        f"- matrix seed: `{args.seed}`",
        "",
        "Schedule attempts:",
        "",
    ]
    for name, root, d, erasure_slot in schedule_attempts(args.L, args.phase):
        lines.append(f"- {name}: root `{root}`, d `{d}`, erasure_slot `{sorted(erasure_slot)}`")

    lines.extend(
        [
            "",
            summarize_bad_masks(practical_bad, args.L),
            "",
            "| pe | practical schedule UE | ideal rank-peeling UE | TCom geometric UE |",
            "|---:|---:|---:|---:|",
        ]
    )
    for pe in args.pes:
        practical = bad_mask_probability(practical_bad, args.L, pe)
        ideal = bad_mask_probability(ideal_bad, args.L, pe)
        geometric = geometric_unrecoverable_probability(args.L, args.M, pe)
        lines.append(
            f"| {pe:.3f} | {format_probability(practical)} | "
            f"{format_probability(ideal)} | {format_probability(geometric)} |"
        )

    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `practical schedule UE` is a collision-free PDP proxy for the current root schedule and sequential NaN handling.",
            "- The gap from `ideal rank-peeling UE` estimates the practical root/order penalty before considering symbol collisions or false paths.",
            "- The gap from empirical PDP then points to remaining effects such as path selection, list behavior, SIC, and collisions.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--M", type=int, default=3)
    parser.add_argument("--phase", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pes", type=float, nargs="+", default=list(DEFAULT_PES))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    content = build_report(args)
    if args.output is None:
        print(content)
    else:
        args.output.write_text(content + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
