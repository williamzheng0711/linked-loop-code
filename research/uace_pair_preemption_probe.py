#!/usr/bin/env python3
"""Pair-level preemption probe for LLC/UACE identity profiles.

The identity-profile bounds sum over many profiles.  For decoder error, those
profiles are highly dependent: for one tagged/alternate user pair, what matters
is whether *any* final-valid profile appears before the true path in the current
DFS row order.  This script estimates that pair event by Monte Carlo over two
random encoded users while exhaustively enumerating canonical two-color
profiles for selected accepted erasure masks.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from uace_bound_explorer import bad_mask_probability, cantor_pairing, format_probability
from uace_ordered_profile_bound import (
    accepted_rotated_masks,
    available_savers,
    filter_masks,
    profiles_for,
    rotate_code,
    saver_sections,
    section_offsets,
    who_decides_p_sec,
)
from uace_schedule_bound import mask_to_sections, schedule_attempts, schedule_bad_masks


DEFAULT_KS = (30, 40, 100)
DEFAULT_PES = (0.1, 0.2, 0.3)


@dataclass(frozen=True)
class AttemptMaskResult:
    attempt: str
    mask: int
    checked_pairs: int
    pair_preemptions: int
    valid_profile_pairs: int
    profiles_checked_per_pair: int


def load_playground():
    repo_root = Path(__file__).resolve().parents[1]
    playground = repo_root / "playground"
    sys.path.insert(0, str(playground))
    import general_lib  # type: ignore
    import static_repo  # type: ignore

    return static_repo, general_lib


def rotate_codewords(codewords: np.ndarray, root: int, length: int, section_bits: int) -> np.ndarray:
    rotated = codewords.copy()
    rotated[:, range(length * section_bits)] = codewords[
        :, np.mod(np.arange(root * section_bits, root * section_bits + length * section_bits), length * section_bits)
    ]
    return rotated


def symbol_value(codeword: np.ndarray, section: int, section_bits: int) -> int:
    bits = codeword[section * section_bits : (section + 1) * section_bits]
    return int(np.asarray(bits, dtype=int) @ (2 ** np.arange(section_bits - 1, -1, -1)))


def candidate_preempts_true(profile: tuple[int, ...], known_sections: tuple[int, ...], codewords: np.ndarray, section_bits: int) -> bool:
    for section, color in zip(known_sections, profile):
        if color == 0:
            continue
        alt = symbol_value(codewords[1], section, section_bits)
        tag = symbol_value(codewords[0], section, section_bits)
        if alt == tag:
            continue
        return alt < tag
    return False


def observed_parity(codewords: np.ndarray, color: int, section: int, message_lens: np.ndarray, section_bits: int) -> np.ndarray:
    start = section * section_bits + int(message_lens[section])
    stop = (section + 1) * section_bits
    return np.asarray(codewords[color, start:stop], dtype=np.uint8)


def info_bits(codewords: np.ndarray, color: int, section: int, message_lens: np.ndarray, section_bits: int) -> np.ndarray:
    start = section * section_bits
    stop = start + int(message_lens[section])
    return np.asarray(codewords[color, start:stop], dtype=np.uint8)


def candidate_info(
    section: int,
    *,
    color_by_section: dict[int, int],
    recovered: dict[int, np.ndarray],
    codewords: np.ndarray,
    message_lens: np.ndarray,
    section_bits: int,
) -> np.ndarray | None:
    if section in recovered:
        return recovered[section]
    if section in color_by_section:
        return info_bits(codewords, color_by_section[section], section, message_lens, section_bits)
    return None


def candidate_parity(
    section: int,
    *,
    length: int,
    memory: int,
    color_by_section: dict[int, int],
    recovered: dict[int, np.ndarray],
    codewords: np.ndarray,
    message_lens: np.ndarray,
    parity_lens: np.ndarray,
    gijs: dict,
    section_bits: int,
) -> np.ndarray | None:
    parity = np.zeros(int(parity_lens[section]), dtype=np.uint8)
    for decider in who_decides_p_sec(length, section, memory):
        info = candidate_info(
            decider,
            color_by_section=color_by_section,
            recovered=recovered,
            codewords=codewords,
            message_lens=message_lens,
            section_bits=section_bits,
        )
        if info is None:
            return None
        parity ^= np.asarray(info @ np.asarray(gijs[cantor_pairing(decider, section)], dtype=np.uint8), dtype=np.uint8) & 1
    return parity


def solve_info(
    known: np.ndarray,
    *,
    lost_section: int,
    solve_data: dict[int, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    columns, inv = solve_data[lost_section]
    return np.asarray(known[columns] @ inv, dtype=np.uint8) & 1


def recover_unsolved_concrete(
    *,
    focus_path: list[int],
    current_section: int,
    color_by_section: dict[int, int],
    recovered: dict[int, np.ndarray],
    codewords: np.ndarray,
    length: int,
    memory: int,
    message_lens: np.ndarray,
    parity_lens: np.ndarray,
    gis: np.ndarray,
    gijs: dict,
    solve_data: dict[int, tuple[np.ndarray, np.ndarray]],
    section_bits: int,
) -> bool:
    losts = [idx for idx, item in enumerate(focus_path) if item < 0]
    for lost in [item for item in losts if item not in recovered]:
        savers = available_savers(length, memory, lost, current_section)
        if not savers:
            return False
        if sum(int(parity_lens[saver]) for saver in savers) < int(message_lens[lost]):
            continue
        known_vectors = []
        for saver in savers:
            if saver not in color_by_section:
                return False
            minuend = observed_parity(codewords, color_by_section[saver], saver, message_lens, section_bits)
            subtrahend = np.zeros(int(parity_lens[saver]), dtype=np.uint8)
            for decider in who_decides_p_sec(length, saver, memory):
                if decider == lost:
                    continue
                info = candidate_info(
                    decider,
                    color_by_section=color_by_section,
                    recovered=recovered,
                    codewords=codewords,
                    message_lens=message_lens,
                    section_bits=section_bits,
                )
                if info is None:
                    return False
                subtrahend ^= np.asarray(info @ np.asarray(gijs[cantor_pairing(decider, saver)], dtype=np.uint8), dtype=np.uint8) & 1
            known_vectors.append(minuend ^ subtrahend)
        known = np.concatenate(known_vectors)
        answer = solve_info(known, lost_section=lost, solve_data=solve_data)
        if len(savers) == memory:
            if not np.array_equal(np.asarray(answer @ np.asarray(gis[lost], dtype=np.uint8), dtype=np.uint8) & 1, known):
                return False
        recovered[lost] = answer
    return True


def ordered_profile_valid_concrete(
    *,
    mask: int,
    profile: tuple[int, ...],
    codewords: np.ndarray,
    length: int,
    memory: int,
    d: int,
    erasure_slot: set[int],
    message_lens: np.ndarray,
    parity_lens: np.ndarray,
    gis: np.ndarray,
    gijs: dict,
    solve_data: dict[int, tuple[np.ndarray, np.ndarray]],
    section_bits: int,
) -> tuple[bool, bool]:
    erased = {section for section in range(length) if (mask >> section) & 1}
    known_sections = tuple(section for section in range(length) if section not in erased)
    color_by_section = dict(zip(known_sections, profile))
    focus_path = [0]
    recovered: dict[int, np.ndarray] = {}

    for section in range(1, length):
        if section in erased:
            can_add = (
                focus_path.count(-1) < d
                and (d - len(erasure_slot) > 0 or section in erasure_slot)
                and all(idx in recovered for idx, item in enumerate(focus_path) if item < 0)
            )
            if not can_add:
                return False, False
            focus_path.append(-1)
            if section == length - 1:
                if not recover_unsolved_concrete(
                    focus_path=focus_path,
                    current_section=section,
                    color_by_section=color_by_section,
                    recovered=recovered,
                    codewords=codewords,
                    length=length,
                    memory=memory,
                    message_lens=message_lens,
                    parity_lens=parity_lens,
                    gis=gis,
                    gijs=gijs,
                    solve_data=solve_data,
                    section_bits=section_bits,
                ):
                    return False, False
            continue

        if section < memory:
            focus_path.append(color_by_section[section])
            continue

        can_decide = section != length - 1
        if can_decide:
            for decider in who_decides_p_sec(length, section, memory):
                if decider < len(focus_path) and focus_path[decider] == -1 and decider not in recovered:
                    can_decide = False
                    break
        if can_decide:
            cand = candidate_parity(
                section,
                length=length,
                memory=memory,
                color_by_section=color_by_section,
                recovered=recovered,
                codewords=codewords,
                message_lens=message_lens,
                parity_lens=parity_lens,
                gijs=gijs,
                section_bits=section_bits,
            )
            if cand is None:
                return False, False
            obs = observed_parity(codewords, color_by_section[section], section, message_lens, section_bits)
            if not np.array_equal(cand, obs):
                return False, False
        else:
            if not recover_unsolved_concrete(
                focus_path=focus_path + [color_by_section[section]],
                current_section=section,
                color_by_section=color_by_section,
                recovered=recovered,
                codewords=codewords,
                length=length,
                memory=memory,
                message_lens=message_lens,
                parity_lens=parity_lens,
                gis=gis,
                gijs=gijs,
                solve_data=solve_data,
                section_bits=section_bits,
            ):
                return False, False
        focus_path.append(color_by_section[section])

    for section in known_sections:
        cand = candidate_parity(
            section,
            length=length,
            memory=memory,
            color_by_section=color_by_section,
            recovered=recovered,
            codewords=codewords,
            message_lens=message_lens,
            parity_lens=parity_lens,
            gijs=gijs,
            section_bits=section_bits,
        )
        if cand is None:
            return False, False
        obs = observed_parity(codewords, color_by_section[section], section, message_lens, section_bits)
        if not np.array_equal(cand, obs):
            return False, False

    decoded_parts = []
    tagged_parts = []
    for section in range(length):
        info = candidate_info(
            section,
            color_by_section=color_by_section,
            recovered=recovered,
            codewords=codewords,
            message_lens=message_lens,
            section_bits=section_bits,
        )
        if info is None:
            return False, False
        decoded_parts.append(info)
        tagged_parts.append(info_bits(codewords, 0, section, message_lens, section_bits))
    return True, not np.array_equal(np.concatenate(decoded_parts), np.concatenate(tagged_parts))


def run_attempt_mask(
    *,
    args: argparse.Namespace,
    attempt: str,
    root: int,
    d: int,
    erasure_slot: set[int],
    mask: int,
    rng: np.random.Generator,
) -> AttemptMaskResult:
    static_repo, general_lib = load_playground()
    message_lens, parity_lens, gis, gijs, solve_data = rotate_code(args.L, args.M, args.seed, root)
    profiles = profiles_for(args.L - mask.bit_count(), 2)
    random.seed(args.seed)
    raw_msg_lens, raw_par_lens = static_repo.get_allocation(args.L)
    raw_gis, _columns, _invs = static_repo.get_G_info(args.L, args.M, raw_msg_lens, raw_par_lens, seed=args.seed)
    raw_gijs = static_repo.partition_Gs(args.L, args.M, raw_par_lens, raw_gis)
    preemptions = 0
    valid_profile_pairs = 0
    for _trial in range(args.pair_trials):
        tx_bits = rng.integers(0, 2, size=(2, static_repo.B), dtype=int)
        # Encode with the unrotated code, then rotate the codeword sections for
        # the chosen-root decoder.
        codewords = general_lib.encode(tx_bits, 2, args.L, args.L * static_repo.J, args.M, raw_msg_lens, raw_par_lens, raw_gijs)
        codewords = rotate_codewords(codewords, root, args.L, static_repo.J)
        known_sections = tuple(section for section in range(args.L) if ((mask >> section) & 1) == 0)
        pair_has_preemption = False
        pair_has_valid = False
        for profile in profiles:
            if not candidate_preempts_true(profile, known_sections, codewords, static_repo.J):
                continue
            valid, is_false = ordered_profile_valid_concrete(
                mask=mask,
                profile=profile,
                codewords=codewords,
                length=args.L,
                memory=args.M,
                d=d,
                erasure_slot=erasure_slot,
                message_lens=message_lens,
                parity_lens=parity_lens,
                gis=gis,
                gijs=gijs,
                solve_data=solve_data,
                section_bits=static_repo.J,
            )
            if valid:
                pair_has_valid = True
                if is_false:
                    pair_has_preemption = True
                    break
        valid_profile_pairs += int(pair_has_valid)
        preemptions += int(pair_has_preemption)
    return AttemptMaskResult(
        attempt=attempt,
        mask=mask,
        checked_pairs=args.pair_trials,
        pair_preemptions=preemptions,
        valid_profile_pairs=valid_profile_pairs,
        profiles_checked_per_pair=len(profiles),
    )


def build_report(args: argparse.Namespace) -> str:
    rng = np.random.default_rng(args.seed)
    results: list[AttemptMaskResult] = []
    for name, root, d, erasure_slot in schedule_attempts(args.L, args.phase):
        if args.only_attempt and args.only_attempt not in name:
            continue
        masks = accepted_rotated_masks(
            length=args.L,
            memory=args.M,
            seed=args.seed,
            root=root,
            d=d,
            erasure_slot=erasure_slot,
            weights=set(args.weights),
        )
        masks = filter_masks(args, masks)
        for mask in masks:
            results.append(
                run_attempt_mask(
                    args=args,
                    attempt=name,
                    root=root,
                    d=d,
                    erasure_slot=erasure_slot,
                    mask=mask,
                    rng=rng,
                )
            )

    total_pairs = sum(item.checked_pairs for item in results)
    total_preempt = sum(item.pair_preemptions for item in results)
    avg_pair_preempt = total_preempt / total_pairs if total_pairs else 0.0
    if total_preempt == 0 and total_pairs > 0:
        avg_pair_preempt_upper95 = 1.0 - (0.05 ** (1.0 / total_pairs))
    else:
        avg_pair_preempt_upper95 = avg_pair_preempt
    schedule_bad = schedule_bad_masks(args.L, args.M, args.phase, args.seed)

    lines = [
        "# UACE Pair-Level Preemption Probe",
        "",
        "This report is generated by `research/uace_pair_preemption_probe.py`.",
        "",
        f"- `L = {args.L}`",
        f"- `M = {args.M}`",
        f"- phase: `{args.phase}`",
        f"- weights: `{args.weights}`",
        f"- pair trials per mask: `{args.pair_trials}`",
        f"- gap representatives: `{args.gap_representatives}`",
        "",
        "This Monte Carlo estimate groups all two-color profiles for one tagged/alternate user pair into one event: whether any valid false profile preempts the true path in row order.",
        "",
        "| attempt | mask sections | checked pairs | pair preemptions | valid-profile pairs | profiles per pair |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item.attempt} | {mask_to_sections(item.mask, args.L)} | {item.checked_pairs} | "
            f"{item.pair_preemptions} | {item.valid_profile_pairs} | {item.profiles_checked_per_pair} |"
        )

    lines.extend(
        [
            "",
            "## Load-Scale Projection",
            "",
            "| K | pe | schedule UE | avg pair preempt | avg pair preempt 95% upper | per-user pair-union scale | pair-union 95% upper |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for k in args.Ks:
        for pe in args.pes:
            schedule_ue = bad_mask_probability(schedule_bad, args.L, pe)
            pair_union = 1.0 - (1.0 - avg_pair_preempt) ** max(k - 1, 0)
            pair_union_upper = 1.0 - (1.0 - avg_pair_preempt_upper95) ** max(k - 1, 0)
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(k),
                        f"{pe:.3f}",
                        format_probability(schedule_ue),
                        format_probability(avg_pair_preempt),
                        format_probability(avg_pair_preempt_upper95),
                        format_probability(pair_union),
                        format_probability(pair_union_upper),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "Readout:",
            "",
            "- This is a Monte Carlo diagnostic, not a theorem.  Its value is to measure profile coalescence before designing an exact ordering DP.",
            "- If pair preemption is zero or very small while profile first moments are large, the missing theorem must bound the pair-level first-preempt event rather than individual profiles.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--M", type=int, default=3)
    parser.add_argument("--phase", type=int, default=3, choices=(2, 3))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights", type=int, nargs="+", default=[2])
    parser.add_argument("--only-attempt", type=str, default="phase-III root")
    parser.add_argument("--gap-representatives", action="store_true")
    parser.add_argument("--max-masks-per-attempt", type=int, default=0)
    parser.add_argument("--pair-trials", type=int, default=20)
    parser.add_argument("--Ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--pes", type=float, nargs="+", default=list(DEFAULT_PES))
    parser.add_argument("--output", type=Path, default=Path("research/uace_pair_preemption_probe.md"))
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
