#!/usr/bin/env python3
"""Symbolic pair-level preemption probe for LLC/UACE.

The profile first-moment reports count parity-valid identity profiles one by
one.  The actual decoder returns the first final-valid path in section-row
order, so the relevant two-user event is stronger:

    Does any valid mixed profile using one alternate user appear before the
    tagged user's true profile?

This script keeps the current decoder's ordered recovery equations, but
evaluates them as GF(2) bitsets over two random users' information bits.  It
also samples the alternate user's erasures, so a profile is available only on
sections where that alternate user's symbol is present in the A-list.

The output is a validation artifact, not a replacement for an analytic proof.
Its value is that it measures the pair-level union event after profile
coalescence, row-order preemption, and alternate-user erasures are included.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from uace_bound_explorer import bad_mask_probability, format_probability
from uace_ordered_profile_bound import (
    accepted_rotated_masks,
    candidate_info_forms,
    candidate_parity_forms,
    observed_parity_forms,
    profiles_for,
    recover_unsolved,
    rotate_code,
    section_offsets,
    user_section_forms,
    who_decides_p_sec,
)
from uace_schedule_bound import mask_to_sections, schedule_attempts, schedule_bad_masks


DEFAULT_KS = (30, 40, 100)
DEFAULT_PES = (0.1, 0.2, 0.3)


@dataclass(frozen=True)
class ProfileRecord:
    profile: tuple[int, ...]
    equations: tuple[int, ...]
    rank: int
    alt_positions: tuple[int, ...]
    alt_sections: tuple[int, ...]


@dataclass(frozen=True)
class MaskBundle:
    attempt: str
    mask: int
    multiplicity: int
    known_sections: tuple[int, ...]
    profiles: tuple[ProfileRecord, ...]
    first_moment: float
    min_rank: int


@dataclass(frozen=True)
class ProbeRow:
    attempt: str
    mask: int
    multiplicity: int
    pe: float
    trials: int
    pair_preemptions: int
    valid_pairs: int
    mean_valid_profiles: float
    first_moment: float
    min_rank: int


def row_basis(rows: list[int]) -> tuple[int, ...]:
    """Return an independent GF(2) row basis encoded as Python integers."""
    basis: dict[int, int] = {}
    for row in rows:
        value = row
        while value:
            pivot = value.bit_length() - 1
            if pivot in basis:
                value ^= basis[pivot]
            else:
                basis[pivot] = value
                break
    return tuple(basis[pivot] for pivot in sorted(basis, reverse=True))


def parity_eval(row: int, sample: int) -> int:
    return (row & sample).bit_count() & 1


def eval_forms(forms: tuple[int, ...], sample: int) -> tuple[int, ...]:
    return tuple(parity_eval(row, sample) for row in forms)


def eval_symbol(forms: tuple[int, ...], sample: int) -> int:
    value = 0
    width = len(forms)
    for idx, row in enumerate(forms):
        value |= parity_eval(row, sample) << (width - 1 - idx)
    return value


def ordered_profile_equations(
    *,
    mask: int,
    profile: tuple[int, ...],
    length: int,
    memory: int,
    d: int,
    erasure_slot: set[int],
    message_lens: np.ndarray,
    parity_lens: np.ndarray,
    gis: np.ndarray,
    gijs: dict,
    solve_data: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[int, ...] | None:
    """Return the ordered decoder equations for one fixed two-color profile."""
    erased = {section for section in range(length) if (mask >> section) & 1}
    known_sections = tuple(section for section in range(length) if section not in erased)
    if 0 in erased or len(profile) != len(known_sections):
        return None

    color_by_section = dict(zip(known_sections, profile))
    offsets, bits_per_user = section_offsets(message_lens)

    focus_path = [0]
    recovered: dict[int, tuple[int, ...]] = {}
    equations: list[int] = []

    for section in range(1, length):
        if section in erased:
            can_add_erasure = (
                focus_path.count(-1) < d
                and (d - len(erasure_slot) > 0 or section in erasure_slot)
                and all(idx in recovered for idx, item in enumerate(focus_path) if item < 0)
            )
            if not can_add_erasure:
                return None
            focus_path.append(-1)
            if section == length - 1:
                if not recover_unsolved(
                    focus_path=focus_path,
                    current_section=section,
                    color_by_section=color_by_section,
                    recovered=recovered,
                    equations=equations,
                    length=length,
                    memory=memory,
                    message_lens=message_lens,
                    parity_lens=parity_lens,
                    gis=gis,
                    gijs=gijs,
                    offsets=offsets,
                    bits_per_user=bits_per_user,
                    solve_data=solve_data,
                ):
                    return None
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
            candidate = candidate_parity_forms(
                section=section,
                length=length,
                memory=memory,
                color_by_section=color_by_section,
                recovered=recovered,
                message_lens=message_lens,
                parity_lens=parity_lens,
                gijs=gijs,
                offsets=offsets,
                bits_per_user=bits_per_user,
            )
            if candidate is None:
                return None
            observed = observed_parity_forms(
                color=color_by_section[section],
                section=section,
                length=length,
                memory=memory,
                message_lens=message_lens,
                gijs=gijs,
                offsets=offsets,
                bits_per_user=bits_per_user,
            )
            for lhs, rhs in zip(candidate, observed):
                eq = lhs ^ rhs
                if eq:
                    equations.append(eq)
        else:
            if not recover_unsolved(
                focus_path=focus_path + [color_by_section[section]],
                current_section=section,
                color_by_section=color_by_section,
                recovered=recovered,
                equations=equations,
                length=length,
                memory=memory,
                message_lens=message_lens,
                parity_lens=parity_lens,
                gis=gis,
                gijs=gijs,
                offsets=offsets,
                bits_per_user=bits_per_user,
                solve_data=solve_data,
            ):
                return None
        focus_path.append(color_by_section[section])

    for section in known_sections:
        candidate = candidate_parity_forms(
            section=section,
            length=length,
            memory=memory,
            color_by_section=color_by_section,
            recovered=recovered,
            message_lens=message_lens,
            parity_lens=parity_lens,
            gijs=gijs,
            offsets=offsets,
            bits_per_user=bits_per_user,
        )
        if candidate is None:
            return None
        observed = observed_parity_forms(
            color=color_by_section[section],
            section=section,
            length=length,
            memory=memory,
            message_lens=message_lens,
            gijs=gijs,
            offsets=offsets,
            bits_per_user=bits_per_user,
        )
        for lhs, rhs in zip(candidate, observed):
            eq = lhs ^ rhs
            if eq:
                equations.append(eq)

    return row_basis(equations)


def symbol_forms_by_color(
    *,
    length: int,
    memory: int,
    message_lens: np.ndarray,
    gijs: dict,
) -> dict[tuple[int, int], tuple[int, ...]]:
    offsets, bits_per_user = section_offsets(message_lens)
    forms = {}
    for color in (0, 1):
        for section in range(length):
            info = user_section_forms(
                color=color,
                section=section,
                section_offsets=offsets,
                bits_per_user=bits_per_user,
                message_lens=message_lens,
            )
            parity = observed_parity_forms(
                color=color,
                section=section,
                length=length,
                memory=memory,
                message_lens=message_lens,
                gijs=gijs,
                offsets=offsets,
                bits_per_user=bits_per_user,
            )
            forms[(color, section)] = tuple(info) + tuple(parity)
    return forms


def selected_masks_with_multiplicity(args: argparse.Namespace, masks: list[int]) -> list[tuple[int, int]]:
    if args.gap_representatives:
        groups: dict[tuple[int, int] | None, list[int]] = {}
        for mask in sorted(masks):
            sections = mask_to_sections(mask, args.L)
            key = None
            if len(sections) == 2:
                gap = (sections[1] - sections[0]) % args.L
                key = tuple(sorted((gap, args.L - gap)))
            groups.setdefault(key, []).append(mask)
        return [(items[0], len(items)) for items in groups.values()]
    if args.max_masks_per_attempt > 0:
        masks = masks[: args.max_masks_per_attempt]
    return [(mask, 1) for mask in masks]


def build_bundle(
    *,
    args: argparse.Namespace,
    attempt: str,
    root: int,
    d: int,
    erasure_slot: set[int],
    mask: int,
    multiplicity: int,
) -> MaskBundle:
    message_lens, parity_lens, gis, gijs, solve_data = rotate_code(args.L, args.M, args.seed, root)
    known_sections = tuple(section for section in range(args.L) if ((mask >> section) & 1) == 0)
    records = []
    first_moment = 0.0
    min_rank = 10**9
    for profile in profiles_for(len(known_sections), 2):
        alt_positions = tuple(pos for pos, color in enumerate(profile) if color == 1)
        if not alt_positions:
            continue
        equations = ordered_profile_equations(
            mask=mask,
            profile=profile,
            length=args.L,
            memory=args.M,
            d=d,
            erasure_slot=erasure_slot,
            message_lens=message_lens,
            parity_lens=parity_lens,
            gis=gis,
            gijs=gijs,
            solve_data=solve_data,
        )
        if equations is None:
            continue
        rank = len(equations)
        first_moment += 2.0 ** (-rank)
        min_rank = min(min_rank, rank)
        records.append(
            ProfileRecord(
                profile=profile,
                equations=equations,
                rank=rank,
                alt_positions=alt_positions,
                alt_sections=tuple(known_sections[pos] for pos in alt_positions),
            )
        )
    return MaskBundle(
        attempt=attempt,
        mask=mask,
        multiplicity=multiplicity,
        known_sections=known_sections,
        profiles=tuple(records),
        first_moment=first_moment,
        min_rank=min_rank if records else -1,
    )


def profile_preempts(
    *,
    record: ProfileRecord,
    known_sections: tuple[int, ...],
    symbols: dict[tuple[int, int], int],
) -> bool:
    for section, color in zip(known_sections, record.profile):
        if color == 0:
            continue
        alt_value = symbols[(1, section)]
        tag_value = symbols[(0, section)]
        if alt_value == tag_value:
            continue
        return alt_value < tag_value
    return False


def profile_available(record: ProfileRecord, alt_erased: int) -> bool:
    return all(((alt_erased >> section) & 1) == 0 for section in record.alt_sections)


def run_bundle_trials(
    *,
    args: argparse.Namespace,
    bundle: MaskBundle,
    symbol_forms: dict[tuple[int, int], tuple[int, ...]],
    pe: float,
    rng_bits: random.Random,
    rng_np: np.random.Generator,
) -> ProbeRow:
    total_bits = 2 * int(sum(rotate_code(args.L, args.M, args.seed, 0)[0]))
    preemptions = 0
    valid_pairs = 0
    valid_profiles = 0
    for _trial in range(args.pair_trials):
        sample = rng_bits.getrandbits(total_bits)
        alt_erased = 0
        for section, erased in enumerate(rng_np.random(args.L) < pe):
            if bool(erased):
                alt_erased |= 1 << section

        symbols = {}
        interesting_sections = set(bundle.known_sections)
        for section in interesting_sections:
            symbols[(0, section)] = eval_symbol(symbol_forms[(0, section)], sample)
            symbols[(1, section)] = eval_symbol(symbol_forms[(1, section)], sample)

        pair_valid = False
        pair_preempts = False
        for record in bundle.profiles:
            if not profile_available(record, alt_erased):
                continue
            if any(parity_eval(row, sample) for row in record.equations):
                continue
            pair_valid = True
            valid_profiles += 1
            if profile_preempts(record=record, known_sections=bundle.known_sections, symbols=symbols):
                pair_preempts = True
                break
        valid_pairs += int(pair_valid)
        preemptions += int(pair_preempts)

    return ProbeRow(
        attempt=bundle.attempt,
        mask=bundle.mask,
        multiplicity=bundle.multiplicity,
        pe=pe,
        trials=args.pair_trials,
        pair_preemptions=preemptions,
        valid_pairs=valid_pairs,
        mean_valid_profiles=valid_profiles / args.pair_trials if args.pair_trials else 0.0,
        first_moment=bundle.first_moment,
        min_rank=bundle.min_rank,
    )


def clopper_zero_upper(successes: int, trials: int, alpha: float = 0.05) -> float:
    if trials <= 0:
        return 1.0
    if successes == 0:
        return 1.0 - alpha ** (1.0 / trials)
    return successes / trials


def build_bundles(args: argparse.Namespace) -> tuple[list[MaskBundle], dict[str, dict[tuple[int, int], tuple[int, ...]]]]:
    bundles: list[MaskBundle] = []
    symbol_forms_cache: dict[str, dict[tuple[int, int], tuple[int, ...]]] = {}
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
        masks_with_multiplicity = selected_masks_with_multiplicity(args, masks)
        message_lens, _parity_lens, _gis, gijs, _solve_data = rotate_code(args.L, args.M, args.seed, root)
        symbol_forms_cache[name] = symbol_forms_by_color(
            length=args.L,
            memory=args.M,
            message_lens=message_lens,
            gijs=gijs,
        )
        for mask, multiplicity in masks_with_multiplicity:
            bundles.append(
                build_bundle(
                    args=args,
                    attempt=name,
                    root=root,
                    d=d,
                    erasure_slot=erasure_slot,
                    mask=mask,
                    multiplicity=multiplicity,
                )
            )
    return bundles, symbol_forms_cache


def build_report(args: argparse.Namespace) -> str:
    bundles, symbol_forms_cache = build_bundles(args)
    rng_bits = random.Random(args.seed + 99173)
    rng_np = np.random.default_rng(args.seed + 314159)

    rows: list[ProbeRow] = []
    for bundle in bundles:
        for pe in args.pes:
            rows.append(
                run_bundle_trials(
                    args=args,
                    bundle=bundle,
                    symbol_forms=symbol_forms_cache[bundle.attempt],
                    pe=pe,
                    rng_bits=rng_bits,
                    rng_np=rng_np,
                )
            )

    schedule_bad = schedule_bad_masks(args.L, args.M, args.phase, args.seed)
    lines = [
        "# UACE Symbolic Pair-Level Preemption Probe",
        "",
        "This report is generated by `research/uace_pair_symbolic_probe.py`.",
        "",
        f"- `L = {args.L}`",
        f"- `M = {args.M}`",
        f"- phase: `{args.phase}`",
        f"- weights: `{args.weights}`",
        f"- pair trials per mask and pe: `{args.pair_trials}`",
        f"- gap representatives: `{args.gap_representatives}`",
        "",
        "For each fixed accepted tagged-user mask, the trial samples two random LLC messages and the alternate user's erasures.  It then asks whether at least one mixed two-color profile is both parity-valid and earlier than the true tagged path in the decoder row order.",
        "",
        "## Per-Mask Pair Event",
        "",
        "| attempt | mask sections | mult. | pe | profiles | min rank | first moment | preemptions/trials | valid pairs/trials | mean valid profiles | pair preempt upper |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        upper = clopper_zero_upper(row.pair_preemptions, row.trials)
        profile_count = next(
            len(bundle.profiles)
            for bundle in bundles
            if bundle.attempt == row.attempt and bundle.mask == row.mask
        )
        lines.append(
            f"| {row.attempt} | {mask_to_sections(row.mask, args.L)} | {row.multiplicity} | "
            f"{row.pe:.3f} | {profile_count} | {row.min_rank} | {row.first_moment:.6g} | "
            f"{row.pair_preemptions}/{row.trials} | {row.valid_pairs}/{row.trials} | "
            f"{row.mean_valid_profiles:.6g} | {upper:.6g} |"
        )

    lines.extend(
        [
            "",
            "## K-Scale Projection",
            "",
            "The projection below uses the per-mask empirical rate for nonzero observations, and the one-sided 95% zero-event upper bound when no preemption is observed.  Multiplicity is the number of accepted masks represented by a displayed gap representative.",
            "",
            "| K | pe | schedule UE | empirical pair-union term | 95% pair-union upper |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for k in args.Ks:
        for pe in args.pes:
            schedule_ue = bad_mask_probability(schedule_bad, args.L, pe)
            empirical_extra = 0.0
            upper_extra = 0.0
            for row in rows:
                if abs(row.pe - pe) > 1e-12:
                    continue
                q_emp = row.pair_preemptions / row.trials if row.trials else 0.0
                q_upper = clopper_zero_upper(row.pair_preemptions, row.trials)
                pair_emp = 1.0 - (1.0 - q_emp) ** max(k - 1, 0)
                pair_upper = 1.0 - (1.0 - q_upper) ** max(k - 1, 0)
                mask_prob = (pe ** row.mask.bit_count()) * ((1.0 - pe) ** (args.L - row.mask.bit_count()))
                empirical_extra += row.multiplicity * mask_prob * pair_emp
                upper_extra += row.multiplicity * mask_prob * pair_upper
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(k),
                        f"{pe:.3f}",
                        format_probability(schedule_ue),
                        format_probability(empirical_extra),
                        format_probability(upper_extra),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "Readout:",
            "",
            "- This is a pair-level union measurement, so it is deliberately much closer to decoder behavior than a profile-by-profile first moment.",
            "- The `first moment` column is conditional on a fully available alternate user and is shown only to expose how loose the profile count is before row ordering and alternate erasures are included.",
            "- The K-scale term is an extra path-preemption term on top of the schedule PDP, not a replacement for the schedule term.",
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
    parser.add_argument("--pair-trials", type=int, default=1000)
    parser.add_argument("--Ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--pes", type=float, nargs="+", default=list(DEFAULT_PES))
    parser.add_argument("--output", type=Path, default=Path("research/uace_pair_symbolic_probe.md"))
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
