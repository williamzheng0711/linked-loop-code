#!/usr/bin/env python3
"""Exact GF(2) feasibility check for two-user LLC/UACE pair preemption.

For a fixed tagged erasure mask and a fixed two-color identity profile, the
current ordered LLC decoder contributes homogeneous GF(2) equations in the two
users' information bits.  The event that the alternate user's path appears
before the tagged path in A-list row order is also a finite union of affine
GF(2) systems: all earlier alternate-selected symbols are equal, and at the
first differing symbol the alternate bit is 0 while the tagged bit is 1.

Therefore a two-user first-preempt event can be checked exactly by Gaussian
elimination over GF(2), without Monte Carlo or an SMT solver.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from uace_ordered_profile_bound import accepted_rotated_masks, profiles_for, rotate_code
from uace_pair_symbolic_probe import ordered_profile_equations
from uace_schedule_bound import mask_to_sections, schedule_attempts
from uace_ordered_profile_bound import observed_parity_forms, section_offsets, user_section_forms


DEFAULT_KS = (30, 40, 100)
DEFAULT_PES = (0.1, 0.2, 0.3)


@dataclass(frozen=True)
class MaskExactRow:
    colors: int
    attempt: str
    mask: int
    multiplicity: int
    profiles_checked: int
    parity_valid_profiles: int
    preempt_feasible_profiles: int
    preempt_terms: int
    preempt_union_bound: float
    preempt_erasure_weighted: dict[float, float]
    min_preempt_rank: int
    min_rank: int
    witness_profile: tuple[int, ...] | None
    witness_section: int | None
    witness_bit: int | None


def add_affine_to_basis(
    basis: dict[int, tuple[int, int]],
    row: int,
    rhs: int,
) -> bool:
    """Add one affine GF(2) equation to a basis.

    Returns False if the augmented system becomes inconsistent.
    """
    value = row
    bit = rhs & 1
    while value:
        pivot = value.bit_length() - 1
        if pivot not in basis:
            basis[pivot] = (value, bit)
            return True
        base_row, base_rhs = basis[pivot]
        value ^= base_row
        bit ^= base_rhs
    return bit == 0


def build_homogeneous_basis(rows: tuple[int, ...]) -> dict[int, tuple[int, int]]:
    basis: dict[int, tuple[int, int]] = {}
    for row in rows:
        ok = add_affine_to_basis(basis, row, 0)
        if not ok:
            raise ValueError("homogeneous equations cannot be inconsistent")
    return basis


def affine_consistent(
    base_basis: dict[int, tuple[int, int]],
    constraints: list[tuple[int, int]],
) -> bool:
    basis = dict(base_basis)
    for row, rhs in constraints:
        if row == 0:
            if rhs & 1:
                return False
            continue
        if not add_affine_to_basis(basis, row, rhs):
            return False
    return True


def extend_basis(
    base_basis: dict[int, tuple[int, int]],
    constraints: list[tuple[int, int]],
) -> dict[int, tuple[int, int]] | None:
    basis = dict(base_basis)
    for row, rhs in constraints:
        if row == 0:
            if rhs & 1:
                return None
            continue
        if not add_affine_to_basis(basis, row, rhs):
            return None
    return basis


def preemption_feasible(
    *,
    equations: tuple[int, ...],
    profile: tuple[int, ...],
    known_sections: tuple[int, ...],
    symbol_forms: dict[tuple[int, int], tuple[int, ...]],
) -> tuple[bool, int | None, int | None]:
    """Check whether this profile can be parity-valid and row-order preempt."""
    base_basis = build_homogeneous_basis(equations)
    earlier_equalities: list[tuple[int, int]] = []

    for pos, color in enumerate(profile):
        if color == 0:
            continue
        section = known_sections[pos]
        tag = symbol_forms[(0, section)]
        alt = symbol_forms[(1, section)]
        prefix_equalities: list[tuple[int, int]] = []
        for bit in range(len(tag)):
            constraints = (
                earlier_equalities
                + prefix_equalities
                + [(alt[bit], 0), (tag[bit], 1)]
            )
            if affine_consistent(base_basis, constraints):
                return True, section, bit
            prefix_equalities.append((alt[bit] ^ tag[bit], 0))

        earlier_equalities.extend((alt[bit] ^ tag[bit], 0) for bit in range(len(tag)))

    return False, None, None


def preemption_terms(
    *,
    equations: tuple[int, ...],
    profile: tuple[int, ...],
    known_sections: tuple[int, ...],
    symbol_forms: dict[tuple[int, int], tuple[int, ...]],
) -> list[tuple[int, int, int, int]]:
    """Return feasible row-order preemption disjuncts.

    Each tuple is `(section, bit, rank, alt_section_count)`.  The rank is the
    affine system rank for this disjoint first-difference event.
    """
    base_basis = build_homogeneous_basis(equations)
    earlier_equalities: list[tuple[int, int]] = []
    alt_symbols_so_far: set[tuple[int, int]] = set()
    terms: list[tuple[int, int, int, int]] = []

    for pos, color in enumerate(profile):
        if color == 0:
            continue
        section = known_sections[pos]
        tag = symbol_forms[(0, section)]
        alt = symbol_forms[(color, section)]
        prefix_equalities: list[tuple[int, int]] = []
        for bit in range(len(tag)):
            constraints = (
                earlier_equalities
                + prefix_equalities
                + [(alt[bit], 0), (tag[bit], 1)]
            )
            basis = extend_basis(base_basis, constraints)
            if basis is not None:
                terms.append((section, bit, len(basis), len(alt_symbols_so_far | {(color, section)})))
            prefix_equalities.append((alt[bit] ^ tag[bit], 0))

        earlier_equalities.extend((alt[bit] ^ tag[bit], 0) for bit in range(len(tag)))
        alt_symbols_so_far.add((color, section))

    return terms


def symbol_forms_by_color(
    *,
    length: int,
    memory: int,
    message_lens: np.ndarray,
    gijs: dict,
    colors: int,
) -> dict[tuple[int, int], tuple[int, ...]]:
    offsets, _bits_per_user = section_offsets(message_lens)
    forms = {}
    for color in range(colors):
        for section in range(length):
            info = user_section_forms(
                color=color,
                section=section,
                section_offsets=offsets,
                bits_per_user=int(sum(message_lens)),
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
                bits_per_user=int(sum(message_lens)),
            )
            forms[(color, section)] = tuple(info) + tuple(parity)
    return forms


def falling_factorial(n: int, k: int) -> int:
    out = 1
    for offset in range(k):
        out *= max(n - offset, 0)
    return out


def mask_from_sections(sections: list[int]) -> int:
    mask = 0
    for section in sections:
        mask |= 1 << section
    return mask


def circular_gap_pair(sections: tuple[int, ...], length: int) -> tuple[int, int] | None:
    if len(sections) != 2:
        return None
    a, b = sections
    gap = (b - a) % length
    return tuple(sorted((gap, length - gap)))


def selected_masks_with_multiplicity(args: argparse.Namespace, masks: list[int]) -> list[tuple[int, int]]:
    if args.only_mask_sections:
        target = mask_from_sections(args.only_mask_sections)
        if target not in masks:
            sections = tuple(args.only_mask_sections)
            raise ValueError(f"requested mask sections {sections} are not accepted by this attempt")
        multiplicity = args.mask_multiplicity if args.mask_multiplicity > 0 else 1
        return [(target, multiplicity)]

    if args.gap_representatives:
        groups: dict[tuple[int, int] | None, list[int]] = {}
        for mask in sorted(masks):
            groups.setdefault(circular_gap_pair(mask_to_sections(mask, args.L), args.L), []).append(mask)
        return [(items[0], len(items)) for items in groups.values()]

    if args.max_masks_per_attempt > 0:
        masks = masks[: args.max_masks_per_attempt]
    return [(mask, 1) for mask in masks]


def analyze_mask(
    *,
    args: argparse.Namespace,
    attempt: str,
    root: int,
    d: int,
    erasure_slot: set[int],
    mask: int,
    multiplicity: int,
) -> MaskExactRow:
    message_lens, parity_lens, gis, gijs, solve_data = rotate_code(args.L, args.M, args.seed, root)
    symbol_forms = symbol_forms_by_color(
        length=args.L,
        memory=args.M,
        message_lens=message_lens,
        gijs=gijs,
        colors=args.colors,
    )
    known_sections = tuple(section for section in range(args.L) if ((mask >> section) & 1) == 0)
    checked = 0
    parity_valid = 0
    preempt_feasible_count = 0
    preempt_term_count = 0
    preempt_union_bound = 0.0
    preempt_erasure_weighted = {pe: 0.0 for pe in args.pes}
    min_preempt_rank = 10**9
    min_rank = 10**9
    witness_profile = None
    witness_section = None
    witness_bit = None

    for profile_index, profile in enumerate(profiles_for(len(known_sections), args.colors)):
        if profile_index < args.profile_offset:
            continue
        if args.max_profiles and checked >= args.max_profiles:
            break
        checked += 1
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
        parity_valid += 1
        min_rank = min(min_rank, len(equations))
        terms = preemption_terms(
            equations=equations,
            profile=profile,
            known_sections=known_sections,
            symbol_forms=symbol_forms,
        )
        if terms:
            preempt_feasible_count += 1
            preempt_term_count += len(terms)
            for section, bit, rank, alt_section_count in terms:
                contribution = 2.0 ** (-rank)
                preempt_union_bound += contribution
                min_preempt_rank = min(min_preempt_rank, rank)
                for pe in args.pes:
                    preempt_erasure_weighted[pe] += contribution * ((1.0 - pe) ** alt_section_count)
                if witness_profile is None:
                    witness_profile = profile
                    witness_section = section
                    witness_bit = bit
            if args.stop_on_witness:
                break

    return MaskExactRow(
        colors=args.colors,
        attempt=attempt,
        mask=mask,
        multiplicity=multiplicity,
        profiles_checked=checked,
        parity_valid_profiles=parity_valid,
        preempt_feasible_profiles=preempt_feasible_count,
        preempt_terms=preempt_term_count,
        preempt_union_bound=preempt_union_bound,
        preempt_erasure_weighted=preempt_erasure_weighted,
        min_preempt_rank=min_preempt_rank if min_preempt_rank < 10**9 else -1,
        min_rank=min_rank if min_rank < 10**9 else -1,
        witness_profile=witness_profile,
        witness_section=witness_section,
        witness_bit=witness_bit,
    )


def build_rows(args: argparse.Namespace) -> list[MaskExactRow]:
    rows: list[MaskExactRow] = []
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
        if not args.gap_representatives and args.max_masks_per_attempt > 0:
            masks = masks[: args.max_masks_per_attempt]
        masks_with_multiplicity = selected_masks_with_multiplicity(args, masks)
        for mask, multiplicity in masks_with_multiplicity:
            rows.append(
                analyze_mask(
                    args=args,
                    attempt=name,
                    root=root,
                    d=d,
                    erasure_slot=erasure_slot,
                    mask=mask,
                    multiplicity=multiplicity,
                )
            )
    return rows


def build_report(args: argparse.Namespace) -> str:
    rows = build_rows(args)
    total_checked = sum(row.profiles_checked for row in rows)
    total_feasible = sum(row.preempt_feasible_profiles for row in rows)
    total_union_bound = sum(row.multiplicity * row.preempt_union_bound for row in rows)
    lines = [
        "# UACE Exact Pair-Preemption Feasibility",
        "",
        "This report is generated by `research/uace_pair_preemption_exact.py`.",
        "",
        f"- `L = {args.L}`",
        f"- `M = {args.M}`",
        f"- phase: `{args.phase}`",
        f"- weights: `{args.weights}`",
        f"- colors: `{args.colors}`",
        f"- gap representatives: `{args.gap_representatives}`",
        f"- only mask sections: `{args.only_mask_sections or 'all'}`",
        f"- mask multiplicity override: `{args.mask_multiplicity or 'auto'}`",
        f"- profile offset: `{args.profile_offset}`",
        f"- max profiles per mask: `{args.max_profiles or 'all'}`",
        "",
        "For each fixed identity profile, parity validity and row-order preemption are checked as one or more affine GF(2) systems.  No random messages are sampled.",
        "",
    ]
    weighted_headers = [f"weighted q pe={pe:.3f}" for pe in args.pes]
    row_headers = [
        "attempt",
        "mask sections",
        "mult.",
        "profiles checked",
        "parity-valid profiles",
        "min parity rank",
        "feasible profiles",
        "preempt terms",
        "min preempt rank",
        "pair union bound",
        *weighted_headers,
        "witness section",
        "witness bit",
    ]
    lines.append("| " + " | ".join(row_headers) + " |")
    lines.append("|" + "|".join("---" if idx < 2 else "---:" for idx, _ in enumerate(row_headers)) + "|")
    for row in rows:
        weighted_parts = [f"{row.preempt_erasure_weighted[pe]:.6g}" for pe in args.pes]
        lines.append(
            f"| {row.attempt} | {mask_to_sections(row.mask, args.L)} | {row.multiplicity} | "
            f"{row.profiles_checked} | {row.parity_valid_profiles} | {row.min_rank} | "
            f"{row.preempt_feasible_profiles} | {row.preempt_terms} | {row.min_preempt_rank} | "
            f"{row.preempt_union_bound:.6g} | "
            + " | ".join(weighted_parts)
            + " | "
            f"{'' if row.witness_section is None else row.witness_section} | "
            f"{'' if row.witness_bit is None else row.witness_bit} |"
        )

    lines.extend(
        [
            "",
            "## Erasure-Weighted Pair Bounds",
            "",
            "The raw pair union bound ignores alternate-user erasures.  The erasure-weighted version multiplies each first-difference term by `(1-pe)^s`, where `s` is the number of alternate-selected sections that must be available up through the first differing section.  This is still a union bound, but it is closer to the UACE pair event.",
            "",
            "| pe | multiplicity-weighted raw pair bound | multiplicity-weighted erasure-weighted pair bound |",
            "|---:|---:|---:|",
        ]
    )
    for pe in args.pes:
        weighted = sum(row.multiplicity * row.preempt_erasure_weighted[pe] for row in rows)
        lines.append(f"| {pe:.3f} | {total_union_bound:.6g} | {weighted:.6g} |")

    lines.extend(
        [
            "",
            "## K-Scale Extra-Term Projection",
            "",
            "All checked masks have weight 2.  The table multiplies the fixed-profile-family bound by the tagged-mask probability `pe^2(1-pe)^(L-2)` and then lifts from `colors-1` ordered alternate users to `(K-1)_(colors-1)` choices.  The columns are conservative union bounds.",
            "",
            "| K | pe | ordered-user choices | raw union extra | erasure-weighted union extra |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for k in args.Ks:
        for pe in args.pes:
            mask_prob = (pe**2) * ((1.0 - pe) ** (args.L - 2))
            raw_q = total_union_bound
            weighted_q = sum(row.multiplicity * row.preempt_erasure_weighted[pe] for row in rows)
            choices = falling_factorial(k - 1, args.colors - 1)
            raw_union = mask_prob * min(1.0, choices * raw_q)
            weighted_union = mask_prob * min(1.0, choices * weighted_q)
            lines.append(
                f"| {k} | {pe:.3f} | {choices} | {raw_union:.6g} | {weighted_union:.6g} |"
            )

    lines.extend(
        [
            "",
            "## Readout",
            "",
            f"- Total profiles checked: `{total_checked}`.",
            f"- Total preempt-feasible profiles: `{total_feasible}`.",
            f"- Multiplicity-weighted raw pair union bound: `{total_union_bound:.6g}`.",
        ]
    )
    if total_feasible == 0:
        lines.append("- For the checked finite instance, no mixed profile can both pass the ordered decoder equations and appear before the tagged path in row order.")
    else:
        lines.append("- At least one feasible preempting profile exists; inspect the witness columns and refine the probability calculation.")
    if args.profile_offset or args.max_profiles:
        lines.append("- This is a profile-window report, not a full profile enumeration.  Its union totals are partial diagnostic contributions for the checked window.")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--M", type=int, default=3)
    parser.add_argument("--phase", type=int, default=3, choices=(2, 3))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights", type=int, nargs="+", default=[2])
    parser.add_argument("--colors", type=int, default=2)
    parser.add_argument("--only-attempt", type=str, default="phase-III root")
    parser.add_argument("--gap-representatives", action="store_true")
    parser.add_argument("--max-masks-per-attempt", type=int, default=0)
    parser.add_argument("--only-mask-sections", type=int, nargs="+", default=None)
    parser.add_argument("--mask-multiplicity", type=int, default=0)
    parser.add_argument("--profile-offset", type=int, default=0)
    parser.add_argument("--max-profiles", type=int, default=0)
    parser.add_argument("--Ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--pes", type=float, nargs="+", default=list(DEFAULT_PES))
    parser.add_argument("--stop-on-witness", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("research/uace_pair_preemption_exact.md"))
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
