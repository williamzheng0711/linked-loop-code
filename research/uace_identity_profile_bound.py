#!/usr/bin/env python3
"""Identity-profile first moments for LLC/UACE false paths.

The rank-corrected A-list first moment treats chosen section symbols as
independent random symbols.  This script exposes a different finite-length
object: a false path can be described by the true-user identity supplying each
non-erased section.  Profiles with few identities model path switches and
returns directly, and their parity exponent can be computed exactly by GF(2)
rank over the underlying users' information bits.

The resulting rows are component bounds, not yet the final all-profile DP.
They quantify the dominant low-identity path-switch classes that a publishable
dependency-aware theorem must control.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from uace_bound_explorer import (
    bad_mask_probability,
    cantor_pairing,
    format_probability,
    load_repo_code_matrices,
    who_decides_p_sec,
)
from uace_schedule_bound import attempt_succeeds, mask_to_sections, schedule_attempts, schedule_bad_masks


DEFAULT_KS = (30, 40, 100)
DEFAULT_PES = (0.1, 0.2, 0.3)


@dataclass(frozen=True)
class ProfileRow:
    attempt: str
    weight: int
    colors: int
    exponent: int
    profiles: int
    masks: int
    known_sections: int


@dataclass(frozen=True)
class RankSpecs:
    known_sections: tuple[int, ...]
    unknown_rank: int
    row_specs: tuple[
        tuple[int, int, tuple[tuple[int, tuple[int, ...]], ...], tuple[int, ...]],
        ...,
    ]


def bit_rank(rows: list[int]) -> int:
    """GF(2) rank for rows encoded as Python integers."""
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
    return len(basis)


def add_block(row: int, base_col: int, block_col: np.ndarray) -> int:
    """XOR one parity-bit column of a generator block into a bitset row."""
    for idx, bit in enumerate(block_col):
        if int(bit) & 1:
            row ^= 1 << (base_col + idx)
    return row


def block_bits(base_col: int, block_col: np.ndarray) -> int:
    row = 0
    for idx, bit in enumerate(block_col):
        if int(bit) & 1:
            row ^= 1 << (base_col + idx)
    return row


def user_col_base(
    *,
    unknown_cols: int,
    color: int,
    section: int,
    section_offsets: tuple[int, ...],
    bits_per_user: int,
) -> int:
    return unknown_cols + color * bits_per_user + section_offsets[section]


def unknown_col_base(
    *,
    erased_offsets: dict[int, int],
    section: int,
) -> int:
    return erased_offsets[section]


def build_rank_specs(
    *,
    mask: int,
    colors: int,
    length: int,
    memory: int,
    message_lens: np.ndarray,
    parity_lens: np.ndarray,
    gijs: dict,
) -> RankSpecs:
    """Precompute equation rows that are independent of the profile colors.

    A later profile only chooses which precomputed color term is active for
    each known section.
    """
    known_sections = tuple(section for section in range(length) if ((mask >> section) & 1) == 0)
    known_pos = {section: pos for pos, section in enumerate(known_sections)}

    erased_sections = tuple(section for section in range(length) if (mask >> section) & 1)
    erased_offsets: dict[int, int] = {}
    unknown_cols = 0
    for section in erased_sections:
        erased_offsets[section] = unknown_cols
        unknown_cols += int(message_lens[section])

    section_offsets: list[int] = []
    running = 0
    for section in range(length):
        section_offsets.append(running)
        running += int(message_lens[section])
    section_offsets_tuple = tuple(section_offsets)
    bits_per_user = running

    row_specs = []
    unknown_rows: list[int] = []

    for section in known_sections:
        width = int(parity_lens[section])
        deciders = who_decides_p_sec(length, section, memory)
        for bit in range(width):
            unknown_row = 0
            unknown_full = 0
            candidate_terms: list[tuple[int, tuple[int, ...]]] = []

            # Candidate parity built from the assembled path.
            for decider in deciders:
                block_col = np.asarray(gijs[cantor_pairing(decider, section)], dtype=np.uint8)[:, bit]
                if decider in erased_offsets:
                    base = unknown_col_base(erased_offsets=erased_offsets, section=decider)
                    bits = block_bits(base, block_col)
                    unknown_full ^= bits
                    unknown_row ^= bits
                else:
                    terms = []
                    for color in range(colors):
                        base = user_col_base(
                            unknown_cols=unknown_cols,
                            color=color,
                            section=decider,
                            section_offsets=section_offsets_tuple,
                            bits_per_user=bits_per_user,
                        )
                        terms.append(block_bits(base, block_col))
                    candidate_terms.append((known_pos[decider], tuple(terms)))

            # Observed parity of the selected section symbol.
            observed_terms = []
            for color in range(colors):
                observed = 0
                for decider in deciders:
                    block_col = np.asarray(gijs[cantor_pairing(decider, section)], dtype=np.uint8)[:, bit]
                    base = user_col_base(
                        unknown_cols=unknown_cols,
                        color=color,
                        section=decider,
                        section_offsets=section_offsets_tuple,
                        bits_per_user=bits_per_user,
                    )
                    observed ^= block_bits(base, block_col)
                observed_terms.append(observed)

            row_specs.append((known_pos[section], unknown_full, tuple(candidate_terms), tuple(observed_terms)))
            if unknown_row:
                unknown_rows.append(unknown_row)

    return RankSpecs(
        known_sections=known_sections,
        unknown_rank=bit_rank(unknown_rows),
        row_specs=tuple(row_specs),
    )


def profile_exponent_from_specs(profile: tuple[int, ...], specs: RankSpecs) -> int:
    if len(profile) != len(specs.known_sections):
        raise ValueError("profile length does not match known section count")
    full_rows: list[int] = []
    for section_pos, unknown_full, candidate_terms, observed_terms in specs.row_specs:
        row = unknown_full
        for decider_pos, terms in candidate_terms:
            row ^= terms[profile[decider_pos]]
        row ^= observed_terms[profile[section_pos]]
        if row:
            full_rows.append(row)

    return bit_rank(full_rows) - specs.unknown_rank


def profile_exponent(
    *,
    mask: int,
    profile: tuple[int, ...],
    length: int,
    memory: int,
    message_lens: np.ndarray,
    parity_lens: np.ndarray,
    gijs: dict,
) -> int:
    """Exact exponent for one section-identity profile.

    Known section `l` takes its information and parity bits from user
    `profile[pos(l)]`.  Erased candidate sections are existential variables.
    For each observed non-erased parity section, we impose equality between
    the candidate parity and the selected user's transmitted parity.
    """
    colors = max(profile) + 1 if profile else 0
    specs = build_rank_specs(
        mask=mask,
        colors=colors,
        length=length,
        memory=memory,
        message_lens=message_lens,
        parity_lens=parity_lens,
        gijs=gijs,
    )
    return profile_exponent_from_specs(profile, specs)


def restricted_growth_profiles(length: int, colors: int):
    """Yield canonical identity profiles using exactly `colors` colors."""
    if colors <= 0:
        return
    profile = [0]

    def rec(pos: int, max_color: int):
        if pos == length:
            if max_color + 1 == colors:
                yield tuple(profile)
            return
        upper = min(colors - 1, max_color + 1)
        for color in range(upper + 1):
            # Prune if not enough positions remain to introduce missing colors.
            new_max = max(max_color, color)
            missing = colors - (new_max + 1)
            if missing > length - pos - 1:
                continue
            profile.append(color)
            yield from rec(pos + 1, new_max)
            profile.pop()

    if length == 0:
        return
    yield from rec(1, 0)


@lru_cache(maxsize=None)
def profiles_for(length: int, colors: int) -> tuple[tuple[int, ...], ...]:
    return tuple(restricted_growth_profiles(length, colors))


def falling_factorial(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    out = 1
    for offset in range(k):
        out *= n - offset
    return out


def profile_switch_count(profile: tuple[int, ...]) -> int:
    return sum(int(profile[idx] != profile[idx - 1]) for idx in range(1, len(profile)))


def accepted_masks_for_attempt(
    *,
    length: int,
    memory: int,
    root: int,
    d: int,
    erasure_slot: set[int],
    message_lens: np.ndarray,
    gijs: dict,
    weights: set[int],
) -> list[int]:
    masks = []
    for mask in range(1 << length):
        if weights and mask.bit_count() not in weights:
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
            masks.append(mask)
    return masks


def circular_gap_pair(sections: tuple[int, ...], length: int) -> tuple[int, int] | None:
    if len(sections) != 2:
        return None
    a, b = sections
    gap = (b - a) % length
    return tuple(sorted((gap, length - gap)))


def filter_masks(args: argparse.Namespace, masks: list[int]) -> list[int]:
    if args.gap_representatives:
        by_gap: dict[tuple[int, int] | None, int] = {}
        for mask in sorted(masks):
            gap = circular_gap_pair(mask_to_sections(mask, args.L), args.L)
            by_gap.setdefault(gap, mask)
        masks = list(by_gap.values())
    if args.max_masks_per_attempt and args.max_masks_per_attempt > 0:
        masks = masks[: args.max_masks_per_attempt]
    return masks


def build_profile_rows(args: argparse.Namespace) -> tuple[list[ProfileRow], dict[tuple[str, int, int], Counter[tuple[int, int]]]]:
    message_lens, parity_lens, gijs = load_repo_code_matrices(args.L, args.M, args.seed)
    rows: list[ProfileRow] = []
    switch_profiles: dict[tuple[str, int, int], Counter[tuple[int, int]]] = {}

    for name, root, d, erasure_slot in schedule_attempts(args.L, args.phase):
        if args.only_attempt and args.only_attempt not in name:
            continue
        masks = accepted_masks_for_attempt(
            length=args.L,
            memory=args.M,
            root=root,
            d=d,
            erasure_slot=erasure_slot,
            message_lens=message_lens,
            gijs=gijs,
            weights=set(args.weights),
        )
        masks = filter_masks(args, masks)
        for colors in args.colors:
            by_key: Counter[tuple[int, int, int]] = Counter()
            by_switch: Counter[tuple[int, int]] = Counter()
            for mask in masks:
                known_count = args.L - mask.bit_count()
                specs = build_rank_specs(
                    mask=mask,
                    colors=colors,
                    length=args.L,
                    memory=args.M,
                    message_lens=message_lens,
                    parity_lens=parity_lens,
                    gijs=gijs,
                )
                for profile in profiles_for(known_count, colors):
                    exponent = profile_exponent_from_specs(profile, specs)
                    by_key[(mask.bit_count(), known_count, exponent)] += 1
                    by_switch[(profile_switch_count(profile), exponent)] += 1
            for (weight, known_count, exponent), count in sorted(by_key.items()):
                rows.append(
                    ProfileRow(
                        attempt=name,
                        weight=weight,
                        colors=colors,
                        exponent=exponent,
                        profiles=count,
                        masks=len([mask for mask in masks if mask.bit_count() == weight]),
                        known_sections=known_count,
                    )
                )
            switch_profiles[(name, colors, args.L - (args.weights[0] if len(args.weights) == 1 else -1))] = by_switch
    return rows, switch_profiles


def build_report(args: argparse.Namespace) -> str:
    rows, switch_profiles = build_profile_rows(args)
    schedule_bad = schedule_bad_masks(args.L, args.M, args.phase, args.seed)

    lines = [
        "# UACE Identity-Profile Path-Switch Bound",
        "",
        "This report is generated by `research/uace_identity_profile_bound.py`.",
        "",
        f"- `L = {args.L}`",
        f"- `M = {args.M}`",
        f"- phase: `{args.phase}`",
        f"- weights: `{args.weights}`",
        f"- colors: `{args.colors}`",
        "",
        "A `color` is one actual user identity used by the candidate path.  The all-one-color profile is the true-user path and is excluded here; two-color profiles are the minimal path-switch / path-return class.",
        "",
        "For a fixed identity profile, the exponent is computed from the exact final-parity linear system equating candidate parities to the selected users' transmitted parities, with erased candidate sections left existential.  This is a necessary-condition first moment for the current sequential decoder: it does not yet model intermediate recovery rejection or first-valid-path ordering.",
        "",
        "## Exponent Inventory",
        "",
        "| attempt | weight | colors | known sections | accepted masks | canonical profiles | exponent |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.attempt} | {row.weight} | {row.colors} | {row.known_sections} | "
            f"{row.masks} | {row.profiles} | {row.exponent} |"
        )

    lines.extend(
        [
            "",
            "## Contribution Scale",
            "",
            "For `c` colors and `m=L-w` known sections, a canonical profile contributes",
            "",
            "$$",
            "(K)_c(1-p_e)^m2^{-e},",
            "$$",
            "",
            "where $(K)_c=K(K-1)\\cdots(K-c+1)$ assigns distinct true users to the canonical colors.",
            "",
            "| K | pe | schedule UE | colors | E low-color profiles | per-user scale | dominant exponent |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for k in args.Ks:
        for pe in args.pes:
            schedule_ue = bad_mask_probability(schedule_bad, args.L, pe)
            by_color: defaultdict[int, float] = defaultdict(float)
            dominant: dict[int, tuple[float, int]] = {}
            for row in rows:
                contribution = (
                    row.profiles
                    * falling_factorial(k, row.colors)
                    * ((1.0 - pe) ** row.known_sections)
                    * (2.0 ** (-row.exponent))
                )
                by_color[row.colors] += contribution
                if row.colors not in dominant or contribution > dominant[row.colors][0]:
                    dominant[row.colors] = (contribution, row.exponent)
            for colors in sorted(by_color):
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(k),
                            f"{pe:.3f}",
                            format_probability(schedule_ue),
                            str(colors),
                            format_probability(by_color[colors]),
                            format_probability(by_color[colors] / k),
                            str(dominant[colors][1]),
                        ]
                    )
                    + " |"
                )

    lines.extend(
        [
            "",
            "## Switch/Return Shape Summary",
            "",
            "| attempt | colors | switches in known-section order | exponent | canonical profiles |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for (attempt, colors, _known), counter in sorted(switch_profiles.items()):
        for (switches, exponent), count in sorted(counter.items()):
            lines.append(f"| {attempt} | {colors} | {switches} | {exponent} | {count} |")

    lines.extend(
        [
            "",
            "Readout:",
            "",
            "- This is a structured necessary-condition component for low-identity path switches, not the final all-profile dependency-aware DP.",
            "- Large values mean the final-parity-only abstraction is too loose unless sequential recovery rejection and first-valid-path ordering are added.",
            "- The next publishable bound should therefore refine these identity profiles by decoder order and recovery state, rather than treating every final-parity-consistent profile as an actual decoder error.",
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
    parser.add_argument("--weights", type=int, nargs="+", default=[2])
    parser.add_argument("--colors", type=int, nargs="+", default=[2])
    parser.add_argument("--only-attempt", type=str, default="phase-III root")
    parser.add_argument("--max-masks-per-attempt", type=int, default=0)
    parser.add_argument("--gap-representatives", action="store_true")
    parser.add_argument("--Ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--pes", type=float, nargs="+", default=list(DEFAULT_PES))
    parser.add_argument("--output", type=Path, default=Path("research/uace_identity_profile_bound.md"))
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
