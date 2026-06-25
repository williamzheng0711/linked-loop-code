#!/usr/bin/env python3
"""Order-aware identity-profile bound for the current LLC/UACE decoder.

This script refines `uace_identity_profile_bound.py`.  The identity-profile
script only checks final parity consistency, treating erased candidate sections
as existential variables.  The current decoder is stricter: it scans sections
in order, recovers erased sections from the first available saver equations,
checks full-saver consistency when all M savers are visible, and only then runs
the final parity check.

For a fixed erasure mask and fixed section-identity profile, this script
symbolically executes that ordered decoder path.  All true-user information bits
are represented as GF(2) variables.  Decoder-recovered erased sections are
linear forms in those variables.  Every parity/recovery rejection contributes a
linear equation; the exponent is the rank of the accumulated equations.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from uace_bound_explorer import bad_mask_probability, cantor_pairing, format_probability, load_repo_code_matrices
from uace_schedule_bound import attempt_succeeds, mask_to_sections, rotate_mask, schedule_attempts, schedule_bad_masks


DEFAULT_KS = (30, 40, 100)
DEFAULT_PES = (0.1, 0.2, 0.3)


@dataclass(frozen=True)
class OrderedRow:
    attempt: str
    weight: int
    colors: int
    known_sections: int
    accepted_masks: int
    canonical_profiles: int
    exponent: int
    rejected_profiles: int


def load_playground():
    repo_root = Path(__file__).resolve().parents[1]
    playground = repo_root / "playground"
    sys.path.insert(0, str(playground))
    import general_utils as gu  # type: ignore
    import static_repo  # type: ignore

    return static_repo, gu


def bit_rank(rows: list[int]) -> int:
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


def block_apply(forms: tuple[int, ...], block: np.ndarray) -> list[int]:
    """Return linear forms for forms @ block over GF(2)."""
    block = np.asarray(block, dtype=np.uint8)
    out = []
    for col in range(block.shape[1]):
        row = 0
        for idx, bit in enumerate(block[:, col]):
            if int(bit) & 1:
                row ^= forms[idx]
        out.append(row)
    return out


def xor_vectors(*vectors: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if not vectors:
        return tuple()
    out = list(vectors[0])
    for vector in vectors[1:]:
        for idx, item in enumerate(vector):
            out[idx] ^= item
    return tuple(out)


def basis_forms(start: int, width: int) -> tuple[int, ...]:
    return tuple(1 << (start + idx) for idx in range(width))


def user_section_forms(
    *,
    color: int,
    section: int,
    section_offsets: tuple[int, ...],
    bits_per_user: int,
    message_lens: np.ndarray,
) -> tuple[int, ...]:
    base = color * bits_per_user + section_offsets[section]
    return basis_forms(base, int(message_lens[section]))


def section_offsets(message_lens: np.ndarray) -> tuple[tuple[int, ...], int]:
    offsets = []
    running = 0
    for width in message_lens:
        offsets.append(running)
        running += int(width)
    return tuple(offsets), running


def observed_parity_forms(
    *,
    color: int,
    section: int,
    length: int,
    memory: int,
    message_lens: np.ndarray,
    gijs: dict,
    offsets: tuple[int, ...],
    bits_per_user: int,
) -> tuple[int, ...]:
    pieces = []
    for decider in who_decides_p_sec(length, section, memory):
        info = user_section_forms(
            color=color,
            section=decider,
            section_offsets=offsets,
            bits_per_user=bits_per_user,
            message_lens=message_lens,
        )
        pieces.append(block_apply(info, gijs[cantor_pairing(decider, section)]))
    return xor_vectors(*pieces)


def candidate_info_forms(
    *,
    section: int,
    color_by_section: dict[int, int],
    recovered: dict[int, tuple[int, ...]],
    message_lens: np.ndarray,
    offsets: tuple[int, ...],
    bits_per_user: int,
) -> tuple[int, ...] | None:
    if section in recovered:
        return recovered[section]
    if section in color_by_section:
        return user_section_forms(
            color=color_by_section[section],
            section=section,
            section_offsets=offsets,
            bits_per_user=bits_per_user,
            message_lens=message_lens,
        )
    return None


def candidate_parity_forms(
    *,
    section: int,
    length: int,
    memory: int,
    color_by_section: dict[int, int],
    recovered: dict[int, tuple[int, ...]],
    message_lens: np.ndarray,
    parity_lens: np.ndarray,
    gijs: dict,
    offsets: tuple[int, ...],
    bits_per_user: int,
) -> tuple[int, ...] | None:
    pieces = []
    for decider in who_decides_p_sec(length, section, memory):
        info = candidate_info_forms(
            section=decider,
            color_by_section=color_by_section,
            recovered=recovered,
            message_lens=message_lens,
            offsets=offsets,
            bits_per_user=bits_per_user,
        )
        if info is None:
            return None
        pieces.append(block_apply(info, gijs[cantor_pairing(decider, section)]))
    if not pieces:
        return tuple(0 for _ in range(int(parity_lens[section])))
    return xor_vectors(*pieces)


def who_decides_p_sec(length: int, section: int, memory: int) -> list[int]:
    return [(section - offset) % length for offset in range(memory, 0, -1)]


def saver_sections(length: int, lost_section: int, memory: int) -> list[int]:
    return [(lost_section + offset) % length for offset in range(1, memory + 1)]


def available_savers(length: int, memory: int, lost_section: int, current_section: int) -> list[int]:
    return [
        saver
        for saver in saver_sections(length, lost_section, memory)
        if all(((saver - offset) % length) <= current_section for offset in range(memory + 1))
    ]


def solve_info_forms(
    known: tuple[int, ...],
    *,
    lost_section: int,
    solve_data: dict[int, tuple[np.ndarray, np.ndarray]],
    message_lens: np.ndarray,
) -> tuple[int, ...]:
    columns, inv = solve_data[lost_section]
    selected = [known[int(col)] for col in columns]
    out = []
    inv = np.asarray(inv, dtype=np.uint8)
    for col in range(inv.shape[1]):
        form = 0
        for idx, bit in enumerate(inv[:, col]):
            if int(bit) & 1:
                form ^= selected[idx]
        out.append(form)
    if len(out) != int(message_lens[lost_section]):
        raise ValueError("bad solve output width")
    return tuple(out)


def recover_unsolved(
    *,
    focus_path: list[int],
    current_section: int,
    color_by_section: dict[int, int],
    recovered: dict[int, tuple[int, ...]],
    equations: list[int],
    length: int,
    memory: int,
    message_lens: np.ndarray,
    parity_lens: np.ndarray,
    gis: np.ndarray,
    gijs: dict,
    offsets: tuple[int, ...],
    bits_per_user: int,
    solve_data: dict[int, tuple[np.ndarray, np.ndarray]],
) -> bool:
    losts = [idx for idx, item in enumerate(focus_path) if item < 0]
    unsolved = [idx for idx in losts if idx not in recovered]
    for lost_section in unsolved:
        savers = available_savers(length, memory, lost_section, current_section)
        if not savers:
            return False
        if sum(int(parity_lens[saver]) for saver in savers) < int(message_lens[lost_section]):
            continue

        known_vectors: list[tuple[int, ...]] = []
        for saver in savers:
            if saver not in color_by_section:
                return False
            minuend = observed_parity_forms(
                color=color_by_section[saver],
                section=saver,
                length=length,
                memory=memory,
                message_lens=message_lens,
                gijs=gijs,
                offsets=offsets,
                bits_per_user=bits_per_user,
            )
            sub_pieces = []
            for decider in who_decides_p_sec(length, saver, memory):
                if decider == lost_section:
                    continue
                info = candidate_info_forms(
                    section=decider,
                    color_by_section=color_by_section,
                    recovered=recovered,
                    message_lens=message_lens,
                    offsets=offsets,
                    bits_per_user=bits_per_user,
                )
                if info is None:
                    return False
                sub_pieces.append(block_apply(info, gijs[cantor_pairing(decider, saver)]))
            subtrahend = xor_vectors(*sub_pieces) if sub_pieces else tuple(0 for _ in range(int(parity_lens[saver])))
            known_vectors.append(xor_vectors(minuend, subtrahend))

        known = tuple(item for vector in known_vectors for item in vector)
        answer = solve_info_forms(known, lost_section=lost_section, solve_data=solve_data, message_lens=message_lens)

        if len(savers) == memory:
            reconstructed = block_apply(answer, np.asarray(gis[lost_section], dtype=np.uint8))
            for lhs, rhs in zip(reconstructed, known):
                eq = lhs ^ rhs
                if eq:
                    equations.append(eq)
        recovered[lost_section] = answer
    return True


def ordered_profile_exponent(
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
) -> int | None:
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

    for section in range(length):
        if section in erased:
            continue
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

    return bit_rank(equations)


def restricted_growth_profiles(length: int, colors: int):
    if colors <= 0 or length <= 0:
        return
    profile = [0]

    def rec(pos: int, max_color: int):
        if pos == length:
            if max_color + 1 == colors:
                yield tuple(profile)
            return
        upper = min(colors - 1, max_color + 1)
        for color in range(upper + 1):
            new_max = max(max_color, color)
            missing = colors - (new_max + 1)
            if missing > length - pos - 1:
                continue
            profile.append(color)
            yield from rec(pos + 1, new_max)
            profile.pop()

    yield from rec(1, 0)


@lru_cache(maxsize=None)
def profiles_for(length: int, colors: int) -> tuple[tuple[int, ...], ...]:
    return tuple(restricted_growth_profiles(length, colors))


def falling_factorial(n: int, k: int) -> int:
    out = 1
    for offset in range(k):
        out *= n - offset
    return out


def rotate_code(length: int, memory: int, seed: int, root: int):
    static_repo, gu = load_playground()
    random.seed(seed)
    message_lens, parity_lens = static_repo.get_allocation(length)
    gis, columns_index, sub_g_invs = static_repo.get_G_info(length, memory, message_lens, parity_lens, seed=seed)
    message_lens = message_lens.copy()
    parity_lens = parity_lens.copy()
    gis = gis.copy()
    columns_index = columns_index.copy()
    sub_g_invs = sub_g_invs.copy()
    message_lens[range(length)] = message_lens[np.mod(np.arange(root, root + length), length)]
    parity_lens[range(length)] = parity_lens[np.mod(np.arange(root, root + length), length)]
    gis[range(length)] = gis[np.mod(np.arange(root, root + length), length)]
    columns_index[range(length)] = columns_index[np.mod(np.arange(root, root + length), length)]
    sub_g_invs[range(length)] = sub_g_invs[np.mod(np.arange(root, root + length), length)]
    gijs = static_repo.partition_Gs(length, memory, parity_lens, gis)
    raw_solve_cache = gu.build_solve_cache(length, memory, columns_index, sub_g_invs, gis)
    solve_data = {
        section: (np.asarray(value[0], dtype=int), np.asarray(value[1], dtype=np.uint8))
        for section, value in raw_solve_cache.items()
    }
    return message_lens, parity_lens, gis, gijs, solve_data


def accepted_rotated_masks(
    *,
    length: int,
    memory: int,
    seed: int,
    root: int,
    d: int,
    erasure_slot: set[int],
    weights: set[int],
) -> list[int]:
    message_lens, _parity_lens, gijs = load_repo_code_matrices(length, memory, seed)
    masks = set()
    for original_mask in range(1 << length):
        if weights and original_mask.bit_count() not in weights:
            continue
        if attempt_succeeds(
            original_mask,
            length=length,
            memory=memory,
            root=root,
            d=d,
            erasure_slot=erasure_slot,
            message_lens=message_lens,
            gijs=gijs,
        ):
            masks.add(rotate_mask(original_mask, length, root))
    return sorted(masks)


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
            by_gap.setdefault(circular_gap_pair(mask_to_sections(mask, args.L), args.L), mask)
        masks = list(by_gap.values())
    if args.max_masks_per_attempt > 0:
        masks = masks[: args.max_masks_per_attempt]
    return masks


def build_rows(args: argparse.Namespace) -> list[OrderedRow]:
    rows = []
    for name, root, d, erasure_slot in schedule_attempts(args.L, args.phase):
        if args.only_attempt and args.only_attempt not in name:
            continue
        message_lens, parity_lens, gis, gijs, solve_data = rotate_code(args.L, args.M, args.seed, root)
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
        for colors in args.colors:
            by_key: Counter[tuple[int, int, int]] = Counter()
            rejected = 0
            for mask in masks:
                known_count = args.L - mask.bit_count()
                for profile in profiles_for(known_count, colors):
                    exponent = ordered_profile_exponent(
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
                    if exponent is None:
                        rejected += 1
                        continue
                    by_key[(mask.bit_count(), known_count, exponent)] += 1
            for (weight, known_count, exponent), count in sorted(by_key.items()):
                rows.append(
                    OrderedRow(
                        attempt=name,
                        weight=weight,
                        colors=colors,
                        known_sections=known_count,
                        accepted_masks=len(masks),
                        canonical_profiles=count,
                        exponent=exponent,
                        rejected_profiles=rejected,
                    )
                )
    return rows


def build_report(args: argparse.Namespace) -> str:
    rows = build_rows(args)
    schedule_bad = schedule_bad_masks(args.L, args.M, args.phase, args.seed)
    lines = [
        "# UACE Ordered Identity-Profile Bound",
        "",
        "This report is generated by `research/uace_ordered_profile_bound.py`.",
        "",
        f"- `L = {args.L}`",
        f"- `M = {args.M}`",
        f"- phase: `{args.phase}`",
        f"- weights: `{args.weights}`",
        f"- colors: `{args.colors}`",
        "",
        "The exponent is the rank of the linear equations accumulated by the current sequential decoder along a fixed identity profile.  Erased sections are recovered exactly as the implementation recovers them, so they are no longer existential free variables.",
        "",
        "## Exponent Inventory",
        "",
        "| attempt | weight | colors | known sections | accepted masks | surviving profiles | exponent | rejected profiles |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.attempt} | {row.weight} | {row.colors} | {row.known_sections} | "
            f"{row.accepted_masks} | {row.canonical_profiles} | {row.exponent} | {row.rejected_profiles} |"
        )

    lines.extend(
        [
            "",
            "## Contribution Scale",
            "",
            "| K | pe | schedule UE | colors | E ordered profiles | per-user scale | dominant exponent |",
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
                    row.canonical_profiles
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
            "Readout:",
            "",
            "- This is still a profile first moment, but it is order-aware: it includes intermediate recovery equations and rejection checks before final parity.",
            "- Comparing this report with `uace_identity_profile_bound.md` quantifies how much protection comes from sequential recovery rather than final parity alone.",
            "- If `rejected profiles` is zero and the exponent inventory matches the final-parity identity-profile report, then sequential recovery alone is not the missing protection; the next refinement must model first-valid-path ordering and dependency/coalescence among many profiles.",
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
    parser.add_argument("--output", type=Path, default=Path("research/uace_ordered_profile_bound.md"))
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
