#!/usr/bin/env python3
"""Direct two-user pair preemption probe for the LLC/UACE decoder.

For a tagged user with a fixed accepted erasure mask, this script adds one
alternate user, samples the alternate user's erasures, and runs the same
first-valid-path search used by the research-side fast decoder.  A pair
preemption occurs when the first valid path from the tagged root decodes to a
message different from the tagged user's message.

This is the closest executable object to the pair-level theorem target:
it includes decoder row order, erasure availability of the alternate user,
recovery equations, and the final parity check.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from uace_bound_explorer import bad_mask_probability, format_probability
from uace_ordered_profile_bound import accepted_rotated_masks, filter_masks
from uace_schedule_bound import mask_to_sections, schedule_attempts, schedule_bad_masks


DEFAULT_KS = (30, 40, 100)
DEFAULT_PES = (0.1, 0.2, 0.3)


@dataclass(frozen=True)
class PairRow:
    attempt: str
    mask: int
    multiplicity: int
    pe: float
    trials: int
    preemptions: int
    path_failures: int
    aborted: int
    node_visits: int


def load_playground():
    repo_root = Path(__file__).resolve().parents[1]
    playground = repo_root / "playground"
    sys.path.insert(0, str(playground))
    import abch_utils  # type: ignore
    import general_lib  # type: ignore
    import general_utils as gu  # type: ignore
    import static_repo  # type: ignore

    return static_repo, general_lib, gu, abch_utils


def rotate_codewords(codewords: np.ndarray, root: int, length: int, section_bits: int) -> np.ndarray:
    rotated = codewords.copy()
    rotated[:, range(length * section_bits)] = codewords[
        :,
        np.mod(
            np.arange(root * section_bits, root * section_bits + length * section_bits),
            length * section_bits,
        ),
    ]
    return rotated


def symbol_value(codeword: np.ndarray, section: int, section_bits: int) -> int:
    bits = codeword[section * section_bits : (section + 1) * section_bits]
    return int(np.asarray(bits, dtype=int) @ (2 ** np.arange(section_bits - 1, -1, -1)))


def build_pair_grand(
    *,
    abch_utils,
    rotated_codewords: np.ndarray,
    tagged_mask: int,
    alt_mask: int,
    length: int,
    section_bits: int,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    b_symbols = -1 * np.ones((2, length), dtype=int)
    for section in range(length):
        if ((tagged_mask >> section) & 1) == 0:
            b_symbols[0, section] = symbol_value(rotated_codewords[0], section, section_bits)
        if ((alt_mask >> section) & 1) == 0:
            b_symbols[1, section] = symbol_value(rotated_codewords[1], section, section_bits)

    a_symbols = -1 * np.ones((2, length), dtype=int)
    for section in range(length):
        unique = np.unique(b_symbols[:, section])
        unique = unique[unique != -1]
        a_symbols[: len(unique), section] = unique

    tagged_root_symbol = symbol_value(rotated_codewords[0], 0, section_bits)
    root_rows = np.where(a_symbols[:, 0] == tagged_root_symbol)[0]
    root_row = int(root_rows[0]) if len(root_rows) else None
    return abch_utils.symbol_to_binary(2, length, a_symbols), a_symbols, root_row


def rotated_message_bits(rotated_codeword: np.ndarray, message_lens: np.ndarray, length: int, section_bits: int) -> np.ndarray:
    pieces = []
    for section in range(length):
        start = section * section_bits
        stop = start + int(message_lens[section])
        pieces.append(rotated_codeword[start:stop])
    return np.concatenate(pieces).astype(int)


def build_decoder_caches(gu, grand, length, memory, section_bits, message_lens, gijs):
    deciders_cache, avail_savers_cache = gu.build_decoder_lookup(length, memory)
    valid_ks_by_section = [np.flatnonzero(grand[:, section * section_bits] != -1) for section in range(length)]
    parity_lookup_by_section = gu.build_parity_lookup_by_section(grand, length, message_lens, valid_ks_by_section)
    parity_cache = gu.build_parity_cache(grand, length, memory, message_lens, gijs)
    return deciders_cache, avail_savers_cache, valid_ks_by_section, parity_lookup_by_section, parity_cache


def rotate_decoder_matrices(static_repo, gu, *, length: int, memory: int, seed: int, root: int):
    random.seed(seed)
    message_lens, parity_lens = static_repo.get_allocation(length)
    gis, columns_index, sub_g_invs = static_repo.get_G_info(
        length, memory, message_lens, parity_lens, seed=seed
    )

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
    solve_cache = gu.build_solve_cache(length, memory, columns_index, sub_g_invs, gis)
    return message_lens, parity_lens, gis, columns_index, sub_g_invs, gijs, solve_cache


def sample_mask(length: int, pe: float, rng: np.random.Generator) -> int:
    mask = 0
    for section, erased in enumerate(rng.random(length) < pe):
        if bool(erased):
            mask |= 1 << section
    return mask


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


def run_pair_row(
    *,
    args: argparse.Namespace,
    attempt: str,
    root: int,
    d: int,
    erasure_slot: set[int],
    mask: int,
    multiplicity: int,
    pe: float,
    rng: np.random.Generator,
) -> PairRow:
    static_repo, general_lib, gu, abch_utils = load_playground()
    import uace_fast_empirical_probe as fast_probe

    fast_probe.gu_global = gu

    random.seed(args.seed)
    np.random.seed(args.seed)
    raw_msg_lens, raw_par_lens = static_repo.get_allocation(args.L)
    raw_gis, _raw_columns, _raw_invs = static_repo.get_G_info(
        args.L, args.M, raw_msg_lens, raw_par_lens, seed=args.seed
    )
    raw_gijs = static_repo.partition_Gs(args.L, args.M, raw_par_lens, raw_gis)

    msg_lens, par_lens, gis, columns_index, sub_g_invs, gijs, solve_cache = rotate_decoder_matrices(
        static_repo,
        gu,
        length=args.L,
        memory=args.M,
        seed=args.seed,
        root=root,
    )

    preemptions = 0
    path_failures = 0
    aborted = 0
    node_visits = 0
    for _trial in range(args.pair_trials):
        tx_bits = rng.integers(0, 2, size=(2, static_repo.B), dtype=int)
        codewords = general_lib.encode(
            tx_bits,
            2,
            args.L,
            args.L * static_repo.J,
            args.M,
            raw_msg_lens,
            raw_par_lens,
            raw_gijs,
        )
        rotated = rotate_codewords(codewords, root, args.L, static_repo.J)
        alt_mask = sample_mask(args.L, pe, rng)
        grand, _symbols, root_row = build_pair_grand(
            abch_utils=abch_utils,
            rotated_codewords=rotated,
            tagged_mask=mask,
            alt_mask=alt_mask,
            length=args.L,
            section_bits=static_repo.J,
        )
        if root_row is None:
            path_failures += 1
            continue

        deciders_cache, avail_savers_cache, valid_ks_by_section, parity_lookup_by_section, parity_cache = (
            build_decoder_caches(gu, grand, args.L, args.M, static_repo.J, msg_lens, gijs)
        )
        caches = (
            valid_ks_by_section,
            parity_cache,
            deciders_cache,
            avail_savers_cache,
            parity_lookup_by_section,
            solve_cache,
        )
        path, visits, did_abort = fast_probe.first_valid_path_for_root(
            root=root_row,
            d=d,
            grand=grand,
            K=2,
            L=args.L,
            M=args.M,
            message_lens=msg_lens,
            parity_lens=par_lens,
            gis=gis,
            gijs=gijs,
            columns_index=columns_index,
            sub_g_invs=sub_g_invs,
            erasure_slot=erasure_slot,
            caches=caches,
            LLC=__import__("linkedloop"),
            max_nodes=args.max_nodes_per_pair,
        )
        node_visits += visits
        if did_abort:
            aborted += 1
            path_failures += 1
            continue
        if path is None:
            path_failures += 1
            continue

        decoded = gu.output_message_oop(grand, [path], args.L, static_repo.J)[0]
        tagged = rotated_message_bits(rotated[0], msg_lens, args.L, static_repo.J)
        if not np.array_equal(decoded, tagged):
            preemptions += 1

    return PairRow(
        attempt=attempt,
        mask=mask,
        multiplicity=multiplicity,
        pe=pe,
        trials=args.pair_trials,
        preemptions=preemptions,
        path_failures=path_failures,
        aborted=aborted,
        node_visits=node_visits,
    )


def clopper_zero_upper(successes: int, trials: int, alpha: float = 0.05) -> float:
    if trials <= 0:
        return 1.0
    if successes == 0:
        return 1.0 - alpha ** (1.0 / trials)
    return successes / trials


def build_rows(args: argparse.Namespace) -> list[PairRow]:
    rng = np.random.default_rng(args.seed + 20250622)
    rows = []
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
        if not args.gap_representatives:
            masks = filter_masks(args, masks)
        for mask, multiplicity in selected_masks_with_multiplicity(args, masks):
            for pe in args.pes:
                rows.append(
                    run_pair_row(
                        args=args,
                        attempt=name,
                        root=root,
                        d=d,
                        erasure_slot=erasure_slot,
                        mask=mask,
                        multiplicity=multiplicity,
                        pe=pe,
                        rng=rng,
                    )
                )
    return rows


def build_report(args: argparse.Namespace) -> str:
    rows = build_rows(args)
    schedule_bad = schedule_bad_masks(args.L, args.M, args.phase, args.seed)
    lines = [
        "# UACE Direct Pair Decoder Probe",
        "",
        "This report is generated by `research/uace_pair_decoder_probe.py`.",
        "",
        f"- `L = {args.L}`",
        f"- `M = {args.M}`",
        f"- phase: `{args.phase}`",
        f"- weights: `{args.weights}`",
        f"- pair trials per mask and pe: `{args.pair_trials}`",
        f"- gap representatives: `{args.gap_representatives}`",
        f"- max nodes per pair: `{args.max_nodes_per_pair}`",
        "",
        "Each row conditions on a tagged user's accepted erasure mask, adds one random alternate user, samples alternate erasures with probability `pe`, and runs the actual first-valid-path search from the tagged root.",
        "",
        "## Per-Mask Pair Event",
        "",
        "| attempt | mask sections | mult. | pe | preemptions/trials | path failures | aborted | mean node visits | pair preempt upper |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        upper = clopper_zero_upper(row.preemptions, row.trials)
        mean_visits = row.node_visits / row.trials if row.trials else 0.0
        lines.append(
            f"| {row.attempt} | {mask_to_sections(row.mask, args.L)} | {row.multiplicity} | "
            f"{row.pe:.3f} | {row.preemptions}/{row.trials} | {row.path_failures} | "
            f"{row.aborted} | {mean_visits:.2f} | {upper:.6g} |"
        )

    lines.extend(
        [
            "",
            "## K-Scale Projection",
            "",
            "The extra term below is a pair-level projection over represented accepted masks.  The binomial column uses",
            "",
            "$$",
            "\\sum_{a,m} p_e^{|m|}(1-p_e)^{L-|m|} c_{a,m}\\left[1-(1-q_{a,m})^{K-1}\\right],",
            "$$",
            "",
            "while the union column replaces the bracketed term by $\\min\\{1,(K-1)q_{a,m}\\}$.  The latter is the conservative theorem-shaped upper bound; the former is the predictive independence-scale approximation.",
            "",
            "| K | pe | schedule UE | empirical binomial extra | empirical union extra | 95% binomial upper | 95% union upper |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for k in args.Ks:
        for pe in args.pes:
            schedule_ue = bad_mask_probability(schedule_bad, args.L, pe)
            empirical_binom_extra = 0.0
            empirical_union_extra = 0.0
            upper_binom_extra = 0.0
            upper_union_extra = 0.0
            for row in rows:
                if abs(row.pe - pe) > 1e-12:
                    continue
                q_emp = row.preemptions / row.trials if row.trials else 0.0
                q_upper = clopper_zero_upper(row.preemptions, row.trials)
                pair_emp_binom = 1.0 - (1.0 - q_emp) ** max(k - 1, 0)
                pair_upper_binom = 1.0 - (1.0 - q_upper) ** max(k - 1, 0)
                pair_emp_union = min(1.0, max(k - 1, 0) * q_emp)
                pair_upper_union = min(1.0, max(k - 1, 0) * q_upper)
                mask_prob = (pe ** row.mask.bit_count()) * ((1.0 - pe) ** (args.L - row.mask.bit_count()))
                empirical_binom_extra += row.multiplicity * mask_prob * pair_emp_binom
                empirical_union_extra += row.multiplicity * mask_prob * pair_emp_union
                upper_binom_extra += row.multiplicity * mask_prob * pair_upper_binom
                upper_union_extra += row.multiplicity * mask_prob * pair_upper_union
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(k),
                        f"{pe:.3f}",
                        format_probability(schedule_ue),
                        format_probability(empirical_binom_extra),
                        format_probability(empirical_union_extra),
                        format_probability(upper_binom_extra),
                        format_probability(upper_union_extra),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "Readout:",
            "",
            "- This probe measures the pair-level event that the next theorem should bound analytically.",
            "- Zero preemptions here are stronger evidence than zero full-decoder PHP, because the conditioning intentionally exposes accepted two-erasure masks where profile first moments were loosest.",
            "- The 95% column is still a statistical upper bound, not an LLC theorem.",
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
    parser.add_argument("--max-nodes-per-pair", type=int, default=10000)
    parser.add_argument("--Ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--pes", type=float, nargs="+", default=list(DEFAULT_PES))
    parser.add_argument("--output", type=Path, default=Path("research/uace_pair_decoder_probe.md"))
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
