#!/usr/bin/env python3
"""Validate the erasure-mask schedule classifier against the actual decoder.

The validation uses K=1, so there are no symbol collisions or false paths.  In
this regime, discrepancies between the actual decoder and schedule classifier
point to root/order/modeling mismatches rather than A-channel ambiguity.
"""

from __future__ import annotations

import argparse
import itertools
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from uace_empirical_probe import append_unique, rows_as_set
from uace_schedule_bound import mask_to_sections, schedule_succeeds
from uace_bound_explorer import rank_peeling_succeeds


@dataclass(frozen=True)
class MaskResult:
    mask: int
    weight: int
    actual_success: bool
    schedule_success: bool
    rank_success: bool


def load_playground():
    repo_root = Path(__file__).resolve().parents[1]
    playground = repo_root / "playground"
    sys.path.insert(0, str(playground))

    import abch_utils  # type: ignore
    import general_lib  # type: ignore
    import static_repo  # type: ignore
    import joblib

    general_lib.Parallel = lambda n_jobs=None: joblib.Parallel(n_jobs=1, backend="threading")
    return static_repo, general_lib, abch_utils


def masks_up_to_weight(length: int, max_weight: int) -> list[int]:
    masks = []
    for weight in range(max_weight + 1):
        for combo in itertools.combinations(range(length), weight):
            mask = 0
            for section in combo:
                mask |= 1 << section
            masks.append(mask)
    return masks


def erased_symbols_for_mask(tx_symbols: np.ndarray, mask: int, length: int) -> np.ndarray:
    rx_symbols = tx_symbols.copy()
    for section in range(length):
        if (mask >> section) & 1:
            rx_symbols[0, section] = -1
    return rx_symbols


def decode_mask(args: argparse.Namespace, mask: int) -> MaskResult:
    static_repo, general_lib, abch_utils = load_playground()
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    message_lens, parity_lens = static_repo.get_allocation(args.L)
    gis, columns_index, sub_g_invs = static_repo.get_G_info(
        args.L, args.M, message_lens, parity_lens, seed=args.seed
    )
    gijs = static_repo.partition_Gs(args.L, args.M, parity_lens, gis)

    tx_bits = rng.integers(0, 2, size=(1, static_repo.B), dtype=int)
    tx_cdwds = general_lib.encode(
        tx_bits,
        1,
        args.L,
        args.L * static_repo.J,
        args.M,
        message_lens,
        parity_lens,
        gijs,
    )
    tx_symbols = abch_utils.binary_to_symbol(tx_cdwds, args.L, 1)
    rx_symbols = erased_symbols_for_mask(tx_symbols, mask, args.L)
    grand_list = abch_utils.symbol_to_binary(1, args.L, rx_symbols)

    decoded = np.empty(shape=(0, 0), dtype=int)
    rx_p1, grand_list = general_lib.phase1_decoder(
        grand_list,
        args.L,
        gijs,
        message_lens,
        parity_lens,
        1,
        args.M,
        SIC=False,
        toPrint=False,
    )
    decoded = append_unique(decoded, rx_p1)

    if args.phase >= 2:
        rx_p21, grand_list = general_lib.phase2plus_decoder(
            1,
            grand_list,
            args.L,
            gis,
            columns_index,
            sub_g_invs,
            message_lens,
            parity_lens,
            1,
            args.M,
            SIC=False,
            toPrint=False,
        )
        decoded = append_unique(decoded, rx_p21)

        if args.L > 8:
            rx_p22, grand_list = general_lib.phase2plus_decoder(
                1,
                grand_list,
                args.L,
                gis,
                columns_index,
                sub_g_invs,
                message_lens,
                parity_lens,
                1,
                args.M,
                SIC=False,
                pChosenRoots=[8],
                toPrint=False,
            )
            decoded = append_unique(decoded, rx_p22)

    if args.phase >= 3:
        for roots in (None, [6], [6, 10]):
            kwargs = {} if roots is None else {"pChosenRoots": roots}
            rx_p3, grand_list = general_lib.phase2plus_decoder(
                2,
                grand_list,
                args.L,
                gis,
                columns_index,
                sub_g_invs,
                message_lens,
                parity_lens,
                1,
                args.M,
                SIC=False,
                toPrint=False,
                **kwargs,
            )
            decoded = append_unique(decoded, rx_p3)

    true_set = rows_as_set(tx_bits)
    actual_success = bool(true_set & rows_as_set(decoded))
    schedule_success = schedule_succeeds(
        mask,
        length=args.L,
        memory=args.M,
        phase=args.phase,
        message_lens=message_lens,
        gijs=gijs,
    )
    rank_success = rank_peeling_succeeds(mask, args.L, args.M, message_lens, gijs)
    return MaskResult(mask, mask.bit_count(), actual_success, schedule_success, rank_success)


def summarize(results: list[MaskResult], args: argparse.Namespace) -> str:
    mismatches = [item for item in results if item.actual_success != item.schedule_success]
    rank_gap = [item for item in results if item.schedule_success != item.rank_success]
    lines = [
        "# UACE Mask Decoder Validation",
        "",
        "This report is generated by `research/uace_mask_decoder_validation.py`.",
        "",
        f"- `K = 1`",
        f"- `L = {args.L}`",
        f"- `M = {args.M}`",
        f"- phase: `{args.phase}`",
        f"- max erasure weight checked: `{args.max_weight}`",
        f"- masks checked: `{len(results)}`",
        f"- actual/schedule mismatches: `{len(mismatches)}`",
        f"- schedule/rank disagreements: `{len(rank_gap)}`",
        "",
        "| weight | masks | actual success | schedule success | rank success |",
        "|---:|---:|---:|---:|---:|",
    ]
    for weight in range(args.max_weight + 1):
        bucket = [item for item in results if item.weight == weight]
        if not bucket:
            continue
        lines.append(
            f"| {weight} | {len(bucket)} | "
            f"{sum(item.actual_success for item in bucket)} | "
            f"{sum(item.schedule_success for item in bucket)} | "
            f"{sum(item.rank_success for item in bucket)} |"
        )

    if mismatches:
        lines.extend(["", "First actual/schedule mismatches:", ""])
        lines.append("| mask sections | actual | schedule | rank |")
        lines.append("|---|---:|---:|---:|")
        for item in mismatches[: args.show]:
            lines.append(
                f"| {mask_to_sections(item.mask, args.L)} | "
                f"{int(item.actual_success)} | {int(item.schedule_success)} | {int(item.rank_success)} |"
            )

    if rank_gap:
        lines.extend(["", "First schedule/rank disagreements:", ""])
        lines.append("| mask sections | actual | schedule | rank |")
        lines.append("|---|---:|---:|---:|")
        for item in rank_gap[: args.show]:
            lines.append(
                f"| {mask_to_sections(item.mask, args.L)} | "
                f"{int(item.actual_success)} | {int(item.schedule_success)} | {int(item.rank_success)} |"
            )

    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Zero actual/schedule mismatches means the erasure-only schedule classifier matches the current decoder on the checked masks.",
            "- Schedule/rank disagreements are recoverability headroom: the ideal rank-peeling process succeeds where the current phase/root schedule fails.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--M", type=int, default=3)
    parser.add_argument("--phase", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--max-weight", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = [decode_mask(args, mask) for mask in masks_up_to_weight(args.L, args.max_weight)]
    content = summarize(results, args)
    if args.output is None:
        print(content)
    else:
        args.output.write_text(content + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
