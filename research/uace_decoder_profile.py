#!/usr/bin/env python3
"""Profile current LLC decoder path growth on one UACE trial."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

from uace_empirical_probe import append_unique, uace_channel


def load_playground():
    repo_root = Path(__file__).resolve().parents[1]
    playground = repo_root / "playground"
    sys.path.insert(0, str(playground))
    import abch_utils  # type: ignore
    import general_lib  # type: ignore
    import general_utils  # type: ignore
    import linkedloop as LLC  # type: ignore
    import static_repo  # type: ignore

    return static_repo, general_lib, general_utils, LLC, abch_utils


def profile_phase2_attempt(
    *,
    name: str,
    d: int,
    chosen_roots: list[int] | None,
    grand_list: np.ndarray,
    L: int,
    M: int,
    K: int,
    message_lens: np.ndarray,
    parity_lens: np.ndarray,
    gis: np.ndarray,
    columns_index: np.ndarray,
    sub_g_invs: np.ndarray,
    cap: int,
) -> str:
    static_repo, general_lib, gu, LLC, _abch_utils = load_playground()

    chosen_root = 0 if chosen_roots is None else chosen_roots[-1]
    erasure_slot = [np.mod(0 - root, L) for root in chosen_roots] if chosen_roots is not None else []

    message_lens = message_lens.copy()
    parity_lens = parity_lens.copy()
    gis = gis.copy()
    columns_index = columns_index.copy()
    sub_g_invs = sub_g_invs.copy()
    grand = grand_list.copy()

    message_lens[range(L)] = message_lens[np.mod(np.arange(chosen_root, chosen_root + L), L)]
    parity_lens[range(L)] = parity_lens[np.mod(np.arange(chosen_root, chosen_root + L), L)]
    gis[range(L)] = gis[np.mod(np.arange(chosen_root, chosen_root + L), L)]
    columns_index[range(L)] = columns_index[np.mod(np.arange(chosen_root, chosen_root + L), L)]
    sub_g_invs[range(L)] = sub_g_invs[np.mod(np.arange(chosen_root, chosen_root + L), L)]
    grand[:, range(L * static_repo.J)] = grand[:, np.mod(np.arange(chosen_root * static_repo.J, chosen_root * static_repo.J + L * static_repo.J), L * static_repo.J)]
    gijs = static_repo.partition_Gs(L, M, parity_lens, gis)

    decoders_cache, avail_savers_cache = gu.build_decoder_lookup(L, M)
    solve_cache = gu.build_solve_cache(L, M, columns_index, sub_g_invs, gis)
    valid_ks_by_section = [np.flatnonzero(grand[:, l * static_repo.J] != -1) for l in range(L)]
    parity_lookup_by_section = gu.build_parity_lookup_by_section(grand, L, message_lens, valid_ks_by_section)
    parity_cache = gu.build_parity_cache(grand, L, M, message_lens, gijs)
    k_effective = [x for x in range(K) if grand[x, 0] != -1]

    lines = [
        f"## {name}",
        "",
        f"- chosen root: `{chosen_root}`",
        f"- erasure slot: `{erasure_slot}`",
        f"- effective roots: `{len(k_effective)}`",
        "",
        "| root idx | section | paths before | paths after | seconds | capped |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    attempt_start = time.time()
    capped = False

    for root_count, root in enumerate(k_effective):
        paths = [LLC.GLinkedLoop([root], message_lens)]
        for section in range(1, L):
            before = len(paths)
            if before == 0:
                break
            t0 = time.time()
            new_all = []
            for path in paths:
                new_all.extend(
                    gu.Path_goes_section_l(
                        section,
                        path,
                        d,
                        grand,
                        K,
                        message_lens,
                        parity_lens,
                        L,
                        M,
                        gis,
                        gijs,
                        columns_index,
                        sub_g_invs,
                        erasure_slot,
                        valid_ks_by_section=valid_ks_by_section,
                        parity_cache=parity_cache,
                        deciders_cache=decoders_cache,
                        avail_savers_cache=avail_savers_cache,
                        parity_lookup_by_section=parity_lookup_by_section,
                        solve_cache=solve_cache,
                    )
                )
            paths = new_all
            elapsed = time.time() - t0
            capped_here = len(paths) > cap
            lines.append(
                f"| {root_count} | {section} | {before} | {len(paths)} | {elapsed:.4f} | {int(capped_here)} |"
            )
            if capped_here:
                capped = True
                break
        if capped:
            break
    lines.extend(["", f"attempt seconds before cap/end: `{time.time() - attempt_start:.3f}`", ""])
    return "\n".join(lines)


def build_report(args: argparse.Namespace) -> str:
    static_repo, general_lib, _gu, _LLC, abch_utils = load_playground()
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    message_lens, parity_lens = static_repo.get_allocation(args.L)
    gis, columns_index, sub_g_invs = static_repo.get_G_info(
        args.L, args.M, message_lens, parity_lens, seed=args.seed
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
    rx_symbols, erasure_counts, erasure_masks = uace_channel(tx_symbols, args.pe, rng)
    grand_list = abch_utils.symbol_to_binary(args.K, args.L, rx_symbols)

    lines = [
        "# UACE Decoder Path Profile",
        "",
        f"- `K = {args.K}`",
        f"- `L = {args.L}`",
        f"- `M = {args.M}`",
        f"- `pe = {args.pe}`",
        f"- `seed = {args.seed}`",
        f"- mean erasures per user: `{float(np.mean(erasure_counts)):.3f}`",
        f"- users with >=2 erasures: `{float(np.mean(erasure_counts >= 2)):.6f}`",
        "",
    ]

    if args.run_phase1:
        t0 = time.time()
        rx_p1, grand_list = general_lib.phase1_decoder(
            grand_list,
            args.L,
            gijs,
            message_lens,
            parity_lens,
            args.K,
            args.M,
            SIC=False,
            toPrint=False,
        )
        lines.extend([f"- phase-I decoded rows: `{rx_p1.shape[0]}`", f"- phase-I seconds: `{time.time() - t0:.3f}`", ""])

    attempts = [("phase-II root 0", 1, None), ("phase-II root 8", 1, [8])]
    if args.phase >= 3:
        attempts.extend(
            [
                ("phase-III root 0", 2, None),
                ("phase-III root 6", 2, [6]),
                ("phase-III root 10", 2, [6, 10]),
            ]
        )
    for name, d, roots in attempts:
        lines.append(
            profile_phase2_attempt(
                name=name,
                d=d,
                chosen_roots=roots,
                grand_list=grand_list,
                L=args.L,
                M=args.M,
                K=args.K,
                message_lens=message_lens,
                parity_lens=parity_lens,
                gis=gis,
                columns_index=columns_index,
                sub_g_invs=sub_g_invs,
                cap=args.cap,
            )
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--K", type=int, default=40)
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--M", type=int, default=3)
    parser.add_argument("--pe", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--phase", type=int, default=2, choices=(2, 3))
    parser.add_argument("--cap", type=int, default=250000)
    parser.add_argument("--run-phase1", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
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
