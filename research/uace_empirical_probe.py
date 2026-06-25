#!/usr/bin/env python3
"""Empirical probe for LLC/UACE simulations.

This is a research-side wrapper around the existing playground encoder and
decoder functions.  It does not modify decoder behavior.  The wrapper adds
structured PDP/PHP metrics and erasure-count statistics so the analytical
bound tables can be compared with simulation trends.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from uace_bound_explorer import rank_peeling_succeeds
from uace_schedule_bound import schedule_succeeds


@dataclass(frozen=True)
class TrialMetrics:
    seed: int
    pdp: float
    php: float
    decoded: int
    correct: int
    false_positive: int
    p0_erasure_emp: float
    p1_erasure_emp: float
    pge2_erasure_emp: float
    schedule_fail_emp: float
    rank_fail_emp: float


def load_playground():
    repo_root = Path(__file__).resolve().parents[1]
    playground = repo_root / "playground"
    sys.path.insert(0, str(playground))

    import abch_utils  # type: ignore
    import general_lib  # type: ignore
    import static_repo  # type: ignore
    import joblib

    requested_jobs = int(os.environ.get("LLC_PROBE_N_JOBS", "1"))
    requested_backend = os.environ.get("LLC_PROBE_BACKEND", "threading")
    if requested_jobs:
        general_lib.Parallel = lambda n_jobs=None: joblib.Parallel(
            n_jobs=requested_jobs, backend=requested_backend
        )
    return static_repo, general_lib, abch_utils


def uace_channel(tx_symbols: np.ndarray, pe: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k, length = tx_symbols.shape
    erasure_mask = rng.random((k, length)) < pe
    b_output = -1 * np.ones_like(tx_symbols)
    for section in range(length):
        kept = tx_symbols[~erasure_mask[:, section], section]
        b_output[: len(kept), section] = kept

    a_output = -1 * np.ones_like(tx_symbols)
    for section in range(length):
        unique = np.unique(b_output[:, section])
        unique = unique[unique != -1]
        a_output[: len(unique), section] = unique
    mask_values = np.zeros(k, dtype=int)
    for user in range(k):
        for section in range(length):
            if erasure_mask[user, section]:
                mask_values[user] |= 1 << section
    return a_output, erasure_mask.sum(axis=1), mask_values


def rows_as_set(bits: np.ndarray) -> set[bytes]:
    if bits.size == 0:
        return set()
    if bits.ndim == 1:
        bits = bits.reshape(1, -1)
    return {np.asarray(row, dtype=np.uint8).tobytes() for row in bits}


def append_unique(existing: np.ndarray, new: np.ndarray) -> np.ndarray:
    if new.size == 0:
        return existing
    if new.ndim == 1:
        new = new.reshape(1, -1)
    if existing.size == 0:
        return np.unique(new, axis=0)
    return np.unique(np.vstack((existing, new)), axis=0)


def run_trial(args: argparse.Namespace, seed: int) -> TrialMetrics:
    static_repo, general_lib, abch_utils = load_playground()
    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)

    message_lens, parity_lens = static_repo.get_allocation(args.L)
    gis, columns_index, sub_g_invs = static_repo.get_G_info(
        args.L, args.M, message_lens, parity_lens, seed=seed
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

    schedule_fail_emp = float(
        np.mean(
            [
                not schedule_succeeds(
                    int(mask),
                    length=args.L,
                    memory=args.M,
                    phase=args.phase,
                    message_lens=message_lens,
                    gijs=gijs,
                )
                for mask in erasure_masks
            ]
        )
    )
    rank_fail_emp = float(
        np.mean(
            [
                not rank_peeling_succeeds(
                    int(mask),
                    args.L,
                    args.M,
                    message_lens,
                    gijs,
                )
                for mask in erasure_masks
            ]
        )
    )

    rx_p1, grand_list = general_lib.phase1_decoder(
        grand_list, args.L, gijs, message_lens, parity_lens, args.K, args.M, SIC=bool(args.sic), toPrint=False
    )
    if args.match_existing_truncation and rx_p1.shape[0] > args.K:
        rx_p1 = rx_p1[np.arange(args.K)]

    decoded = np.empty(shape=(0, 0), dtype=int)
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
            args.K,
            args.M,
            SIC=bool(args.sic),
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
                args.K,
                args.M,
                SIC=bool(args.sic),
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
                args.K,
                args.M,
                SIC=bool(args.sic),
                toPrint=False,
                **kwargs,
            )
            decoded = append_unique(decoded, rx_p3)

    true_set = rows_as_set(tx_bits)
    decoded_set = rows_as_set(decoded)
    correct = len(true_set & decoded_set)
    false_positive = len(decoded_set - true_set)
    pdp = 1.0 - (correct / args.K)
    php = false_positive / args.K

    return TrialMetrics(
        seed=seed,
        pdp=pdp,
        php=php,
        decoded=len(decoded_set),
        correct=correct,
        false_positive=false_positive,
        p0_erasure_emp=float(np.mean(erasure_counts == 0)),
        p1_erasure_emp=float(np.mean(erasure_counts == 1)),
        pge2_erasure_emp=float(np.mean(erasure_counts >= 2)),
        schedule_fail_emp=schedule_fail_emp,
        rank_fail_emp=rank_fail_emp,
    )


def summarize(metrics: list[TrialMetrics], args: argparse.Namespace) -> str:
    def mean(field: str) -> float:
        return float(np.mean([getattr(item, field) for item in metrics]))

    def std(field: str) -> float:
        return float(np.std([getattr(item, field) for item in metrics], ddof=0))

    lines = [
        "# UACE Empirical Probe",
        "",
        f"- `K = {args.K}`",
        f"- `L = {args.L}`",
        f"- `J = 16`",
        f"- `M = {args.M}`",
        f"- `p_e = {args.pe}`",
        f"- `phase = {args.phase}`",
        f"- `SIC = {args.sic}`",
        f"- trials: `{args.trials}`",
        "",
        "| metric | mean | std |",
        "|---|---:|---:|",
    ]
    for field in (
        "pdp",
        "php",
        "decoded",
        "correct",
        "false_positive",
        "p0_erasure_emp",
        "p1_erasure_emp",
        "pge2_erasure_emp",
        "schedule_fail_emp",
        "rank_fail_emp",
    ):
        lines.append(f"| {field} | {mean(field):.6f} | {std(field):.6f} |")

    lines.extend(
        [
            "",
            "Per-trial rows:",
            "",
            "| seed | PDP | PHP | decoded | correct | false positive | P0 era | P1 era | P>=2 era | schedule fail | rank fail |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in metrics:
        lines.append(
            f"| {item.seed} | {item.pdp:.6f} | {item.php:.6f} | {item.decoded} | "
            f"{item.correct} | {item.false_positive} | {item.p0_erasure_emp:.6f} | "
            f"{item.p1_erasure_emp:.6f} | {item.pge2_erasure_emp:.6f} | "
            f"{item.schedule_fail_emp:.6f} | {item.rank_fail_emp:.6f} |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--M", type=int, default=3)
    parser.add_argument("--pe", type=float, default=0.1)
    parser.add_argument("--phase", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--sic", type=int, default=0, choices=(0, 1))
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--match-existing-truncation", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = [run_trial(args, args.seed + idx) for idx in range(args.trials)]
    content = summarize(metrics, args)
    if args.output is None:
        print(content)
    else:
        args.output.write_text(content + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
