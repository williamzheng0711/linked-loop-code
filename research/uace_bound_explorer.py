#!/usr/bin/env python3
"""Numerical bound explorer for LLC over the UACE.

The script deliberately avoids touching the simulator.  It computes finite-
length analytical ingredients that can be compared against PDP/PHP simulation
curves in the TCom parameter regime.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_PES = (0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2)


@dataclass(frozen=True)
class BoundRow:
    pe: float
    p_zero_erasure: float
    p_one_erasure: float
    p_two_or_more_erasures: float
    p_geometric_ue: float
    p_clean_root_fail: float
    p_rank_peeling_ue: float | None
    rho_tagged_collision: float
    expected_a_list_size: float
    p_collision_run_exact: float
    p_collision_run_union: float
    tcom_phase1_pdp: float
    tcom_phase1_php: float
    phase2_pdp_proxy: float


def clamp_probability(x: float) -> float:
    return max(0.0, min(1.0, x))


def popcount(mask: int) -> int:
    """Return the number of set bits, compatible with Python 3.9."""
    mask_int = int(mask)
    if hasattr(mask_int, "bit_count"):
        return mask_int.bit_count()
    return bin(mask_int).count("1")


def binomial_pmf(n: int, k: int, p: float) -> float:
    if k < 0 or k > n:
        return 0.0
    return math.comb(n, k) * (p**k) * ((1.0 - p) ** (n - k))


def erasure_count_tail(l: int, pe: float, minimum: int) -> float:
    return sum(binomial_pmf(l, k, pe) for k in range(minimum, l + 1))


def tagged_collision_probability(k: int, j: int, pe: float) -> float:
    """Probability that a tagged non-erased symbol is hidden by another user."""
    q = 2**j
    return 1.0 - (1.0 - (1.0 - pe) / q) ** (k - 1)


def expected_a_list_size(k: int, j: int, pe: float) -> float:
    q = 2**j
    return q * (1.0 - (1.0 - (1.0 - pe) / q) ** k)


def mask_has_circular_run(mask: int, length: int, run: int) -> bool:
    if run <= 0:
        return True
    for start in range(length):
        if all((mask >> ((start + offset) % length)) & 1 for offset in range(run)):
            return True
    return False


def mask_has_clean_window(mask: int, length: int, window: int) -> bool:
    for start in range(length):
        if all(((mask >> ((start + offset) % length)) & 1) == 0 for offset in range(window)):
            return True
    return False


def mask_has_window_overload(mask: int, length: int, window: int, limit: int) -> bool:
    for start in range(length):
        count = sum((mask >> ((start + offset) % length)) & 1 for offset in range(window))
        if count > limit:
            return True
    return False


def bernoulli_mask_probability(mask: int, length: int, p: float) -> float:
    weight = popcount(mask)
    return (p**weight) * ((1.0 - p) ** (length - weight))


def circular_run_probability(length: int, run: int, p: float) -> float:
    """Exact circular run probability by brute force.

    This is intentionally simple because the LLC regimes of interest have
    L <= 16.  The function remains fine up to roughly L = 24.
    """
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    total = 0.0
    for mask in range(1 << length):
        if mask_has_circular_run(mask, length, run):
            total += bernoulli_mask_probability(mask, length, p)
    return clamp_probability(total)


def geometric_unrecoverable_probability(length: int, memory: int, pe: float) -> float:
    """Exact probability of the TCom-style geometric UE event.

    The event combines (i) more than one erasure in a circular window of length
    M and (ii) absence of a clean length-M root window.
    """
    total = 0.0
    for mask in range(1 << length):
        overloaded = mask_has_window_overload(mask, length, memory, limit=1)
        no_clean_root = not mask_has_clean_window(mask, length, memory)
        if overloaded or no_clean_root:
            total += bernoulli_mask_probability(mask, length, pe)
    return clamp_probability(total)


def clean_root_failure_probability(length: int, memory: int, pe: float) -> float:
    total = 0.0
    for mask in range(1 << length):
        if not mask_has_clean_window(mask, length, memory):
            total += bernoulli_mask_probability(mask, length, pe)
    return clamp_probability(total)


def gf2_rank(matrix: np.ndarray) -> int:
    mat = np.array(matrix, dtype=np.uint8, copy=True) & 1
    if mat.size == 0:
        return 0
    rows, cols = mat.shape
    rank = 0
    for col in range(cols):
        pivot = None
        for row in range(rank, rows):
            if mat[row, col]:
                pivot = row
                break
        if pivot is None:
            continue
        if pivot != rank:
            mat[[rank, pivot]] = mat[[pivot, rank]]
        for row in range(rows):
            if row != rank and mat[row, col]:
                mat[row] ^= mat[rank]
        rank += 1
        if rank == rows:
            break
    return rank


def cantor_pairing(i: int, j: int) -> int:
    return ((i + j) * (i + j + 1)) // 2 + i


def who_decides_p_sec(length: int, section: int, memory: int) -> list[int]:
    return [(section - offset) % length for offset in range(memory, 0, -1)]


def saver_sections(length: int, lost_section: int, memory: int) -> list[int]:
    return [(lost_section + offset) % length for offset in range(1, memory + 1)]


def load_repo_code_matrices(length: int, memory: int, seed: int):
    """Load the exact matrix profile used by the playground code."""
    repo_root = Path(__file__).resolve().parents[1]
    playground = repo_root / "playground"
    sys.path.insert(0, str(playground))
    import static_repo  # type: ignore

    random.seed(seed)
    message_lens, parity_lens = static_repo.get_allocation(length)
    gis, _columns_index, _sub_g_invs = static_repo.get_G_info(
        length, memory, message_lens, parity_lens, seed=seed
    )
    gijs = static_repo.partition_Gs(length, memory, parity_lens, gis)
    return np.array(message_lens, dtype=int), np.array(parity_lens, dtype=int), gijs


def rank_peeling_succeeds(
    mask: int,
    length: int,
    memory: int,
    message_lens: np.ndarray,
    gijs: dict,
) -> bool:
    erased = {idx for idx in range(length) if (mask >> idx) & 1}
    known_info = set(range(length)) - erased
    unresolved = set(erased)
    if not unresolved:
        return True

    made_progress = True
    while made_progress and unresolved:
        made_progress = False
        for lost in list(unresolved):
            blocks = []
            for saver in saver_sections(length, lost, memory):
                if saver in erased:
                    continue
                deciders = who_decides_p_sec(length, saver, memory)
                if all(decider == lost or decider in known_info for decider in deciders):
                    blocks.append(np.array(gijs[cantor_pairing(lost, saver)], dtype=np.uint8))
            if not blocks:
                continue
            transfer = np.concatenate(blocks, axis=1)
            if gf2_rank(transfer) >= int(message_lens[lost]):
                unresolved.remove(lost)
                known_info.add(lost)
                made_progress = True
    return not unresolved


def rank_peeling_bad_masks(
    length: int,
    memory: int,
    seed: int,
) -> list[int] | None:
    try:
        message_lens, _parity_lens, gijs = load_repo_code_matrices(length, memory, seed)
    except Exception as exc:  # pragma: no cover - used as a research fallback.
        print(f"warning: could not load repo matrices for rank peeling: {exc}", file=sys.stderr)
        return None

    bad_masks = []
    for mask in range(1 << length):
        if not rank_peeling_succeeds(mask, length, memory, message_lens, gijs):
            bad_masks.append(mask)
    return bad_masks


def bad_mask_probability(bad_masks: list[int] | None, length: int, pe: float) -> float | None:
    if bad_masks is None:
        return None
    return clamp_probability(sum(bernoulli_mask_probability(mask, length, pe) for mask in bad_masks))


def tcom_phase1_bounds(k: int, length: int, j: int, memory: int) -> tuple[float, float]:
    """TCom-style phase-I PDP/PHP union bounds.

    The extraction from the PDF shows the intended scaling:
    (L-M) (K 2^-J)^M for drops and its square for hallucinations.
    """
    branch = k / (2**j)
    starts = max(length - memory, 0)
    pdp = starts * (branch**memory)
    php = (starts**2) * (branch ** (2 * memory))
    return clamp_probability(pdp), clamp_probability(php)


def compute_row(
    k: int,
    length: int,
    j: int,
    memory: int,
    pe: float,
    rank_bad_masks: list[int] | None,
) -> BoundRow:
    p_zero = (1.0 - pe) ** length
    p_one = length * pe * ((1.0 - pe) ** (length - 1))
    p_ge2 = erasure_count_tail(length, pe, 2)
    rho = tagged_collision_probability(k, j, pe)
    exact_run = circular_run_probability(length, memory, rho)
    union_run = clamp_probability(length * (rho**memory))
    tcom_pdp, tcom_php = tcom_phase1_bounds(k, length, j, memory)
    p_geo_ue = geometric_unrecoverable_probability(length, memory, pe)
    p_root_fail = clean_root_failure_probability(length, memory, pe)
    p_rank_ue = bad_mask_probability(rank_bad_masks, length, pe)

    # First phase-II PDP proxy: a one-erasure decoder necessarily drops users
    # with two or more erased sections, plus a small exact collision-run term.
    phase2_pdp_proxy = clamp_probability(p_ge2 + exact_run)

    return BoundRow(
        pe=pe,
        p_zero_erasure=p_zero,
        p_one_erasure=p_one,
        p_two_or_more_erasures=p_ge2,
        p_geometric_ue=p_geo_ue,
        p_clean_root_fail=p_root_fail,
        p_rank_peeling_ue=p_rank_ue,
        rho_tagged_collision=rho,
        expected_a_list_size=expected_a_list_size(k, j, pe),
        p_collision_run_exact=exact_run,
        p_collision_run_union=union_run,
        tcom_phase1_pdp=tcom_pdp,
        tcom_phase1_php=tcom_php,
        phase2_pdp_proxy=phase2_pdp_proxy,
    )


def format_probability(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value == 0.0:
        return "0"
    if 1e-4 <= abs(value) < 1e4:
        return f"{value:.6f}"
    return f"{value:.3e}"


def markdown_table(rows: Iterable[BoundRow], args: argparse.Namespace) -> str:
    row_list = list(rows)
    lines = [
        "# Initial UACE Bound Table",
        "",
        "This table is generated by `research/uace_bound_explorer.py`.",
        "",
        "Regime:",
        "",
        f"- `K = {args.K}`",
        f"- `L = {args.L}`",
        f"- `J = {args.J}`",
        f"- `M = {args.M}`",
        f"- matrix seed for rank peeling: `{args.seed}`",
        "",
        "Columns:",
        "",
        "- `P>=2 era`: unavoidable PDP floor for a decoder that only corrects up to one erased section.",
        "- `UE geom`: exact circular-window unrecoverable probability from the TCom-style separation condition.",
        "- `UE rank`: exact rank-peeling unrecoverable probability for the repo matrix profile.",
        "- `rho`: exact tagged A-channel collision probability.",
        "- `run exact`: exact circular run probability for `M` consecutive tagged collisions.",
        "- `run union`: union bound `L rho^M` for the same event.",
        "- `TCOM PDP/PHP`: phase-I scaling bounds from the TCom analysis.",
        "- `phase-II proxy`: `P>=2 era + run exact`, a first crude trend proxy for one-erasure decoding.",
        "",
        "| pe | P0 era | P1 era | P>=2 era | UE geom | UE rank | rho | E[Y_l] | run exact | run union | TCOM PDP | TCOM PHP | phase-II proxy |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in row_list:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{row.pe:.3f}",
                    format_probability(row.p_zero_erasure),
                    format_probability(row.p_one_erasure),
                    format_probability(row.p_two_or_more_erasures),
                    format_probability(row.p_geometric_ue),
                    format_probability(row.p_rank_peeling_ue),
                    format_probability(row.rho_tagged_collision),
                    format_probability(row.expected_a_list_size),
                    format_probability(row.p_collision_run_exact),
                    format_probability(row.p_collision_run_union),
                    format_probability(row.tcom_phase1_pdp),
                    format_probability(row.tcom_phase1_php),
                    format_probability(row.phase2_pdp_proxy),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "Interpretation notes:",
            "",
            "- For `K=100,J=16,M=3`, the exact collision-run term is tiny.  UACE PDP trends under phase-II-only decoding should therefore be dominated by erasure count/recoverability, not by phase-I collision runs.",
            "- `UE geom` is much smaller than `P>=2 era` at moderate `pe`; this quantifies the value of phase-II-e style multi-erasure recovery if the decoder can exploit separated erasures.",
            "- `UE rank` can be lower than `UE geom` because actual rank-peeling can sometimes recover patterns that violate the simple one-erasure-per-window rule.  This is a concrete route to a tighter theorem.",
            "- The original TCom phase-I bounds are essentially constant in `pe` here because they do not use the exact `(1-pe)` occupancy thinning.  That is one reason they cannot track full PDP curves by themselves.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--J", type=int, default=16)
    parser.add_argument("--M", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pes", type=float, nargs="+", default=list(DEFAULT_PES))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rank_bad_masks = rank_peeling_bad_masks(args.L, args.M, args.seed)
    rows = [
        compute_row(args.K, args.L, args.J, args.M, pe, rank_bad_masks)
        for pe in args.pes
    ]
    content = markdown_table(rows, args)
    if args.output is None:
        print(content)
    else:
        args.output.write_text(content + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
