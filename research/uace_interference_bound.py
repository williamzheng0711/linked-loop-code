#!/usr/bin/env python3
"""First-moment interference terms for LLC/UACE.

The schedule bound is erasure-only and K-independent.  This script adds a
K-dependent, first-moment estimate for parity-consistent false survivors in the
current LLC decoder schedule.  The estimate is deliberately theorem-shaped:
it counts candidate A-list paths by exact occupancy scale and discounts them
by the number of parity bits that a false path must satisfy.
"""

from __future__ import annotations

import argparse
import math
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path

from uace_bound_explorer import (
    bad_mask_probability,
    expected_a_list_size,
    format_probability,
    load_repo_code_matrices,
    popcount,
    tagged_collision_probability,
)
from uace_parity_rank_profile import parity_rank_exponent
from uace_schedule_bound import attempt_succeeds, schedule_attempts, schedule_bad_masks


DEFAULT_KS = (30, 40, 100)
DEFAULT_PES = (0.05, 0.1, 0.15, 0.2, 0.3)


@dataclass(frozen=True)
class AttemptMoment:
    name: str
    d: int
    erasure_shapes: int
    erasure_shape_counts: tuple[tuple[int, int], ...]
    rank_shape_counts: tuple[tuple[int, int, int], ...]
    weight_contributions: tuple[tuple[int, int, int, float], ...]
    parity_bits_checked: int
    expected_false_survivors: float


def comb(n: int, r: int) -> int:
    if r < 0 or r > n:
        return 0
    return math.comb(n, r)


@lru_cache(maxsize=None)
def stirling2(n: int, k: int) -> int:
    if n == 0 and k == 0:
        return 1
    if n == 0 or k == 0 or k > n:
        return 0
    return k * stirling2(n - 1, k) + stirling2(n - 1, k - 1)


def falling_factorial(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    out = 1
    for item in range(k):
        out *= n - item
    return out


def occupancy_pmf(k: int, j: int, pe: float) -> list[float]:
    """Exact PMF of one-section A-list size under random symbols.

    This is equivalent to mixing the Stirling occupancy law over
    N ~ Binomial(K, 1-pe), but the Markov recursion below is numerically stable:
    each user is erased, hits an occupied symbol, or opens a new symbol.
    """
    q = 2**j
    keep = 1.0 - pe
    pmf = [1.0]
    for _user in range(k):
        nxt = [0.0 for _ in range(len(pmf) + 1)]
        for s, prob in enumerate(pmf):
            stay = pe + keep * (s / q)
            grow = keep * ((q - s) / q)
            nxt[s] += prob * stay
            nxt[s + 1] += prob * grow
        pmf = nxt
    return pmf


def occupancy_moment(k: int, j: int, pe: float, order: int) -> float:
    pmf = occupancy_pmf(k, j, pe)
    return sum((idx**order) * prob for idx, prob in enumerate(pmf))


@lru_cache(maxsize=None)
def accepted_erasure_shape_rank_counts(
    length: int,
    memory: int,
    seed: int,
    root: int,
    d: int,
    erasure_slot_tuple: tuple[int, ...],
) -> tuple[tuple[int, int, int], ...]:
    """Exact accepted erasure-shape/rank counts for one decoder attempt.

    The old first-moment proxy used only a combinatorial upper count such as
    ``binom(L-1,d)``.  The current decoder is more structured: root rotation,
    erasure-slot guards, sequential recovery, dirty saver sections, and local
    ranks reject many masks.  This finite enumeration is cheap for L=16 and is
    the right object for a theorem tied to the current implementation.
    """
    if d == 0:
        message_lens, parity_lens, gijs = load_repo_code_matrices(length, memory, seed)
        exponent, _rank_full, _rank_unknown = parity_rank_exponent(0, length, memory, message_lens, parity_lens, gijs)
        return ((0, exponent, 1),)

    message_lens, parity_lens, gijs = load_repo_code_matrices(length, memory, seed)
    erasure_slot = set(erasure_slot_tuple)
    by_weight_rank: dict[tuple[int, int], int] = {}
    for mask in range(1 << length):
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
            weight = popcount(mask)
            exponent, _rank_full, _rank_unknown = parity_rank_exponent(mask, length, memory, message_lens, parity_lens, gijs)
            key = (weight, exponent)
            by_weight_rank[key] = by_weight_rank.get(key, 0) + 1
    return tuple((weight, exponent, count) for (weight, exponent), count in sorted(by_weight_rank.items()))


def attempt_false_moment(
    *,
    name: str,
    length: int,
    memory: int,
    seed: int,
    parity_bits_per_section: int,
    a_list_size: float,
    root: int,
    d: int,
    erasure_slot: set[int],
) -> AttemptMoment:
    rank_counts = accepted_erasure_shape_rank_counts(
        length,
        memory,
        seed,
        root,
        d,
        tuple(sorted(erasure_slot)),
    )
    by_weight: dict[int, int] = {}
    for weight, _exponent, count in rank_counts:
        by_weight[weight] = by_weight.get(weight, 0) + count
    counts = tuple(sorted(by_weight.items()))
    shapes = sum(count for _weight, count in counts)
    expected = 0.0
    min_checked = length * parity_bits_per_section
    weight_contributions = []
    for weight, exponent, count in rank_counts:
        min_checked = min(min_checked, exponent)
        candidate_paths = count * (a_list_size ** (length - weight))
        contribution = candidate_paths * (2.0 ** (-exponent))
        expected += contribution
        weight_contributions.append((weight, count, exponent, contribution))
    return AttemptMoment(
        name=name,
        d=d,
        erasure_shapes=shapes,
        erasure_shape_counts=counts,
        rank_shape_counts=rank_counts,
        weight_contributions=tuple(weight_contributions),
        parity_bits_checked=min_checked,
        expected_false_survivors=expected,
    )


def interference_moments(
    *,
    k: int,
    length: int,
    j: int,
    memory: int,
    phase: int,
    pe: float,
    parity_bits_per_section: int,
    seed: int = 0,
) -> list[AttemptMoment]:
    a_size = expected_a_list_size(k, j, pe)
    rows = []
    for name, root, d, erasure_slot in schedule_attempts(length, phase):
        rows.append(
            attempt_false_moment(
                name=name,
                length=length,
                memory=memory,
                seed=seed,
                parity_bits_per_section=parity_bits_per_section,
                a_list_size=a_size,
                root=root,
                d=d,
                erasure_slot=erasure_slot,
            )
        )
    return rows


def build_report(args: argparse.Namespace) -> str:
    bad_masks = {
        phase: schedule_bad_masks(args.L, args.M, phase, args.seed)
        for phase in args.phases
    }

    lines = [
        "# UACE Interference First-Moment Bound",
        "",
        "This report is generated by `research/uace_interference_bound.py`.",
        "",
        f"- `L = {args.L}`",
        f"- `M = {args.M}`",
        f"- `J = {args.J}`",
        f"- parity bits per section: `{args.R}`",
        f"- matrix seed: `{args.seed}`",
        "",
        "Definitions:",
        "",
        "- `schedule UE` is the exact erasure-only term for the current schedule.",
        "- `lambda_A` is the exact expected A-list occupancy per section.",
        "- The factor `lambda_A^(L-w)` is an exact product moment under the independent-section random-symbol abstraction; it is not a Jensen replacement for `E[S^(L-w)]`.",
        "- `E false` is the first-moment upper-scale estimate of parity-consistent false survivors over all attempts up to the listed phase.",
        "- `PHP <= E false/K` is the Markov bound for per-user hallucination probability.",
        "- `PDP pred` is `schedule UE + E false/K`; for K=30/K=40 it is numerically indistinguishable from the schedule term, while K=100 phase-III low-erasure points expose a non-negligible first-moment PHP bound.",
        "- `tagged rho` and `any tagged collision` are diagnostic collision scales, not added directly to PDP.",
        "",
    ]

    for phase in args.phases:
        lines.extend(
            [
                f"## Phase {phase}",
                "",
                "| K | pe | schedule UE | lambda_A | E false | PHP <= E false/K | PDP pred | tagged rho | any tagged collision | dominant false attempt |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for k in args.Ks:
            for pe in args.pes:
                schedule_ue = bad_mask_probability(bad_masks[phase], args.L, pe)
                a_size = expected_a_list_size(k, args.J, pe)
                moments = interference_moments(
                    k=k,
                    length=args.L,
                    j=args.J,
                    memory=args.M,
                    phase=phase,
                    pe=pe,
                    parity_bits_per_section=args.R,
                    seed=args.seed,
                )
                e_false = sum(item.expected_false_survivors for item in moments)
                php_bound = min(1.0, e_false / k)
                pdp_pred = min(1.0, (schedule_ue or 0.0) + php_bound)
                rho = tagged_collision_probability(k, args.J, pe)
                any_collision = 1.0 - (1.0 - ((1.0 - pe) * rho)) ** args.L
                dominant = max(moments, key=lambda item: item.expected_false_survivors)
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(k),
                            f"{pe:.3f}",
                            format_probability(schedule_ue),
                            f"{a_size:.3f}",
                            format_probability(e_false),
                            format_probability(php_bound),
                            format_probability(pdp_pred),
                            format_probability(rho),
                            format_probability(any_collision),
                            f"{dominant.name} ($d={dominant.d}$)",
                        ]
                    )
                    + " |"
                )
        lines.append("")
        lines.extend(
            [
                "Accepted erasure-shape profiles:",
                "",
                "| attempt | accepted shape profile $A_a(w)$ | accepted shapes |",
                "|---|---:|---:|",
            ]
        )
        for moment in interference_moments(
            k=args.Ks[0],
            length=args.L,
            j=args.J,
            memory=args.M,
            phase=phase,
            pe=args.pes[0],
            parity_bits_per_section=args.R,
            seed=args.seed,
        ):
            profile = ", ".join(f"w={weight}:{count}" for weight, count in moment.erasure_shape_counts)
            lines.append(f"| {moment.name} | {profile} | {moment.erasure_shapes} |")
        lines.append("")

    lines.extend(
        [
            "One-section occupancy distribution checks:",
            "",
            "| K | pe | E[S] | Var(S) | P(S=0) | E[S^2] |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for k in args.Ks:
        for pe in args.pes:
            mean = occupancy_moment(k, args.J, pe, 1)
            second = occupancy_moment(k, args.J, pe, 2)
            var = second - mean**2
            pmf = occupancy_pmf(k, args.J, pe)
            lines.append(
                f"| {k} | {pe:.3f} | {mean:.6f} | {var:.6f} | {pmf[0]:.3e} | {second:.6f} |"
            )
    lines.append("")

    lines.extend(
        [
            "Attempt-level formula:",
            "",
            "For an attempt `a`, let `A_a(w,e)` be the exact number of erasure masks of weight `w` accepted by that current-decoder attempt with parity-rank exponent `e = rank([A B]) - rank(A)`.  The first-moment term used here is",
            "",
            "$$",
            "\\mathbb{E}N_{\\mathrm{false},a}",
            "\\le",
            "\\sum_{w=0}^L\\sum_e A_a(w,e)\\,\\lambda_A^{L-w}\\,2^{-e},",
            "$$",
            "",
            "where the exponent comes from the existential erased-info linear system $A x_{\\rm erased}+Bz_{\\rm known}=0$, and",
            "",
            "$$",
            "\\lambda_A",
            "=",
            "2^J\\left[1-\\left(1-\\frac{1-p_e}{2^J}\\right)^K\\right].",
            "$$",
            "",
            "Under independent section lists, $\\mathbb{E}\\prod_{i=1}^m S_i=\\lambda_A^m$ exactly.  This is still a first-moment theorem rather than a full dependency-aware path DP: it treats false path syndromes as random.  Its value is that all main ingredients are now finite-length objects: exact A-channel occupancy/product scale, exact accepted erasure-shape enumeration, and exact GF(2) parity-rank exponents.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--M", type=int, default=3)
    parser.add_argument("--J", type=int, default=16)
    parser.add_argument("--R", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--Ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--pes", type=float, nargs="+", default=list(DEFAULT_PES))
    parser.add_argument("--phases", type=int, nargs="+", default=[2, 3], choices=(1, 2, 3))
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
