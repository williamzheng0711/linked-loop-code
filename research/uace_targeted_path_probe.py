#!/usr/bin/env python3
"""Targeted path-interference probe for LLC/UACE.

This script focuses the expensive path search on users whose erasure masks are
recoverable by the current schedule.  It asks whether a parity-consistent path
search from the user's true root is preempted by a wrong path.

It is a semi-genie validator: it does not scan roots belonging only to
schedule-failed users.  Therefore it is designed to validate the extra
interference term on top of the schedule PDP, not to benchmark decoder runtime.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from uace_bound_explorer import bad_mask_probability, rank_peeling_succeeds
from uace_fast_empirical_probe import first_valid_path_for_root, rotate_decoder_state, uace_channel
from uace_interference_bound import interference_moments
from uace_schedule_bound import attempt_succeeds, schedule_attempts, schedule_bad_masks, schedule_succeeds


def load_playground():
    repo_root = Path(__file__).resolve().parents[1]
    playground = repo_root / "playground"
    sys.path.insert(0, str(playground))
    import abch_utils  # type: ignore
    import general_lib  # type: ignore
    import general_utils as gu  # type: ignore
    import linkedloop as LLC  # type: ignore
    import static_repo  # type: ignore

    return static_repo, general_lib, gu, LLC, abch_utils


@dataclass(frozen=True)
class TargetedTrial:
    seed: int
    schedule_fail_emp: float
    rank_fail_emp: float
    schedule_success_users: int
    checked_users: int
    path_fail_users: int
    wrong_path_users: int
    aborted_users: int
    targeted_php: float
    node_visits: int
    seconds: float


def bit_row_key(bits: np.ndarray) -> bytes:
    return np.asarray(bits, dtype=np.uint8).tobytes()


def first_success_attempt(mask: int, *, length: int, memory: int, phase: int, message_lens, gijs):
    for name, root, d, erasure_slot in schedule_attempts(length, phase):
        if d == 0:
            if mask == 0:
                return name, root, d, erasure_slot
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
            return name, root, d, erasure_slot
    return None


def build_attempt_context(static_repo, gu, L, M, chosen_roots, grand_list, message_lens, parity_lens, gis, columns_index, sub_g_invs):
    chosen_root, erasure_slot, grand, msg_lens, par_lens, rot_gis, rot_columns, rot_invs, gijs = rotate_decoder_state(
        static_repo, L, M, chosen_roots, grand_list, message_lens, parity_lens, gis, columns_index, sub_g_invs
    )
    deciders_cache, avail_savers_cache = gu.build_decoder_lookup(L, M)
    solve_cache = gu.build_solve_cache(L, M, rot_columns, rot_invs, rot_gis)
    valid_ks_by_section = [np.flatnonzero(grand[:, l * static_repo.J] != -1) for l in range(L)]
    parity_lookup_by_section = gu.build_parity_lookup_by_section(grand, L, msg_lens, valid_ks_by_section)
    parity_cache = gu.build_parity_cache(grand, L, M, msg_lens, gijs)
    caches = (valid_ks_by_section, parity_cache, deciders_cache, avail_savers_cache, parity_lookup_by_section, solve_cache)
    root_lookup = {
        bit_row_key(grand[row, 0:static_repo.J]): int(row)
        for row in valid_ks_by_section[0]
    }
    return {
        "chosen_root": chosen_root,
        "erasure_slot": erasure_slot,
        "grand": grand,
        "message_lens": msg_lens,
        "parity_lens": par_lens,
        "gis": rot_gis,
        "columns_index": rot_columns,
        "sub_g_invs": rot_invs,
        "gijs": gijs,
        "caches": caches,
        "root_lookup": root_lookup,
    }


def unrotate_decoded(decoded_msg: np.ndarray, chosen_root: int, message_lens: np.ndarray, L: int) -> np.ndarray:
    if decoded_msg.size == 0:
        return decoded_msg
    w = int(sum(message_lens))
    decoded_msg = decoded_msg.copy()
    decoded_msg[:, range(w)] = decoded_msg[:, np.mod(np.arange(w) + sum(message_lens[0:L - chosen_root]), w)]
    return decoded_msg


def run_trial(args: argparse.Namespace, seed: int) -> TargetedTrial:
    static_repo, general_lib, gu, LLC, abch_utils = load_playground()
    import uace_fast_empirical_probe as fast_probe

    fast_probe.gu_global = gu

    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)
    start = time.time()

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
    rx_symbols, _erasure_counts, erasure_masks = uace_channel(tx_symbols, args.pe, rng)
    grand_list = abch_utils.symbol_to_binary(args.K, args.L, rx_symbols)

    schedule_success = [
        schedule_succeeds(
            int(mask),
            length=args.L,
            memory=args.M,
            phase=args.phase,
            message_lens=message_lens,
            gijs=gijs,
        )
        for mask in erasure_masks
    ]
    rank_fail_emp = float(
        np.mean(
            [
                not rank_peeling_succeeds(int(mask), args.L, args.M, message_lens, gijs)
                for mask in erasure_masks
            ]
        )
    )

    contexts = {}
    for name, root, d, erasure_slot in schedule_attempts(args.L, args.phase):
        if d == 0:
            continue
        chosen_roots = None
        if name == "phase-II root 8":
            chosen_roots = [8]
        elif name == "phase-III root 6":
            chosen_roots = [6]
        elif name == "phase-III root 10":
            chosen_roots = [6, 10]
        contexts[name] = build_attempt_context(
            static_repo,
            gu,
            args.L,
            args.M,
            chosen_roots,
            grand_list,
            message_lens,
            parity_lens,
            gis,
            columns_index,
            sub_g_invs,
        )

    checked = 0
    path_fail = 0
    wrong_path = 0
    aborted = 0
    node_visits = 0
    false_messages: set[bytes] = set()

    for user, ok in enumerate(schedule_success):
        if not ok:
            continue
        attempt = first_success_attempt(
            int(erasure_masks[user]),
            length=args.L,
            memory=args.M,
            phase=args.phase,
            message_lens=message_lens,
            gijs=gijs,
        )
        if attempt is None:
            path_fail += 1
            continue
        name, root, d, _erasure_slot = attempt
        if d == 0:
            checked += 1
            continue

        ctx = contexts[name]
        root_bits = tx_cdwds[user, root * static_repo.J : (root + 1) * static_repo.J]
        root_row = ctx["root_lookup"].get(bit_row_key(root_bits))
        checked += 1
        if root_row is None:
            path_fail += 1
            continue

        path, visits, did_abort = first_valid_path_for_root(
            root=root_row,
            d=d,
            grand=ctx["grand"],
            K=args.K,
            L=args.L,
            M=args.M,
            message_lens=ctx["message_lens"],
            parity_lens=ctx["parity_lens"],
            gis=ctx["gis"],
            gijs=ctx["gijs"],
            columns_index=ctx["columns_index"],
            sub_g_invs=ctx["sub_g_invs"],
            erasure_slot=ctx["erasure_slot"],
            caches=ctx["caches"],
            LLC=LLC,
            max_nodes=args.max_nodes_per_user,
        )
        node_visits += visits
        if did_abort:
            aborted += 1
            path_fail += 1
            continue
        if path is None:
            path_fail += 1
            continue

        decoded = gu.output_message_oop(ctx["grand"], [path], args.L, static_repo.J)
        decoded = unrotate_decoded(decoded, ctx["chosen_root"], ctx["message_lens"], args.L)
        decoded_key = bit_row_key(decoded[0])
        true_key = bit_row_key(tx_bits[user])
        if decoded_key != true_key:
            wrong_path += 1
            path_fail += 1
            false_messages.add(decoded_key)

    return TargetedTrial(
        seed=seed,
        schedule_fail_emp=float(1.0 - np.mean(schedule_success)),
        rank_fail_emp=rank_fail_emp,
        schedule_success_users=int(sum(schedule_success)),
        checked_users=checked,
        path_fail_users=path_fail,
        wrong_path_users=wrong_path,
        aborted_users=aborted,
        targeted_php=len(false_messages) / args.K,
        node_visits=node_visits,
        seconds=time.time() - start,
    )


def build_report(args: argparse.Namespace, rows: list[TargetedTrial]) -> str:
    def mean(field: str) -> float:
        return float(np.mean([getattr(row, field) for row in rows]))

    schedule_bad = schedule_bad_masks(args.L, args.M, args.phase, args.seed)
    schedule_ue = bad_mask_probability(schedule_bad, args.L, args.pe)
    moments = interference_moments(
        k=args.K,
        length=args.L,
        j=16,
        memory=args.M,
        phase=args.phase,
        pe=args.pe,
        parity_bits_per_section=8,
    )
    php_bound = sum(item.expected_false_survivors for item in moments) / args.K

    lines = [
        "# UACE Targeted Path Probe",
        "",
        f"- `K = {args.K}`",
        f"- `L = {args.L}`",
        f"- `M = {args.M}`",
        f"- `pe = {args.pe}`",
        f"- `phase = {args.phase}`",
        f"- trials: `{args.trials}`",
        f"- max nodes per schedule-success user: `{args.max_nodes_per_user}`",
        "",
        "Theory:",
        "",
        f"- schedule UE: `{schedule_ue:.6f}`",
        f"- first-moment PHP bound: `{php_bound:.3e}`",
        "",
        "| metric | mean |",
        "|---|---:|",
    ]
    for field in (
        "schedule_fail_emp",
        "rank_fail_emp",
        "schedule_success_users",
        "checked_users",
        "path_fail_users",
        "wrong_path_users",
        "aborted_users",
        "targeted_php",
        "node_visits",
        "seconds",
    ):
        lines.append(f"| {field} | {mean(field):.6f} |")

    lines.extend(
        [
            "",
            "Per-trial rows:",
            "",
            "| seed | schedule fail | schedule-success users | checked | path fail | wrong path | aborted | targeted PHP | node visits | seconds |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.seed} | {row.schedule_fail_emp:.6f} | {row.schedule_success_users} | "
            f"{row.checked_users} | {row.path_fail_users} | {row.wrong_path_users} | "
            f"{row.aborted_users} | {row.targeted_php:.6f} | {row.node_visits} | {row.seconds:.2f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `path_fail_users` among schedule-success users is the directly observed extra path-interference term in this targeted probe.",
            "- `wrong_path_users` counts preemption by a parity-consistent path whose decoded message is not the tagged user's message.",
            "- The probe is exact for the targeted users only when `aborted = 0`; otherwise it is conservative for PDP.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--K", type=int, default=30)
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--M", type=int, default=3)
    parser.add_argument("--pe", type=float, default=0.3)
    parser.add_argument("--phase", type=int, default=3, choices=(2, 3))
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-nodes-per-user", type=int, default=500_000)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [run_trial(args, args.seed + idx) for idx in range(args.trials)]
    content = build_report(args, rows)
    if args.output:
        args.output.write_text(content + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
