#!/usr/bin/env python3
"""Compare exact pair-preemption bounds with direct two-user decoder probes."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

from uace_bound_explorer import format_probability


DEFAULT_EXACT = Path("research/uace_pair_preemption_exact.md")
DEFAULT_PROBES = (
    "research/uace_pair_decoder_probe.md",
    "research/uace_pair_decoder_probe_pe010_trials1000.md",
    "research/uace_pair_decoder_probe_root10_pe010_trials20000.md",
)


@dataclass(frozen=True)
class ExactKRow:
    k: int
    pe: float
    choices: int
    raw_extra: float
    weighted_extra: float


@dataclass(frozen=True)
class ExactMaskRow:
    attempt: str
    mask: tuple[int, ...]
    multiplicity: int
    raw_q: float


@dataclass(frozen=True)
class ProbeKRow:
    source: Path
    label: str
    k: int
    pe: float
    empirical_union: float
    upper_union: float
    total_pair_trials: int
    preemptions: int
    represented_rows: int


def zero_event_required_trials(rate: float, alpha: float = 0.05) -> int | None:
    if not (0.0 < rate < 1.0):
        return None
    return math.ceil(math.log(alpha) / math.log(1.0 - rate))


def parse_mask(text: str) -> tuple[int, ...]:
    return tuple(int(item) for item in re.findall(r"\d+", text))


def parse_exact(path: Path) -> tuple[list[ExactMaskRow], list[ExactKRow]]:
    text = path.read_text(encoding="utf-8")
    masks: list[ExactMaskRow] = []
    krows: list[ExactKRow] = []
    in_k_table = False
    for line in text.splitlines():
        if line.startswith("| attempt | mask sections |"):
            in_k_table = False
            continue
        if line.startswith("| K | pe | ordered-user choices |"):
            in_k_table = True
            continue
        if line.startswith("| phase-"):
            parts = [item.strip() for item in line.strip("|").split("|")]
            if len(parts) < 10:
                continue
            masks.append(
                ExactMaskRow(
                    attempt=parts[0],
                    mask=parse_mask(parts[1]),
                    multiplicity=int(parts[2]),
                    raw_q=float(parts[9]),
                )
            )
        elif in_k_table and re.match(r"\| \d+ \| [0-9.]+ \|", line):
            parts = [item.strip() for item in line.strip("|").split("|")]
            if len(parts) != 5:
                continue
            krows.append(
                ExactKRow(
                    k=int(parts[0]),
                    pe=float(parts[1]),
                    choices=int(parts[2]),
                    raw_extra=float(parts[3]),
                    weighted_extra=float(parts[4]),
                )
            )
    return masks, krows


def parse_probe(path: Path) -> list[ProbeKRow]:
    text = path.read_text(encoding="utf-8")
    pair_trials = int(re.search(r"pair trials per mask and pe: `(\d+)`", text).group(1))
    represented_rows = 0
    preemptions = 0
    pe_values: set[float] = set()
    for line in text.splitlines():
        if not line.startswith("| phase-"):
            continue
        parts = [item.strip() for item in line.strip("|").split("|")]
        represented_rows += 1
        pe_values.add(float(parts[3]))
        preemptions += int(parts[4].split("/")[0])

    label = path.stem.replace("uace_pair_decoder_probe_", "")
    rows: list[ProbeKRow] = []
    for line in text.splitlines():
        if not re.match(r"\| \d+ \| [0-9.]+ \|", line):
            continue
        parts = [item.strip() for item in line.strip("|").split("|")]
        if len(parts) != 7:
            continue
        rows.append(
            ProbeKRow(
                source=path,
                label=label,
                k=int(parts[0]),
                pe=float(parts[1]),
                empirical_union=float(parts[4]),
                upper_union=float(parts[6]),
                total_pair_trials=pair_trials * sum(1 for pe in pe_values if abs(pe - float(parts[1])) < 1e-12) * (represented_rows // max(len(pe_values), 1)),
                preemptions=preemptions,
                represented_rows=represented_rows // max(len(pe_values), 1),
            )
        )
    return rows


def root10_raw_projection(mask_rows: list[ExactMaskRow], k: int, pe: float, length: int) -> float:
    total = 0.0
    for row in mask_rows:
        if row.attempt != "phase-III root 10" or row.mask != (6, 10):
            continue
        mask_prob = (pe ** len(row.mask)) * ((1.0 - pe) ** (length - len(row.mask)))
        total += row.multiplicity * mask_prob * min(1.0, (k - 1) * row.raw_q)
    return total


def build_report(args: argparse.Namespace) -> str:
    mask_rows, exact_rows = parse_exact(args.exact)
    probe_rows = [row for path in args.probes if Path(path).exists() for row in parse_probe(Path(path))]
    exact_by_key = {(row.k, row.pe): row for row in exact_rows}
    lines = [
        "# UACE Pair-Level Validation Summary",
        "",
        "This report is generated by `research/uace_pair_validation_summary.py`.",
        "",
        f"- exact pair source: `{args.exact}`",
        "",
        "## Exact Pair Correction",
        "",
        "| K | pe | exact raw pair extra | exact erasure-weighted pair extra |",
        "|---:|---:|---:|---:|",
    ]
    for row in exact_rows:
        if row.k in args.Ks and row.pe in args.pes:
            lines.append(
                f"| {row.k} | {row.pe:.3f} | {format_probability(row.raw_extra)} | {format_probability(row.weighted_extra)} |"
            )

    lines.extend(
        [
            "",
            "## Direct Pair Decoder Probe Resolution",
            "",
            "The direct probe runs the actual first-valid-path decoder on two users conditioned on accepted tagged two-erasure masks.  It is the closest executable check of the pair-preemption event, but zero events still have finite resolution.",
            "",
            "| probe | K | pe | represented mask rows | pair trials/pe | preemptions | exact weighted extra | empirical union extra | 95% union upper | upper / exact | pair trials needed at same design |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in probe_rows:
        if "root10" in row.source.name:
            continue
        if row.k not in args.Ks or row.pe not in args.pes:
            continue
        exact = exact_by_key.get((row.k, row.pe))
        if not exact:
            continue
        ratio = row.upper_union / exact.weighted_extra if exact.weighted_extra > 0 else float("nan")
        needed = math.ceil((row.total_pair_trials / max(row.represented_rows, 1)) * ratio)
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.source.name}`",
                    str(row.k),
                    f"{row.pe:.3f}",
                    str(row.represented_rows),
                    str(row.total_pair_trials),
                    str(row.preemptions),
                    format_probability(exact.weighted_extra),
                    format_probability(row.empirical_union),
                    format_probability(row.upper_union),
                    format_probability(ratio),
                    str(needed),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Root10 Deep Probe",
            "",
            "The root10 deep probe only covers the represented mask `(6,10)`, so it should be compared with the root10 raw analytic contribution, not the full pair correction.",
            "",
            "| K | pe | root10 raw analytic extra | root10 direct 95% union upper | upper / root10 raw | pair trials needed for root10 scale |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    root10_rows = [row for row in probe_rows if "root10" in row.source.name]
    for row in root10_rows:
        if row.k not in args.Ks or row.pe not in args.pes:
            continue
        root10 = root10_raw_projection(mask_rows, row.k, row.pe, args.L)
        ratio = row.upper_union / root10 if root10 > 0 else float("nan")
        needed = math.ceil((row.total_pair_trials / max(row.represented_rows, 1)) * ratio)
        lines.append(
            f"| {row.k} | {row.pe:.3f} | {format_probability(root10)} | {format_probability(row.upper_union)} | {format_probability(ratio)} | {needed} |"
        )

    lines.extend(
        [
            "",
            "Readout:",
            "",
            "- Direct pair probes observed zero preemptions, which rules out gross decoder-level pair failures.",
            "- The full gap-representative 1000-trial probe is still hundreds of times too coarse to empirically resolve the exact `1e-4` K=40 correction.",
            "- The 20000-trial root10 probe gives a much tighter local upper bound, but root10 is not the dominant multiplicity class in the total pair correction.",
            "- Therefore the pair correction is presently supported mainly by the exact affine-rank enumeration, with direct decoder probes serving as implementation sanity checks.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact", type=Path, default=DEFAULT_EXACT)
    parser.add_argument("--probes", nargs="+", default=list(DEFAULT_PROBES))
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--Ks", type=int, nargs="+", default=[30, 40])
    parser.add_argument("--pes", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    parser.add_argument("--output", type=Path, default=Path("research/uace_pair_validation_summary.md"))
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
