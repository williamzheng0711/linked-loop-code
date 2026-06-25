#!/usr/bin/env python3
"""Aggregate resumable multi-color pair-preemption profile chunks.

`uace_pair_preemption_exact.py` can enumerate a window of identity profiles via
`--profile-offset` and `--max-profiles`.  This script parses those markdown
reports, groups rows by attempt and mask, removes fully duplicated profile
windows, and accumulates conservative raw / erasure-weighted union terms.

The aggregate is still a finite-window diagnostic unless every represented
mask has full profile coverage.  Its value is that the coverage and overlap
accounting are explicit and reproducible.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

from uace_bound_explorer import format_probability


DEFAULT_REPORTS = (
    "research/uace_pair_preemption_exact_c3_gap_profiles000k_001k.md",
    "research/uace_pair_preemption_exact_c3_root0_mask01_04_profiles001k_006k.md",
    "research/uace_pair_preemption_exact_c3_root6_mask01_04_profiles001k_006k.md",
    "research/uace_pair_preemption_exact_c3_root10_profiles000k_005k.md",
    "research/uace_pair_preemption_exact_c3_root10_profiles005k_010k.md",
    "research/uace_pair_preemption_exact_c3_root10_profiles010k_015k.md",
    "research/uace_pair_preemption_exact_c3_root10_profiles015k_020k.md",
)
DEFAULT_KS = (30, 40, 100)
DEFAULT_PES = (0.1, 0.2, 0.3)


@dataclass(frozen=True)
class ChunkRow:
    source: Path
    colors: int
    offset: int
    max_profiles: int | None
    attempt: str
    mask_sections: tuple[int, ...]
    multiplicity: int
    checked: int
    parity_valid: int
    feasible: int
    min_preempt_rank: int
    raw_q: float
    weighted_q: dict[float, float]

    @property
    def start(self) -> int:
        return self.offset

    @property
    def end(self) -> int:
        return self.offset + self.checked

    @property
    def key(self) -> tuple[int, str, tuple[int, ...]]:
        return (self.colors, self.attempt, self.mask_sections)


@dataclass(frozen=True)
class IncludedRow:
    row: ChunkRow
    status: str


def stirling2(n: int, k: int) -> int:
    table = [[0 for _ in range(k + 1)] for _ in range(n + 1)]
    table[0][0] = 1
    for i in range(1, n + 1):
        for j in range(1, min(i, k) + 1):
            table[i][j] = j * table[i - 1][j] + table[i - 1][j - 1]
    return table[n][k]


def falling_factorial(n: int, k: int) -> int:
    out = 1
    for idx in range(k):
        out *= max(n - idx, 0)
    return out


def extract(pattern: str, text: str, default: str = "") -> str:
    match = re.search(pattern, text)
    return match.group(1) if match else default


def parse_mask_sections(text: str) -> tuple[int, ...]:
    return tuple(int(item) for item in re.findall(r"\d+", text))


def parse_float(text: str) -> float:
    if not text or text == "n/a":
        return 0.0
    return float(text)


def parse_report(path: Path) -> list[ChunkRow]:
    text = path.read_text(encoding="utf-8")
    colors = int(extract(r"- colors: `(\d+)`", text, "0"))
    offset = int(extract(r"- profile offset: `(\d+)`", text, "0"))
    max_profiles_text = extract(r"- max profiles per mask: `([^`]+)`", text, "all")
    max_profiles = None if max_profiles_text == "all" else int(max_profiles_text)

    rows: list[ChunkRow] = []
    header: list[str] | None = None
    for line in text.splitlines():
        if line.startswith("| attempt | mask sections |"):
            header = [item.strip() for item in line.strip("|").split("|")]
            continue
        if header is None or not line.startswith("| phase-"):
            continue
        parts = [item.strip() for item in line.strip("|").split("|")]
        if len(parts) != len(header):
            continue
        data = dict(zip(header, parts))
        weighted: dict[float, float] = {}
        for name, value in data.items():
            match = re.fullmatch(r"weighted q pe=([0-9.]+)", name)
            if match:
                weighted[float(match.group(1))] = parse_float(value)
        rows.append(
            ChunkRow(
                source=path,
                colors=colors,
                offset=offset,
                max_profiles=max_profiles,
                attempt=data["attempt"],
                mask_sections=parse_mask_sections(data["mask sections"]),
                multiplicity=int(data["mult."]),
                checked=int(data["profiles checked"]),
                parity_valid=int(data["parity-valid profiles"]),
                feasible=int(data["feasible profiles"]),
                min_preempt_rank=int(data["min preempt rank"]),
                raw_q=parse_float(data["pair union bound"]),
                weighted_q=weighted,
            )
        )
    return rows


def include_nonoverlapping(rows: list[ChunkRow]) -> list[IncludedRow]:
    by_key: dict[tuple[int, str, tuple[int, ...]], list[ChunkRow]] = {}
    for row in rows:
        by_key.setdefault(row.key, []).append(row)

    out: list[IncludedRow] = []
    for _key, bucket in sorted(by_key.items()):
        covered: list[tuple[int, int]] = []
        for row in sorted(bucket, key=lambda item: (item.start, -(item.end - item.start), item.source.name)):
            contained = any(start <= row.start and row.end <= end for start, end in covered)
            overlaps = any(not (row.end <= start or row.start >= end) for start, end in covered)
            if contained:
                out.append(IncludedRow(row=row, status="skipped duplicate"))
                continue
            if overlaps:
                out.append(IncludedRow(row=row, status="skipped partial-overlap"))
                continue
            covered.append((row.start, row.end))
            covered.sort()
            out.append(IncludedRow(row=row, status="included"))
    return out


def interval_coverage(included: list[IncludedRow], known_sections: int) -> dict[tuple[int, str, tuple[int, ...]], float]:
    by_key: dict[tuple[int, str, tuple[int, ...]], list[ChunkRow]] = {}
    for item in included:
        if item.status != "included":
            continue
        by_key.setdefault(item.row.key, []).append(item.row)
    coverage = {}
    for key, rows in by_key.items():
        colors = key[0]
        canonical = stirling2(known_sections, colors)
        covered = sum(row.checked for row in rows)
        coverage[key] = covered / canonical if canonical else 0.0
    return coverage


def build_report(args: argparse.Namespace) -> str:
    paths = [Path(item) for item in args.reports if Path(item).exists()]
    rows = [row for path in paths for row in parse_report(path)]
    included = include_nonoverlapping(rows)
    included_rows = [item.row for item in included if item.status == "included"]
    coverage = interval_coverage(included, args.known_sections)
    colors_seen = sorted({row.colors for row in included_rows})
    colors = colors_seen[0] if len(colors_seen) == 1 else 0
    canonical = stirling2(args.known_sections, colors) if colors else 0
    raw_q = sum(row.multiplicity * row.raw_q for row in included_rows)
    weighted_q: dict[float, float | None] = {}
    for pe in args.pes:
        if any(pe not in row.weighted_q for row in included_rows):
            weighted_q[pe] = None
        else:
            weighted_q[pe] = sum(row.multiplicity * row.weighted_q[pe] for row in included_rows)
    min_rank = min((row.min_preempt_rank for row in included_rows if row.min_preempt_rank >= 0), default=-1)

    lines = [
        "# UACE Multi-Color Chunk Aggregate",
        "",
        "This report is generated by `research/uace_multicolor_chunk_aggregate.py`.",
        "",
        f"- reports parsed: `{len(paths)}`",
        f"- rows parsed: `{len(rows)}`",
        f"- rows included after overlap removal: `{len(included_rows)}`",
        f"- colors: `{colors if colors else 'mixed'}`",
        f"- canonical profiles per two-erasure mask: `{canonical if canonical else 'n/a'}`",
        f"- minimum included preempt rank: `{min_rank}`",
        "",
        "## Included Profile Windows",
        "",
        "| status | source | colors | attempt | mask | mult. | profile window | checked | feasible | min rank | raw q | coverage for mask |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in sorted(included, key=lambda x: (x.row.key, x.row.start, x.row.source.name)):
        row = item.row
        cov = coverage.get(row.key, 0.0)
        lines.append(
            "| "
            + " | ".join(
                [
                    item.status,
                    f"`{row.source.name}`",
                    str(row.colors),
                    row.attempt,
                    str(row.mask_sections),
                    str(row.multiplicity),
                    f"{row.start}-{row.end}",
                    str(row.checked),
                    str(row.feasible),
                    str(row.min_preempt_rank),
                    format_probability(row.raw_q),
                    format_probability(cov),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Aggregate Fixed-Profile-Family Bound",
            "",
            "| pe | multiplicity-weighted raw q | multiplicity-weighted erasure-weighted q |",
            "|---:|---:|---:|",
        ]
    )
    for pe in args.pes:
        lines.append(f"| {pe:.3f} | {format_probability(raw_q)} | {format_probability(weighted_q[pe])} |")

    lines.extend(
        [
            "",
            "## K-Scale Extra-Term Projection",
            "",
            "The projection uses the same conservative lifting as `uace_pair_preemption_exact.py`: weight-2 tagged-mask probability times ordered alternate-user choices.",
            "",
            "| K | pe | ordered choices | raw extra | erasure-weighted extra |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for k in args.Ks:
        choices = falling_factorial(k - 1, max(colors - 1, 0))
        for pe in args.pes:
            mask_prob = (pe**2) * ((1.0 - pe) ** (args.L - 2))
            raw_extra = mask_prob * min(1.0, choices * raw_q)
            weighted_extra = (
                None
                if weighted_q[pe] is None
                else mask_prob * min(1.0, choices * weighted_q[pe])
            )
            lines.append(
                f"| {k} | {pe:.3f} | {choices} | {format_probability(raw_extra)} | {format_probability(weighted_extra)} |"
            )

    lines.extend(
        [
            "",
            "Readout:",
            "",
            "- This aggregate removes fully duplicated profile windows, e.g. a 0--5000 window contained in a later 0--50000 report for the same attempt/mask.",
            "- Rows marked `skipped partial-overlap` are not included because the markdown reports do not provide enough information to split a partially overlapping window.",
            "- The result is a rigorous finite-window union contribution for the included chunks, but not a full multi-color theorem unless every represented mask reaches coverage 1.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", default=list(DEFAULT_REPORTS))
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--known-sections", type=int, default=14)
    parser.add_argument("--Ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--pes", type=float, nargs="+", default=list(DEFAULT_PES))
    parser.add_argument("--output", type=Path, default=Path("research/uace_multicolor_chunk_aggregate.md"))
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
