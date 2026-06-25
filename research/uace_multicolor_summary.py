#!/usr/bin/env python3
"""Summarize LLC/UACE multi-alternate preemption bounds."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from uace_bound_explorer import format_probability


DEFAULT_REPORTS = (
    "research/uace_pair_preemption_exact.md",
    "research/uace_pair_preemption_exact_c3_probe.md",
    "research/uace_pair_preemption_exact_c3_root10_probe.md",
    "research/uace_pair_preemption_exact_c3_root10_full.md",
)
DEFAULT_AGGREGATE = "research/uace_multicolor_chunk_aggregate.md"


@dataclass(frozen=True)
class MultiColorRow:
    source: Path
    colors: int
    label: str
    profiles_checked: int
    max_profiles: str
    canonical_profiles: int
    profile_coverage: float
    k: int
    pe: float
    extra: float


@dataclass(frozen=True)
class AggregateRow:
    source: Path
    colors: int
    included_windows: int
    max_coverage: float
    k: int
    pe: float
    extra: float


def stirling2(n: int, k: int) -> int:
    if n == 0 and k == 0:
        return 1
    if n == 0 or k == 0 or k > n:
        return 0
    table = [[0 for _ in range(k + 1)] for _ in range(n + 1)]
    table[0][0] = 1
    for i in range(1, n + 1):
        for j in range(1, min(i, k) + 1):
            table[i][j] = j * table[i - 1][j] + table[i - 1][j - 1]
    return table[n][k]


def extract(pattern: str, text: str, default: str = "") -> str:
    match = re.search(pattern, text)
    return match.group(1) if match else default


def parse_report(path: Path, known_sections: int) -> list[MultiColorRow]:
    text = path.read_text(encoding="utf-8")
    colors = int(extract(r"- colors: `(\d+)`", text, "0"))
    max_profiles = extract(r"- max profiles per mask: `([^`]+)`", text, "")
    profile_offset = extract(r"- profile offset: `([^`]+)`", text, "0")
    total_profiles = int(extract(r"Total profiles checked: `(\d+)`", text, "0"))
    gap = extract(r"- gap representatives: `([^`]+)`", text, "")
    if "root10_full" in path.name:
        label = "3-color root10 full"
    elif "root10" in path.name:
        label = "3-color root10 truncation"
    elif colors == 3:
        label = "3-color gap-rep truncation"
    else:
        label = "2-color exact"
    if gap:
        label = f"{label}; gap reps={gap}"
    if profile_offset and profile_offset != "0":
        label = f"{label}; offset={profile_offset}"
    canonical_profiles = stirling2(known_sections, colors) if colors else 0
    if max_profiles == "all":
        profile_coverage = 1.0
    else:
        try:
            profile_coverage = min(1.0, int(max_profiles) / canonical_profiles)
        except (TypeError, ValueError, ZeroDivisionError):
            profile_coverage = 0.0

    rows: list[MultiColorRow] = []
    for line in text.splitlines():
        if not re.match(r"\| \d+ \| [0-9.]+ \|", line):
            continue
        parts = [item.strip() for item in line.strip("|").split("|")]
        if len(parts) == 5:
            rows.append(
                MultiColorRow(
                    source=path,
                    colors=colors,
                    label=label,
                    profiles_checked=total_profiles,
                    max_profiles=max_profiles,
                    canonical_profiles=canonical_profiles,
                    profile_coverage=profile_coverage,
                    k=int(parts[0]),
                    pe=float(parts[1]),
                    extra=float(parts[4]),
                )
            )
    return rows


def parse_aggregate(path: Path) -> list[AggregateRow]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    colors = int(extract(r"- colors: `(\d+)`", text, "0"))
    included_windows = int(extract(r"- rows included after overlap removal: `(\d+)`", text, "0"))
    coverages = [
        float(match.group(1))
        for match in re.finditer(
            r"\| (?:included|skipped duplicate|skipped partial-overlap) \| .*? \| ([0-9.eE+-]+) \|$",
            text,
            re.MULTILINE,
        )
    ]
    max_coverage = max(coverages, default=0.0)

    rows: list[AggregateRow] = []
    in_projection = False
    for line in text.splitlines():
        if line.startswith("## K-Scale Extra-Term Projection"):
            in_projection = True
            continue
        if in_projection and line.startswith("## "):
            break
        if not in_projection or not re.match(r"\| \d+ \| [0-9.]+ \|", line):
            continue
        parts = [item.strip() for item in line.strip("|").split("|")]
        if len(parts) != 5:
            continue
        rows.append(
            AggregateRow(
                source=path,
                colors=colors,
                included_windows=included_windows,
                max_coverage=max_coverage,
                k=int(parts[0]),
                pe=float(parts[1]),
                extra=float(parts[4]),
            )
        )
    return rows


def build_report(args: argparse.Namespace) -> str:
    paths = [Path(item) for item in args.reports if Path(item).exists()]
    rows = [row for path in paths for row in parse_report(path, args.known_sections)]
    aggregate_rows = parse_aggregate(args.aggregate_file)
    exact_by_key = {
        (row.k, row.pe): row.extra
        for row in rows
        if row.colors == 2 and "exact" in row.label
    }
    lines = [
        "# UACE Multi-Color Preemption Summary",
        "",
        "This report is generated by `research/uace_multicolor_summary.py`.",
        "",
        f"- canonical known sections per two-erasure profile: `{args.known_sections}`",
        "",
        "| source | colors | label | profiles checked | canonical profiles/mask | coverage/mask | K | pe | erasure-weighted extra | ratio to 2-color exact |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row.k not in args.Ks or row.pe not in args.pes:
            continue
        baseline = exact_by_key.get((row.k, row.pe))
        ratio = row.extra / baseline if baseline and baseline > 0 else float("nan")
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.source}`",
                    str(row.colors),
                    row.label,
                    str(row.profiles_checked),
                    str(row.canonical_profiles),
                    format_probability(row.profile_coverage),
                    str(row.k),
                    f"{row.pe:.3f}",
                    format_probability(row.extra),
                    format_probability(ratio),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Chunk-Aware Aggregate",
            "",
            "The newer overlap-removed aggregate is generated by",
            "`research/uace_multicolor_chunk_aggregate.py` and written to",
            f"`{args.aggregate_file}`.  The continuation manifest is",
            "`research/uace_multicolor_manifest.json`.",
            "",
            "| source | colors | included windows | max coverage/mask | K | pe | erasure-weighted extra | ratio to 2-color exact |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate_rows:
        if row.k not in args.Ks or row.pe not in args.pes:
            continue
        baseline = exact_by_key.get((row.k, row.pe))
        ratio = row.extra / baseline if baseline and baseline > 0 else float("nan")
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.source}`",
                    str(row.colors),
                    str(row.included_windows),
                    format_probability(row.max_coverage),
                    str(row.k),
                    f"{row.pe:.3f}",
                    format_probability(row.extra),
                    format_probability(ratio),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "Readout:",
            "",
            "- The 2-color row is the current exact pair-preemption correction used by the composite PDP/PHP predictor.",
            "- The 3-color rows are multi-alternate diagnostics.  Truncation rows are evidence, not theorem-level bounds.",
            "- Offset-window rows show that resumable profile enumeration is now supported; they are partial diagnostic contributions, not full bounds.",
            "- The chunk-aware aggregate removes duplicated windows before summing finite-window contributions.",
            "- Coverage is measured per represented mask/profile family; it does not mean the whole phase-III multi-color space has been exhausted.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", default=list(DEFAULT_REPORTS))
    parser.add_argument("--aggregate-file", type=Path, default=Path(DEFAULT_AGGREGATE))
    parser.add_argument("--Ks", type=int, nargs="+", default=[40])
    parser.add_argument("--pes", type=float, nargs="+", default=[0.1])
    parser.add_argument("--known-sections", type=int, default=14)
    parser.add_argument("--output", type=Path, default=Path("research/uace_multicolor_summary.md"))
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
