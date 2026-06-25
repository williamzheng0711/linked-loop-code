#!/usr/bin/env python3
"""Build a machine-readable manifest for 3-color profile chunks.

The full 3-color profile space has S(14,3)=788970 profiles per represented
two-erasure mask.  This manifest records which profile intervals have already
been enumerated, how much coverage that gives, and which chunk should be run
next for each represented mask.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from uace_bound_explorer import format_probability
from uace_multicolor_chunk_aggregate import ChunkRow, parse_report, stirling2


DEFAULT_REPORTS = (
    "research/uace_pair_preemption_exact_c3_gap_profiles000k_001k.md",
    "research/uace_pair_preemption_exact_c3_root0_mask01_04_profiles001k_006k.md",
    "research/uace_pair_preemption_exact_c3_root6_mask01_04_profiles001k_006k.md",
    "research/uace_pair_preemption_exact_c3_root10_profiles000k_005k.md",
    "research/uace_pair_preemption_exact_c3_root10_profiles005k_010k.md",
    "research/uace_pair_preemption_exact_c3_root10_profiles010k_015k.md",
    "research/uace_pair_preemption_exact_c3_root10_profiles015k_020k.md",
)


@dataclass(frozen=True)
class ManifestEntry:
    colors: int
    attempt: str
    mask: tuple[int, ...]
    multiplicity: int
    intervals: tuple[tuple[int, int], ...]
    covered_profiles: int
    canonical_profiles: int
    coverage: float
    next_start: int
    next_stop: int


def merge_intervals(rows: list[ChunkRow]) -> tuple[tuple[int, int], ...]:
    intervals = sorted((row.start, row.end) for row in rows)
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            old_start, old_end = merged[-1]
            merged[-1] = (old_start, max(old_end, end))
    return tuple(merged)


def first_gap(intervals: tuple[tuple[int, int], ...], canonical: int) -> int:
    cursor = 0
    for start, end in intervals:
        if cursor < start:
            return cursor
        cursor = max(cursor, end)
    return min(cursor, canonical)


def build_entries(args: argparse.Namespace) -> list[ManifestEntry]:
    rows = [row for report in args.reports if Path(report).exists() for row in parse_report(Path(report))]
    by_key: dict[tuple[int, str, tuple[int, ...]], list[ChunkRow]] = {}
    for row in rows:
        by_key.setdefault(row.key, []).append(row)

    entries: list[ManifestEntry] = []
    for (colors, attempt, mask), bucket in sorted(by_key.items()):
        canonical = stirling2(args.known_sections, colors)
        intervals = merge_intervals(bucket)
        covered = sum(end - start for start, end in intervals)
        gap = first_gap(intervals, canonical)
        next_stop = min(gap + args.chunk_size, canonical)
        entries.append(
            ManifestEntry(
                colors=colors,
                attempt=attempt,
                mask=mask,
                multiplicity=bucket[0].multiplicity,
                intervals=intervals,
                covered_profiles=covered,
                canonical_profiles=canonical,
                coverage=covered / canonical if canonical else 0.0,
                next_start=gap,
                next_stop=next_stop,
            )
        )
    return entries


def command_for_entry(entry: ManifestEntry, args: argparse.Namespace) -> str:
    if entry.next_start >= entry.canonical_profiles:
        return "complete"
    max_profiles = entry.next_stop - entry.next_start
    attempt_arg = f'--only-attempt "{entry.attempt}"'
    mask_arg = (
        "--only-mask-sections "
        + " ".join(str(section) for section in entry.mask)
        + f" --mask-multiplicity {entry.multiplicity}"
    )
    if entry.attempt == "phase-III root 10" and entry.mask == (6, 10):
        output = (
            f"research/uace_pair_preemption_exact_c3_root10_profiles"
            f"{entry.next_start // 1000:03d}k_{entry.next_stop // 1000:03d}k.md"
        )
        return (
            "python3 research/uace_pair_preemption_exact.py --colors 3 "
            f"{attempt_arg} {mask_arg} --profile-offset {entry.next_start} "
            f"--max-profiles {max_profiles} --output {output}"
        )
    output = (
        f"research/uace_pair_preemption_exact_c3_"
        f"{entry.attempt.replace('phase-III ', '').replace(' ', '')}_"
        f"mask{'_'.join(f'{section:02d}' for section in entry.mask)}_profiles"
        f"{entry.next_start // 1000:03d}k_{entry.next_stop // 1000:03d}k.md"
    )
    return (
        "python3 research/uace_pair_preemption_exact.py --colors 3 "
        f"{attempt_arg} {mask_arg} --profile-offset {entry.next_start} "
        f"--max-profiles {max_profiles} --output {output}"
    )


def write_json(args: argparse.Namespace, entries: list[ManifestEntry]) -> None:
    payload = {
        "known_sections": args.known_sections,
        "chunk_size": args.chunk_size,
        "entries": [
            {
                "colors": entry.colors,
                "attempt": entry.attempt,
                "mask": list(entry.mask),
                "multiplicity": entry.multiplicity,
                "intervals": [list(item) for item in entry.intervals],
                "covered_profiles": entry.covered_profiles,
                "canonical_profiles": entry.canonical_profiles,
                "coverage": entry.coverage,
                "next_start": entry.next_start,
                "next_stop": entry.next_stop,
                "next_command": command_for_entry(entry, args),
            }
            for entry in entries
        ],
    }
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_report(args: argparse.Namespace, entries: list[ManifestEntry]) -> str:
    lines = [
        "# UACE 3-Color Chunk Manifest",
        "",
        "This report is generated by `research/uace_multicolor_manifest.py`.",
        "",
        f"- known sections per represented two-erasure path: `{args.known_sections}`",
        f"- default next chunk size: `{args.chunk_size}`",
        f"- JSON manifest: `{args.json_output}`",
        "",
        "| colors | attempt | mask | mult. | intervals | coverage | next interval | next command |",
        "|---:|---|---|---:|---|---:|---:|---|",
    ]
    for entry in entries:
        intervals = ", ".join(f"{start}-{end}" for start, end in entry.intervals)
        next_interval = (
            "complete"
            if entry.next_start >= entry.canonical_profiles
            else f"{entry.next_start}-{entry.next_stop}"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(entry.colors),
                    entry.attempt,
                    str(entry.mask),
                    str(entry.multiplicity),
                    intervals,
                    format_probability(entry.coverage),
                    next_interval,
                    f"`{command_for_entry(entry, args)}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Readout:",
            "",
            "- This manifest is a reproducibility layer for the remaining multi-color theorem work.",
            "- It records profile intervals, not just report filenames, so duplicate windows and future chunks can be audited.",
            "- Each next command now targets one attempt/mask interval, including its represented multiplicity.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", default=list(DEFAULT_REPORTS))
    parser.add_argument("--known-sections", type=int, default=14)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--json-output", type=Path, default=Path("research/uace_multicolor_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("research/uace_multicolor_manifest.md"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entries = build_entries(args)
    write_json(args, entries)
    args.output.write_text(build_report(args, entries) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
