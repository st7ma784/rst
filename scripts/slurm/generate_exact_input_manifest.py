#!/usr/bin/env python3
"""Resolve manifest input globs into exact file lists.

Input manifest is expected to have the header defined in scripts/slurm/README.md.
This tool rewrites the input_glob column to a special value:

  @/absolute/or/relative/path/to/list_file.txt

The list file contains one exact filename per line in deterministic sort order.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import pathlib
import sys
from typing import List


def safe_name(value: str) -> str:
    keep = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def resolve_files(pattern: str) -> List[str]:
    matches = glob.glob(pattern, recursive=True)
    return sorted(os.path.abspath(m) for m in matches)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build exact-input manifest for Slurm pipeline")
    parser.add_argument("--manifest", required=True, help="Input manifest CSV")
    parser.add_argument("--output", required=True, help="Output manifest CSV")
    parser.add_argument(
        "--list-dir",
        required=True,
        help="Directory where per-row input list files are written",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow rows with zero matching files (default: error)",
    )

    args = parser.parse_args()

    in_path = pathlib.Path(args.manifest)
    out_path = pathlib.Path(args.output)
    list_dir = pathlib.Path(args.list_dir)
    list_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("ERROR: Input manifest has no header", file=sys.stderr)
            return 2

        rows = list(reader)
        required = {"date", "radar", "input_glob"}
        missing = [k for k in required if k not in reader.fieldnames]
        if missing:
            print(f"ERROR: Missing required columns: {', '.join(missing)}", file=sys.stderr)
            return 2

    out_rows = []
    failures = 0

    for idx, row in enumerate(rows, start=1):
        date = row.get("date", "")
        radar = row.get("radar", "")
        pattern = row.get("input_glob", "")
        if not pattern:
            print(f"ERROR: Row {idx} has empty input_glob", file=sys.stderr)
            failures += 1
            continue

        files = resolve_files(pattern)
        if not files and not args.allow_empty:
            print(
                f"ERROR: Row {idx} ({date}/{radar}) pattern matched no files: {pattern}",
                file=sys.stderr,
            )
            failures += 1
            continue

        list_name = f"{idx:06d}_{safe_name(date)}_{safe_name(radar)}.files"
        list_path = list_dir / list_name
        with list_path.open("w", encoding="utf-8") as lf:
            for p in files:
                lf.write(p)
                lf.write("\n")

        row["input_glob"] = "@" + str(list_path)
        out_rows.append(row)

    if failures > 0 and not args.allow_empty:
        print(f"ERROR: resolve failed for {failures} row(s)", file=sys.stderr)
        return 3

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else None)
        if writer.fieldnames is None:
            print("ERROR: No rows to write", file=sys.stderr)
            return 4
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Resolved rows: {len(out_rows)}")
    print(f"Output manifest: {out_path}")
    print(f"List directory: {list_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
