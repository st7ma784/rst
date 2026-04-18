#!/usr/bin/env python3
"""Generate a rerun manifest containing only failed rows from a prior Slurm run.

Failure criteria:
- Missing fit marker for row:   meta/<date>/<radar>/fit.done
- Missing grid marker for row:  meta/<date>/<radar>/grid.done
- Missing map marker for group: meta/<date>/<hemisphere>/map.done (optional)

By default, group map failure includes all rows in that (date, hemisphere) group.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass
class RowStatus:
    index: int
    date: str
    radar: str
    hemisphere: str
    reasons: List[str]


def normalize_hemi(value: str) -> str:
    v = (value or "").strip().lower()
    if v in {"n", "north"}:
        return "north"
    if v in {"s", "south", "sh"}:
        return "south"
    return v or "north"


def load_manifest(path: pathlib.Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Manifest has no header")
        rows = list(reader)
    return list(reader.fieldnames), rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Create rerun manifest from previous run state")
    parser.add_argument("--run-dir", required=True, help="Run directory created by submit_rst_pipeline.sh")
    parser.add_argument(
        "--manifest",
        help="Manifest to evaluate (default: <run-dir>/resolved_manifest.csv, fallback to input_manifest.csv)",
    )
    parser.add_argument("--output", required=True, help="Output rerun manifest CSV")
    parser.add_argument("--reasons-out", help="Optional CSV report with failure reasons")
    parser.add_argument(
        "--skip-map-check",
        action="store_true",
        help="Do not treat missing map.done as failure trigger",
    )
    args = parser.parse_args()

    run_dir = pathlib.Path(args.run_dir).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        print(f"ERROR: run dir not found: {run_dir}", file=sys.stderr)
        return 2

    if args.manifest:
        manifest = pathlib.Path(args.manifest).resolve()
    else:
        resolved = run_dir / "resolved_manifest.csv"
        raw = run_dir / "input_manifest.csv"
        manifest = resolved if resolved.exists() else raw

    if not manifest.exists():
        print(f"ERROR: manifest not found: {manifest}", file=sys.stderr)
        return 2

    try:
        fieldnames, rows = load_manifest(manifest)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    for key in ("date", "radar", "hemisphere"):
        if key not in fieldnames:
            print(f"ERROR: manifest missing required column '{key}'", file=sys.stderr)
            return 2

    failed_idx: Dict[int, RowStatus] = {}
    group_rows: Dict[Tuple[str, str], List[int]] = {}

    for i, row in enumerate(rows, start=1):
        date = (row.get("date") or "").strip()
        radar = (row.get("radar") or "").strip().lower()
        hemi = normalize_hemi(row.get("hemisphere") or "")

        if not date or not radar:
            status = RowStatus(i, date, radar, hemi, ["invalid_row"])
            failed_idx[i] = status
            continue

        group_rows.setdefault((date, hemi), []).append(i)

        fit_done = run_dir / "meta" / date / radar / "fit.done"
        grid_done = run_dir / "meta" / date / radar / "grid.done"

        reasons: List[str] = []
        if not fit_done.exists():
            reasons.append("fit_missing")
        if not grid_done.exists():
            reasons.append("grid_missing")

        if reasons:
            failed_idx[i] = RowStatus(i, date, radar, hemi, reasons)

    if not args.skip_map_check:
        for (date, hemi), idxs in group_rows.items():
            map_done = run_dir / "meta" / date / hemi / "map.done"
            if map_done.exists():
                continue
            for idx in idxs:
                status = failed_idx.get(idx)
                if status is None:
                    row = rows[idx - 1]
                    status = RowStatus(
                        idx,
                        (row.get("date") or "").strip(),
                        (row.get("radar") or "").strip().lower(),
                        normalize_hemi(row.get("hemisphere") or ""),
                        [],
                    )
                    failed_idx[idx] = status
                if "map_missing_group" not in status.reasons:
                    status.reasons.append("map_missing_group")

    output = pathlib.Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    failed_rows = [rows[idx - 1] for idx in sorted(failed_idx.keys())]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failed_rows)

    reasons_path = pathlib.Path(args.reasons_out).resolve() if args.reasons_out else output.with_suffix(".reasons.csv")
    with reasons_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_index", "date", "radar", "hemisphere", "reasons"])
        for idx in sorted(failed_idx.keys()):
            st = failed_idx[idx]
            writer.writerow([st.index, st.date, st.radar, st.hemisphere, ";".join(sorted(st.reasons))])

    print(f"Source manifest: {manifest}")
    print(f"Input rows: {len(rows)}")
    print(f"Failed rows: {len(failed_rows)}")
    print(f"Rerun manifest: {output}")
    print(f"Failure reasons: {reasons_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
