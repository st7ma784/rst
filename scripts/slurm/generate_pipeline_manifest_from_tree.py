#!/usr/bin/env python3
"""Generate a Slurm pipeline manifest by scanning RAWACF files in a directory tree.

This script creates rows keyed by (date, radar), so each row can become one
fit->grid task entry in the manifest.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

HEADER = [
    "date",
    "radar",
    "hemisphere",
    "input_glob",
    "fit_mode",
    "fitacf_version",
    "tdiff_method",
    "tdiff_value",
    "channel_mode",
    "scan_length_sec",
    "grid_interval_sec",
    "grid_extra_flags",
    "map_model",
    "map_order",
    "map_doping",
    "imf_mode",
    "imf_file",
    "bx",
    "by",
    "bz",
    "extra_map_flags",
    "use_cuda",
]

DATE_RE = re.compile(r"^\d{8}$")
RADAR_RE = re.compile(r"^[a-z]{3,4}$")

# Practical defaults for common network usage.
KNOWN_HEMISPHERE = {
    "gbr": "north",
    "sch": "north",
    "kap": "north",
    "sas": "north",
    "pgr": "north",
    "kod": "north",
    "sto": "north",
    "pyk": "north",
    "han": "north",
    "ksr": "north",
    "wal": "north",
    "cly": "north",
    "rkn": "north",
    "inv": "north",
    "san": "south",
    "sys": "south",
    "sye": "south",
    "tig": "south",
    "ker": "south",
    "unw": "south",
    "hal": "south",
}


def load_hemisphere_map(path: Optional[pathlib.Path]) -> Dict[str, str]:
    mapping = dict(KNOWN_HEMISPHERE)
    if path is None:
        return mapping

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"radar", "hemisphere"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError("Hemisphere map must contain radar,hemisphere columns")

        for row in reader:
            radar = (row.get("radar") or "").strip().lower()
            hemi = (row.get("hemisphere") or "").strip().lower()
            if not radar:
                continue
            if hemi not in {"north", "south"}:
                raise ValueError(f"Invalid hemisphere '{hemi}' for radar '{radar}'")
            mapping[radar] = hemi

    return mapping


def parse_date_radar(path: pathlib.Path) -> Optional[Tuple[str, str]]:
    name = path.name
    if ".rawacf" not in name:
        return None

    parts = name.split(".")
    date = ""
    radar = ""

    for token in parts:
        if DATE_RE.match(token):
            date = token
            break

    for idx, token in enumerate(parts):
        if token.startswith("rawacf") and idx > 0:
            cand = parts[idx - 1].lower()
            if RADAR_RE.match(cand):
                radar = cand
                break

    if not date or not radar:
        m = re.search(r"(\d{8})\.[0-9]{4}\.[0-9]{2}\.([a-z]{3,4})\.rawacf", name)
        if m:
            date = m.group(1)
            radar = m.group(2).lower()

    if not date or not radar:
        return None
    return date, radar


def in_date_range(date_str: str, start: Optional[str], end: Optional[str]) -> bool:
    if start and date_str < start:
        return False
    if end and date_str > end:
        return False
    return True


def walk_rawacf_files(root: pathlib.Path) -> Iterable[pathlib.Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if ".rawacf" in p.name:
            yield p


def validate_yyyymmdd(value: Optional[str], arg_name: str) -> None:
    if value is None:
        return
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"{arg_name} must be YYYYMMDD") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate RST Slurm manifest from RAWACF tree")
    parser.add_argument("--raw-root", required=True, help="Root directory containing RAWACF files")
    parser.add_argument("--output", required=True, help="Output manifest CSV")
    parser.add_argument("--hemisphere-map", help="CSV with radar,hemisphere columns")
    parser.add_argument("--default-hemisphere", default="north", choices=["north", "south"])
    parser.add_argument("--start-date", help="Inclusive start date YYYYMMDD")
    parser.add_argument("--end-date", help="Inclusive end date YYYYMMDD")
    parser.add_argument("--fit-mode", default="fitacf3")
    parser.add_argument("--fitacf-version", default="3.0")
    parser.add_argument("--tdiff-method", default="-")
    parser.add_argument("--tdiff-value", default="-")
    parser.add_argument("--channel-mode", default="all", choices=["all", "a", "b"])
    parser.add_argument("--scan-length-sec", default="60")
    parser.add_argument("--grid-interval-sec", default="120")
    parser.add_argument("--grid-extra-flags", default="-")
    parser.add_argument("--map-model", default="PSR")
    parser.add_argument("--map-order", default="8")
    parser.add_argument("--map-doping", default="l")
    parser.add_argument("--imf-mode", default="none", choices=["none", "file", "fixed", "ace", "wind"])
    parser.add_argument("--imf-file", default="-")
    parser.add_argument("--bx", default="-")
    parser.add_argument("--by", default="-")
    parser.add_argument("--bz", default="-")
    parser.add_argument("--extra-map-flags", default="-")
    parser.add_argument("--use-cuda", default="auto")
    args = parser.parse_args()

    try:
        validate_yyyymmdd(args.start_date, "--start-date")
        validate_yyyymmdd(args.end_date, "--end-date")
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    raw_root = pathlib.Path(args.raw_root).resolve()
    if not raw_root.exists() or not raw_root.is_dir():
        print(f"ERROR: raw root not found or not a directory: {raw_root}", file=sys.stderr)
        return 2

    hemi_path = pathlib.Path(args.hemisphere_map).resolve() if args.hemisphere_map else None
    if hemi_path and not hemi_path.exists():
        print(f"ERROR: hemisphere map does not exist: {hemi_path}", file=sys.stderr)
        return 2

    try:
        hemisphere_map = load_hemisphere_map(hemi_path)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    grouped: Dict[Tuple[str, str], List[pathlib.Path]] = defaultdict(list)
    scanned = 0

    for path in walk_rawacf_files(raw_root):
        scanned += 1
        parsed = parse_date_radar(path)
        if parsed is None:
            continue
        date, radar = parsed
        if not in_date_range(date, args.start_date, args.end_date):
            continue
        grouped[(date, radar)].append(path)

    out_path = pathlib.Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)

        row_count = 0
        for date, radar in sorted(grouped.keys()):
            hemi = hemisphere_map.get(radar, args.default_hemisphere)
            # Recursive pattern; exact resolver turns this into deterministic per-row lists.
            input_glob = str(raw_root / "**" / f"{date}*.{radar}.rawacf*")
            writer.writerow(
                [
                    date,
                    radar,
                    hemi,
                    input_glob,
                    args.fit_mode,
                    args.fitacf_version,
                    args.tdiff_method,
                    args.tdiff_value,
                    args.channel_mode,
                    args.scan_length_sec,
                    args.grid_interval_sec,
                    args.grid_extra_flags,
                    args.map_model,
                    args.map_order,
                    args.map_doping,
                    args.imf_mode,
                    args.imf_file,
                    args.bx,
                    args.by,
                    args.bz,
                    args.extra_map_flags,
                    args.use_cuda,
                ]
            )
            row_count += 1

    print(f"Scanned files: {scanned}")
    print(f"Manifest rows: {row_count}")
    print(f"Output manifest: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
