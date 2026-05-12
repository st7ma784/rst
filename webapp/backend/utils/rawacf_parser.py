"""
Shared binary parser for the synthetic test rawacf format produced by
test_data/generate_test_data.py.

Header layout (44 bytes, little-endian):
  uint32  radar_id
  uint16  year, month, day, hour, minute, second   (6 × 2 = 12 bytes)
  uint16  nrang, nave, mplgs, mpinc, frang, rsep,
          bmnum, channel, cp, scan                  (10 × 2 = 20 bytes)
  float32 noise_lev, tfreq_khz                       (2 × 4 = 8 bytes)

Followed by four contiguous float32 blocks each of shape (nrang, mplgs):
  acf_real, acf_imag, xcf_real, xcf_imag
"""

import struct
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np

_HDR_FMT    = "<I" + "H" * 16 + "ff"       # 4 + 32 + 8 = 44 bytes
_HDR_SIZE   = struct.calcsize(_HDR_FMT)     # 44
_HDR_FIELDS = [
    "radar_id",
    "year", "month", "day", "hour", "minute", "second",
    "nrang", "nave", "mplgs", "mpinc", "frang", "rsep",
    "bmnum", "channel", "cp", "scan",
    "noise_lev", "tfreq",
]


def _parse_record(raw: bytes, offset: int) -> Tuple[Dict, int]:
    """Parse one record starting at `offset`. Returns (record_dict, next_offset)."""
    if offset + _HDR_SIZE > len(raw):
        raise StopIteration

    hdr_vals = struct.unpack_from(_HDR_FMT, raw, offset)
    header   = dict(zip(_HDR_FIELDS, hdr_vals))
    nrang    = int(header["nrang"])
    mplgs    = int(header["mplgs"])

    if nrang <= 0 or mplgs <= 0:
        raise StopIteration

    n        = nrang * mplgs
    rec_size = _HDR_SIZE + 4 * n * 4

    if offset + rec_size > len(raw):
        raise StopIteration

    def _arr(off: int) -> np.ndarray:
        return np.frombuffer(raw, dtype="<f4", count=n, offset=off).reshape(nrang, mplgs).copy()

    base     = offset + _HDR_SIZE
    acf_real = _arr(base);               base += n * 4
    acf_imag = _arr(base);               base += n * 4
    xcf_real = _arr(base);               base += n * 4
    xcf_imag = _arr(base)

    return {
        "header":   header,
        "acf_real": acf_real,
        "acf_imag": acf_imag,
        "xcf_real": xcf_real,
        "xcf_imag": xcf_imag,
    }, offset + rec_size


def parse_rawacf_file(
    file_path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Parse the first record of a rawacf file.
    Kept for backward compatibility with single-record callers.
    """
    records = parse_rawacf_all_records(file_path)
    if not records:
        raise ValueError("No records found in file")
    r = records[0]
    return r["acf_real"], r["acf_imag"], r["xcf_real"], r["xcf_imag"], r["header"]


class NotRawACFError(ValueError):
    """Raised when a file is not in the expected rawacf binary format."""


def detect_file_type(file_path) -> str:
    """
    Return 'rawacf', 'fitacf', or 'unknown' based on file content.
    Uses the header heuristic: a valid rawacf has plausible nrang (1–500)
    and mplgs (1–50) at the known byte offsets.
    """
    raw = Path(file_path).read_bytes()
    if len(raw) < _HDR_SIZE:
        return "unknown"
    try:
        vals  = struct.unpack_from(_HDR_FMT, raw, 0)
        hdr   = dict(zip(_HDR_FIELDS, vals))
        nrang = int(hdr["nrang"])
        mplgs = int(hdr["mplgs"])
        tfreq = float(hdr["tfreq"])
        if 1 <= nrang <= 500 and 1 <= mplgs <= 50 and 1000 <= tfreq <= 50000:
            # Check that the data section is the right size
            expected = _HDR_SIZE + 4 * nrang * mplgs * 4
            if len(raw) >= expected:
                return "rawacf"
    except Exception:
        pass
    # Check filename extension
    name = str(file_path).lower()
    if "fitacf" in name or name.endswith(".fitacf"):
        return "fitacf"
    if "grid" in name or name.endswith(".grid"):
        return "grid"
    return "unknown"


def parse_rawacf_all_records(file_path) -> list:
    """
    Parse ALL records from a rawacf file.

    Returns a list of dicts, each with keys:
      header, acf_real, acf_imag, xcf_real, xcf_imag
    """
    raw     = Path(file_path).read_bytes()
    records = []
    offset  = 0
    while offset < len(raw):
        try:
            rec, offset = _parse_record(raw, offset)
            records.append(rec)
        except StopIteration:
            break
    return records
