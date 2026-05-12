"""
Numerical comparison utilities for comparing backend outputs.

All tolerances are defined here so they can be tightened as backends converge.
"""

from typing import Any, Dict, List, Optional, Tuple
import math


# ── Tolerances ────────────────────────────────────────────────────────────────

# These are deliberately generous while backends are still being aligned.
# Tighten when pythonv2, CUDArst and RST are all using the same physics.
VELOCITY_ABS_TOLERANCE   = 50.0    # m/s   — within 50 m/s absolute
VELOCITY_REL_TOLERANCE   = 0.10    # 10 %  — or within 10% of larger value
POWER_ABS_TOLERANCE      = 2.0     # linear power (not dB)
POWER_REL_TOLERANCE      = 0.15    # 15 %
WIDTH_ABS_TOLERANCE      = 100.0   # m/s
WIDTH_REL_TOLERANCE      = 0.20    # 20 %
GOOD_RANGES_TOLERANCE    = 0.20    # fraction — good_ranges counts can differ by 20%
GRID_VEL_ABS_TOLERANCE   = 100.0   # m/s


class CompareResult:
    def __init__(self, field: str, passed: bool, detail: str):
        self.field  = field
        self.passed = passed
        self.detail = detail

    def __repr__(self):
        tag = "PASS" if self.passed else "FAIL"
        return f"[{tag}] {self.field}: {self.detail}"


def _close(a, b, abs_tol, rel_tol) -> bool:
    if a is None or b is None:
        return True   # missing data treated as N/A not a failure
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isnan(a) or math.isnan(b):
        return False
    diff = abs(a - b)
    if diff <= abs_tol:
        return True
    return diff <= rel_tol * max(abs(a), abs(b), 1e-9)


def _list_agreement(
    name: str, a: List, b: List, abs_tol: float, rel_tol: float
) -> CompareResult:
    """Check element-wise agreement between two lists. Reports worst mismatch."""
    if not a and not b:
        return CompareResult(name, True, "both empty")
    if not a or not b:
        return CompareResult(name, False, f"lengths differ: {len(a)} vs {len(b)}")

    n          = min(len(a), len(b))
    mismatches = 0
    worst_diff = 0.0

    for i in range(n):
        ai = a[i] if a[i] is not None else float("nan")
        bi = b[i] if b[i] is not None else float("nan")
        if not _close(ai, bi, abs_tol, rel_tol):
            mismatches += 1
            worst_diff  = max(worst_diff, abs(ai - bi))

    if mismatches == 0:
        return CompareResult(name, True, f"all {n} elements agree")
    pct = 100 * mismatches / n
    return CompareResult(
        name, mismatches / n < 0.10,   # pass if < 10 % of values disagree
        f"{mismatches}/{n} ({pct:.1f}%) disagree; worst diff={worst_diff:.2f}"
    )


def compare_acf_results(a: Dict, b: Dict) -> List[CompareResult]:
    results = []

    # nranges must match exactly
    ok = a.get("nranges") == b.get("nranges")
    results.append(CompareResult("acf.nranges", ok,
                                 f"{a.get('nranges')} vs {b.get('nranges')}"))

    results.append(_list_agreement(
        "acf.power", a.get("acf_power", []), b.get("acf_power", []),
        POWER_ABS_TOLERANCE, POWER_REL_TOLERANCE
    ))
    return results


def compare_fitacf_results(a: Dict, b: Dict) -> List[CompareResult]:
    results = []

    # nranges
    ok = a.get("nranges") == b.get("nranges")
    results.append(CompareResult("fitacf.nranges", ok,
                                 f"{a.get('nranges')} vs {b.get('nranges')}"))

    # good_ranges — allow 20% difference
    ga, gb = a.get("good_ranges", 0), b.get("good_ranges", 0)
    if max(ga, gb) > 0:
        ok = abs(ga - gb) / max(ga, gb) <= GOOD_RANGES_TOLERANCE
    else:
        ok = True
    results.append(CompareResult("fitacf.good_ranges", ok, f"{ga} vs {gb}"))

    # velocity, power, width arrays
    results.append(_list_agreement(
        "fitacf.velocity", a.get("velocity", []), b.get("velocity", []),
        VELOCITY_ABS_TOLERANCE, VELOCITY_REL_TOLERANCE
    ))
    results.append(_list_agreement(
        "fitacf.power", a.get("power", []), b.get("power", []),
        POWER_ABS_TOLERANCE, POWER_REL_TOLERANCE
    ))
    results.append(_list_agreement(
        "fitacf.spectral_width", a.get("spectral_width", []), b.get("spectral_width", []),
        WIDTH_ABS_TOLERANCE, WIDTH_REL_TOLERANCE
    ))
    return results


def compare_grid_results(a: Dict, b: Dict) -> List[CompareResult]:
    results = []
    results.append(_list_agreement(
        "grid.velocity", a.get("velocity", []), b.get("velocity", []),
        GRID_VEL_ABS_TOLERANCE, 0.20
    ))
    return results


def compare_stage_results(
    stage: str, a: Dict, b: Dict
) -> List[CompareResult]:
    if stage == "acf":
        return compare_acf_results(a, b)
    elif stage == "fitacf":
        return compare_fitacf_results(a, b)
    elif stage == "grid":
        return compare_grid_results(a, b)
    return []


def compare_pipelines(
    result_a: Dict, result_b: Dict,
    label_a: str = "A", label_b: str = "B"
) -> Dict[str, List[CompareResult]]:
    """
    Full pipeline comparison. Returns dict mapping stage → list of CompareResult.
    """
    stages_a = result_a.get("stages", {})
    stages_b = result_b.get("stages", {})
    all_stages = set(stages_a) | set(stages_b)

    out = {}
    for stage in sorted(all_stages):
        data_a = stages_a.get(stage, {})
        data_b = stages_b.get(stage, {})
        out[stage] = compare_stage_results(stage, data_a, data_b)

    return out


def summarise(comparison: Dict[str, List[CompareResult]]) -> str:
    lines = []
    total_pass = total_fail = 0
    for stage, checks in comparison.items():
        p = sum(1 for c in checks if c.passed)
        f = sum(1 for c in checks if not c.passed)
        total_pass += p
        total_fail += f
        lines.append(f"  {stage}: {p} pass, {f} fail")
        for c in checks:
            if not c.passed:
                lines.append(f"    ✗ {c}")
    lines.append(f"Total: {total_pass} pass, {total_fail} fail")
    return "\n".join(lines)
