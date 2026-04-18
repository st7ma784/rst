#!/usr/bin/env python3
"""Run FITACF CPU/GPU batch benchmark and regression checks against fixed legacy reference."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np

from superdarn_gpu.core.backends import BackendContext, synchronize
from superdarn_gpu.processing.fitacf import FitACFProcessor


def _load_reference(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_input(reference: Dict[str, Any], scale: int = 1) -> Dict[str, Any]:
    nrang = int(reference["nrang"])
    mplgs = int(reference["mplgs"])
    mpinc_us = float(reference["mpinc_us"])
    noise_floor = float(reference.get("noise_floor", 2.5))

    acf = np.zeros((nrang, mplgs), dtype=np.complex64)
    power = np.zeros(nrang, dtype=np.float32)
    noise = np.full(nrang, noise_floor, dtype=np.float32)

    profiles = reference["range_profiles"]
    for profile in profiles:
        r = int(profile["range"])
        v = float(profile["velocity"])
        w = float(profile["spectral_width"])
        p = float(profile["power"])
        for lag in range(mplgs):
            lag_time = lag * mpinc_us * 1e-6
            decay = np.exp(-w * lag_time / 100.0)
            phase = v * lag_time / 200.0
            acf[r, lag] = np.complex64(p * decay * (np.cos(phase) + 1j * np.sin(phase)))
        power[r] = np.real(acf[r, 0]).astype(np.float32)

    # Deterministic low-level background for non-target ranges.
    for r in range(nrang):
        if power[r] > 0:
            continue
        for lag in range(mplgs):
            re = 0.5 + 0.01 * ((r + lag) % 5)
            im = 0.3 + 0.01 * ((r * lag + 1) % 7)
            acf[r, lag] = np.complex64(re + 1j * im)
        power[r] = np.real(acf[r, 0]).astype(np.float32)

    prm = SimpleNamespace(mpinc=int(mpinc_us), mplgs=mplgs, nrang=nrang, nave=32)

    if scale > 1:
        acf = np.tile(acf, (scale, 1))
        power = np.tile(power, scale)
        noise = np.tile(noise, scale)
        prm = SimpleNamespace(mpinc=int(mpinc_us), mplgs=mplgs, nrang=nrang * scale, nave=32)

    return {
        "acf": acf,
        "power": power,
        "noise": noise,
        "prm": prm,
    }


def _as_numpy(arr: Any) -> np.ndarray:
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)


def _run_backend(backend: str, data: Dict[str, Any], iterations: int, warmup: int) -> Dict[str, Any]:
    with BackendContext(backend):
        processor = FitACFProcessor()

        for _ in range(max(0, warmup)):
            _ = processor.process(data)
            if backend == "cupy":
                synchronize()

        times: List[float] = []
        final_result = None
        for _ in range(max(1, iterations)):
            t0 = time.perf_counter()
            final_result = processor.process(data)
            if backend == "cupy":
                synchronize()
            times.append(time.perf_counter() - t0)

        assert final_result is not None

        return {
            "backend": backend,
            "timings_sec": times,
            "mean_sec": float(np.mean(times)),
            "std_sec": float(np.std(times)),
            "result": {
                "velocity": _as_numpy(final_result.velocity),
                "spectral_width": _as_numpy(final_result.spectral_width),
                "power": _as_numpy(final_result.power),
                "qflg": _as_numpy(final_result.qflg),
            },
        }


def _compute_regression_metrics(reference: Dict[str, Any], output: Dict[str, Any]) -> Dict[str, float]:
    ranges = [int(r) for r in reference["target_ranges"]]
    expected_by_range = {int(p["range"]): p for p in reference["range_profiles"]}

    vel_expected = np.array([expected_by_range[r]["velocity"] for r in ranges], dtype=np.float64)
    wid_expected = np.array([expected_by_range[r]["spectral_width"] for r in ranges], dtype=np.float64)
    pwr_expected = np.array([expected_by_range[r]["power"] for r in ranges], dtype=np.float64)

    vel_fit = np.array([output["velocity"][r] for r in ranges], dtype=np.float64)
    wid_fit = np.array([output["spectral_width"][r] for r in ranges], dtype=np.float64)
    pwr_fit = np.array([output["power"][r] for r in ranges], dtype=np.float64)

    vel_mae = float(np.mean(np.abs(vel_fit - vel_expected)))
    width_mae = float(np.mean(np.abs(wid_fit - wid_expected)))
    power_rel_mae = float(np.mean(np.abs(pwr_fit - pwr_expected) / np.maximum(np.abs(pwr_expected), 1e-9)))

    return {
        "velocity_mae": vel_mae,
        "width_mae": width_mae,
        "power_rel_mae": power_rel_mae,
    }


def _compute_backend_delta_metrics(reference: Dict[str, Any], a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    ranges = [int(r) for r in reference["target_ranges"]]

    vel_a = np.array([a["velocity"][r] for r in ranges], dtype=np.float64)
    wid_a = np.array([a["spectral_width"][r] for r in ranges], dtype=np.float64)
    pwr_a = np.array([a["power"][r] for r in ranges], dtype=np.float64)

    vel_b = np.array([b["velocity"][r] for r in ranges], dtype=np.float64)
    wid_b = np.array([b["spectral_width"][r] for r in ranges], dtype=np.float64)
    pwr_b = np.array([b["power"][r] for r in ranges], dtype=np.float64)

    return {
        "velocity_mae": float(np.mean(np.abs(vel_a - vel_b))),
        "width_mae": float(np.mean(np.abs(wid_a - wid_b))),
        "power_rel_mae": float(np.mean(np.abs(pwr_a - pwr_b) / np.maximum(np.abs(pwr_a), 1e-9))),
    }


def _is_cupy_available() -> bool:
    try:
        import cupy as cp

        return bool(cp.cuda.is_available())
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("tests/reference/fitacf_legacy_reference.json"))
    parser.add_argument("--backend", choices=["numpy", "cupy", "both"], default="both")
    parser.add_argument("--mode", choices=["benchmark", "regression", "all"], default="all")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--scale", type=int, default=32, help="Dataset replication factor for benchmark throughput")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/fitacf_batch_benchmark.json"))
    parser.add_argument("--require-cupy", action="store_true", help="Fail if CuPy backend is unavailable")
    args = parser.parse_args()

    reference = _load_reference(args.reference)

    run_numpy = args.backend in {"numpy", "both"}
    run_cupy = args.backend in {"cupy", "both"}

    cupy_available = _is_cupy_available()
    if run_cupy and not cupy_available:
        if args.require_cupy:
            print("ERROR: CuPy backend requested but not available")
            return 2
        print("WARNING: CuPy backend unavailable; continuing with NumPy only")
        run_cupy = False

    regression_data = _build_input(reference, scale=1)
    benchmark_data = _build_input(reference, scale=max(1, args.scale))

    report: Dict[str, Any] = {
        "reference": str(args.reference),
        "mode": args.mode,
        "backend": args.backend,
        "cupy_available": cupy_available,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "scale": args.scale,
    }

    backend_reports: Dict[str, Any] = {}

    if run_numpy:
        data = benchmark_data if args.mode in {"benchmark", "all"} else regression_data
        backend_reports["numpy"] = _run_backend("numpy", data, args.iterations, args.warmup)

    if run_cupy:
        data = benchmark_data if args.mode in {"benchmark", "all"} else regression_data
        backend_reports["cupy"] = _run_backend("cupy", data, args.iterations, args.warmup)

    if args.mode in {"regression", "all"}:
        regression_summary: Dict[str, Any] = {}
        tolerances = reference.get("tolerances", {})

        # Always compute regression from scale=1 data.
        numpy_reg: Optional[Dict[str, Any]] = None
        cupy_reg: Optional[Dict[str, Any]] = None

        if run_numpy:
            numpy_reg = _run_backend("numpy", regression_data, 1, 0)
            metrics = _compute_regression_metrics(reference, numpy_reg["result"])
            regression_summary["numpy"] = metrics

        if run_cupy:
            cupy_reg = _run_backend("cupy", regression_data, 1, 0)
            metrics = _compute_regression_metrics(reference, cupy_reg["result"])
            regression_summary["cupy"] = metrics

        if numpy_reg is not None and cupy_reg is not None:
            regression_summary["numpy_cupy_delta"] = _compute_backend_delta_metrics(
                reference,
                numpy_reg["result"],
                cupy_reg["result"],
            )

        regression_pass = True
        checks: Dict[str, bool] = {}

        if "numpy" in regression_summary:
            npm = regression_summary["numpy"]
            checks["numpy_velocity"] = npm["velocity_mae"] <= float(tolerances.get("velocity_mae_max", 260.0))
            checks["numpy_width"] = npm["width_mae"] <= float(tolerances.get("width_mae_max", 120.0))
            checks["numpy_power"] = npm["power_rel_mae"] <= float(tolerances.get("power_rel_mae_max", 0.35))

        if "cupy" in regression_summary:
            cpm = regression_summary["cupy"]
            checks["cupy_velocity"] = cpm["velocity_mae"] <= float(tolerances.get("velocity_mae_max", 260.0))
            checks["cupy_width"] = cpm["width_mae"] <= float(tolerances.get("width_mae_max", 120.0))
            checks["cupy_power"] = cpm["power_rel_mae"] <= float(tolerances.get("power_rel_mae_max", 0.35))

        if "numpy_cupy_delta" in regression_summary:
            delta = regression_summary["numpy_cupy_delta"]
            checks["numpy_cupy_velocity"] = delta["velocity_mae"] <= float(tolerances.get("numpy_cupy_velocity_mae_max", 30.0))
            checks["numpy_cupy_width"] = delta["width_mae"] <= float(tolerances.get("numpy_cupy_width_mae_max", 40.0))
            checks["numpy_cupy_power"] = delta["power_rel_mae"] <= float(tolerances.get("numpy_cupy_power_rel_mae_max", 0.08))

        regression_pass = all(checks.values()) if checks else False
        regression_summary["checks"] = checks
        regression_summary["passed"] = regression_pass
        report["regression"] = regression_summary

    if args.mode in {"benchmark", "all"}:
        benchmark_summary: Dict[str, Any] = {}
        if "numpy" in backend_reports:
            benchmark_summary["numpy_mean_sec"] = backend_reports["numpy"]["mean_sec"]
            benchmark_summary["numpy_std_sec"] = backend_reports["numpy"]["std_sec"]
        if "cupy" in backend_reports:
            benchmark_summary["cupy_mean_sec"] = backend_reports["cupy"]["mean_sec"]
            benchmark_summary["cupy_std_sec"] = backend_reports["cupy"]["std_sec"]
        if "numpy" in backend_reports and "cupy" in backend_reports and backend_reports["cupy"]["mean_sec"] > 0:
            benchmark_summary["speedup_cupy_vs_numpy"] = (
                backend_reports["numpy"]["mean_sec"] / backend_reports["cupy"]["mean_sec"]
            )
        report["benchmark"] = benchmark_summary

    args.output.parent.mkdir(parents=True, exist_ok=True)

    serializable_report = dict(report)
    serializable_report["backend_details"] = {
        k: {key: value for key, value in v.items() if key != "result"}
        for k, v in backend_reports.items()
    }

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(serializable_report, f, indent=2)

    print(json.dumps(serializable_report, indent=2))

    if args.mode in {"regression", "all"}:
        if not report.get("regression", {}).get("passed", False):
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
