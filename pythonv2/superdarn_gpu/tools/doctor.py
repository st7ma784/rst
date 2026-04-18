"""Environment diagnostics for SuperDARN GPU usability checks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from typing import Any, Dict

from ..core.backends import get_backend


def _run_cmd(cmd: list[str]) -> Dict[str, Any]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
        }


def gather_diagnostics(run_autotune: bool = False) -> Dict[str, Any]:
    env_requested = os.environ.get("SUPERDARN_BACKEND", "cupy")
    diag: Dict[str, Any] = {
        "requested_backend": env_requested,
        "effective_backend": get_backend().value,
        "rst_disable_cuda": os.environ.get("RST_DISABLE_CUDA", "0"),
    }

    cupy_info: Dict[str, Any] = {"installed": False, "cuda_available": False}
    try:
        import cupy as cp

        cupy_info["installed"] = True
        cupy_info["version"] = cp.__version__
        cupy_info["cuda_available"] = bool(cp.cuda.is_available())
        if cupy_info["cuda_available"]:
            props = cp.cuda.runtime.getDeviceProperties(0)
            name = props.get("name", b"unknown")
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="ignore")
            cupy_info["device_name"] = str(name)
    except Exception as exc:
        cupy_info["error"] = str(exc)

    diag["cupy"] = cupy_info
    diag["nvidia_smi"] = _run_cmd(["nvidia-smi"])

    if run_autotune:
        try:
            from .autotune import benchmark_fitacf_batch_sizes

            diag["fitacf_autotune"] = benchmark_fitacf_batch_sizes(
                candidate_sizes=(256, 512, 1024),
                repeats=2,
                warmup=1,
                backend="cupy" if cupy_info.get("cuda_available") else "numpy",
            )
        except Exception as exc:
            diag["fitacf_autotune"] = {"error": str(exc)}

    return diag


def _print_human(diag: Dict[str, Any]) -> None:
    print("SuperDARN Doctor")
    print("===============")
    print(f"Requested backend: {diag['requested_backend']}")
    print(f"Effective backend: {diag['effective_backend']}")
    print(f"RST_DISABLE_CUDA: {diag['rst_disable_cuda']}")

    cupy = diag.get("cupy", {})
    cupy_state = "YES" if cupy.get("installed") else "NO"
    print(f"CuPy installed: {cupy_state}")
    if cupy.get("installed"):
        print(f"CuPy version: {cupy.get('version', 'unknown')}")
        print(f"CuPy CUDA available: {bool(cupy.get('cuda_available'))}")
        if cupy.get("device_name"):
            print(f"CUDA device: {cupy['device_name']}")
    if cupy.get("error"):
        print(f"CuPy error: {cupy['error']}")

    nvsmi = diag.get("nvidia_smi", {})
    print(f"nvidia-smi ok: {bool(nvsmi.get('ok'))}")
    if nvsmi.get("stdout"):
        lines = nvsmi["stdout"].splitlines()
        print("nvidia-smi output (first 3 lines):")
        for line in lines[:3]:
            print(f"  {line}")

    autotune = diag.get("fitacf_autotune")
    if autotune:
        if "error" in autotune:
            print(f"FITACF autotune: ERROR ({autotune['error']})")
        else:
            print("FITACF autotune:")
            print(f"  backend={autotune['backend']} best_batch_size={autotune['best_batch_size']}")
            print(f"  timings_sec={autotune['timings_sec']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--autotune", action="store_true", help="Run quick FITACF batch-size auto-tune")
    args = parser.parse_args()

    diag = gather_diagnostics(run_autotune=args.autotune)
    if args.json:
        print(json.dumps(diag, indent=2))
    else:
        _print_human(diag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
