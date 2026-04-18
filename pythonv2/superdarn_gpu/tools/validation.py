"""Validation helpers for quick environment and processing sanity checks."""

from __future__ import annotations

import argparse

from .doctor import gather_diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a quick SuperDARN validation check")
    parser.add_argument("--require-gpu", action="store_true", help="Fail if GPU backend is not active")
    args = parser.parse_args()

    diag = gather_diagnostics(run_autotune=False)
    backend = diag.get("effective_backend")

    print("SuperDARN validation summary")
    print(f"effective_backend={backend}")
    print(f"cupy_cuda_available={diag.get('cupy', {}).get('cuda_available', False)}")

    if args.require_gpu and backend != "cupy":
        print("Validation failed: GPU backend required but not active")
        return 1

    print("Validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
