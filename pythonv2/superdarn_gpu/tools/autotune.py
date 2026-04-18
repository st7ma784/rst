"""Auto-tuning helpers for FITACF batch sizing."""

from __future__ import annotations

import time
import os
from typing import Dict, Iterable, List, Any

import numpy as np

from ..core.backends import BackendContext, synchronize
from ..processing.fitacf import FitACFProcessor, FitACFConfig


def _synthetic_fitacf_input(nrang: int = 128, mplgs: int = 18, mpinc_us: int = 1500) -> Dict[str, Any]:
    """Create deterministic synthetic ACF data for quick batch-size timing."""
    acf = np.zeros((nrang, mplgs), dtype=np.complex64)
    power = np.zeros(nrang, dtype=np.float32)
    noise = np.full(nrang, 2.0, dtype=np.float32)

    for r in range(nrang):
        velocity = -300.0 + (600.0 * r / max(1, nrang - 1))
        width = 120.0 + 0.8 * r
        pwr = 250.0 + 100.0 * np.cos(r / 10.0)
        for lag in range(mplgs):
            lag_time = lag * mpinc_us * 1e-6
            decay = np.exp(-width * lag_time / 100.0)
            phase = velocity * lag_time / 200.0
            acf[r, lag] = np.complex64(pwr * decay * (np.cos(phase) + 1j * np.sin(phase)))
        power[r] = float(np.real(acf[r, 0]))

    class _Prm:
        mpinc = mpinc_us
        nave = 32

    return {"acf": acf, "power": power, "noise": noise, "prm": _Prm()}


def benchmark_fitacf_batch_sizes(
    candidate_sizes: Iterable[int] = (256, 512, 1024),
    repeats: int = 3,
    warmup: int = 1,
    nrang: int = 128,
    mplgs: int = 18,
    backend: str = "cupy",
) -> Dict[str, Any]:
    """Benchmark candidate FITACF batch sizes and return timing summary."""
    candidates: List[int] = [int(x) for x in candidate_sizes if int(x) > 0]
    if not candidates:
        raise ValueError("At least one positive batch size is required")

    data = _synthetic_fitacf_input(nrang=nrang, mplgs=mplgs)
    timings: Dict[int, float] = {}

    with BackendContext(backend):
        prev_auto = os.environ.get("SUPERDARN_AUTOTUNE_BATCH")
        os.environ["SUPERDARN_AUTOTUNE_BATCH"] = "0"
        try:
            for batch_size in candidates:
                processor = FitACFProcessor(config=FitACFConfig(batch_size=batch_size))

                for _ in range(max(0, warmup)):
                    _ = processor.process(data)
                    if backend == "cupy":
                        synchronize()

                run_times = []
                for _ in range(max(1, repeats)):
                    t0 = time.perf_counter()
                    _ = processor.process(data)
                    if backend == "cupy":
                        synchronize()
                    run_times.append(time.perf_counter() - t0)

                timings[batch_size] = float(np.mean(run_times))
        finally:
            if prev_auto is None:
                os.environ.pop("SUPERDARN_AUTOTUNE_BATCH", None)
            else:
                os.environ["SUPERDARN_AUTOTUNE_BATCH"] = prev_auto

    best_batch_size = min(timings, key=timings.get)
    return {
        "backend": backend,
        "nrang": nrang,
        "mplgs": mplgs,
        "repeats": repeats,
        "warmup": warmup,
        "timings_sec": timings,
        "best_batch_size": int(best_batch_size),
    }


def recommend_fitacf_batch_size(
    candidate_sizes: Iterable[int] = (256, 512, 1024),
    backend: str = "cupy",
) -> int:
    """Return the best observed batch size from a short timing sweep."""
    result = benchmark_fitacf_batch_sizes(
        candidate_sizes=candidate_sizes,
        repeats=2,
        warmup=1,
        nrang=128,
        mplgs=18,
        backend=backend,
    )
    return int(result["best_batch_size"])
