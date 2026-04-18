"""Benchmark command wrappers for SuperDARN tools."""

from __future__ import annotations

import argparse

from .autotune import benchmark_fitacf_batch_sizes


def main() -> int:
    parser = argparse.ArgumentParser(description="Run quick SuperDARN FITACF benchmark utilities")
    parser.add_argument("--backend", choices=["cupy", "numpy"], default="numpy")
    parser.add_argument("--sizes", default="256,512,1024", help="Comma-separated batch sizes")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    sizes = tuple(int(x) for x in args.sizes.split(",") if x.strip())
    out = benchmark_fitacf_batch_sizes(
        candidate_sizes=sizes,
        repeats=max(1, args.repeats),
        warmup=max(0, args.warmup),
        backend=args.backend,
    )

    print("FITACF batch-size benchmark")
    print(f"backend={out['backend']} best_batch_size={out['best_batch_size']}")
    for size, timing in sorted(out["timings_sec"].items()):
        print(f"  batch_size={size} mean_sec={timing:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
