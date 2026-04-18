"""Top-level CLI wrapper for SuperDARN utility commands."""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="superdarn-process",
        description="SuperDARN utility entrypoint",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="help",
        choices=["help", "doctor", "benchmark", "validate"],
        help="Utility command to run",
    )
    args, _ = parser.parse_known_args()

    if args.command == "doctor":
        from .doctor import main as doctor_main

        return doctor_main()
    if args.command == "benchmark":
        from .benchmarks import main as benchmark_main

        return benchmark_main()
    if args.command == "validate":
        from .validation import main as validation_main

        return validation_main()

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
