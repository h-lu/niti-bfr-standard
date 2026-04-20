from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from niti_bfr.wire_benchmark import (
    WIRE_CALIBRATION_BENCHMARK_NAMES,
    flatten_wire_benchmark_summary,
    get_wire_benchmark_scenario,
    list_wire_benchmark_names,
    run_wire_benchmark,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run wire-like synthetic benchmark scenarios.")
    parser.add_argument(
        "benchmarks",
        nargs="*",
        help="Benchmark names to run. Defaults to the two calibration benchmarks.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all registered wire benchmarks, including the baseline wire_demo.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available benchmarks and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.list:
        for name in list_wire_benchmark_names():
            scenario = get_wire_benchmark_scenario(name)
            print(f"{name}: {scenario.description}")
        return

    benchmark_names = (
        list_wire_benchmark_names()
        if args.all
        else list(args.benchmarks) if args.benchmarks else list(WIRE_CALIBRATION_BENCHMARK_NAMES)
    )
    runs = [run_wire_benchmark(name, root=ROOT) for name in benchmark_names]
    suite_dir = ROOT / "outputs" / "wire_benchmark_suite"
    suite_dir.mkdir(parents=True, exist_ok=True)
    suite_summary = [flatten_wire_benchmark_summary(run.summary) for run in runs]

    (suite_dir / "benchmark_summary.json").write_text(
        json.dumps(suite_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (suite_dir / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(suite_summary[0].keys()))
        writer.writeheader()
        writer.writerows(suite_summary)

    for run in runs:
        print(f"{run.scenario.name} saved to {run.output_dir}")
    print(f"suite summary saved to {suite_dir}")


if __name__ == "__main__":
    main()
