from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from niti_bfr.braided_benchmark import (
    BRAIDED_CALIBRATION_BENCHMARK_NAMES,
    get_braided_benchmark_scenario,
    list_braided_benchmark_names,
    run_braided_benchmark,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run braided synthetic benchmark scenarios.")
    parser.add_argument(
        "benchmarks",
        nargs="*",
        help="Benchmark names to run. Defaults to the two calibration benchmarks.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all registered braided benchmarks, including the baseline braided_demo.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available benchmarks and exit.",
    )
    return parser.parse_args()


def _summary_row(summary: dict[str, object]) -> dict[str, object]:
    af_comparison = summary["af_comparison"]
    error_summary = summary["error_summary"]
    threshold_sensitivity = summary["diameter_threshold_sensitivity"]["baseline_max_delta"]
    return {
        "benchmark_name": summary["benchmark_name"],
        "description": summary["benchmark_description"],
        "output_dir": summary["demo_output"],
        "quality_median": summary["quality_median"],
        "body_mask_attachment_leak_fraction": summary["body_mask_attachment_leak_fraction"],
        "centerline_disagreement_median": summary["centerline_disagreement_median"],
        "endpoint_jump_p95_px": summary["endpoint_jump_p95_px"],
        "axis_peak_position_stability_p95": summary["axis_peak_position_stability_p95"],
        "length_axis_mae_px": error_summary["length_axis_mae_px"],
        "diameter_max_mae_px": error_summary["diameter_max_mae_px"],
        "area_proj_mae_px2": error_summary["area_proj_mae_px2"],
        "length_axis_af95_error_c": af_comparison["length_axis"]["af95_error_c"],
        "length_axis_aftan_error_c": af_comparison["length_axis"]["aftan_error_c"],
        "diameter_af95_error_c": af_comparison["diameter_max"]["af95_error_c"],
        "diameter_aftan_error_c": af_comparison["diameter_max"]["aftan_error_c"],
        "area_af95_error_c": af_comparison["area_proj"]["af95_error_c"],
        "area_aftan_error_c": af_comparison["area_proj"]["aftan_error_c"],
        "diameter_threshold_sensitivity_af95_c": threshold_sensitivity.get("af95_c"),
        "diameter_threshold_sensitivity_aftan_c": threshold_sensitivity.get("aftan_c"),
    }


def main() -> None:
    args = _parse_args()
    if args.list:
        for name in list_braided_benchmark_names():
            scenario = get_braided_benchmark_scenario(name)
            print(f"{name}: {scenario.description}")
        return

    benchmark_names = (
        list_braided_benchmark_names()
        if args.all
        else list(args.benchmarks) if args.benchmarks else list(BRAIDED_CALIBRATION_BENCHMARK_NAMES)
    )

    runs = [run_braided_benchmark(name, root=ROOT) for name in benchmark_names]
    suite_dir = ROOT / "outputs" / "braided_benchmark_suite"
    suite_dir.mkdir(parents=True, exist_ok=True)
    suite_summary = [_summary_row(run.summary) for run in runs]

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
