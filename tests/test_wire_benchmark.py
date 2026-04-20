from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import yaml

from niti_bfr.wire_benchmark import (
    WIRE_CALIBRATION_BENCHMARK_NAMES,
    build_wire_benchmark_config,
    flatten_wire_benchmark_summary,
    get_wire_benchmark_scenario,
    list_wire_benchmark_names,
    load_project_config,
    run_wire_benchmark,
)


class WireBenchmarkRegistryTests(unittest.TestCase):
    def test_calibration_benchmarks_are_registered(self) -> None:
        all_names = list_wire_benchmark_names()
        for name in WIRE_CALIBRATION_BENCHMARK_NAMES:
            self.assertIn(name, all_names)
            scenario = get_wire_benchmark_scenario(name)
            self.assertTrue(scenario.description)
            self.assertGreater(len(scenario.calibration_focus), 0)

    def test_calibration_benchmarks_override_distinct_stressors(self) -> None:
        config = load_project_config()
        base = build_wire_benchmark_config(config, "wire_demo")
        noisy = build_wire_benchmark_config(config, "wire_demo_high_noise")
        low_dynamic = build_wire_benchmark_config(config, "wire_demo_low_dynamic")

        self.assertGreater(noisy["synthetic"]["render"]["noise_sigma"], base["synthetic"]["render"]["noise_sigma"])
        self.assertGreater(noisy["synthetic"]["render"]["blur_sigma"], base["synthetic"]["render"]["blur_sigma"])
        self.assertLess(noisy["synthetic"]["line_thickness_px"], base["synthetic"]["line_thickness_px"])

        self.assertLess(low_dynamic["synthetic"]["kappa_m"], base["synthetic"]["kappa_m"])
        self.assertGreater(low_dynamic["synthetic"]["kappa_a"], base["synthetic"]["kappa_a"])
        self.assertGreater(
            low_dynamic["synthetic"]["transition_width_c"],
            base["synthetic"]["transition_width_c"],
        )


class WireBenchmarkRunTests(unittest.TestCase):
    def test_benchmark_summary_includes_route_level_results_and_suite_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_project_config()
            config["synthetic"]["fps"] = 8
            config["synthetic"]["duration_sec"] = 2.0
            config_path = Path(tmpdir) / "minimal-benchmark.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

            run = run_wire_benchmark("wire_demo", output_root=tmpdir, config_path=config_path)

        summary = run.summary
        self.assertIn("route_results", summary)
        self.assertIn("route_results_by_alias", summary)
        self.assertIn("route_benchmark_overview", summary)
        self.assertIn("route_benchmark_overview_by_alias", summary)
        self.assertEqual(summary["route_alias_order"], ["A", "B", "C"])
        self.assertIn("object_reportability_status", summary)
        self.assertIn("object_recommended_route_alias", summary)

        route_results = summary["route_results"]
        route_results_by_alias = summary["route_results_by_alias"]
        route_overview_by_alias = summary["route_benchmark_overview_by_alias"]
        self.assertIsInstance(route_results, list)
        self.assertEqual(len(route_results), 3)
        self.assertEqual(set(route_results_by_alias), {"A", "B", "C"})
        self.assertEqual(set(route_overview_by_alias), {"A", "B", "C"})

        for alias in ("A", "B", "C"):
            route_entry = route_results_by_alias[alias]
            overview_entry = route_overview_by_alias[alias]
            self.assertEqual(route_entry["alias"], alias)
            self.assertEqual(overview_entry["alias"], alias)
            self.assertIn("metric_key", route_entry)
            self.assertIn("reportability_status", route_entry)
            self.assertIn("gate_reason", route_entry)
            self.assertIn("truth_metric_key", overview_entry)
            self.assertIn("measured_af95_c", overview_entry)
            self.assertIn("af95_error_c", overview_entry)

        suite_row = flatten_wire_benchmark_summary(summary)
        self.assertEqual(suite_row["benchmark_name"], "wire_demo")
        self.assertIn("route_A_status", suite_row)
        self.assertIn("route_B_af95_error_c", suite_row)
        self.assertIn("route_C_available", suite_row)
        self.assertIn("quality_median", suite_row)
        self.assertIn("object_formal_route_alias", suite_row)
        self.assertIn("object_recommended_route_alias", suite_row)


if __name__ == "__main__":
    unittest.main()
