from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import yaml

from niti_bfr.braided_benchmark import (
    BRAIDED_CALIBRATION_BENCHMARK_NAMES,
    build_braided_benchmark_config,
    get_braided_benchmark_scenario,
    list_braided_benchmark_names,
    load_project_config,
    run_braided_benchmark,
)


class BraidedBenchmarkRegistryTests(unittest.TestCase):
    def test_calibration_benchmarks_are_registered(self) -> None:
        all_names = list_braided_benchmark_names()
        for name in BRAIDED_CALIBRATION_BENCHMARK_NAMES:
            self.assertIn(name, all_names)
            scenario = get_braided_benchmark_scenario(name)
            self.assertTrue(scenario.description)
            self.assertGreater(len(scenario.calibration_focus), 0)

    def test_calibration_benchmarks_override_distinct_stressors(self) -> None:
        config = load_project_config()
        base = build_braided_benchmark_config(config, "braided_demo")
        asymmetric = build_braided_benchmark_config(config, "braided_demo_asymmetric")
        attachment = build_braided_benchmark_config(config, "braided_demo_attachment_stress")

        self.assertGreater(asymmetric["peak_shift_norm"], base["peak_shift_norm"])
        self.assertLess(asymmetric["left_profile_power"], base["left_profile_power"])
        self.assertGreater(asymmetric["right_profile_power"], base["right_profile_power"])

        self.assertGreater(attachment["render"]["support_length_px"], base["render"]["support_length_px"])
        self.assertGreater(attachment["render"]["tip_cap_radius_px"], base["render"]["tip_cap_radius_px"])
        self.assertGreater(attachment["render"]["noise_sigma"], base["render"]["noise_sigma"])

    def test_benchmark_summary_includes_route_level_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_project_config()
            config["braided_synthetic"]["fps"] = 8
            config["braided_synthetic"]["duration_sec"] = 2.0
            config_path = Path(tmpdir) / "minimal-benchmark.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            run = run_braided_benchmark("braided_demo", output_root=tmpdir, config_path=config_path)

        summary = run.summary
        self.assertIn("route_results", summary)
        self.assertIn("route_results_by_alias", summary)
        self.assertIn("object_reportability_status", summary)
        self.assertIn("object_recommended_route_alias", summary)

        route_results = summary["route_results"]
        route_results_by_alias = summary["route_results_by_alias"]
        self.assertIsInstance(route_results, list)
        self.assertIsInstance(route_results_by_alias, dict)
        self.assertEqual(set(route_results_by_alias), {"A", "B", "C"})
        self.assertEqual(len(route_results), 3)

        for alias in ("A", "B", "C"):
            entry = route_results_by_alias[alias]
            self.assertEqual(entry["alias"], alias)
            self.assertIn("metric_key", entry)
            self.assertIn("reportability_status", entry)
            self.assertIn("gate_reason", entry)


if __name__ == "__main__":
    unittest.main()
