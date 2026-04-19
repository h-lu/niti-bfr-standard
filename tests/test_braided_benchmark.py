from __future__ import annotations

import unittest

from niti_bfr.braided_benchmark import (
    BRAIDED_CALIBRATION_BENCHMARK_NAMES,
    build_braided_benchmark_config,
    get_braided_benchmark_scenario,
    list_braided_benchmark_names,
    load_project_config,
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


if __name__ == "__main__":
    unittest.main()
