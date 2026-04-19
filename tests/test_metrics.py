from __future__ import annotations

import unittest

from niti_bfr.metrics import summarize_numeric_sweep


class SummarizeNumericSweepTests(unittest.TestCase):
    def test_reports_spreads_and_baseline_deltas(self) -> None:
        summary = summarize_numeric_sweep(
            {
                "137": {
                    "quality_median": 0.91,
                    "diameter_max_mae_px": 2.4,
                    "af95_c": 58.5,
                    "aftan_c": 57.8,
                },
                "145": {
                    "quality_median": 0.93,
                    "diameter_max_mae_px": 1.8,
                    "af95_c": 58.0,
                    "aftan_c": 57.0,
                },
                "153": {
                    "quality_median": 0.88,
                    "diameter_max_mae_px": 3.1,
                    "af95_c": 59.2,
                    "aftan_c": 58.1,
                },
            },
            baseline_key="145",
        )

        self.assertEqual(summary["variant_count"], 3)
        self.assertEqual(summary["baseline_key"], "145")
        self.assertEqual(summary["swept_keys"], ["137", "145", "153"])
        self.assertAlmostEqual(summary["spreads"]["quality_median"], 0.05)
        self.assertAlmostEqual(summary["spreads"]["diameter_max_mae_px"], 1.3)
        self.assertAlmostEqual(summary["baseline_max_delta"]["diameter_max_mae_px"], 1.3)
        self.assertAlmostEqual(summary["baseline_max_delta"]["af95_c"], 1.2)
        self.assertAlmostEqual(summary["baseline_max_delta"]["aftan_c"], 1.1)
        self.assertAlmostEqual(summary["max_abs"]["af95_c"], 59.2)

    def test_ignores_non_numeric_and_nonfinite_values(self) -> None:
        summary = summarize_numeric_sweep(
            {
                "base": {
                    "quality_median": 0.9,
                    "note": "baseline",
                    "af95_c": 58.0,
                },
                "alt": {
                    "quality_median": float("nan"),
                    "note": "alt",
                    "af95_c": 57.5,
                },
            },
            baseline_key="base",
        )

        self.assertNotIn("note", summary["spreads"])
        self.assertAlmostEqual(summary["spreads"]["af95_c"], 0.5)
        self.assertAlmostEqual(summary["baseline_max_delta"]["af95_c"], 0.5)
        self.assertAlmostEqual(summary["spreads"]["quality_median"], 0.0)


if __name__ == "__main__":
    unittest.main()
