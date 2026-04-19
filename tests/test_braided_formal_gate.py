from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from niti_bfr.metrics import MetricEvaluation, RecoveryFit
from niti_bfr.pipeline import AnalysisResult, _formal_braided_af_gate, compute_braided_acceptance
from niti_bfr.webapp import _build_summary, _public_formal_metric_label


class BraidedFormalGateTests(unittest.TestCase):
    def _base_series(self) -> pd.DataFrame:
        n = 20
        recovery = np.concatenate(
            [
                np.linspace(0.0, 0.88, n - 5, endpoint=True),
                np.linspace(0.92, 1.0, 5, endpoint=True),
            ]
        )
        return pd.DataFrame(
            {
                "frame": np.arange(n),
                "temperature_c": np.linspace(20.0, 80.0, n),
                "length_axis_formal_px": np.linspace(120.0, 90.0, n),
                "length_axis_recovery": recovery,
                "quality": np.full(n, 0.9),
                "centerline_disagreement": np.full(n, 0.01),
                "endpoint_jump_px": np.full(n, 2.0),
                "branch_component_count_after_pruning": np.zeros(n),
                "axis_peak_position_stability": np.full(n, 0.02),
                "body_mask_attachment_leak_fraction": np.zeros(n),
                "excluded_attachment_area_px2": np.zeros(n),
                "component_area_px2": np.full(n, 1000.0),
                "body_mask_area_px2": np.full(n, 800.0),
                "attachment_border_touch_count": np.zeros(n),
            }
        )

    def _reports(self) -> dict[str, MetricEvaluation]:
        return {
            "length_axis": MetricEvaluation(
                label="length_axis",
                increasing=False,
                fit=RecoveryFit(x_m=-120.0, x_a=-90.0, t0=45.0, width=4.0),
                af95_c=60.0,
                aftan_c=58.0,
                fit_rmse=0.5,
                monotonic_violation_fraction=0.0,
                dynamic_range=30.0,
            )
        }

    def test_gate_accepts_clean_series(self) -> None:
        allowed, reason = _formal_braided_af_gate(self._base_series(), self._reports())
        self.assertTrue(allowed)
        self.assertIsNone(reason)

    def test_gate_rejects_attachment_leakage(self) -> None:
        series = self._base_series()
        series["body_mask_attachment_leak_fraction"] = 0.06
        allowed, reason = _formal_braided_af_gate(series, self._reports())
        self.assertFalse(allowed)
        self.assertEqual(reason, "body_mask_attachment_leak_fraction")

    def test_gate_tolerates_high_absolute_branch_complexity_when_other_qc_is_clean(self) -> None:
        series = self._base_series()
        series["branch_component_count_after_pruning"] = 42.0
        allowed, reason = _formal_braided_af_gate(series, self._reports())
        self.assertTrue(allowed)
        self.assertIsNone(reason)

    def test_gate_tolerates_border_touch_qc_when_leakage_is_clean(self) -> None:
        series = self._base_series()
        series["attachment_border_touch_count"] = 1.0
        allowed, reason = _formal_braided_af_gate(series, self._reports())
        self.assertTrue(allowed)
        self.assertIsNone(reason)

    def test_gate_rejects_small_body_fraction(self) -> None:
        series = self._base_series()
        series["body_mask_area_px2"] = 400.0
        allowed, reason = _formal_braided_af_gate(series, self._reports())
        self.assertFalse(allowed)
        self.assertEqual(reason, "body_mask_area_fraction")

    def test_gate_rejects_endpoint_fraction_instability_below_px_floor(self) -> None:
        series = self._base_series()
        series["length_axis_formal_px"] = 100.0
        series["endpoint_jump_px"] = 10.5
        allowed, reason = _formal_braided_af_gate(series, self._reports())
        self.assertFalse(allowed)
        self.assertEqual(reason, "endpoint_jump")

    def test_gate_tolerates_subpixel_axis_jitter(self) -> None:
        series = self._base_series()
        series["length_axis_formal_px"] = np.linspace(120.0, 90.0, len(series)) + 0.2 * np.sin(np.linspace(0.0, 8.0, len(series)))
        reports = self._reports()
        reports["length_axis"] = MetricEvaluation(
            label="length_axis",
            increasing=False,
            fit=reports["length_axis"].fit,
            af95_c=reports["length_axis"].af95_c,
            aftan_c=reports["length_axis"].aftan_c,
            fit_rmse=reports["length_axis"].fit_rmse,
            monotonic_violation_fraction=0.55,
            dynamic_range=reports["length_axis"].dynamic_range,
        )
        allowed, reason = _formal_braided_af_gate(series, reports)
        self.assertTrue(allowed)
        self.assertIsNone(reason)

    def test_acceptance_exposes_real_video_thresholds(self) -> None:
        acceptance = compute_braided_acceptance(self._base_series())
        self.assertTrue(acceptance["accepted"])
        self.assertEqual(acceptance["reasons"], [])
        self.assertEqual(acceptance["metrics"]["valid_frames"], 20)
        self.assertAlmostEqual(acceptance["metrics"]["endpoint_jump_limit_px"], 12.0)
        self.assertAlmostEqual(acceptance["thresholds"]["endpoint_jump_fraction_p95_max"], 0.08)
        self.assertAlmostEqual(acceptance["thresholds"]["axis_monotonic_violation_fraction_max"], 0.20)


class PublicSummaryTests(unittest.TestCase):
    def test_public_formal_metric_hidden_for_quicklook(self) -> None:
        series = pd.DataFrame(
            {
                "frame": np.arange(12),
                "quality": np.full(12, 0.8),
                "length_axis_px": np.full(12, 120.0),
                "centerline_disagreement": np.full(12, 0.01),
                "endpoint_jump_px": np.full(12, 2.0),
                "body_mask_attachment_leak_fraction": np.full(12, 0.06),
            }
        )
        result = AnalysisResult(
            series=series,
            fit=None,
            af95_c=None,
            aftan_c=None,
            metric_reports=None,
            primary_metric_label=None,
            mode="quicklook",
            formal_metric_label="length_axis",
            formal_gate_reason="body_mask_attachment_leak_fraction",
        )
        self.assertIsNone(_public_formal_metric_label(result))
        summary = _build_summary(
            {
                "id": "run-1",
                "preset": "braided_like",
                "requested_mode": "formal_af",
                "video_filename": "demo.mp4",
                "temperature_filename": "demo.csv",
            },
            result,
        )
        self.assertIsNone(summary["formal_metric_label"])
        self.assertEqual(summary["actual_mode"], "quicklook")
        self.assertEqual(summary["formal_gate_reason"], "body_mask_attachment_leak_fraction")
        self.assertFalse(summary["acceptance"]["accepted"])
        self.assertIn("body_mask_attachment_leak_fraction", summary["acceptance"]["reasons"])


if __name__ == "__main__":
    unittest.main()
