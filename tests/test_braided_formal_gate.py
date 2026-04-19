from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from niti_bfr.metrics import MetricEvaluation, RecoveryFit
from niti_bfr.pipeline import AnalysisResult, _formal_braided_af_gate
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

    def test_gate_rejects_small_body_fraction(self) -> None:
        series = self._base_series()
        series["body_mask_area_px2"] = 400.0
        allowed, reason = _formal_braided_af_gate(series, self._reports())
        self.assertFalse(allowed)
        self.assertEqual(reason, "body_mask_area_fraction")


class PublicSummaryTests(unittest.TestCase):
    def test_public_formal_metric_hidden_for_quicklook(self) -> None:
        result = AnalysisResult(
            series=pd.DataFrame({"frame": [0], "quality": [0.8]}),
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


if __name__ == "__main__":
    unittest.main()
