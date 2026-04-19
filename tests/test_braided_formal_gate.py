from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from niti_bfr.metrics import MetricEvaluation, RecoveryFit
from niti_bfr.pipeline import AnalysisResult, _formal_braided_af_gate, _select_braided_formal_metric, compute_braided_acceptance
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
                "endpoint_gap_alt_centerline_px": np.full(n, 2.0),
                "endpoint_jump_px": np.full(n, 2.0),
                "endpoint_frame_jump_px": np.full(n, 2.0),
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

    def _diameter_report(self, *, fit_rmse: float = 0.25, monotonic_violation_fraction: float = 0.0) -> MetricEvaluation:
        return MetricEvaluation(
            label="diameter_max",
            increasing=True,
            fit=RecoveryFit(x_m=20.0, x_a=40.0, t0=45.0, width=4.0),
            af95_c=59.0,
            aftan_c=57.0,
            fit_rmse=fit_rmse,
            monotonic_violation_fraction=monotonic_violation_fraction,
            dynamic_range=20.0,
        )

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
        series["endpoint_gap_alt_centerline_px"] = 10.5
        series["endpoint_jump_px"] = 10.5
        allowed, reason = _formal_braided_af_gate(series, self._reports())
        self.assertFalse(allowed)
        self.assertEqual(reason, "endpoint_jump")

    def test_real_video_gate_rejects_large_frame_jump_but_synthetic_profile_can_tolerate_it(self) -> None:
        series = self._base_series()
        series["endpoint_frame_jump_px"] = 10.5
        allowed_real, reason_real = _formal_braided_af_gate(series, self._reports(), acceptance_profile="real_video")
        allowed_synth, reason_synth = _formal_braided_af_gate(series, self._reports(), acceptance_profile="synthetic")
        self.assertFalse(allowed_real)
        self.assertEqual(reason_real, "endpoint_frame_jump")
        self.assertTrue(allowed_synth)
        self.assertIsNone(reason_synth)

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

    def test_gate_can_validate_diameter_metric(self) -> None:
        series = self._base_series()
        series["diameter_max_px"] = np.linspace(20.0, 40.0, len(series))
        series["diameter_max_recovery"] = series["length_axis_recovery"]
        reports = self._reports()
        reports["diameter_max"] = self._diameter_report()

        allowed, reason = _formal_braided_af_gate(series, reports, metric_label="diameter_max")
        self.assertTrue(allowed)
        self.assertIsNone(reason)

    def test_select_braided_formal_metric_can_promote_diameter_when_it_scores_better(self) -> None:
        series = self._base_series()
        series["diameter_max_px"] = np.linspace(20.0, 40.0, len(series))
        series["diameter_max_recovery"] = series["length_axis_recovery"]
        reports = self._reports()
        reports["length_axis"] = MetricEvaluation(
            label="length_axis",
            increasing=False,
            fit=reports["length_axis"].fit,
            af95_c=reports["length_axis"].af95_c,
            aftan_c=reports["length_axis"].aftan_c,
            fit_rmse=3.0,
            monotonic_violation_fraction=reports["length_axis"].monotonic_violation_fraction,
            dynamic_range=reports["length_axis"].dynamic_range,
        )
        reports["diameter_max"] = self._diameter_report(fit_rmse=0.2)

        metric_label, gate_reason = _select_braided_formal_metric(series, reports)
        self.assertEqual(metric_label, "diameter_max")
        self.assertIsNone(gate_reason)

    def test_acceptance_exposes_real_video_thresholds(self) -> None:
        acceptance = compute_braided_acceptance(self._base_series())
        self.assertTrue(acceptance["accepted"])
        self.assertEqual(acceptance["reasons"], [])
        self.assertEqual(acceptance["metrics"]["valid_frames"], 20)
        self.assertAlmostEqual(acceptance["metrics"]["endpoint_jump_limit_px"], 12.0)
        self.assertAlmostEqual(acceptance["metrics"]["endpoint_frame_jump_limit_px"], 12.0)
        self.assertAlmostEqual(acceptance["thresholds"]["endpoint_gap_alt_centerline_fraction_p95_max"], 0.08)
        self.assertAlmostEqual(acceptance["thresholds"]["axis_monotonic_violation_fraction_max"], 0.20)


class PublicSummaryTests(unittest.TestCase):
    def test_public_formal_metric_hidden_for_quicklook(self) -> None:
        series = pd.DataFrame(
            {
                "frame": np.arange(12),
                "quality": np.full(12, 0.8),
                "length_axis_px": np.full(12, 120.0),
                "centerline_disagreement": np.full(12, 0.01),
                "endpoint_gap_alt_centerline_px": np.full(12, 2.0),
                "endpoint_jump_px": np.full(12, 2.0),
                "endpoint_frame_jump_px": np.full(12, 14.0),
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
            provisional_metric_label="length_axis",
            provisional_af95_c=61.0,
            provisional_aftan_c=58.0,
            reportability_status="reportable_with_warning",
            warning_codes=["body_mask_attachment_leak_fraction"],
            acceptance_profile="real_video",
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
        self.assertEqual(summary["reportability_status"], "reportable_with_warning")
        self.assertEqual(summary["provisional_metric_label"], "length_axis")
        self.assertEqual(summary["provisional_af95_c"], 61.0)
        self.assertEqual(summary["formal_gate_reason"], "body_mask_attachment_leak_fraction")
        self.assertEqual(summary["formal_qc_scope"], "current braided formal-gate QC snapshot; not an overall A/B/C verdict")
        self.assertEqual(summary["formal_candidate_metrics"][0]["label"], "length_axis")
        self.assertEqual(summary["formal_candidate_metrics"][1]["label"], "diameter_max")
        self.assertFalse(summary["formal_qc"]["accepted"])
        self.assertIn("body_mask_attachment_leak_fraction", summary["formal_qc"]["reasons"])


if __name__ == "__main__":
    unittest.main()
