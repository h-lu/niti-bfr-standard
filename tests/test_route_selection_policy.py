from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from niti_bfr.metrics import MetricEvaluation, RecoveryFit
from niti_bfr.pipeline import _select_braided_formal_metric, _select_braided_primary_metric, _select_wire_provisional_metric


def _metric_eval(label: str, *, increasing: bool, fit_rmse: float, dynamic_range: float) -> MetricEvaluation:
    return MetricEvaluation(
        label=label,
        increasing=increasing,
        fit=RecoveryFit(x_m=0.0, x_a=1.0, t0=45.0, width=4.0),
        af95_c=60.0,
        aftan_c=57.0,
        fit_rmse=fit_rmse,
        monotonic_violation_fraction=0.0,
        dynamic_range=dynamic_range,
    )


class RouteSelectionPolicyTests(unittest.TestCase):
    def _wire_series(self) -> pd.DataFrame:
        n = 20
        return pd.DataFrame(
            {
                "frame": np.arange(n),
                "temperature_c": np.linspace(20.0, 80.0, n),
                "x_route_a_px": np.linspace(55.0, 90.0, n),
                "x_route_a_recovery": np.concatenate(
                    [np.linspace(0.0, 0.88, n - 5), np.linspace(0.92, 1.0, 5)]
                ),
                "route_a_anchor_x": np.full(n, 10.0),
                "route_a_anchor_y": np.full(n, 15.0),
                "route_a_tip_x": np.linspace(65.0, 100.0, n),
                "route_a_tip_y": np.full(n, 15.0),
                "kappa_fit_px_inv": np.linspace(0.08, 0.01, n),
                "kappa_fit_recovery": np.linspace(0.0, 0.82, n),
                "kappa_route_c_px_inv": np.linspace(0.081, 0.011, n),
                "kappa_route_c_recovery": np.concatenate(
                    [np.linspace(0.0, 0.90, n - 5), np.linspace(0.94, 1.0, 5)]
                ),
                "quality": np.full(n, 0.9),
            }
        )

    def test_wire_selector_prefers_route_b_as_reported_metric_even_when_formal_b_is_blocked(self) -> None:
        series = self._wire_series()
        reports = {
            "x_route_a": _metric_eval("x_route_a", increasing=True, fit_rmse=0.6, dynamic_range=35.0),
            "kappa_fit": _metric_eval("kappa_fit", increasing=False, fit_rmse=0.01, dynamic_range=0.07),
            "kappa_route_c": _metric_eval("kappa_route_c", increasing=False, fit_rmse=0.005, dynamic_range=0.07),
        }

        selected = _select_wire_provisional_metric(series, reports)
        self.assertEqual(selected, "kappa_fit")

    def test_wire_selector_prefers_route_b_when_all_routes_pass(self) -> None:
        series = self._wire_series()
        series["kappa_fit_recovery"] = np.concatenate(
            [np.linspace(0.0, 0.90, len(series) - 5), np.linspace(0.94, 1.0, 5)]
        )
        reports = {
            "x_route_a": _metric_eval("x_route_a", increasing=True, fit_rmse=0.6, dynamic_range=35.0),
            "kappa_fit": _metric_eval("kappa_fit", increasing=False, fit_rmse=0.01, dynamic_range=0.07),
            "kappa_route_c": _metric_eval("kappa_route_c", increasing=False, fit_rmse=0.005, dynamic_range=0.07),
        }

        selected = _select_wire_provisional_metric(series, reports)
        self.assertEqual(selected, "kappa_fit")

    def test_wire_selector_can_fall_back_to_route_a_when_b_and_c_are_unavailable(self) -> None:
        series = self._wire_series()
        reports = {
            "x_route_a": _metric_eval("x_route_a", increasing=True, fit_rmse=0.6, dynamic_range=35.0),
        }

        selected = _select_wire_provisional_metric(series, reports)
        self.assertEqual(selected, "x_route_a")

    def _braided_series(self) -> pd.DataFrame:
        n = 20
        recovery = np.concatenate(
            [np.linspace(0.0, 0.88, n - 5), np.linspace(0.92, 1.0, 5)]
        )
        return pd.DataFrame(
            {
                "frame": np.arange(n),
                "temperature_c": np.linspace(20.0, 80.0, n),
                "quality": np.full(n, 0.9),
                "length_axis_formal_px": np.linspace(120.0, 90.0, n),
                "length_axis_recovery": recovery,
                "diameter_max_px": np.linspace(20.0, 40.0, n),
                "diameter_max_recovery": recovery,
                "area_proj_formal_px2": np.linspace(500.0, 720.0, n),
                "area_proj_recovery": recovery,
                "centerline_disagreement": np.full(n, 0.01),
                "endpoint_gap_alt_centerline_px": np.full(n, 2.0),
                "endpoint_jump_px": np.full(n, 2.0),
                "endpoint_frame_jump_px": np.full(n, 2.0),
                "axis_peak_position_stability": np.full(n, 0.02),
                "body_mask_attachment_leak_fraction": np.zeros(n),
                "excluded_attachment_area_px2": np.zeros(n),
                "component_area_px2": np.full(n, 1000.0),
                "body_mask_area_px2": np.full(n, 800.0),
                "area_proj_definition_gap_px2": np.full(n, 10.0),
            }
        )

    def test_braided_primary_selector_can_choose_diameter_candidate(self) -> None:
        series = self._braided_series()
        reports = {
            "length_axis": _metric_eval("length_axis", increasing=False, fit_rmse=0.3, dynamic_range=30.0),
            "diameter_max": _metric_eval("diameter_max", increasing=True, fit_rmse=0.05, dynamic_range=20.0),
        }

        selected = _select_braided_primary_metric(series, reports)
        self.assertEqual(selected, "diameter_max")

    def test_braided_formal_selector_returns_none_and_reason_when_all_candidates_blocked(self) -> None:
        series = self._braided_series()
        series["centerline_disagreement"] = 0.12
        reports = {
            "length_axis": _metric_eval("length_axis", increasing=False, fit_rmse=0.1, dynamic_range=30.0),
            "diameter_max": _metric_eval("diameter_max", increasing=True, fit_rmse=0.05, dynamic_range=20.0),
        }

        selected, reason = _select_braided_formal_metric(series, reports)
        self.assertIsNone(selected)
        self.assertEqual(reason, "centerline_disagreement")

    def test_braided_formal_selector_can_promote_diameter_candidate(self) -> None:
        series = self._braided_series()
        reports = {
            "length_axis": _metric_eval("length_axis", increasing=False, fit_rmse=3.0, dynamic_range=30.0),
            "diameter_max": _metric_eval("diameter_max", increasing=True, fit_rmse=0.05, dynamic_range=20.0),
            "area_proj": _metric_eval("area_proj", increasing=True, fit_rmse=0.08, dynamic_range=220.0),
        }

        selected, reason = _select_braided_formal_metric(series, reports)
        self.assertEqual(selected, "diameter_max")
        self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()
