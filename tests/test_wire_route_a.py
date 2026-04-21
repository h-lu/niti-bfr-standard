from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from niti_bfr.metrics import MetricEvaluation, RecoveryFit
from niti_bfr.pipeline import _formal_wire_metric_gate


def _route_a_report(
    *,
    dynamic_range: float = 35.0,
    monotonic_violation_fraction: float = 0.0,
) -> MetricEvaluation:
    return MetricEvaluation(
        label="x_route_a",
        increasing=True,
        fit=RecoveryFit(x_m=55.0, x_a=90.0, t0=45.0, width=4.0),
        af95_c=61.0,
        aftan_c=58.0,
        fit_rmse=0.8,
        monotonic_violation_fraction=monotonic_violation_fraction,
        dynamic_range=dynamic_range,
    )


class WireRouteAGateTests(unittest.TestCase):
    def _base_series(self) -> pd.DataFrame:
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
            }
        )

    def test_gate_accepts_clean_route_a_series(self) -> None:
        allowed, reason = _formal_wire_metric_gate(self._base_series(), {"x_route_a": _route_a_report()}, "x_route_a")
        self.assertTrue(allowed)
        self.assertIsNone(reason)

    def test_gate_rejects_endpoint_instability(self) -> None:
        series = self._base_series()
        series.loc[10:, "route_a_tip_x"] += 18.0

        allowed, reason = _formal_wire_metric_gate(series, {"x_route_a": _route_a_report()}, "x_route_a")
        self.assertFalse(allowed)
        self.assertEqual(reason, "x_route_a_endpoint_instability")

    def test_gate_rejects_small_dynamic_range(self) -> None:
        allowed, reason = _formal_wire_metric_gate(
            self._base_series(),
            {"x_route_a": _route_a_report(dynamic_range=3.0)},
            "x_route_a",
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "x_route_a_dynamic_range_too_small")


if __name__ == "__main__":
    unittest.main()
