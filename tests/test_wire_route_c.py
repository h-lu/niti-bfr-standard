from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from niti_bfr.metrics import MetricEvaluation, RecoveryFit
from niti_bfr.pipeline import _formal_wire_metric_gate


def _route_c_report(
    *,
    dynamic_range: float = 0.07,
) -> MetricEvaluation:
    return MetricEvaluation(
        label="kappa_route_c",
        increasing=False,
        fit=RecoveryFit(x_m=-0.078, x_a=-0.012, t0=45.0, width=4.0),
        af95_c=60.5,
        aftan_c=57.5,
        fit_rmse=0.001,
        monotonic_violation_fraction=0.0,
        dynamic_range=dynamic_range,
    )


def _route_b_report() -> MetricEvaluation:
    return MetricEvaluation(
        label="kappa_fit",
        increasing=False,
        fit=RecoveryFit(x_m=-0.08, x_a=-0.01, t0=45.0, width=4.0),
        af95_c=60.0,
        aftan_c=57.0,
        fit_rmse=0.001,
        monotonic_violation_fraction=0.0,
        dynamic_range=0.07,
    )


class WireRouteCGateTests(unittest.TestCase):
    def _base_series(self) -> pd.DataFrame:
        n = 20
        base_kappa = np.linspace(0.08, 0.01, n)
        route_c = base_kappa + 0.001 * np.sin(np.linspace(0.0, 5.0, n))
        return pd.DataFrame(
            {
                "temperature_c": np.linspace(20.0, 80.0, n),
                "kappa_fit_px_inv": base_kappa,
                "kappa_fit_recovery": np.concatenate(
                    [np.linspace(0.0, 0.88, n - 5), np.linspace(0.92, 1.0, 5)]
                ),
                "kappa_route_c_px_inv": route_c,
                "kappa_route_c_recovery": np.concatenate(
                    [np.linspace(0.0, 0.9, n - 5), np.linspace(0.94, 1.0, 5)]
                ),
            }
        )

    def test_gate_accepts_clean_temporal_route(self) -> None:
        reports = {"kappa_fit": _route_b_report(), "kappa_route_c": _route_c_report()}
        allowed, reason = _formal_wire_metric_gate(self._base_series(), reports, "kappa_route_c")
        self.assertTrue(allowed)
        self.assertIsNone(reason)

    def test_gate_rejects_large_route_b_deviation(self) -> None:
        series = self._base_series()
        series["kappa_route_c_px_inv"] = np.linspace(0.18, 0.11, len(series))
        reports = {
            "kappa_fit": _route_b_report(),
            "kappa_route_c": _route_c_report(dynamic_range=0.07),
        }
        allowed, reason = _formal_wire_metric_gate(series, reports, "kappa_route_c")
        self.assertFalse(allowed)
        self.assertEqual(reason, "kappa_route_c_route_b_deviation_too_large")

    def test_gate_rejects_small_dynamic_range(self) -> None:
        reports = {"kappa_fit": _route_b_report(), "kappa_route_c": _route_c_report(dynamic_range=0.001)}
        allowed, reason = _formal_wire_metric_gate(self._base_series(), reports, "kappa_route_c")
        self.assertFalse(allowed)
        self.assertEqual(reason, "kappa_route_c_dynamic_range_too_small")


if __name__ == "__main__":
    unittest.main()
