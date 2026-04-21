from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from niti_bfr.metrics import MetricEvaluation, RecoveryFit
from niti_bfr.pipeline import _formal_wire_metric_gate


def _route_b_report(*, monotonic_violation_fraction: float = 0.0) -> MetricEvaluation:
    return MetricEvaluation(
        label="kappa_fit",
        increasing=False,
        fit=RecoveryFit(x_m=-0.08, x_a=-0.01, t0=45.0, width=4.0),
        af95_c=60.0,
        aftan_c=57.0,
        fit_rmse=0.001,
        monotonic_violation_fraction=monotonic_violation_fraction,
        dynamic_range=0.07,
    )


class WireRouteBGateTests(unittest.TestCase):
    def _base_series(self) -> pd.DataFrame:
        n = 20
        return pd.DataFrame(
            {
                "temperature_c": np.linspace(20.0, 80.0, n),
                "kappa_fit_px_inv": np.linspace(0.08, 0.01, n),
                "kappa_fit_recovery": np.concatenate(
                    [np.linspace(0.0, 0.88, n - 5), np.linspace(0.92, 1.0, 5)]
                ),
                "quality": np.full(n, 0.9),
            }
        )

    def test_gate_accepts_clean_kappa_fit_series(self) -> None:
        allowed, reason = _formal_wire_metric_gate(self._base_series(), {"kappa_fit": _route_b_report()}, "kappa_fit")
        self.assertTrue(allowed)
        self.assertIsNone(reason)

    def test_gate_rejects_when_kappa_report_missing(self) -> None:
        allowed, reason = _formal_wire_metric_gate(self._base_series(), {}, "kappa_fit")
        self.assertFalse(allowed)
        self.assertEqual(reason, "insufficient_kappa_points")

    def test_gate_rejects_non_monotonic_kappa(self) -> None:
        series = self._base_series()
        series["kappa_fit_px_inv"] = np.concatenate(
            [np.linspace(0.08, 0.03, 10), np.linspace(0.031, 0.05, 10)]
        )
        allowed, reason = _formal_wire_metric_gate(
            series,
            {"kappa_fit": _route_b_report(monotonic_violation_fraction=0.4)},
            "kappa_fit",
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "kappa_fit_not_monotonic_enough")

    def test_gate_accepts_small_raw_oscillation_when_smoothed_trend_is_monotonic(self) -> None:
        series = self._base_series()
        series["kappa_fit_px_inv"] = np.linspace(0.08, 0.01, len(series)) + 0.002 * np.where(
            np.arange(len(series)) % 2 == 0, 1.0, -1.0
        )

        allowed, reason = _formal_wire_metric_gate(
            series,
            {"kappa_fit": _route_b_report(monotonic_violation_fraction=0.4)},
            "kappa_fit",
        )
        self.assertTrue(allowed)
        self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()
