from __future__ import annotations

import unittest
from typing import Any

import numpy as np
import pandas as pd

from niti_bfr.metrics import MetricEvaluation, RecoveryFit
from niti_bfr.pipeline import AnalysisResult, _build_braided_route_results, _build_wire_route_results
from niti_bfr.webapp import _build_summary


class RouteLevelResultsSummaryTests(unittest.TestCase):
    def _metric_report(
        self,
        label: str,
        *,
        increasing: bool,
        af95_c: float,
        aftan_c: float,
        dynamic_range: float,
        fit_rmse: float = 0.25,
        monotonic_violation_fraction: float = 0.0,
    ) -> MetricEvaluation:
        if increasing:
            fit = RecoveryFit(x_m=10.0, x_a=25.0, t0=45.0, width=4.0)
        else:
            fit = RecoveryFit(x_m=-25.0, x_a=-10.0, t0=45.0, width=4.0)
        return MetricEvaluation(
            label=label,
            increasing=increasing,
            fit=fit,
            af95_c=af95_c,
            aftan_c=aftan_c,
            fit_rmse=fit_rmse,
            monotonic_violation_fraction=monotonic_violation_fraction,
            dynamic_range=dynamic_range,
        )

    def _build_result(
        self,
        *,
        series: pd.DataFrame,
        metric_reports: dict[str, MetricEvaluation],
        primary_metric_label: str | None,
        mode: str,
        formal_metric_label: str | None,
        provisional_metric_label: str | None,
        route_results: Any,
    ) -> AnalysisResult:
        result = AnalysisResult(
            series=series,
            fit=None,
            af95_c=None,
            aftan_c=None,
            metric_reports=metric_reports,
            primary_metric_label=primary_metric_label,
            mode=mode,
            formal_metric_label=formal_metric_label,
            formal_gate_reason=None,
            provisional_metric_label=provisional_metric_label,
            provisional_af95_c=61.0 if provisional_metric_label else None,
            provisional_aftan_c=58.0 if provisional_metric_label else None,
            reportability_status="reportable_with_warning" if provisional_metric_label else "quicklook_only",
            warning_codes=[],
            acceptance_profile="real_video",
        )
        setattr(result, "route_results", route_results)
        return result

    def _build_wire_series(self, n: int = 20) -> pd.DataFrame:
        plateau_tail = np.concatenate(
            [
                np.linspace(0.0, 0.88, n - 5, endpoint=True),
                np.linspace(0.92, 1.0, 5, endpoint=True),
            ]
        )
        return pd.DataFrame(
            {
                "frame": np.arange(n),
                "quality": np.full(n, 0.82),
                "temperature_c": np.linspace(20.0, 80.0, n),
                "x_route_a_px": np.linspace(5.0, 20.0, n),
                "x_route_a_recovery": plateau_tail,
                "x_fit_px": np.linspace(6.0, 21.0, n),
                "x_route_c_px": np.linspace(6.1, 20.8, n),
                "kappa_fit_px_inv": np.linspace(0.10, 0.02, n),
                "kappa_fit_recovery": plateau_tail,
                "kappa_route_c_px_inv": np.linspace(0.095, 0.025, n),
                "kappa_route_c_recovery": plateau_tail,
            }
        )

    def _extract_route_entries(self, summary: dict[str, Any]) -> list[dict[str, Any]]:
        self.assertIn("route_results", summary)
        route_results = summary["route_results"]
        self.assertIsInstance(route_results, (dict, list))

        if isinstance(route_results, dict):
            raw_entries = []
            for key, value in route_results.items():
                entry = dict(value or {})
                entry.setdefault("_container_key", key)
                raw_entries.append(entry)
        else:
            raw_entries = [dict(entry) for entry in route_results]

        entries: list[dict[str, Any]] = []
        for entry in raw_entries:
            container_key = entry.get("_container_key")
            alias = entry.get("alias") or entry.get("route_alias")
            metric_key = entry.get("metric_key")
            if alias is None and container_key in {"A", "B", "C"}:
                alias = container_key
            if metric_key is None and isinstance(container_key, str) and container_key not in {"A", "B", "C"}:
                metric_key = container_key
            entries.append(
                {
                    **entry,
                    "alias": alias,
                    "metric_key": metric_key,
                }
            )
        return entries

    def _find_route_entry(
        self,
        route_results: list[dict[str, Any]],
        *,
        alias: str | None = None,
        metric_key: str | None = None,
    ) -> dict[str, Any]:
        for entry in route_results:
            if alias is not None and entry.get("alias") == alias:
                return entry
            if metric_key is not None and entry.get("metric_key") == metric_key:
                return entry
        details = f"alias={alias!r}, metric_key={metric_key!r}"
        raise AssertionError(f"missing route entry for {details}")

    def test_braided_summary_exposes_three_routes_and_keeps_area_proj_out_of_formal_candidates(self) -> None:
        n = 12
        series = pd.DataFrame(
            {
                "frame": np.arange(n),
                "quality": np.full(n, 0.85),
                "temperature_c": np.linspace(20.0, 80.0, n),
                "length_axis_px": np.linspace(120.0, 90.0, n),
                "diameter_max_px": np.linspace(20.0, 36.0, n),
                "area_proj_px2": np.linspace(500.0, 720.0, n),
                "centerline_disagreement": np.full(n, 0.01),
                "endpoint_gap_alt_centerline_px": np.full(n, 2.0),
                "endpoint_jump_px": np.full(n, 2.0),
                "endpoint_frame_jump_px": np.full(n, 2.0),
                "body_mask_attachment_leak_fraction": np.zeros(n),
            }
        )
        metric_reports = {
            "length_axis": self._metric_report(
                "length_axis",
                increasing=False,
                af95_c=60.0,
                aftan_c=58.0,
                dynamic_range=30.0,
            ),
            "diameter_max": self._metric_report(
                "diameter_max",
                increasing=True,
                af95_c=59.0,
                aftan_c=57.0,
                dynamic_range=16.0,
            ),
            "area_proj": self._metric_report(
                "area_proj",
                increasing=True,
                af95_c=58.5,
                aftan_c=56.5,
                dynamic_range=220.0,
            ),
        }
        result = self._build_result(
            series=series,
            metric_reports=metric_reports,
            primary_metric_label="length_axis",
            mode="quicklook",
            formal_metric_label=None,
            provisional_metric_label="length_axis",
            route_results={
                "A": {
                    "alias": "A",
                    "metric_key": "length_axis",
                    "display_label": "A:length_axis",
                    "af95_c": 60.0,
                    "aftan_c": 58.0,
                    "reportability_status": "provisional",
                    "accepted_as_formal_candidate": True,
                },
                "B": {
                    "alias": "B",
                    "metric_key": "diameter_max",
                    "display_label": "B:diameter_max",
                    "af95_c": 59.0,
                    "aftan_c": 57.0,
                    "reportability_status": "provisional",
                    "accepted_as_formal_candidate": True,
                },
                "C": {
                    "alias": "C",
                    "metric_key": "area_proj",
                    "display_label": "C:area_proj",
                    "af95_c": 58.5,
                    "aftan_c": 56.5,
                    "reportability_status": "provisional",
                    "accepted_as_formal_candidate": False,
                },
            },
        )
        summary = _build_summary(
            {
                "id": "braided-run",
                "preset": "braided_like",
                "requested_mode": "formal_af",
                "video_filename": "braided.mp4",
                "temperature_filename": "braided.csv",
            },
            result,
        )

        entries = self._extract_route_entries(summary)
        aliases = {entry["alias"] for entry in entries if entry.get("alias")}
        metric_keys = {entry["metric_key"] for entry in entries if entry.get("metric_key")}

        self.assertGreaterEqual(len(entries), 3)
        self.assertTrue({"A", "B", "C"}.issubset(aliases) or {"length_axis", "diameter_max", "area_proj"}.issubset(metric_keys))

        area_entries = [entry for entry in entries if entry.get("alias") == "C" or entry.get("metric_key") == "area_proj"]
        self.assertTrue(area_entries)

        candidate_labels = {
            item.get("label") or item.get("metric_key")
            for item in summary["formal_candidate_metrics"]
        }
        self.assertIn("length_axis", candidate_labels)
        self.assertIn("diameter_max", candidate_labels)
        self.assertNotIn("area_proj", candidate_labels)

    def test_wire_like_summary_exposes_route_level_results_minimal_schema(self) -> None:
        n = 10
        series = pd.DataFrame(
            {
                "frame": np.arange(n),
                "quality": np.full(n, 0.82),
                "temperature_c": np.linspace(20.0, 70.0, n),
                "x_route_a_px": np.linspace(5.0, 15.0, n),
                "x_fit_px": np.linspace(6.0, 16.0, n),
                "x_route_c_px": np.linspace(6.2, 15.8, n),
                "kappa_fit_px_inv": np.linspace(0.08, 0.02, n),
                "kappa_route_c_px_inv": np.linspace(0.075, 0.025, n),
            }
        )
        metric_reports = {
            "x_route_a": self._metric_report(
                "x_route_a",
                increasing=True,
                af95_c=59.0,
                aftan_c=57.0,
                dynamic_range=10.0,
            ),
            "kappa_fit": self._metric_report(
                "kappa_fit",
                increasing=False,
                af95_c=58.0,
                aftan_c=56.0,
                dynamic_range=0.06,
            ),
            "kappa_route_c": self._metric_report(
                "kappa_route_c",
                increasing=False,
                af95_c=57.5,
                aftan_c=55.5,
                dynamic_range=0.05,
            ),
        }
        result = self._build_result(
            series=series,
            metric_reports=metric_reports,
            primary_metric_label="kappa_fit",
            mode="quicklook",
            formal_metric_label=None,
            provisional_metric_label=None,
            route_results=[
                {
                    "route_alias": "A",
                    "metric_key": "x_route_a",
                    "display_label": "A:x_route_a",
                    "af95_c": 59.0,
                    "aftan_c": 57.0,
                    "reportability_status": "quicklook_only",
                },
                {
                    "route_alias": "B",
                    "metric_key": "kappa_fit",
                    "display_label": "B:kappa_fit",
                    "af95_c": 58.0,
                    "aftan_c": 56.0,
                    "reportability_status": "quicklook_only",
                },
                {
                    "route_alias": "C",
                    "metric_key": "kappa_route_c",
                    "display_label": "C:kappa_route_c",
                    "af95_c": 57.5,
                    "aftan_c": 55.5,
                    "reportability_status": "quicklook_only",
                },
            ],
        )
        summary = _build_summary(
            {
                "id": "wire-run",
                "preset": "wire_like",
                "requested_mode": "quicklook",
                "video_filename": "wire.mp4",
                "temperature_filename": "wire.csv",
            },
            result,
        )

        entries = self._extract_route_entries(summary)
        self.assertGreaterEqual(len(entries), 3)

        route_metric_families = {
            "A": {"x_route_a"},
            "B": {"x_fit", "kappa_fit"},
            "C": {"x_route_c", "kappa_route_c"},
        }
        for route_alias, metric_family in route_metric_families.items():
            matches = [
                entry
                for entry in entries
                if entry.get("alias") == route_alias or entry.get("metric_key") in metric_family
            ]
            self.assertTrue(matches, msg=f"missing route-level result for {route_alias}")
            match = matches[0]
            self.assertIn("metric_key", match)
            self.assertIn("af95_c", match)
            self.assertIn("aftan_c", match)
            self.assertIn("reportability_status", match)

    def test_braided_route_results_keep_route_level_passes_separate_from_object_level_formal_selection(self) -> None:
        n = 20
        plateau_tail = np.concatenate(
            [
                np.linspace(0.0, 0.88, n - 5, endpoint=True),
                np.linspace(0.92, 1.0, 5, endpoint=True),
            ]
        )
        series = pd.DataFrame(
            {
                "frame": np.arange(n),
                "quality": np.full(n, 0.9),
                "temperature_c": np.linspace(20.0, 80.0, n),
                "length_axis_formal_px": np.linspace(120.0, 90.0, n),
                "length_axis_recovery": plateau_tail,
                "diameter_max_px": np.linspace(20.0, 40.0, n),
                "diameter_max_recovery": plateau_tail,
                "area_proj_formal_px2": np.linspace(500.0, 720.0, n),
                "area_proj_recovery": plateau_tail,
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
        metric_reports = {
            "length_axis": self._metric_report(
                "length_axis",
                increasing=False,
                af95_c=60.0,
                aftan_c=58.0,
                dynamic_range=30.0,
                fit_rmse=3.0,
            ),
            "diameter_max": self._metric_report(
                "diameter_max",
                increasing=True,
                af95_c=59.0,
                aftan_c=57.0,
                dynamic_range=20.0,
                fit_rmse=0.2,
            ),
            "area_proj": self._metric_report(
                "area_proj",
                increasing=True,
                af95_c=58.5,
                aftan_c=56.5,
                dynamic_range=220.0,
            ),
        }

        route_results = _build_braided_route_results(
            series,
            metric_reports,
            primary_metric_label="length_axis",
            formal_metric_label="diameter_max",
            acceptance_profile="real_video",
            temperature_available=True,
        )

        a_entry = self._find_route_entry(route_results, alias="A")
        b_entry = self._find_route_entry(route_results, alias="B")
        c_entry = self._find_route_entry(route_results, alias="C")

        self.assertEqual(a_entry["reportability_status"], "formal_passed")
        self.assertTrue(a_entry["accepted_as_formal_candidate"])
        self.assertTrue(a_entry["selected_as_primary"])
        self.assertFalse(a_entry["selected_as_formal"])

        self.assertEqual(b_entry["reportability_status"], "formal_passed")
        self.assertTrue(b_entry["formal_candidate"])
        self.assertTrue(b_entry["accepted_as_formal_candidate"])
        self.assertFalse(b_entry["selected_as_primary"])
        self.assertTrue(b_entry["selected_as_formal"])

        self.assertEqual(c_entry["metric_key"], "area_proj")
        self.assertEqual(c_entry["reportability_status"], "provisional")
        self.assertFalse(c_entry["formal_candidate"])
        self.assertFalse(c_entry["accepted_as_formal_candidate"])

    def test_wire_like_summary_keeps_b_route_formal_when_kappa_fit_is_object_level_formal(self) -> None:
        series = self._build_wire_series()
        metric_reports = {
            "x_route_a": self._metric_report(
                "x_route_a",
                increasing=True,
                af95_c=59.0,
                aftan_c=57.0,
                dynamic_range=15.0,
            ),
            "kappa_fit": self._metric_report(
                "kappa_fit",
                increasing=False,
                af95_c=58.0,
                aftan_c=56.0,
                dynamic_range=0.08,
            ),
            "kappa_route_c": self._metric_report(
                "kappa_route_c",
                increasing=False,
                af95_c=57.5,
                aftan_c=55.5,
                dynamic_range=0.07,
            ),
        }
        route_results = _build_wire_route_results(
            series,
            metric_reports,
            primary_metric_label="kappa_fit",
            formal_metric_label="kappa_fit",
            temperature_available=True,
        )
        result = AnalysisResult(
            series=series,
            fit=None,
            af95_c=58.0,
            aftan_c=56.0,
            metric_reports=metric_reports,
            primary_metric_label="kappa_fit",
            mode="formal_af",
            formal_metric_label="kappa_fit",
            formal_gate_reason=None,
            reportability_status="formal",
            warning_codes=[],
            acceptance_profile="real_video",
            route_results=route_results,
        )
        summary = _build_summary(
            {
                "id": "wire-formal",
                "preset": "wire_like",
                "requested_mode": "formal_af",
                "video_filename": "wire.mp4",
                "temperature_filename": "wire.csv",
            },
            result,
        )

        entries = self._extract_route_entries(summary)
        b_entry = self._find_route_entry(entries, alias="B")

        self.assertEqual(summary["formal_metric_label"], "kappa_fit")
        self.assertEqual(b_entry["metric_key"], "kappa_fit")
        self.assertEqual(b_entry["reportability_status"], "formal_passed")
        self.assertNotEqual(b_entry["reportability_status"], "formal_blocked")
        self.assertIsNone(b_entry["gate_reason"])
        self.assertTrue(b_entry["accepted_as_formal_candidate"])
        self.assertIn("selected_as_primary", b_entry)
        self.assertIn("selected_as_formal", b_entry)
        self.assertTrue(b_entry["selected_as_primary"])
        self.assertTrue(b_entry["selected_as_formal"])

    def test_wire_like_route_results_include_status_and_gate_reason_for_a_and_c_with_temperature(self) -> None:
        series = self._build_wire_series()
        metric_reports = {
            "kappa_fit": self._metric_report(
                "kappa_fit",
                increasing=False,
                af95_c=58.0,
                aftan_c=56.0,
                dynamic_range=0.08,
            ),
        }
        route_results = _build_wire_route_results(
            series,
            metric_reports,
            primary_metric_label="kappa_fit",
            formal_metric_label="kappa_fit",
            temperature_available=True,
        )

        a_entry = self._find_route_entry(route_results, alias="A")
        c_entry = self._find_route_entry(route_results, alias="C")

        for entry, metric_key in ((a_entry, "x_route_a"), (c_entry, "kappa_route_c")):
            self.assertEqual(entry["metric_key"], metric_key)
            self.assertIn("reportability_status", entry)
            self.assertIn("gate_reason", entry)
            self.assertEqual(entry["reportability_status"], "formal_blocked")
            self.assertEqual(entry["gate_reason"], f"{metric_key}_insufficient_points")
            self.assertIn("warning_codes", entry)
            self.assertEqual(entry["warning_codes"], [f"{metric_key}_insufficient_points"])
            self.assertIn("selected_as_primary", entry)
            self.assertIn("selected_as_formal", entry)
            self.assertFalse(entry["selected_as_primary"])
            self.assertFalse(entry["selected_as_formal"])


if __name__ == "__main__":
    unittest.main()
