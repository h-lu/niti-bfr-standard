from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from niti_bfr.metrics import MetricEvaluation, RecoveryFit
from niti_bfr.export_contract import flatten_route_results
from niti_bfr.pipeline import _build_braided_route_results, _build_wire_route_results


def _metric_eval(label: str, *, increasing: bool, af95_c: float, aftan_c: float, dynamic_range: float) -> MetricEvaluation:
    return MetricEvaluation(
        label=label,
        increasing=increasing,
        fit=RecoveryFit(x_m=0.0, x_a=1.0, t0=45.0, width=4.0),
        af95_c=af95_c,
        aftan_c=aftan_c,
        fit_rmse=0.05,
        monotonic_violation_fraction=0.0,
        dynamic_range=dynamic_range,
    )


class RouteResultSchemaTests(unittest.TestCase):
    def test_wire_routes_always_expose_abc_schema(self) -> None:
        series = pd.DataFrame(
            {
                "temperature_c": np.linspace(20.0, 80.0, 20),
                "x_route_a_px": np.linspace(50.0, 90.0, 20),
                "x_route_a_recovery": np.linspace(0.0, 1.0, 20),
                "kappa_fit_px_inv": np.linspace(0.08, 0.01, 20),
                "kappa_fit_recovery": np.linspace(0.0, 1.0, 20),
                "kappa_route_c_px_inv": np.linspace(0.08, 0.01, 20),
                "kappa_route_c_recovery": np.linspace(0.0, 1.0, 20),
                "route_a_anchor_x": np.full(20, 1.0),
                "route_a_anchor_y": np.full(20, 2.0),
                "route_a_tip_x": np.linspace(10.0, 40.0, 20),
                "route_a_tip_y": np.full(20, 2.0),
                "quality": np.full(20, 0.9),
            }
        )
        reports = {
            "x_route_a": _metric_eval("x_route_a", increasing=True, af95_c=60.0, aftan_c=58.0, dynamic_range=40.0),
            "kappa_fit": _metric_eval("kappa_fit", increasing=False, af95_c=59.0, aftan_c=57.0, dynamic_range=0.07),
            "kappa_route_c": _metric_eval(
                "kappa_route_c",
                increasing=False,
                af95_c=59.5,
                aftan_c=57.5,
                dynamic_range=0.07,
            ),
        }

        route_results = _build_wire_route_results(
            series,
            reports,
            primary_metric_label="kappa_fit",
            formal_metric_label="kappa_fit",
            temperature_available=True,
        )

        self.assertEqual([entry["alias"] for entry in route_results], ["A", "B", "C"])
        for entry in route_results:
            self.assertTrue(
                {
                    "metric_key",
                    "display_label",
                    "af95_c",
                    "aftan_c",
                    "reportability_status",
                    "gate_reason",
                    "warning_codes",
                    "accepted_as_formal_candidate",
                }.issubset(entry)
            )

    def test_wire_routes_keep_fixed_alias_mapping_and_formal_selection_flags(self) -> None:
        series = pd.DataFrame(
            {
                "frame": np.arange(20),
                "temperature_c": np.linspace(20.0, 80.0, 20),
                "x_route_a_px": np.linspace(50.0, 90.0, 20),
                "x_route_a_recovery": np.linspace(0.0, 1.0, 20),
                "kappa_fit_px_inv": np.linspace(0.08, 0.01, 20),
                "kappa_fit_recovery": np.linspace(0.0, 1.0, 20),
                "kappa_route_c_px_inv": np.linspace(0.081, 0.011, 20),
                "kappa_route_c_recovery": np.linspace(0.0, 1.0, 20),
                "route_a_anchor_x": np.full(20, 1.0),
                "route_a_anchor_y": np.full(20, 2.0),
                "route_a_tip_x": np.linspace(10.0, 40.0, 20),
                "route_a_tip_y": np.full(20, 2.0),
                "quality": np.full(20, 0.9),
            }
        )
        reports = {
            "x_route_a": _metric_eval("x_route_a", increasing=True, af95_c=60.0, aftan_c=58.0, dynamic_range=40.0),
            "kappa_fit": _metric_eval("kappa_fit", increasing=False, af95_c=59.0, aftan_c=57.0, dynamic_range=0.07),
            "kappa_route_c": _metric_eval(
                "kappa_route_c",
                increasing=False,
                af95_c=59.5,
                aftan_c=57.5,
                dynamic_range=0.07,
            ),
        }

        route_results = _build_wire_route_results(
            series,
            reports,
            primary_metric_label="kappa_fit",
            formal_metric_label="kappa_fit",
            temperature_available=True,
        )

        self.assertEqual(
            [(entry["alias"], entry["metric_key"]) for entry in route_results],
            [("A", "x_route_a"), ("B", "kappa_fit"), ("C", "kappa_route_c")],
        )
        self.assertEqual(route_results[0]["auxiliary_metric_keys"], [])
        self.assertEqual(route_results[1]["auxiliary_metric_keys"], ["x_fit"])
        self.assertEqual(route_results[2]["auxiliary_metric_keys"], ["x_route_c"])
        self.assertFalse(route_results[0]["formal_candidate"])
        self.assertTrue(route_results[1]["formal_candidate"])
        self.assertFalse(route_results[2]["formal_candidate"])
        self.assertEqual(route_results[1]["formal_role"], "object_formal")
        self.assertTrue(route_results[1]["selected_as_primary"])
        self.assertTrue(route_results[1]["selected_as_formal"])
        self.assertEqual(route_results[0]["formal_role"], "route_result")
        self.assertEqual(route_results[2]["formal_role"], "route_result")
        self.assertIn("route_qc_summary", route_results[2])
        self.assertEqual(
            set(route_results[2]["route_qc_summary"]),
            {
                "valid_points",
                "quality_median",
                "monotonic_violation_fraction",
                "tail_recovery_median",
                "stability_metric",
            },
        )
        self.assertEqual(
            route_results[2]["route_qc_summary"]["stability_metric"]["key"],
            "route_b_deviation_fraction",
        )

    def test_braided_routes_always_expose_abc_schema(self) -> None:
        series = pd.DataFrame(
            {
                "temperature_c": np.linspace(20.0, 80.0, 20),
                "length_axis_formal_px": np.linspace(120.0, 90.0, 20),
                "length_axis_recovery": np.linspace(0.0, 1.0, 20),
                "diameter_max_px": np.linspace(20.0, 40.0, 20),
                "diameter_max_recovery": np.linspace(0.0, 1.0, 20),
                "area_proj_formal_px2": np.linspace(300.0, 420.0, 20),
                "area_proj_recovery": np.linspace(0.0, 1.0, 20),
                "quality": np.full(20, 0.9),
                "centerline_disagreement": np.full(20, 0.01),
                "endpoint_gap_alt_centerline_px": np.full(20, 2.0),
                "endpoint_jump_px": np.full(20, 2.0),
                "endpoint_frame_jump_px": np.full(20, 2.0),
                "axis_peak_position_stability": np.full(20, 0.02),
                "body_mask_attachment_leak_fraction": np.zeros(20),
                "excluded_attachment_area_px2": np.zeros(20),
                "component_area_px2": np.full(20, 1000.0),
                "body_mask_area_px2": np.full(20, 800.0),
                "area_proj_definition_gap_px2": np.full(20, 10.0),
            }
        )
        reports = {
            "length_axis": _metric_eval("length_axis", increasing=False, af95_c=60.0, aftan_c=58.0, dynamic_range=30.0),
            "diameter_max": _metric_eval("diameter_max", increasing=True, af95_c=59.0, aftan_c=57.0, dynamic_range=20.0),
            "area_proj": _metric_eval("area_proj", increasing=True, af95_c=61.0, aftan_c=58.5, dynamic_range=120.0),
        }

        route_results = _build_braided_route_results(
            series,
            reports,
            primary_metric_label="length_axis",
            formal_metric_label="length_axis",
            acceptance_profile="real_video",
            temperature_available=True,
        )

        self.assertEqual([entry["alias"] for entry in route_results], ["A", "B", "C"])
        for entry in route_results:
            self.assertTrue(
                {
                    "metric_key",
                    "display_label",
                    "af95_c",
                    "aftan_c",
                    "reportability_status",
                    "gate_reason",
                    "warning_codes",
                    "accepted_as_formal_candidate",
                }.issubset(entry)
            )

    def test_braided_route_c_stays_provisional_when_selected_as_primary(self) -> None:
        plateau_tail = np.concatenate(
            [
                np.linspace(0.0, 0.88, 15, endpoint=True),
                np.linspace(0.92, 1.0, 5, endpoint=True),
            ]
        )
        series = pd.DataFrame(
            {
                "temperature_c": np.linspace(20.0, 80.0, 20),
                "length_axis_formal_px": np.linspace(120.0, 90.0, 20),
                "length_axis_recovery": plateau_tail,
                "diameter_max_px": np.linspace(20.0, 40.0, 20),
                "diameter_max_recovery": plateau_tail,
                "area_proj_formal_px2": np.linspace(300.0, 420.0, 20),
                "area_proj_recovery": plateau_tail,
                "quality": np.full(20, 0.9),
                "centerline_disagreement": np.full(20, 0.01),
                "endpoint_gap_alt_centerline_px": np.full(20, 2.0),
                "endpoint_jump_px": np.full(20, 2.0),
                "endpoint_frame_jump_px": np.full(20, 2.0),
                "axis_peak_position_stability": np.full(20, 0.02),
                "body_mask_attachment_leak_fraction": np.zeros(20),
                "excluded_attachment_area_px2": np.zeros(20),
                "component_area_px2": np.full(20, 1000.0),
                "body_mask_area_px2": np.full(20, 800.0),
                "area_proj_definition_gap_px2": np.full(20, 10.0),
            }
        )
        reports = {
            "length_axis": _metric_eval("length_axis", increasing=False, af95_c=60.0, aftan_c=58.0, dynamic_range=30.0),
            "diameter_max": _metric_eval("diameter_max", increasing=True, af95_c=59.0, aftan_c=57.0, dynamic_range=20.0),
            "area_proj": _metric_eval("area_proj", increasing=True, af95_c=61.0, aftan_c=58.5, dynamic_range=120.0),
        }

        route_results = _build_braided_route_results(
            series,
            reports,
            primary_metric_label="area_proj",
            formal_metric_label="diameter_max",
            acceptance_profile="real_video",
            temperature_available=True,
        )

        route_c = route_results[2]
        self.assertEqual(route_c["alias"], "C")
        self.assertEqual(route_c["metric_key"], "area_proj")
        self.assertEqual(route_c["reportability_status"], "provisional")
        self.assertFalse(route_c["formal_candidate"])
        self.assertFalse(route_c["accepted_as_formal_candidate"])
        self.assertTrue(route_c["selected_as_primary"])
        self.assertFalse(route_c["selected_as_formal"])
        self.assertEqual(route_c["formal_role"], "object_primary")
        self.assertEqual(
            route_c["route_qc_summary"]["stability_metric"]["key"],
            "area_definition_gap_fraction_p95",
        )

    def test_flatten_route_results_expands_nested_qc_fields_for_csv_export(self) -> None:
        flattened = flatten_route_results(
            [
                {
                    "alias": "A",
                    "metric_key": "x_route_a",
                    "route_qc_summary": {
                        "valid_points": 20,
                        "quality_median": 0.9,
                        "monotonic_violation_fraction": 0.0,
                        "tail_recovery_median": 0.95,
                        "stability_metric": {
                            "key": "endpoint_jump_max_px",
                            "value": 3.5,
                        },
                    },
                }
            ]
        )

        self.assertEqual(len(flattened), 1)
        row = flattened[0]
        self.assertEqual(row["qc_valid_points"], 20)
        self.assertEqual(row["qc_quality_median"], 0.9)
        self.assertEqual(row["qc_stability_metric_key"], "endpoint_jump_max_px")
        self.assertEqual(row["qc_stability_metric_value"], 3.5)


if __name__ == "__main__":
    unittest.main()
