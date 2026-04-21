from __future__ import annotations

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import pandas as pd
from fastapi.testclient import TestClient

from niti_bfr.export_contract import ROUTE_RESULTS_SCHEMA_VERSION
from niti_bfr import webapp
from niti_bfr.webapp import (
    BENCHMARK_FAMILY_DISPLAY,
    SAMPLE_RUNS,
    _build_benchmark_page_data,
    _expected_plot_outputs,
    _mode_label,
    _postprocess_needed,
    _plot_route_titles_for_series,
    _prepare_summary_for_display,
    _preset_label,
    _refresh_plot_outputs_for_display,
    _run_result_hint,
    _worker_output_label,
)


class WebappFrontendLabelTests(unittest.TestCase):
    def _write_json(self, path: Path, payload: dict[str, object] | list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _storage_patch_context(self, tmpdir: str):
        base = Path(tmpdir)
        data_root = base / "var" / "webapp"
        runs_root = data_root / "runs"
        previews_root = data_root / "previews"
        db_path = data_root / "runs.db"
        data_root.mkdir(parents=True, exist_ok=True)
        runs_root.mkdir(parents=True, exist_ok=True)
        previews_root.mkdir(parents=True, exist_ok=True)
        return mock.patch.multiple(
            webapp,
            DATA_ROOT=data_root,
            RUNS_ROOT=runs_root,
            PREVIEWS_ROOT=previews_root,
            DB_PATH=db_path,
        )

    def _benchmark_patch_context(self, outputs_root: Path):
        display = copy.deepcopy(BENCHMARK_FAMILY_DISPLAY)
        display["wire"]["suite_path"] = outputs_root / "wire_benchmark_suite" / "benchmark_summary.json"
        display["braided"]["suite_path"] = outputs_root / "braided_benchmark_suite" / "benchmark_summary.json"
        return mock.patch.multiple(
            webapp,
            PROJECT_OUTPUTS_ROOT=outputs_root,
            BENCHMARK_FAMILY_DISPLAY=display,
        )

    def test_braided_sample_is_exposed(self) -> None:
        self.assertIn("braided_synthetic_quicklook", SAMPLE_RUNS)
        self.assertEqual(SAMPLE_RUNS["braided_synthetic_quicklook"]["preset"], "braided_demo")
        self.assertEqual(SAMPLE_RUNS["braided_synthetic_quicklook"]["requested_mode"], "quicklook")

    def test_labels_are_human_readable(self) -> None:
        self.assertEqual(_preset_label("braided_like"), "编织类 braided-like")
        self.assertEqual(_mode_label("formal_af"), "正式 Af")

    def test_worker_route_titles_are_human_readable(self) -> None:
        self.assertEqual(webapp._worker_route_title("demo", "A"), "两端距离")
        self.assertEqual(webapp._worker_route_title("demo", "B"), "整体弯曲程度")
        self.assertEqual(webapp._worker_route_title("braided_demo", "C"), "投影面积")

    def test_route_curve_labels_follow_object_type(self) -> None:
        self.assertEqual(_worker_output_label("route_a_metric_over_time.png", "demo"), "两端距离变化图")
        self.assertEqual(_worker_output_label("route_c_recovery_vs_temperature.png", "demo"), "连续跟踪后的弯曲程度温度曲线")
        self.assertEqual(_worker_output_label("route_a_metric_over_time.png", "braided_demo"), "主体长度变化图")
        self.assertEqual(_worker_output_label("route_b_recovery_vs_temperature.png", "braided_demo"), "主体宽度温度曲线")
        self.assertEqual(_worker_output_label("route_c_recovery_vs_temperature.png", "braided_demo"), "投影面积温度曲线")
        self.assertEqual(_worker_output_label("direction_metric_over_time.png", "braided_demo"), "方向法变化图")
        self.assertEqual(_worker_output_label("direction_recovery_vs_temperature.png", "braided_demo"), "方向法温度曲线")

    def test_plot_route_titles_follow_series_family(self) -> None:
        wire_titles = _plot_route_titles_for_series(pd.DataFrame({"x_route_a_px": [1.0], "kappa_fit_px_inv": [0.1]}))
        braided_titles = _plot_route_titles_for_series(pd.DataFrame({"length_axis_px": [10.0], "diameter_max_px": [2.0]}))
        self.assertEqual(wire_titles, {"A": "两端距离", "B": "整体弯曲程度", "C": "连续跟踪后的弯曲程度"})
        self.assertEqual(braided_titles, {"A": "主体长度", "B": "主体宽度", "C": "投影面积"})

    def test_result_hint_shows_formal_downgrade(self) -> None:
        run = {
            "status": "completed",
            "requested_mode": "formal_af",
            "actual_mode": "quicklook",
            "formal_gate_reason": "body_mask_attachment_leak_fraction",
            "reportability_status": "reportable_with_warning",
        }
        hint = _run_result_hint(run)
        self.assertIn("formal Af 未放行", hint)
        self.assertIn("临时结果", hint)

    def test_result_hint_shows_formal_success(self) -> None:
        run = {
            "status": "completed",
            "requested_mode": "formal_af",
            "actual_mode": "formal_af",
            "formal_gate_reason": None,
        }
        self.assertEqual(_run_result_hint(run), "formal Af 已放行。")

    def test_prepare_summary_backfills_wire_route_results_from_metric_reports(self) -> None:
        run = {
            "id": "wire-run-1",
            "preset": "wire_like",
            "requested_mode": "formal_af",
            "actual_mode": "quicklook",
            "temperature_filename": "wire.csv",
        }
        summary = {
            "preset": "wire_like",
            "requested_mode": "formal_af",
            "actual_mode": "quicklook",
            "formal_gate_reason": "insufficient_kappa_points",
            "provisional_metric_label": "kappa_route_c",
            "metric_reports": {
                "x_route_a": {"af95_c": 61.0, "aftan_c": 58.0, "fit_rmse": 0.4, "monotonic_violation_fraction": 0.0},
                "kappa_fit": {"af95_c": 60.0, "aftan_c": 57.0, "fit_rmse": 0.05, "monotonic_violation_fraction": 0.2},
                "kappa_route_c": {
                    "af95_c": 60.5,
                    "aftan_c": 57.5,
                    "fit_rmse": 0.03,
                    "monotonic_violation_fraction": 0.0,
                },
            },
        }

        prepared = _prepare_summary_for_display(run, summary)
        self.assertIsNotNone(prepared)
        assert prepared is not None
        self.assertEqual(prepared["route_alias_order"], ["A", "B", "C"])
        self.assertEqual(prepared["route_results_schema_version"], ROUTE_RESULTS_SCHEMA_VERSION)
        self.assertEqual([entry["alias"] for entry in prepared["route_results"]], ["A", "B", "C"])
        self.assertEqual(prepared["recommended_route_alias"], "C")
        self.assertEqual(prepared["route_results_by_alias"]["A"]["reportability_status"], "formal_blocked")
        self.assertEqual(prepared["route_results_by_alias"]["B"]["gate_reason"], "insufficient_kappa_points")
        self.assertEqual(prepared["route_results_by_alias"]["C"]["reportability_status"], "provisional")

    def test_prepare_summary_keeps_braided_area_route_visible(self) -> None:
        run = {
            "id": "braided-run-1",
            "preset": "braided_like",
            "requested_mode": "formal_af",
            "actual_mode": "quicklook",
            "temperature_filename": "braided.csv",
        }
        summary = {
            "preset": "braided_like",
            "requested_mode": "formal_af",
            "actual_mode": "quicklook",
            "formal_gate_reason": "centerline_disagreement",
            "metric_reports": {
                "length_axis": {"af95_c": 60.0, "aftan_c": 58.0},
                "diameter_max": {"af95_c": 59.0, "aftan_c": 57.0},
                "area_proj": {"af95_c": 61.0, "aftan_c": 58.5},
            },
            "route_results": [
                {"alias": "A", "metric_key": "length_axis", "reportability_status": "formal_blocked", "gate_reason": "centerline_disagreement"},
                {"alias": "B", "metric_key": "diameter_max", "reportability_status": "formal_blocked", "gate_reason": "centerline_disagreement"},
            ],
        }

        prepared = _prepare_summary_for_display(run, summary)
        self.assertIsNotNone(prepared)
        assert prepared is not None
        route_c = prepared["route_results_by_alias"]["C"]
        self.assertEqual(route_c["metric_key"], "area_proj")
        self.assertEqual(route_c["display_label"], "C:area_proj")
        self.assertFalse(route_c["formal_candidate"])
        self.assertFalse(route_c["accepted_as_formal_candidate"])
        self.assertEqual(route_c["reportability_status"], "formal_blocked")

    def test_prepare_summary_keeps_direction_result(self) -> None:
        run = {
            "id": "braided-run-direction",
            "preset": "braided_like",
            "requested_mode": "formal_af",
            "actual_mode": "formal_af",
            "temperature_filename": "braided.csv",
        }
        summary = {
            "preset": "braided_like",
            "requested_mode": "formal_af",
            "actual_mode": "formal_af",
            "direction_result": {
                "enabled": True,
                "angle_deg": 12.0,
                "metric_key": "direction_span",
                "reportability_status": "formal_passed",
            },
        }

        prepared = _prepare_summary_for_display(run, summary)
        self.assertIsNotNone(prepared)
        assert prepared is not None
        self.assertEqual(prepared["direction_result"]["angle_deg"], 12.0)
        self.assertTrue(prepared["direction_result"]["enabled"])

    def test_prepare_summary_backfills_smoothed_values_from_analysis_csv(self) -> None:
        with TemporaryDirectory() as tmp:
            with self._storage_patch_context(tmp):
                webapp._ensure_storage()
                run_id = "wire-run-smoothed-backfill"
                run_dir = webapp.RUNS_ROOT / run_id
                outputs_dir = run_dir / "outputs"
                outputs_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    {
                        "temperature_c": [20.0, 28.0, 36.0, 44.0, 52.0, 58.0, 62.0, 66.0],
                        "x_route_a_recovery": [0.0, 0.03, 0.1, 0.45, 0.92, 0.98, 1.0, 1.0],
                        "kappa_fit_recovery": [0.0, 0.02, 0.08, 0.25, 0.82, 0.97, 1.0, 1.0],
                        "kappa_route_c_recovery": [0.0, 0.01, 0.06, 0.2, 0.8, 0.96, 1.0, 1.0],
                    }
                ).to_csv(outputs_dir / "analysis.csv", index=False)

                run = {
                    "id": run_id,
                    "preset": "demo",
                    "requested_mode": "formal_af",
                    "actual_mode": "formal_af",
                    "temperature_filename": "temp.csv",
                    "run_dir": str(run_dir),
                }
                summary = {
                    "preset": "demo",
                    "requested_mode": "formal_af",
                    "actual_mode": "formal_af",
                    "reportability_status": "formal",
                    "formal_metric_label": "kappa_fit",
                    "provisional_metric_label": "kappa_fit",
                    "route_results": [
                        {"alias": "A", "metric_key": "x_route_a", "af95_c": 50.0, "aftan_c": 48.0},
                        {"alias": "B", "metric_key": "kappa_fit", "af95_c": 53.0, "aftan_c": 51.0},
                        {"alias": "C", "metric_key": "kappa_route_c", "af95_c": 53.1, "aftan_c": 51.1},
                    ],
                }

                prepared = _prepare_summary_for_display(run, summary)

        self.assertIsNotNone(prepared)
        assert prepared is not None
        self.assertIsNotNone(prepared["route_results_by_alias"]["B"]["smoothed_af95_c"])
        self.assertIsNotNone(prepared["route_results_by_alias"]["B"]["smoothed_aftan_c"])
        self.assertIsNotNone(prepared["object_smoothed_af95_c"])
        self.assertEqual(prepared["object_smoothed_route_alias"], "B")

    def test_refresh_plot_outputs_for_display_rewrites_legacy_temperature_plots(self) -> None:
        with TemporaryDirectory() as tmp:
            with self._storage_patch_context(tmp):
                webapp._ensure_storage()
                run_id = "wire-run-plot-refresh"
                run_dir = webapp.RUNS_ROOT / run_id
                outputs_dir = run_dir / "outputs"
                outputs_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    {
                        "temperature_c": [20.0, 28.0, 36.0, 44.0, 52.0, 58.0, 62.0, 66.0],
                        "x_route_a_recovery": [0.0, 0.03, 0.1, 0.45, 0.92, 0.98, 1.0, 1.0],
                        "kappa_fit_recovery": [0.0, 0.02, 0.08, 0.25, 0.82, 0.97, 1.0, 1.0],
                        "kappa_route_c_recovery": [0.0, 0.01, 0.06, 0.2, 0.8, 0.96, 1.0, 1.0],
                    }
                ).to_csv(outputs_dir / "analysis.csv", index=False)

                run = {
                    "id": run_id,
                    "preset": "demo",
                    "requested_mode": "formal_af",
                    "actual_mode": "formal_af",
                    "temperature_filename": "temp.csv",
                    "run_dir": str(run_dir),
                }
                summary = {
                    "preset": "demo",
                    "requested_mode": "formal_af",
                    "actual_mode": "formal_af",
                    "reportability_status": "formal",
                    "formal_metric_label": "kappa_fit",
                    "provisional_metric_label": "kappa_fit",
                    "route_results": [
                        {"alias": "A", "metric_key": "x_route_a", "af95_c": 50.0, "aftan_c": 48.0},
                        {"alias": "B", "metric_key": "kappa_fit", "af95_c": 53.0, "aftan_c": 51.0},
                        {"alias": "C", "metric_key": "kappa_route_c", "af95_c": 53.1, "aftan_c": 51.1},
                    ],
                }

                prepared = _prepare_summary_for_display(run, summary)
                with mock.patch.object(webapp, "_schedule_postprocess", autospec=True) as schedule:
                    _refresh_plot_outputs_for_display(run, prepared)

                schedule.assert_called_once()

    def test_postprocess_needed_when_only_partial_plot_set_exists(self) -> None:
        with TemporaryDirectory() as tmp:
            with self._storage_patch_context(tmp):
                webapp._ensure_storage()
                run_id = "braided-run-partial-plots"
                run_dir = webapp.RUNS_ROOT / run_id
                outputs_dir = run_dir / "outputs"
                outputs_dir.mkdir(parents=True, exist_ok=True)
                (outputs_dir / "route_a_metric_over_time.png").write_bytes(b"partial")

                run = {
                    "id": run_id,
                    "preset": "braided_like",
                    "requested_mode": "quicklook",
                    "actual_mode": "quicklook",
                    "temperature_filename": None,
                    "run_dir": str(run_dir),
                    "direction_metric_enabled": False,
                    "status": "completed",
                }
                prepared = {
                    "preset": "braided_like",
                    "actual_mode": "quicklook",
                    "asset_generation_status": "failed",
                    "process_video_filename": webapp.PROCESS_VIDEO_FILENAME,
                    "temperature_c_min": None,
                    "temperature_c_max": None,
                }

                self.assertTrue(_postprocess_needed(run, prepared))

    def test_postprocess_ignores_disabled_direction_metric_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            with self._storage_patch_context(tmp):
                webapp._ensure_storage()
                run_id = "braided-run-direction-disabled"
                run_dir = webapp.RUNS_ROOT / run_id
                outputs_dir = run_dir / "outputs"
                outputs_dir.mkdir(parents=True, exist_ok=True)

                run = {
                    "id": run_id,
                    "preset": "braided_like",
                    "requested_mode": "quicklook",
                    "actual_mode": "quicklook",
                    "temperature_filename": None,
                    "run_dir": str(run_dir),
                    "direction_metric_enabled": False,
                    "status": "completed",
                }
                prepared = {
                    "preset": "braided_like",
                    "actual_mode": "quicklook",
                    "asset_generation_status": "completed",
                    "process_video_filename": webapp.PROCESS_VIDEO_FILENAME,
                    "temperature_c_min": None,
                    "temperature_c_max": None,
                    "direction_result": {"enabled": False, "metric_key": "direction_span"},
                }

                expected = _expected_plot_outputs(run, prepared)
                self.assertNotIn("direction_metric_over_time.png", expected)
                for filename in expected:
                    (outputs_dir / filename).write_bytes(b"plot")
                (outputs_dir / webapp.PROCESS_VIDEO_FILENAME).write_bytes(b"video")

                self.assertFalse(_postprocess_needed(run, prepared))

    def test_refresh_plot_outputs_for_display_schedules_when_partial_plots_exist(self) -> None:
        with TemporaryDirectory() as tmp:
            with self._storage_patch_context(tmp):
                webapp._ensure_storage()
                run_id = "wire-run-refresh-partial"
                run_dir = webapp.RUNS_ROOT / run_id
                outputs_dir = run_dir / "outputs"
                outputs_dir.mkdir(parents=True, exist_ok=True)
                (outputs_dir / "route_a_metric_over_time.png").write_bytes(b"partial")
                (outputs_dir / webapp.PROCESS_VIDEO_FILENAME).write_bytes(b"video")

                run = {
                    "id": run_id,
                    "preset": "demo",
                    "requested_mode": "quicklook",
                    "actual_mode": "quicklook",
                    "temperature_filename": None,
                    "run_dir": str(run_dir),
                    "status": "completed",
                }
                prepared = {
                    "preset": "demo",
                    "actual_mode": "quicklook",
                    "asset_generation_status": "completed",
                    "process_video_filename": webapp.PROCESS_VIDEO_FILENAME,
                    "temperature_c_min": None,
                    "temperature_c_max": None,
                }

                with mock.patch.object(webapp, "_schedule_postprocess", autospec=True) as schedule:
                    _refresh_plot_outputs_for_display(run, prepared)

                schedule.assert_called_once()

    def test_delete_run_blocks_pending_or_running_asset_generation(self) -> None:
        for asset_status in ("pending", "running"):
            with self.subTest(asset_status=asset_status):
                with TemporaryDirectory() as tmp:
                    with self._storage_patch_context(tmp):
                        webapp._ensure_storage()
                        run_id = f"delete-blocked-{asset_status}"
                        run_dir = webapp.RUNS_ROOT / run_id
                        outputs_dir = run_dir / "outputs"
                        outputs_dir.mkdir(parents=True, exist_ok=True)

                        webapp._insert_run(
                            {
                                "id": run_id,
                                "created_at": webapp._utc_now(),
                                "status": "completed",
                                "run_name": "blocked delete",
                                "preset": "braided_like",
                                "requested_mode": "quicklook",
                                "frame_stride": 1,
                                "actual_mode": "quicklook",
                                "formal_metric_label": None,
                                "formal_gate_reason": None,
                                "af95_c": None,
                                "aftan_c": None,
                                "original_frame_count": 12,
                                "analyzed_frame_count": 12,
                                "annotated_video_filename": None,
                                "video_filename": "demo.mp4",
                                "temperature_filename": None,
                                "run_dir": str(run_dir),
                                "error_text": None,
                                "direction_angle_deg": None,
                                "direction_metric_enabled": 0,
                            }
                        )
                        self._write_json(
                            outputs_dir / "summary.json",
                            {
                                "preset": "braided_like",
                                "requested_mode": "quicklook",
                                "actual_mode": "quicklook",
                                "asset_generation_status": asset_status,
                            },
                        )

                        client = TestClient(webapp.app)
                        response = client.post(f"/runs/{run_id}/delete")

                        self.assertEqual(response.status_code, 409)
                        self.assertTrue(run_dir.exists())
                        self.assertIsNotNone(webapp._get_run(run_id))

    def test_benchmark_page_falls_back_to_analysis_metrics(self) -> None:
        with TemporaryDirectory() as tmp:
            outputs_root = Path(tmp)
            wire_out = outputs_root / "wire_demo_case"
            braided_out = outputs_root / "braided_demo_case"

            self._write_json(
                wire_out / "analysis_metrics.json",
                {
                    "benchmark_name": "wire_demo_case",
                    "benchmark_description": "wire fallback case",
                    "demo_output": str(wire_out),
                    "preset": "demo",
                    "mode": "formal_af",
                    "reportability_status": "reportable",
                    "formal_metric_label": "kappa_fit",
                    "primary_metric_label": "kappa_fit",
                    "formal_gate_reason": None,
                    "af95_c": 52.0,
                    "aftan_c": 50.0,
                    "route_results": [
                        {
                            "alias": "A",
                            "metric_key": "x_route_a",
                            "display_label": "A:x_route_a",
                            "reportability_status": "formal_blocked",
                            "gate_reason": "route_a_not_primary",
                            "af95_c": 49.4,
                            "aftan_c": 48.1,
                        },
                        {
                            "alias": "B",
                            "metric_key": "kappa_fit",
                            "display_label": "B:kappa_fit",
                            "reportability_status": "formal_passed",
                            "af95_c": 52.0,
                            "aftan_c": 50.0,
                            "formal_candidate": True,
                            "accepted_as_formal_candidate": True,
                            "selected_as_primary": True,
                            "selected_as_formal": True,
                        },
                        {
                            "alias": "C",
                            "metric_key": "kappa_route_c",
                            "display_label": "C:kappa_route_c",
                            "reportability_status": "provisional",
                            "af95_c": 52.2,
                            "aftan_c": 50.2,
                        },
                    ],
                    "route_benchmark_overview_by_alias": {
                        "A": {"metric_key": "x_route_a", "measured_af95_c": 49.4, "truth_af95_c": 49.0, "af95_error_c": 0.4, "measured_aftan_c": 48.1, "truth_aftan_c": 47.8, "aftan_error_c": 0.3},
                        "B": {"metric_key": "kappa_fit", "measured_af95_c": 52.0, "truth_af95_c": 51.8, "af95_error_c": 0.2, "measured_aftan_c": 50.0, "truth_aftan_c": 49.9, "aftan_error_c": 0.1},
                        "C": {"metric_key": "kappa_route_c", "measured_af95_c": 52.2, "truth_af95_c": 51.8, "af95_error_c": 0.4, "measured_aftan_c": 50.2, "truth_aftan_c": 49.9, "aftan_error_c": 0.3},
                    },
                    "wire_qc": {
                        "quality_median": 0.93,
                        "route_a_endpoint_jump_p95_px": 2.8,
                        "centerline_points_median": 41,
                        "quadratic_fraction": 0.88,
                    },
                },
            )
            self._write_json(
                braided_out / "analysis_metrics.json",
                {
                    "benchmark_name": "braided_demo_case",
                    "benchmark_description": "braided fallback case",
                    "demo_output": str(braided_out),
                    "preset": "braided_demo",
                    "mode": "quicklook",
                    "reportability_status": "reportable_with_warning",
                    "provisional_metric_label": "diameter_max",
                    "primary_metric_label": "diameter_max",
                    "formal_gate_reason": "endpoint_jump",
                    "provisional_af95_c": 53.1,
                    "provisional_aftan_c": 50.8,
                    "route_results": [
                        {
                            "alias": "A",
                            "metric_key": "length_axis",
                            "display_label": "A:length_axis",
                            "reportability_status": "formal_blocked",
                            "gate_reason": "endpoint_jump",
                            "af95_c": 53.4,
                            "aftan_c": 50.7,
                            "formal_candidate": True,
                        },
                        {
                            "alias": "B",
                            "metric_key": "diameter_max",
                            "display_label": "B:diameter_max",
                            "reportability_status": "provisional",
                            "af95_c": 53.1,
                            "aftan_c": 50.8,
                            "formal_candidate": True,
                            "accepted_as_formal_candidate": True,
                            "selected_as_primary": True,
                        },
                        {
                            "alias": "C",
                            "metric_key": "area_proj",
                            "display_label": "C:area_proj",
                            "reportability_status": "formal_blocked",
                            "gate_reason": "endpoint_jump",
                            "af95_c": 52.5,
                            "aftan_c": 50.0,
                        },
                    ],
                    "af_comparison_by_alias": {
                        "A": {"metric_key": "length_axis", "measured_af95_c": 53.4, "truth_af95_c": 53.0, "af95_error_c": 0.4, "measured_aftan_c": 50.7, "truth_aftan_c": 50.9, "aftan_error_c": -0.2},
                        "B": {"metric_key": "diameter_max", "measured_af95_c": 53.1, "truth_af95_c": 52.9, "af95_error_c": 0.2, "measured_aftan_c": 50.8, "truth_aftan_c": 50.9, "aftan_error_c": -0.1},
                        "C": {"metric_key": "area_proj", "measured_af95_c": 52.5, "truth_af95_c": 52.1, "af95_error_c": 0.4, "measured_aftan_c": 50.0, "truth_aftan_c": 50.2, "aftan_error_c": -0.2},
                    },
                    "quality_median": 0.91,
                    "body_mask_attachment_leak_fraction": 0.01,
                    "centerline_disagreement_median": 0.03,
                    "endpoint_jump_p95_px": 28.0,
                },
            )

            with self._benchmark_patch_context(outputs_root):
                page = _build_benchmark_page_data()

            self.assertEqual(page["families"]["wire"]["benchmark_count"], 1)
            self.assertEqual(page["families"]["braided"]["benchmark_count"], 1)
            self.assertEqual(page["families"]["wire"]["source_label"], "回退到 analysis_metrics")
            braided_benchmark = page["families"]["braided"]["benchmarks"][0]
            self.assertEqual(braided_benchmark["object_recommended_route_alias"], "B")
            self.assertEqual(braided_benchmark["routes_by_alias"]["B"]["reportability_status"], "provisional")
            self.assertAlmostEqual(page["comparison_rows"][1]["wire"]["mean_abs_af95_error_c"], 0.2)

    def test_benchmark_page_prefers_suite_summary_when_present(self) -> None:
        with TemporaryDirectory() as tmp:
            outputs_root = Path(tmp)
            wire_out = outputs_root / "wire_suite_case"
            self._write_json(
                wire_out / "analysis_metrics.json",
                {
                    "benchmark_name": "wire_suite_case",
                    "benchmark_description": "wire suite detail",
                    "demo_output": str(wire_out),
                    "preset": "demo",
                    "mode": "formal_af",
                    "reportability_status": "reportable",
                    "formal_metric_label": "kappa_fit",
                    "route_results": [
                        {"alias": "A", "metric_key": "x_route_a", "display_label": "A:x_route_a", "reportability_status": "formal_blocked"},
                        {"alias": "B", "metric_key": "kappa_fit", "display_label": "B:kappa_fit", "reportability_status": "formal_passed", "selected_as_formal": True},
                        {"alias": "C", "metric_key": "kappa_route_c", "display_label": "C:kappa_route_c", "reportability_status": "provisional"},
                    ],
                    "route_benchmark_overview_by_alias": {
                        "A": {"metric_key": "x_route_a", "measured_af95_c": 49.2, "truth_af95_c": 49.0, "af95_error_c": 0.2},
                        "B": {"metric_key": "kappa_fit", "measured_af95_c": 52.0, "truth_af95_c": 51.8, "af95_error_c": 0.2},
                        "C": {"metric_key": "kappa_route_c", "measured_af95_c": 52.1, "truth_af95_c": 51.8, "af95_error_c": 0.3},
                    },
                    "wire_qc": {"quality_median": 0.9, "route_a_endpoint_jump_p95_px": 3.0, "centerline_points_median": 40, "quadratic_fraction": 0.8},
                },
            )
            self._write_json(
                outputs_root / "wire_benchmark_suite" / "benchmark_summary.json",
                [
                    {
                        "benchmark_name": "wire_suite_case",
                        "description": "suite row",
                        "output_dir": str(wire_out),
                        "reportability_status": "reportable",
                        "route_A_metric_key": "x_route_a",
                        "route_A_display_label": "A:x_route_a",
                        "route_A_available": True,
                        "route_A_status": "formal_blocked",
                        "route_A_af95_error_c": 0.2,
                        "route_B_metric_key": "kappa_fit",
                        "route_B_display_label": "B:kappa_fit",
                        "route_B_available": True,
                        "route_B_status": "formal_passed",
                        "route_B_selected_as_formal": True,
                        "route_B_af95_error_c": 0.2,
                        "route_C_metric_key": "kappa_route_c",
                        "route_C_display_label": "C:kappa_route_c",
                        "route_C_available": True,
                        "route_C_status": "provisional",
                        "route_C_af95_error_c": 0.3,
                    }
                ],
            )

            with self._benchmark_patch_context(outputs_root):
                page = _build_benchmark_page_data()

            self.assertTrue(page["families"]["wire"]["suite_available"])
            self.assertEqual(page["families"]["wire"]["source_label"], "优先读取汇总文件")
            self.assertEqual(page["families"]["wire"]["benchmarks"][0]["source_label"], "汇总文件 + analysis_metrics")

    def test_run_detail_uses_worker_friendly_chinese_labels(self) -> None:
        with TemporaryDirectory() as tmp:
            with self._storage_patch_context(tmp):
                webapp._ensure_storage()
                run_id = "run-detail-worker"
                run_dir = webapp.RUNS_ROOT / run_id
                outputs_dir = run_dir / "outputs"
                outputs_dir.mkdir(parents=True, exist_ok=True)

                webapp._insert_run(
                    {
                        "id": run_id,
                        "created_at": webapp._utc_now(),
                        "status": "completed",
                        "run_name": "wire-like formal",
                        "preset": "demo",
                        "requested_mode": "formal_af",
                        "frame_stride": 1,
                        "actual_mode": "formal_af",
                        "formal_metric_label": "kappa_fit",
                        "formal_gate_reason": None,
                        "af95_c": 52.8,
                        "aftan_c": 50.8,
                        "original_frame_count": 480,
                        "analyzed_frame_count": 480,
                        "annotated_video_filename": "annotated_overview.mp4",
                        "video_filename": "demo.mp4",
                        "temperature_filename": "temp.csv",
                        "run_dir": str(run_dir),
                        "error_text": None,
                    }
                )
                self._write_json(
                    outputs_dir / "summary.json",
                    {
                        "preset": "demo",
                        "requested_mode": "formal_af",
                        "actual_mode": "formal_af",
                        "reportability_status": "formal",
                        "formal_metric_label": "kappa_fit",
                        "provisional_metric_label": "kappa_fit",
                        "object_smoothed_af95_c": 52.81,
                        "object_smoothed_aftan_c": 50.84,
                        "temperature_c_min": 20.0,
                        "temperature_c_max": 65.0,
                        "analyzed_frame_count": 480,
                        "route_results": [
                            {"alias": "A", "metric_key": "x_route_a", "display_label": "A:x_route_a", "reportability_status": "provisional", "af95_c": 49.6, "aftan_c": 48.2, "smoothed_af95_c": 49.68, "smoothed_aftan_c": 47.94},
                            {"alias": "B", "metric_key": "kappa_fit", "display_label": "B:kappa_fit", "reportability_status": "formal_passed", "selected_as_formal": True, "selected_as_primary": True, "af95_c": 52.8, "aftan_c": 50.8, "smoothed_af95_c": 52.81, "smoothed_aftan_c": 50.84},
                            {"alias": "C", "metric_key": "kappa_route_c", "display_label": "C:kappa_route_c", "reportability_status": "formal_blocked", "gate_reason": "kappa_route_c_dynamic_range_too_small", "af95_c": 52.9, "aftan_c": 51.0, "smoothed_af95_c": 52.90, "smoothed_aftan_c": 51.02},
                        ],
                    },
                )
                for name in ["annotated_overview.mp4", "analysis_process.mp4", "route_b_metric_over_time.png", "route_results.csv"]:
                    (outputs_dir / name).write_bytes(b"demo")

                client = TestClient(webapp.app)
                response = client.get(f"/runs/{run_id}")

        self.assertEqual(response.status_code, 200)
        text = response.text
        self.assertIn("细丝对象测量结果", text)
        self.assertIn("测量方式一", text)
        self.assertIn("两端距离", text)
        self.assertIn("整体弯曲程度", text)
        self.assertIn("95%恢复温度", text)
        self.assertIn("分析过程视频", text)
        self.assertIn("稳健参考", text)
        self.assertIn("52.81℃", text)
        self.assertNotIn("A:x_route_a", text)
        self.assertNotIn("B:kappa_fit", text)
        self.assertNotIn("formal Af", text)
        self.assertNotIn("quicklook", text)

    def test_braided_run_detail_uses_braided_curve_labels(self) -> None:
        with TemporaryDirectory() as tmp:
            with self._storage_patch_context(tmp):
                webapp._ensure_storage()
                run_id = "run-detail-braided-curves"
                run_dir = webapp.RUNS_ROOT / run_id
                outputs_dir = run_dir / "outputs"
                outputs_dir.mkdir(parents=True, exist_ok=True)

                webapp._insert_run(
                    {
                        "id": run_id,
                        "created_at": webapp._utc_now(),
                        "status": "completed",
                        "run_name": "braided formal",
                        "preset": "braided_demo",
                        "requested_mode": "formal_af",
                        "frame_stride": 1,
                        "actual_mode": "formal_af",
                        "formal_metric_label": "diameter_max",
                        "formal_gate_reason": None,
                        "af95_c": 52.1,
                        "aftan_c": 50.9,
                        "original_frame_count": 100,
                        "analyzed_frame_count": 100,
                        "annotated_video_filename": None,
                        "video_filename": "braided.mp4",
                        "temperature_filename": "temp.csv",
                        "run_dir": str(run_dir),
                        "error_text": None,
                    }
                )
                self._write_json(
                    outputs_dir / "summary.json",
                    {
                        "preset": "braided_demo",
                        "requested_mode": "formal_af",
                        "actual_mode": "formal_af",
                        "reportability_status": "formal",
                        "formal_metric_label": "diameter_max",
                        "provisional_metric_label": "diameter_max",
                        "temperature_c_min": 20.0,
                        "temperature_c_max": 65.0,
                        "analyzed_frame_count": 100,
                        "route_results": [
                            {"alias": "A", "metric_key": "length_axis", "display_label": "A:length_axis", "reportability_status": "formal_blocked", "af95_c": 53.4, "aftan_c": 50.7},
                            {"alias": "B", "metric_key": "diameter_max", "display_label": "B:diameter_max", "reportability_status": "formal_passed", "selected_as_formal": True, "selected_as_primary": True, "af95_c": 53.1, "aftan_c": 50.8},
                            {"alias": "C", "metric_key": "area_proj", "display_label": "C:area_proj", "reportability_status": "provisional", "af95_c": 52.5, "aftan_c": 50.0},
                        ],
                    },
                )
                for name in [
                    "analysis_process.mp4",
                    "route_a_recovery_vs_temperature.png",
                    "route_b_recovery_vs_temperature.png",
                    "route_c_recovery_vs_temperature.png",
                ]:
                    (outputs_dir / name).write_bytes(b"demo")

                client = TestClient(webapp.app)
                response = client.get(f"/runs/{run_id}")

        self.assertEqual(response.status_code, 200)
        text = response.text
        self.assertIn("编织对象测量结果", text)
        self.assertIn("主体长度温度曲线", text)
        self.assertIn("主体宽度温度曲线", text)
        self.assertIn("投影面积温度曲线", text)
        self.assertNotIn("两端距离温度曲线", text)
        self.assertNotIn("整体弯曲程度温度曲线", text)


if __name__ == "__main__":
    unittest.main()
