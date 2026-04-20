from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from niti_bfr import webapp
from niti_bfr.pipeline import AnalysisResult, analyze_video


class RealFrontendFlowTests(unittest.TestCase):
    def _patched_storage(self, tmpdir: str):
        base = Path(tmpdir)
        data_root = base / "var" / "webapp"
        runs_root = data_root / "runs"
        db_path = data_root / "runs.db"
        data_root.mkdir(parents=True, exist_ok=True)
        runs_root.mkdir(parents=True, exist_ok=True)
        return mock.patch.multiple(
            webapp,
            DATA_ROOT=data_root,
            RUNS_ROOT=runs_root,
            DB_PATH=db_path,
        )

    def test_home_page_removes_sample_and_benchmark_entrypoints(self) -> None:
        with TemporaryDirectory() as tmp:
            with self._patched_storage(tmp):
                webapp._ensure_storage()
                client = TestClient(webapp.app)
                response = client.get("/")

        self.assertEqual(response.status_code, 200)
        text = response.text
        self.assertIn("抽帧步长", text)
        self.assertNotIn("Benchmark 汇总", text)
        self.assertNotIn("直接运行这个示例", text)
        self.assertNotIn("分析模式", text)
        self.assertIn("开始分析", text)

    def test_home_page_recent_runs_hide_single_result_temperature_copy(self) -> None:
        with TemporaryDirectory() as tmp:
            with self._patched_storage(tmp):
                webapp._ensure_storage()
                run_id = "recent-run-copy"
                run_dir = webapp.RUNS_ROOT / run_id
                outputs_dir = run_dir / "outputs"
                outputs_dir.mkdir(parents=True, exist_ok=True)
                webapp._insert_run(
                    {
                        "id": run_id,
                        "created_at": webapp._utc_now(),
                        "status": "completed",
                        "run_name": "recent",
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
                        "annotated_video_filename": None,
                        "video_filename": "demo.mp4",
                        "temperature_filename": "temp.csv",
                        "run_dir": str(run_dir),
                        "error_text": None,
                    }
                )
                (outputs_dir / "summary.json").write_text(
                    '{"object_formal_af95_c": 52.8, "object_provisional_af95_c": 52.8}',
                    encoding="utf-8",
                )
                client = TestClient(webapp.app)
                response = client.get("/")

        self.assertEqual(response.status_code, 200)
        text = response.text
        self.assertNotIn("正式结果温度", text)
        self.assertNotIn("参考结果温度", text)

    def test_history_page_supports_deleting_completed_runs(self) -> None:
        with TemporaryDirectory() as tmp:
            with self._patched_storage(tmp), mock.patch.object(webapp, "_execute_run", autospec=True):
                webapp._ensure_storage()
                client = TestClient(webapp.app)
                response = client.post(
                    "/runs",
                    data={"preset": "wire_like", "frame_stride": "1", "run_name": "to-delete"},
                    files={"video_file": ("demo.mp4", b"not-a-real-video", "video/mp4")},
                    follow_redirects=False,
                )
                run_id = response.headers["location"].rsplit("/", 1)[-1]
                run_dir = webapp.RUNS_ROOT / run_id
                (run_dir / "outputs").mkdir(parents=True, exist_ok=True)
                (run_dir / "outputs" / "summary.json").write_text("{}", encoding="utf-8")
                webapp._update_run(run_id, {"status": "completed"})

                history_response = client.get("/history")
                delete_response = client.post(f"/runs/{run_id}/delete", follow_redirects=False)
                self.assertEqual(history_response.status_code, 200)
                self.assertIn("删除记录", history_response.text)
                self.assertEqual(delete_response.status_code, 303)
                self.assertTrue(delete_response.headers["location"].endswith("/history?deleted=1"))
                self.assertIsNone(webapp._get_run(run_id))
                self.assertFalse(run_dir.exists())

    def test_create_run_derives_requested_mode_and_persists_frame_stride(self) -> None:
        with TemporaryDirectory() as tmp:
            with self._patched_storage(tmp), mock.patch.object(webapp, "_execute_run", autospec=True):
                webapp._ensure_storage()
                client = TestClient(webapp.app)
                response = client.post(
                    "/runs",
                    data={"preset": "wire_like", "frame_stride": "5", "run_name": "real-run"},
                    files={"video_file": ("demo.mp4", b"not-a-real-video", "video/mp4")},
                    follow_redirects=False,
                )
                rows = webapp._list_runs(limit=1)

        self.assertEqual(response.status_code, 303)
        self.assertEqual(len(rows), 1)
        row = dict(rows[0])
        self.assertEqual(row["requested_mode"], "quicklook")
        self.assertEqual(row["frame_stride"], 5)

    def test_create_run_with_temperature_csv_derives_formal_request(self) -> None:
        with TemporaryDirectory() as tmp:
            with self._patched_storage(tmp), mock.patch.object(webapp, "_execute_run", autospec=True):
                webapp._ensure_storage()
                client = TestClient(webapp.app)
                response = client.post(
                    "/runs",
                    data={"preset": "braided_like", "frame_stride": "2"},
                    files={
                        "video_file": ("demo.mp4", b"not-a-real-video", "video/mp4"),
                        "temperature_file": ("temp.csv", b"frame,temperature_c\n0,20\n", "text/csv"),
                    },
                    follow_redirects=False,
                )
                rows = webapp._list_runs(limit=1)

        self.assertEqual(response.status_code, 303)
        self.assertEqual(dict(rows[0])["requested_mode"], "formal_af")

    def test_create_run_rejects_invalid_frame_stride(self) -> None:
        with TemporaryDirectory() as tmp:
            with self._patched_storage(tmp), mock.patch.object(webapp, "_execute_run", autospec=True):
                webapp._ensure_storage()
                client = TestClient(webapp.app)
                response = client.post(
                    "/runs",
                    data={"preset": "wire_like", "frame_stride": "0"},
                    files={"video_file": ("demo.mp4", b"not-a-real-video", "video/mp4")},
                )

        self.assertEqual(response.status_code, 400)
        self.assertIn("frame_stride", response.text)

    def test_wire_analysis_respects_frame_stride_and_reports_counts(self) -> None:
        class DummyCapture:
            def __init__(self, frames: list[np.ndarray]) -> None:
                self.frames = frames
                self.index = 0

            def isOpened(self) -> bool:
                return True

            def get(self, prop: int) -> float:
                if prop == 5:
                    return 10.0
                return 0.0

            def read(self):
                if self.index >= len(self.frames):
                    return False, None
                frame = self.frames[self.index]
                self.index += 1
                return True, frame

            def release(self) -> None:
                return None

        class DummyGeom:
            def __init__(self, frame_idx: int) -> None:
                self.route_a_anchor_xy = np.array([0.0, 0.0])
                self.route_a_tip_xy = np.array([10.0 + frame_idx, 2.0])
                self.x_route_a_px = 10.0 + frame_idx
                self.anchor_xy = np.array([1.0, 1.0])
                self.tip_xy = np.array([11.0 + frame_idx, 2.0])
                self.x_px = 11.0 + frame_idx
                self.quality = 0.9
                self.sampled_centerline_xy = np.array([[0.0, 0.0], [1.0, 1.0]])
                self.x_fit_px = 12.0 + frame_idx
                self.kappa_fit_px_inv = 0.2
                self.quadratic_rmse_px = 0.1
                self.circle_rmse_px = 0.1
                self.model_name = "quadratic"

        frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(6)]
        extraction = webapp._build_wire_extraction_config(
            {"analysis": {"demo_extraction": {"roi_xyxy": [0, 0, 4, 4], "blur_ksize": 1, "threshold_dark": 0, "open_kernel": 1, "close_kernel": 1}}},
            "demo",
        )
        with mock.patch("niti_bfr.pipeline.cv2.VideoCapture", return_value=DummyCapture(frames)), mock.patch(
            "niti_bfr.pipeline.extract_geometry",
            side_effect=[DummyGeom(idx) for idx in range(0, 6, 2)],
        ):
            result = analyze_video("dummy.mp4", extraction=extraction, frame_stride=2)

        self.assertEqual(result.original_frame_count, 6)
        self.assertEqual(result.analyzed_frame_count, 3)
        self.assertEqual(result.frame_stride, 2)
        self.assertEqual(result.series["frame"].tolist(), [0, 2, 4])

    def test_build_summary_keeps_frame_counts_for_results_page(self) -> None:
        result = AnalysisResult(
            series=pd.DataFrame({"frame": [0, 2, 4], "quality": [0.9, 0.9, 0.9]}),
            fit=None,
            af95_c=None,
            aftan_c=None,
            reportability_status="quicklook_only",
            route_results=[],
            input_fps=20.0,
            original_frame_count=8,
            analyzed_frame_count=3,
            frame_stride=2,
        )
        summary = webapp._build_summary(
            {
                "id": "run-1",
                "preset": "wire_like",
                "requested_mode": "quicklook",
                "video_filename": "demo.mp4",
                "temperature_filename": None,
                "annotated_video_filename": None,
            },
            result,
        )

        self.assertEqual(summary["frame_stride"], 2)
        self.assertEqual(summary["original_frame_count"], 8)
        self.assertEqual(summary["analyzed_frame_count"], 3)
        self.assertIsNone(summary["annotated_video_filename"])

    def test_execute_run_writes_process_video_summary_when_render_succeeds(self) -> None:
        with TemporaryDirectory() as tmp:
            with self._patched_storage(tmp):
                webapp._ensure_storage()
                run_id = "run-process-video"
                run_dir = webapp.RUNS_ROOT / run_id
                inputs_dir = run_dir / "inputs"
                outputs_dir = run_dir / "outputs"
                inputs_dir.mkdir(parents=True, exist_ok=True)
                outputs_dir.mkdir(parents=True, exist_ok=True)
                (inputs_dir / "demo.mp4").write_bytes(b"fake-video")

                webapp._insert_run(
                    {
                        "id": run_id,
                        "created_at": webapp._utc_now(),
                        "status": "queued",
                        "run_name": "process-video",
                        "preset": "wire_like",
                        "requested_mode": "quicklook",
                        "frame_stride": 1,
                        "actual_mode": None,
                        "formal_metric_label": None,
                        "formal_gate_reason": None,
                        "af95_c": None,
                        "aftan_c": None,
                        "original_frame_count": None,
                        "analyzed_frame_count": None,
                        "annotated_video_filename": None,
                        "video_filename": "demo.mp4",
                        "temperature_filename": None,
                        "run_dir": str(run_dir),
                        "error_text": None,
                    }
                )

                fake_result = AnalysisResult(
                    series=pd.DataFrame(
                        {
                            "frame": [0, 1],
                            "time_sec": [0.0, 0.05],
                            "quality": [0.9, 0.92],
                            "x_route_a_px": [10.0, 11.0],
                            "x_fit_px": [9.5, 10.5],
                            "kappa_fit_px_inv": [0.2, 0.18],
                            "x_route_c_px": [10.2, 11.2],
                            "kappa_route_c_px_inv": [0.19, 0.17],
                            "route_c_anchor_x": [1.0, 1.0],
                            "route_c_anchor_y": [1.0, 1.0],
                            "route_c_tip_x": [9.0, 10.0],
                            "route_c_tip_y": [2.0, 2.0],
                        }
                    ),
                    fit=None,
                    af95_c=None,
                    aftan_c=None,
                    reportability_status="quicklook_only",
                    route_results=[],
                    input_fps=20.0,
                    original_frame_count=2,
                    analyzed_frame_count=2,
                    frame_stride=1,
                )

                with mock.patch.object(webapp, "analyze_video", return_value=fake_result), mock.patch.object(
                    webapp,
                    "render_process_debug_video",
                    side_effect=lambda **kwargs: Path(kwargs["output_path"]).write_bytes(b"process"),
                ), mock.patch.object(webapp, "_write_plots", autospec=True), mock.patch.object(
                    webapp, "route_results_dataframe", return_value=pd.DataFrame()
                ):
                    webapp._execute_run(run_id)

                summary = webapp._load_summary(run_id)
                assert summary is not None
                self.assertIsNone(summary["annotated_video_filename"])
                self.assertEqual(summary["process_video_filename"], webapp.PROCESS_VIDEO_FILENAME)
                self.assertTrue((outputs_dir / webapp.PROCESS_VIDEO_FILENAME).exists())
                row = dict(webapp._get_run(run_id))
                self.assertEqual(row["status"], "completed")
                self.assertIsNone(row["annotated_video_filename"])


if __name__ == "__main__":
    unittest.main()
