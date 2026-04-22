from __future__ import annotations

import asyncio
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient

from niti_bfr import webapp
from niti_bfr.pipeline import AnalysisResult, analyze_video


class _FakeRequest:
    query_params: dict[str, str]

    def __init__(self, query_params: dict[str, str] | None = None) -> None:
        self.query_params = dict(query_params or {})

    def url_for(self, name: str, **path_params: object) -> str:
        if name == "home":
            return "/"
        if name == "history":
            return "/history"
        if name == "run_detail":
            return f"/runs/{path_params['run_id']}"
        if name == "delete_run":
            return f"/runs/{path_params['run_id']}/delete"
        if name == "create_run":
            return "/runs"
        if name == "create_video_preview":
            return "/preview-video"
        if name == "files":
            return f"/files/{path_params['path']}"
        return f"/{name}"


class _FakeUpload:
    def __init__(self, filename: str, data: bytes) -> None:
        self.filename = filename
        self._data = data
        self._read = False

    async def read(self, _size: int) -> bytes:
        if self._read:
            return b""
        self._read = True
        return self._data

    async def close(self) -> None:
        return None


class RealFrontendFlowTests(unittest.TestCase):
    def _patched_storage(self, tmpdir: str):
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

    def _create_run_direct(
        self,
        *,
        preset: str,
        frame_stride: str = "1",
        direction_angle_deg: str = "",
        initial_roi_xyxy: str = "",
        temperature_data: bytes | None = None,
    ):
        return asyncio.run(
            webapp.create_run(
                _FakeRequest(),
                BackgroundTasks(),
                video_file=_FakeUpload("demo.mp4", b"not-a-real-video"),
                temperature_file=(
                    _FakeUpload("temp.csv", temperature_data)
                    if temperature_data is not None
                    else None
                ),
                preset=preset,
                frame_stride=frame_stride,
                run_name="",
                direction_angle_deg=direction_angle_deg,
                initial_roi_xyxy=initial_roi_xyxy,
            )
        )

    def test_home_page_removes_sample_and_benchmark_entrypoints(self) -> None:
        with TemporaryDirectory() as tmp:
            with self._patched_storage(tmp):
                webapp._ensure_storage()
                response = webapp.home(_FakeRequest())
                response.body

        self.assertEqual(response.status_code, 200)
        text = response.body.decode("utf-8")
        self.assertIn("抽帧步长", text)
        self.assertNotIn("Benchmark 汇总", text)
        self.assertNotIn("直接运行这个示例", text)
        self.assertNotIn("分析模式", text)
        self.assertIn("开始分析", text)
        self.assertIn("细丝对象固定 ROI", text)
        self.assertIn("编织对象 ROI 与方向确认", text)
        self.assertIn("requiresRoiPreview", text)
        self.assertIn("requiresDirectionConfirmation", text)
        self.assertIn("先选固定 ROI，再确认方向", text)
        self.assertIn("该 ROI 将用于所有帧", text)
        self.assertIn('id="confirm_direction_button"', text)
        self.assertIn('confirmDirectionButton.addEventListener("click", confirmDirection)', text)
        self.assertIn("点击确认方向或按回车", text)
        self.assertIn("const MIN_ROI_SOURCE_PX = 2", text)
        self.assertIn("function sourceRoiFromBox", text)
        self.assertIn("sourceRoi.width < MIN_ROI_SOURCE_PX", text)
        self.assertNotIn("roiBox.x1 - roiBox.x0 < 0.01", text)
        self.assertLess(text.index("尚未选择固定 ROI"), text.index("方向角度：0°"))
        self.assertIn('name="initial_roi_xyxy"', text)
        self.assertIn("preview_roi", text)

    def test_home_page_references_editorial_asset_and_asset_route_serves_file(self) -> None:
        with TemporaryDirectory() as tmp:
            with self._patched_storage(tmp):
                webapp._ensure_storage()
                client = TestClient(webapp.app)
                response = client.get("/")
                asset_response = client.get("/assets/niti-editorial-hero.png")

        self.assertEqual(response.status_code, 200)
        self.assertIn("/assets/niti-editorial-hero.png", response.text)
        self.assertEqual(asset_response.status_code, 200)
        self.assertTrue(asset_response.headers["content-type"].startswith("image/"))

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

    def test_home_and_history_do_not_backfill_summaries_from_analysis_csv(self) -> None:
        with TemporaryDirectory() as tmp:
            with self._patched_storage(tmp):
                webapp._ensure_storage()
                run_id = "list-page-no-backfill"
                run_dir = webapp.RUNS_ROOT / run_id
                outputs_dir = run_dir / "outputs"
                outputs_dir.mkdir(parents=True, exist_ok=True)
                webapp._insert_run(
                    {
                        "id": run_id,
                        "created_at": webapp._utc_now(),
                        "status": "completed",
                        "run_name": "list no backfill",
                        "preset": "demo",
                        "requested_mode": "formal_af",
                        "frame_stride": 1,
                        "actual_mode": "formal_af",
                        "formal_metric_label": "kappa_fit",
                        "formal_gate_reason": None,
                        "af95_c": 52.8,
                        "aftan_c": 50.8,
                        "original_frame_count": 3,
                        "analyzed_frame_count": 3,
                        "annotated_video_filename": None,
                        "video_filename": "demo.mp4",
                        "temperature_filename": "temp.csv",
                        "run_dir": str(run_dir),
                        "error_text": None,
                    }
                )
                (outputs_dir / "summary.json").write_text(
                    '{"preset":"demo","requested_mode":"formal_af","actual_mode":"formal_af",'
                    '"formal_metric_label":"kappa_fit","route_results":[{"alias":"B","metric_key":"kappa_fit"}]}',
                    encoding="utf-8",
                )
                (outputs_dir / "analysis.csv").write_text(
                    "temperature_c,kappa_fit_recovery\n20,0\n60,1\n",
                    encoding="utf-8",
                )
                with (
                    mock.patch.object(webapp.pd, "read_csv", autospec=True) as read_csv,
                    mock.patch.object(webapp, "_write_summary", autospec=True) as write_summary,
                ):
                    home_response = webapp.home(_FakeRequest())
                    history_response = webapp.history(_FakeRequest())
                    home_response.body
                    history_response.body

        self.assertEqual(home_response.status_code, 200)
        self.assertEqual(history_response.status_code, 200)
        read_csv.assert_not_called()
        write_summary.assert_not_called()

    def test_history_page_supports_deleting_completed_runs(self) -> None:
        with TemporaryDirectory() as tmp:
            with self._patched_storage(tmp), mock.patch.object(webapp, "_execute_run", autospec=True):
                webapp._ensure_storage()
                client = TestClient(webapp.app)
                response = client.post(
                    "/runs",
                    data={
                        "preset": "wire_like",
                        "frame_stride": "1",
                        "run_name": "to-delete",
                        "initial_roi_xyxy": "[1,2,20,30]",
                    },
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
                self.assertIn("仅视频", history_response.text)
                self.assertNotIn("结果摘要", history_response.text)
                self.assertNotIn("路线 A", history_response.text)
                self.assertNotIn("路线 B", history_response.text)
                self.assertNotIn("正式 Af", history_response.text)
                self.assertNotIn("quicklook", history_response.text)
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
                    data={
                        "preset": "wire_like",
                        "frame_stride": "5",
                        "run_name": "real-run",
                        "initial_roi_xyxy": "[7,8,70,80]",
                    },
                    files={"video_file": ("demo.mp4", b"not-a-real-video", "video/mp4")},
                    follow_redirects=False,
                )
                rows = webapp._list_runs(limit=1)

        self.assertEqual(response.status_code, 303)
        self.assertEqual(len(rows), 1)
        row = dict(rows[0])
        self.assertEqual(row["requested_mode"], "quicklook")
        self.assertEqual(row["frame_stride"], 5)
        self.assertIsNone(row["direction_angle_deg"])
        self.assertEqual(row["direction_metric_enabled"], 0)
        self.assertEqual(row["initial_roi_xyxy"], "[7,8,70,80]")

    def test_create_wire_run_requires_initial_roi(self) -> None:
        with TemporaryDirectory() as tmp:
            with self._patched_storage(tmp), mock.patch.object(webapp, "_execute_run", autospec=True):
                webapp._ensure_storage()
                with self.assertRaises(webapp.HTTPException) as ctx:
                    self._create_run_direct(preset="wire_like", frame_stride="1")

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("initial_roi_xyxy", ctx.exception.detail)
        self.assertIn("wire_like", ctx.exception.detail)

    def test_create_wire_run_persists_initial_roi_without_direction(self) -> None:
        with TemporaryDirectory() as tmp:
            with self._patched_storage(tmp), mock.patch.object(webapp, "_execute_run", autospec=True):
                webapp._ensure_storage()
                response = self._create_run_direct(
                    preset="wire_like",
                    frame_stride="2",
                    direction_angle_deg="17",
                    initial_roi_xyxy="[12,24,220,260]",
                )
                rows = webapp._list_runs(limit=1)

        self.assertEqual(response.status_code, 303)
        row = dict(rows[0])
        self.assertEqual(row["initial_roi_xyxy"], "[12,24,220,260]")
        self.assertIsNone(row["direction_angle_deg"])
        self.assertEqual(row["direction_metric_enabled"], 0)

    def test_create_wire_run_clips_initial_roi_to_video_frame(self) -> None:
        with TemporaryDirectory() as tmp:
            with self._patched_storage(tmp), mock.patch.object(webapp, "_video_frame_size", return_value=(100, 80)):
                webapp._ensure_storage()
                response = self._create_run_direct(
                    preset="wire_like",
                    frame_stride="1",
                    initial_roi_xyxy="[10,20,150,90]",
                )
                rows = webapp._list_runs(limit=1)

        self.assertEqual(response.status_code, 303)
        self.assertEqual(dict(rows[0])["initial_roi_xyxy"], "[10,20,100,80]")

    def test_create_wire_run_rejects_roi_outside_video_frame(self) -> None:
        with TemporaryDirectory() as tmp:
            with self._patched_storage(tmp), mock.patch.object(webapp, "_video_frame_size", return_value=(100, 80)):
                webapp._ensure_storage()
                with self.assertRaises(webapp.HTTPException) as ctx:
                    self._create_run_direct(
                        preset="wire_like",
                        frame_stride="1",
                        initial_roi_xyxy="[120,20,150,40]",
                    )

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("overlap", ctx.exception.detail)

    def test_create_run_with_temperature_csv_derives_formal_request(self) -> None:
        with TemporaryDirectory() as tmp:
            with self._patched_storage(tmp), mock.patch.object(webapp, "_execute_run", autospec=True):
                webapp._ensure_storage()
                response = self._create_run_direct(
                    preset="braided_like",
                    frame_stride="2",
                    direction_angle_deg="9",
                    initial_roi_xyxy="[10,20,210,220]",
                    temperature_data=b"frame,temperature_c\n0,20\n",
                )
                rows = webapp._list_runs(limit=1)

        self.assertEqual(response.status_code, 303)
        self.assertEqual(dict(rows[0])["requested_mode"], "formal_af")

    def test_create_braided_run_persists_direction_angle(self) -> None:
        with TemporaryDirectory() as tmp:
            with self._patched_storage(tmp), mock.patch.object(webapp, "_execute_run", autospec=True):
                webapp._ensure_storage()
                response = self._create_run_direct(
                    preset="braided_like",
                    frame_stride="2",
                    direction_angle_deg="17",
                    initial_roi_xyxy="[12,24,220,260]",
                )
                rows = webapp._list_runs(limit=1)

        self.assertEqual(response.status_code, 303)
        row = dict(rows[0])
        self.assertEqual(row["direction_angle_deg"], 17.0)
        self.assertEqual(row["direction_metric_enabled"], 1)
        self.assertEqual(row["initial_roi_xyxy"], "[12,24,220,260]")

    def test_create_braided_run_requires_direction_angle(self) -> None:
        with TemporaryDirectory() as tmp:
            with self._patched_storage(tmp), mock.patch.object(webapp, "_execute_run", autospec=True):
                webapp._ensure_storage()
                with self.assertRaises(webapp.HTTPException) as ctx:
                    self._create_run_direct(
                        preset="braided_like",
                        frame_stride="1",
                        initial_roi_xyxy="[12,24,220,260]",
                    )

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("direction_angle_deg", ctx.exception.detail)
        self.assertIn("initial_roi_xyxy", ctx.exception.detail)

    def test_create_braided_run_requires_initial_roi(self) -> None:
        with TemporaryDirectory() as tmp:
            with self._patched_storage(tmp), mock.patch.object(webapp, "_execute_run", autospec=True):
                webapp._ensure_storage()
                with self.assertRaises(webapp.HTTPException) as ctx:
                    self._create_run_direct(
                        preset="braided_like",
                        frame_stride="1",
                        direction_angle_deg="4",
                    )

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("initial_roi_xyxy", ctx.exception.detail)

    def test_create_braided_run_requires_initial_roi_before_direction_angle(self) -> None:
        with TemporaryDirectory() as tmp:
            with self._patched_storage(tmp), mock.patch.object(webapp, "_execute_run", autospec=True):
                webapp._ensure_storage()
                with self.assertRaises(webapp.HTTPException) as ctx:
                    self._create_run_direct(
                        preset="braided_like",
                        frame_stride="1",
                    )

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("initial_roi_xyxy", ctx.exception.detail)
        self.assertNotIn("direction_angle_deg", ctx.exception.detail)

    def test_create_braided_run_rejects_invalid_initial_roi(self) -> None:
        with TemporaryDirectory() as tmp:
            with self._patched_storage(tmp), mock.patch.object(webapp, "_execute_run", autospec=True):
                webapp._ensure_storage()
                with self.assertRaises(webapp.HTTPException) as ctx:
                    self._create_run_direct(
                        preset="braided_like",
                        frame_stride="1",
                        direction_angle_deg="4",
                        initial_roi_xyxy="[20,20,20,40]",
                    )

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("initial_roi_xyxy", ctx.exception.detail)

    def test_ensure_storage_backfills_direction_columns(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            data_root = base / "var" / "webapp"
            runs_root = data_root / "runs"
            previews_root = data_root / "previews"
            db_path = data_root / "runs.db"
            data_root.mkdir(parents=True, exist_ok=True)
            runs_root.mkdir(parents=True, exist_ok=True)
            previews_root.mkdir(parents=True, exist_ok=True)
            with mock.patch.multiple(webapp, DATA_ROOT=data_root, RUNS_ROOT=runs_root, PREVIEWS_ROOT=previews_root, DB_PATH=db_path):
                with sqlite3.connect(db_path) as conn:
                    conn.execute(
                        """
                        CREATE TABLE runs (
                            id TEXT PRIMARY KEY,
                            created_at TEXT NOT NULL,
                            status TEXT NOT NULL,
                            run_name TEXT,
                            preset TEXT NOT NULL,
                            requested_mode TEXT NOT NULL,
                            actual_mode TEXT,
                            formal_metric_label TEXT,
                            formal_gate_reason TEXT,
                            af95_c REAL,
                            aftan_c REAL,
                            video_filename TEXT NOT NULL,
                            temperature_filename TEXT,
                            run_dir TEXT NOT NULL,
                            error_text TEXT
                        )
                        """
                    )
                    conn.commit()
                webapp._ensure_storage()
                with webapp._connect_db() as conn:
                    columns = {row["name"] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}

        self.assertIn("direction_angle_deg", columns)
        self.assertIn("direction_metric_enabled", columns)
        self.assertIn("initial_roi_xyxy", columns)

    def test_preview_video_endpoint_returns_file_url(self) -> None:
        with TemporaryDirectory() as tmp:
            with self._patched_storage(tmp), mock.patch.object(webapp, "_transcode_preview_video", autospec=True) as transcode_mock:
                webapp._ensure_storage()

                def _fake_transcode(source_path: Path, output_path: Path) -> Path:
                    output_path.write_bytes(source_path.read_bytes())
                    return output_path

                transcode_mock.side_effect = _fake_transcode
                client = TestClient(webapp.app)
                response = client.post(
                    "/preview-video",
                    files={"video_file": ("demo.mp4", b"fake-video", "video/mp4")},
                )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("/files/previews/", payload["preview_url"])

    def test_files_route_only_serves_safe_output_and_preview_media(self) -> None:
        with TemporaryDirectory() as tmp:
            with self._patched_storage(tmp):
                webapp._ensure_storage()
                run_id = "file-scope"
                outputs_dir = webapp.RUNS_ROOT / run_id / "outputs"
                inputs_dir = webapp.RUNS_ROOT / run_id / "inputs"
                preview_dir = webapp.PREVIEWS_ROOT / "preview-1"
                outputs_dir.mkdir(parents=True, exist_ok=True)
                inputs_dir.mkdir(parents=True, exist_ok=True)
                preview_dir.mkdir(parents=True, exist_ok=True)
                (outputs_dir / "plot.png").write_bytes(b"png")
                (outputs_dir / "summary.json").write_text("{}", encoding="utf-8")
                (inputs_dir / "source.mp4").write_bytes(b"input")
                (preview_dir / "preview.mp4").write_bytes(b"preview")
                (preview_dir / "preview_poster.jpg").write_bytes(b"poster")
                (preview_dir / "source.mp4").write_bytes(b"preview-source")
                webapp.DB_PATH.write_bytes(b"db")
                (webapp.DATA_ROOT / "debug.log").write_text("log", encoding="utf-8")

                allowed_output = webapp.serve_file(f"runs/{run_id}/outputs/plot.png")
                allowed_preview = webapp.serve_file("previews/preview-1/preview.mp4")
                allowed_preview_poster = webapp.serve_file("previews/preview-1/preview_poster.jpg")
                blocked = [
                    "runs.db",
                    "debug.log",
                    f"runs/{run_id}/inputs/source.mp4",
                    f"runs/{run_id}/outputs/summary.json",
                    "previews/preview-1/source.mp4",
                    "unknown/plot.png",
                    f"runs/{run_id}/outputs/../inputs/source.mp4",
                ]

                prefix_route_names = {getattr(route, "name", "") for route in webapp.app.routes}

        self.assertEqual(allowed_output.status_code, 200)
        self.assertEqual(allowed_preview.status_code, 200)
        self.assertEqual(allowed_preview_poster.status_code, 200)
        self.assertIn("files/niti", prefix_route_names)
        for path in blocked:
            with self.subTest(path=path), self.assertRaises(webapp.HTTPException) as ctx:
                webapp.serve_file(path)
            self.assertEqual(ctx.exception.status_code, 404)

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

    def test_wire_analysis_releases_capture_when_unexpected_error_escapes(self) -> None:
        class DummyCapture:
            def __init__(self) -> None:
                self.released = False
                self.index = 0

            def isOpened(self) -> bool:
                return True

            def get(self, _prop: int) -> float:
                return 10.0

            def read(self):
                if self.index:
                    return False, None
                self.index += 1
                return True, np.zeros((8, 8, 3), dtype=np.uint8)

            def release(self) -> None:
                self.released = True

        capture = DummyCapture()
        extraction = webapp._build_wire_extraction_config(
            {"analysis": {"demo_extraction": {"roi_xyxy": [0, 0, 4, 4], "blur_ksize": 1, "threshold_dark": 0, "open_kernel": 1, "close_kernel": 1}}},
            "demo",
        )
        with mock.patch("niti_bfr.pipeline.cv2.VideoCapture", return_value=capture), mock.patch(
            "niti_bfr.pipeline.extract_geometry",
            side_effect=ValueError("unexpected"),
        ):
            with self.assertRaises(ValueError):
                analyze_video("dummy.mp4", extraction=extraction)

        self.assertTrue(capture.released)

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
                        "initial_roi_xyxy": "[6,7,60,70]",
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

                with mock.patch.object(webapp, "analyze_video", return_value=fake_result) as analyze_mock, mock.patch.object(
                    webapp,
                    "render_process_debug_video",
                    side_effect=lambda **kwargs: Path(kwargs["output_path"]).write_bytes(b"process"),
                ) as render_mock, mock.patch.object(webapp, "_write_plots", autospec=True), mock.patch.object(
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
                self.assertEqual(summary["initial_roi_xyxy"], [6, 7, 60, 70])
                self.assertEqual(analyze_mock.call_args.kwargs["extraction"].roi_xyxy, (6, 7, 60, 70))
                self.assertEqual(render_mock.call_args.kwargs["extraction"].roi_xyxy, (6, 7, 60, 70))

    def test_execute_run_passes_retained_geometries_to_process_video_asset(self) -> None:
        cases = [
            (
                "wire_like",
                "analyze_video",
                pd.DataFrame({"frame": [0], "time_sec": [0.0], "quality": [0.9]}),
            ),
            (
                "braided_like",
                "analyze_braided_video_quicklook",
                pd.DataFrame({"frame": [0], "time_sec": [0.0], "quality": [0.9], "length_axis_px": [20.0]}),
            ),
        ]
        for preset, analyzer_name, series in cases:
            with self.subTest(preset=preset), TemporaryDirectory() as tmp:
                with self._patched_storage(tmp):
                    webapp._ensure_storage()
                    run_id = f"run-cached-geometry-{preset}"
                    run_dir = webapp.RUNS_ROOT / run_id
                    inputs_dir = run_dir / "inputs"
                    outputs_dir = run_dir / "outputs"
                    inputs_dir.mkdir(parents=True, exist_ok=True)
                    outputs_dir.mkdir(parents=True, exist_ok=True)
                    (inputs_dir / "demo.mp4").write_bytes(b"fake-video")
                    cached_geometries = {0: object()}

                    webapp._insert_run(
                        {
                            "id": run_id,
                            "created_at": webapp._utc_now(),
                            "status": "queued",
                            "run_name": "cached geometry",
                            "preset": preset,
                            "requested_mode": "quicklook",
                            "frame_stride": 1,
                            "direction_angle_deg": 15.0 if preset == "braided_like" else None,
                            "direction_metric_enabled": 1 if preset == "braided_like" else 0,
                            "initial_roi_xyxy": "[6,7,60,70]",
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
                        series=series,
                        fit=None,
                        af95_c=None,
                        aftan_c=None,
                        reportability_status="quicklook_only",
                        route_results=[],
                        input_fps=20.0,
                        original_frame_count=1,
                        analyzed_frame_count=1,
                        frame_stride=1,
                        frame_geometries=cached_geometries,
                    )

                    def fake_render(**kwargs):
                        Path(kwargs["output_path"]).write_bytes(b"process")

                    with (
                        mock.patch.object(webapp, analyzer_name, return_value=fake_result) as analyze_mock,
                        mock.patch.object(webapp, "compute_braided_acceptance", return_value={}),
                        mock.patch.object(webapp, "render_process_debug_video", side_effect=fake_render) as render_mock,
                        mock.patch.object(webapp, "_write_plots", autospec=True),
                        mock.patch.object(webapp, "route_results_dataframe", return_value=pd.DataFrame()),
                    ):
                        webapp._execute_run(run_id)

                    self.assertTrue(analyze_mock.call_args.kwargs["retain_frame_geometries"])
                    self.assertIs(render_mock.call_args.kwargs["result"].frame_geometries, cached_geometries)

    def test_execute_braided_run_uses_persisted_initial_roi(self) -> None:
        with TemporaryDirectory() as tmp:
            with self._patched_storage(tmp):
                webapp._ensure_storage()
                run_id = "run-braided-initial-roi"
                run_dir = webapp.RUNS_ROOT / run_id
                inputs_dir = run_dir / "inputs"
                outputs_dir = run_dir / "outputs"
                inputs_dir.mkdir(parents=True, exist_ok=True)
                outputs_dir.mkdir(parents=True, exist_ok=True)
                (inputs_dir / "braided.mp4").write_bytes(b"fake-video")

                webapp._insert_run(
                    {
                        "id": run_id,
                        "created_at": webapp._utc_now(),
                        "status": "queued",
                        "run_name": "braided roi",
                        "preset": "braided_like",
                        "requested_mode": "quicklook",
                        "frame_stride": 3,
                        "direction_angle_deg": 12.0,
                        "direction_metric_enabled": 1,
                        "initial_roi_xyxy": "[12,24,220,260]",
                        "actual_mode": None,
                        "formal_metric_label": None,
                        "formal_gate_reason": None,
                        "af95_c": None,
                        "aftan_c": None,
                        "original_frame_count": None,
                        "analyzed_frame_count": None,
                        "annotated_video_filename": None,
                        "video_filename": "braided.mp4",
                        "temperature_filename": None,
                        "run_dir": str(run_dir),
                        "error_text": None,
                    }
                )

                fake_result = AnalysisResult(
                    series=pd.DataFrame({"frame": [0], "quality": [0.9], "length_axis_px": [20.0]}),
                    fit=None,
                    af95_c=None,
                    aftan_c=None,
                    reportability_status="quicklook_only",
                    route_results=[],
                    input_fps=20.0,
                    original_frame_count=1,
                    analyzed_frame_count=1,
                    frame_stride=3,
                )

                with (
                    mock.patch.object(webapp, "analyze_braided_video_quicklook", return_value=fake_result) as analyze_mock,
                    mock.patch.object(webapp, "compute_braided_acceptance", return_value={}),
                    mock.patch.object(webapp, "_schedule_postprocess", autospec=True) as schedule_mock,
                    mock.patch.object(webapp, "route_results_dataframe", return_value=pd.DataFrame()),
                ):
                    webapp._execute_run(run_id)

                extraction = analyze_mock.call_args.kwargs["extraction"]
                summary = webapp._load_summary(run_id)
                assert summary is not None

        self.assertEqual(extraction.initial_roi_xyxy, (12, 24, 220, 260))
        self.assertNotEqual(extraction.roi_xyxy, (12, 24, 220, 260))
        self.assertEqual(summary["initial_roi_xyxy"], [12, 24, 220, 260])
        self.assertEqual(analyze_mock.call_args.kwargs["frame_stride"], 3)
        self.assertEqual(analyze_mock.call_args.kwargs["direction_angle_deg"], 12.0)
        self.assertEqual(schedule_mock.call_args.kwargs["extraction_cfg"].initial_roi_xyxy, (12, 24, 220, 260))


if __name__ == "__main__":
    unittest.main()
