from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import niti_bfr.annotated_overview_video as annotated_overview_video
import niti_bfr.process_debug_video as process_debug_video
from niti_bfr.extract_braided import BraidedExtractionConfig, BraidedTrackingState
from niti_bfr.extract import ExtractionConfig
from niti_bfr.annotated_overview_video import render_annotated_overview_video
from niti_bfr.pipeline import AnalysisResult
from niti_bfr.process_debug_video import render_process_debug_video


class _DummyCapture:
    def __init__(self, frame_shape: tuple[int, int, int]) -> None:
        self.frame_shape = frame_shape

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        if prop == 5:
            return 10.0
        if prop == 3:
            return float(self.frame_shape[1])
        if prop == 4:
            return float(self.frame_shape[0])
        return 0.0

    def release(self) -> None:
        return None


class _DummyWriter:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        return None


class ProcessDebugVideoTests(unittest.TestCase):
    def test_process_debug_renderer_reuses_cached_wire_geometry(self) -> None:
        frame_shape = (12, 20, 3)
        frames = [np.zeros(frame_shape, dtype=np.uint8) for _ in range(2)]
        cached_geometries = {
            0: SimpleNamespace(frame=0),
            1: SimpleNamespace(frame=1),
        }
        result = AnalysisResult(
            series=pd.DataFrame({"frame": [0, 1]}),
            fit=None,
            af95_c=None,
            aftan_c=None,
            frame_geometries=cached_geometries,
        )
        extraction = ExtractionConfig(roi_xyxy=(0, 0, 20, 12))
        writer = _DummyWriter()
        rendered_geometries: list[SimpleNamespace] = []

        def fake_render(**kwargs):
            rendered_geometries.append(kwargs["geom"])
            return np.zeros((12, 20 + 420, 3), dtype=np.uint8)

        with TemporaryDirectory() as tmp, mock.patch(
            "niti_bfr.process_debug_video.cv2.VideoCapture",
            return_value=_DummyCapture(frame_shape),
        ), mock.patch(
            "niti_bfr.process_debug_video._open_browser_compatible_writer",
            return_value=writer,
        ), mock.patch(
            "niti_bfr.process_debug_video._read_frame_at",
            side_effect=[(frames[0], 1), (frames[1], 2)],
        ), mock.patch(
            "niti_bfr.process_debug_video.extract_geometry",
        ) as extract_mock, mock.patch(
            "niti_bfr.process_debug_video._render_wire_debug_frame",
            side_effect=fake_render,
        ):
            render_process_debug_video(
                "dummy.mp4",
                Path(tmp) / "process.mp4",
                extraction=extraction,
                result=result,
                object_type="wire_like",
            )

        extract_mock.assert_not_called()
        self.assertEqual(rendered_geometries, [cached_geometries[0], cached_geometries[1]])
        self.assertEqual(len(writer.frames), 2)

    def test_process_debug_renderer_reuses_cached_braided_geometry(self) -> None:
        frame_shape = (12, 20, 3)
        frames = [np.zeros(frame_shape, dtype=np.uint8) for _ in range(2)]
        cached_geometries = {
            0: SimpleNamespace(tracking_state=BraidedTrackingState((1, 1, 18, 10))),
            1: SimpleNamespace(tracking_state=BraidedTrackingState((2, 2, 19, 11))),
        }
        result = AnalysisResult(
            series=pd.DataFrame({"frame": [0, 1]}),
            fit=None,
            af95_c=None,
            aftan_c=None,
            frame_geometries=cached_geometries,
        )
        extraction = BraidedExtractionConfig(roi_xyxy=(0, 0, 20, 12))
        writer = _DummyWriter()
        rendered_geometries: list[SimpleNamespace] = []

        def fake_render(**kwargs):
            rendered_geometries.append(kwargs["geom"])
            return np.zeros((12, 20 + 420, 3), dtype=np.uint8)

        with TemporaryDirectory() as tmp, mock.patch(
            "niti_bfr.process_debug_video.cv2.VideoCapture",
            return_value=_DummyCapture(frame_shape),
        ), mock.patch(
            "niti_bfr.process_debug_video._open_browser_compatible_writer",
            return_value=writer,
        ), mock.patch(
            "niti_bfr.process_debug_video._read_frame_at",
            side_effect=[(frames[0], 1), (frames[1], 2)],
        ), mock.patch(
            "niti_bfr.process_debug_video.extract_braided_geometry",
        ) as extract_mock, mock.patch(
            "niti_bfr.process_debug_video._render_braided_debug_frame",
            side_effect=fake_render,
        ), mock.patch(
            "niti_bfr.process_debug_video._next_braided_tracking_state",
        ) as next_state_mock:
            render_process_debug_video(
                "dummy.mp4",
                Path(tmp) / "process.mp4",
                extraction=extraction,
                result=result,
                object_type="braided_like",
            )

        extract_mock.assert_not_called()
        next_state_mock.assert_not_called()
        self.assertEqual(rendered_geometries, [cached_geometries[0], cached_geometries[1]])
        self.assertEqual(len(writer.frames), 2)

    def test_braided_renderer_reuses_geom_tracking_state(self) -> None:
        frame_shape = (12, 20, 3)
        frames = [np.zeros(frame_shape, dtype=np.uint8) for _ in range(2)]
        result = AnalysisResult(
            series=pd.DataFrame({"frame": [0, 1]}),
            fit=None,
            af95_c=None,
            aftan_c=None,
        )
        extraction = BraidedExtractionConfig(roi_xyxy=(0, 0, 20, 12))
        writer = _DummyWriter()
        tracked_state = BraidedTrackingState((2, 2, 18, 10))
        seen_tracking_states: list[BraidedTrackingState | None] = []

        def fake_extract(_frame_bgr, _extraction, tracking_state=None):
            seen_tracking_states.append(tracking_state)
            return SimpleNamespace(tracking_state=tracked_state)

        with TemporaryDirectory() as tmp, mock.patch(
            "niti_bfr.process_debug_video.cv2.VideoCapture",
            return_value=_DummyCapture(frame_shape),
        ), mock.patch(
            "niti_bfr.process_debug_video._open_browser_compatible_writer",
            return_value=writer,
        ), mock.patch(
            "niti_bfr.process_debug_video._read_frame_at",
            side_effect=[(frames[0], 1), (frames[1], 2)],
        ), mock.patch(
            "niti_bfr.process_debug_video.extract_braided_geometry",
            side_effect=fake_extract,
        ), mock.patch(
            "niti_bfr.process_debug_video._render_braided_debug_frame",
            return_value=np.zeros((12, 20 + 420, 3), dtype=np.uint8),
        ), mock.patch(
            "niti_bfr.process_debug_video._next_braided_tracking_state",
        ) as next_state_mock:
            render_process_debug_video(
                "dummy.mp4",
                Path(tmp) / "process.mp4",
                extraction=extraction,
                result=result,
            )

        self.assertEqual(seen_tracking_states, [None, tracked_state])
        next_state_mock.assert_not_called()
        self.assertEqual(len(writer.frames), 2)

    def test_braided_renderer_falls_back_to_pipeline_tracking_state(self) -> None:
        frame_shape = (12, 20, 3)
        frames = [np.zeros(frame_shape, dtype=np.uint8) for _ in range(2)]
        result = AnalysisResult(
            series=pd.DataFrame({"frame": [0, 1]}),
            fit=None,
            af95_c=None,
            aftan_c=None,
        )
        extraction = BraidedExtractionConfig(roi_xyxy=(0, 0, 20, 12))
        writer = _DummyWriter()
        fallback_state = BraidedTrackingState((1, 1, 19, 11))
        seen_tracking_states: list[BraidedTrackingState | None] = []

        def fake_extract(_frame_bgr, _extraction, tracking_state=None):
            seen_tracking_states.append(tracking_state)
            return SimpleNamespace(tracking_state=None)

        with TemporaryDirectory() as tmp, mock.patch(
            "niti_bfr.process_debug_video.cv2.VideoCapture",
            return_value=_DummyCapture(frame_shape),
        ), mock.patch(
            "niti_bfr.process_debug_video._open_browser_compatible_writer",
            return_value=writer,
        ), mock.patch(
            "niti_bfr.process_debug_video._read_frame_at",
            side_effect=[(frames[0], 1), (frames[1], 2)],
        ), mock.patch(
            "niti_bfr.process_debug_video.extract_braided_geometry",
            side_effect=fake_extract,
        ), mock.patch(
            "niti_bfr.process_debug_video._render_braided_debug_frame",
            return_value=np.zeros((12, 20 + 420, 3), dtype=np.uint8),
        ), mock.patch(
            "niti_bfr.process_debug_video._next_braided_tracking_state",
            return_value=fallback_state,
        ) as next_state_mock:
            render_process_debug_video(
                "dummy.mp4",
                Path(tmp) / "process.mp4",
                extraction=extraction,
                result=result,
            )

        self.assertEqual(seen_tracking_states, [None, fallback_state])
        self.assertEqual(next_state_mock.call_count, 2)
        self.assertEqual(len(writer.frames), 2)

    def test_braided_renderer_uses_fixed_initial_roi_without_tracking_updates(self) -> None:
        frame_shape = (12, 20, 3)
        frames = [np.zeros(frame_shape, dtype=np.uint8) for _ in range(2)]
        result = AnalysisResult(
            series=pd.DataFrame({"frame": [0, 1]}),
            fit=None,
            af95_c=None,
            aftan_c=None,
        )
        initial_roi = (3, 1, 17, 11)
        extraction = BraidedExtractionConfig(roi_xyxy=(0, 0, 20, 12), initial_roi_xyxy=initial_roi)
        writer = _DummyWriter()
        seen_tracking_states: list[BraidedTrackingState | None] = []
        rendered_rois: list[tuple[int, int, int, int]] = []

        def fake_extract(_frame_bgr, _extraction, tracking_state=None):
            seen_tracking_states.append(tracking_state)
            return SimpleNamespace(tracking_state=BraidedTrackingState((1, 1, 19, 11)))

        def fake_render(**kwargs):
            rendered_rois.append(kwargs["roi_xyxy"])
            return np.zeros((12, 20 + 420, 3), dtype=np.uint8)

        with TemporaryDirectory() as tmp, mock.patch(
            "niti_bfr.process_debug_video.cv2.VideoCapture",
            return_value=_DummyCapture(frame_shape),
        ), mock.patch(
            "niti_bfr.process_debug_video._open_browser_compatible_writer",
            return_value=writer,
        ), mock.patch(
            "niti_bfr.process_debug_video._read_frame_at",
            side_effect=[(frames[0], 1), (frames[1], 2)],
        ), mock.patch(
            "niti_bfr.process_debug_video.extract_braided_geometry",
            side_effect=fake_extract,
        ), mock.patch(
            "niti_bfr.process_debug_video._render_braided_debug_frame",
            side_effect=fake_render,
        ), mock.patch(
            "niti_bfr.process_debug_video._next_braided_tracking_state",
        ) as next_state_mock:
            render_process_debug_video(
                "dummy.mp4",
                Path(tmp) / "process.mp4",
                extraction=extraction,
                result=result,
            )

        self.assertEqual(seen_tracking_states, [None, None])
        self.assertEqual(rendered_rois, [initial_roi, initial_roi])
        next_state_mock.assert_not_called()
        self.assertEqual(len(writer.frames), 2)

    def test_wire_annotated_overview_uses_extraction_roi(self) -> None:
        frame_shape = (12, 20, 3)
        frames = [np.zeros(frame_shape, dtype=np.uint8) for _ in range(2)]
        result = AnalysisResult(
            series=pd.DataFrame({"frame": [0, 1], "quality": [0.9, 0.9]}),
            fit=None,
            af95_c=None,
            aftan_c=None,
            route_results=[],
        )
        extraction = ExtractionConfig(roi_xyxy=(3, 2, 17, 10))
        writer = _DummyWriter()
        seen_rois: list[tuple[int, int, int, int]] = []

        def fake_wire_overlay(overlay, _row, _header_lines, extraction_arg):
            seen_rois.append(extraction_arg.roi_xyxy)
            return overlay

        with TemporaryDirectory() as tmp, mock.patch(
            "niti_bfr.annotated_overview_video.cv2.VideoCapture",
            return_value=_DummyCapture(frame_shape),
        ), mock.patch(
            "niti_bfr.annotated_overview_video._open_browser_compatible_writer",
            return_value=writer,
        ), mock.patch(
            "niti_bfr.annotated_overview_video._read_frame_at",
            side_effect=[(frames[0], 1), (frames[1], 2)],
        ), mock.patch(
            "niti_bfr.annotated_overview_video._make_wire_overlay",
            side_effect=fake_wire_overlay,
        ):
            render_annotated_overview_video(
                "dummy.mp4",
                Path(tmp) / "overview.mp4",
                extraction=extraction,
                result=result,
                object_type="wire_like",
            )

        self.assertEqual(seen_rois, [(3, 2, 17, 10), (3, 2, 17, 10)])
        self.assertEqual(len(writer.frames), 2)

    def test_wire_debug_renderer_draws_route_b_as_fitted_curve(self) -> None:
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        extraction = ExtractionConfig(roi_xyxy=(2, 3, 30, 22))
        geom = SimpleNamespace(
            contour_xy=np.array([[4, 4], [4, 20], [28, 20], [28, 4]], dtype=float),
            sampled_centerline_xy=np.array([[6, 6], [10, 10], [14, 14]], dtype=float),
            fitted_curve_xy=np.array([[7, 5], [13, 9], [21, 12]], dtype=float),
            route_a_anchor_xy=np.array([5, 5], dtype=float),
            route_a_tip_xy=np.array([26, 18], dtype=float),
            anchor_xy=np.array([6, 6], dtype=float),
            tip_xy=np.array([24, 17], dtype=float),
        )
        row = SimpleNamespace(
            frame=0,
            time_sec=0.0,
            temperature_c=20.0,
            x_route_a_px=20.0,
            x_route_a_recovery=0.1,
            x_fit_px=19.0,
            kappa_fit_px_inv=0.02,
            x_route_c_px=18.0,
            kappa_route_c_px_inv=0.01,
            quality=0.9,
            model_name="quadratic",
        )
        series = pd.DataFrame(
            {
                "frame": [0],
                "x_route_a_recovery": [0.1],
                "kappa_fit_recovery": [0.2],
                "kappa_route_c_recovery": [0.3],
            }
        )
        polyline_calls: list[np.ndarray] = []

        def capture_polylines(_image, pts, *args, **kwargs):
            polyline_calls.append(np.array(pts[0], copy=True))
            return None

        with mock.patch.object(process_debug_video.cv2, "polylines", side_effect=capture_polylines), mock.patch.object(
            process_debug_video, "_make_canvas", side_effect=lambda overlay: overlay
        ), mock.patch.object(process_debug_video, "_draw_text_block"), mock.patch.object(
            process_debug_video, "_draw_trend_plot"
        ):
            process_debug_video._render_wire_debug_frame(
                frame_bgr=frame,
                geom=geom,
                row=row,
                series=series,
                frame_idx=0,
                extraction=extraction,
            )

        self.assertGreaterEqual(len(polyline_calls), 2)
        np.testing.assert_array_equal(
            polyline_calls[1],
            np.round(geom.fitted_curve_xy).astype(np.int32).reshape(-1, 1, 2),
        )

    def test_wire_annotated_overview_labels_route_b_as_shape_fit(self) -> None:
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        extraction = ExtractionConfig(roi_xyxy=(2, 3, 30, 22))
        geom = SimpleNamespace(
            contour_xy=np.array([[4, 4], [4, 20], [28, 20], [28, 4]], dtype=float),
            fitted_curve_xy=np.array([[7, 5], [13, 9], [21, 12]], dtype=float),
            route_a_anchor_xy=np.array([5, 5], dtype=float),
            route_a_tip_xy=np.array([26, 18], dtype=float),
            anchor_xy=np.array([6, 6], dtype=float),
            tip_xy=np.array([24, 17], dtype=float),
        )
        row = SimpleNamespace(frame=0, time_sec=0.0)
        legend_entries: list[tuple[str, str, tuple[int, int, int]]] = []

        def capture_legend(_overlay, entries):
            legend_entries.extend(entries)

        with mock.patch.object(annotated_overview_video, "extract_geometry", return_value=geom), mock.patch.object(
            annotated_overview_video, "_draw_legend_box", side_effect=capture_legend
        ), mock.patch.object(annotated_overview_video, "_draw_header_badge"):
            annotated_overview_video._make_wire_overlay(frame.copy(), row, [], extraction)

        self.assertIn(("B", "shape fit / 拟合主线", annotated_overview_video.ROUTE_COLORS["B"]), legend_entries)

    def test_wire_annotated_overview_reuses_cached_geometry(self) -> None:
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        extraction = ExtractionConfig(roi_xyxy=(2, 3, 30, 22))
        geom = SimpleNamespace(
            contour_xy=np.array([[4, 4], [4, 20], [28, 20], [28, 4]], dtype=float),
            fitted_curve_xy=np.array([[7, 5], [13, 9], [21, 12]], dtype=float),
            route_a_anchor_xy=np.array([5, 5], dtype=float),
            route_a_tip_xy=np.array([26, 18], dtype=float),
            anchor_xy=np.array([6, 6], dtype=float),
            tip_xy=np.array([24, 17], dtype=float),
        )
        row = SimpleNamespace(frame=0, time_sec=0.0)

        with mock.patch.object(annotated_overview_video, "extract_geometry") as extract_mock, mock.patch.object(
            annotated_overview_video, "_draw_legend_box"
        ), mock.patch.object(annotated_overview_video, "_draw_header_badge"):
            annotated_overview_video._make_wire_overlay(frame.copy(), row, [], extraction, geom)

        extract_mock.assert_not_called()

    def test_braided_annotated_overview_uses_cached_geometry_source_roi_for_mask(self) -> None:
        frame = np.zeros((12, 20, 3), dtype=np.uint8)
        extraction = BraidedExtractionConfig(roi_xyxy=(0, 0, 20, 12))
        source_roi = (5, 2, 9, 6)
        geom = SimpleNamespace(
            source_roi_xyxy=source_roi,
            body_tube_mask=np.ones((4, 4), dtype=np.uint8),
            contour_xy=np.array([[5, 2], [5, 5], [8, 5], [8, 2]], dtype=float),
            body_contour_xy=np.array([[5, 2], [5, 5], [8, 5], [8, 2]], dtype=float),
            sampled_centerline_xy=np.array([[5, 3], [8, 3]], dtype=float),
            sampled_width_segments_xy=np.empty((0, 2, 2), dtype=float),
            anchor_xy=np.array([5, 3], dtype=float),
            tip_xy=np.array([8, 3], dtype=float),
        )
        row = SimpleNamespace(frame=0, time_sec=0.0)
        seen_rois: list[tuple[int, int, int, int]] = []

        def capture_blend(frame_bgr, mask, roi_xyxy, color_bgr, *, alpha):
            seen_rois.append(roi_xyxy)
            self.assertEqual(mask.shape, (4, 4))
            return frame_bgr

        with mock.patch.object(annotated_overview_video, "extract_braided_geometry") as extract_mock, mock.patch.object(
            annotated_overview_video, "_blend_mask_on_roi", side_effect=capture_blend
        ), mock.patch.object(annotated_overview_video, "_draw_legend_box"), mock.patch.object(
            annotated_overview_video, "_draw_header_badge"
        ):
            annotated_overview_video._make_braided_overlay(frame.copy(), row, [], extraction, geom)

        extract_mock.assert_not_called()
        self.assertEqual(seen_rois, [source_roi])

    def test_braided_annotated_overview_falls_back_to_extraction_roi_for_old_cached_geometry(self) -> None:
        frame = np.zeros((12, 20, 3), dtype=np.uint8)
        extraction = BraidedExtractionConfig(roi_xyxy=(0, 0, 20, 12))
        geom = SimpleNamespace(
            body_tube_mask=np.ones((12, 20), dtype=np.uint8),
            contour_xy=np.array([[1, 1], [1, 10], [18, 10], [18, 1]], dtype=float),
            body_contour_xy=np.array([[1, 1], [1, 10], [18, 10], [18, 1]], dtype=float),
            sampled_centerline_xy=np.array([[1, 5], [18, 5]], dtype=float),
            sampled_width_segments_xy=np.empty((0, 2, 2), dtype=float),
            anchor_xy=np.array([1, 5], dtype=float),
            tip_xy=np.array([18, 5], dtype=float),
        )
        row = SimpleNamespace(frame=0, time_sec=0.0)
        seen_rois: list[tuple[int, int, int, int]] = []

        def capture_blend(frame_bgr, _mask, roi_xyxy, _color_bgr, *, alpha):
            seen_rois.append(roi_xyxy)
            return frame_bgr

        with mock.patch.object(annotated_overview_video, "_blend_mask_on_roi", side_effect=capture_blend), mock.patch.object(
            annotated_overview_video, "_draw_legend_box"
        ), mock.patch.object(annotated_overview_video, "_draw_header_badge"):
            annotated_overview_video._make_braided_overlay(frame.copy(), row, [], extraction, geom)

        self.assertEqual(seen_rois, [extraction.roi_xyxy])


if __name__ == "__main__":
    unittest.main()
