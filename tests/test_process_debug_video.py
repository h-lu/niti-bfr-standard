from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from niti_bfr.extract_braided import BraidedExtractionConfig, BraidedTrackingState
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


if __name__ == "__main__":
    unittest.main()
