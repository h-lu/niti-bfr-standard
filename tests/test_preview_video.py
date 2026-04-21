from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import numpy as np

from niti_bfr.process_debug_video import BrowserCompatibleWriter
from niti_bfr.webapp import _transcode_preview_video


class _DummyCapture:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = frames
        self.index = 0

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        if prop == 5:
            return 30.0
        if prop == 3:
            return float(self.frames[0].shape[1])
        if prop == 4:
            return float(self.frames[0].shape[0])
        return 0.0

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def set(self, _prop: int, _value: float) -> bool:
        return True

    def release(self) -> None:
        return None


class _DummyWriter:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.released = False

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        self.released = True

    def isOpened(self) -> bool:
        return True


class PreviewVideoTests(unittest.TestCase):
    def test_transcode_preview_uses_browser_writer_proxy_and_returns_actual_path(self) -> None:
        frames = [np.zeros((12, 20, 3), dtype=np.uint8) for _ in range(3)]
        wrapped_writer = _DummyWriter()

        with TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "preview.mp4"
            actual_output_path = output_path.with_suffix(".webm")
            with mock.patch("niti_bfr.webapp.cv2.VideoCapture", return_value=_DummyCapture(frames)), mock.patch(
                "niti_bfr.webapp._open_browser_compatible_writer",
                return_value=BrowserCompatibleWriter(writer=wrapped_writer, output_path=actual_output_path),
            ):
                rendered_path = _transcode_preview_video(Path("source.mp4"), output_path)

        self.assertEqual(rendered_path, actual_output_path)
        self.assertEqual(len(wrapped_writer.frames), 2)
        self.assertTrue(wrapped_writer.released)


if __name__ == "__main__":
    unittest.main()
