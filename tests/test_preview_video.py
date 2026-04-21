from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from niti_bfr.webapp import _transcode_preview_video


class PreviewVideoTests(unittest.TestCase):
    def test_transcode_preview_delegates_to_ffmpeg_transcoder(self) -> None:
        with TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source.mp4"
            output_path = Path(tmp) / "preview.mp4"
            rendered_path = output_path.with_suffix(".webm")
            source_path.write_bytes(b"dummy")
            with mock.patch(
                "niti_bfr.webapp.transcode_video_for_browser",
                return_value=rendered_path,
            ) as transcode_mock:
                actual_path = _transcode_preview_video(source_path, output_path)

        self.assertEqual(actual_path, rendered_path)
        transcode_mock.assert_called_once_with(source_path, output_path, max_width=960, max_fps=15.0)


if __name__ == "__main__":
    unittest.main()
