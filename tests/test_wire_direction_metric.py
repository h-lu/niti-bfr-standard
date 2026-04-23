from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from niti_bfr import webapp
from niti_bfr.extract import ExtractionConfig
from niti_bfr.pipeline import analyze_video


class WireDirectionMetricTests(unittest.TestCase):
    def test_analyze_wire_video_populates_two_direction_methods_and_plots(self) -> None:
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
                span = 5.0 + frame_idx
                self.route_a_anchor_xy = np.array([0.0, 0.0])
                self.route_a_tip_xy = np.array([span, 1.0])
                self.x_route_a_px = span
                self.anchor_xy = np.array([0.0, 0.0])
                self.tip_xy = np.array([span, 1.0])
                self.x_px = span
                self.x_fit_px = span
                self.kappa_fit_px_inv = 0.20 - 0.015 * frame_idx
                self.quality = 0.95
                self.sampled_centerline_xy = np.array([[2.0, 4.0], [2.0 + span, 4.0]])
                self.fit_samples_xy = self.sampled_centerline_xy
                self.fitted_curve_xy = self.sampled_centerline_xy
                self.contour_xy = np.array(
                    [[2.0, 3.0], [2.0 + span, 3.0], [2.0 + span, 5.0], [2.0, 5.0]]
                )
                self.mask = np.zeros((10, 20), dtype=np.uint8)
                width = int(round(span)) + 1
                self.mask[3:6, 2 : 2 + width] = 255
                self.quadratic_rmse_px = 0.1
                self.circle_rmse_px = 0.2
                self.model_name = "quadratic"

        frames = [np.zeros((10, 20, 3), dtype=np.uint8) for _ in range(8)]
        extraction = ExtractionConfig(roi_xyxy=(0, 0, 20, 10), blur_ksize=1, open_kernel=1, close_kernel=1)
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            temperature_csv = tmp_path / "temperature.csv"
            pd.DataFrame(
                {
                    "frame": list(range(8)),
                    "temperature_c": np.linspace(20.0, 80.0, 8),
                }
            ).to_csv(temperature_csv, index=False)
            with mock.patch("niti_bfr.pipeline.cv2.VideoCapture", return_value=DummyCapture(frames)), mock.patch(
                "niti_bfr.pipeline.extract_geometry",
                side_effect=[DummyGeom(idx) for idx in range(8)],
            ):
                result = analyze_video(
                    "dummy.mp4",
                    extraction=extraction,
                    temperature_csv=temperature_csv,
                    direction_angle_deg=0.0,
                )

            self.assertIn("direction_centerline_span_px", result.series.columns)
            self.assertIn("direction_mask_span_px", result.series.columns)
            self.assertIn("direction_centerline_recovery", result.series.columns)
            self.assertIn("direction_mask_recovery", result.series.columns)
            self.assertTrue(result.series["direction_centerline_span_px"].notna().all())
            self.assertTrue(result.series["direction_mask_span_px"].notna().all())
            self.assertIsNotNone(result.direction_result)
            self.assertEqual(len(result.direction_results or []), 2)
            metric_keys = {entry["metric_key"] for entry in result.direction_results or []}
            self.assertEqual(metric_keys, {"direction_centerline_span", "direction_mask_span"})
            for entry in result.direction_results or []:
                self.assertTrue(entry["enabled"])
                self.assertEqual(entry["reportability_status"], "formal_passed")
                self.assertIsNotNone(entry["af95_c"])

            webapp._write_plots(tmp_path, result)
            self.assertTrue((tmp_path / "direction_metric_over_time.png").exists())
            self.assertTrue((tmp_path / "direction_recovery_vs_temperature.png").exists())
            self.assertTrue((tmp_path / "direction_centerline_metric_over_time.png").exists())
            self.assertTrue((tmp_path / "direction_centerline_recovery_vs_temperature.png").exists())
            self.assertTrue((tmp_path / "direction_mask_metric_over_time.png").exists())
            self.assertTrue((tmp_path / "direction_mask_recovery_vs_temperature.png").exists())


if __name__ == "__main__":
    unittest.main()
