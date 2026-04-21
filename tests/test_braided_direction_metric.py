from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from niti_bfr.extract_braided import BraidedExtractionConfig, compute_directional_span_from_mask
from niti_bfr.pipeline import analyze_braided_video_quicklook


class BraidedDirectionMetricTests(unittest.TestCase):
    def test_compute_directional_span_from_mask_tracks_projection_span(self) -> None:
        mask = np.zeros((6, 12), dtype=np.uint8)
        mask[1:5, 2:10] = 255

        self.assertAlmostEqual(compute_directional_span_from_mask(mask, 0.0), 7.0)
        self.assertAlmostEqual(compute_directional_span_from_mask(mask, 90.0), 3.0)

    def test_analyze_braided_video_populates_direction_result(self) -> None:
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
                width = 4 + frame_idx
                self.source_roi_xyxy = (0, 0, 20, 12)
                self.anchor_xy = np.array([1.0, 1.0])
                self.tip_xy = np.array([10.0, 1.0])
                self.quality = 0.9
                self.length_env_px = 24.0 - frame_idx
                self.length_axis_px = 20.0 - frame_idx
                self.length_axis_skeleton_px = 20.0 - frame_idx
                self.length_axis_body_bins_px = 20.0 - frame_idx
                self.length_axis_alt_px = 20.0 - frame_idx + 0.1
                self.length_axis_disagreement_px = 0.2
                self.centerline_disagreement = 0.01
                self.endpoint_gap_alt_centerline_px = 0.3
                self.diameter_max_px = 5.0 + 0.4 * frame_idx
                self.diameter_max_orth_px = 5.0 + 0.4 * frame_idx
                self.diameter_max_thickness_px = 5.0 + 0.4 * frame_idx
                self.diameter_max_feret_px = 5.0 + 0.4 * frame_idx
                self.diameter_p95_px = 4.0 + 0.3 * frame_idx
                self.diameter_mid_median_px = 4.0 + 0.3 * frame_idx
                self.diameter_mid_p90_px = 4.0 + 0.3 * frame_idx
                self.diameter_peak_span_px = 3.0 + 0.1 * frame_idx
                self.diameter_peak_pos_norm = 0.5
                self.area_proj_px2 = 50.0 + 4.0 * frame_idx
                self.area_proj_contour_width_integral_px2 = 50.0 + 4.0 * frame_idx
                self.area_proj_contour_px2 = 50.0 + 4.0 * frame_idx
                self.area_proj_definition_gap_px2 = 0.5
                self.body_mask_area_px2 = 50.0 + 4.0 * frame_idx
                self.component_area_px2 = 50.0 + 4.0 * frame_idx
                self.excluded_attachment_area_px2 = 0.0
                self.body_mask_attachment_leak_fraction = 0.0
                self.attachment_count = 0
                self.attachment_border_touch_count = 0
                self.attachment_max_elongation = 0.0
                self.attachment_max_solidity = 1.0
                self.attachment_max_orientation_mismatch_deg = 0.0
                self.attachment_max_distance_to_main_axis_px = 0.0
                self.x_peak_norm = 0.5
                self.taper_left_px = 1.0
                self.taper_right_px = 1.0
                self.landing_zone_left_px = 1.0
                self.landing_zone_right_px = 1.0
                self.transition_zone_left_px = 1.0
                self.transition_zone_right_px = 1.0
                self.compaction_zone_length_px = 2.0
                self.zone_symmetry = 1.0
                self.branch_component_count_after_pruning = 0
                self.branch_count_after_pruning = 0
                self.endpoint_jump_px = 0.2
                self.axis_peak_position_stability = 0.01
                self.sampled_centerline_xy = np.array([[1.0, 1.0], [10.0, 1.0]])
                self.body_tube_mask = np.zeros((12, 20), dtype=np.uint8)
                self.body_tube_mask[3:9, 4 : 4 + width] = 255

        frames = [np.zeros((12, 20, 3), dtype=np.uint8) for _ in range(8)]
        extraction = BraidedExtractionConfig(roi_xyxy=(0, 0, 20, 12))
        with TemporaryDirectory() as tmp:
            temperature_csv = Path(tmp) / "temperature.csv"
            pd.DataFrame(
                {
                    "frame": list(range(8)),
                    "temperature_c": np.linspace(20.0, 80.0, 8),
                }
            ).to_csv(temperature_csv, index=False)
            with mock.patch("niti_bfr.pipeline.cv2.VideoCapture", return_value=DummyCapture(frames)), mock.patch(
                "niti_bfr.pipeline.extract_braided_geometry",
                side_effect=[DummyGeom(idx) for idx in range(8)],
            ):
                result = analyze_braided_video_quicklook(
                    "dummy.mp4",
                    extraction=extraction,
                    temperature_csv=temperature_csv,
                    direction_angle_deg=0.0,
                )

        self.assertIn("direction_span_px", result.series.columns)
        self.assertIn("direction_recovery", result.series.columns)
        self.assertTrue(result.series["direction_span_px"].notna().all())
        self.assertLess(float(result.series["direction_recovery"].iloc[0]), 0.5)
        self.assertGreater(float(result.series["direction_recovery"].iloc[-1]), 0.5)
        self.assertLess(
            float(result.series["direction_recovery"].iloc[0]),
            float(result.series["direction_recovery"].iloc[-1]),
        )
        self.assertIsNotNone(result.direction_result)
        assert result.direction_result is not None
        self.assertTrue(result.direction_result["enabled"])
        self.assertEqual(result.direction_result["metric_key"], "direction_span")
        self.assertEqual(result.direction_result["reportability_status"], "formal_passed")
        self.assertIsNotNone(result.direction_result["af95_c"])
