from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from niti_bfr.extract_braided import BraidedExtractionConfig, BraidedTrackingState, _component_mask, extract_braided_geometry


class BraidedExtractionSelectionTests(unittest.TestCase):
    def test_component_mask_bridges_split_braided_body_before_largest_component(self) -> None:
        frame = np.full((80, 120, 3), 255, dtype=np.uint8)
        frame[20:60, 8:46] = 0
        frame[20:60, 52:104] = 0
        config = BraidedExtractionConfig(roi_xyxy=(0, 0, 120, 80), blur_ksize=1, threshold_dark=145, close_kernel=5)

        component, _ = _component_mask(frame, config)
        rows, cols = np.where(component > 0)

        self.assertGreater(len(rows), 0)
        self.assertLessEqual(int(cols.min()), 10)
        self.assertGreaterEqual(int(cols.max()), 101)

    def test_component_mask_uses_filled_outer_contour_instead_of_mesh_pixels(self) -> None:
        frame = np.full((120, 160, 3), 255, dtype=np.uint8)
        frame[30:90, 20:140] = 0
        frame[38:82, 28:132] = 255
        frame[30:90:8, 20:140] = 0
        frame[30:90, 20:140:10] = 0
        config = BraidedExtractionConfig(roi_xyxy=(0, 0, 160, 120), blur_ksize=1, threshold_dark=145, close_kernel=5)

        component, _ = _component_mask(frame, config)

        self.assertEqual(int(component[60, 80]), 255)
        self.assertGreater(int(np.count_nonzero(component)), 5000)

    def test_extract_braided_geometry_prefers_complete_base_roi_over_stale_tracking_crop(self) -> None:
        frame = np.zeros((32, 64, 3), dtype=np.uint8)
        config = BraidedExtractionConfig(roi_xyxy=(0, 0, 64, 32))
        tracked_roi = (20, 0, 64, 32)
        tracked_result = SimpleNamespace(
            length_axis_px=166.0,
            area_proj_px2=2428.0,
            attachment_count=2,
            endpoint_jump_px=184.0,
            centerline_disagreement=0.12,
            quality=0.777,
            tracking_state=BraidedTrackingState(tracked_roi),
        )
        base_result = SimpleNamespace(
            length_axis_px=357.0,
            area_proj_px2=5432.0,
            attachment_count=1,
            endpoint_jump_px=0.5,
            centerline_disagreement=0.08,
            quality=0.910,
            tracking_state=BraidedTrackingState(config.roi_xyxy),
        )

        def fake_extract(_frame_bgr, _config, *, roi_xyxy):
            if roi_xyxy == tracked_roi:
                return tracked_result
            return base_result

        with mock.patch("niti_bfr.extract_braided._extract_braided_geometry_once", side_effect=fake_extract):
            result = extract_braided_geometry(frame, config, tracking_state=BraidedTrackingState(tracked_roi))

        self.assertIs(result, base_result)

    def test_extract_braided_geometry_keeps_tracking_candidate_when_it_scores_best(self) -> None:
        frame = np.zeros((32, 64, 3), dtype=np.uint8)
        config = BraidedExtractionConfig(roi_xyxy=(0, 0, 64, 32))
        tracked_roi = (8, 0, 64, 32)
        tracked_result = SimpleNamespace(
            length_axis_px=360.0,
            area_proj_px2=5400.0,
            attachment_count=1,
            endpoint_jump_px=0.5,
            centerline_disagreement=0.04,
            quality=0.93,
            tracking_state=BraidedTrackingState(tracked_roi),
        )
        base_result = SimpleNamespace(
            length_axis_px=342.0,
            area_proj_px2=5200.0,
            attachment_count=1,
            endpoint_jump_px=3.0,
            centerline_disagreement=0.08,
            quality=0.87,
            tracking_state=BraidedTrackingState(config.roi_xyxy),
        )

        def fake_extract(_frame_bgr, _config, *, roi_xyxy):
            if roi_xyxy == tracked_roi:
                return tracked_result
            return base_result

        with mock.patch("niti_bfr.extract_braided._extract_braided_geometry_once", side_effect=fake_extract):
            result = extract_braided_geometry(frame, config, tracking_state=BraidedTrackingState(tracked_roi))

        self.assertIs(result, tracked_result)


if __name__ == "__main__":
    unittest.main()
