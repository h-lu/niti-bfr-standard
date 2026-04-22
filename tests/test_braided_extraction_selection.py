from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from niti_bfr.extract_braided import BraidedExtractionConfig, BraidedTrackingState, _component_mask, extract_braided_geometry
from niti_bfr.pipeline import analyze_braided_video_quicklook


class _DummyCapture:
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


class _DummyBraidedGeom:
    def __init__(self, frame_idx: int, tracking_state: BraidedTrackingState | None) -> None:
        self.tracking_state = tracking_state
        self.anchor_xy = np.array([1.0, 2.0])
        self.tip_xy = np.array([21.0 + frame_idx, 2.0])
        self.quality = 0.9
        self.length_env_px = 20.0 + frame_idx
        self.length_axis_px = 19.5 + frame_idx
        self.length_axis_skeleton_px = 19.4 + frame_idx
        self.length_axis_body_bins_px = 19.6 + frame_idx
        self.length_axis_alt_px = 19.3 + frame_idx
        self.length_axis_disagreement_px = 0.2
        self.centerline_disagreement = 0.01
        self.endpoint_gap_alt_centerline_px = 0.3
        self.diameter_max_px = 8.0 + frame_idx
        self.diameter_max_orth_px = 8.0 + frame_idx
        self.diameter_max_thickness_px = 7.8 + frame_idx
        self.diameter_max_feret_px = 8.2 + frame_idx
        self.diameter_p95_px = 7.6 + frame_idx
        self.diameter_mid_median_px = 7.0 + frame_idx
        self.diameter_mid_p90_px = 7.4 + frame_idx
        self.diameter_peak_span_px = 4.0
        self.diameter_peak_pos_norm = 0.5
        self.area_proj_px2 = 120.0 + frame_idx
        self.area_proj_contour_width_integral_px2 = 118.0 + frame_idx
        self.area_proj_contour_px2 = 116.0 + frame_idx
        self.area_proj_definition_gap_px2 = 4.0
        self.body_mask_area_px2 = 110.0
        self.component_area_px2 = 120.0
        self.excluded_attachment_area_px2 = 0.0
        self.body_mask_attachment_leak_fraction = 0.0
        self.attachment_count = 0
        self.attachment_border_touch_count = 0
        self.attachment_max_elongation = 0.0
        self.attachment_max_solidity = 0.0
        self.attachment_max_orientation_mismatch_deg = 0.0
        self.attachment_max_distance_to_main_axis_px = 0.0
        self.x_peak_norm = 0.5
        self.taper_left_px = 1.0
        self.taper_right_px = 1.0
        self.landing_zone_left_px = 2.0
        self.landing_zone_right_px = 2.0
        self.transition_zone_left_px = 3.0
        self.transition_zone_right_px = 3.0
        self.compaction_zone_length_px = 4.0
        self.zone_symmetry = 1.0
        self.branch_component_count_after_pruning = 1
        self.branch_count_after_pruning = 1
        self.endpoint_jump_px = 0.3
        self.axis_peak_position_stability = 0.0
        self.sampled_centerline_xy = np.array([[1.0, 2.0], [21.0, 2.0]])
        self.body_tube_mask = np.ones((4, 4), dtype=np.uint8)


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

    def test_extract_braided_geometry_uses_manual_roi_as_fixed_roi_without_tracking_state(self) -> None:
        frame = np.zeros((32, 64, 3), dtype=np.uint8)
        initial_roi = (10, 2, 42, 30)
        config = BraidedExtractionConfig(roi_xyxy=(0, 0, 64, 32), initial_roi_xyxy=initial_roi)
        initial_result = SimpleNamespace(
            length_axis_px=340.0,
            area_proj_px2=5200.0,
            attachment_count=1,
            endpoint_jump_px=0.5,
            centerline_disagreement=0.05,
            quality=0.91,
            tracking_state=BraidedTrackingState((8, 0, 48, 32)),
        )
        seen_rois: list[tuple[int, int, int, int]] = []

        def fake_extract(_frame_bgr, _config, *, roi_xyxy):
            seen_rois.append(roi_xyxy)
            return initial_result

        with mock.patch("niti_bfr.extract_braided._extract_braided_geometry_once", side_effect=fake_extract):
            result = extract_braided_geometry(frame, config)

        self.assertIs(result, initial_result)
        self.assertEqual(seen_rois, [initial_roi])

    def test_extract_braided_geometry_uses_manual_roi_as_fixed_roi(self) -> None:
        frame = np.zeros((32, 64, 3), dtype=np.uint8)
        initial_roi = (10, 2, 42, 30)
        tracked_roi = (8, 0, 48, 32)
        config = BraidedExtractionConfig(roi_xyxy=(0, 0, 64, 32), initial_roi_xyxy=initial_roi)
        initial_result = SimpleNamespace(
            length_axis_px=360.0,
            area_proj_px2=5400.0,
            attachment_count=1,
            endpoint_jump_px=0.5,
            centerline_disagreement=0.04,
            quality=0.93,
            tracking_state=BraidedTrackingState(initial_roi),
        )
        seen_rois: list[tuple[int, int, int, int]] = []

        def fake_extract(_frame_bgr, _config, *, roi_xyxy):
            seen_rois.append(roi_xyxy)
            return initial_result

        with mock.patch("niti_bfr.extract_braided._extract_braided_geometry_once", side_effect=fake_extract):
            result = extract_braided_geometry(frame, config, tracking_state=BraidedTrackingState(tracked_roi))

        self.assertIs(result, initial_result)
        self.assertEqual(seen_rois, [initial_roi])
        self.assertEqual(result.tracking_state, BraidedTrackingState(initial_roi))

    def test_analyze_braided_video_reuses_fixed_manual_roi_without_tracking_updates(self) -> None:
        frames = [np.zeros((32, 64, 3), dtype=np.uint8) for _ in range(3)]
        initial_roi = (10, 2, 42, 30)
        config = BraidedExtractionConfig(roi_xyxy=(0, 0, 64, 32), initial_roi_xyxy=initial_roi)
        seen_rois: list[tuple[int, int, int, int]] = []

        def fake_extract_once(_frame_bgr, _config, *, roi_xyxy):
            seen_rois.append(roi_xyxy)
            return _DummyBraidedGeom(
                len(seen_rois),
                tracking_state=BraidedTrackingState((0, 0, 1, 1)),
            )

        with mock.patch("niti_bfr.pipeline.cv2.VideoCapture", return_value=_DummyCapture(frames)), mock.patch(
            "niti_bfr.extract_braided._extract_braided_geometry_once",
            side_effect=fake_extract_once,
        ), mock.patch("niti_bfr.pipeline._next_braided_tracking_state") as next_state_mock:
            result = analyze_braided_video_quicklook("dummy.mp4", extraction=config)

        self.assertEqual(seen_rois, [initial_roi, initial_roi, initial_roi])
        next_state_mock.assert_not_called()
        self.assertEqual(result.series["frame"].tolist(), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
