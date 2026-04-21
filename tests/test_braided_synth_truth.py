from __future__ import annotations

import unittest

import numpy as np

from niti_bfr.extract_braided import cumulative_path_length, rasterize_braided_body_tube_mask
from niti_bfr.synth_braided import _truth_metrics_from_geometry, truth_metrics_from_body_mask


class BraidedSynthTruthTests(unittest.TestCase):
    def test_truth_metrics_use_body_only_axis_span_and_area_definition(self) -> None:
        centerline_xy = np.column_stack([np.arange(0.0, 70.0, 10.0), np.full(7, 40.0)])
        width_profile_px = np.array([4.0, 9.0, 18.0, 24.0, 18.0, 9.0, 4.0], dtype=float)
        radius = 0.5 * width_profile_px
        upper = np.column_stack([centerline_xy[:, 0], centerline_xy[:, 1] - radius])
        lower = np.column_stack([centerline_xy[:, 0], centerline_xy[:, 1] + radius])
        contour_xy = np.vstack([upper, lower[::-1]])

        truth = _truth_metrics_from_geometry(
            contour_xy=contour_xy,
            centerline_xy=centerline_xy,
            width_profile_px=width_profile_px,
            taper_threshold_ratio=0.25,
            compaction_threshold_ratio=0.75,
            image_shape=(96, 96),
        )

        curve_positions_px = cumulative_path_length(centerline_xy)
        expected_axis_span = float(curve_positions_px[-1] - curve_positions_px[0])
        expected_rel_positions = curve_positions_px - float(curve_positions_px[0])
        expected_area = float(np.trapezoid(width_profile_px, expected_rel_positions))

        self.assertTrue(np.isclose(truth["length_axis_true_px"], expected_axis_span))
        self.assertTrue(np.isclose(truth["area_proj_true_px2"], expected_area))

    def test_truth_metrics_from_body_mask_recovers_a_b_c_geometry(self) -> None:
        centerline_xy = np.column_stack([np.arange(12.0, 84.0, 12.0), np.full(6, 36.0)])
        width_profile_px = np.array([8.0, 14.0, 20.0, 20.0, 14.0, 8.0], dtype=float)
        body_mask = np.ones(len(width_profile_px), dtype=bool)
        raster = rasterize_braided_body_tube_mask(
            centerline_xy=centerline_xy,
            width_profile_px=width_profile_px,
            body_mask=body_mask,
            image_shape=(96, 96),
        )

        truth = truth_metrics_from_body_mask(
            raster,
            taper_threshold_ratio=0.25,
            compaction_threshold_ratio=0.75,
            width_sampling_step_px=4.0,
            centerline_smooth_window=5,
        )

        self.assertGreater(truth["length_axis_true_px"], 0.0)
        self.assertGreater(truth["diameter_true_px"], 0.0)
        self.assertGreater(truth["area_proj_true_px2"], 0.0)
        self.assertAlmostEqual(truth["body_mask_area_true_px2"], float(np.count_nonzero(raster > 0)))
        self.assertLess(abs(truth["diameter_true_px"] - float(np.max(width_profile_px))), 4.0)


if __name__ == "__main__":
    unittest.main()
