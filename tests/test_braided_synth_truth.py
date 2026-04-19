from __future__ import annotations

import unittest

import numpy as np

from niti_bfr.extract_braided import compute_braided_body_mask, cumulative_path_length
from niti_bfr.synth_braided import _truth_metrics_from_geometry


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

        body_mask = compute_braided_body_mask(width_profile_px)
        curve_positions_px = cumulative_path_length(centerline_xy)
        expected_axis_span = float(curve_positions_px[body_mask][-1] - curve_positions_px[body_mask][0])
        expected_rel_positions = curve_positions_px - float(curve_positions_px[body_mask][0])
        expected_area = float(np.trapezoid(width_profile_px[body_mask], expected_rel_positions[body_mask]))

        self.assertLess(expected_axis_span, float(curve_positions_px[-1]))
        self.assertTrue(np.isclose(truth["length_axis_true_px"], expected_axis_span))
        self.assertTrue(np.isclose(truth["area_proj_true_px2"], expected_area))


if __name__ == "__main__":
    unittest.main()
