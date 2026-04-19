from __future__ import annotations

import unittest

import numpy as np

from niti_bfr.extract_braided import _endpoint_gap, _select_centerline_paths


class BraidedCenterlineSelectionTests(unittest.TestCase):
    def test_select_centerline_paths_prefers_consensus_length(self) -> None:
        mask = np.ones((32, 64), dtype=np.uint8) * 255
        candidates = {
            "skeleton": np.array([[8.0, 16.0], [18.0, 16.0], [28.0, 16.0], [38.0, 16.0]], dtype=float),
            "body_bins": np.array([[8.0, 16.0], [19.0, 16.0], [30.0, 16.0], [41.0, 16.0], [52.0, 16.0]], dtype=float),
            "prior": np.array([[8.0, 16.0], [20.0, 16.0], [32.0, 16.0], [44.0, 16.0], [56.0, 16.0]], dtype=float),
        }

        primary_name, primary_xy, secondary_name, secondary_xy = _select_centerline_paths(mask, candidates)

        self.assertEqual(primary_name, "body_bins")
        self.assertIn(secondary_name, {"skeleton", "prior"})
        self.assertTrue(np.array_equal(primary_xy, candidates["body_bins"]))
        self.assertTrue(np.array_equal(secondary_xy, candidates[secondary_name]))

    def test_endpoint_gap_ignores_path_direction(self) -> None:
        primary = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]], dtype=float)
        reversed_secondary = primary[::-1].copy()

        self.assertAlmostEqual(_endpoint_gap(primary, reversed_secondary), 0.0)

    def test_endpoint_gap_uses_shared_span_when_one_path_is_truncated(self) -> None:
        full_span = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0], [15.0, 0.0], [20.0, 0.0]], dtype=float)
        truncated = np.array([[5.0, 0.0], [10.0, 0.0], [15.0, 0.0]], dtype=float)

        self.assertAlmostEqual(_endpoint_gap(full_span, truncated), 0.0)

    def test_select_centerline_paths_aligns_secondary_orientation_to_primary(self) -> None:
        mask = np.ones((32, 64), dtype=np.uint8) * 255
        candidates = {
            "body_bins": np.array([[8.0, 16.0], [20.0, 16.0], [32.0, 16.0], [44.0, 16.0]], dtype=float),
            "skeleton": np.array([[44.0, 16.0], [32.0, 16.0], [20.0, 16.0], [8.0, 16.0]], dtype=float),
            "prior": np.array([[8.0, 15.5], [20.0, 15.5], [32.0, 15.5], [44.0, 15.5]], dtype=float),
        }

        _, primary_xy, secondary_name, secondary_xy = _select_centerline_paths(mask, candidates)

        self.assertIn(secondary_name, {"skeleton", "prior"})
        self.assertLess(np.linalg.norm(primary_xy[0] - secondary_xy[0]), 1.0)
        self.assertLess(np.linalg.norm(primary_xy[-1] - secondary_xy[-1]), 1.0)

    def test_select_centerline_paths_rejects_low_span_skeleton_even_if_length_matches(self) -> None:
        mask = np.ones((64, 96), dtype=np.uint8) * 255
        candidates = {
            "body_bins": np.array([[8.0, 32.0], [28.0, 31.0], [48.0, 31.0], [68.0, 32.0], [88.0, 33.0]], dtype=float),
            "skeleton": np.array([[38.0, 44.0], [42.0, 36.0], [46.0, 28.0], [50.0, 20.0], [54.0, 12.0]], dtype=float),
            "prior": np.array([[10.0, 31.5], [30.0, 31.0], [50.0, 31.0], [70.0, 31.5], [90.0, 32.0]], dtype=float),
        }

        primary_name, primary_xy, secondary_name, secondary_xy = _select_centerline_paths(mask, candidates)

        self.assertEqual(primary_name, "body_bins")
        self.assertEqual(secondary_name, "prior")
        self.assertTrue(np.array_equal(primary_xy, candidates["body_bins"]))
        self.assertTrue(np.array_equal(secondary_xy, candidates["prior"]))

    def test_select_centerline_paths_keeps_global_length_consensus_when_low_span_candidate_is_filtered(self) -> None:
        mask = np.ones((40, 80), dtype=np.uint8) * 255
        mask[20, 30] = 0
        candidates = {
            "body_bins": np.array([[10.0, 20.0], [30.0, 20.0], [50.0, 20.0], [70.0, 20.0]], dtype=float),
            "skeleton": np.array([[39.0, 30.0], [39.0, 20.0], [39.0, 10.0]], dtype=float),
            "prior": np.array([[5.0, 21.0], [30.0, 21.0], [55.0, 21.0], [75.0, 21.0]], dtype=float),
        }

        primary_name, primary_xy, secondary_name, secondary_xy = _select_centerline_paths(mask, candidates)

        self.assertEqual(primary_name, "body_bins")
        self.assertEqual(secondary_name, "prior")
        self.assertTrue(np.array_equal(primary_xy, candidates["body_bins"]))
        self.assertTrue(np.array_equal(secondary_xy, candidates["prior"]))


if __name__ == "__main__":
    unittest.main()
