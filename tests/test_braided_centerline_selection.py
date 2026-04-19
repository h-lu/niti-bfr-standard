from __future__ import annotations

import unittest

import numpy as np

from niti_bfr.extract_braided import _select_centerline_paths


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


if __name__ == "__main__":
    unittest.main()
