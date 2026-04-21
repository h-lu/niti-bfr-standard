from __future__ import annotations

import unittest

import numpy as np

from niti_bfr.extract import ExtractionConfig, _component_mask


class WireExtractionSelectionTests(unittest.TestCase):
    def test_component_mask_bridges_split_wire_before_largest_component(self) -> None:
        mask = np.zeros((80, 120), dtype=np.uint8)
        mask[10:44, 58:62] = 255
        for offset in range(34):
            y = 43 + offset
            x = 58 - offset
            mask[max(0, y - 1) : min(mask.shape[0], y + 2), max(0, x - 1) : min(mask.shape[1], x + 2)] = 255
        for offset in range(26):
            y = 50 + offset
            x = 32 - offset
            mask[max(0, y - 1) : min(mask.shape[0], y + 2), max(0, x - 1) : min(mask.shape[1], x + 2)] = 255

        gap_slice = (slice(46, 52), slice(35, 43))
        mask[gap_slice] = 0

        component = _component_mask(mask, ExtractionConfig(close_kernel=7, component_bridge_kernel=11))
        rows, cols = np.where(component > 0)

        self.assertGreater(len(rows), 0)
        self.assertLessEqual(int(cols.min()), 8)
        self.assertGreaterEqual(int(cols.max()), 59)

    def test_component_mask_merges_distal_fragment_when_it_stays_on_wire_axis(self) -> None:
        mask = np.zeros((96, 128), dtype=np.uint8)
        mask[6:52, 70:74] = 255
        for offset in range(42):
            y = 50 + offset
            x = 70 - offset
            mask[max(0, y - 1) : min(mask.shape[0], y + 2), max(0, x - 1) : min(mask.shape[1], x + 2)] = 255
        for offset in range(14):
            y = 81 + offset
            x = 25 - offset
            mask[max(0, y - 1) : min(mask.shape[0], y + 2), max(0, x - 1) : min(mask.shape[1], x + 2)] = 255

        mask[74:82, 22:34] = 0

        component = _component_mask(mask, ExtractionConfig(close_kernel=7, component_bridge_kernel=11))
        rows, cols = np.where(component > 0)

        self.assertGreater(len(rows), 0)
        self.assertLessEqual(int(cols.min()), 11)
        self.assertGreaterEqual(int(cols.max()), 73)


if __name__ == "__main__":
    unittest.main()
