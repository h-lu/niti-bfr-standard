from __future__ import annotations

import cv2
import numpy as np

from niti_bfr.extract_braided import (
    BraidedExtractionConfig,
    _prune_attachment_candidates_from_component,
)


def test_attachment_pruning_removes_border_rod_and_tip_cap() -> None:
    component = np.zeros((90, 180), dtype=np.uint8)
    cv2.ellipse(component, (90, 45), (46, 14), 0, 0, 360, 255, -1)
    cv2.line(component, (6, 22), (6, 44), 255, 7)
    cv2.circle(component, (160, 45), 8, 255, -1)

    body_tube_prior_mask = np.zeros_like(component)
    cv2.ellipse(body_tube_prior_mask, (90, 45), (44, 13), 0, 0, 360, 255, -1)

    cleaned, rejected = _prune_attachment_candidates_from_component(
        component,
        body_tube_prior_mask,
        body_center_xy=np.array([90.0, 45.0]),
        body_axis=np.array([1.0, 0.0]),
        body_positions=np.array([-44.0, 44.0]),
        width_profile_px=np.array([8.0, 18.0, 28.0, 18.0, 8.0]),
        config=BraidedExtractionConfig(min_component_area=80, attachment_min_area_px2=12),
    )

    assert int(cleaned[30, 6]) == 0
    assert int(cleaned[45, 160]) == 0
    assert int(cleaned[45, 90]) == 255
    assert int(rejected[30, 6]) == 255
    assert int(rejected[45, 160]) == 255


def test_attachment_pruning_keeps_long_aligned_residual_band() -> None:
    component = np.zeros((90, 180), dtype=np.uint8)
    cv2.ellipse(component, (90, 45), (50, 16), 0, 0, 360, 255, -1)

    body_tube_prior_mask = np.zeros_like(component)
    cv2.ellipse(body_tube_prior_mask, (90, 45), (44, 12), 0, 0, 360, 255, -1)

    cleaned, rejected = _prune_attachment_candidates_from_component(
        component,
        body_tube_prior_mask,
        body_center_xy=np.array([90.0, 45.0]),
        body_axis=np.array([1.0, 0.0]),
        body_positions=np.array([-44.0, 44.0]),
        width_profile_px=np.array([8.0, 18.0, 28.0, 18.0, 8.0]),
        config=BraidedExtractionConfig(min_component_area=80, attachment_min_area_px2=12),
    )

    assert np.array_equal(cleaned, component)
    assert np.count_nonzero(rejected) == 0
