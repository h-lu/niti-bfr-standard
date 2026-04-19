from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import cv2
import numpy as np
from skimage.morphology import skeletonize


@dataclass
class BraidedExtractionConfig:
    roi_xyxy: tuple[int, int, int, int] = (180, 0, 760, 520)
    blur_ksize: int = 5
    threshold_dark: int = 145
    open_kernel: int = 1
    close_kernel: int = 5
    min_component_area: int = 200
    width_sampling_step_px: float = 4.0
    taper_threshold_ratio: float = 0.25
    compaction_threshold_ratio: float = 0.75
    qc_max_segments: int = 15
    tube_radius_scale: float = 1.0
    body_min_halfwidth_px: float = 3.0
    centerline_smooth_window: int = 7
    diameter_peak_threshold_ratio: float = 0.95
    attachment_min_area_px2: int = 24
    attachment_orientation_mismatch_deg: float = 35.0
    attachment_far_axis_distance_ratio: float = 0.78
    attachment_short_axis_span_ratio: float = 0.18
    attachment_short_normal_span_ratio: float = 0.55
    attachment_prune_dilate_kernel: int = 1
    diameter_proxy_window_start_norm: float = 0.6
    diameter_proxy_window_end_norm: float = 0.8


@dataclass
class BraidedExtractionResult:
    anchor_xy: np.ndarray
    tip_xy: np.ndarray
    length_env_px: float
    length_axis_px: float
    length_axis_skeleton_px: float
    length_axis_body_bins_px: float
    length_axis_alt_px: float
    length_axis_disagreement_px: float
    diameter_max_px: float
    diameter_max_orth_px: float
    diameter_max_thickness_px: float
    diameter_max_feret_px: float
    diameter_p95_px: float
    diameter_mid_median_px: float
    diameter_mid_p90_px: float
    diameter_peak_span_px: float
    diameter_peak_pos_norm: float
    area_proj_px2: float
    area_proj_contour_width_integral_px2: float
    area_proj_contour_px2: float
    area_proj_definition_gap_px2: float
    body_mask_area_px2: float
    component_area_px2: float
    excluded_attachment_area_px2: float
    body_mask_attachment_leak_fraction: float
    attachment_count: int
    attachment_border_touch_count: int
    attachment_max_elongation: float
    attachment_max_solidity: float
    attachment_max_orientation_mismatch_deg: float
    attachment_max_distance_to_main_axis_px: float
    x_peak_norm: float
    taper_left_px: float
    taper_right_px: float
    landing_zone_left_px: float
    landing_zone_right_px: float
    transition_zone_left_px: float
    transition_zone_right_px: float
    compaction_zone_length_px: float
    zone_symmetry: float
    branch_component_count_after_pruning: int
    branch_count_after_pruning: int
    centerline_disagreement: float
    endpoint_jump_px: float
    axis_peak_position_stability: float
    quality: float
    mask: np.ndarray
    body_tube_mask: np.ndarray
    excluded_attachment_mask: np.ndarray
    contour_xy: np.ndarray
    body_contour_xy: np.ndarray
    axis_line_xy: np.ndarray
    sampled_axis_xy: np.ndarray
    sampled_centerline_xy: np.ndarray
    sampled_width_segments_xy: np.ndarray
    width_profile_pos_px: np.ndarray
    width_profile_diameter_px: np.ndarray
    orth_width_profile_pos_px: np.ndarray
    orth_width_profile_diameter_px: np.ndarray


@dataclass
class BraidedZoneMetrics:
    x_peak_norm: float
    taper_left_px: float
    taper_right_px: float
    landing_zone_left_px: float
    landing_zone_right_px: float
    transition_zone_left_px: float
    transition_zone_right_px: float
    compaction_zone_length_px: float
    zone_symmetry: float


def _clip_roi(frame_bgr: np.ndarray, roi_xyxy: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = frame_bgr.shape[:2]
    x0, y0, x1, y1 = roi_xyxy
    x0 = int(np.clip(x0, 0, w - 1))
    y0 = int(np.clip(y0, 0, h - 1))
    x1 = int(np.clip(x1, x0 + 1, w))
    y1 = int(np.clip(y1, y0 + 1, h))
    roi = frame_bgr[y0:y1, x0:x1]
    return roi, np.array([x0, y0], dtype=float)


def _largest_component(mask: np.ndarray, min_area: int) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    best_label = 0
    best_area = 0
    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if area > best_area:
            best_label = label
            best_area = area
    if best_label == 0:
        raise RuntimeError("braided component not found")
    return (labels == best_label).astype(np.uint8) * 255


def _component_mask(frame_bgr: np.ndarray, config: BraidedExtractionConfig) -> tuple[np.ndarray, np.ndarray]:
    roi_bgr, roi_offset_xy = _clip_roi(frame_bgr, config.roi_xyxy)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blur_ksize = max(1, int(config.blur_ksize))
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    if blur_ksize > 1:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    _, mask = cv2.threshold(gray, int(config.threshold_dark), 255, cv2.THRESH_BINARY_INV)
    if config.open_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.open_kernel, config.open_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if config.close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.close_kernel, config.close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    component = _largest_component(mask, min_area=int(config.min_component_area))
    return component, roi_offset_xy


def _component_contour(component: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("braided contour not found")
    contour = max(contours, key=cv2.contourArea)
    return contour[:, 0, :].astype(float)


def _mask_contour(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("braided body contour not found")
    contour = max(contours, key=cv2.contourArea)
    return contour[:, 0, :].astype(float)


def _unit_direction(vector_xy: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector_xy))
    if norm < 1e-9:
        raise RuntimeError("degenerate braided axis")
    return vector_xy / norm


def _principal_axis(points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center_xy = points_xy.mean(axis=0)
    centered = points_xy - center_xy[None, :]
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = _unit_direction(vh[0])
    if axis[0] < 0.0:
        axis = -axis
    normal = np.array([-axis[1], axis[0]], dtype=float)
    proj_axis = centered @ axis
    proj_normal = centered @ normal
    return center_xy, axis, proj_axis, proj_normal


def _project_points(
    points_xy: np.ndarray,
    center_xy: np.ndarray,
    axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    normal = np.array([-axis[1], axis[0]], dtype=float)
    centered = points_xy - center_xy[None, :]
    proj_axis = centered @ axis
    proj_normal = centered @ normal
    return proj_axis, proj_normal


def _sample_envelope_profile(
    envelope_xy: np.ndarray,
    center_xy: np.ndarray,
    axis: np.ndarray,
    step_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    proj_axis, proj_normal = _project_points(envelope_xy, center_xy, axis)
    step_px = max(float(step_px), 1.0)
    min_proj = float(np.min(proj_axis))
    max_proj = float(np.max(proj_axis))
    positions = np.arange(min_proj, max_proj + 0.5 * step_px, step_px, dtype=float)
    if len(positions) < 3:
        positions = np.linspace(min_proj, max_proj, 3)
    half_step = max(0.5 * step_px, 1.0)
    axis_points: list[np.ndarray] = []
    diameters: list[float] = []
    for pos in positions:
        in_bin = np.abs(proj_axis - pos) <= half_step
        if not np.any(in_bin):
            axis_points.append(np.array([np.nan, np.nan], dtype=float))
            diameters.append(np.nan)
            continue
        normal_vals = proj_normal[in_bin]
        point_bin = envelope_xy[in_bin]
        lo_idx = int(np.argmin(normal_vals))
        hi_idx = int(np.argmax(normal_vals))
        low_point = point_bin[lo_idx]
        high_point = point_bin[hi_idx]
        axis_points.append(0.5 * (low_point + high_point))
        diameters.append(float(normal_vals[hi_idx] - normal_vals[lo_idx]))
    return positions, np.asarray(axis_points), np.asarray(diameters, dtype=float), np.asarray(positions, dtype=float)


def _qc_segments(
    sampled_axis_xy: np.ndarray,
    width_profile_px: np.ndarray,
    tangents_xy: np.ndarray,
    max_segments: int,
) -> np.ndarray:
    valid_idx = np.flatnonzero(np.isfinite(width_profile_px) & np.all(np.isfinite(sampled_axis_xy), axis=1))
    if len(valid_idx) == 0:
        return np.empty((0, 2, 2), dtype=float)
    keep = valid_idx
    if len(keep) > max_segments:
        keep = np.linspace(0, len(valid_idx) - 1, max_segments, dtype=int)
        keep = valid_idx[keep]
    segments = []
    for idx in keep:
        tangent = tangents_xy[idx]
        if not np.all(np.isfinite(tangent)):
            continue
        normal = _unit_direction(np.array([-tangent[1], tangent[0]], dtype=float))
        radius = 0.5 * float(width_profile_px[idx])
        center_xy = sampled_axis_xy[idx]
        segments.append(np.stack([center_xy - radius * normal, center_xy + radius * normal], axis=0))
    if not segments:
        return np.empty((0, 2, 2), dtype=float)
    return np.asarray(segments, dtype=float)


def _moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    window = max(1, int(window))
    if window <= 1 or len(values) == 0:
        return values.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _smooth_path(points_xy: np.ndarray, window: int = 5) -> np.ndarray:
    if len(points_xy) == 0:
        return points_xy.copy()
    smoothed = points_xy.copy()
    valid = np.all(np.isfinite(points_xy), axis=1)
    valid_idx = np.flatnonzero(valid)
    if len(valid_idx) < 3:
        return smoothed
    segment = points_xy[valid_idx].copy()
    for dim in range(segment.shape[1]):
        segment[:, dim] = _moving_average(segment[:, dim], window=window)
    smoothed[valid_idx] = segment
    return smoothed


def cumulative_path_length(points_xy: np.ndarray) -> np.ndarray:
    cumulative = np.full(len(points_xy), np.nan, dtype=float)
    if len(points_xy) == 0:
        return cumulative
    valid = np.all(np.isfinite(points_xy), axis=1)
    valid_idx = np.flatnonzero(valid)
    if len(valid_idx) == 0:
        return cumulative
    valid_points = points_xy[valid_idx]
    deltas = np.linalg.norm(np.diff(valid_points, axis=0), axis=1)
    cumulative_valid = np.concatenate([[0.0], np.cumsum(deltas, dtype=float)])
    cumulative[valid_idx] = cumulative_valid
    return cumulative


def _resample_polyline(points_xy: np.ndarray, step_px: float) -> np.ndarray:
    if len(points_xy) < 2:
        return points_xy.copy()
    curve = cumulative_path_length(points_xy)
    total = float(curve[-1])
    if not np.isfinite(total) or total <= 1e-9:
        return points_xy.copy()
    sample_step = max(float(step_px), 1.0)
    sample_pos = np.arange(0.0, total + 0.5 * sample_step, sample_step, dtype=float)
    if sample_pos[-1] < total:
        sample_pos = np.append(sample_pos, total)
    x = np.interp(sample_pos, curve, points_xy[:, 0])
    y = np.interp(sample_pos, curve, points_xy[:, 1])
    return np.column_stack([x, y])


def _find_body_bounds(
    width_profile_px: np.ndarray,
    peak_idx: int,
    diameter_max_px: float,
) -> tuple[int, int]:
    valid_idx = np.flatnonzero(np.isfinite(width_profile_px))
    if len(valid_idx) == 0:
        raise RuntimeError("braided width profile unavailable")
    raw = np.nan_to_num(_moving_average(width_profile_px, window=5), nan=0.0)
    body_threshold = max(0.22 * diameter_max_px, 8.0)
    body_support = np.isfinite(width_profile_px) & (raw >= body_threshold)
    if not np.any(body_support):
        return int(valid_idx[0]), int(valid_idx[-1])

    center_idx = int(np.clip(peak_idx, valid_idx[0], valid_idx[-1]))
    if not body_support[center_idx]:
        supported = np.flatnonzero(body_support)
        center_idx = int(supported[np.argmin(np.abs(supported - center_idx))])

    left_idx = center_idx
    while left_idx > int(valid_idx[0]) and body_support[left_idx - 1]:
        left_idx -= 1
    right_idx = center_idx
    while right_idx < int(valid_idx[-1]) and body_support[right_idx + 1]:
        right_idx += 1
    return left_idx, right_idx


def compute_braided_body_mask(
    width_profile_px: np.ndarray,
    peak_idx: int | None = None,
    diameter_max_px: float | None = None,
) -> np.ndarray:
    valid = np.isfinite(width_profile_px)
    if not np.any(valid):
        raise RuntimeError("braided width profile unavailable")
    if peak_idx is None:
        peak_idx = int(np.nanargmax(width_profile_px))
    if diameter_max_px is None:
        diameter_max_px = float(np.nanmax(width_profile_px))
    body_left_idx, body_right_idx = _find_body_bounds(
        width_profile_px=width_profile_px,
        peak_idx=int(peak_idx),
        diameter_max_px=float(diameter_max_px),
    )
    body_mask = np.zeros_like(valid, dtype=bool)
    body_mask[body_left_idx : body_right_idx + 1] = True
    body_mask &= valid
    return body_mask


def rasterize_braided_body_tube_mask(
    centerline_xy: np.ndarray,
    width_profile_px: np.ndarray,
    body_mask: np.ndarray,
    image_shape: tuple[int, int],
    radius_scale: float = 1.0,
    min_halfwidth_px: float = 3.0,
) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    valid_idx = np.flatnonzero(body_mask & np.isfinite(width_profile_px) & np.all(np.isfinite(centerline_xy), axis=1))
    if len(valid_idx) == 0:
        return mask
    for idx in valid_idx:
        radius = max(float(min_halfwidth_px), 0.5 * float(width_profile_px[idx]) * float(radius_scale))
        center = tuple(np.round(centerline_xy[idx]).astype(int))
        cv2.circle(mask, center, max(1, int(round(radius))), 255, -1, cv2.LINE_AA)
    for left_idx, right_idx in zip(valid_idx[:-1], valid_idx[1:]):
        p0 = tuple(np.round(centerline_xy[left_idx]).astype(int))
        p1 = tuple(np.round(centerline_xy[right_idx]).astype(int))
        radius0 = max(float(min_halfwidth_px), 0.5 * float(width_profile_px[left_idx]) * float(radius_scale))
        radius1 = max(float(min_halfwidth_px), 0.5 * float(width_profile_px[right_idx]) * float(radius_scale))
        thickness = max(1, int(round(2.0 * max(radius0, radius1))))
        cv2.line(mask, p0, p1, 255, thickness, cv2.LINE_AA)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _build_body_prior_from_component(
    component: np.ndarray,
    config: BraidedExtractionConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    contour_local = _component_contour(component)
    rows, cols = np.where(component > 0)
    if len(rows) < int(config.min_component_area):
        raise RuntimeError("braided component too small")
    pixel_local_xy = np.column_stack([cols, rows]).astype(float)

    center_xy, axis, proj_axis, _ = _principal_axis(pixel_local_xy)
    step_px = max(float(config.width_sampling_step_px), 1.0)
    positions, axis_points_xy, width_profile_px, _ = _sample_envelope_profile(
        envelope_xy=contour_local,
        center_xy=center_xy,
        axis=axis,
        step_px=step_px,
    )
    valid = np.isfinite(width_profile_px)
    if not np.any(valid):
        raise RuntimeError("braided width profile unavailable")

    diameter_seed_px = float(np.nanmax(width_profile_px))
    peak_idx = int(np.nanargmax(width_profile_px))
    body_mask = compute_braided_body_mask(
        width_profile_px=width_profile_px,
        peak_idx=peak_idx,
        diameter_max_px=diameter_seed_px,
    )
    body_positions = positions[body_mask]
    if len(body_positions) == 0:
        raise RuntimeError("braided body span unavailable")

    body_pixel_mask = (proj_axis >= float(body_positions[0]) - step_px) & (proj_axis <= float(body_positions[-1]) + 0.5 * step_px)
    body_pixel_xy = pixel_local_xy[body_pixel_mask]
    if len(body_pixel_xy) < int(config.min_component_area):
        body_pixel_xy = pixel_local_xy
    body_center_xy, body_axis, _, _ = _principal_axis(body_pixel_xy)
    positions, axis_points_xy, width_profile_px, _ = _sample_envelope_profile(
        envelope_xy=contour_local,
        center_xy=body_center_xy,
        axis=body_axis,
        step_px=step_px,
    )
    axis_points_xy = _smooth_path(axis_points_xy, window=max(3, int(config.centerline_smooth_window)))
    valid = np.isfinite(width_profile_px)
    if not np.any(valid):
        raise RuntimeError("braided refined width profile unavailable")

    peak_idx = int(np.nanargmax(width_profile_px))
    body_mask = compute_braided_body_mask(
        width_profile_px=width_profile_px,
        peak_idx=peak_idx,
        diameter_max_px=float(np.nanmax(width_profile_px)),
    )
    body_positions = positions[body_mask]
    if len(body_positions) == 0:
        raise RuntimeError("braided refined body span unavailable")

    body_tube_prior_mask = rasterize_braided_body_tube_mask(
        centerline_xy=axis_points_xy,
        width_profile_px=width_profile_px,
        body_mask=body_mask,
        image_shape=component.shape,
        radius_scale=float(config.tube_radius_scale),
        min_halfwidth_px=float(config.body_min_halfwidth_px),
    )
    body_tube_prior_mask = cv2.bitwise_and(body_tube_prior_mask, component)
    if np.count_nonzero(body_tube_prior_mask) < int(config.min_component_area):
        raise RuntimeError("braided body tube prior mask too small")
    body_tube_prior_mask = _largest_component(
        body_tube_prior_mask,
        min_area=max(32, int(config.min_component_area * 0.4)),
    )
    return (
        contour_local,
        pixel_local_xy,
        step_px,
        body_center_xy,
        body_axis,
        positions,
        axis_points_xy,
        width_profile_px,
        body_mask,
        body_positions,
        body_tube_prior_mask,
    )


def _mask_from_pixels(
    image_shape: tuple[int, int],
    pixel_xy: np.ndarray,
) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    if len(pixel_xy) == 0:
        return mask
    cols = np.clip(np.round(pixel_xy[:, 0]).astype(int), 0, image_shape[1] - 1)
    rows = np.clip(np.round(pixel_xy[:, 1]).astype(int), 0, image_shape[0] - 1)
    mask[rows, cols] = 255
    return mask


def _clip_component_to_axis_span(
    component: np.ndarray,
    pixel_local_xy: np.ndarray,
    center_xy: np.ndarray,
    axis: np.ndarray,
    clip_min: float,
    clip_max: float,
    min_area: int,
) -> np.ndarray:
    proj_axis, _ = _project_points(pixel_local_xy, center_xy, axis)
    keep = (proj_axis >= float(clip_min)) & (proj_axis <= float(clip_max))
    clipped = _mask_from_pixels(component.shape, pixel_local_xy[keep])
    if np.count_nonzero(clipped) < int(min_area):
        raise RuntimeError("braided axis-span clip too small")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clipped = cv2.morphologyEx(clipped, cv2.MORPH_CLOSE, kernel)
    return _largest_component(clipped, min_area=max(16, int(min_area * 0.4)))


def _project_span_along_axis(
    points_xy: np.ndarray,
    center_xy: np.ndarray,
    axis: np.ndarray,
) -> tuple[float, float]:
    proj_axis, _ = _project_points(points_xy, center_xy, axis)
    return float(np.min(proj_axis)), float(np.max(proj_axis))


def _integrate_projected_area_px2(
    positions: np.ndarray,
    width_profile_px: np.ndarray,
    body_mask: np.ndarray,
) -> float:
    valid = body_mask & np.isfinite(positions) & np.isfinite(width_profile_px)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    pos = positions[valid]
    width = width_profile_px[valid]
    order = np.argsort(pos)
    return float(np.trapezoid(width[order], pos[order]))


def _window_width_stats(
    positions_rel_px: np.ndarray,
    width_profile_px: np.ndarray,
    window_start_norm: float,
    window_end_norm: float,
) -> tuple[float, float]:
    valid = np.isfinite(positions_rel_px) & np.isfinite(width_profile_px)
    if np.count_nonzero(valid) == 0:
        return float("nan"), float("nan")

    positions_valid = positions_rel_px[valid]
    widths_valid = width_profile_px[valid]
    lo = float(np.clip(min(window_start_norm, window_end_norm), 0.0, 1.0))
    hi = float(np.clip(max(window_start_norm, window_end_norm), 0.0, 1.0))
    span_px = float(np.max(positions_valid) - np.min(positions_valid)) if len(positions_valid) >= 2 else 0.0

    if span_px > 1e-9:
        pos_norm = (positions_valid - float(np.min(positions_valid))) / span_px
        window = (pos_norm >= lo) & (pos_norm <= hi)
    else:
        window = np.ones(len(widths_valid), dtype=bool)

    if np.count_nonzero(window) < 3:
        window = np.ones(len(widths_valid), dtype=bool)

    widths_window = widths_valid[window]
    return float(np.nanmedian(widths_window)), float(np.nanpercentile(widths_window, 90))


def compute_braided_zone_metrics(
    positions: np.ndarray,
    width_profile_px: np.ndarray,
    body_mask: np.ndarray,
    diameter_max_px: float,
    taper_threshold_ratio: float,
    compaction_threshold_ratio: float,
) -> BraidedZoneMetrics:
    body_positions = positions[body_mask]
    if len(body_positions) == 0:
        raise RuntimeError("braided body span unavailable for zone metrics")

    length_axis_px = float(body_positions[-1] - body_positions[0]) if len(body_positions) >= 2 else 0.0
    peak_idx = int(np.nanargmax(width_profile_px))
    peak_pos = float(positions[peak_idx])
    x_peak_norm = 0.5
    if length_axis_px > 1e-9:
        x_peak_norm = float(np.clip((peak_pos - body_positions[0]) / length_axis_px, 0.0, 1.0))

    taper_threshold = float(taper_threshold_ratio) * diameter_max_px
    support = np.flatnonzero(body_mask & (width_profile_px >= taper_threshold))
    if len(support) == 0:
        landing_zone_left_px = np.nan
        landing_zone_right_px = np.nan
        taper_left_px = np.nan
        taper_right_px = np.nan
        support_left_boundary = float(body_positions[0])
        support_right_boundary = float(body_positions[-1])
    else:
        support_left_boundary = float(positions[support[0]])
        support_right_boundary = float(positions[support[-1]])
        landing_zone_left_px = max(0.0, support_left_boundary - body_positions[0])
        landing_zone_right_px = max(0.0, body_positions[-1] - support_right_boundary)
        taper_left_px = landing_zone_left_px
        taper_right_px = landing_zone_right_px

    compaction_threshold = float(compaction_threshold_ratio) * diameter_max_px
    compaction_support = np.flatnonzero(body_mask & (width_profile_px >= compaction_threshold))
    if len(compaction_support) == 0:
        compaction_zone_length_px = np.nan
        transition_zone_left_px = np.nan
        transition_zone_right_px = np.nan
    else:
        compaction_left_boundary = float(positions[compaction_support[0]])
        compaction_right_boundary = float(positions[compaction_support[-1]])
        compaction_zone_length_px = max(0.0, compaction_right_boundary - compaction_left_boundary)
        transition_zone_left_px = max(0.0, compaction_left_boundary - support_left_boundary)
        transition_zone_right_px = max(0.0, support_right_boundary - compaction_right_boundary)

    if not np.isfinite(length_axis_px) or length_axis_px <= 1e-9:
        zone_symmetry = np.nan
    else:
        left_extent = np.nan_to_num(landing_zone_left_px, nan=0.0) + np.nan_to_num(transition_zone_left_px, nan=0.0)
        right_extent = np.nan_to_num(landing_zone_right_px, nan=0.0) + np.nan_to_num(transition_zone_right_px, nan=0.0)
        zone_symmetry = float(np.clip(1.0 - abs(left_extent - right_extent) / length_axis_px, 0.0, 1.0))

    return BraidedZoneMetrics(
        x_peak_norm=x_peak_norm,
        taper_left_px=float(taper_left_px),
        taper_right_px=float(taper_right_px),
        landing_zone_left_px=float(landing_zone_left_px),
        landing_zone_right_px=float(landing_zone_right_px),
        transition_zone_left_px=float(transition_zone_left_px),
        transition_zone_right_px=float(transition_zone_right_px),
        compaction_zone_length_px=float(compaction_zone_length_px),
        zone_symmetry=float(zone_symmetry),
    )


def _skeleton_graph(mask: np.ndarray) -> tuple[list[tuple[float, float]], list[list[tuple[int, float]]], np.ndarray]:
    skeleton = skeletonize(mask > 0)
    rows, cols = np.where(skeleton)
    if len(rows) < 2:
        raise RuntimeError("braided body skeleton too short")
    coords = list(zip(cols.astype(float), rows.astype(float), strict=False))
    index = {(int(r), int(c)): idx for idx, (c, r) in enumerate(coords)}
    adjacency: list[list[tuple[int, float]]] = [[] for _ in coords]
    offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    for idx, (c, r) in enumerate(coords):
        rr = int(r)
        cc = int(c)
        for dr, dc in offsets:
            neighbor = (rr + dr, cc + dc)
            other_idx = index.get(neighbor)
            if other_idx is None:
                continue
            weight = math.sqrt(2.0) if dr != 0 and dc != 0 else 1.0
            adjacency[idx].append((other_idx, weight))
    degrees = np.asarray([len(neighbors) for neighbors in adjacency], dtype=int)
    return [(c, r) for c, r in coords], adjacency, degrees


def _dijkstra_farthest(
    adjacency: list[list[tuple[int, float]]],
    start_idx: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    dist = np.full(len(adjacency), np.inf, dtype=float)
    prev = np.full(len(adjacency), -1, dtype=int)
    dist[start_idx] = 0.0
    queue: list[tuple[float, int]] = [(0.0, start_idx)]
    while queue:
        current_dist, node = heapq.heappop(queue)
        if current_dist > dist[node]:
            continue
        for neighbor, weight in adjacency[node]:
            cand = current_dist + weight
            if cand + 1e-9 < dist[neighbor]:
                dist[neighbor] = cand
                prev[neighbor] = node
                heapq.heappush(queue, (cand, neighbor))
    farthest = int(np.nanargmax(np.where(np.isfinite(dist), dist, -np.inf)))
    return farthest, dist, prev


def _reconstruct_path(prev: np.ndarray, end_idx: int) -> list[int]:
    path = [int(end_idx)]
    node = int(end_idx)
    while prev[node] >= 0:
        node = int(prev[node])
        path.append(node)
    path.reverse()
    return path


def _filtered_adjacency(
    adjacency: list[list[tuple[int, float]]],
    active: np.ndarray,
) -> tuple[list[list[tuple[int, float]]], np.ndarray]:
    filtered: list[list[tuple[int, float]]] = [[] for _ in adjacency]
    degrees = np.zeros(len(adjacency), dtype=int)
    active_bool = np.asarray(active, dtype=bool)
    for idx, neighbors in enumerate(adjacency):
        if not active_bool[idx]:
            continue
        keep = [(other, weight) for other, weight in neighbors if active_bool[other]]
        filtered[idx] = keep
        degrees[idx] = len(keep)
    return filtered, degrees


def _prune_terminal_spurs(
    adjacency: list[list[tuple[int, float]]],
    step_px: float,
) -> tuple[np.ndarray, int]:
    active = np.ones(len(adjacency), dtype=bool)
    spur_length_limit = max(18.0, 6.0 * float(step_px))

    while True:
        filtered, degrees = _filtered_adjacency(adjacency, active)
        endpoints = np.flatnonzero(active & (degrees == 1))
        if len(endpoints) == 0:
            break

        changed = False
        for endpoint in endpoints:
            if not active[endpoint]:
                continue
            filtered, degrees = _filtered_adjacency(adjacency, active)
            if degrees[endpoint] != 1:
                continue
            path_nodes = [int(endpoint)]
            path_length = 0.0
            prev = -1
            current = int(endpoint)
            terminus = int(endpoint)

            while True:
                next_candidates = [(other, weight) for other, weight in filtered[current] if other != prev]
                if len(next_candidates) != 1:
                    terminus = int(current)
                    break
                nxt, weight = next_candidates[0]
                path_length += float(weight)
                prev = current
                current = int(nxt)
                if degrees[current] != 2:
                    terminus = int(current)
                    break
                path_nodes.append(current)

            if terminus == endpoint:
                continue
            if degrees[terminus] <= 1:
                continue
            if path_length > spur_length_limit:
                continue
            for node in path_nodes:
                if node != terminus and active[node]:
                    active[node] = False
                    changed = True

        if not changed:
            break

    filtered, degrees = _filtered_adjacency(adjacency, active)
    branch_nodes = np.flatnonzero(active & (degrees > 2))
    seen: set[int] = set()
    branch_components = 0
    branch_set = set(int(node) for node in branch_nodes)
    for node in branch_nodes:
        node = int(node)
        if node in seen:
            continue
        branch_components += 1
        stack = [node]
        seen.add(node)
        while stack:
            current = stack.pop()
            for other, _ in filtered[current]:
                if other in branch_set and other not in seen:
                    seen.add(other)
                    stack.append(other)

    return active, int(branch_components)


def _extract_centerline_from_body(mask: np.ndarray, smooth_window: int, step_px: float) -> tuple[np.ndarray, int]:
    coords, adjacency, degrees = _skeleton_graph(mask)
    active, branch_component_count = _prune_terminal_spurs(adjacency, step_px=step_px)
    filtered, filtered_degrees = _filtered_adjacency(adjacency, active)
    active_nodes = np.flatnonzero(active & (filtered_degrees > 0))
    if len(active_nodes) == 0:
        active_nodes = np.flatnonzero(active)
    if len(active_nodes) < 2:
        raise RuntimeError("braided pruned skeleton too short")
    start_idx = int(active_nodes[0])
    anchor_idx, _, _ = _dijkstra_farthest(filtered, start_idx)
    tip_idx, _, prev = _dijkstra_farthest(filtered, anchor_idx)
    path_indices = _reconstruct_path(prev, tip_idx)
    path_xy = np.asarray([coords[idx] for idx in path_indices], dtype=float)
    if len(path_xy) < 2:
        raise RuntimeError("braided centerline path too short")
    path_xy = _smooth_path(path_xy, window=smooth_window)
    path_xy = _resample_polyline(path_xy, step_px=max(step_px * 0.75, 1.0))
    return path_xy, branch_component_count


def _extract_centerline_from_body_bins(mask: np.ndarray, smooth_window: int, step_px: float) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.where(mask > 0)
    if len(rows) < 2:
        raise RuntimeError("braided body pixels unavailable")
    points_xy = np.column_stack([cols, rows]).astype(float)
    center_xy, axis, proj_axis, proj_normal = _principal_axis(points_xy)
    normal = np.array([-axis[1], axis[0]], dtype=float)
    min_proj = float(np.min(proj_axis))
    max_proj = float(np.max(proj_axis))
    positions = np.arange(min_proj, max_proj + 0.5 * step_px, max(step_px, 1.0), dtype=float)
    half_step = max(0.5 * step_px, 1.0)
    center_offsets: list[float] = []
    center_positions: list[float] = []
    for pos in positions:
        in_bin = np.abs(proj_axis - pos) <= half_step
        if not np.any(in_bin):
            continue
        center_positions.append(float(np.median(proj_axis[in_bin])))
        center_offsets.append(float(np.median(proj_normal[in_bin])))
    if len(center_offsets) < 2:
        raise RuntimeError("braided body-bin centerline too short")
    center_positions_arr = _moving_average(np.asarray(center_positions, dtype=float), window=smooth_window)
    center_offsets_arr = _moving_average(np.asarray(center_offsets, dtype=float), window=smooth_window)
    centerline_xy = center_xy[None, :] + center_positions_arr[:, None] * axis[None, :] + center_offsets_arr[:, None] * normal[None, :]
    centerline_xy = _resample_polyline(centerline_xy, step_px=max(step_px * 0.75, 1.0))
    return centerline_xy, axis


def _align_path_orientation(reference_xy: np.ndarray, candidate_xy: np.ndarray) -> np.ndarray:
    if len(reference_xy) < 1 or len(candidate_xy) < 1:
        return candidate_xy
    forward_gap = float(
        np.linalg.norm(reference_xy[0] - candidate_xy[0])
        + np.linalg.norm(reference_xy[-1] - candidate_xy[-1])
    )
    reverse_gap = float(
        np.linalg.norm(reference_xy[0] - candidate_xy[-1])
        + np.linalg.norm(reference_xy[-1] - candidate_xy[0])
    )
    if reverse_gap < forward_gap:
        return candidate_xy[::-1].copy()
    return candidate_xy


def _nearest_distance(distance_map: np.ndarray, point_xy: np.ndarray) -> float:
    x = int(np.clip(round(float(point_xy[0])), 0, distance_map.shape[1] - 1))
    y = int(np.clip(round(float(point_xy[1])), 0, distance_map.shape[0] - 1))
    return float(distance_map[y, x])


def _sample_binary(mask: np.ndarray, point_xy: np.ndarray) -> bool:
    x = int(round(float(point_xy[0])))
    y = int(round(float(point_xy[1])))
    if x < 0 or y < 0 or y >= mask.shape[0] or x >= mask.shape[1]:
        return False
    return bool(mask[y, x] > 0)


def _path_tangents(points_xy: np.ndarray) -> np.ndarray:
    if len(points_xy) == 0:
        return np.empty((0, 2), dtype=float)
    delta = np.gradient(points_xy, axis=0)
    norms = np.linalg.norm(delta, axis=1, keepdims=True)
    return delta / np.maximum(norms, 1e-9)


def _path_length(points_xy: np.ndarray) -> float:
    curve = cumulative_path_length(points_xy)
    if len(curve) == 0:
        return float("nan")
    return float(curve[-1])


def _path_support_fraction(mask: np.ndarray, points_xy: np.ndarray) -> float:
    if len(points_xy) == 0:
        return 0.0
    supported = np.asarray([_sample_binary(mask, point_xy) for point_xy in points_xy], dtype=float)
    if len(supported) == 0:
        return 0.0
    return float(np.mean(supported))


def _path_axis_span(points_xy: np.ndarray, axis_xy: np.ndarray) -> float:
    if len(points_xy) == 0:
        return 0.0
    axis = _unit_direction(np.asarray(axis_xy, dtype=float))
    proj = np.asarray(points_xy, dtype=float) @ axis
    return float(np.max(proj) - np.min(proj))


def _endpoint_gap(primary_xy: np.ndarray, secondary_xy: np.ndarray) -> float:
    if len(primary_xy) == 0 or len(secondary_xy) == 0:
        return float("nan")
    primary_length = _path_length(primary_xy)
    secondary_length = _path_length(secondary_xy)
    if np.isfinite(primary_length) and np.isfinite(secondary_length) and primary_length <= secondary_length:
        shorter_xy, longer_xy = primary_xy, secondary_xy
    else:
        shorter_xy, longer_xy = secondary_xy, primary_xy

    endpoint_gaps: list[float] = []
    for endpoint_xy in (shorter_xy[0], shorter_xy[-1]):
        deltas = longer_xy - endpoint_xy[None, :]
        endpoint_gaps.append(float(np.min(np.linalg.norm(deltas, axis=1))))
    return float(np.mean(endpoint_gaps))


def _select_centerline_paths(
    body_mask: np.ndarray,
    candidates: dict[str, np.ndarray],
) -> tuple[str, np.ndarray, str, np.ndarray]:
    rows, cols = np.where(body_mask > 0)
    if len(rows) < 2:
        raise RuntimeError("braided body pixels unavailable for centerline selection")
    _, body_axis, _, _ = _principal_axis(np.column_stack([cols, rows]).astype(float))

    lengths: dict[str, float] = {}
    supports: dict[str, float] = {}
    axis_spans: dict[str, float] = {}
    for name, points_xy in candidates.items():
        length_px = _path_length(points_xy)
        if not np.isfinite(length_px) or length_px <= 1e-9:
            continue
        lengths[name] = length_px
        supports[name] = _path_support_fraction(body_mask, points_xy)
        axis_spans[name] = _path_axis_span(points_xy, body_axis)
    if len(lengths) < 2:
        raise RuntimeError("braided centerline candidates unavailable")

    max_axis_span_px = max(axis_spans.values())
    span_floor_px = 0.85 * max_axis_span_px
    eligible_primary_names = [name for name, span_px in axis_spans.items() if span_px >= span_floor_px]
    if not eligible_primary_names:
        eligible_primary_names = list(lengths)

    median_length_px = float(np.median(np.asarray([lengths[name] for name in eligible_primary_names], dtype=float)))
    primary_preference = {"body_bins": 0, "skeleton": 1, "prior": 2}
    secondary_preference = {"skeleton": 0, "body_bins": 1, "prior": 2}

    primary_name = min(
        eligible_primary_names,
        key=lambda name: (
            abs(lengths[name] - median_length_px) / max(median_length_px, 1e-9),
            1.0 - supports[name],
            primary_preference.get(name, 99),
        ),
    )
    primary_xy = candidates[primary_name]
    primary_length_px = lengths[primary_name]

    secondary_names = [name for name in eligible_primary_names if name != primary_name]
    if not secondary_names:
        secondary_names = [name for name in lengths if name != primary_name]
    secondary_name = min(
        secondary_names,
        key=lambda name: (
            abs(lengths[name] - primary_length_px) / max(primary_length_px, 1e-9),
            1.0 - supports[name],
            secondary_preference.get(name, 99),
        ),
    )
    secondary_xy = _align_path_orientation(primary_xy, candidates[secondary_name])
    return primary_name, primary_xy, secondary_name, secondary_xy


def _orthogonal_extent(mask: np.ndarray, point_xy: np.ndarray, normal_xy: np.ndarray, sign: float) -> float:
    step = 0.5
    extent = 0.0
    for idx in range(1, 2400):
        dist = idx * step
        probe = point_xy + sign * dist * normal_xy
        if not _sample_binary(mask, probe):
            break
        extent = dist
    return extent


def _orthogonal_widths(mask: np.ndarray, centerline_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tangents_xy = _path_tangents(centerline_xy)
    widths = np.full(len(centerline_xy), np.nan, dtype=float)
    segments = np.full((len(centerline_xy), 2, 2), np.nan, dtype=float)
    for idx, (point_xy, tangent_xy) in enumerate(zip(centerline_xy, tangents_xy, strict=False)):
        if not _sample_binary(mask, point_xy):
            continue
        normal = np.array([-tangent_xy[1], tangent_xy[0]], dtype=float)
        if not np.all(np.isfinite(normal)):
            continue
        normal = _unit_direction(normal)
        neg_extent = _orthogonal_extent(mask, point_xy, normal, sign=-1.0)
        pos_extent = _orthogonal_extent(mask, point_xy, normal, sign=1.0)
        widths[idx] = neg_extent + pos_extent
        segments[idx, 0] = point_xy - neg_extent * normal
        segments[idx, 1] = point_xy + pos_extent * normal
    return widths, segments


def _attachment_metrics(
    attachment_mask: np.ndarray,
    body_center_xy: np.ndarray,
    body_axis: np.ndarray,
    min_area_px2: int,
) -> dict[str, float]:
    if np.count_nonzero(attachment_mask) == 0:
        return {
            "attachment_count": 0,
            "attachment_border_touch_count": 0,
            "attachment_max_elongation": 0.0,
            "attachment_max_solidity": 0.0,
            "attachment_max_orientation_mismatch_deg": 0.0,
            "attachment_max_distance_to_main_axis_px": 0.0,
        }

    num, labels, stats, _ = cv2.connectedComponentsWithStats(attachment_mask)
    border_touch_count = 0
    attachment_count = 0
    max_elongation = 0.0
    max_solidity = 0.0
    max_orientation_mismatch_deg = 0.0
    max_distance_to_main_axis_px = 0.0
    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(min_area_px2):
            continue
        component = (labels == label).astype(np.uint8)
        rows, cols = np.where(component > 0)
        if len(rows) < 3:
            continue
        attachment_count += 1
        x0 = int(stats[label, cv2.CC_STAT_LEFT])
        y0 = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        border_touch = x0 == 0 or y0 == 0 or x0 + w >= attachment_mask.shape[1] or y0 + h >= attachment_mask.shape[0]
        border_touch_count += int(border_touch)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        hull_area = max(float(cv2.contourArea(hull)), 1e-9)
        solidity = float(cv2.contourArea(contour) / hull_area)
        points_xy = np.column_stack([cols, rows]).astype(float)
        _, attachment_axis, _, _ = _principal_axis(points_xy)
        alignment = float(abs(np.clip(np.dot(attachment_axis, body_axis), -1.0, 1.0)))
        orientation_mismatch_deg = float(np.degrees(np.arccos(alignment)))
        _, proj_normal_body = _project_points(points_xy, body_center_xy, body_axis)
        side_distance = float(np.mean(np.abs(proj_normal_body)))
        span_x = float(max(w, 1))
        span_y = float(max(h, 1))
        elongation = float(max(span_x, span_y) / max(min(span_x, span_y), 1.0))
        max_elongation = max(max_elongation, elongation)
        max_solidity = max(max_solidity, solidity)
        max_orientation_mismatch_deg = max(max_orientation_mismatch_deg, orientation_mismatch_deg)
        max_distance_to_main_axis_px = max(max_distance_to_main_axis_px, side_distance)
    return {
        "attachment_count": int(attachment_count),
        "attachment_border_touch_count": int(border_touch_count),
        "attachment_max_elongation": float(max_elongation),
        "attachment_max_solidity": float(max_solidity),
        "attachment_max_orientation_mismatch_deg": float(max_orientation_mismatch_deg),
        "attachment_max_distance_to_main_axis_px": float(max_distance_to_main_axis_px),
    }


def _prune_attachment_candidates_from_component(
    component: np.ndarray,
    body_tube_prior_mask: np.ndarray,
    body_center_xy: np.ndarray,
    body_axis: np.ndarray,
    body_positions: np.ndarray,
    width_profile_px: np.ndarray,
    config: BraidedExtractionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    attachment_mask = cv2.subtract(component, body_tube_prior_mask)
    if np.count_nonzero(attachment_mask) == 0:
        return component.copy(), np.zeros_like(component)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(attachment_mask)
    rejected_mask = np.zeros_like(component)
    body_span_px = float(body_positions[-1] - body_positions[0]) if len(body_positions) >= 2 else 0.0
    diameter_max_px = float(np.nanmax(width_profile_px)) if np.any(np.isfinite(width_profile_px)) else 0.0
    short_axis_span_limit_px = max(18.0, float(config.attachment_short_axis_span_ratio) * max(body_span_px, diameter_max_px))
    short_normal_span_limit_px = max(8.0, float(config.attachment_short_normal_span_ratio) * max(diameter_max_px, 1.0))
    far_axis_distance_limit_px = max(6.0, float(config.attachment_far_axis_distance_ratio) * max(diameter_max_px, 1.0))
    mismatch_limit_deg = float(config.attachment_orientation_mismatch_deg)
    dilated_prior = cv2.dilate(body_tube_prior_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(config.attachment_min_area_px2):
            continue
        candidate = (labels == label).astype(np.uint8) * 255
        rows, cols = np.where(candidate > 0)
        if len(rows) < 3:
            continue
        points_xy = np.column_stack([cols, rows]).astype(float)
        proj_axis, proj_normal = _project_points(points_xy, body_center_xy, body_axis)
        mean_axis_pos_px = float(np.mean(proj_axis))
        axis_span_px = float(np.max(proj_axis) - np.min(proj_axis))
        normal_span_px = float(np.max(proj_normal) - np.min(proj_normal))
        mean_side_distance_px = float(np.mean(np.abs(proj_normal)))
        x0 = int(stats[label, cv2.CC_STAT_LEFT])
        y0 = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        border_touch = x0 == 0 or y0 == 0 or x0 + w >= component.shape[1] or y0 + h >= component.shape[0]
        _, candidate_axis, _, _ = _principal_axis(points_xy)
        alignment = float(abs(np.clip(np.dot(candidate_axis, body_axis), -1.0, 1.0)))
        orientation_mismatch_deg = float(np.degrees(np.arccos(alignment)))
        short_spur = axis_span_px <= short_axis_span_limit_px and normal_span_px >= short_normal_span_limit_px
        off_axis = mean_side_distance_px >= far_axis_distance_limit_px
        misaligned_spur = orientation_mismatch_deg >= mismatch_limit_deg and mean_side_distance_px >= 0.35 * max(diameter_max_px, 1.0)
        detached = not bool(np.any((candidate > 0) & (dilated_prior > 0)))
        outboard_margin_px = max(4.0, 0.08 * max(body_span_px, diameter_max_px))
        outboard_spur = detached and short_spur and (
            mean_axis_pos_px <= float(body_positions[0]) - outboard_margin_px
            or mean_axis_pos_px >= float(body_positions[-1]) + outboard_margin_px
        )
        detached_off_axis = detached and off_axis
        border_spur = border_touch and axis_span_px <= max(short_axis_span_limit_px * 3.0, 0.6 * max(body_span_px, diameter_max_px))
        if border_spur or outboard_spur or (short_spur and (off_axis or misaligned_spur)) or detached_off_axis:
            rejected_mask = cv2.bitwise_or(rejected_mask, candidate)

    if np.count_nonzero(rejected_mask) == 0:
        return component.copy(), rejected_mask

    dilate_kernel = max(1, int(config.attachment_prune_dilate_kernel))
    if dilate_kernel > 1:
        if dilate_kernel % 2 == 0:
            dilate_kernel += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
        rejected_mask = cv2.dilate(rejected_mask, kernel)
    cleaned_component = cv2.subtract(component, rejected_mask)
    if np.count_nonzero(cleaned_component) < max(32, int(config.min_component_area * 0.75)):
        return component.copy(), np.zeros_like(component)
    cleaned_component = _largest_component(
        cleaned_component,
        min_area=max(32, int(config.min_component_area * 0.4)),
    )
    return cleaned_component, rejected_mask


def extract_braided_geometry(frame_bgr: np.ndarray, config: BraidedExtractionConfig) -> BraidedExtractionResult:
    source_component, roi_offset_xy = _component_mask(frame_bgr, config)
    source_contour_local = _component_contour(source_component)
    component = source_component.copy()
    (
        contour_local,
        pixel_local_xy,
        step_px,
        body_center_xy,
        body_axis,
        positions,
        axis_points_xy,
        width_profile_px,
        body_mask,
        body_positions,
        body_tube_prior_mask,
    ) = _build_body_prior_from_component(component, config)
    component, _ = _prune_attachment_candidates_from_component(
        component,
        body_tube_prior_mask,
        body_center_xy=body_center_xy,
        body_axis=body_axis,
        body_positions=body_positions,
        width_profile_px=width_profile_px,
        config=config,
    )
    if np.count_nonzero(cv2.absdiff(component, source_component)) > 0:
        (
            contour_local,
            pixel_local_xy,
            step_px,
            body_center_xy,
            body_axis,
            positions,
            axis_points_xy,
            width_profile_px,
            body_mask,
            body_positions,
            body_tube_prior_mask,
        ) = _build_body_prior_from_component(component, config)

    clip_min = float(body_positions[0]) - 0.5 * step_px
    clip_max = float(body_positions[-1]) + 0.5 * step_px
    prior_rows, prior_cols = np.where(body_tube_prior_mask > 0)
    prior_pixel_xy = np.column_stack([prior_cols, prior_rows]).astype(float)
    body_tube_mask = _clip_component_to_axis_span(
        component=body_tube_prior_mask,
        pixel_local_xy=prior_pixel_xy,
        center_xy=body_center_xy,
        axis=body_axis,
        clip_min=clip_min,
        clip_max=clip_max,
        min_area=max(32, int(config.min_component_area * 0.4)),
    )
    if np.count_nonzero(body_tube_mask) < int(config.min_component_area):
        raise RuntimeError("braided observed body mask too small")

    body_rows, body_cols = np.where(body_tube_mask > 0)
    body_pixel_xy = np.column_stack([body_cols, body_rows]).astype(float)
    body_center_xy, body_axis, body_proj_axis, _ = _principal_axis(body_pixel_xy)
    body_contour_local = _mask_contour(body_tube_mask)
    positions, axis_points_xy, width_profile_px, _ = _sample_envelope_profile(
        envelope_xy=body_contour_local,
        center_xy=body_center_xy,
        axis=body_axis,
        step_px=step_px,
    )
    axis_points_xy = _smooth_path(axis_points_xy, window=max(3, int(config.centerline_smooth_window)))
    valid = np.isfinite(width_profile_px)
    if not np.any(valid):
        raise RuntimeError("braided observed-body width profile unavailable")

    diameter_alt_px = float(np.nanmax(width_profile_px))
    peak_idx = int(np.nanargmax(width_profile_px))
    body_mask = compute_braided_body_mask(
        width_profile_px=width_profile_px,
        peak_idx=peak_idx,
        diameter_max_px=diameter_alt_px,
    )
    body_positions = positions[body_mask]
    if len(body_positions) == 0:
        raise RuntimeError("braided observed-body span unavailable")

    clip_min = float(body_positions[0]) - 0.5 * step_px
    clip_max = float(body_positions[-1]) + 0.5 * step_px
    body_tube_mask = _clip_component_to_axis_span(
        component=body_tube_mask,
        pixel_local_xy=body_pixel_xy,
        center_xy=body_center_xy,
        axis=body_axis,
        clip_min=clip_min,
        clip_max=clip_max,
        min_area=max(32, int(config.min_component_area * 0.4)),
    )
    body_rows, body_cols = np.where(body_tube_mask > 0)
    body_pixel_xy = np.column_stack([body_cols, body_rows]).astype(float)
    body_center_xy, body_axis, body_proj_axis, body_proj_normal = _principal_axis(body_pixel_xy)
    body_contour_local = _mask_contour(body_tube_mask)
    positions, axis_points_xy, width_profile_px, _ = _sample_envelope_profile(
        envelope_xy=body_contour_local,
        center_xy=body_center_xy,
        axis=body_axis,
        step_px=step_px,
    )
    axis_points_xy = _smooth_path(axis_points_xy, window=max(3, int(config.centerline_smooth_window)))
    valid = np.isfinite(width_profile_px)
    if not np.any(valid):
        raise RuntimeError("braided final observed-body width profile unavailable")

    peak_idx = int(np.nanargmax(width_profile_px))
    body_mask = compute_braided_body_mask(
        width_profile_px=width_profile_px,
        peak_idx=peak_idx,
        diameter_max_px=float(np.nanmax(width_profile_px)),
    )
    body_positions = positions[body_mask]
    if len(body_positions) == 0:
        raise RuntimeError("braided final observed-body span unavailable")

    body_bin_centerline_local, _ = _extract_centerline_from_body_bins(
        body_tube_mask,
        smooth_window=max(3, int(config.centerline_smooth_window)),
        step_px=step_px,
    )
    prior_centerline_local, _ = _extract_centerline_from_body_bins(
        body_tube_prior_mask,
        smooth_window=max(3, int(config.centerline_smooth_window)),
        step_px=step_px,
    )
    skeleton_centerline_local, branch_component_count_after_pruning = _extract_centerline_from_body(
        body_tube_mask,
        smooth_window=max(3, int(config.centerline_smooth_window)),
        step_px=step_px,
    )
    body_bin_centerline_local = _smooth_path(body_bin_centerline_local, window=max(3, int(config.centerline_smooth_window)))
    body_bin_centerline_local = _resample_polyline(body_bin_centerline_local, step_px=max(0.75 * step_px, 1.0))
    prior_centerline_local = _smooth_path(prior_centerline_local, window=max(3, int(config.centerline_smooth_window)))
    prior_centerline_local = _resample_polyline(prior_centerline_local, step_px=max(0.75 * step_px, 1.0))
    skeleton_centerline_local = _smooth_path(skeleton_centerline_local, window=max(3, int(config.centerline_smooth_window)))
    skeleton_centerline_local = _resample_polyline(skeleton_centerline_local, step_px=max(0.75 * step_px, 1.0))
    skeleton_centerline_local = _align_path_orientation(body_bin_centerline_local, skeleton_centerline_local)
    prior_centerline_local = _align_path_orientation(skeleton_centerline_local, prior_centerline_local)
    _, centerline_local, _, alt_centerline_local = _select_centerline_paths(
        body_tube_mask,
        {
            "skeleton": skeleton_centerline_local,
            "body_bins": body_bin_centerline_local,
            "prior": prior_centerline_local,
        },
    )
    body_bin_centerline_curve_px = cumulative_path_length(body_bin_centerline_local)
    skeleton_curve_px = cumulative_path_length(skeleton_centerline_local)
    centerline_curve_px = cumulative_path_length(centerline_local)
    alt_centerline_curve_px = cumulative_path_length(alt_centerline_local)
    length_axis_skeleton_px = float(skeleton_curve_px[-1]) if len(skeleton_curve_px) else float("nan")
    length_axis_body_bins_px = float(body_bin_centerline_curve_px[-1]) if len(body_bin_centerline_curve_px) else float("nan")

    body_axis_points_xy = axis_points_xy[body_mask]
    if len(body_axis_points_xy) < 2:
        raise RuntimeError("braided provisional body centerline too short")
    body_axis_points_xy = _resample_polyline(body_axis_points_xy, step_px=max(0.75 * step_px, 1.0))
    body_axis_points_xy = _smooth_path(body_axis_points_xy, window=max(3, int(config.centerline_smooth_window)))
    body_axis_points_xy = _align_path_orientation(centerline_local, body_axis_points_xy)
    length_axis_px = float(centerline_curve_px[-1]) if len(centerline_curve_px) else float("nan")
    length_axis_alt_px = float(alt_centerline_curve_px[-1]) if len(alt_centerline_curve_px) else float("nan")
    length_axis_disagreement_px = float(abs(length_axis_px - length_axis_alt_px))
    centerline_disagreement = float(length_axis_disagreement_px / max(length_axis_px, 1e-9))

    anchor_local = centerline_local[0]
    tip_local = centerline_local[-1]
    endpoint_jump_px = _endpoint_gap(centerline_local, alt_centerline_local)

    distance_map = cv2.distanceTransform(body_tube_mask, cv2.DIST_L2, 5)
    thickness_widths_px = np.asarray([2.0 * _nearest_distance(distance_map, point_xy) for point_xy in centerline_local], dtype=float)
    orth_widths_px, _ = _orthogonal_widths(body_tube_mask, centerline_local)
    if not np.any(np.isfinite(orth_widths_px)):
        raise RuntimeError("braided orthogonal width profile unavailable")

    width_profile_trimmed_px = width_profile_px.copy()
    width_profile_trimmed_px[~body_mask] = np.nan
    if not np.any(np.isfinite(width_profile_trimmed_px)):
        raise RuntimeError("braided trimmed width profile unavailable")

    orth_curve_pos_px = cumulative_path_length(centerline_local)
    centerline_axis_proj, _ = _project_points(centerline_local, body_center_xy, body_axis)
    centerline_body_mask = (
        np.isfinite(orth_curve_pos_px)
        & np.isfinite(orth_widths_px)
        & np.isfinite(centerline_axis_proj)
        & (centerline_axis_proj >= clip_min)
        & (centerline_axis_proj <= clip_max)
    )
    if np.count_nonzero(centerline_body_mask) < 2:
        centerline_body_mask = np.isfinite(orth_curve_pos_px) & np.isfinite(orth_widths_px)
    if np.any(centerline_body_mask):
        orth_curve_pos_rel_px = orth_curve_pos_px - float(np.nanmin(orth_curve_pos_px[centerline_body_mask]))
    else:
        orth_curve_pos_rel_px = orth_curve_pos_px.copy()
    body_positions_rel_px = positions.copy() - float(body_positions[0])
    diameter_max_orth_px = float(np.nanmax(orth_widths_px[centerline_body_mask])) if np.any(centerline_body_mask) else float(np.nanmax(orth_widths_px))
    diameter_max_thickness_px = float(np.nanmax(thickness_widths_px))
    diameter_max_feret_px = float(np.max(body_proj_normal) - np.min(body_proj_normal))
    diameter_p95_px = float(np.nanpercentile(width_profile_trimmed_px[np.isfinite(width_profile_trimmed_px)], 95))
    diameter_mid_median_px, diameter_mid_p90_px = _window_width_stats(
        positions_rel_px=body_positions_rel_px,
        width_profile_px=width_profile_trimmed_px,
        window_start_norm=float(config.diameter_proxy_window_start_norm),
        window_end_norm=float(config.diameter_proxy_window_end_norm),
    )
    diameter_max_px = diameter_max_orth_px

    peak_idx = int(np.nanargmax(width_profile_trimmed_px))
    diameter_peak_pos_norm = 0.5
    body_span_px = float(body_positions[-1] - body_positions[0]) if len(body_positions) >= 2 else 0.0
    if body_span_px > 1e-9:
        diameter_peak_pos_norm = float(np.clip((positions[peak_idx] - body_positions[0]) / body_span_px, 0.0, 1.0))
    peak_support = body_mask & np.isfinite(width_profile_trimmed_px) & (
        width_profile_trimmed_px >= float(config.diameter_peak_threshold_ratio) * diameter_max_orth_px
    )
    peak_span_pos = body_positions_rel_px[peak_support]
    diameter_peak_span_px = float(peak_span_pos[-1] - peak_span_pos[0]) if len(peak_span_pos) >= 2 else 0.0
    axis_peak_position_stability = float("nan")

    area_proj_px2 = _integrate_projected_area_px2(
        positions=orth_curve_pos_rel_px,
        width_profile_px=orth_widths_px,
        body_mask=centerline_body_mask,
    )
    area_proj_contour_width_integral_px2 = _integrate_projected_area_px2(
        positions=body_positions_rel_px,
        width_profile_px=width_profile_trimmed_px,
        body_mask=body_mask,
    )
    area_proj_contour_px2 = float(cv2.contourArea(body_contour_local.astype(np.float32)))
    area_proj_definition_gap_px2 = float(area_proj_px2 - area_proj_contour_px2)
    body_mask_area_px2 = float(np.count_nonzero(body_tube_mask > 0))
    component_area_px2 = float(np.count_nonzero(source_component > 0))
    excluded_attachment_mask = cv2.subtract(source_component, body_tube_mask)
    excluded_attachment_area_px2 = float(np.count_nonzero(excluded_attachment_mask > 0))
    body_mask_attachment_leak_fraction = float(excluded_attachment_area_px2 / max(component_area_px2, 1.0))
    attachment_metrics = _attachment_metrics(
        excluded_attachment_mask,
        body_center_xy=body_center_xy,
        body_axis=body_axis,
        min_area_px2=int(config.attachment_min_area_px2),
    )

    body_contour_min, body_contour_max = _project_span_along_axis(body_contour_local, body_center_xy, body_axis)
    length_env_px = float(body_contour_max - body_contour_min)

    zone_metrics = compute_braided_zone_metrics(
        positions=body_positions_rel_px,
        width_profile_px=width_profile_px,
        body_mask=body_mask,
        diameter_max_px=diameter_max_orth_px,
        taper_threshold_ratio=float(config.taper_threshold_ratio),
        compaction_threshold_ratio=float(config.compaction_threshold_ratio),
    )

    quality = float(
        np.clip(
            0.30 * np.count_nonzero(np.isfinite(width_profile_trimmed_px)) / max(len(width_profile_trimmed_px), 1)
            + 0.25 * body_mask_area_px2 / max(component_area_px2, 1.0)
            + 0.20 * np.clip(1.0 - centerline_disagreement / 0.20, 0.0, 1.0)
            + 0.15 * np.clip(1.0 - endpoint_jump_px / max(18.0, 0.10 * max(length_axis_px, 1.0)), 0.0, 1.0)
            + 0.10 * np.clip(1.0 - body_mask_attachment_leak_fraction / 0.12, 0.0, 1.0),
            0.0,
            1.0,
        )
    )

    contour_xy = source_contour_local + roi_offset_xy[None, :]
    body_contour_xy = body_contour_local + roi_offset_xy[None, :]
    axis_line_xy = np.stack([anchor_local, tip_local], axis=0) + roi_offset_xy[None, :]
    sampled_axis_xy = body_axis_points_xy + roi_offset_xy[None, :]
    sampled_centerline_xy = centerline_local + roi_offset_xy[None, :]
    tangents_xy = _path_tangents(centerline_local)
    sampled_width_segments_xy = _qc_segments(
        sampled_axis_xy=sampled_centerline_xy,
        width_profile_px=orth_widths_px,
        tangents_xy=tangents_xy,
        max_segments=int(config.qc_max_segments),
    )

    return BraidedExtractionResult(
        anchor_xy=anchor_local + roi_offset_xy,
        tip_xy=tip_local + roi_offset_xy,
        length_env_px=float(length_env_px),
        length_axis_px=float(length_axis_px),
        length_axis_skeleton_px=float(length_axis_skeleton_px),
        length_axis_body_bins_px=float(length_axis_body_bins_px),
        length_axis_alt_px=float(length_axis_alt_px),
        length_axis_disagreement_px=float(length_axis_disagreement_px),
        diameter_max_px=float(diameter_max_px),
        diameter_max_orth_px=float(diameter_max_orth_px),
        diameter_max_thickness_px=float(diameter_max_thickness_px),
        diameter_max_feret_px=float(diameter_max_feret_px),
        diameter_p95_px=float(diameter_p95_px),
        diameter_mid_median_px=float(diameter_mid_median_px),
        diameter_mid_p90_px=float(diameter_mid_p90_px),
        diameter_peak_span_px=float(diameter_peak_span_px),
        diameter_peak_pos_norm=float(diameter_peak_pos_norm),
        area_proj_px2=float(area_proj_px2),
        area_proj_contour_width_integral_px2=float(area_proj_contour_width_integral_px2),
        area_proj_contour_px2=float(area_proj_contour_px2),
        area_proj_definition_gap_px2=float(area_proj_definition_gap_px2),
        body_mask_area_px2=float(body_mask_area_px2),
        component_area_px2=float(component_area_px2),
        excluded_attachment_area_px2=float(excluded_attachment_area_px2),
        body_mask_attachment_leak_fraction=float(body_mask_attachment_leak_fraction),
        attachment_count=int(attachment_metrics["attachment_count"]),
        attachment_border_touch_count=int(attachment_metrics["attachment_border_touch_count"]),
        attachment_max_elongation=float(attachment_metrics["attachment_max_elongation"]),
        attachment_max_solidity=float(attachment_metrics["attachment_max_solidity"]),
        attachment_max_orientation_mismatch_deg=float(attachment_metrics["attachment_max_orientation_mismatch_deg"]),
        attachment_max_distance_to_main_axis_px=float(attachment_metrics["attachment_max_distance_to_main_axis_px"]),
        x_peak_norm=float(zone_metrics.x_peak_norm),
        taper_left_px=float(zone_metrics.taper_left_px),
        taper_right_px=float(zone_metrics.taper_right_px),
        landing_zone_left_px=float(zone_metrics.landing_zone_left_px),
        landing_zone_right_px=float(zone_metrics.landing_zone_right_px),
        transition_zone_left_px=float(zone_metrics.transition_zone_left_px),
        transition_zone_right_px=float(zone_metrics.transition_zone_right_px),
        compaction_zone_length_px=float(zone_metrics.compaction_zone_length_px),
        zone_symmetry=float(zone_metrics.zone_symmetry),
        branch_component_count_after_pruning=int(branch_component_count_after_pruning),
        branch_count_after_pruning=int(branch_component_count_after_pruning),
        centerline_disagreement=float(centerline_disagreement),
        endpoint_jump_px=float(endpoint_jump_px),
        axis_peak_position_stability=float(axis_peak_position_stability),
        quality=quality,
        mask=source_component,
        body_tube_mask=body_tube_mask,
        excluded_attachment_mask=excluded_attachment_mask,
        contour_xy=contour_xy,
        body_contour_xy=body_contour_xy,
        axis_line_xy=axis_line_xy,
        sampled_axis_xy=sampled_axis_xy,
        sampled_centerline_xy=sampled_centerline_xy,
        sampled_width_segments_xy=sampled_width_segments_xy,
        width_profile_pos_px=body_positions_rel_px,
        width_profile_diameter_px=width_profile_trimmed_px,
        orth_width_profile_pos_px=orth_curve_pos_px,
        orth_width_profile_diameter_px=orth_widths_px,
    )
