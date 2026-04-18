from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


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
    qc_max_segments: int = 15


@dataclass
class BraidedExtractionResult:
    anchor_xy: np.ndarray
    tip_xy: np.ndarray
    length_env_px: float
    length_axis_px: float
    diameter_max_px: float
    x_peak_norm: float
    taper_left_px: float
    taper_right_px: float
    quality: float
    mask: np.ndarray
    contour_xy: np.ndarray
    axis_line_xy: np.ndarray
    sampled_axis_xy: np.ndarray
    sampled_width_segments_xy: np.ndarray
    width_profile_pos_px: np.ndarray
    width_profile_diameter_px: np.ndarray


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


def _sample_width_profile(
    pixel_xy: np.ndarray,
    center_xy: np.ndarray,
    axis: np.ndarray,
    proj_axis: np.ndarray,
    proj_normal: np.ndarray,
    step_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        pixel_bin = pixel_xy[in_bin]
        axis_points.append(pixel_bin.mean(axis=0))
        diameters.append(float(np.max(normal_vals) - np.min(normal_vals)))
    return positions, np.asarray(axis_points), np.asarray(diameters, dtype=float), np.asarray(positions, dtype=float)


def _qc_segments(
    sampled_axis_xy: np.ndarray,
    width_profile_px: np.ndarray,
    axis: np.ndarray,
    max_segments: int,
) -> np.ndarray:
    valid_idx = np.flatnonzero(np.isfinite(width_profile_px))
    if len(valid_idx) == 0:
        return np.empty((0, 2, 2), dtype=float)
    keep = valid_idx
    if len(keep) > max_segments:
        keep = np.linspace(0, len(valid_idx) - 1, max_segments, dtype=int)
        keep = valid_idx[keep]
    normal = np.array([-axis[1], axis[0]], dtype=float)
    segments = []
    for idx in keep:
        center_xy = sampled_axis_xy[idx]
        radius = 0.5 * float(width_profile_px[idx])
        if not np.all(np.isfinite(center_xy)) or not np.isfinite(radius):
            continue
        segments.append(np.stack([center_xy - radius * normal, center_xy + radius * normal], axis=0))
    if not segments:
        return np.empty((0, 2, 2), dtype=float)
    return np.asarray(segments, dtype=float)


def extract_braided_geometry(frame_bgr: np.ndarray, config: BraidedExtractionConfig) -> BraidedExtractionResult:
    component, roi_offset_xy = _component_mask(frame_bgr, config)
    contour_local = _component_contour(component)
    rows, cols = np.where(component > 0)
    if len(rows) < int(config.min_component_area):
        raise RuntimeError("braided component too small")
    pixel_local_xy = np.column_stack([cols, rows]).astype(float)
    center_xy, axis, proj_axis, proj_normal = _principal_axis(pixel_local_xy)
    min_proj = float(np.min(proj_axis))
    max_proj = float(np.max(proj_axis))
    length_env_px = max_proj - min_proj
    anchor_local = center_xy + min_proj * axis
    tip_local = center_xy + max_proj * axis

    positions, axis_points_xy, width_profile_px, width_profile_pos_px = _sample_width_profile(
        pixel_xy=pixel_local_xy,
        center_xy=center_xy,
        axis=axis,
        proj_axis=proj_axis,
        proj_normal=proj_normal,
        step_px=float(config.width_sampling_step_px),
    )
    valid = np.isfinite(width_profile_px)
    if not np.any(valid):
        raise RuntimeError("braided width profile unavailable")

    diameter_max_px = float(np.nanmax(width_profile_px))
    if diameter_max_px <= 0.0:
        raise RuntimeError("braided max diameter unavailable")

    valid_positions = positions[valid]
    length_axis_px = float(valid_positions[-1] - valid_positions[0]) if len(valid_positions) >= 2 else float(length_env_px)
    peak_idx = int(np.nanargmax(width_profile_px))
    peak_pos = float(positions[peak_idx])
    x_peak_norm = 0.5
    if length_axis_px > 1e-9:
        x_peak_norm = float(np.clip((peak_pos - valid_positions[0]) / length_axis_px, 0.0, 1.0))

    taper_threshold = float(config.taper_threshold_ratio) * diameter_max_px
    above = np.flatnonzero(valid & (width_profile_px >= taper_threshold))
    if len(above) == 0:
        taper_left_px = np.nan
        taper_right_px = np.nan
    else:
        left_boundary = float(positions[above[0]])
        right_boundary = float(positions[above[-1]])
        taper_left_px = max(0.0, left_boundary - valid_positions[0])
        taper_right_px = max(0.0, valid_positions[-1] - right_boundary)

    quality = float(np.clip(np.count_nonzero(valid) / max(len(width_profile_px), 1), 0.0, 1.0))
    contour_xy = contour_local + roi_offset_xy[None, :]
    axis_line_xy = np.stack([anchor_local, tip_local], axis=0) + roi_offset_xy[None, :]
    sampled_axis_xy = axis_points_xy + roi_offset_xy[None, :]
    sampled_width_segments_xy = _qc_segments(sampled_axis_xy, width_profile_px, axis, int(config.qc_max_segments))

    return BraidedExtractionResult(
        anchor_xy=anchor_local + roi_offset_xy,
        tip_xy=tip_local + roi_offset_xy,
        length_env_px=float(length_env_px),
        length_axis_px=float(length_axis_px),
        diameter_max_px=diameter_max_px,
        x_peak_norm=x_peak_norm,
        taper_left_px=float(taper_left_px),
        taper_right_px=float(taper_right_px),
        quality=quality,
        mask=component,
        contour_xy=contour_xy,
        axis_line_xy=axis_line_xy,
        sampled_axis_xy=sampled_axis_xy,
        sampled_width_segments_xy=sampled_width_segments_xy,
        width_profile_pos_px=width_profile_pos_px,
        width_profile_diameter_px=width_profile_px,
    )
