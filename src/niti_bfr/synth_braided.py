from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .extract_braided import (
    compute_braided_body_mask,
    compute_braided_zone_metrics,
    cumulative_path_length,
    rasterize_braided_body_tube_mask,
)


@dataclass
class BraidedSyntheticModelConfig:
    center_xy: tuple[float, float] = (640.0, 260.0)
    length_m_px: float = 520.0
    length_a_px: float = 410.0
    diameter_m_px: float = 120.0
    diameter_a_px: float = 220.0
    transition_temp_c: float = 47.0
    transition_width_c: float = 2.0
    peak_shift_norm: float = 0.56
    left_profile_power: float = 0.88
    right_profile_power: float = 1.06
    bow_m_px: float = 14.0
    bow_a_px: float = 8.0
    axis_angle_m_deg: float = -1.5
    axis_angle_a_deg: float = 2.0
    center_shift_a_xy: tuple[float, float] = (10.0, -6.0)


@dataclass
class BraidedSyntheticRenderConfig:
    image_width: int = 1280
    image_height: int = 848
    fps: int = 20
    duration_sec: float = 24.0
    background_gray: int = 236
    body_gray: int = 120
    wire_gray: int = 70
    outline_gray: int = 60
    braid_spacing_px: int = 28
    wire_thickness_px: int = 2
    outline_thickness_px: int = 2
    support_gray: int = 178
    support_length_px: int = 52
    support_radius_px: int = 6
    tip_cap_gray: int = 98
    tip_cap_length_px: int = 20
    tip_cap_radius_px: int = 10
    noise_sigma: float = 2.0
    blur_sigma: float = 0.8


class BraidedSyntheticModel:
    def __init__(self, config: BraidedSyntheticModelConfig) -> None:
        self.config = config

    def _sigmoid(self, temperature_c: float) -> float:
        cfg = self.config
        return 1.0 / (1.0 + np.exp(-(temperature_c - cfg.transition_temp_c) / cfg.transition_width_c))

    def length_px(self, temperature_c: float) -> float:
        cfg = self.config
        alpha = self._sigmoid(temperature_c)
        return float(cfg.length_m_px + alpha * (cfg.length_a_px - cfg.length_m_px))

    def diameter_px(self, temperature_c: float) -> float:
        cfg = self.config
        alpha = self._sigmoid(temperature_c)
        return float(cfg.diameter_m_px + alpha * (cfg.diameter_a_px - cfg.diameter_m_px))

    def bow_px(self, temperature_c: float) -> float:
        cfg = self.config
        alpha = self._sigmoid(temperature_c)
        return float(cfg.bow_m_px + alpha * (cfg.bow_a_px - cfg.bow_m_px))

    def axis_angle_rad(self, temperature_c: float) -> float:
        cfg = self.config
        alpha = self._sigmoid(temperature_c)
        angle_deg = cfg.axis_angle_m_deg + alpha * (cfg.axis_angle_a_deg - cfg.axis_angle_m_deg)
        return float(np.deg2rad(angle_deg))

    def center_xy_at(self, temperature_c: float) -> np.ndarray:
        cfg = self.config
        alpha = self._sigmoid(temperature_c)
        base = np.asarray(cfg.center_xy, dtype=float)
        shift = alpha * np.asarray(cfg.center_shift_a_xy, dtype=float)
        return base + shift

    def _width_factor(self, u: np.ndarray) -> np.ndarray:
        cfg = self.config
        peak = float(np.clip(cfg.peak_shift_norm, 0.2, 0.8))
        left = np.clip(u / peak, 0.0, 1.0)
        right = np.clip((1.0 - u) / (1.0 - peak), 0.0, 1.0)
        factor = np.empty_like(u)
        left_mask = u <= peak
        factor[left_mask] = np.sin(0.5 * np.pi * left[left_mask]) ** cfg.left_profile_power
        factor[~left_mask] = np.sin(0.5 * np.pi * right[~left_mask]) ** cfg.right_profile_power
        return factor

    def profile(
        self,
        temperature_c: float,
        sample_count: int = 240,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        length_px = self.length_px(temperature_c)
        diameter_px = self.diameter_px(temperature_c)
        bow_px = self.bow_px(temperature_c)
        center_xy = self.center_xy_at(temperature_c)
        axis_angle = self.axis_angle_rad(temperature_c)

        u = np.linspace(0.0, 1.0, sample_count)
        x_local = (u - 0.5) * length_px
        y_local = bow_px * np.sin(np.pi * u)
        centerline_local = np.column_stack([x_local, y_local])
        delta = np.gradient(centerline_local, axis=0)
        tangent = delta / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-9)
        normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])

        radius = 0.5 * diameter_px * self._width_factor(u)
        upper_local = centerline_local + radius[:, None] * normal
        lower_local = centerline_local - radius[:, None] * normal

        rotation = np.array(
            [[np.cos(axis_angle), -np.sin(axis_angle)], [np.sin(axis_angle), np.cos(axis_angle)]],
            dtype=float,
        )
        centerline = centerline_local @ rotation.T + center_xy[None, :]
        upper = upper_local @ rotation.T + center_xy[None, :]
        lower = lower_local @ rotation.T + center_xy[None, :]
        contour = np.vstack([upper, lower[::-1]])
        width = 2.0 * radius
        return contour, centerline, width


def _project_span(points_xy: np.ndarray) -> tuple[np.ndarray, float, float]:
    center_xy = points_xy.mean(axis=0)
    centered = points_xy - center_xy[None, :]
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    if axis[0] < 0.0:
        axis = -axis
    proj = centered @ axis
    return axis, float(np.min(proj)), float(np.max(proj))


def _truth_metrics_from_geometry(
    contour_xy: np.ndarray,
    centerline_xy: np.ndarray,
    width_profile_px: np.ndarray,
    taper_threshold_ratio: float,
    compaction_threshold_ratio: float,
    image_shape: tuple[int, int],
) -> dict[str, float]:
    curve_positions_px = cumulative_path_length(centerline_xy)
    diameter_true_px = float(np.nanmax(width_profile_px))
    peak_idx = int(np.nanargmax(width_profile_px))
    body_mask = compute_braided_body_mask(
        width_profile_px=width_profile_px,
        peak_idx=peak_idx,
        diameter_max_px=diameter_true_px,
    )
    body_curve_positions_px = curve_positions_px[body_mask]
    axis_span = (
        float(body_curve_positions_px[-1] - body_curve_positions_px[0])
        if len(body_curve_positions_px) >= 2
        else 0.0
    )
    body_centerline_xy = centerline_xy[body_mask]
    body_axis, _, _ = _project_span(body_centerline_xy)
    body_centerline_center = body_centerline_xy.mean(axis=0)
    centerline_proj = (body_centerline_xy - body_centerline_center[None, :]) @ body_axis
    axis_alt_span = float(np.max(centerline_proj) - np.min(centerline_proj)) if len(centerline_proj) else 0.0
    body_tube_mask = rasterize_braided_body_tube_mask(
        centerline_xy=centerline_xy,
        width_profile_px=width_profile_px,
        body_mask=body_mask,
        image_shape=image_shape,
    )
    body_mask_area_true_px2 = float(np.count_nonzero(body_tube_mask > 0))
    body_rows, body_cols = np.where(body_tube_mask > 0)
    body_pixel_xy = np.column_stack([body_cols, body_rows]).astype(float)
    mask_axis, _, _ = _project_span(body_pixel_xy)
    body_center = body_pixel_xy.mean(axis=0)
    body_proj = (body_pixel_xy - body_center[None, :]) @ mask_axis
    body_normal = np.array([-mask_axis[1], mask_axis[0]], dtype=float)
    body_normal_proj = (body_pixel_xy - body_center[None, :]) @ body_normal
    contours, _ = cv2.findContours(body_tube_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    body_contour_area_px2 = float(cv2.contourArea(max(contours, key=cv2.contourArea))) if contours else body_mask_area_true_px2
    positions_curve_rel = curve_positions_px - float(body_curve_positions_px[0])
    area_proj_true_px2 = (
        float(np.trapz(width_profile_px[body_mask], positions_curve_rel[body_mask]))
        if np.count_nonzero(body_mask) >= 2
        else float("nan")
    )
    zone_metrics = compute_braided_zone_metrics(
        positions=positions_curve_rel,
        width_profile_px=width_profile_px,
        body_mask=body_mask,
        diameter_max_px=diameter_true_px,
        taper_threshold_ratio=taper_threshold_ratio,
        compaction_threshold_ratio=compaction_threshold_ratio,
    )
    diameter_peak_pos_norm_true = zone_metrics.x_peak_norm
    peak_support = width_profile_px[body_mask] >= 0.95 * diameter_true_px
    peak_body_pos = positions_curve_rel[body_mask][peak_support]
    diameter_peak_span_true_px = float(peak_body_pos[-1] - peak_body_pos[0]) if len(peak_body_pos) >= 2 else 0.0
    return {
        "length_env_true_px": float(np.max(body_proj) - np.min(body_proj)),
        "length_axis_true_px": axis_span,
        "length_axis_alt_true_px": axis_alt_span,
        "length_axis_definition_gap_true_px": float(abs(axis_span - axis_alt_span)),
        "diameter_true_px": diameter_true_px,
        "diameter_max_orth_true_px": diameter_true_px,
        "diameter_max_thickness_true_px": diameter_true_px,
        "diameter_max_feret_true_px": float(np.max(body_normal_proj) - np.min(body_normal_proj)),
        "diameter_p95_true_px": float(np.nanpercentile(width_profile_px[body_mask], 95)),
        "diameter_peak_span_true_px": diameter_peak_span_true_px,
        "diameter_peak_pos_norm_true": diameter_peak_pos_norm_true,
        "area_proj_true_px2": area_proj_true_px2,
        "area_proj_contour_true_px2": body_contour_area_px2,
        "area_proj_definition_gap_true_px2": float(area_proj_true_px2 - body_contour_area_px2),
        "body_mask_area_true_px2": body_mask_area_true_px2,
        "body_mask_attachment_leak_fraction_true": 0.0,
        "x_peak_norm_true": zone_metrics.x_peak_norm,
        "taper_left_true_px": zone_metrics.taper_left_px,
        "taper_right_true_px": zone_metrics.taper_right_px,
        "landing_zone_left_true_px": zone_metrics.landing_zone_left_px,
        "landing_zone_right_true_px": zone_metrics.landing_zone_right_px,
        "transition_zone_left_true_px": zone_metrics.transition_zone_left_px,
        "transition_zone_right_true_px": zone_metrics.transition_zone_right_px,
        "compaction_zone_length_true_px": zone_metrics.compaction_zone_length_px,
        "zone_symmetry_true": zone_metrics.zone_symmetry,
    }


def _draw_support_and_tip(
    frame: np.ndarray,
    centerline_xy: np.ndarray,
    config: BraidedSyntheticRenderConfig,
) -> None:
    start_xy = centerline_xy[0]
    start_dir = centerline_xy[1] - centerline_xy[0]
    start_dir = start_dir / max(np.linalg.norm(start_dir), 1e-9)
    end_xy = centerline_xy[-1]
    end_dir = centerline_xy[-1] - centerline_xy[-2]
    end_dir = end_dir / max(np.linalg.norm(end_dir), 1e-9)

    support_end = start_xy - config.support_length_px * start_dir
    cv2.line(
        frame,
        np.round(start_xy).astype(int),
        np.round(support_end).astype(int),
        (config.support_gray, config.support_gray, config.support_gray),
        max(2, config.support_radius_px * 2),
        cv2.LINE_AA,
    )
    cv2.circle(
        frame,
        np.round(support_end).astype(int),
        config.support_radius_px,
        (config.support_gray, config.support_gray, config.support_gray),
        -1,
        cv2.LINE_AA,
    )

    tip_center = end_xy + 0.5 * config.tip_cap_length_px * end_dir
    angle_deg = float(np.degrees(np.arctan2(end_dir[1], end_dir[0])))
    cv2.ellipse(
        frame,
        center=np.round(tip_center).astype(int),
        axes=(config.tip_cap_length_px // 2, config.tip_cap_radius_px),
        angle=angle_deg,
        startAngle=0,
        endAngle=360,
        color=(config.tip_cap_gray, config.tip_cap_gray, config.tip_cap_gray),
        thickness=-1,
        lineType=cv2.LINE_AA,
    )


def render_braided_frame(
    model: BraidedSyntheticModel,
    temperature_c: float,
    config: BraidedSyntheticRenderConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = np.full((config.image_height, config.image_width, 3), config.background_gray, dtype=np.uint8)
    yy, xx = np.mgrid[0 : config.image_height, 0 : config.image_width]
    vignette = 1.0 - 0.06 * (
        ((xx - 0.55 * config.image_width) / config.image_width) ** 2
        + ((yy - 0.48 * config.image_height) / config.image_height) ** 2
    )
    frame = np.clip(frame.astype(np.float32) * vignette[..., None], 0, 255).astype(np.uint8)

    contour_xy, centerline_xy, width_profile_px = model.profile(temperature_c)
    contour_int = np.round(contour_xy).astype(np.int32)
    device_mask = np.zeros((config.image_height, config.image_width), dtype=np.uint8)
    cv2.fillPoly(device_mask, [contour_int], 255)
    frame[device_mask > 0] = config.body_gray

    mesh_mask = np.zeros_like(device_mask)
    diag_limit = config.image_width + config.image_height
    spacing = max(8, int(config.braid_spacing_px))
    for offset in range(-diag_limit, diag_limit + spacing, spacing):
        p0 = (offset, 0)
        p1 = (offset + config.image_height, config.image_height)
        cv2.line(mesh_mask, p0, p1, 255, config.wire_thickness_px, cv2.LINE_AA)
        q0 = (offset, config.image_height)
        q1 = (offset + config.image_height, 0)
        cv2.line(mesh_mask, q0, q1, 255, config.wire_thickness_px, cv2.LINE_AA)
    mesh_mask = cv2.bitwise_and(mesh_mask, device_mask)
    frame[mesh_mask > 0] = config.wire_gray

    _draw_support_and_tip(frame, centerline_xy, config)
    cv2.polylines(
        frame,
        [contour_int.reshape(-1, 1, 2)],
        isClosed=True,
        color=(config.outline_gray, config.outline_gray, config.outline_gray),
        thickness=config.outline_thickness_px,
        lineType=cv2.LINE_AA,
    )

    if config.blur_sigma > 0:
        frame = cv2.GaussianBlur(frame, (0, 0), sigmaX=config.blur_sigma, sigmaY=config.blur_sigma)
    if config.noise_sigma > 0:
        noise = rng.normal(0.0, config.noise_sigma, frame.shape).astype(np.float32)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return frame, contour_xy, centerline_xy, width_profile_px


def write_braided_synthetic_dataset(
    output_dir: str | Path,
    model: BraidedSyntheticModel,
    render_config: BraidedSyntheticRenderConfig,
    schedule: pd.DataFrame,
    taper_threshold_ratio: float,
    compaction_threshold_ratio: float = 0.75,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    video_path = output_dir / "synthetic.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        render_config.fps,
        (render_config.image_width, render_config.image_height),
    )
    rows: list[dict[str, float]] = []
    for row in schedule.itertuples(index=False):
        temperature_c = float(row.temperature_c)
        frame, contour_xy, centerline_xy, width_profile_px = render_braided_frame(model, temperature_c, render_config, rng)
        writer.write(frame)
        rows.append(
            {
                "frame": int(row.frame),
                "time_sec": float(row.time_sec),
                "temperature_c": temperature_c,
                **_truth_metrics_from_geometry(
                    contour_xy=contour_xy,
                    centerline_xy=centerline_xy,
                    width_profile_px=width_profile_px,
                    taper_threshold_ratio=taper_threshold_ratio,
                    compaction_threshold_ratio=compaction_threshold_ratio,
                    image_shape=(render_config.image_height, render_config.image_width),
                ),
            }
        )
    writer.release()
    truth = pd.DataFrame(rows)
    axis_ref = float(truth["length_axis_true_px"].max())
    env_ref = float(truth["length_env_true_px"].max())
    truth["length_axis_ref_true_px"] = axis_ref
    truth["length_env_ref_true_px"] = env_ref
    truth["foreshortening_axis_true"] = 1.0 - truth["length_axis_true_px"] / max(axis_ref, 1e-9)
    truth["foreshortening_env_true"] = 1.0 - truth["length_env_true_px"] / max(env_ref, 1e-9)
    truth.to_csv(output_dir / "truth.csv", index=False)
    return truth
