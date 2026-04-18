from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


@dataclass
class BraidedSyntheticModelConfig:
    center_xy: tuple[float, float] = (640.0, 260.0)
    length_m_px: float = 520.0
    length_a_px: float = 410.0
    diameter_m_px: float = 120.0
    diameter_a_px: float = 220.0
    transition_temp_c: float = 47.0
    transition_width_c: float = 2.0
    profile_power: float = 0.9


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

    def profile(self, temperature_c: float, sample_count: int = 240) -> tuple[np.ndarray, np.ndarray]:
        length_px = self.length_px(temperature_c)
        diameter_px = self.diameter_px(temperature_c)
        u = np.linspace(0.0, 1.0, sample_count)
        x = (u - 0.5) * length_px + self.config.center_xy[0]
        radius = 0.5 * diameter_px * np.sin(np.pi * u) ** self.config.profile_power
        y_top = self.config.center_xy[1] - radius
        y_bottom = self.config.center_xy[1] + radius
        upper = np.column_stack([x, y_top])
        lower = np.column_stack([x[::-1], y_bottom[::-1]])
        contour = np.vstack([upper, lower])
        width = 2.0 * radius
        return contour, width


def render_braided_frame(
    model: BraidedSyntheticModel,
    temperature_c: float,
    config: BraidedSyntheticRenderConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = np.full((config.image_height, config.image_width, 3), config.background_gray, dtype=np.uint8)
    contour_xy, width_profile_px = model.profile(temperature_c)
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
    return frame, contour_xy, width_profile_px


def write_braided_synthetic_dataset(
    output_dir: str | Path,
    model: BraidedSyntheticModel,
    render_config: BraidedSyntheticRenderConfig,
    schedule: pd.DataFrame,
    taper_threshold_ratio: float,
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
        frame, contour_xy, width_profile_px = render_braided_frame(model, temperature_c, render_config, rng)
        writer.write(frame)
        length_true_px = float(model.length_px(temperature_c))
        diameter_true_px = float(np.max(width_profile_px))
        threshold = taper_threshold_ratio * diameter_true_px
        support = np.flatnonzero(width_profile_px >= threshold)
        if len(support) == 0:
            taper_left_px = np.nan
            taper_right_px = np.nan
        else:
            x_profile = contour_xy[: len(width_profile_px), 0]
            taper_left_px = float(x_profile[support[0]] - x_profile[0])
            taper_right_px = float(x_profile[-1] - x_profile[support[-1]])
        rows.append(
            {
                "frame": int(row.frame),
                "time_sec": float(row.time_sec),
                "temperature_c": temperature_c,
                "length_true_px": length_true_px,
                "diameter_true_px": diameter_true_px,
                "x_peak_norm_true": 0.5,
                "taper_left_true_px": taper_left_px,
                "taper_right_true_px": taper_right_px,
            }
        )
    writer.release()
    truth = pd.DataFrame(rows)
    truth.to_csv(output_dir / "truth.csv", index=False)
    return truth
