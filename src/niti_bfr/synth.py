from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .metrics import af_95, af_tan, recovery_ratio
from .model import RecoveryModel, RecoveryModelConfig


@dataclass
class SyntheticRenderConfig:
    image_width: int = 1280
    image_height: int = 848
    fps: int = 20
    duration_sec: float = 24.0
    line_thickness_px: int = 5
    background_gray: int = 236
    noise_sigma: float = 2.0
    blur_sigma: float = 0.8


def generate_temperature_schedule(
    fps: int,
    duration_sec: float,
    start_c: float,
    end_c: float,
) -> pd.DataFrame:
    n = int(round(fps * duration_sec))
    frame = np.arange(n)
    time_sec = frame / fps
    temperature_c = np.linspace(start_c, end_c, n)
    return pd.DataFrame({"frame": frame, "time_sec": time_sec, "temperature_c": temperature_c})


def render_frame(
    centerline_xy: np.ndarray,
    config: SyntheticRenderConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    frame = np.full(
        (config.image_height, config.image_width, 3),
        config.background_gray,
        dtype=np.uint8,
    )
    cv2.ellipse(
        frame,
        center=(config.image_width // 2, config.image_height // 2),
        axes=(config.image_width // 2 - 24, config.image_height // 2 - 24),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=(228, 228, 228),
        thickness=12,
    )
    pts = np.round(centerline_xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(
        frame,
        [pts],
        isClosed=False,
        color=(35, 35, 35),
        thickness=config.line_thickness_px,
        lineType=cv2.LINE_AA,
    )
    if config.blur_sigma > 0:
        frame = cv2.GaussianBlur(frame, (0, 0), sigmaX=config.blur_sigma, sigmaY=config.blur_sigma)
    if config.noise_sigma > 0:
        noise = rng.normal(0.0, config.noise_sigma, frame.shape).astype(np.float32)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return frame


def write_synthetic_dataset(
    output_dir: str | Path,
    model: RecoveryModel,
    render_config: SyntheticRenderConfig,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
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
        centerline = model.centerline(temperature_c)
        frame = render_frame(centerline, render_config, rng)
        writer.write(frame)
        x_px = float(np.linalg.norm(centerline[-1] - centerline[0]))
        kappa_true = float(model.curvature(temperature_c))
        rows.append(
            {
                "frame": int(row.frame),
                "time_sec": float(row.time_sec),
                "temperature_c": temperature_c,
                "x_true_px": x_px,
                "kappa_true_px_inv": kappa_true,
                "anchor_x": float(centerline[0, 0]),
                "anchor_y": float(centerline[0, 1]),
                "tip_x": float(centerline[-1, 0]),
                "tip_y": float(centerline[-1, 1]),
            }
        )
    writer.release()
    truth = pd.DataFrame(rows)
    x_m = float(truth["x_true_px"].iloc[0])
    x_a = float(truth["x_true_px"].iloc[-1])
    kappa_m = float(truth["kappa_true_px_inv"].iloc[0])
    kappa_a = float(truth["kappa_true_px_inv"].iloc[-1])
    truth["x_recovery_true"] = recovery_ratio(truth["x_true_px"], x_m=x_m, x_a=x_a)
    truth["kappa_recovery_true"] = np.clip(
        (kappa_m - truth["kappa_true_px_inv"]) / max(kappa_m - kappa_a, 1e-9),
        0.0,
        1.0,
    )
    truth.to_csv(output_dir / "temperature_truth.csv", index=False)
    metrics = {
        "x_m_px": x_m,
        "x_a_px": x_a,
        "x_af95_c": af_95(truth["temperature_c"], truth["x_recovery_true"]),
        "x_aftan_c": af_tan(truth["temperature_c"], truth["x_recovery_true"]),
        "kappa_m_px_inv": kappa_m,
        "kappa_a_px_inv": kappa_a,
        "kappa_af95_c": af_95(truth["temperature_c"], truth["kappa_recovery_true"]),
        "kappa_aftan_c": af_tan(truth["temperature_c"], truth["kappa_recovery_true"]),
    }
    (output_dir / "metrics.json").write_text(
        "{\n"
        + ",\n".join(f'  "{k}": {float(v):.6f}' for k, v in metrics.items())
        + "\n}\n",
        encoding="utf-8",
    )
    return truth


def build_model_from_dict(cfg: dict) -> RecoveryModel:
    model_cfg = RecoveryModelConfig(
        needle_length_px=float(cfg["needle_length_px"]),
        anchor_xy=tuple(float(v) for v in cfg["anchor_xy"]),
        anchor_angle_deg=float(cfg["anchor_angle_deg"]),
        kappa_m=float(cfg["kappa_m"]),
        kappa_a=float(cfg["kappa_a"]),
        transition_temp_c=float(cfg["transition_temp_c"]),
        transition_width_c=float(cfg["transition_width_c"]),
    )
    return RecoveryModel(model_cfg)
