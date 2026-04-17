from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RecoveryModelConfig:
    needle_length_px: float = 255.0
    anchor_xy: tuple[float, float] = (645.0, 38.0)
    anchor_angle_deg: float = 108.0
    kappa_m: float = 0.0115
    kappa_a: float = 0.00005
    transition_temp_c: float = 47.0
    transition_width_c: float = 2.0
    samples_along_length: int = 160


class RecoveryModel:
    """Single-bend, inextensible planar recovery model."""

    def __init__(self, config: RecoveryModelConfig):
        self.config = config

    def curvature(self, temperature_c: np.ndarray | float) -> np.ndarray:
        t = np.asarray(temperature_c, dtype=float)
        c = self.config
        return c.kappa_a + (c.kappa_m - c.kappa_a) / (
            1.0 + np.exp((t - c.transition_temp_c) / c.transition_width_c)
        )

    def centerline(self, temperature_c: float) -> np.ndarray:
        c = self.config
        s = np.linspace(0.0, c.needle_length_px, c.samples_along_length)
        ds = s[1] - s[0]
        theta0 = np.deg2rad(c.anchor_angle_deg)
        kappa = float(self.curvature(temperature_c))
        theta = theta0 + kappa * s
        x = c.anchor_xy[0] + np.cumsum(np.cos(theta) * ds)
        y = c.anchor_xy[1] + np.cumsum(np.sin(theta) * ds)
        x[0], y[0] = c.anchor_xy
        return np.column_stack([x, y])

    def x_metric(self, temperature_c: float) -> float:
        line = self.centerline(temperature_c)
        return float(np.linalg.norm(line[-1] - line[0]))
