from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RouteCConfig:
    endpoint_blend_route_a: float = 0.2
    ema_alpha: float = 0.35
    ema_alpha_floor: float = 0.08
    enforce_monotonic_x: bool = True
    enforce_monotonic_kappa: bool = True


def _fill_nan_linear(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values.copy()
    idx = np.arange(len(values), dtype=float)
    valid = np.isfinite(values)
    if not np.any(valid):
        return np.full_like(values, np.nan)
    if np.count_nonzero(valid) == 1:
        return np.full_like(values, float(values[valid][0]))
    filled = values.copy()
    filled[~valid] = np.interp(idx[~valid], idx[valid], values[valid])
    return filled


def _ema(values: np.ndarray, alpha: float | np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    alpha_arr = np.asarray(alpha, dtype=float)
    if alpha_arr.ndim == 0:
        alpha_arr = np.full(len(values), float(alpha_arr))
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        a = float(np.clip(alpha_arr[i], 0.0, 1.0))
        out[i] = a * values[i] + (1.0 - a) * out[i - 1]
    return out


def _bidirectional_ema(values: np.ndarray, alpha: float | np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values.copy()
    alpha_arr = np.asarray(alpha, dtype=float)
    if alpha_arr.ndim == 0:
        alpha_arr = np.full(len(values), float(alpha_arr))
    forward = _ema(values, alpha_arr)
    backward = _ema(values[::-1], alpha_arr[::-1])[::-1]
    return 0.5 * (forward + backward)


def _isotonic_regression(values: np.ndarray, increasing: bool) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    if not increasing:
        return -_isotonic_regression(-y, increasing=True)
    blocks: list[list[float]] = []
    weights: list[int] = []
    starts: list[int] = []
    for idx, value in enumerate(y):
        blocks.append([float(value)])
        weights.append(1)
        starts.append(idx)
        while len(blocks) >= 2:
            mean_prev = sum(blocks[-2]) / weights[-2]
            mean_curr = sum(blocks[-1]) / weights[-1]
            if mean_prev <= mean_curr:
                break
            merged_block = blocks[-2] + blocks[-1]
            merged_weight = weights[-2] + weights[-1]
            blocks[-2] = merged_block
            weights[-2] = merged_weight
            blocks.pop()
            weights.pop()
            starts.pop()
    out = np.empty_like(y)
    pos = 0
    for block, weight in zip(blocks, weights):
        mean_value = sum(block) / weight
        out[pos : pos + weight] = mean_value
        pos += weight
    return out


def temporal_regularize(
    values: np.ndarray,
    increasing: bool,
    alpha: float | np.ndarray,
    enforce_monotonic: bool,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values.copy()
    valid = np.isfinite(values)
    if not np.any(valid):
        return np.full_like(values, np.nan)
    filled = _fill_nan_linear(values)
    smooth = _bidirectional_ema(filled, alpha)
    if enforce_monotonic:
        smooth = _isotonic_regression(smooth, increasing=increasing)
    out = smooth.copy()
    out[~valid] = np.nan
    return out


def temporal_regularize_angle(values_rad: np.ndarray, alpha: float | np.ndarray) -> np.ndarray:
    values = np.asarray(values_rad, dtype=float)
    if len(values) == 0:
        return values.copy()
    valid = np.isfinite(values)
    if not np.any(valid):
        return np.full_like(values, np.nan)
    out = np.full_like(values, np.nan)
    unwrapped_valid = np.unwrap(values[valid])
    filled = _fill_nan_linear(
        np.interp(
            np.arange(len(values), dtype=float),
            np.flatnonzero(valid).astype(float),
            unwrapped_valid,
        )
    )
    smooth = _bidirectional_ema(filled, alpha)
    out[:] = smooth
    out[~valid] = np.nan
    return out


def apply_route_c(series: pd.DataFrame, config: RouteCConfig) -> pd.DataFrame:
    series = series.copy()
    w = float(np.clip(config.endpoint_blend_route_a, 0.0, 1.0))
    alpha = float(np.clip(config.ema_alpha, 0.01, 0.99))
    alpha_floor = float(np.clip(config.ema_alpha_floor, 0.01, alpha))
    quality = series.get("quality", pd.Series(np.ones(len(series), dtype=float))).to_numpy(dtype=float)
    quality = np.clip(np.nan_to_num(quality, nan=0.0), 0.0, 1.0)
    alpha_series = alpha_floor + (alpha - alpha_floor) * quality

    anchor_route_a_x = series["route_a_anchor_x"].to_numpy(dtype=float)
    anchor_route_a_y = series["route_a_anchor_y"].to_numpy(dtype=float)
    tip_route_a_x = series["route_a_tip_x"].to_numpy(dtype=float)
    tip_route_a_y = series["route_a_tip_y"].to_numpy(dtype=float)
    anchor_route_b_x = series["anchor_x"].to_numpy(dtype=float)
    anchor_route_b_y = series["anchor_y"].to_numpy(dtype=float)
    tip_route_b_x = series["tip_x"].to_numpy(dtype=float)
    tip_route_b_y = series["tip_y"].to_numpy(dtype=float)

    anchor_raw_x = (1.0 - w) * anchor_route_b_x + w * anchor_route_a_x
    anchor_raw_y = (1.0 - w) * anchor_route_b_y + w * anchor_route_a_y
    tip_raw_x = (1.0 - w) * tip_route_b_x + w * tip_route_a_x
    tip_raw_y = (1.0 - w) * tip_route_b_y + w * tip_route_a_y

    anchor_smooth_x = temporal_regularize(
        anchor_raw_x,
        increasing=False,
        alpha=alpha_series,
        enforce_monotonic=False,
    )
    anchor_smooth_y = temporal_regularize(
        anchor_raw_y,
        increasing=False,
        alpha=alpha_series,
        enforce_monotonic=False,
    )

    dx_raw = tip_raw_x - anchor_raw_x
    dy_raw = tip_raw_y - anchor_raw_y
    x_raw = np.sqrt(dx_raw**2 + dy_raw**2)
    theta_raw = np.arctan2(dy_raw, dx_raw)

    x_smooth = temporal_regularize(
        x_raw,
        increasing=True,
        alpha=alpha_series,
        enforce_monotonic=config.enforce_monotonic_x,
    )
    theta_smooth = temporal_regularize_angle(theta_raw, alpha=alpha_series)

    tip_smooth_x = anchor_smooth_x + x_smooth * np.cos(theta_smooth)
    tip_smooth_y = anchor_smooth_y + x_smooth * np.sin(theta_smooth)

    series["route_c_anchor_x"] = anchor_smooth_x
    series["route_c_anchor_y"] = anchor_smooth_y
    series["route_c_tip_x"] = tip_smooth_x
    series["route_c_tip_y"] = tip_smooth_y
    series["route_c_axis_theta_rad"] = theta_smooth
    series["x_route_c_px"] = x_smooth
    series["kappa_route_c_px_inv"] = temporal_regularize(
        series["kappa_fit_px_inv"].to_numpy(dtype=float),
        increasing=False,
        alpha=alpha_series,
        enforce_monotonic=config.enforce_monotonic_kappa,
    )
    return series
