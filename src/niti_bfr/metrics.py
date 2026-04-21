from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


def recovery_ratio(x: np.ndarray, x_m: float, x_a: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    denom = x_a - x_m
    if abs(denom) < 1e-9:
        raise ValueError("x_a and x_m are too close to define recovery ratio")
    return np.clip((x - x_m) / denom, 0.0, 1.0)


def infer_metric_direction(
    values: np.ndarray,
    *,
    head_fraction: float = 0.1,
    tail_fraction: float = 0.1,
    default_increasing: bool = True,
    min_relative_change: float = 0.02,
) -> bool:
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    if len(valid) < 4:
        return bool(default_increasing)
    head_count = max(2, int(np.ceil(len(valid) * float(head_fraction))))
    tail_count = max(2, int(np.ceil(len(valid) * float(tail_fraction))))
    head = float(np.nanmedian(valid[:head_count]))
    tail = float(np.nanmedian(valid[-tail_count:]))
    scale = max(float(np.nanmax(valid) - np.nanmin(valid)), abs(head), abs(tail), 1e-9)
    if abs(tail - head) < float(min_relative_change) * scale:
        return bool(default_increasing)
    return bool(tail > head)


def _sigmoid_x(temp_c: np.ndarray, x_m: float, x_a: float, t0: float, w: float) -> np.ndarray:
    return x_m + (x_a - x_m) / (1.0 + np.exp(-(temp_c - t0) / w))


@dataclass
class RecoveryFit:
    x_m: float
    x_a: float
    t0: float
    width: float


@dataclass
class MetricEvaluation:
    label: str
    increasing: bool
    fit: RecoveryFit
    af95_c: float
    aftan_c: float
    fit_rmse: float
    monotonic_violation_fraction: float
    dynamic_range: float


def fit_recovery_curve(temp_c: np.ndarray, x: np.ndarray) -> RecoveryFit:
    temp_c = np.asarray(temp_c, dtype=float)
    x = np.asarray(x, dtype=float)
    order = np.argsort(temp_c)
    temp_c = temp_c[order]
    x = x[order]
    p0 = [float(np.min(x)), float(np.max(x)), float(np.median(temp_c)), 2.0]
    bounds = (
        [np.min(x) - 20.0, np.max(x) - 5.0, np.min(temp_c), 0.1],
        [np.min(x) + 20.0, np.max(x) + 80.0, np.max(temp_c), 20.0],
    )
    params, _ = curve_fit(_sigmoid_x, temp_c, x, p0=p0, bounds=bounds, maxfev=20000)
    return RecoveryFit(
        x_m=float(params[0]),
        x_a=float(params[1]),
        t0=float(params[2]),
        width=float(params[3]),
    )


def recovery_ratio_directional(
    values: np.ndarray,
    low_temp_ref: float,
    high_temp_ref: float,
    increasing: bool,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if increasing:
        return recovery_ratio(values, low_temp_ref, high_temp_ref)
    denom = low_temp_ref - high_temp_ref
    if abs(denom) < 1e-9:
        raise ValueError("reference values are too close to define recovery ratio")
    return np.clip((low_temp_ref - values) / denom, 0.0, 1.0)


def evaluate_metric(
    temp_c: np.ndarray,
    values: np.ndarray,
    label: str,
    increasing: bool,
) -> MetricEvaluation:
    temp_c = np.asarray(temp_c, dtype=float)
    values = np.asarray(values, dtype=float)
    order = np.argsort(temp_c)
    temp_c = temp_c[order]
    values = values[order]

    fit_values = values if increasing else -values
    fit = fit_recovery_curve(temp_c, fit_values)
    fitted = fit.x_m + (fit.x_a - fit.x_m) / (1.0 + np.exp(-(temp_c - fit.t0) / fit.width))
    recovery = recovery_ratio(fit_values, fit.x_m, fit.x_a)
    rmse = float(np.sqrt(np.mean((fit_values - fitted) ** 2)))
    diff = np.diff(values)
    if increasing:
        violations = diff < 0.0
    else:
        violations = diff > 0.0
    monotonic_violation_fraction = float(np.mean(violations)) if len(diff) else 0.0
    dynamic_range = float(np.max(values) - np.min(values))
    return MetricEvaluation(
        label=label,
        increasing=increasing,
        fit=fit,
        af95_c=af_95(temp_c, recovery),
        aftan_c=af_tan(temp_c, recovery),
        fit_rmse=rmse,
        monotonic_violation_fraction=monotonic_violation_fraction,
        dynamic_range=dynamic_range,
    )


def af_95(temp_c: np.ndarray, recovery: np.ndarray, threshold: float = 0.95) -> float:
    temp_c = np.asarray(temp_c, dtype=float)
    recovery = np.asarray(recovery, dtype=float)
    order = np.argsort(temp_c)
    temp_c = temp_c[order]
    recovery = recovery[order]
    if np.all(recovery < threshold):
        return float("nan")
    idx = int(np.argmax(recovery >= threshold))
    if idx == 0:
        return float(temp_c[0])
    x0, x1 = recovery[idx - 1], recovery[idx]
    t0, t1 = temp_c[idx - 1], temp_c[idx]
    frac = (threshold - x0) / max(x1 - x0, 1e-9)
    return float(t0 + frac * (t1 - t0))


def af_tan(temp_c: np.ndarray, recovery: np.ndarray) -> float:
    temp_c = np.asarray(temp_c, dtype=float)
    recovery = np.asarray(recovery, dtype=float)
    order = np.argsort(temp_c)
    temp_c = temp_c[order]
    recovery = recovery[order]
    n = len(temp_c)
    if n < 7:
        return float("nan")
    window = min(n if n % 2 == 1 else n - 1, 11)
    window = max(window, 5)
    smooth = savgol_filter(recovery, window_length=window, polyorder=2, mode="interp")
    slope = np.gradient(smooth, temp_c)
    idx = int(np.argmax(slope))
    t0 = temp_c[idx]
    r0 = smooth[idx]
    m = slope[idx]
    if m <= 1e-9:
        return float("nan")
    upper = float(np.median(smooth[int(0.85 * n) :]))
    return float(t0 + (upper - r0) / m)


def summarize_numeric_sweep(
    sweep_results: Mapping[str, Mapping[str, float | int | None]],
    *,
    baseline_key: str | None = None,
) -> dict[str, Any]:
    numeric_fields: set[str] = set()
    for metrics in sweep_results.values():
        for field, value in metrics.items():
            try:
                numeric = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric):
                numeric_fields.add(field)

    spreads: dict[str, float] = {}
    max_abs: dict[str, float] = {}
    baseline_max_delta: dict[str, float] = {}

    for field in sorted(numeric_fields):
        values_by_key: dict[str, float] = {}
        for sweep_key, metrics in sweep_results.items():
            if field not in metrics:
                continue
            try:
                numeric = float(metrics[field])  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric):
                values_by_key[str(sweep_key)] = numeric
        if not values_by_key:
            continue
        values = np.asarray(list(values_by_key.values()), dtype=float)
        spreads[field] = float(np.max(values) - np.min(values))
        max_abs[field] = float(np.max(np.abs(values)))
        if baseline_key is not None and baseline_key in values_by_key:
            baseline_max_delta[field] = float(np.max(np.abs(values - values_by_key[baseline_key])))

    return {
        "variant_count": int(len(sweep_results)),
        "swept_keys": [str(key) for key in sweep_results.keys()],
        "baseline_key": None if baseline_key is None else str(baseline_key),
        "spreads": spreads,
        "max_abs": max_abs,
        "baseline_max_delta": baseline_max_delta,
    }
