from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .extract_braided import BraidedExtractionConfig, extract_braided_geometry
from .extract import ExtractionConfig, extract_geometry
from .metrics import MetricEvaluation, RecoveryFit, evaluate_metric, recovery_ratio_directional
from .temporal import RouteCConfig, apply_route_c


@dataclass
class AnalysisResult:
    series: pd.DataFrame
    fit: RecoveryFit | None
    af95_c: float | None
    aftan_c: float | None
    metric_reports: dict[str, MetricEvaluation] | None = None
    primary_metric_label: str | None = None
    mode: str = "quicklook"
    formal_metric_label: str | None = None
    formal_gate_reason: str | None = None


def _evaluate_temperature_metrics(series: pd.DataFrame) -> dict[str, MetricEvaluation]:
    valid = series.dropna(
        subset=[
            "temperature_c",
            "x_route_a_px",
            "x_fit_px",
            "x_route_c_px",
            "kappa_fit_px_inv",
            "kappa_route_c_px_inv",
        ]
    )
    reports = {
        "x_route_a": evaluate_metric(
            valid["temperature_c"].to_numpy(),
            valid["x_route_a_px"].to_numpy(),
            label="x_route_a",
            increasing=True,
        ),
        "x_fit": evaluate_metric(
            valid["temperature_c"].to_numpy(),
            valid["x_fit_px"].to_numpy(),
            label="x_fit",
            increasing=True,
        ),
        "x_route_c": evaluate_metric(
            valid["temperature_c"].to_numpy(),
            valid["x_route_c_px"].to_numpy(),
            label="x_route_c",
            increasing=True,
        ),
        "kappa_fit": evaluate_metric(
            valid["temperature_c"].to_numpy(),
            valid["kappa_fit_px_inv"].to_numpy(),
            label="kappa_fit",
            increasing=False,
        ),
        "kappa_route_c": evaluate_metric(
            valid["temperature_c"].to_numpy(),
            valid["kappa_route_c_px_inv"].to_numpy(),
            label="kappa_route_c",
            increasing=False,
        ),
    }
    x_route_a_eval = reports["x_route_a"]
    x_eval = reports["x_fit"]
    x_route_c_eval = reports["x_route_c"]
    k_eval = reports["kappa_fit"]
    k_route_c_eval = reports["kappa_route_c"]
    series["x_route_a_recovery"] = recovery_ratio_directional(
        series["x_route_a_px"].to_numpy(),
        x_route_a_eval.fit.x_m,
        x_route_a_eval.fit.x_a,
        increasing=True,
    )
    series["x_fit_recovery"] = recovery_ratio_directional(
        series["x_fit_px"].to_numpy(),
        x_eval.fit.x_m,
        x_eval.fit.x_a,
        increasing=True,
    )
    series["x_route_c_recovery"] = recovery_ratio_directional(
        series["x_route_c_px"].to_numpy(),
        x_route_c_eval.fit.x_m,
        x_route_c_eval.fit.x_a,
        increasing=True,
    )
    series["kappa_fit_recovery"] = recovery_ratio_directional(
        series["kappa_fit_px_inv"].to_numpy(),
        -k_eval.fit.x_m,
        -k_eval.fit.x_a,
        increasing=False,
    )
    series["kappa_route_c_recovery"] = recovery_ratio_directional(
        series["kappa_route_c_px_inv"].to_numpy(),
        -k_route_c_eval.fit.x_m,
        -k_route_c_eval.fit.x_a,
        increasing=False,
    )
    return reports


def _formal_af_gate(series: pd.DataFrame, reports: dict[str, MetricEvaluation]) -> tuple[bool, str | None]:
    valid = series.dropna(subset=["temperature_c", "kappa_fit_px_inv", "kappa_fit_recovery", "quality"])
    if len(valid) < 15:
        return False, "insufficient_kappa_points"

    quality_median = float(valid["quality"].median())
    if quality_median < 0.10:
        return False, "unstable_kappa_extraction"

    kappa_report = reports["kappa_fit"]
    if kappa_report.monotonic_violation_fraction > 0.25:
        return False, "kappa_fit_not_monotonic_enough"

    tail_count = max(5, int(np.ceil(len(valid) * 0.1)))
    tail_recovery = valid["kappa_fit_recovery"].to_numpy()[-tail_count:]
    if float(np.nanmedian(tail_recovery)) < 0.90:
        return False, "missing_high_temp_plateau"

    return True, None


def analyze_video(
    video_path: str | Path,
    extraction: ExtractionConfig,
    temperature_csv: str | Path | None = None,
    temperature_time_offset_sec: float = 0.0,
    route_c: RouteCConfig | None = None,
) -> AnalysisResult:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    rows: list[dict[str, float]] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        try:
            geom = extract_geometry(frame, extraction)
            route_a_anchor = geom.route_a_anchor_xy
            route_a_tip = geom.route_a_tip_xy
            x_route_a = geom.x_route_a_px
            anchor = geom.anchor_xy
            tip = geom.tip_xy
            x = geom.x_px
            quality = geom.quality
            centerline_points = len(geom.sampled_centerline_xy)
            x_fit = geom.x_fit_px
            kappa_fit = geom.kappa_fit_px_inv
            quadratic_rmse_px = geom.quadratic_rmse_px
            circle_rmse_px = geom.circle_rmse_px
            model_name = geom.model_name
        except RuntimeError:
            route_a_anchor = np.array([np.nan, np.nan])
            route_a_tip = np.array([np.nan, np.nan])
            x_route_a = np.nan
            anchor = np.array([np.nan, np.nan])
            tip = np.array([np.nan, np.nan])
            x = np.nan
            quality = 0.0
            centerline_points = 0
            x_fit = np.nan
            kappa_fit = np.nan
            quadratic_rmse_px = np.nan
            circle_rmse_px = np.nan
            model_name = "failed"
        rows.append(
            {
                "frame": frame_idx,
                "time_sec": frame_idx / fps,
                "route_a_anchor_x": route_a_anchor[0],
                "route_a_anchor_y": route_a_anchor[1],
                "route_a_tip_x": route_a_tip[0],
                "route_a_tip_y": route_a_tip[1],
                "x_route_a_px": x_route_a,
                "anchor_x": anchor[0],
                "anchor_y": anchor[1],
                "tip_x": tip[0],
                "tip_y": tip[1],
                "x_px": x,
                "x_fit_px": x_fit,
                "kappa_fit_px_inv": kappa_fit,
                "quality": quality,
                "centerline_points": centerline_points,
                "quadratic_rmse_px": quadratic_rmse_px,
                "circle_rmse_px": circle_rmse_px,
                "model_name": model_name,
            }
        )
        frame_idx += 1
    cap.release()

    series = pd.DataFrame(rows)
    series = apply_route_c(series, route_c or RouteCConfig())
    fit = None
    af95_c = None
    aftan_c = None
    metric_reports = None
    primary_metric_label = None
    mode = "quicklook"
    formal_metric_label = None
    formal_gate_reason = "temperature_sync_missing"

    if temperature_csv is not None:
        temp = pd.read_csv(temperature_csv).copy()
        if "time_sec" in temp.columns:
            temp["time_sec"] = temp["time_sec"].astype(float) + temperature_time_offset_sec
        if "frame" in temp.columns:
            merged = series.merge(temp, on="frame", how="left")
        elif "time_sec" in temp.columns:
            if "temperature_c" not in temp.columns:
                raise ValueError("temperature file must contain temperature_c")
            temp = temp.sort_values("time_sec")
            merged = series.copy()
            merged["temperature_c"] = np.interp(
                merged["time_sec"].to_numpy(),
                temp["time_sec"].to_numpy(),
                temp["temperature_c"].to_numpy(),
                left=np.nan,
                right=np.nan,
            )
        else:
            raise ValueError("temperature file must contain frame or time_sec")
        if "temperature_c" not in merged.columns:
            raise ValueError("temperature file must contain temperature_c")
        metric_reports = _evaluate_temperature_metrics(merged)
        formal_metric_label = "kappa_fit"
        formal_allowed, formal_gate_reason = _formal_af_gate(merged, metric_reports)
        if formal_allowed:
            formal_report = metric_reports[formal_metric_label]
            fit = formal_report.fit
            af95_c = formal_report.af95_c
            aftan_c = formal_report.aftan_c
            primary_metric_label = formal_metric_label
            mode = "formal_af"
            formal_gate_reason = None
        series = merged

    return AnalysisResult(
        series=series,
        fit=fit,
        af95_c=af95_c,
        aftan_c=aftan_c,
        metric_reports=metric_reports,
        primary_metric_label=primary_metric_label,
        mode=mode,
        formal_metric_label=formal_metric_label,
        formal_gate_reason=formal_gate_reason,
    )


def analyze_braided_video_quicklook(
    video_path: str | Path,
    extraction: BraidedExtractionConfig,
) -> AnalysisResult:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    rows: list[dict[str, float]] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        try:
            geom = extract_braided_geometry(frame, extraction)
            row = {
                "frame": frame_idx,
                "time_sec": frame_idx / fps,
                "anchor_x": geom.anchor_xy[0],
                "anchor_y": geom.anchor_xy[1],
                "tip_x": geom.tip_xy[0],
                "tip_y": geom.tip_xy[1],
                "quality": geom.quality,
                "length_env_px": geom.length_env_px,
                "length_axis_px": geom.length_axis_px,
                "diameter_max_px": geom.diameter_max_px,
                "x_peak_norm": geom.x_peak_norm,
                "taper_left_px": geom.taper_left_px,
                "taper_right_px": geom.taper_right_px,
            }
        except RuntimeError:
            row = {
                "frame": frame_idx,
                "time_sec": frame_idx / fps,
                "anchor_x": np.nan,
                "anchor_y": np.nan,
                "tip_x": np.nan,
                "tip_y": np.nan,
                "quality": 0.0,
                "length_env_px": np.nan,
                "length_axis_px": np.nan,
                "diameter_max_px": np.nan,
                "x_peak_norm": np.nan,
                "taper_left_px": np.nan,
                "taper_right_px": np.nan,
            }
        rows.append(row)
        frame_idx += 1
    cap.release()

    return AnalysisResult(
        series=pd.DataFrame(rows),
        fit=None,
        af95_c=None,
        aftan_c=None,
        metric_reports=None,
        primary_metric_label=None,
        mode="quicklook",
        formal_metric_label=None,
        formal_gate_reason="braided_quicklook_only",
    )
