from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

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


def _metric_preference_score(report: MetricEvaluation) -> float:
    return float(report.fit_rmse + 200.0 * report.monotonic_violation_fraction)


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
        preferred_items = sorted(metric_reports.items(), key=lambda item: _metric_preference_score(item[1]))
        primary_metric_label, primary_report = preferred_items[0]
        fit = primary_report.fit
        af95_c = primary_report.af95_c
        aftan_c = primary_report.aftan_c
        series = merged

    return AnalysisResult(
        series=series,
        fit=fit,
        af95_c=af95_c,
        aftan_c=aftan_c,
        metric_reports=metric_reports,
        primary_metric_label=primary_metric_label,
    )
