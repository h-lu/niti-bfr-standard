from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .extract_braided import BraidedExtractionConfig, extract_braided_geometry
from .extract import ExtractionConfig, extract_geometry
from .metrics import (
    MetricEvaluation,
    RecoveryFit,
    evaluate_metric,
    infer_metric_direction,
    recovery_ratio_directional,
)
from .temporal import RouteCConfig, apply_route_c

BRAIDED_METRIC_ALIAS_TO_KEY = {
    "A": "length_axis",
    "B": "diameter_max",
    "C": "area_proj",
}

BRAIDED_METRIC_KEY_TO_ALIAS = {value: key for key, value in BRAIDED_METRIC_ALIAS_TO_KEY.items()}

BRAIDED_METRIC_DISPLAY = {
    "length_axis": "A:length_axis",
    "diameter_max": "B:diameter_max",
    "area_proj": "C:area_proj",
}

BRAIDED_FORMAL_CANDIDATES = ("length_axis", "diameter_max")

WIRE_ROUTE_ALIAS_TO_KEY = {
    "A": "x_route_a",
    "B": "kappa_fit",
    "C": "kappa_route_c",
}

WIRE_ROUTE_DISPLAY = {
    "x_route_a": "A:x_route_a",
    "kappa_fit": "B:kappa_fit",
    "kappa_route_c": "C:kappa_route_c",
}

BRAIDED_ACCEPTANCE_THRESHOLDS: dict[str, dict[str, float]] = {
    "real_video": {
        "min_valid_frames": 10.0,
        "quality_median_min": 0.45,
        "quality_lt_0_5_fraction_max": 0.20,
        "axis_monotonic_violation_fraction_max": 0.20,
        "centerline_disagreement_median_max": 0.08,
        "endpoint_gap_alt_centerline_fraction_p95_max": 0.08,
        "endpoint_gap_alt_centerline_px_floor": 12.0,
        "endpoint_frame_jump_fraction_p95_max": 0.08,
        "endpoint_frame_jump_px_floor": 12.0,
        "axis_peak_position_stability_p95_max": 0.18,
        "body_mask_attachment_leak_fraction_p90_max": 0.03,
        "excluded_attachment_area_fraction_median_max": 0.12,
        "body_mask_area_fraction_median_min": 0.55,
    },
    "synthetic": {
        "min_valid_frames": 10.0,
        "quality_median_min": 0.45,
        "quality_lt_0_5_fraction_max": 0.20,
        "axis_monotonic_violation_fraction_max": 0.20,
        "centerline_disagreement_median_max": 0.08,
        "endpoint_gap_alt_centerline_fraction_p95_max": 0.08,
        "endpoint_gap_alt_centerline_px_floor": 12.0,
        "endpoint_frame_jump_fraction_p95_max": 0.12,
        "endpoint_frame_jump_px_floor": 20.0,
        "axis_peak_position_stability_p95_max": 0.18,
        "body_mask_attachment_leak_fraction_p90_max": 0.03,
        "excluded_attachment_area_fraction_median_max": 0.12,
        "body_mask_area_fraction_median_min": 0.55,
    },
}


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
    provisional_metric_label: str | None = None
    provisional_af95_c: float | None = None
    provisional_aftan_c: float | None = None
    reportability_status: str = "quicklook_only"
    warning_codes: list[str] | None = None
    acceptance_profile: str | None = None
    route_results: list[dict[str, Any]] | None = None


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


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return numeric


def _wire_metric_columns(metric_label: str) -> tuple[str, str]:
    mapping = {
        "x_route_a": ("x_route_a_px", "x_route_a_recovery"),
        "x_fit": ("x_fit_px", "x_fit_recovery"),
        "x_route_c": ("x_route_c_px", "x_route_c_recovery"),
        "kappa_fit": ("kappa_fit_px_inv", "kappa_fit_recovery"),
        "kappa_route_c": ("kappa_route_c_px_inv", "kappa_route_c_recovery"),
    }
    if metric_label not in mapping:
        raise KeyError(f"unsupported wire metric: {metric_label}")
    return mapping[metric_label]


def _formal_wire_metric_gate(
    series: pd.DataFrame,
    reports: dict[str, MetricEvaluation],
    metric_label: str,
) -> tuple[bool, str | None]:
    if metric_label == "kappa_fit":
        return _formal_af_gate(series, reports)

    eval_col, recovery_col = _wire_metric_columns(metric_label)
    metric_report = reports.get(metric_label)
    if metric_report is None:
        return False, f"{metric_label}_insufficient_points"

    valid = series.dropna(subset=["temperature_c", eval_col, recovery_col, "quality"])
    if len(valid) < 15:
        return False, f"{metric_label}_insufficient_points"

    quality_median = float(valid["quality"].median())
    if quality_median < 0.10:
        return False, f"{metric_label}_unstable_extraction"

    dynamic_range_floor = 0.005 if metric_label == "kappa_route_c" else 5.0
    if metric_report.dynamic_range < dynamic_range_floor:
        return False, f"{metric_label}_dynamic_range_too_small"

    if metric_report.monotonic_violation_fraction > 0.25:
        return False, f"{metric_label}_not_monotonic_enough"

    tail_count = max(5, int(np.ceil(len(valid) * 0.1)))
    tail_recovery = valid[recovery_col].to_numpy(dtype=float)[-tail_count:]
    if float(np.nanmedian(tail_recovery)) < 0.90:
        return False, "missing_high_temp_plateau"

    return True, None


def _build_route_result(
    *,
    alias: str,
    metric_key: str,
    display_label: str,
    value_series_col: str,
    recovery_series_col: str | None,
    metric_report: MetricEvaluation | None,
    reportability_status: str,
    gate_reason: str | None,
    warning_codes: list[str] | None,
    formal_candidate: bool,
    accepted_as_formal_candidate: bool,
    selected_as_primary: bool,
    selected_as_formal: bool,
    auxiliary_metric_keys: list[str] | None = None,
) -> dict[str, Any]:
    if selected_as_formal:
        formal_role = "object_formal"
    elif selected_as_primary:
        formal_role = "object_primary"
    elif formal_candidate:
        formal_role = "formal_candidate"
    else:
        formal_role = "route_result"
    return {
        "alias": alias,
        "metric_key": metric_key,
        "display_label": display_label,
        "value_series_col": value_series_col,
        "recovery_series_col": recovery_series_col,
        "af95_c": _finite_or_none(metric_report.af95_c) if metric_report is not None else None,
        "aftan_c": _finite_or_none(metric_report.aftan_c) if metric_report is not None else None,
        "fit_rmse": _finite_or_none(metric_report.fit_rmse) if metric_report is not None else None,
        "monotonic_violation_fraction": (
            _finite_or_none(metric_report.monotonic_violation_fraction) if metric_report is not None else None
        ),
        "dynamic_range": _finite_or_none(metric_report.dynamic_range) if metric_report is not None else None,
        "reportability_status": reportability_status,
        "gate_reason": gate_reason,
        "warning_codes": list(warning_codes or []),
        "formal_candidate": formal_candidate,
        "accepted_as_formal_candidate": accepted_as_formal_candidate,
        "selected_as_primary": selected_as_primary,
        "selected_as_formal": selected_as_formal,
        "formal_role": formal_role,
        "auxiliary_metric_keys": list(auxiliary_metric_keys or []),
    }


def _build_wire_route_results(
    series: pd.DataFrame,
    reports: dict[str, MetricEvaluation] | None,
    *,
    primary_metric_label: str | None,
    formal_metric_label: str | None,
    temperature_available: bool,
) -> list[dict[str, Any]]:
    route_specs = [
        ("A", "x_route_a", "A:x_route_a", "x_route_a_px", "x_route_a_recovery", []),
        ("B", "kappa_fit", "B:kappa_fit", "kappa_fit_px_inv", "kappa_fit_recovery", ["x_fit"]),
        ("C", "kappa_route_c", "C:kappa_route_c", "kappa_route_c_px_inv", "kappa_route_c_recovery", ["x_route_c"]),
    ]
    route_results: list[dict[str, Any]] = []
    reports = reports or {}
    for alias, metric_key, display_label, value_col, recovery_col, aux_keys in route_specs:
        metric_report = reports.get(metric_key)
        formal_candidate = metric_key == "kappa_fit"
        if not temperature_available:
            status = "quicklook_only"
            gate_reason = "temperature_sync_missing"
            accepted = False
        elif metric_report is None:
            status = "formal_blocked"
            gate_reason = f"{metric_key}_insufficient_points"
            accepted = False
        else:
            allowed, gate_reason = _formal_wire_metric_gate(series, reports, metric_key)
            if allowed:
                status = "formal_passed" if formal_candidate else "provisional"
                gate_reason = None
                accepted = formal_candidate
            else:
                status = "formal_blocked"
                accepted = False
        route_results.append(
            _build_route_result(
                alias=alias,
                metric_key=metric_key,
                display_label=display_label,
                value_series_col=value_col,
                recovery_series_col=recovery_col,
                metric_report=metric_report,
                reportability_status=status,
                gate_reason=gate_reason,
                warning_codes=[] if gate_reason is None else [gate_reason],
                formal_candidate=formal_candidate,
                accepted_as_formal_candidate=accepted,
                selected_as_primary=primary_metric_label == metric_key,
                selected_as_formal=formal_metric_label == metric_key,
                auxiliary_metric_keys=aux_keys,
            )
        )
    return route_results


def _evaluate_braided_temperature_metrics(series: pd.DataFrame) -> dict[str, MetricEvaluation]:
    axis_eval_col = "length_axis_formal_px" if "length_axis_formal_px" in series.columns else "length_axis_px"
    area_eval_col = "area_proj_formal_px2" if "area_proj_formal_px2" in series.columns else "area_proj_px2"
    length_env_valid = series.dropna(subset=["temperature_c", "length_env_px"])
    length_axis_valid = series.dropna(subset=["temperature_c", axis_eval_col])
    diameter_valid = series.dropna(subset=["temperature_c", "diameter_max_px"])
    area_valid = series.dropna(subset=["temperature_c", area_eval_col])

    reports: dict[str, MetricEvaluation] = {}
    if len(length_env_valid) >= 4:
        reports["length_env"] = evaluate_metric(
            length_env_valid["temperature_c"].to_numpy(),
            length_env_valid["length_env_px"].to_numpy(),
            label="length_env",
            increasing=False,
        )
    if len(length_axis_valid) >= 4:
        reports["length_axis"] = evaluate_metric(
            length_axis_valid["temperature_c"].to_numpy(),
            length_axis_valid[axis_eval_col].to_numpy(),
            label="length_axis",
            increasing=False,
        )
    if len(diameter_valid) >= 4:
        reports["diameter_max"] = evaluate_metric(
            diameter_valid["temperature_c"].to_numpy(),
            diameter_valid["diameter_max_px"].to_numpy(),
            label="diameter_max",
            increasing=True,
        )
    if len(area_valid) >= 4:
        area_increasing = infer_metric_direction(area_valid[area_eval_col].to_numpy(), default_increasing=True)
        reports["area_proj"] = evaluate_metric(
            area_valid["temperature_c"].to_numpy(),
            area_valid[area_eval_col].to_numpy(),
            label="area_proj",
            increasing=area_increasing,
        )

    series["length_env_recovery"] = np.nan
    series["length_axis_recovery"] = np.nan
    series["diameter_max_recovery"] = np.nan
    series["area_proj_recovery"] = np.nan

    length_env_eval = reports.get("length_env")
    if length_env_eval is not None:
        series["length_env_recovery"] = recovery_ratio_directional(
            series["length_env_px"].to_numpy(),
            -length_env_eval.fit.x_m,
            -length_env_eval.fit.x_a,
            increasing=False,
        )
    length_axis_eval = reports.get("length_axis")
    if length_axis_eval is not None:
        series["length_axis_recovery"] = recovery_ratio_directional(
            series[axis_eval_col].to_numpy(),
            -length_axis_eval.fit.x_m,
            -length_axis_eval.fit.x_a,
            increasing=False,
        )
    diameter_eval = reports.get("diameter_max")
    if diameter_eval is not None:
        series["diameter_max_recovery"] = recovery_ratio_directional(
            series["diameter_max_px"].to_numpy(),
            diameter_eval.fit.x_m,
            diameter_eval.fit.x_a,
            increasing=True,
        )
    area_eval = reports.get("area_proj")
    if area_eval is not None:
        series["area_proj_recovery"] = recovery_ratio_directional(
            series[area_eval_col].to_numpy(),
            area_eval.fit.x_m,
            area_eval.fit.x_a,
            increasing=area_eval.increasing,
        )
    return reports


def _build_braided_route_results(
    series: pd.DataFrame,
    reports: dict[str, MetricEvaluation] | None,
    *,
    primary_metric_label: str | None,
    formal_metric_label: str | None,
    acceptance_profile: str,
    temperature_available: bool,
) -> list[dict[str, Any]]:
    route_specs = [
        ("A", "length_axis", "A:length_axis", "length_axis_formal_px", "length_axis_recovery"),
        ("B", "diameter_max", "B:diameter_max", "diameter_max_px", "diameter_max_recovery"),
        ("C", "area_proj", "C:area_proj", "area_proj_formal_px2", "area_proj_recovery"),
    ]
    route_results: list[dict[str, Any]] = []
    reports = reports or {}
    for alias, metric_key, display_label, value_col, recovery_col in route_specs:
        metric_report = reports.get(metric_key)
        formal_candidate = metric_key in BRAIDED_FORMAL_CANDIDATES
        if not temperature_available:
            status = "quicklook_only"
            gate_reason = "temperature_sync_missing"
            accepted = False
        elif metric_report is None:
            status = "formal_blocked"
            gate_reason = _braided_formal_reason(metric_key, "insufficient_points")
            accepted = False
        else:
            allowed, gate_reason = _formal_braided_af_gate(
                series,
                reports,
                metric_label=metric_key,
                acceptance_profile=acceptance_profile,
            )
            if allowed:
                status = "formal_passed" if formal_candidate else "provisional"
                gate_reason = None
                accepted = formal_candidate
            else:
                status = "formal_blocked"
                accepted = False
        route_results.append(
            _build_route_result(
                alias=alias,
                metric_key=metric_key,
                display_label=display_label,
                value_series_col=value_col,
                recovery_series_col=recovery_col,
                metric_report=metric_report,
                reportability_status=status,
                gate_reason=gate_reason,
                warning_codes=[] if gate_reason is None else [gate_reason],
                formal_candidate=formal_candidate,
                accepted_as_formal_candidate=accepted,
                selected_as_primary=primary_metric_label == metric_key,
                selected_as_formal=formal_metric_label == metric_key,
            )
        )
    return route_results


def _augment_braided_qc_series(series: pd.DataFrame) -> pd.DataFrame:
    if series.empty:
        return series
    augmented = series.copy()

    if "branch_component_count_after_pruning" in augmented.columns and "branch_count_after_pruning" not in augmented.columns:
        augmented["branch_count_after_pruning"] = augmented["branch_component_count_after_pruning"]
    if "branch_count_after_pruning" in augmented.columns and "branch_component_count_after_pruning" not in augmented.columns:
        augmented["branch_component_count_after_pruning"] = augmented["branch_count_after_pruning"]

    def _rowwise_nanmax(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        finite = np.isfinite(values)
        safe = np.where(finite, values, -np.inf)
        out = np.max(safe, axis=1)
        out[~np.any(finite, axis=1)] = np.nan
        return out

    if "length_axis_px" in augmented.columns:
        augmented["length_axis_formal_px"] = augmented["length_axis_px"]
    if "area_proj_px2" in augmented.columns:
        augmented["area_proj_formal_px2"] = augmented["area_proj_px2"]
    if "endpoint_gap_alt_centerline_px" not in augmented.columns and "endpoint_jump_px" in augmented.columns:
        augmented["endpoint_gap_alt_centerline_px"] = augmented["endpoint_jump_px"].to_numpy(dtype=float)
    if "length_axis_px" in augmented.columns and "length_axis_alt_px" in augmented.columns:
        augmented["centerline_disagreement"] = augmented["length_axis_disagreement_px"] / augmented["length_axis_px"].clip(lower=1e-9)
    elif "length_axis_disagreement_px" not in augmented.columns:
        augmented["length_axis_disagreement_px"] = np.nan
        augmented["centerline_disagreement"] = np.nan
    if {"anchor_x", "anchor_y", "tip_x", "tip_y"}.issubset(augmented.columns):
        anchor = augmented[["anchor_x", "anchor_y"]].to_numpy(dtype=float)
        tip = augmented[["tip_x", "tip_y"]].to_numpy(dtype=float)
        anchor_jump = np.full(len(augmented), np.nan, dtype=float)
        tip_jump = np.full(len(augmented), np.nan, dtype=float)
        if len(augmented) >= 2:
            anchor_jump[1:] = np.linalg.norm(np.diff(anchor, axis=0), axis=1)
            tip_jump[1:] = np.linalg.norm(np.diff(tip, axis=0), axis=1)
        augmented["anchor_jump_px"] = anchor_jump
        augmented["tip_jump_px"] = tip_jump
        augmented["endpoint_frame_jump_px"] = _rowwise_nanmax(np.column_stack([anchor_jump, tip_jump]))
    axis_eval_col = "length_axis_formal_px" if "length_axis_formal_px" in augmented.columns else "length_axis_px"
    if axis_eval_col in augmented.columns:
        if "endpoint_gap_alt_centerline_px" in augmented.columns:
            augmented["endpoint_gap_alt_centerline_fraction"] = _safe_fraction(
                augmented["endpoint_gap_alt_centerline_px"],
                augmented[axis_eval_col],
            )
            # Keep the legacy endpoint_jump_* columns as aliases for same-frame centerline disagreement.
            augmented["endpoint_jump_px"] = augmented["endpoint_gap_alt_centerline_px"]
            augmented["endpoint_jump_fraction"] = augmented["endpoint_gap_alt_centerline_fraction"]
        if "endpoint_frame_jump_px" in augmented.columns:
            augmented["endpoint_frame_jump_fraction"] = _safe_fraction(
                augmented["endpoint_frame_jump_px"],
                augmented[axis_eval_col],
            )
    if "diameter_peak_pos_norm" in augmented.columns:
        stability = np.full(len(augmented), np.nan, dtype=float)
        if len(augmented) >= 2:
            stability[1:] = np.abs(np.diff(augmented["diameter_peak_pos_norm"].to_numpy(dtype=float)))
        if "axis_peak_position_stability" in augmented.columns:
            augmented["axis_peak_position_stability"] = _rowwise_nanmax(
                np.column_stack([augmented["axis_peak_position_stability"].to_numpy(dtype=float), stability])
            )
        else:
            augmented["axis_peak_position_stability"] = stability
    return augmented


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


def _finite_percentile(values: np.ndarray | pd.Series, q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def _finite_median(values: np.ndarray | pd.Series) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(np.median(arr))


def _safe_fraction(numerator: np.ndarray | pd.Series, denominator: np.ndarray | pd.Series) -> np.ndarray:
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    out = np.full(num.shape, np.nan, dtype=float)
    valid = np.isfinite(num) & np.isfinite(den)
    out[valid] = num[valid] / np.maximum(den[valid], 1e-9)
    return out


def _finite_value(value: float) -> float | None:
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _braided_monotonic_violation_fraction(
    temp_c: np.ndarray | pd.Series,
    values: np.ndarray | pd.Series,
    *,
    increasing: bool,
) -> float | None:
    temp_arr = np.asarray(temp_c, dtype=float)
    value_arr = np.asarray(values, dtype=float)
    valid = np.isfinite(temp_arr) & np.isfinite(value_arr)
    if np.count_nonzero(valid) < 4:
        return None

    temp_valid = temp_arr[valid]
    value_valid = value_arr[valid]
    order = np.argsort(temp_valid)
    value_valid = value_valid[order]

    window = min(9, len(value_valid))
    if window % 2 == 0:
        window -= 1
    if window >= 3:
        smooth = pd.Series(value_valid).rolling(window=window, center=True, min_periods=1).median().to_numpy(dtype=float)
    else:
        smooth = value_valid

    dynamic_range = float(np.nanmax(smooth) - np.nanmin(smooth)) if len(smooth) else 0.0
    tolerance = max(0.5, 0.005 * dynamic_range)
    diff = np.diff(smooth)
    if increasing:
        violations = diff < -tolerance
    else:
        violations = diff > tolerance
    return float(np.mean(violations)) if len(diff) else 0.0


def _braided_acceptance_thresholds(profile: str) -> dict[str, float]:
    if profile not in BRAIDED_ACCEPTANCE_THRESHOLDS:
        available = ", ".join(sorted(BRAIDED_ACCEPTANCE_THRESHOLDS))
        raise KeyError(f"unsupported braided acceptance profile: {profile} (available: {available})")
    return {key: float(value) for key, value in BRAIDED_ACCEPTANCE_THRESHOLDS[profile].items()}


def _braided_endpoint_limit_px(
    series: pd.DataFrame,
    axis_eval_col: str,
    *,
    thresholds: dict[str, float],
    fraction_key: str,
    floor_key: str,
) -> float | None:
    if axis_eval_col not in series.columns:
        return None
    axis_median = _finite_median(series[axis_eval_col])
    if not np.isfinite(axis_median):
        return None
    return float(max(thresholds[floor_key], thresholds[fraction_key] * axis_median))


def compute_braided_acceptance(series: pd.DataFrame, *, acceptance_profile: str = "real_video") -> dict[str, Any]:
    axis_eval_col = "length_axis_formal_px" if "length_axis_formal_px" in series.columns else "length_axis_px"
    if axis_eval_col in series.columns and "quality" in series.columns:
        valid = series.dropna(subset=[axis_eval_col, "quality"])
    elif axis_eval_col in series.columns:
        valid = series.dropna(subset=[axis_eval_col])
    else:
        valid = series.iloc[0:0].copy()

    metrics: dict[str, Any] = {
        "axis_eval_col": axis_eval_col,
        "valid_frames": int(len(valid)),
        "quality_median": None,
        "quality_lt_0_5_fraction": None,
        "axis_monotonic_violation_fraction": None,
        "centerline_disagreement_median": None,
        "endpoint_gap_alt_centerline_p95_px": None,
        "endpoint_gap_alt_centerline_fraction_p95": None,
        "endpoint_gap_alt_centerline_limit_px": None,
        "endpoint_frame_jump_p95_px": None,
        "endpoint_frame_jump_fraction_p95": None,
        "endpoint_frame_jump_limit_px": None,
        "endpoint_jump_p95_px": None,
        "endpoint_jump_fraction_p95": None,
        "endpoint_jump_limit_px": None,
        "branch_component_count_after_pruning_p95": None,
        "axis_peak_position_stability_p95": None,
        "body_mask_attachment_leak_fraction_p90": None,
        "excluded_attachment_area_fraction_median": None,
        "body_mask_area_fraction_median": None,
        "attachment_border_touch_count_p90": None,
    }
    thresholds = _braided_acceptance_thresholds(acceptance_profile)

    if len(valid) > 0:
        if "quality" in valid.columns:
            metrics["quality_median"] = _finite_value(_finite_median(valid["quality"]))
            metrics["quality_lt_0_5_fraction"] = float(np.mean(valid["quality"].to_numpy(dtype=float) < 0.5))
        if {"temperature_c", axis_eval_col}.issubset(valid.columns):
            axis_monotonic_violation = _braided_monotonic_violation_fraction(
                valid["temperature_c"],
                valid[axis_eval_col],
                increasing=False,
            )
            metrics["axis_monotonic_violation_fraction"] = _finite_value(
                axis_monotonic_violation if axis_monotonic_violation is not None else float("nan")
            )
        if "centerline_disagreement" in valid.columns:
            metrics["centerline_disagreement_median"] = _finite_value(_finite_median(valid["centerline_disagreement"]))
        endpoint_gap_col = "endpoint_gap_alt_centerline_px" if "endpoint_gap_alt_centerline_px" in valid.columns else "endpoint_jump_px"
        endpoint_gap_fraction_col = (
            "endpoint_gap_alt_centerline_fraction"
            if "endpoint_gap_alt_centerline_fraction" in valid.columns
            else "endpoint_jump_fraction"
        )
        if endpoint_gap_col in valid.columns:
            metrics["endpoint_gap_alt_centerline_p95_px"] = _finite_value(_finite_percentile(valid[endpoint_gap_col], 95))
            endpoint_gap_fraction = (
                valid[endpoint_gap_fraction_col]
                if endpoint_gap_fraction_col in valid.columns
                else _safe_fraction(valid[endpoint_gap_col], valid[axis_eval_col])
            )
            metrics["endpoint_gap_alt_centerline_fraction_p95"] = _finite_value(
                _finite_percentile(endpoint_gap_fraction, 95)
            )
            metrics["endpoint_gap_alt_centerline_limit_px"] = _finite_value(
                _braided_endpoint_limit_px(
                    valid,
                    axis_eval_col,
                    thresholds=thresholds,
                    fraction_key="endpoint_gap_alt_centerline_fraction_p95_max",
                    floor_key="endpoint_gap_alt_centerline_px_floor",
                )
                or float("nan")
            )
            metrics["endpoint_jump_p95_px"] = metrics["endpoint_gap_alt_centerline_p95_px"]
            metrics["endpoint_jump_fraction_p95"] = metrics["endpoint_gap_alt_centerline_fraction_p95"]
            metrics["endpoint_jump_limit_px"] = metrics["endpoint_gap_alt_centerline_limit_px"]
        if "endpoint_frame_jump_px" in valid.columns:
            metrics["endpoint_frame_jump_p95_px"] = _finite_value(_finite_percentile(valid["endpoint_frame_jump_px"], 95))
            endpoint_frame_fraction = (
                valid["endpoint_frame_jump_fraction"]
                if "endpoint_frame_jump_fraction" in valid.columns
                else _safe_fraction(valid["endpoint_frame_jump_px"], valid[axis_eval_col])
            )
            metrics["endpoint_frame_jump_fraction_p95"] = _finite_value(
                _finite_percentile(endpoint_frame_fraction, 95)
            )
            metrics["endpoint_frame_jump_limit_px"] = _finite_value(
                _braided_endpoint_limit_px(
                    valid,
                    axis_eval_col,
                    thresholds=thresholds,
                    fraction_key="endpoint_frame_jump_fraction_p95_max",
                    floor_key="endpoint_frame_jump_px_floor",
                )
                or float("nan")
            )
        branch_qc_col = None
        if "branch_component_count_after_pruning" in valid.columns:
            branch_qc_col = "branch_component_count_after_pruning"
        elif "branch_count_after_pruning" in valid.columns:
            branch_qc_col = "branch_count_after_pruning"
        if branch_qc_col is not None:
            metrics["branch_component_count_after_pruning_p95"] = _finite_value(_finite_percentile(valid[branch_qc_col], 95))
        if "axis_peak_position_stability" in valid.columns:
            metrics["axis_peak_position_stability_p95"] = _finite_value(_finite_percentile(valid["axis_peak_position_stability"], 95))
        if "body_mask_attachment_leak_fraction" in valid.columns:
            metrics["body_mask_attachment_leak_fraction_p90"] = _finite_value(
                _finite_percentile(valid["body_mask_attachment_leak_fraction"], 90)
            )
        if {"excluded_attachment_area_px2", "component_area_px2"}.issubset(valid.columns):
            excluded_attachment_fraction = _safe_fraction(
                valid["excluded_attachment_area_px2"],
                valid["component_area_px2"],
            )
            metrics["excluded_attachment_area_fraction_median"] = _finite_value(_finite_median(excluded_attachment_fraction))
        if {"body_mask_area_px2", "component_area_px2"}.issubset(valid.columns):
            body_component_fraction = _safe_fraction(
                valid["body_mask_area_px2"],
                valid["component_area_px2"],
            )
            metrics["body_mask_area_fraction_median"] = _finite_value(_finite_median(body_component_fraction))
        if "attachment_border_touch_count" in valid.columns:
            metrics["attachment_border_touch_count_p90"] = _finite_value(
                _finite_percentile(valid["attachment_border_touch_count"], 90)
            )

    reasons: list[str] = []
    if metrics["valid_frames"] < int(thresholds["min_valid_frames"]):
        reasons.append("insufficient_axis_points")
    if metrics["quality_median"] is not None and metrics["quality_median"] < thresholds["quality_median_min"]:
        reasons.append("unstable_axis_extraction")
    if (
        metrics["quality_lt_0_5_fraction"] is not None
        and metrics["quality_lt_0_5_fraction"] > thresholds["quality_lt_0_5_fraction_max"]
    ):
        reasons.append("quality_low_fraction")
    if (
        metrics["axis_monotonic_violation_fraction"] is not None
        and metrics["axis_monotonic_violation_fraction"] > thresholds["axis_monotonic_violation_fraction_max"]
    ):
        reasons.append("length_axis_not_monotonic_enough")
    if (
        metrics["centerline_disagreement_median"] is not None
        and metrics["centerline_disagreement_median"] > thresholds["centerline_disagreement_median_max"]
    ):
        reasons.append("centerline_disagreement")
    if (
        metrics["endpoint_gap_alt_centerline_fraction_p95"] is not None
        and metrics["endpoint_gap_alt_centerline_fraction_p95"] > thresholds["endpoint_gap_alt_centerline_fraction_p95_max"]
    ):
        reasons.append("endpoint_jump")
    elif (
        metrics["endpoint_gap_alt_centerline_p95_px"] is not None
        and metrics["endpoint_gap_alt_centerline_limit_px"] is not None
        and metrics["endpoint_gap_alt_centerline_p95_px"] > metrics["endpoint_gap_alt_centerline_limit_px"]
    ):
        reasons.append("endpoint_jump")
    if (
        metrics["endpoint_frame_jump_fraction_p95"] is not None
        and metrics["endpoint_frame_jump_fraction_p95"] > thresholds["endpoint_frame_jump_fraction_p95_max"]
    ):
        reasons.append("endpoint_frame_jump")
    elif (
        metrics["endpoint_frame_jump_p95_px"] is not None
        and metrics["endpoint_frame_jump_limit_px"] is not None
        and metrics["endpoint_frame_jump_p95_px"] > metrics["endpoint_frame_jump_limit_px"]
    ):
        reasons.append("endpoint_frame_jump")
    if (
        metrics["axis_peak_position_stability_p95"] is not None
        and metrics["axis_peak_position_stability_p95"] > thresholds["axis_peak_position_stability_p95_max"]
    ):
        reasons.append("axis_peak_position_stability")
    if (
        metrics["body_mask_attachment_leak_fraction_p90"] is not None
        and metrics["body_mask_attachment_leak_fraction_p90"] > thresholds["body_mask_attachment_leak_fraction_p90_max"]
    ):
        reasons.append("body_mask_attachment_leak_fraction")
    if (
        metrics["excluded_attachment_area_fraction_median"] is not None
        and metrics["excluded_attachment_area_fraction_median"] > thresholds["excluded_attachment_area_fraction_median_max"]
    ):
        reasons.append("excluded_attachment_area_fraction")
    if (
        metrics["body_mask_area_fraction_median"] is not None
        and metrics["body_mask_area_fraction_median"] < thresholds["body_mask_area_fraction_median_min"]
    ):
        reasons.append("body_mask_area_fraction")
    deduped_reasons = list(dict.fromkeys(reasons))
    return {
        "accepted": len(deduped_reasons) == 0,
        "reasons": deduped_reasons,
        "metrics": metrics,
        "thresholds": thresholds,
        "acceptance_profile": acceptance_profile,
    }


def _braided_metric_columns(metric_label: str) -> tuple[str, str]:
    mapping = {
        "length_axis": ("length_axis_formal_px", "length_axis_recovery"),
        "diameter_max": ("diameter_max_px", "diameter_max_recovery"),
        "area_proj": ("area_proj_formal_px2", "area_proj_recovery"),
    }
    if metric_label not in mapping:
        raise KeyError(f"unsupported braided metric: {metric_label}")
    return mapping[metric_label]


def _braided_formal_reason(metric_label: str, suffix: str) -> str:
    if metric_label == "length_axis":
        mapping = {
            "insufficient_points": "insufficient_axis_points",
            "unstable_extraction": "unstable_axis_extraction",
            "dynamic_range_too_small": "axis_dynamic_range_too_small",
            "not_monotonic_enough": "length_axis_not_monotonic_enough",
        }
        return mapping.get(suffix, suffix)
    return f"{metric_label}_{suffix}"


def _braided_formal_candidate_score(
    series: pd.DataFrame,
    reports: dict[str, MetricEvaluation],
    metric_label: str,
) -> tuple[float, float, float]:
    eval_col, _ = _braided_metric_columns(metric_label)
    report = reports[metric_label]
    valid = series.dropna(subset=["temperature_c", eval_col])
    smoothed_monotonic = _braided_monotonic_violation_fraction(
        valid["temperature_c"],
        valid[eval_col],
        increasing=report.increasing,
    )
    monotonic_score = float(smoothed_monotonic if smoothed_monotonic is not None else report.monotonic_violation_fraction)
    fit_score = float(report.fit_rmse / max(report.dynamic_range, 1e-9))
    preference = 0.0 if metric_label == "diameter_max" else 1.0
    return monotonic_score, fit_score, preference


def _select_braided_formal_metric(
    series: pd.DataFrame,
    reports: dict[str, MetricEvaluation],
    *,
    acceptance_profile: str = "real_video",
) -> tuple[str | None, str | None]:
    winners: list[tuple[tuple[float, float, float], str]] = []
    fallback_reason: str | None = None
    for metric_label in BRAIDED_FORMAL_CANDIDATES:
        allowed, reason = _formal_braided_af_gate(
            series,
            reports,
            metric_label=metric_label,
            acceptance_profile=acceptance_profile,
        )
        if allowed:
            winners.append((_braided_formal_candidate_score(series, reports, metric_label), metric_label))
        elif metric_label == "length_axis":
            fallback_reason = reason
    if winners:
        winners.sort(key=lambda item: item[0])
        return winners[0][1], None
    return None, fallback_reason


def _formal_braided_af_gate(
    series: pd.DataFrame,
    reports: dict[str, MetricEvaluation],
    metric_label: str = "length_axis",
    *,
    acceptance_profile: str = "real_video",
) -> tuple[bool, str | None]:
    eval_col, recovery_col = _braided_metric_columns(metric_label)
    metric_report = reports.get(metric_label)
    thresholds = _braided_acceptance_thresholds(acceptance_profile)
    if metric_report is None:
        return False, _braided_formal_reason(metric_label, "insufficient_points")
    valid = series.dropna(subset=["temperature_c", eval_col, recovery_col, "quality"])
    if len(valid) < 15:
        return False, _braided_formal_reason(metric_label, "insufficient_points")

    quality_median = float(valid["quality"].median())
    if quality_median < 0.10:
        return False, _braided_formal_reason(metric_label, "unstable_extraction")

    if metric_report.dynamic_range < 5.0:
        return False, _braided_formal_reason(metric_label, "dynamic_range_too_small")

    metric_monotonic_violation = _braided_monotonic_violation_fraction(
        valid["temperature_c"],
        valid[eval_col],
        increasing=metric_report.increasing,
    )
    if (
        metric_monotonic_violation is not None
        and metric_monotonic_violation > thresholds["axis_monotonic_violation_fraction_max"]
    ):
        return False, _braided_formal_reason(metric_label, "not_monotonic_enough")

    acceptance = compute_braided_acceptance(valid, acceptance_profile=acceptance_profile)
    acceptance_metrics = acceptance["metrics"]
    acceptance_thresholds = acceptance["thresholds"]

    if acceptance_metrics["centerline_disagreement_median"] is not None:
        if acceptance_metrics["centerline_disagreement_median"] > acceptance_thresholds["centerline_disagreement_median_max"]:
            return False, "centerline_disagreement"

    if acceptance_metrics["endpoint_gap_alt_centerline_fraction_p95"] is not None:
        if (
            acceptance_metrics["endpoint_gap_alt_centerline_fraction_p95"]
            > acceptance_thresholds["endpoint_gap_alt_centerline_fraction_p95_max"]
        ):
            return False, "endpoint_jump"
    if (
        acceptance_metrics["endpoint_gap_alt_centerline_p95_px"] is not None
        and acceptance_metrics["endpoint_gap_alt_centerline_limit_px"] is not None
    ):
        if (
            acceptance_metrics["endpoint_gap_alt_centerline_p95_px"]
            > acceptance_metrics["endpoint_gap_alt_centerline_limit_px"]
        ):
            return False, "endpoint_jump"
    if acceptance_profile == "real_video":
        if acceptance_metrics["endpoint_frame_jump_fraction_p95"] is not None:
            if (
                acceptance_metrics["endpoint_frame_jump_fraction_p95"]
                > acceptance_thresholds["endpoint_frame_jump_fraction_p95_max"]
            ):
                return False, "endpoint_frame_jump"
        if (
            acceptance_metrics["endpoint_frame_jump_p95_px"] is not None
            and acceptance_metrics["endpoint_frame_jump_limit_px"] is not None
            and acceptance_metrics["endpoint_frame_jump_p95_px"] > acceptance_metrics["endpoint_frame_jump_limit_px"]
        ):
            return False, "endpoint_frame_jump"

    if acceptance_metrics["axis_peak_position_stability_p95"] is not None:
        if (
            acceptance_metrics["axis_peak_position_stability_p95"]
            > acceptance_thresholds["axis_peak_position_stability_p95_max"]
        ):
            return False, "axis_peak_position_stability"

    if acceptance_metrics["body_mask_attachment_leak_fraction_p90"] is not None:
        if (
            acceptance_metrics["body_mask_attachment_leak_fraction_p90"]
            > acceptance_thresholds["body_mask_attachment_leak_fraction_p90_max"]
        ):
            return False, "body_mask_attachment_leak_fraction"

    if acceptance_metrics["excluded_attachment_area_fraction_median"] is not None:
        if (
            acceptance_metrics["excluded_attachment_area_fraction_median"]
            > acceptance_thresholds["excluded_attachment_area_fraction_median_max"]
        ):
            return False, "excluded_attachment_area_fraction"

    if acceptance_metrics["body_mask_area_fraction_median"] is not None:
        if acceptance_metrics["body_mask_area_fraction_median"] < acceptance_thresholds["body_mask_area_fraction_median_min"]:
            return False, "body_mask_area_fraction"

    tail_count = max(5, int(np.ceil(len(valid) * 0.1)))
    tail_recovery = valid[recovery_col].to_numpy()[-tail_count:]
    if float(np.nanmedian(tail_recovery)) < 0.90:
        return False, "missing_high_temp_plateau"

    return True, None


def _select_braided_primary_metric(series: pd.DataFrame, reports: dict[str, MetricEvaluation]) -> str | None:
    candidates: list[tuple[tuple[float, float, float], str]] = []
    for metric_label in BRAIDED_FORMAL_CANDIDATES:
        metric_report = reports.get(metric_label)
        if metric_report is None:
            continue
        eval_col, _ = _braided_metric_columns(metric_label)
        valid = series.dropna(subset=["temperature_c", eval_col])
        if len(valid) < 4:
            continue
        candidates.append((_braided_formal_candidate_score(series, reports, metric_label), metric_label))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


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
    reportability_status = "quicklook_only"
    warning_codes: list[str] = []
    route_results = _build_wire_route_results(
        series,
        metric_reports,
        primary_metric_label=primary_metric_label,
        formal_metric_label=formal_metric_label,
        temperature_available=False,
    )

    if temperature_csv is not None:
        temp = pd.read_csv(temperature_csv).copy()
        if "time_sec" in temp.columns:
            temp["time_sec"] = temp["time_sec"].astype(float) + temperature_time_offset_sec
        if "frame" in temp.columns:
            temp = temp.drop(columns=["time_sec"], errors="ignore")
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
        candidate_formal_metric_label = "kappa_fit"
        formal_allowed, formal_gate_reason = _formal_af_gate(merged, metric_reports)
        if formal_allowed:
            formal_metric_label = candidate_formal_metric_label
            formal_report = metric_reports[formal_metric_label]
            fit = formal_report.fit
            af95_c = formal_report.af95_c
            aftan_c = formal_report.aftan_c
            primary_metric_label = formal_metric_label
            mode = "formal_af"
            formal_gate_reason = None
            reportability_status = "formal"
        else:
            reportability_status = "reportable_with_warning"
            warning_codes = [formal_gate_reason] if formal_gate_reason is not None else []
        series = merged
        route_results = _build_wire_route_results(
            series,
            metric_reports,
            primary_metric_label=primary_metric_label,
            formal_metric_label=formal_metric_label,
            temperature_available=True,
        )

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
        reportability_status=reportability_status,
        warning_codes=warning_codes,
        route_results=route_results,
    )


def analyze_braided_video_quicklook(
    video_path: str | Path,
    extraction: BraidedExtractionConfig,
    temperature_csv: str | Path | None = None,
    temperature_time_offset_sec: float = 0.0,
    acceptance_profile: str = "real_video",
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
                "length_axis_skeleton_px": geom.length_axis_skeleton_px,
                "length_axis_body_bins_px": geom.length_axis_body_bins_px,
                "length_axis_alt_px": geom.length_axis_alt_px,
                "length_axis_disagreement_px": geom.length_axis_disagreement_px,
                "centerline_disagreement": geom.centerline_disagreement,
                "endpoint_gap_alt_centerline_px": geom.endpoint_gap_alt_centerline_px,
                "diameter_max_px": geom.diameter_max_px,
                "diameter_max_orth_px": geom.diameter_max_orth_px,
                "diameter_max_thickness_px": geom.diameter_max_thickness_px,
                "diameter_max_feret_px": geom.diameter_max_feret_px,
                "diameter_p95_px": geom.diameter_p95_px,
                "diameter_mid_median_px": geom.diameter_mid_median_px,
                "diameter_mid_p90_px": geom.diameter_mid_p90_px,
                "diameter_peak_span_px": geom.diameter_peak_span_px,
                "diameter_peak_pos_norm": geom.diameter_peak_pos_norm,
                "area_proj_px2": geom.area_proj_px2,
                "area_proj_contour_width_integral_px2": geom.area_proj_contour_width_integral_px2,
                "area_proj_contour_px2": geom.area_proj_contour_px2,
                "area_proj_definition_gap_px2": geom.area_proj_definition_gap_px2,
                "body_mask_area_px2": geom.body_mask_area_px2,
                "component_area_px2": geom.component_area_px2,
                "excluded_attachment_area_px2": geom.excluded_attachment_area_px2,
                "body_mask_attachment_leak_fraction": geom.body_mask_attachment_leak_fraction,
                "attachment_count": geom.attachment_count,
                "attachment_border_touch_count": geom.attachment_border_touch_count,
                "attachment_max_elongation": geom.attachment_max_elongation,
                "attachment_max_solidity": geom.attachment_max_solidity,
                "attachment_max_orientation_mismatch_deg": geom.attachment_max_orientation_mismatch_deg,
                "attachment_max_distance_to_main_axis_px": geom.attachment_max_distance_to_main_axis_px,
                "x_peak_norm": geom.x_peak_norm,
                "taper_left_px": geom.taper_left_px,
                "taper_right_px": geom.taper_right_px,
                "landing_zone_left_px": geom.landing_zone_left_px,
                "landing_zone_right_px": geom.landing_zone_right_px,
                "transition_zone_left_px": geom.transition_zone_left_px,
                "transition_zone_right_px": geom.transition_zone_right_px,
                "compaction_zone_length_px": geom.compaction_zone_length_px,
                "zone_symmetry": geom.zone_symmetry,
                "branch_component_count_after_pruning": geom.branch_component_count_after_pruning,
                "branch_count_after_pruning": geom.branch_count_after_pruning,
                "endpoint_jump_px": geom.endpoint_jump_px,
                "axis_peak_position_stability": geom.axis_peak_position_stability,
                "centerline_points": len(geom.sampled_centerline_xy),
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
                "length_axis_skeleton_px": np.nan,
                "length_axis_body_bins_px": np.nan,
                "length_axis_alt_px": np.nan,
                "length_axis_disagreement_px": np.nan,
                "centerline_disagreement": np.nan,
                "endpoint_gap_alt_centerline_px": np.nan,
                "diameter_max_px": np.nan,
                "diameter_max_orth_px": np.nan,
                "diameter_max_thickness_px": np.nan,
                "diameter_max_feret_px": np.nan,
                "diameter_p95_px": np.nan,
                "diameter_mid_median_px": np.nan,
                "diameter_mid_p90_px": np.nan,
                "diameter_peak_span_px": np.nan,
                "diameter_peak_pos_norm": np.nan,
                "area_proj_px2": np.nan,
                "area_proj_contour_width_integral_px2": np.nan,
                "area_proj_contour_px2": np.nan,
                "area_proj_definition_gap_px2": np.nan,
                "body_mask_area_px2": np.nan,
                "component_area_px2": np.nan,
                "excluded_attachment_area_px2": np.nan,
                "body_mask_attachment_leak_fraction": np.nan,
                "attachment_count": np.nan,
                "attachment_border_touch_count": np.nan,
                "attachment_max_elongation": np.nan,
                "attachment_max_solidity": np.nan,
                "attachment_max_orientation_mismatch_deg": np.nan,
                "attachment_max_distance_to_main_axis_px": np.nan,
                "x_peak_norm": np.nan,
                "taper_left_px": np.nan,
                "taper_right_px": np.nan,
                "landing_zone_left_px": np.nan,
                "landing_zone_right_px": np.nan,
                "transition_zone_left_px": np.nan,
                "transition_zone_right_px": np.nan,
                "compaction_zone_length_px": np.nan,
                "zone_symmetry": np.nan,
                "branch_component_count_after_pruning": np.nan,
                "branch_count_after_pruning": np.nan,
                "endpoint_jump_px": np.nan,
                "axis_peak_position_stability": np.nan,
                "centerline_points": 0,
            }
        rows.append(row)
        frame_idx += 1
    cap.release()

    series = _augment_braided_qc_series(pd.DataFrame(rows))
    valid_axis = series["length_axis_px"].to_numpy(dtype=float)
    valid_env = series["length_env_px"].to_numpy(dtype=float)
    axis_ref = float(np.nanmax(valid_axis)) if np.isfinite(np.nanmax(valid_axis)) else np.nan
    env_ref = float(np.nanmax(valid_env)) if np.isfinite(np.nanmax(valid_env)) else np.nan
    series["length_axis_ref_px"] = axis_ref
    series["length_env_ref_px"] = env_ref
    if np.isfinite(axis_ref) and axis_ref > 1e-9:
        series["foreshortening_axis"] = 1.0 - series["length_axis_px"] / axis_ref
    else:
        series["foreshortening_axis"] = np.nan
    if np.isfinite(env_ref) and env_ref > 1e-9:
        series["foreshortening_env"] = 1.0 - series["length_env_px"] / env_ref
    else:
        series["foreshortening_env"] = np.nan

    fit = None
    af95_c = None
    aftan_c = None
    metric_reports = None
    primary_metric_label = None
    mode = "quicklook"
    formal_metric_label = None
    formal_gate_reason = "temperature_sync_missing"
    provisional_metric_label = None
    provisional_af95_c = None
    provisional_aftan_c = None
    reportability_status = "quicklook_only"
    warning_codes: list[str] = []
    route_results = _build_braided_route_results(
        series,
        metric_reports,
        primary_metric_label=primary_metric_label,
        formal_metric_label=formal_metric_label,
        acceptance_profile=acceptance_profile,
        temperature_available=False,
    )

    if temperature_csv is not None:
        temp = pd.read_csv(temperature_csv).copy()
        if "time_sec" in temp.columns:
            temp["time_sec"] = temp["time_sec"].astype(float) + temperature_time_offset_sec
        if "frame" in temp.columns:
            temp = temp.drop(columns=["time_sec"], errors="ignore")
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
        metric_reports = _evaluate_braided_temperature_metrics(merged)
        provisional_metric_label = _select_braided_primary_metric(merged, metric_reports)
        if provisional_metric_label is not None:
            provisional_report = metric_reports[provisional_metric_label]
            provisional_af95_c = provisional_report.af95_c
            provisional_aftan_c = provisional_report.aftan_c
            primary_metric_label = provisional_metric_label
        candidate_formal_metric_label, formal_gate_reason = _select_braided_formal_metric(
            merged,
            metric_reports,
            acceptance_profile=acceptance_profile,
        )
        if candidate_formal_metric_label is not None:
            formal_metric_label = candidate_formal_metric_label
            formal_report = metric_reports[formal_metric_label]
            fit = formal_report.fit
            af95_c = formal_report.af95_c
            aftan_c = formal_report.aftan_c
            primary_metric_label = formal_metric_label
            mode = "formal_af"
            formal_gate_reason = None
            reportability_status = "formal"
            warning_codes = []
        elif provisional_metric_label is not None:
            acceptance = compute_braided_acceptance(merged, acceptance_profile=acceptance_profile)
            reportability_status = "reportable_with_warning"
            warning_codes = acceptance["reasons"]
        series = merged
        route_results = _build_braided_route_results(
            series,
            metric_reports,
            primary_metric_label=primary_metric_label,
            formal_metric_label=formal_metric_label,
            acceptance_profile=acceptance_profile,
            temperature_available=True,
        )

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
        provisional_metric_label=provisional_metric_label,
        provisional_af95_c=provisional_af95_c,
        provisional_aftan_c=provisional_aftan_c,
        reportability_status=reportability_status,
        warning_codes=warning_codes,
        acceptance_profile=acceptance_profile if temperature_csv is not None else None,
        route_results=route_results,
    )
