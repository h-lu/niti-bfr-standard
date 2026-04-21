from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .extract import ExtractionConfig, extract_geometry
from .export_contract import canonical_object_result_fields
from .metrics import evaluate_metric
from .pipeline import WIRE_ROUTE_ALIAS_TO_KEY, WIRE_ROUTE_DISPLAY, analyze_video
from .synth import SyntheticRenderConfig, build_model_from_dict, generate_temperature_schedule, write_synthetic_dataset
from .temporal import RouteCConfig

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "minimal.yaml"


@dataclass(frozen=True)
class WireBenchmarkScenario:
    name: str
    description: str
    calibration_focus: tuple[str, ...] = field(default_factory=tuple)
    synthetic_overrides: dict[str, Any] = field(default_factory=dict)
    extraction_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class WireBenchmarkRun:
    scenario: WireBenchmarkScenario
    output_dir: Path
    summary: dict[str, Any]


WIRE_BENCHMARK_SCENARIOS: dict[str, WireBenchmarkScenario] = {
    "wire_demo": WireBenchmarkScenario(
        name="wire_demo",
        description="Baseline wire-like synthetic demo used as the reference A/B/C benchmark.",
        calibration_focus=(
            "baseline A/B/C route agreement",
            "kappa formal gate stability",
            "quicklook-to-formal route visibility",
        ),
    ),
    "wire_demo_high_noise": WireBenchmarkScenario(
        name="wire_demo_high_noise",
        description="Higher blur/noise and a slightly thinner line to stress route availability and QC robustness.",
        calibration_focus=(
            "segmentation robustness under image noise",
            "route A endpoint stability under blur",
            "route C smoothing resilience",
        ),
        synthetic_overrides={
            "line_thickness_px": 4,
            "render": {
                "noise_sigma": 4.5,
                "blur_sigma": 1.4,
            },
        },
        extraction_overrides={
            "threshold_dark": 152,
            "close_kernel": 5,
            "route_c": {
                "ema_alpha": 0.22,
                "ema_alpha_floor": 0.07,
            },
        },
    ),
    "wire_demo_low_dynamic": WireBenchmarkScenario(
        name="wire_demo_low_dynamic",
        description="Reduced curvature span to probe route-level Af stability when recovery contrast is weaker.",
        calibration_focus=(
            "dynamic-range sensitivity",
            "route C provisional visibility when curvature span tightens",
            "object-level formal candidate fallback behavior",
        ),
        synthetic_overrides={
            "kappa_m": 0.0038,
            "kappa_a": 0.00055,
            "transition_width_c": 3.1,
        },
        extraction_overrides={
            "route_c": {
                "ema_alpha": 0.28,
                "ema_alpha_floor": 0.09,
            },
        },
    ),
}

WIRE_CALIBRATION_BENCHMARK_NAMES: tuple[str, str] = (
    "wire_demo_high_noise",
    "wire_demo_low_dynamic",
)

WIRE_ROUTE_ALIAS_ORDER: tuple[str, str, str] = ("A", "B", "C")


def load_project_config(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def get_wire_benchmark_scenario(name: str) -> WireBenchmarkScenario:
    try:
        return WIRE_BENCHMARK_SCENARIOS[name]
    except KeyError as exc:
        available = ", ".join(sorted(WIRE_BENCHMARK_SCENARIOS))
        raise KeyError(f"unknown wire benchmark '{name}', available: {available}") from exc


def list_wire_benchmark_names(*, calibration_only: bool = False) -> list[str]:
    if calibration_only:
        return list(WIRE_CALIBRATION_BENCHMARK_NAMES)
    return list(WIRE_BENCHMARK_SCENARIOS)


def build_wire_benchmark_config(
    base_config: dict[str, Any],
    benchmark_name: str,
) -> dict[str, Any]:
    scenario = get_wire_benchmark_scenario(benchmark_name)
    return {
        "synthetic": _deep_merge_dicts(base_config["synthetic"], scenario.synthetic_overrides),
        "analysis": {
            "wire_like_extraction": _deep_merge_dicts(
                base_config["analysis"]["wire_like_extraction"],
                scenario.extraction_overrides,
            )
        },
    }


def flatten_wire_benchmark_summary(summary: dict[str, Any]) -> dict[str, Any]:
    row = {
        "benchmark_name": summary["benchmark_name"],
        "description": summary["benchmark_description"],
        "output_dir": summary["demo_output"],
        "object_reportability_status": summary.get("object_reportability_status"),
        "object_formal_metric_key": summary.get("object_formal_metric_key"),
        "object_formal_route_alias": summary.get("object_formal_route_alias"),
        "object_provisional_metric_key": summary.get("object_provisional_metric_key"),
        "object_provisional_route_alias": summary.get("object_provisional_route_alias"),
        "object_recommended_metric_key": summary.get("object_recommended_metric_key"),
        "object_recommended_route_alias": summary.get("object_recommended_route_alias"),
        "object_formal_gate_reason": summary.get("object_formal_gate_reason"),
        "reportability_status_legacy": summary["reportability_status"],
        "formal_metric_alias_legacy": summary.get("formal_metric_alias"),
        "primary_metric_alias_legacy": summary.get("primary_metric_alias"),
        "provisional_metric_alias_legacy": summary.get("provisional_metric_alias"),
        "quality_median": summary["wire_qc"]["quality_median"],
        "quality_lt_0_1_fraction": summary["wire_qc"]["quality_lt_0_1_fraction"],
        "route_a_endpoint_jump_p95_px": summary["wire_qc"]["route_a_endpoint_jump_p95_px"],
        "centerline_points_median": summary["wire_qc"]["centerline_points_median"],
        "quadratic_fraction": summary["wire_qc"]["quadratic_fraction"],
    }
    overview_by_alias = summary.get("route_benchmark_overview_by_alias", {})
    for alias in WIRE_ROUTE_ALIAS_ORDER:
        entry = overview_by_alias.get(alias, {})
        prefix = f"route_{alias}"
        row[f"{prefix}_metric_key"] = entry.get("metric_key")
        row[f"{prefix}_display_label"] = entry.get("display_label")
        row[f"{prefix}_available"] = entry.get("available")
        row[f"{prefix}_status"] = entry.get("reportability_status")
        row[f"{prefix}_gate_reason"] = entry.get("gate_reason")
        row[f"{prefix}_accepted_as_formal_candidate"] = entry.get("accepted_as_formal_candidate")
        row[f"{prefix}_selected_as_primary"] = entry.get("selected_as_primary")
        row[f"{prefix}_selected_as_formal"] = entry.get("selected_as_formal")
        row[f"{prefix}_af95_c"] = entry.get("measured_af95_c")
        row[f"{prefix}_aftan_c"] = entry.get("measured_aftan_c")
        row[f"{prefix}_truth_af95_c"] = entry.get("truth_af95_c")
        row[f"{prefix}_truth_aftan_c"] = entry.get("truth_aftan_c")
        row[f"{prefix}_af95_error_c"] = entry.get("af95_error_c")
        row[f"{prefix}_aftan_error_c"] = entry.get("aftan_error_c")
        row[f"{prefix}_dynamic_range"] = entry.get("dynamic_range")
        row[f"{prefix}_fit_rmse"] = entry.get("fit_rmse")
        row[f"{prefix}_monotonic_violation_fraction"] = entry.get("monotonic_violation_fraction")
    return row


def run_wire_benchmark(
    benchmark_name: str,
    *,
    root: str | Path = ROOT,
    config_path: str | Path | None = None,
    output_root: str | Path | None = None,
) -> WireBenchmarkRun:
    root = Path(root)
    resolved_config_path = Path(config_path) if config_path is not None else root / "configs" / "minimal.yaml"
    config = load_project_config(resolved_config_path)
    benchmark_cfg = build_wire_benchmark_config(config, benchmark_name)
    scenario = get_wire_benchmark_scenario(benchmark_name)
    synth_cfg = benchmark_cfg["synthetic"]
    extraction_params = benchmark_cfg["analysis"]["wire_like_extraction"]

    render_cfg = SyntheticRenderConfig(
        image_width=int(synth_cfg["image_width"]),
        image_height=int(synth_cfg["image_height"]),
        fps=int(synth_cfg["fps"]),
        duration_sec=float(synth_cfg["duration_sec"]),
        line_thickness_px=int(synth_cfg["line_thickness_px"]),
        background_gray=int(synth_cfg["render"]["background_gray"]),
        noise_sigma=float(synth_cfg["render"]["noise_sigma"]),
        blur_sigma=float(synth_cfg["render"]["blur_sigma"]),
    )
    model = build_model_from_dict(synth_cfg)
    schedule = generate_temperature_schedule(
        fps=render_cfg.fps,
        duration_sec=render_cfg.duration_sec,
        start_c=float(synth_cfg["temperature"]["start_c"]),
        end_c=float(synth_cfg["temperature"]["end_c"]),
    )
    extraction_cfg = _build_extraction_config(extraction_params)
    route_c_cfg = RouteCConfig(**extraction_params.get("route_c", {}))

    output_root_path = Path(output_root) if output_root is not None else root / "outputs"
    out_dir = output_root_path / scenario.name
    out_dir.mkdir(parents=True, exist_ok=True)

    truth = write_synthetic_dataset(
        output_dir=out_dir,
        model=model,
        render_config=render_cfg,
        schedule=schedule,
    )
    result = analyze_video(
        out_dir / "synthetic.mp4",
        extraction=extraction_cfg,
        temperature_csv=out_dir / "temperature_truth.csv",
        route_c=route_c_cfg,
    )
    result.series.to_csv(out_dir / "analysis.csv", index=False)

    truth_x_eval = evaluate_metric(
        truth["temperature_c"].to_numpy(),
        truth["x_true_px"].to_numpy(),
        label="x_true",
        increasing=True,
    )
    truth_kappa_eval = evaluate_metric(
        truth["temperature_c"].to_numpy(),
        truth["kappa_true_px_inv"].to_numpy(),
        label="kappa_true",
        increasing=False,
    )
    route_results = _copy_route_results(result.route_results)
    route_results_by_alias = _route_results_by_alias(route_results)
    route_benchmark_overview = _build_route_benchmark_overview(
        route_results=route_results,
        truth_x_eval=truth_x_eval,
        truth_kappa_eval=truth_kappa_eval,
    )
    route_benchmark_overview_by_alias = {entry["alias"]: entry for entry in route_benchmark_overview}

    x_truth = truth["x_true_px"].to_numpy(dtype=float)
    kappa_truth = truth["kappa_true_px_inv"].to_numpy(dtype=float)
    error_summary = {
        "x_route_a_mae_px": _mean_abs_error(result.series.get("x_route_a_px"), x_truth),
        "x_fit_mae_px": _mean_abs_error(result.series.get("x_fit_px"), x_truth),
        "x_route_c_mae_px": _mean_abs_error(result.series.get("x_route_c_px"), x_truth),
        "kappa_fit_mae_px_inv": _mean_abs_error(result.series.get("kappa_fit_px_inv"), kappa_truth),
        "kappa_route_c_mae_px_inv": _mean_abs_error(result.series.get("kappa_route_c_px_inv"), kappa_truth),
    }
    wire_qc = _wire_qc_summary(result.series)
    _save_benchmark_plots(out_dir, truth, result.series)
    _save_preview_overlays(out_dir, extraction_cfg, len(truth))

    summary = {
        "benchmark_name": scenario.name,
        "benchmark_description": scenario.description,
        "calibration_focus": list(scenario.calibration_focus),
        "demo_output": str(out_dir),
        "route_aliases": WIRE_ROUTE_ALIAS_TO_KEY,
        "route_display_labels": WIRE_ROUTE_DISPLAY,
        "formal_candidate_metrics": [
            {
                "label": "kappa_fit",
                "alias": "B",
            }
        ],
        "truth_frames": int(len(truth)),
        "mode": result.mode,
        "reportability_status": result.reportability_status,
        "warning_codes": result.warning_codes,
        "formal_metric_label": result.formal_metric_label,
        "formal_metric_alias": None if result.formal_metric_label is None else "B",
        "primary_metric_label": result.primary_metric_label,
        "primary_metric_alias": _wire_metric_alias(result.primary_metric_label),
        "provisional_metric_label": result.provisional_metric_label,
        "provisional_metric_alias": _wire_metric_alias(result.provisional_metric_label),
        "formal_gate_reason": result.formal_gate_reason,
        "af95_c": _finite_or_none(result.af95_c),
        "aftan_c": _finite_or_none(result.aftan_c),
        "provisional_af95_c": _finite_or_none(result.provisional_af95_c),
        "provisional_aftan_c": _finite_or_none(result.provisional_aftan_c),
        "route_results": route_results,
        "route_results_by_alias": route_results_by_alias,
        "route_alias_order": list(WIRE_ROUTE_ALIAS_ORDER),
        "truth_x_af95_c": float(truth_x_eval.af95_c),
        "truth_x_aftan_c": float(truth_x_eval.aftan_c),
        "truth_kappa_af95_c": float(truth_kappa_eval.af95_c),
        "truth_kappa_aftan_c": float(truth_kappa_eval.aftan_c),
        "route_benchmark_overview": route_benchmark_overview,
        "route_benchmark_overview_by_alias": route_benchmark_overview_by_alias,
        "wire_qc": wire_qc,
        "error_summary": error_summary,
    }
    summary.update(
        canonical_object_result_fields(
            preset="wire_like",
            reportability_status=result.reportability_status,
            formal_metric_key=result.formal_metric_label,
            formal_gate_reason=result.formal_gate_reason,
            provisional_metric_key=result.provisional_metric_label,
            af95_c=_finite_or_none(result.af95_c),
            aftan_c=_finite_or_none(result.aftan_c),
            provisional_af95_c=_finite_or_none(result.provisional_af95_c),
            provisional_aftan_c=_finite_or_none(result.provisional_aftan_c),
        )
    )
    (out_dir / "analysis_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return WireBenchmarkRun(scenario=scenario, output_dir=out_dir, summary=summary)


def _deep_merge_dicts(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key, value in base.items():
        if isinstance(value, dict):
            merged[key] = _deep_merge_dicts(value, {})
        elif isinstance(value, list):
            merged[key] = list(value)
        else:
            merged[key] = value
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
            continue
        if isinstance(value, list):
            merged[key] = list(value)
            continue
        merged[key] = value
    return merged


def _build_extraction_config(params: dict[str, Any]) -> ExtractionConfig:
    return ExtractionConfig(
        roi_xyxy=tuple(params["roi_xyxy"]),
        blur_ksize=int(params["blur_ksize"]),
        threshold_dark=int(params["threshold_dark"]),
        open_kernel=int(params["open_kernel"]),
        close_kernel=int(params["close_kernel"]),
        route_a_tip_cluster_radius_px=float(params.get("route_a_tip_cluster_radius_px", 6.0)),
        route_b_endpoint_extension_scale=float(params.get("route_b_endpoint_extension_scale", 0.0)),
        route_b_cap_inset_scale=float(params.get("route_b_cap_inset_scale", 0.08)),
        fit_bin_px=float(params.get("fit_bin_px", 3.0)),
        fit_path_fraction=float(params.get("fit_path_fraction", 0.72)),
        fit_path_fraction_min=float(params.get("fit_path_fraction_min", 0.42)),
        fit_curvature_threshold_ratio=float(params.get("fit_curvature_threshold_ratio", 0.28)),
        fit_margin_prefer_quadratic=float(params.get("fit_margin_prefer_quadratic", 0.05)),
        anchor_prior_xy=tuple(params["anchor_prior_xy"]) if params.get("anchor_prior_xy") is not None else None,
        anchor_prior_weight=float(params.get("anchor_prior_weight", 0.0)),
    )


def _copy_route_results(route_results: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [dict(entry) for entry in (route_results or [])]


def _route_results_by_alias(route_results: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    return {entry["alias"]: entry for entry in _copy_route_results(route_results) if isinstance(entry.get("alias"), str)}


def _wire_metric_alias(metric_key: str | None) -> str | None:
    if metric_key is None:
        return None
    for alias, key in WIRE_ROUTE_ALIAS_TO_KEY.items():
        if key == metric_key:
            return alias
    return None


def _truth_eval_for_alias(alias: str, truth_x_eval: Any, truth_kappa_eval: Any) -> Any:
    return truth_x_eval if alias == "A" else truth_kappa_eval


def _build_route_benchmark_overview(
    *,
    route_results: list[dict[str, Any]],
    truth_x_eval: Any,
    truth_kappa_eval: Any,
) -> list[dict[str, Any]]:
    overview: list[dict[str, Any]] = []
    for alias in WIRE_ROUTE_ALIAS_ORDER:
        route_result = next((entry for entry in route_results if entry.get("alias") == alias), None)
        if route_result is None:
            continue
        truth_eval = _truth_eval_for_alias(alias, truth_x_eval, truth_kappa_eval)
        measured_af95 = _finite_or_none(route_result.get("af95_c"))
        measured_aftan = _finite_or_none(route_result.get("aftan_c"))
        truth_af95 = _finite_or_none(truth_eval.af95_c)
        truth_aftan = _finite_or_none(truth_eval.aftan_c)
        overview.append(
            {
                "alias": alias,
                "metric_key": route_result.get("metric_key"),
                "display_label": route_result.get("display_label"),
                "truth_metric_key": "x_true" if alias == "A" else "kappa_true",
                "truth_display_label": "truth:x_true" if alias == "A" else "truth:kappa_true",
                "available": measured_af95 is not None and measured_aftan is not None,
                "reportability_status": route_result.get("reportability_status"),
                "gate_reason": route_result.get("gate_reason"),
                "warning_codes": list(route_result.get("warning_codes") or []),
                "accepted_as_formal_candidate": bool(route_result.get("accepted_as_formal_candidate")),
                "selected_as_primary": bool(route_result.get("selected_as_primary")),
                "selected_as_formal": bool(route_result.get("selected_as_formal")),
                "measured_af95_c": measured_af95,
                "truth_af95_c": truth_af95,
                "af95_error_c": _delta_or_none(measured_af95, truth_af95),
                "measured_aftan_c": measured_aftan,
                "truth_aftan_c": truth_aftan,
                "aftan_error_c": _delta_or_none(measured_aftan, truth_aftan),
                "fit_rmse": _finite_or_none(route_result.get("fit_rmse")),
                "dynamic_range": _finite_or_none(route_result.get("dynamic_range")),
                "monotonic_violation_fraction": _finite_or_none(route_result.get("monotonic_violation_fraction")),
            }
        )
    return overview


def _wire_qc_summary(series: pd.DataFrame) -> dict[str, float | None]:
    endpoint_jump_p95 = _frame_jump_p95(series.get("route_a_anchor_x"), series.get("route_a_anchor_y"))
    tip_jump_p95 = _frame_jump_p95(series.get("route_a_tip_x"), series.get("route_a_tip_y"))
    route_a_endpoint_jump_p95 = _nanmax_or_none([endpoint_jump_p95, tip_jump_p95])
    return {
        "quality_median": _finite_or_none(series.get("quality").median()) if "quality" in series else None,
        "quality_lt_0_1_fraction": (
            float((series["quality"].to_numpy(dtype=float) < 0.1).mean()) if "quality" in series else None
        ),
        "route_a_endpoint_jump_p95_px": route_a_endpoint_jump_p95,
        "centerline_points_median": (
            _finite_or_none(series.get("centerline_points").median()) if "centerline_points" in series else None
        ),
        "quadratic_fraction": (
            float((series["model_name"] == "quadratic").mean()) if "model_name" in series else None
        ),
    }


def _mean_abs_error(measured: pd.Series | np.ndarray | None, truth: pd.Series | np.ndarray | None) -> float | None:
    if measured is None or truth is None:
        return None
    measured_arr = np.asarray(measured, dtype=float)
    truth_arr = np.asarray(truth, dtype=float)
    valid = np.isfinite(measured_arr) & np.isfinite(truth_arr)
    if not np.any(valid):
        return None
    return float(np.mean(np.abs(measured_arr[valid] - truth_arr[valid])))


def _delta_or_none(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return numeric


def _frame_jump_p95(x_values: pd.Series | np.ndarray | None, y_values: pd.Series | np.ndarray | None) -> float | None:
    if x_values is None or y_values is None:
        return None
    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if np.count_nonzero(valid) < 2:
        return None
    points = np.column_stack([x_arr[valid], y_arr[valid]])
    jumps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if len(jumps) == 0:
        return 0.0
    return float(np.percentile(jumps, 95))


def _nanmax_or_none(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not finite:
        return None
    return float(np.max(finite))


def _load_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read frame {frame_idx}")
    return frame


def _make_overlay(frame_bgr: np.ndarray, extraction_cfg: ExtractionConfig) -> np.ndarray:
    geom = extract_geometry(frame_bgr, extraction_cfg)
    overlay = frame_bgr.copy()
    x0, y0, x1, y1 = extraction_cfg.roi_xyxy
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (80, 180, 255), 2)
    contour = np.round(geom.contour_xy).astype(np.int32).reshape(-1, 1, 2)
    centerline = np.round(geom.sampled_centerline_xy).astype(np.int32).reshape(-1, 1, 2)
    fit_curve = np.round(geom.fitted_curve_xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [contour], isClosed=True, color=(120, 120, 120), thickness=1, lineType=cv2.LINE_AA)
    cv2.line(
        overlay,
        np.round(geom.route_a_anchor_xy).astype(int),
        np.round(geom.route_a_tip_xy).astype(int),
        (220, 0, 220),
        2,
        cv2.LINE_AA,
    )
    cv2.polylines(overlay, [centerline], isClosed=False, color=(0, 140, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.polylines(overlay, [fit_curve], isClosed=False, color=(255, 180, 0), thickness=2, lineType=cv2.LINE_AA)
    return overlay


def _save_preview_overlays(out_dir: Path, extraction_cfg: ExtractionConfig, frame_count: int) -> None:
    preview_dir = out_dir / "qc_frames"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_frames = np.linspace(0, frame_count - 1, 5, dtype=int)
    for frame_idx in preview_frames:
        frame = _load_frame(out_dir / "synthetic.mp4", int(frame_idx))
        overlay = _make_overlay(frame, extraction_cfg)
        cv2.imwrite(str(preview_dir / f"frame_{frame_idx:04d}.png"), overlay)


def _save_benchmark_plots(out_dir: Path, truth: pd.DataFrame, series: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 4.8))
    plt.plot(truth["temperature_c"], truth["x_true_px"], label="truth x", linewidth=2)
    plt.plot(series["temperature_c"], series["x_route_a_px"], label="A measured", alpha=0.85)
    plt.plot(series["temperature_c"], series["x_fit_px"], label="B auxiliary x", alpha=0.8)
    plt.plot(series["temperature_c"], series["x_route_c_px"], label="C auxiliary x", alpha=0.8)
    plt.xlabel("Temperature (C)")
    plt.ylabel("Chord length (px)")
    plt.title("Wire benchmark x-route comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "x_routes_vs_temperature.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.plot(truth["temperature_c"], truth["kappa_true_px_inv"], label="truth kappa", linewidth=2)
    plt.plot(series["temperature_c"], series["kappa_fit_px_inv"], label="B measured", alpha=0.85)
    plt.plot(series["temperature_c"], series["kappa_route_c_px_inv"], label="C measured", alpha=0.85)
    plt.xlabel("Temperature (C)")
    plt.ylabel("Curvature (px^-1)")
    plt.title("Wire benchmark kappa comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "kappa_routes_vs_temperature.png", dpi=160)
    plt.close()

    recovery_cols = {"x_route_a_recovery", "kappa_fit_recovery", "kappa_route_c_recovery"}
    if recovery_cols.issubset(series.columns):
        plt.figure(figsize=(8, 4.8))
        plt.plot(series["temperature_c"], series["x_route_a_recovery"], label="A recovery", linewidth=1.9)
        plt.plot(series["temperature_c"], series["kappa_fit_recovery"], label="B recovery", linewidth=1.9)
        plt.plot(series["temperature_c"], series["kappa_route_c_recovery"], label="C recovery", linewidth=1.9)
        plt.xlabel("Temperature (C)")
        plt.ylabel("Recovery ratio")
        plt.title("Wire benchmark route recovery comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "recovery_vs_temperature.png", dpi=160)
        plt.close()
