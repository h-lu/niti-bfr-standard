from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from niti_bfr.extract_braided import BraidedExtractionConfig, extract_braided_geometry
from niti_bfr.metrics import evaluate_metric, infer_metric_direction, summarize_numeric_sweep
from niti_bfr.pipeline import (
    BRAIDED_FORMAL_CANDIDATES,
    BRAIDED_METRIC_ALIAS_TO_KEY,
    BRAIDED_METRIC_DISPLAY,
    BRAIDED_METRIC_KEY_TO_ALIAS,
    analyze_braided_video_quicklook,
    compute_braided_acceptance,
)
from niti_bfr.synth import generate_temperature_schedule
from niti_bfr.synth_braided import (
    BraidedSyntheticModel,
    BraidedSyntheticModelConfig,
    BraidedSyntheticRenderConfig,
    write_braided_synthetic_dataset,
)


def _mean_abs_error(measured: np.ndarray, truth: np.ndarray) -> float:
    measured = np.asarray(measured, dtype=float)
    truth = np.asarray(truth, dtype=float)
    valid = np.isfinite(measured) & np.isfinite(truth)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.abs(measured[valid] - truth[valid])))


def _axis_eval_series(series) -> np.ndarray:
    col = "length_axis_formal_px" if "length_axis_formal_px" in series.columns else "length_axis_px"
    return series[col].to_numpy()


def _braided_alias(metric_key: str) -> str:
    return BRAIDED_METRIC_KEY_TO_ALIAS.get(metric_key, metric_key)


def _load_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read frame {frame_idx}")
    return frame


def _make_overlay(frame_bgr: np.ndarray, extraction_cfg: BraidedExtractionConfig) -> np.ndarray:
    geom = extract_braided_geometry(frame_bgr, extraction_cfg)
    overlay = frame_bgr.copy()
    x0, y0, x1, y1 = extraction_cfg.roi_xyxy
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (80, 180, 255), 2)
    contour = np.round(geom.contour_xy).astype(np.int32).reshape(-1, 1, 2)
    body_contour = np.round(geom.body_contour_xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [contour], isClosed=True, color=(120, 120, 120), thickness=1, lineType=cv2.LINE_AA)
    cv2.polylines(overlay, [body_contour], isClosed=True, color=(40, 200, 120), thickness=2, lineType=cv2.LINE_AA)
    cv2.line(
        overlay,
        np.round(geom.axis_line_xy[0]).astype(int),
        np.round(geom.axis_line_xy[1]).astype(int),
        (0, 140, 255),
        2,
        cv2.LINE_AA,
    )
    for segment in geom.sampled_width_segments_xy:
        cv2.line(
            overlay,
            np.round(segment[0]).astype(int),
            np.round(segment[1]).astype(int),
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.circle(overlay, np.round(geom.anchor_xy).astype(int), 5, (0, 220, 0), -1)
    cv2.circle(overlay, np.round(geom.tip_xy).astype(int), 5, (0, 0, 255), -1)
    peak_xy = geom.anchor_xy + geom.x_peak_norm * (geom.tip_xy - geom.anchor_xy)
    cv2.circle(overlay, np.round(peak_xy).astype(int), 5, (255, 180, 0), -1)
    axis_vector = geom.tip_xy - geom.anchor_xy
    axis_length = max(float(np.linalg.norm(axis_vector)), 1e-9)
    axis_unit = axis_vector / axis_length
    left_transition_end = geom.anchor_xy + (geom.landing_zone_left_px + geom.transition_zone_left_px) * axis_unit
    right_transition_start = geom.tip_xy - (geom.landing_zone_right_px + geom.transition_zone_right_px) * axis_unit
    for point_xy, color in [
        (geom.anchor_xy + geom.landing_zone_left_px * axis_unit, (0, 200, 255)),
        (left_transition_end, (255, 180, 0)),
        (right_transition_start, (255, 180, 0)),
        (geom.tip_xy - geom.landing_zone_right_px * axis_unit, (0, 200, 255)),
    ]:
        if np.all(np.isfinite(point_xy)):
            cv2.circle(overlay, np.round(point_xy).astype(int), 4, color, -1)
    label = (
        f"axis={geom.length_axis_px:.1f}|bins={geom.length_axis_body_bins_px:.1f}|alt={geom.length_axis_alt_px:.1f}px "
        f"Dmax={geom.diameter_max_px:.1f}|p90={geom.diameter_mid_p90_px:.1f}px "
        f"Cproj={geom.area_proj_px2:.0f}|contourInt={geom.area_proj_contour_width_integral_px2:.0f}px2"
    )
    cv2.putText(overlay, label, (x0 + 8, max(24, y0 + 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
    return overlay


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/minimal.yaml").read_text(encoding="utf-8"))
    synth_cfg = config["braided_synthetic"]
    render_cfg = BraidedSyntheticRenderConfig(
        image_width=int(synth_cfg["image_width"]),
        image_height=int(synth_cfg["image_height"]),
        fps=int(synth_cfg["fps"]),
        duration_sec=float(synth_cfg["duration_sec"]),
        background_gray=int(synth_cfg["render"]["background_gray"]),
        body_gray=int(synth_cfg["render"]["body_gray"]),
        wire_gray=int(synth_cfg["render"]["wire_gray"]),
        outline_gray=int(synth_cfg["render"]["outline_gray"]),
        braid_spacing_px=int(synth_cfg["render"]["braid_spacing_px"]),
        wire_thickness_px=int(synth_cfg["render"]["wire_thickness_px"]),
        outline_thickness_px=int(synth_cfg["render"]["outline_thickness_px"]),
        support_gray=int(synth_cfg["render"]["support_gray"]),
        support_length_px=int(synth_cfg["render"]["support_length_px"]),
        support_radius_px=int(synth_cfg["render"]["support_radius_px"]),
        tip_cap_gray=int(synth_cfg["render"]["tip_cap_gray"]),
        tip_cap_length_px=int(synth_cfg["render"]["tip_cap_length_px"]),
        tip_cap_radius_px=int(synth_cfg["render"]["tip_cap_radius_px"]),
        noise_sigma=float(synth_cfg["render"]["noise_sigma"]),
        blur_sigma=float(synth_cfg["render"]["blur_sigma"]),
    )
    model = BraidedSyntheticModel(
        BraidedSyntheticModelConfig(
            center_xy=tuple(float(v) for v in synth_cfg["center_xy"]),
            length_m_px=float(synth_cfg["length_m_px"]),
            length_a_px=float(synth_cfg["length_a_px"]),
            diameter_m_px=float(synth_cfg["diameter_m_px"]),
            diameter_a_px=float(synth_cfg["diameter_a_px"]),
            transition_temp_c=float(synth_cfg["transition_temp_c"]),
            transition_width_c=float(synth_cfg["transition_width_c"]),
            peak_shift_norm=float(synth_cfg["peak_shift_norm"]),
            left_profile_power=float(synth_cfg["left_profile_power"]),
            right_profile_power=float(synth_cfg["right_profile_power"]),
            bow_m_px=float(synth_cfg["bow_m_px"]),
            bow_a_px=float(synth_cfg["bow_a_px"]),
            axis_angle_m_deg=float(synth_cfg["axis_angle_m_deg"]),
            axis_angle_a_deg=float(synth_cfg["axis_angle_a_deg"]),
            center_shift_a_xy=tuple(float(v) for v in synth_cfg["center_shift_a_xy"]),
        )
    )
    schedule = generate_temperature_schedule(
        fps=render_cfg.fps,
        duration_sec=render_cfg.duration_sec,
        start_c=float(synth_cfg["temperature"]["start_c"]),
        end_c=float(synth_cfg["temperature"]["end_c"]),
    )

    extraction_params = config["analysis"]["braided_device_extraction"]
    extraction_cfg = BraidedExtractionConfig(
        roi_xyxy=tuple(extraction_params["roi_xyxy"]),
        blur_ksize=int(extraction_params["blur_ksize"]),
        threshold_dark=int(extraction_params["threshold_dark"]),
        open_kernel=int(extraction_params["open_kernel"]),
        close_kernel=int(extraction_params["close_kernel"]),
        min_component_area=int(extraction_params.get("min_component_area", 200)),
        width_sampling_step_px=float(extraction_params.get("width_sampling_step_px", 4.0)),
        taper_threshold_ratio=float(extraction_params.get("taper_threshold_ratio", 0.25)),
        compaction_threshold_ratio=float(extraction_params.get("compaction_threshold_ratio", 0.75)),
        qc_max_segments=int(extraction_params.get("qc_max_segments", 15)),
        tube_radius_scale=float(extraction_params.get("tube_radius_scale", 1.0)),
        body_min_halfwidth_px=float(extraction_params.get("body_min_halfwidth_px", 3.0)),
        centerline_smooth_window=int(extraction_params.get("centerline_smooth_window", 7)),
        diameter_peak_threshold_ratio=float(extraction_params.get("diameter_peak_threshold_ratio", 0.95)),
        attachment_min_area_px2=int(extraction_params.get("attachment_min_area_px2", 24)),
        attachment_orientation_mismatch_deg=float(extraction_params.get("attachment_orientation_mismatch_deg", 35.0)),
        attachment_far_axis_distance_ratio=float(extraction_params.get("attachment_far_axis_distance_ratio", 0.78)),
        attachment_short_axis_span_ratio=float(extraction_params.get("attachment_short_axis_span_ratio", 0.18)),
        attachment_short_normal_span_ratio=float(extraction_params.get("attachment_short_normal_span_ratio", 0.55)),
        attachment_prune_dilate_kernel=int(extraction_params.get("attachment_prune_dilate_kernel", 1)),
        diameter_proxy_window_start_norm=float(extraction_params.get("diameter_proxy_window_start_norm", 0.6)),
        diameter_proxy_window_end_norm=float(extraction_params.get("diameter_proxy_window_end_norm", 0.8)),
    )

    out_dir = ROOT / "outputs" / "braided_demo"
    truth = write_braided_synthetic_dataset(
        output_dir=out_dir,
        model=model,
        render_config=render_cfg,
        schedule=schedule,
        taper_threshold_ratio=extraction_cfg.taper_threshold_ratio,
        compaction_threshold_ratio=extraction_cfg.compaction_threshold_ratio,
    )
    result = analyze_braided_video_quicklook(
        out_dir / "synthetic.mp4",
        extraction=extraction_cfg,
        temperature_csv=out_dir / "truth.csv",
    )
    result.series.to_csv(out_dir / "analysis.csv", index=False)

    truth_axis_eval = evaluate_metric(
        truth["temperature_c"].to_numpy(),
        truth["length_axis_true_px"].to_numpy(),
        label="length_axis_true",
        increasing=False,
    )
    truth_env_eval = evaluate_metric(
        truth["temperature_c"].to_numpy(),
        truth["length_env_true_px"].to_numpy(),
        label="length_env_true",
        increasing=False,
    )
    truth_diameter_eval = evaluate_metric(
        truth["temperature_c"].to_numpy(),
        truth["diameter_true_px"].to_numpy(),
        label="diameter_true",
        increasing=True,
    )
    area_increasing = infer_metric_direction(truth["area_proj_true_px2"].to_numpy(), default_increasing=True)
    truth_area_eval = evaluate_metric(
        truth["temperature_c"].to_numpy(),
        truth["area_proj_true_px2"].to_numpy(),
        label="area_proj_true",
        increasing=area_increasing,
    )
    measured_axis_eval = evaluate_metric(
        result.series["temperature_c"].to_numpy(),
        _axis_eval_series(result.series),
        label="length_axis_measured",
        increasing=False,
    )
    measured_diameter_eval = evaluate_metric(
        result.series["temperature_c"].to_numpy(),
        result.series["diameter_max_px"].to_numpy(),
        label="diameter_measured",
        increasing=True,
    )
    measured_area_eval = evaluate_metric(
        result.series["temperature_c"].to_numpy(),
        result.series["area_proj_px2"].to_numpy(),
        label="area_proj_measured",
        increasing=area_increasing,
    )
    diameter_threshold_sweep: dict[str, dict[str, float]] = {}
    for threshold_delta in (-8, 0, 8):
        sweep_cfg = replace(extraction_cfg, threshold_dark=int(extraction_cfg.threshold_dark + threshold_delta))
        sweep_result = analyze_braided_video_quicklook(
            out_dir / "synthetic.mp4",
            extraction=sweep_cfg,
            temperature_csv=out_dir / "truth.csv",
        )
        sweep_eval = evaluate_metric(
            sweep_result.series["temperature_c"].to_numpy(),
            sweep_result.series["diameter_max_px"].to_numpy(),
            label=f"diameter_threshold_{sweep_cfg.threshold_dark}",
            increasing=True,
        )
        diameter_threshold_sweep[str(sweep_cfg.threshold_dark)] = {
            "quality_median": float(sweep_result.series["quality"].median()),
            "diameter_max_mae_px": _mean_abs_error(sweep_result.series["diameter_max_px"], truth["diameter_true_px"]),
            "af95_c": float(sweep_eval.af95_c),
            "aftan_c": float(sweep_eval.aftan_c),
            "af95_error_c": float(sweep_eval.af95_c - truth_diameter_eval.af95_c),
            "aftan_error_c": float(sweep_eval.aftan_c - truth_diameter_eval.aftan_c),
        }
    diameter_threshold_sensitivity = summarize_numeric_sweep(
        diameter_threshold_sweep,
        baseline_key=str(extraction_cfg.threshold_dark),
    )

    plt.figure(figsize=(8, 4.8))
    plt.plot(truth["temperature_c"], truth["length_env_true_px"], label="envelope length true", linewidth=2, alpha=0.85)
    plt.plot(truth["temperature_c"], truth["length_axis_true_px"], label="A true", linewidth=2, alpha=0.85)
    plt.plot(result.series["temperature_c"], result.series["length_env_px"], label="envelope length", alpha=0.8)
    plt.plot(result.series["temperature_c"], result.series["length_axis_px"], label="A measured", alpha=0.8)
    plt.xlabel("Temperature (C)")
    plt.ylabel("Length (px)")
    plt.title("Braided demo length comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "length_vs_temperature.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.plot(truth["temperature_c"], truth["diameter_true_px"], label="B true", linewidth=2)
    plt.plot(result.series["temperature_c"], result.series["diameter_max_px"], label="B measured", alpha=0.9)
    plt.plot(result.series["temperature_c"], result.series["diameter_mid_p90_px"], label="mid-window p90 proxy", alpha=0.8)
    plt.plot(result.series["temperature_c"], result.series["diameter_mid_median_px"], label="mid-window median", alpha=0.75)
    plt.plot(result.series["temperature_c"], result.series["diameter_p95_px"], label="diameter body p95", alpha=0.75)
    plt.plot(result.series["temperature_c"], result.series["diameter_max_thickness_px"], label="diameter thickness", alpha=0.75)
    plt.xlabel("Temperature (C)")
    plt.ylabel("Diameter (px)")
    plt.title("Braided demo D_max vs width proxies")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "diameter_vs_temperature.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.plot(truth["temperature_c"], truth["area_proj_true_px2"], label="C true", linewidth=2)
    plt.plot(result.series["temperature_c"], result.series["area_proj_px2"], label="C measured", alpha=0.85)
    plt.plot(
        result.series["temperature_c"],
        result.series["area_proj_contour_width_integral_px2"],
        label="contour-width integral",
        alpha=0.7,
    )
    plt.xlabel("Temperature (C)")
    plt.ylabel("Projected area (px^2)")
    plt.title("Braided demo A_proj vs truth area")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "area_vs_temperature.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.plot(truth["temperature_c"], truth["body_mask_area_true_px2"], label="body mask true", linewidth=2)
    plt.plot(result.series["temperature_c"], result.series["body_mask_area_px2"], label="body mask measured", alpha=0.85)
    plt.xlabel("Temperature (C)")
    plt.ylabel("Body mask area (px^2)")
    plt.title("Braided body-only area comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "body_mask_vs_temperature.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.plot(result.series["temperature_c"], result.series["length_axis_disagreement_px"], label="axis gap measured", linewidth=1.9)
    plt.plot(result.series["temperature_c"], result.series["centerline_disagreement"], label="axis gap / A", alpha=0.85)
    plt.xlabel("Temperature (C)")
    plt.ylabel("Centerline QC")
    plt.title("Braided axis definition QC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "axis_definition_gap_vs_temperature.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.plot(truth["temperature_c"], truth["diameter_peak_pos_norm_true"], label="peak pos true", linewidth=2)
    plt.plot(result.series["temperature_c"], result.series["diameter_peak_pos_norm"], label="peak pos measured", alpha=0.85)
    plt.xlabel("Temperature (C)")
    plt.ylabel("Peak position (norm)")
    plt.title("Braided Dmax peak position")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "dmax_peak_position_vs_temperature.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.plot(truth["temperature_c"], truth["foreshortening_axis_true"], label="FS axis true", linewidth=2, alpha=0.85)
    plt.plot(truth["temperature_c"], truth["foreshortening_env_true"], label="FS env true", linewidth=2, alpha=0.85)
    plt.plot(result.series["temperature_c"], result.series["foreshortening_axis"], label="FS axis", alpha=0.85)
    plt.plot(result.series["temperature_c"], result.series["foreshortening_env"], label="FS env", alpha=0.85)
    plt.xlabel("Temperature (C)")
    plt.ylabel("Foreshortening")
    plt.title("Braided demo foreshortening comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "foreshortening_vs_temperature.png", dpi=160)
    plt.close()

    if {"length_axis_recovery", "length_env_recovery", "diameter_max_recovery", "area_proj_recovery"}.issubset(result.series.columns):
        plt.figure(figsize=(8, 4.8))
        plt.plot(result.series["temperature_c"], result.series["length_axis_recovery"], label="A recovery", linewidth=2)
        plt.plot(result.series["temperature_c"], result.series["length_env_recovery"], label="env recovery", linewidth=1.8)
        plt.plot(result.series["temperature_c"], result.series["diameter_max_recovery"], label="B recovery", linewidth=1.8)
        plt.plot(result.series["temperature_c"], result.series["area_proj_recovery"], label="C recovery", linewidth=1.8)
        plt.xlabel("Temperature (C)")
        plt.ylabel("Recovery ratio")
        plt.title("Braided demo recovery comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "recovery_vs_temperature.png", dpi=160)
        plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.plot(truth["temperature_c"], truth["landing_zone_left_true_px"], label="landing left true", linewidth=1.8)
    plt.plot(result.series["temperature_c"], result.series["landing_zone_left_px"], label="landing left", alpha=0.85)
    plt.plot(truth["temperature_c"], truth["transition_zone_left_true_px"], label="transition left true", linewidth=1.8)
    plt.plot(result.series["temperature_c"], result.series["transition_zone_left_px"], label="transition left", alpha=0.85)
    plt.plot(truth["temperature_c"], truth["compaction_zone_length_true_px"], label="compaction true", linewidth=1.8)
    plt.plot(result.series["temperature_c"], result.series["compaction_zone_length_px"], label="compaction", alpha=0.85)
    plt.xlabel("Temperature (C)")
    plt.ylabel("Length (px)")
    plt.title("Braided demo zone comparison")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "zones_vs_temperature.png", dpi=160)
    plt.close()

    error_summary = {
        "length_env_mae_px": _mean_abs_error(result.series["length_env_px"], truth["length_env_true_px"]),
        "length_axis_mae_px": _mean_abs_error(result.series["length_axis_px"], truth["length_axis_true_px"]),
        "diameter_max_mae_px": _mean_abs_error(result.series["diameter_max_px"], truth["diameter_true_px"]),
        "diameter_threshold_sensitivity_mae_px": diameter_threshold_sensitivity["baseline_max_delta"].get(
            "diameter_max_mae_px"
        ),
        "diameter_threshold_sensitivity_af95_c": diameter_threshold_sensitivity["baseline_max_delta"].get("af95_c"),
        "diameter_threshold_sensitivity_aftan_c": diameter_threshold_sensitivity["baseline_max_delta"].get("aftan_c"),
        "diameter_peak_pos_norm_mae": _mean_abs_error(result.series["diameter_peak_pos_norm"], truth["diameter_peak_pos_norm_true"]),
        "area_proj_mae_px2": _mean_abs_error(result.series["area_proj_px2"], truth["area_proj_true_px2"]),
        "area_proj_definition_gap_px2": _mean_abs_error(result.series["area_proj_definition_gap_px2"], truth["area_proj_definition_gap_true_px2"]),
        "body_mask_area_mae_px2": _mean_abs_error(result.series["body_mask_area_px2"], truth["body_mask_area_true_px2"]),
        "x_peak_norm_mae": _mean_abs_error(result.series["x_peak_norm"], truth["x_peak_norm_true"]),
        "foreshortening_axis_mae": _mean_abs_error(result.series["foreshortening_axis"], truth["foreshortening_axis_true"]),
        "foreshortening_env_mae": _mean_abs_error(result.series["foreshortening_env"], truth["foreshortening_env_true"]),
        "taper_left_mae_px": _mean_abs_error(result.series["taper_left_px"], truth["taper_left_true_px"]),
        "taper_right_mae_px": _mean_abs_error(result.series["taper_right_px"], truth["taper_right_true_px"]),
        "landing_zone_left_mae_px": _mean_abs_error(result.series["landing_zone_left_px"], truth["landing_zone_left_true_px"]),
        "landing_zone_right_mae_px": _mean_abs_error(result.series["landing_zone_right_px"], truth["landing_zone_right_true_px"]),
        "transition_zone_left_mae_px": _mean_abs_error(result.series["transition_zone_left_px"], truth["transition_zone_left_true_px"]),
        "transition_zone_right_mae_px": _mean_abs_error(result.series["transition_zone_right_px"], truth["transition_zone_right_true_px"]),
        "compaction_zone_length_mae_px": _mean_abs_error(result.series["compaction_zone_length_px"], truth["compaction_zone_length_true_px"]),
        "zone_symmetry_mae": _mean_abs_error(result.series["zone_symmetry"], truth["zone_symmetry_true"]),
    }
    formal_qc = compute_braided_acceptance(result.series)
    summary = {
        "demo_output": str(out_dir),
        "metric_aliases": BRAIDED_METRIC_ALIAS_TO_KEY,
        "metric_display_labels": BRAIDED_METRIC_DISPLAY,
        "formal_candidate_metrics": [
            {
                "label": metric_label,
                "alias": _braided_alias(metric_label),
            }
            for metric_label in BRAIDED_FORMAL_CANDIDATES
        ],
        "formal_qc_scope": "current braided formal-gate QC snapshot; not an overall A/B/C verdict",
        "truth_frames": int(len(truth)),
        "mode": result.mode,
        "formal_metric_label": result.formal_metric_label,
        "formal_metric_alias": None if result.formal_metric_label is None else _braided_alias(result.formal_metric_label),
        "primary_metric_label": result.primary_metric_label,
        "primary_metric_alias": None if result.primary_metric_label is None else _braided_alias(result.primary_metric_label),
        "formal_gate_reason": result.formal_gate_reason,
        "af95_c": None if result.af95_c is None else float(result.af95_c),
        "aftan_c": None if result.aftan_c is None else float(result.aftan_c),
        "truth_axis_af95_c": float(truth_axis_eval.af95_c),
        "truth_axis_aftan_c": float(truth_axis_eval.aftan_c),
        "truth_env_af95_c": float(truth_env_eval.af95_c),
        "truth_env_aftan_c": float(truth_env_eval.aftan_c),
        "truth_diameter_af95_c": float(truth_diameter_eval.af95_c),
        "truth_diameter_aftan_c": float(truth_diameter_eval.aftan_c),
        "truth_area_af95_c": float(truth_area_eval.af95_c),
        "truth_area_aftan_c": float(truth_area_eval.aftan_c),
        "measured_axis_af95_c": float(measured_axis_eval.af95_c),
        "measured_axis_aftan_c": float(measured_axis_eval.aftan_c),
        "measured_diameter_af95_c": float(measured_diameter_eval.af95_c),
        "measured_diameter_aftan_c": float(measured_diameter_eval.aftan_c),
        "measured_area_af95_c": float(measured_area_eval.af95_c),
        "measured_area_aftan_c": float(measured_area_eval.aftan_c),
        "af_comparison": {
            "length_axis": {
                "alias": "A",
                "measured_af95_c": float(measured_axis_eval.af95_c),
                "truth_af95_c": float(truth_axis_eval.af95_c),
                "af95_error_c": float(measured_axis_eval.af95_c - truth_axis_eval.af95_c),
                "measured_aftan_c": float(measured_axis_eval.aftan_c),
                "truth_aftan_c": float(truth_axis_eval.aftan_c),
                "aftan_error_c": float(measured_axis_eval.aftan_c - truth_axis_eval.aftan_c),
            },
            "diameter_max": {
                "alias": "B",
                "measured_af95_c": float(measured_diameter_eval.af95_c),
                "truth_af95_c": float(truth_diameter_eval.af95_c),
                "af95_error_c": float(measured_diameter_eval.af95_c - truth_diameter_eval.af95_c),
                "measured_aftan_c": float(measured_diameter_eval.aftan_c),
                "truth_aftan_c": float(truth_diameter_eval.aftan_c),
                "aftan_error_c": float(measured_diameter_eval.aftan_c - truth_diameter_eval.aftan_c),
            },
            "area_proj": {
                "alias": "C",
                "measured_af95_c": float(measured_area_eval.af95_c),
                "truth_af95_c": float(truth_area_eval.af95_c),
                "af95_error_c": float(measured_area_eval.af95_c - truth_area_eval.af95_c),
                "measured_aftan_c": float(measured_area_eval.aftan_c),
                "truth_aftan_c": float(truth_area_eval.aftan_c),
                "aftan_error_c": float(measured_area_eval.aftan_c - truth_area_eval.aftan_c),
            },
        },
        "af_comparison_by_alias": {
            "A": {
                "metric_key": "length_axis",
                "measured_af95_c": float(measured_axis_eval.af95_c),
                "truth_af95_c": float(truth_axis_eval.af95_c),
                "af95_error_c": float(measured_axis_eval.af95_c - truth_axis_eval.af95_c),
                "measured_aftan_c": float(measured_axis_eval.aftan_c),
                "truth_aftan_c": float(truth_axis_eval.aftan_c),
                "aftan_error_c": float(measured_axis_eval.aftan_c - truth_axis_eval.aftan_c),
            },
            "B": {
                "metric_key": "diameter_max",
                "measured_af95_c": float(measured_diameter_eval.af95_c),
                "truth_af95_c": float(truth_diameter_eval.af95_c),
                "af95_error_c": float(measured_diameter_eval.af95_c - truth_diameter_eval.af95_c),
                "measured_aftan_c": float(measured_diameter_eval.aftan_c),
                "truth_aftan_c": float(truth_diameter_eval.aftan_c),
                "aftan_error_c": float(measured_diameter_eval.aftan_c - truth_diameter_eval.aftan_c),
            },
            "C": {
                "metric_key": "area_proj",
                "measured_af95_c": float(measured_area_eval.af95_c),
                "truth_af95_c": float(truth_area_eval.af95_c),
                "af95_error_c": float(measured_area_eval.af95_c - truth_area_eval.af95_c),
                "measured_aftan_c": float(measured_area_eval.aftan_c),
                "truth_aftan_c": float(truth_area_eval.aftan_c),
                "aftan_error_c": float(measured_area_eval.aftan_c - truth_area_eval.aftan_c),
            },
        },
        "quality_median": float(result.series["quality"].median()),
        "body_mask_attachment_leak_fraction": float(result.series["body_mask_attachment_leak_fraction"].median()),
        "centerline_disagreement_median": float(result.series["centerline_disagreement"].median()),
        "endpoint_jump_p95_px": float(result.series["endpoint_jump_px"].quantile(0.95)),
        "axis_peak_position_stability_p95": float(result.series["axis_peak_position_stability"].quantile(0.95)),
        "formal_qc": formal_qc,
        "diameter_threshold_sweep": diameter_threshold_sweep,
        "diameter_threshold_sensitivity": diameter_threshold_sensitivity,
        "error_summary": error_summary,
    }
    (out_dir / "analysis_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    preview_dir = out_dir / "qc_frames"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_frames = np.linspace(0, len(truth) - 1, 5, dtype=int)
    for frame_idx in preview_frames:
        frame = _load_frame(out_dir / "synthetic.mp4", int(frame_idx))
        overlay = _make_overlay(frame, extraction_cfg)
        cv2.imwrite(str(preview_dir / f"frame_{frame_idx:04d}.png"), overlay)
    print(f"braided demo saved to {out_dir}")


if __name__ == "__main__":
    main()
