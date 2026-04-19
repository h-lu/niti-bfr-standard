from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from niti_bfr.extract_braided import BraidedExtractionConfig, extract_braided_geometry
from niti_bfr.pipeline import analyze_braided_video_quicklook


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
        f"Cproj={geom.area_proj_px2:.0f}px2 leak={geom.body_mask_attachment_leak_fraction:.2f}"
    )
    cv2.putText(overlay, label, (x0 + 8, max(24, y0 + 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
    return overlay


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quicklook analysis for braided-device-like videos.")
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--temperature-csv", type=Path, default=None, help="Optional synced temperature CSV")
    parser.add_argument(
        "--temperature-time-offset-sec",
        type=float,
        default=0.0,
        help="Optional fixed offset applied to temperature time_sec before interpolation",
    )
    parser.add_argument(
        "--config-key",
        default="braided_device_extraction",
        help="Configuration key under analysis in configs/minimal.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = yaml.safe_load((ROOT / "configs/minimal.yaml").read_text(encoding="utf-8"))
    extraction_params = config["analysis"][args.config_key]
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
        diameter_proxy_window_start_norm=float(extraction_params.get("diameter_proxy_window_start_norm", 0.6)),
        diameter_proxy_window_end_norm=float(extraction_params.get("diameter_proxy_window_end_norm", 0.8)),
    )

    video_path = args.video.resolve()
    out_dir = args.output_dir or (ROOT / "outputs" / "braided_like" / video_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    temperature_csv = args.temperature_csv.resolve() if args.temperature_csv else None
    result = analyze_braided_video_quicklook(
        video_path,
        extraction_cfg,
        temperature_csv=temperature_csv,
        temperature_time_offset_sec=float(args.temperature_time_offset_sec),
    )
    result.series["length_env_smooth"] = result.series["length_env_px"].rolling(window=11, center=True, min_periods=1).median()
    result.series["length_axis_smooth"] = result.series["length_axis_px"].rolling(window=11, center=True, min_periods=1).median()
    result.series["diameter_max_smooth"] = result.series["diameter_max_px"].rolling(window=11, center=True, min_periods=1).median()
    if "diameter_mid_median_px" in result.series.columns:
        result.series["diameter_mid_median_smooth"] = result.series["diameter_mid_median_px"].rolling(window=11, center=True, min_periods=1).median()
    result.series["area_proj_smooth_px2"] = result.series["area_proj_px2"].rolling(window=11, center=True, min_periods=1).median()
    result.series["compaction_zone_smooth"] = result.series["compaction_zone_length_px"].rolling(window=11, center=True, min_periods=1).median()
    result.series["body_mask_area_smooth_px2"] = result.series["body_mask_area_px2"].rolling(window=11, center=True, min_periods=1).median()
    result.series["axis_definition_gap_smooth_px"] = result.series["length_axis_disagreement_px"].rolling(window=11, center=True, min_periods=1).median()
    result.series.to_csv(out_dir / "quicklook.csv", index=False)

    plt.figure(figsize=(8, 4.8))
    plt.plot(result.series["time_sec"], result.series["length_env_px"], linewidth=1.0, alpha=0.25, label="envelope raw")
    plt.plot(result.series["time_sec"], result.series["length_env_smooth"], linewidth=2.0, label="envelope median smooth")
    plt.plot(result.series["time_sec"], result.series["length_axis_px"], linewidth=1.0, alpha=0.25, label="A raw")
    plt.plot(result.series["time_sec"], result.series["length_axis_smooth"], linewidth=2.0, label="A smooth")
    plt.xlabel("Time (s)")
    plt.ylabel("Length (px)")
    plt.title(f"{video_path.name} braided quicklook lengths")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "lengths_vs_time.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.plot(result.series["time_sec"], result.series["diameter_max_px"], linewidth=1.0, alpha=0.25, label="B raw")
    plt.plot(result.series["time_sec"], result.series["diameter_max_smooth"], linewidth=2.0, label="B smooth")
    if "diameter_mid_median_smooth" in result.series.columns:
        plt.plot(result.series["time_sec"], result.series["diameter_mid_median_smooth"], linewidth=1.7, label="mid-window median smooth")
    plt.xlabel("Time (s)")
    plt.ylabel("Diameter (px)")
    plt.title(f"{video_path.name} braided quicklook D_max")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "diameter_vs_time.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.plot(result.series["time_sec"], result.series["area_proj_px2"], linewidth=1.0, alpha=0.25, label="C raw")
    plt.plot(result.series["time_sec"], result.series["area_proj_smooth_px2"], linewidth=2.0, label="C smooth")
    plt.xlabel("Time (s)")
    plt.ylabel("Projected area integral (px^2)")
    plt.title(f"{video_path.name} braided quicklook A_proj")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "area_vs_time.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.plot(result.series["time_sec"], result.series["body_mask_area_px2"], linewidth=1.0, alpha=0.25, label="body mask raw")
    plt.plot(result.series["time_sec"], result.series["body_mask_area_smooth_px2"], linewidth=2.0, label="body mask smooth")
    plt.xlabel("Time (s)")
    plt.ylabel("Body mask area (px^2)")
    plt.title(f"{video_path.name} body-only mask area")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "body_mask_vs_time.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.plot(result.series["time_sec"], result.series["length_axis_disagreement_px"], linewidth=1.0, alpha=0.25, label="axis gap raw")
    plt.plot(result.series["time_sec"], result.series["axis_definition_gap_smooth_px"], linewidth=2.0, label="axis gap smooth")
    plt.xlabel("Time (s)")
    plt.ylabel("Definition gap (px)")
    plt.title(f"{video_path.name} axis definition gap")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "axis_definition_gap_vs_time.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.plot(result.series["time_sec"], result.series["foreshortening_axis"], linewidth=2.0, label="FS axis")
    plt.plot(result.series["time_sec"], result.series["foreshortening_env"], linewidth=2.0, label="FS envelope")
    plt.xlabel("Time (s)")
    plt.ylabel("Foreshortening")
    plt.title(f"{video_path.name} braided quicklook foreshortening")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "foreshortening_vs_time.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.plot(result.series["time_sec"], result.series["landing_zone_left_px"], linewidth=1.5, label="landing left")
    plt.plot(result.series["time_sec"], result.series["transition_zone_left_px"], linewidth=1.5, label="transition left")
    plt.plot(result.series["time_sec"], result.series["compaction_zone_length_px"], linewidth=1.0, alpha=0.25, label="compaction raw")
    plt.plot(result.series["time_sec"], result.series["compaction_zone_smooth"], linewidth=2.0, label="compaction smooth")
    plt.plot(result.series["time_sec"], result.series["transition_zone_right_px"], linewidth=1.5, label="transition right")
    plt.plot(result.series["time_sec"], result.series["landing_zone_right_px"], linewidth=1.5, label="landing right")
    plt.xlabel("Time (s)")
    plt.ylabel("Length (px)")
    plt.title(f"{video_path.name} braided quicklook zones")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "zones_vs_time.png", dpi=160)
    plt.close()

    if "temperature_c" in result.series.columns and result.series["temperature_c"].notna().any():
        plt.figure(figsize=(8, 4.8))
        plt.plot(result.series["temperature_c"], result.series["length_env_px"], linewidth=1.4, label="envelope length")
        plt.plot(result.series["temperature_c"], result.series["length_axis_px"], linewidth=1.8, label="A")
        plt.xlabel("Temperature (C)")
        plt.ylabel("Length (px)")
        plt.title(f"{video_path.name} braided length over temperature")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "lengths_vs_temperature.png", dpi=160)
        plt.close()

        if {"length_axis_recovery", "length_env_recovery", "diameter_max_recovery", "area_proj_recovery"}.issubset(result.series.columns):
            plt.figure(figsize=(8, 4.8))
            plt.plot(result.series["temperature_c"], result.series["length_axis_recovery"], linewidth=2.0, label="A recovery")
            plt.plot(result.series["temperature_c"], result.series["length_env_recovery"], linewidth=1.7, label="envelope recovery")
            plt.plot(result.series["temperature_c"], result.series["diameter_max_recovery"], linewidth=1.7, label="B recovery")
            plt.plot(result.series["temperature_c"], result.series["area_proj_recovery"], linewidth=1.7, label="C recovery")
            plt.xlabel("Temperature (C)")
            plt.ylabel("Recovery ratio")
            plt.title(f"{video_path.name} braided recovery over temperature")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "recovery_vs_temperature.png", dpi=160)
            plt.close()

        plt.figure(figsize=(8, 4.8))
        plt.plot(result.series["temperature_c"], result.series["length_axis_px"], linewidth=1.6, label="A")
        plt.plot(result.series["temperature_c"], result.series["diameter_max_px"], linewidth=1.6, label="B")
        plt.plot(result.series["temperature_c"], result.series["length_env_px"], linewidth=1.4, label="env")
        plt.xlabel("Temperature (C)")
        plt.ylabel("Length / diameter (px)")
        plt.title(f"{video_path.name} braided A/B over temperature")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "core_metrics_vs_temperature.png", dpi=160)
        plt.close()

        plt.figure(figsize=(8, 4.8))
        plt.plot(result.series["temperature_c"], result.series["body_mask_area_px2"], linewidth=1.8, label="body mask area")
        plt.plot(result.series["temperature_c"], result.series["length_axis_disagreement_px"], linewidth=1.7, label="axis definition gap")
        plt.plot(result.series["temperature_c"], result.series["diameter_peak_pos_norm"], linewidth=1.7, label="peak pos norm")
        plt.xlabel("Temperature (C)")
        plt.ylabel("Body/QC metrics")
        plt.title(f"{video_path.name} body-only QC over temperature")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "body_qc_vs_temperature.png", dpi=160)
        plt.close()

    frame_count = int(result.series["frame"].max()) + 1
    preview_frames = np.linspace(0, frame_count - 1, min(frame_count, 5), dtype=int)
    preview_dir = out_dir / "qc_frames"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx in preview_frames:
        frame = _load_frame(video_path, int(frame_idx))
        overlay = _make_overlay(frame, extraction_cfg)
        cv2.imwrite(str(preview_dir / f"frame_{frame_idx:04d}.png"), overlay)

    summary = {
        "metric_aliases": {
            "A": "length_axis",
            "B": "diameter_max",
            "C": "area_proj",
        },
        "frames": frame_count,
        "length_env_median_px": float(result.series["length_env_px"].median()),
        "length_axis_median_px": float(result.series["length_axis_px"].median()),
        "diameter_max_median_px": float(result.series["diameter_max_px"].median()),
        "diameter_p95_median_px": float(result.series["diameter_p95_px"].median()),
        "diameter_mid_median_px": float(result.series["diameter_mid_median_px"].median()),
        "diameter_mid_p90_median_px": float(result.series["diameter_mid_p90_px"].median()),
        "area_proj_median_px2": float(result.series["area_proj_px2"].median()),
        "body_mask_area_median_px2": float(result.series["body_mask_area_px2"].median()),
        "length_axis_definition_gap_median_px": float(result.series["length_axis_disagreement_px"].median()),
        "body_mask_attachment_leak_fraction_median": float(result.series["body_mask_attachment_leak_fraction"].median()),
        "x_peak_norm_median": float(result.series["x_peak_norm"].median()),
        "diameter_peak_pos_norm_median": float(result.series["diameter_peak_pos_norm"].median()),
        "foreshortening_axis_max": float(result.series["foreshortening_axis"].max()),
        "foreshortening_env_max": float(result.series["foreshortening_env"].max()),
        "landing_zone_left_median_px": float(result.series["landing_zone_left_px"].median()),
        "landing_zone_right_median_px": float(result.series["landing_zone_right_px"].median()),
        "transition_zone_left_median_px": float(result.series["transition_zone_left_px"].median()),
        "transition_zone_right_median_px": float(result.series["transition_zone_right_px"].median()),
        "compaction_zone_length_median_px": float(result.series["compaction_zone_length_px"].median()),
        "zone_symmetry_median": float(result.series["zone_symmetry"].median()),
        "taper_left_median_px": float(result.series["taper_left_px"].median()),
        "taper_right_median_px": float(result.series["taper_right_px"].median()),
        "quality_median": float(result.series["quality"].median()),
        "quality_lt_0_5_fraction": float((result.series["quality"] < 0.5).mean()),
        "endpoint_jump_p95_px": float(result.series["endpoint_jump_px"].quantile(0.95)),
        "axis_peak_position_stability_p95": float(result.series["axis_peak_position_stability"].quantile(0.95)),
        "mode": result.mode,
        "formal_metric_label": result.formal_metric_label,
        "formal_metric_alias": {"length_axis": "A", "diameter_max": "B", "area_proj": "C"}.get(result.formal_metric_label),
        "primary_metric_label": result.primary_metric_label,
        "primary_metric_alias": {"length_axis": "A", "diameter_max": "B", "area_proj": "C"}.get(result.primary_metric_label),
        "formal_gate_reason": result.formal_gate_reason,
        "af95_c": None if result.af95_c is None else float(result.af95_c),
        "aftan_c": None if result.aftan_c is None else float(result.aftan_c),
    }
    (out_dir / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    print(f"quicklook saved to {out_dir}")


if __name__ == "__main__":
    main()
