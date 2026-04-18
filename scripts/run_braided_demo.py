from __future__ import annotations

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
from niti_bfr.pipeline import analyze_braided_video_quicklook
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
    cv2.polylines(overlay, [contour], isClosed=True, color=(120, 120, 120), thickness=1, lineType=cv2.LINE_AA)
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
    label = (
        f"env={geom.length_env_px:.1f}px axis={geom.length_axis_px:.1f}px "
        f"Dmax={geom.diameter_max_px:.1f}px q={geom.quality:.2f}"
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
            profile_power=float(synth_cfg["profile_power"]),
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
        qc_max_segments=int(extraction_params.get("qc_max_segments", 15)),
    )

    out_dir = ROOT / "outputs" / "braided_demo"
    truth = write_braided_synthetic_dataset(
        output_dir=out_dir,
        model=model,
        render_config=render_cfg,
        schedule=schedule,
        taper_threshold_ratio=extraction_cfg.taper_threshold_ratio,
    )
    result = analyze_braided_video_quicklook(out_dir / "synthetic.mp4", extraction=extraction_cfg)
    result.series["temperature_c"] = truth["temperature_c"]
    result.series.to_csv(out_dir / "analysis.csv", index=False)

    plt.figure(figsize=(8, 4.8))
    plt.plot(truth["temperature_c"], truth["length_true_px"], label="length true", linewidth=2)
    plt.plot(result.series["temperature_c"], result.series["length_env_px"], label="envelope length", alpha=0.8)
    plt.plot(result.series["temperature_c"], result.series["length_axis_px"], label="axis length", alpha=0.8)
    plt.xlabel("Temperature (C)")
    plt.ylabel("Length (px)")
    plt.title("Braided demo length comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "length_vs_temperature.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.plot(truth["temperature_c"], truth["diameter_true_px"], label="diameter true", linewidth=2)
    plt.plot(result.series["temperature_c"], result.series["diameter_max_px"], label="diameter measured", alpha=0.85)
    plt.xlabel("Temperature (C)")
    plt.ylabel("Diameter (px)")
    plt.title("Braided demo diameter comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "diameter_vs_temperature.png", dpi=160)
    plt.close()

    error_summary = {
        "length_env_mae_px": _mean_abs_error(result.series["length_env_px"], truth["length_true_px"]),
        "length_axis_mae_px": _mean_abs_error(result.series["length_axis_px"], truth["length_true_px"]),
        "diameter_max_mae_px": _mean_abs_error(result.series["diameter_max_px"], truth["diameter_true_px"]),
        "x_peak_norm_mae": _mean_abs_error(result.series["x_peak_norm"], truth["x_peak_norm_true"]),
        "taper_left_mae_px": _mean_abs_error(result.series["taper_left_px"], truth["taper_left_true_px"]),
        "taper_right_mae_px": _mean_abs_error(result.series["taper_right_px"], truth["taper_right_true_px"]),
    }
    summary = {
        "demo_output": str(out_dir),
        "truth_frames": int(len(truth)),
        "quality_median": float(result.series["quality"].median()),
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
