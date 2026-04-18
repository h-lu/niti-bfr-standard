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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quicklook analysis for braided-device-like videos.")
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
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
        qc_max_segments=int(extraction_params.get("qc_max_segments", 15)),
    )

    video_path = args.video.resolve()
    out_dir = args.output_dir or (ROOT / "outputs" / "braided_like" / video_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = analyze_braided_video_quicklook(video_path, extraction_cfg)
    result.series["length_env_smooth"] = result.series["length_env_px"].rolling(window=11, center=True, min_periods=1).median()
    result.series["length_axis_smooth"] = result.series["length_axis_px"].rolling(window=11, center=True, min_periods=1).median()
    result.series["diameter_max_smooth"] = result.series["diameter_max_px"].rolling(window=11, center=True, min_periods=1).median()
    result.series.to_csv(out_dir / "quicklook.csv", index=False)

    plt.figure(figsize=(8, 4.8))
    plt.plot(result.series["time_sec"], result.series["length_env_px"], linewidth=1.0, alpha=0.25, label="envelope raw")
    plt.plot(result.series["time_sec"], result.series["length_env_smooth"], linewidth=2.0, label="envelope median smooth")
    plt.plot(result.series["time_sec"], result.series["length_axis_px"], linewidth=1.0, alpha=0.25, label="axis raw")
    plt.plot(result.series["time_sec"], result.series["length_axis_smooth"], linewidth=2.0, label="axis median smooth")
    plt.xlabel("Time (s)")
    plt.ylabel("Length (px)")
    plt.title(f"{video_path.name} braided quicklook lengths")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "lengths_vs_time.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.plot(result.series["time_sec"], result.series["diameter_max_px"], linewidth=1.0, alpha=0.25, label="Dmax raw")
    plt.plot(result.series["time_sec"], result.series["diameter_max_smooth"], linewidth=2.0, label="Dmax median smooth")
    plt.xlabel("Time (s)")
    plt.ylabel("Diameter (px)")
    plt.title(f"{video_path.name} braided quicklook max diameter")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "diameter_vs_time.png", dpi=160)
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
        "frames": frame_count,
        "length_env_median_px": float(result.series["length_env_px"].median()),
        "length_axis_median_px": float(result.series["length_axis_px"].median()),
        "diameter_max_median_px": float(result.series["diameter_max_px"].median()),
        "x_peak_norm_median": float(result.series["x_peak_norm"].median()),
        "taper_left_median_px": float(result.series["taper_left_px"].median()),
        "taper_right_median_px": float(result.series["taper_right_px"].median()),
        "quality_median": float(result.series["quality"].median()),
        "quality_lt_0_5_fraction": float((result.series["quality"] < 0.5).mean()),
    }
    (out_dir / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    print(f"quicklook saved to {out_dir}")


if __name__ == "__main__":
    main()
