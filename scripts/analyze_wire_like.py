from __future__ import annotations

from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from niti_bfr.extract import ExtractionConfig, extract_geometry
from niti_bfr.pipeline import analyze_video
from niti_bfr.temporal import RouteCConfig


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
    skeleton_path = np.round(geom.sampled_centerline_xy).astype(np.int32).reshape(-1, 1, 2)
    fit_samples = np.round(geom.fit_samples_xy).astype(np.int32).reshape(-1, 1, 2)
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
    cv2.polylines(overlay, [skeleton_path], isClosed=False, color=(0, 140, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.polylines(overlay, [fit_samples], isClosed=False, color=(0, 220, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.polylines(overlay, [fit_curve], isClosed=False, color=(255, 180, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(overlay, np.round(geom.route_a_anchor_xy).astype(int), 5, (255, 0, 255), -1)
    cv2.circle(overlay, np.round(geom.route_a_tip_xy).astype(int), 5, (180, 0, 255), -1)
    cv2.circle(overlay, np.round(geom.anchor_xy).astype(int), 6, (0, 220, 0), -1)
    cv2.circle(overlay, np.round(geom.tip_xy).astype(int), 6, (0, 0, 255), -1)
    label = (
        f"A:x={geom.x_route_a_px:.1f}px "
        f"B:{geom.model_name} x={geom.x_fit_px:.1f}px k={geom.kappa_fit_px_inv:.4f} q={geom.quality:.2f}"
    )
    cv2.putText(overlay, label, (x0 + 8, max(24, y0 + 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
    return overlay


def main() -> None:
    video_path = ROOT / "data/wire-like.mp4"
    config = yaml.safe_load((ROOT / "configs/minimal.yaml").read_text(encoding="utf-8"))
    extraction_params = config["analysis"]["wire_like_extraction"]
    extraction_cfg = ExtractionConfig(
        roi_xyxy=tuple(extraction_params["roi_xyxy"]),
        blur_ksize=int(extraction_params["blur_ksize"]),
        threshold_dark=int(extraction_params["threshold_dark"]),
        open_kernel=int(extraction_params["open_kernel"]),
        close_kernel=int(extraction_params["close_kernel"]),
        route_a_tip_cluster_radius_px=float(extraction_params.get("route_a_tip_cluster_radius_px", 6.0)),
        fit_bin_px=float(extraction_params.get("fit_bin_px", 3.0)),
        fit_path_fraction=float(extraction_params.get("fit_path_fraction", 0.72)),
        fit_margin_prefer_quadratic=float(extraction_params.get("fit_margin_prefer_quadratic", 0.05)),
    )
    out_dir = ROOT / "outputs/wire_like"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = analyze_video(
        video_path,
        extraction=extraction_cfg,
        route_c=RouteCConfig(**extraction_params.get("route_c", {})),
    )
    result.series["x_route_a_smooth"] = result.series["x_route_a_px"].rolling(window=11, center=True, min_periods=1).median()
    result.series["x_fit_smooth"] = result.series["x_fit_px"].rolling(window=11, center=True, min_periods=1).median()
    result.series["x_route_c_smooth"] = result.series["x_route_c_px"].rolling(window=11, center=True, min_periods=1).median()
    result.series["kappa_fit_smooth"] = result.series["kappa_fit_px_inv"].rolling(window=11, center=True, min_periods=1).median()
    result.series["kappa_route_c_smooth"] = result.series["kappa_route_c_px_inv"].rolling(window=11, center=True, min_periods=1).median()
    result.series.to_csv(out_dir / "quicklook.csv", index=False)

    plt.figure(figsize=(8, 4.8))
    plt.plot(result.series["time_sec"], result.series["x_route_a_px"], linewidth=1.0, alpha=0.25, label="route A raw")
    plt.plot(result.series["time_sec"], result.series["x_route_a_smooth"], linewidth=2.0, label="route A median smooth")
    plt.plot(result.series["time_sec"], result.series["x_fit_px"], linewidth=1.0, alpha=0.25, label="route B raw")
    plt.plot(result.series["time_sec"], result.series["x_fit_smooth"], linewidth=2.0, label="route B median smooth")
    plt.plot(result.series["time_sec"], result.series["x_route_c_px"], linewidth=1.0, alpha=0.25, label="route C raw")
    plt.plot(result.series["time_sec"], result.series["x_route_c_smooth"], linewidth=2.0, label="route C median smooth")
    plt.xlabel("Time (s)")
    plt.ylabel("x (px)")
    plt.title("wire-like.mp4 route A vs route B x quicklook")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "x_routes_vs_time.png", dpi=160)
    plt.savefig(out_dir / "x_fit_vs_time.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.plot(result.series["time_sec"], result.series["kappa_fit_px_inv"], linewidth=1.0, alpha=0.25, label="route B raw")
    plt.plot(result.series["time_sec"], result.series["kappa_fit_smooth"], linewidth=2.0, label="route B median smooth")
    plt.plot(result.series["time_sec"], result.series["kappa_route_c_px_inv"], linewidth=1.0, alpha=0.25, label="route C raw")
    plt.plot(result.series["time_sec"], result.series["kappa_route_c_smooth"], linewidth=2.0, label="route C median smooth")
    plt.xlabel("Time (s)")
    plt.ylabel("kappa_fit (px^-1)")
    plt.title("wire-like.mp4 kappa_fit quicklook")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "kappa_fit_vs_time.png", dpi=160)
    plt.close()

    frame_count = int(result.series["frame"].max()) + 1
    preview_frames = np.linspace(0, frame_count - 1, 5, dtype=int)
    preview_dir = out_dir / "qc_frames"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx in preview_frames:
        frame = _load_frame(video_path, int(frame_idx))
        overlay = _make_overlay(frame, extraction_cfg)
        cv2.imwrite(str(preview_dir / f"frame_{frame_idx:04d}.png"), overlay)

    summary = {
        "frames": frame_count,
        "x_route_a_median_px": float(result.series["x_route_a_px"].median()),
        "x_route_a_smooth_min_px": float(result.series["x_route_a_smooth"].min()),
        "x_route_a_smooth_max_px": float(result.series["x_route_a_smooth"].max()),
        "x_fit_median_px": float(result.series["x_fit_px"].median()),
        "x_fit_smooth_min_px": float(result.series["x_fit_smooth"].min()),
        "x_fit_smooth_max_px": float(result.series["x_fit_smooth"].max()),
        "x_route_c_median_px": float(result.series["x_route_c_px"].median()),
        "x_route_c_smooth_min_px": float(result.series["x_route_c_smooth"].min()),
        "x_route_c_smooth_max_px": float(result.series["x_route_c_smooth"].max()),
        "kappa_fit_median_px_inv": float(result.series["kappa_fit_px_inv"].median()),
        "kappa_fit_smooth_min_px_inv": float(result.series["kappa_fit_smooth"].min()),
        "kappa_fit_smooth_max_px_inv": float(result.series["kappa_fit_smooth"].max()),
        "kappa_route_c_median_px_inv": float(result.series["kappa_route_c_px_inv"].median()),
        "kappa_route_c_smooth_min_px_inv": float(result.series["kappa_route_c_smooth"].min()),
        "kappa_route_c_smooth_max_px_inv": float(result.series["kappa_route_c_smooth"].max()),
        "quality_median": float(result.series["quality"].median()),
        "quality_lt_0_1_fraction": float((result.series["quality"] < 0.1).mean()),
        "quadratic_fraction": float((result.series["model_name"] == "quadratic").mean()),
    }
    (out_dir / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    print(f"quicklook saved to {out_dir}")


if __name__ == "__main__":
    main()
