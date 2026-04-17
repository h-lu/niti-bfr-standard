from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from niti_bfr.extract import ExtractionConfig, extract_geometry


def _make_extraction_config(cfg: dict) -> ExtractionConfig:
    return ExtractionConfig(
        roi_xyxy=tuple(cfg["roi_xyxy"]),
        blur_ksize=int(cfg["blur_ksize"]),
        threshold_dark=int(cfg["threshold_dark"]),
        open_kernel=int(cfg["open_kernel"]),
        close_kernel=int(cfg["close_kernel"]),
        route_a_tip_cluster_radius_px=float(cfg.get("route_a_tip_cluster_radius_px", 6.0)),
        fit_bin_px=float(cfg.get("fit_bin_px", 3.0)),
        fit_margin_prefer_quadratic=float(cfg.get("fit_margin_prefer_quadratic", 0.05)),
        anchor_prior_xy=tuple(cfg["anchor_prior_xy"]) if cfg.get("anchor_prior_xy") is not None else None,
        anchor_prior_weight=float(cfg.get("anchor_prior_weight", 0.0)),
    )


def _draw_mask_inset(canvas: np.ndarray, mask: np.ndarray, title: str, origin_xy: tuple[int, int]) -> None:
    ox, oy = origin_xy
    inset_w = 180
    scale = inset_w / max(mask.shape[1], 1)
    inset_h = max(int(round(mask.shape[0] * scale)), 1)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_small = cv2.resize(mask_rgb, (inset_w, inset_h), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(canvas, (ox - 2, oy - 22), (ox + inset_w + 2, oy + inset_h + 2), (245, 245, 245), -1)
    cv2.rectangle(canvas, (ox - 2, oy - 22), (ox + inset_w + 2, oy + inset_h + 2), (120, 120, 120), 1)
    cv2.putText(canvas, title, (ox, oy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    canvas[oy : oy + inset_h, ox : ox + inset_w] = mask_small


def _draw_route_a_overlay(frame_bgr: np.ndarray, geom, truth_row: pd.Series, frame_idx: int) -> np.ndarray:
    overlay = frame_bgr.copy()
    x0, y0, x1, y1 = truth_row["roi_xyxy"]
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (80, 180, 255), 2)
    contour = np.round(geom.contour_xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [contour], isClosed=True, color=(110, 110, 110), thickness=1, lineType=cv2.LINE_AA)
    cv2.line(
        overlay,
        np.round(geom.route_a_anchor_xy).astype(int),
        np.round(geom.route_a_tip_xy).astype(int),
        (220, 0, 220),
        2,
        cv2.LINE_AA,
    )
    cv2.circle(overlay, np.round(geom.route_a_anchor_xy).astype(int), 5, (255, 0, 255), -1)
    cv2.circle(overlay, np.round(geom.route_a_tip_xy).astype(int), 5, (180, 0, 255), -1)
    cv2.circle(
        overlay,
        (int(round(float(truth_row["anchor_x"]))), int(round(float(truth_row["anchor_y"])))),
        4,
        (0, 180, 0),
        -1,
    )
    cv2.circle(
        overlay,
        (int(round(float(truth_row["tip_x"]))), int(round(float(truth_row["tip_y"])))),
        4,
        (0, 90, 220),
        -1,
    )

    _draw_mask_inset(overlay, geom.mask, "ROI mask", (18, 48))
    lines = [
        "Demo process: Route A (endpoint/chord)",
        f"frame={frame_idx:03d}  T={truth_row['temperature_c']:.2f} C",
        "magenta line: route A chord",
        "magenta dots: route A anchor/tip",
        "green+blue dots: synthetic truth anchor/tip",
        f"x_route_a={geom.x_route_a_px:.1f}px",
        f"x_true={truth_row['x_true_px']:.1f}px",
        f"instant error={geom.x_route_a_px - truth_row['x_true_px']:+.1f}px",
    ]
    _draw_text_block(overlay, lines, origin_xy=(18, 300))
    return overlay


def _draw_route_b_overlay(frame_bgr: np.ndarray, geom, truth_row: pd.Series, frame_idx: int) -> np.ndarray:
    overlay = frame_bgr.copy()
    x0, y0, x1, y1 = truth_row["roi_xyxy"]
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (80, 180, 255), 2)
    contour = np.round(geom.contour_xy).astype(np.int32).reshape(-1, 1, 2)
    skeleton_path = np.round(geom.sampled_centerline_xy).astype(np.int32).reshape(-1, 1, 2)
    fit_samples = np.round(geom.fit_samples_xy).astype(np.int32).reshape(-1, 1, 2)
    fit_curve = np.round(geom.fitted_curve_xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [contour], isClosed=True, color=(120, 120, 120), thickness=1, lineType=cv2.LINE_AA)
    cv2.polylines(overlay, [skeleton_path], isClosed=False, color=(0, 140, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.polylines(overlay, [fit_samples], isClosed=False, color=(0, 220, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.polylines(overlay, [fit_curve], isClosed=False, color=(255, 180, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(overlay, np.round(geom.anchor_xy).astype(int), 6, (0, 220, 0), -1)
    cv2.circle(overlay, np.round(geom.tip_xy).astype(int), 6, (0, 0, 255), -1)
    cv2.circle(
        overlay,
        (int(round(float(truth_row["anchor_x"]))), int(round(float(truth_row["anchor_y"])))),
        4,
        (0, 180, 0),
        -1,
    )
    cv2.circle(
        overlay,
        (int(round(float(truth_row["tip_x"]))), int(round(float(truth_row["tip_y"])))),
        4,
        (0, 90, 220),
        -1,
    )

    _draw_mask_inset(overlay, geom.mask, "ROI mask", (18, 48))
    lines = [
        "Demo process: Route B (skeleton + local bend fit)",
        f"frame={frame_idx:03d}  T={truth_row['temperature_c']:.2f} C",
        "orange: skeleton main path used for x_fit",
        "cyan: local fit samples",
        "blue: local fitted curve used for kappa_fit",
        "green/red: route B anchor/tip",
        f"x_fit={geom.x_fit_px:.1f}px  x_true={truth_row['x_true_px']:.1f}px",
        f"kappa_fit={geom.kappa_fit_px_inv:.5f}  kappa_true={truth_row['kappa_true_px_inv']:.5f}",
        f"model={geom.model_name}  q={geom.quality:.2f}",
    ]
    _draw_text_block(overlay, lines, origin_xy=(18, 300))
    return overlay


def _draw_text_block(canvas: np.ndarray, lines: list[str], origin_xy: tuple[int, int]) -> None:
    ox, oy = origin_xy
    line_h = 24
    width = 520
    height = line_h * len(lines) + 16
    cv2.rectangle(canvas, (ox - 8, oy - 24), (ox + width, oy - 24 + height), (245, 245, 245), -1)
    cv2.rectangle(canvas, (ox - 8, oy - 24), (ox + width, oy - 24 + height), (120, 120, 120), 1)
    for idx, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (ox, oy + idx * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )


def _draw_combined_overlay(frame_bgr: np.ndarray, route_a_frame: np.ndarray, route_b_frame: np.ndarray) -> np.ndarray:
    margin = 12
    h, w = frame_bgr.shape[:2]
    scale = 0.48
    small_a = cv2.resize(route_a_frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    small_b = cv2.resize(route_b_frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    canvas = frame_bgr.copy()
    y1 = h - small_a.shape[0] - margin
    x1 = margin
    y2 = h - small_b.shape[0] - margin
    x2 = w - small_b.shape[1] - margin
    canvas[y1 : y1 + small_a.shape[0], x1 : x1 + small_a.shape[1]] = small_a
    canvas[y2 : y2 + small_b.shape[0], x2 : x2 + small_b.shape[1]] = small_b
    cv2.rectangle(canvas, (x1 - 2, y1 - 2), (x1 + small_a.shape[1] + 2, y1 + small_a.shape[0] + 2), (255, 255, 255), 2)
    cv2.rectangle(canvas, (x2 - 2, y2 - 2), (x2 + small_b.shape[1] + 2, y2 + small_b.shape[0] + 2), (255, 255, 255), 2)
    return canvas


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/minimal.yaml").read_text(encoding="utf-8"))
    extraction_cfg = _make_extraction_config(config["analysis"]["demo_extraction"])
    out_dir = ROOT / "outputs/demo"
    video_path = out_dir / "synthetic.mp4"
    truth_path = out_dir / "temperature_truth.csv"
    if not video_path.exists() or not truth_path.exists():
        raise RuntimeError("demo files missing, run scripts/run_demo.py first")

    truth = pd.read_csv(truth_path)
    x0, y0, x1, y1 = extraction_cfg.roi_xyxy
    truth["roi_xyxy"] = [(x0, y0, x1, y1)] * len(truth)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_a = cv2.VideoWriter(str(out_dir / "route_a_process.mp4"), fourcc, fps, (width, height))
    writer_b = cv2.VideoWriter(str(out_dir / "route_b_process.mp4"), fourcc, fps, (width, height))
    writer_ab = cv2.VideoWriter(str(out_dir / "route_ab_process.mp4"), fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        geom = extract_geometry(frame, extraction_cfg)
        truth_row = truth.iloc[frame_idx]
        frame_a = _draw_route_a_overlay(frame, geom, truth_row, frame_idx)
        frame_b = _draw_route_b_overlay(frame, geom, truth_row, frame_idx)
        frame_ab = _draw_combined_overlay(frame, frame_a, frame_b)
        writer_a.write(frame_a)
        writer_b.write(frame_b)
        writer_ab.write(frame_ab)
        frame_idx += 1

    cap.release()
    writer_a.release()
    writer_b.release()
    writer_ab.release()
    print(f"wrote process videos to {out_dir}")


if __name__ == "__main__":
    main()
