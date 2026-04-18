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
from niti_bfr.temporal import RouteCConfig, apply_route_c


def _make_extraction_config(cfg: dict) -> ExtractionConfig:
    return ExtractionConfig(
        roi_xyxy=tuple(cfg["roi_xyxy"]),
        blur_ksize=int(cfg["blur_ksize"]),
        threshold_dark=int(cfg["threshold_dark"]),
        open_kernel=int(cfg["open_kernel"]),
        close_kernel=int(cfg["close_kernel"]),
        route_a_tip_cluster_radius_px=float(cfg.get("route_a_tip_cluster_radius_px", 6.0)),
        route_b_endpoint_extension_scale=float(cfg.get("route_b_endpoint_extension_scale", 0.0)),
        route_b_cap_inset_scale=float(cfg.get("route_b_cap_inset_scale", 0.08)),
        fit_bin_px=float(cfg.get("fit_bin_px", 3.0)),
        fit_path_fraction=float(cfg.get("fit_path_fraction", 0.72)),
        fit_path_fraction_min=float(cfg.get("fit_path_fraction_min", 0.42)),
        fit_curvature_threshold_ratio=float(cfg.get("fit_curvature_threshold_ratio", 0.28)),
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


def _draw_text_block(canvas: np.ndarray, lines: list[str], origin_xy: tuple[int, int]) -> None:
    ox, oy = origin_xy
    line_h = 24
    width = 560
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


def _draw_text_block_bottom_left(canvas: np.ndarray, lines: list[str]) -> None:
    line_h = 24
    margin = 18
    width = 560
    height = line_h * len(lines) + 16
    origin_xy = (margin, canvas.shape[0] - margin - height + 40)
    _draw_text_block(canvas, lines, origin_xy=origin_xy)


def _route_a_overlay(frame_bgr: np.ndarray, geom, row: pd.Series, frame_idx: int, roi_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    overlay = frame_bgr.copy()
    x0, y0, x1, y1 = roi_xyxy
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (80, 180, 255), 2)
    contour = np.round(geom.contour_xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [contour], isClosed=True, color=(120, 120, 120), thickness=1, lineType=cv2.LINE_AA)
    cv2.line(
        overlay,
        np.round(geom.route_a_anchor_xy).astype(int),
        np.round(geom.route_a_tip_xy).astype(int),
        (220, 0, 220),
        2,
        cv2.LINE_AA,
    )
    cv2.circle(overlay, np.round(geom.route_a_anchor_xy).astype(int), 5, (255, 0, 255), -1)
    cv2.circle(overlay, np.round(geom.route_a_tip_xy).astype(int), 6, (180, 0, 255), -1)
    _draw_mask_inset(overlay, geom.mask, "ROI mask", (18, 48))
    lines = [
        "wire-like process: Route A (endpoint/chord baseline)",
        f"frame={frame_idx:03d}  time={row['time_sec']:.2f}s",
        "magenta: route A chord",
        "pink dots: route A anchor / tip",
        f"x_route_a={row['x_route_a_px']:.1f}px",
        f"quality={row['quality']:.2f}",
    ]
    _draw_text_block_bottom_left(overlay, lines)
    return overlay


def _route_b_overlay(frame_bgr: np.ndarray, geom, row: pd.Series, frame_idx: int, roi_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    overlay = frame_bgr.copy()
    x0, y0, x1, y1 = roi_xyxy
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (80, 180, 255), 2)
    contour = np.round(geom.contour_xy).astype(np.int32).reshape(-1, 1, 2)
    skeleton_path = np.round(geom.sampled_centerline_xy).astype(np.int32).reshape(-1, 1, 2)
    fit_samples = np.round(geom.fit_samples_xy).astype(np.int32).reshape(-1, 1, 2)
    fit_curve = np.round(geom.fitted_curve_xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [contour], isClosed=True, color=(120, 120, 120), thickness=1, lineType=cv2.LINE_AA)
    cv2.polylines(overlay, [skeleton_path], isClosed=False, color=(0, 140, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.polylines(overlay, [fit_samples], isClosed=False, color=(0, 220, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.polylines(overlay, [fit_curve], isClosed=False, color=(255, 180, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(overlay, np.round(geom.anchor_xy).astype(int), 5, (0, 220, 0), -1)
    cv2.circle(overlay, np.round(geom.tip_xy).astype(int), 6, (0, 0, 255), -1)
    _draw_mask_inset(overlay, geom.mask, "ROI mask", (18, 48))
    lines = [
        "wire-like process: Route B (single-frame shape fit)",
        f"frame={frame_idx:03d}  time={row['time_sec']:.2f}s",
        "orange: route B skeleton path",
        "cyan: fit samples, blue: fitted curve",
        "green/red: route B anchor / tip",
        f"x_fit={row['x_fit_px']:.1f}px",
        f"kappa_fit={row['kappa_fit_px_inv']:.5f}",
        f"model={row['model_name']}  quality={row['quality']:.2f}",
    ]
    _draw_text_block_bottom_left(overlay, lines)
    return overlay


def _route_c_overlay(frame_bgr: np.ndarray, geom, row: pd.Series, frame_idx: int, roi_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    overlay = frame_bgr.copy()
    x0, y0, x1, y1 = roi_xyxy
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (80, 180, 255), 2)
    contour = np.round(geom.contour_xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [contour], isClosed=True, color=(120, 120, 120), thickness=1, lineType=cv2.LINE_AA)
    route_c_anchor = np.array([row["route_c_anchor_x"], row["route_c_anchor_y"]], dtype=float)
    route_c_tip = np.array([row["route_c_tip_x"], row["route_c_tip_y"]], dtype=float)
    cv2.line(
        overlay,
        np.round(route_c_anchor).astype(int),
        np.round(route_c_tip).astype(int),
        (60, 200, 60),
        2,
        cv2.LINE_AA,
    )
    cv2.circle(overlay, np.round(route_c_anchor).astype(int), 5, (60, 200, 60), -1)
    cv2.circle(overlay, np.round(route_c_tip).astype(int), 6, (40, 120, 255), -1)
    cv2.circle(overlay, np.round(geom.route_a_tip_xy).astype(int), 4, (255, 0, 255), -1)
    cv2.circle(overlay, np.round(geom.tip_xy).astype(int), 4, (0, 0, 255), -1)
    _draw_mask_inset(overlay, geom.mask, "ROI mask", (18, 48))
    lines = [
        "wire-like process: Route C (temporal enhancement)",
        f"frame={frame_idx:03d}  time={row['time_sec']:.2f}s",
        "green line: route C temporally regularized chord",
        "orange dot: route C tip",
        "small pink/red dots: route A / B single-frame tips",
        f"x_route_c={row['x_route_c_px']:.1f}px",
        f"kappa_route_c={row['kappa_route_c_px_inv']:.5f}",
    ]
    _draw_text_block_bottom_left(overlay, lines)
    return overlay


def _combined_overlay(frame_bgr: np.ndarray, route_a_frame: np.ndarray, route_b_frame: np.ndarray, route_c_frame: np.ndarray) -> np.ndarray:
    margin = 10
    h, w = frame_bgr.shape[:2]
    scale = 0.32
    small_a = cv2.resize(route_a_frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    small_b = cv2.resize(route_b_frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    small_c = cv2.resize(route_c_frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    canvas = frame_bgr.copy()
    y = h - small_a.shape[0] - margin
    x_positions = [margin, (w - small_b.shape[1]) // 2, w - small_c.shape[1] - margin]
    for x, panel in zip(x_positions, [small_a, small_b, small_c]):
        canvas[y : y + panel.shape[0], x : x + panel.shape[1]] = panel
        cv2.rectangle(canvas, (x - 2, y - 2), (x + panel.shape[1] + 2, y + panel.shape[0] + 2), (255, 255, 255), 2)
    return canvas


def main() -> None:
    video_path = ROOT / "data/wire-like.mp4"
    config = yaml.safe_load((ROOT / "configs/minimal.yaml").read_text(encoding="utf-8"))
    extraction_params = config["analysis"]["wire_like_extraction"]
    extraction_cfg = _make_extraction_config(extraction_params)
    route_c_cfg = RouteCConfig(**extraction_params.get("route_c", {}))
    out_dir = ROOT / "outputs/wire_like"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rows = []
    raw_frames: list[np.ndarray] = []
    geoms = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        geom = extract_geometry(frame, extraction_cfg)
        raw_frames.append(frame.copy())
        geoms.append(geom)
        rows.append(
            {
                "frame": frame_idx,
                "time_sec": frame_idx / fps,
                "route_a_anchor_x": geom.route_a_anchor_xy[0],
                "route_a_anchor_y": geom.route_a_anchor_xy[1],
                "route_a_tip_x": geom.route_a_tip_xy[0],
                "route_a_tip_y": geom.route_a_tip_xy[1],
                "x_route_a_px": geom.x_route_a_px,
                "anchor_x": geom.anchor_xy[0],
                "anchor_y": geom.anchor_xy[1],
                "tip_x": geom.tip_xy[0],
                "tip_y": geom.tip_xy[1],
                "x_fit_px": geom.x_fit_px,
                "kappa_fit_px_inv": geom.kappa_fit_px_inv,
                "quality": geom.quality,
                "model_name": geom.model_name,
            }
        )
        frame_idx += 1
    cap.release()

    series = pd.DataFrame(rows)
    series = apply_route_c(series, route_c_cfg)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_a = cv2.VideoWriter(str(out_dir / "route_a_process.mp4"), fourcc, fps, (width, height))
    writer_b = cv2.VideoWriter(str(out_dir / "route_b_process.mp4"), fourcc, fps, (width, height))
    writer_c = cv2.VideoWriter(str(out_dir / "route_c_process.mp4"), fourcc, fps, (width, height))
    writer_abc = cv2.VideoWriter(str(out_dir / "route_abc_process.mp4"), fourcc, fps, (width, height))

    roi_xyxy = extraction_cfg.roi_xyxy
    for frame_idx, (frame, geom) in enumerate(zip(raw_frames, geoms)):
        row = series.iloc[frame_idx]
        frame_a = _route_a_overlay(frame, geom, row, frame_idx, roi_xyxy)
        frame_b = _route_b_overlay(frame, geom, row, frame_idx, roi_xyxy)
        frame_c = _route_c_overlay(frame, geom, row, frame_idx, roi_xyxy)
        frame_abc = _combined_overlay(frame, frame_a, frame_b, frame_c)
        writer_a.write(frame_a)
        writer_b.write(frame_b)
        writer_c.write(frame_c)
        writer_abc.write(frame_abc)

    writer_a.release()
    writer_b.release()
    writer_c.release()
    writer_abc.release()
    print(f"wrote wire-like process videos to {out_dir}")


if __name__ == "__main__":
    main()
