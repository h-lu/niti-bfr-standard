from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from niti_bfr.extract_braided import (  # noqa: E402
    BraidedExtractionConfig,
    _orthogonal_widths,
    extract_braided_geometry,
)
from niti_bfr.pipeline import analyze_braided_video_quicklook  # noqa: E402

SIDEBAR_WIDTH = 430


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render braided A/B/C debug overlay video.")
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument("--output", type=Path, required=True, help="Output mp4 path")
    parser.add_argument("--temperature-csv", type=Path, default=None, help="Optional synced temperature CSV")
    parser.add_argument("--quicklook-csv", type=Path, default=None, help="Optional precomputed quicklook.csv to reuse")
    parser.add_argument(
        "--config-key",
        default="braided_device_extraction",
        help="Configuration key under analysis in configs/minimal.yaml",
    )
    return parser.parse_args()


def _make_extraction_config(cfg: dict) -> BraidedExtractionConfig:
    return BraidedExtractionConfig(
        roi_xyxy=tuple(cfg["roi_xyxy"]),
        blur_ksize=int(cfg["blur_ksize"]),
        threshold_dark=int(cfg["threshold_dark"]),
        open_kernel=int(cfg["open_kernel"]),
        close_kernel=int(cfg["close_kernel"]),
        min_component_area=int(cfg.get("min_component_area", 200)),
        width_sampling_step_px=float(cfg.get("width_sampling_step_px", 4.0)),
        taper_threshold_ratio=float(cfg.get("taper_threshold_ratio", 0.25)),
        compaction_threshold_ratio=float(cfg.get("compaction_threshold_ratio", 0.75)),
        qc_max_segments=int(cfg.get("qc_max_segments", 15)),
        tube_radius_scale=float(cfg.get("tube_radius_scale", 1.0)),
        body_min_halfwidth_px=float(cfg.get("body_min_halfwidth_px", 3.0)),
        centerline_smooth_window=int(cfg.get("centerline_smooth_window", 7)),
        diameter_peak_threshold_ratio=float(cfg.get("diameter_peak_threshold_ratio", 0.95)),
        attachment_min_area_px2=int(cfg.get("attachment_min_area_px2", 24)),
        attachment_orientation_mismatch_deg=float(cfg.get("attachment_orientation_mismatch_deg", 35.0)),
        attachment_far_axis_distance_ratio=float(cfg.get("attachment_far_axis_distance_ratio", 0.78)),
        attachment_short_axis_span_ratio=float(cfg.get("attachment_short_axis_span_ratio", 0.18)),
        attachment_short_normal_span_ratio=float(cfg.get("attachment_short_normal_span_ratio", 0.55)),
        attachment_prune_dilate_kernel=int(cfg.get("attachment_prune_dilate_kernel", 1)),
        diameter_proxy_window_start_norm=float(cfg.get("diameter_proxy_window_start_norm", 0.6)),
        diameter_proxy_window_end_norm=float(cfg.get("diameter_proxy_window_end_norm", 0.8)),
    )


def _load_or_compute_series(
    video_path: Path,
    extraction_cfg: BraidedExtractionConfig,
    temperature_csv: Path | None,
    quicklook_csv: Path | None,
) -> pd.DataFrame:
    if quicklook_csv is not None and quicklook_csv.exists():
        return pd.read_csv(quicklook_csv)
    result = analyze_braided_video_quicklook(
        video_path,
        extraction_cfg,
        temperature_csv=temperature_csv,
    )
    return result.series.copy()


def _draw_text_block(canvas: np.ndarray, lines: list[str], origin_xy: tuple[int, int], width: int) -> None:
    ox, oy = origin_xy
    line_h = 24
    height = line_h * len(lines) + 18
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


def _blend_body_mask(
    frame_bgr: np.ndarray,
    body_mask: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    color_bgr: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    overlay = frame_bgr.copy()
    x0, y0, x1, y1 = roi_xyxy
    roi = overlay[y0:y1, x0:x1]
    color = np.asarray(color_bgr, dtype=np.uint8)
    support = body_mask > 0
    if np.any(support):
        roi[support] = np.round((1.0 - alpha) * roi[support] + alpha * color[None, :]).astype(np.uint8)
    return overlay


def _max_width_segment_global(
    geom,
    roi_offset_xy: np.ndarray,
) -> np.ndarray | None:
    centerline_local = geom.sampled_centerline_xy - roi_offset_xy[None, :]
    widths_px, segments_local = _orthogonal_widths(geom.body_tube_mask, centerline_local)
    valid = np.isfinite(widths_px)
    if not np.any(valid):
        return None
    max_idx = int(np.nanargmax(np.where(valid, widths_px, np.nan)))
    segment_local = segments_local[max_idx]
    if not np.all(np.isfinite(segment_local)):
        return None
    return segment_local + roi_offset_xy[None, :]


def _draw_metric_plot(
    canvas: np.ndarray,
    series: pd.DataFrame,
    frame_idx: int,
    origin_xy: tuple[int, int],
    size_xy: tuple[int, int],
) -> None:
    ox, oy = origin_xy
    width, height = size_xy
    x1 = ox + width
    y1 = oy + height
    cv2.rectangle(canvas, (ox, oy), (x1, y1), (248, 248, 248), -1)
    cv2.rectangle(canvas, (ox, oy), (x1, y1), (120, 120, 120), 1)
    cv2.putText(canvas, "A/B/C trend", (ox + 10, oy + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)

    plot_x0 = ox + 12
    plot_y0 = oy + 32
    plot_w = width - 24
    plot_h = height - 44
    cv2.rectangle(canvas, (plot_x0, plot_y0), (plot_x0 + plot_w, plot_y0 + plot_h), (255, 255, 255), -1)
    cv2.rectangle(canvas, (plot_x0, plot_y0), (plot_x0 + plot_w, plot_y0 + plot_h), (210, 210, 210), 1)

    metric_specs = [
        ("length_axis_recovery", (0, 140, 255), "A"),
        ("diameter_max_recovery", (220, 60, 220), "B"),
        ("area_proj_recovery", (40, 180, 80), "C"),
    ]
    x_values = np.linspace(plot_x0, plot_x0 + plot_w, len(series), endpoint=True)

    for col, color, label in metric_specs:
        if col not in series.columns:
            continue
        values = series[col].to_numpy(dtype=float)
        valid = np.isfinite(values)
        if np.count_nonzero(valid) < 2:
            continue
        points: list[list[int]] = []
        for idx, value in enumerate(values):
            if not np.isfinite(value):
                continue
            px = int(round(float(x_values[idx])))
            py = int(round(float(plot_y0 + plot_h - np.clip(value, 0.0, 1.0) * plot_h)))
            points.append([px, py])
        if len(points) >= 2:
            cv2.polylines(canvas, [np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA)
        value_now = values[frame_idx]
        if np.isfinite(value_now):
            px = int(round(float(x_values[frame_idx])))
            py = int(round(float(plot_y0 + plot_h - np.clip(value_now, 0.0, 1.0) * plot_h)))
            cv2.circle(canvas, (px, py), 4, color, -1)
            cv2.putText(canvas, label, (px + 6, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    px = int(round(float(x_values[frame_idx])))
    cv2.line(canvas, (px, plot_y0), (px, plot_y0 + plot_h), (150, 150, 150), 1, cv2.LINE_AA)
    for frac in (0.0, 0.5, 1.0):
        py = int(round(float(plot_y0 + plot_h - frac * plot_h)))
        cv2.line(canvas, (plot_x0, py), (plot_x0 + plot_w, py), (235, 235, 235), 1, cv2.LINE_AA)


def _format_value(value: float, *, suffix: str = "", precision: int = 1) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{precision}f}{suffix}"


def _make_debug_frame(
    frame_bgr: np.ndarray,
    geom,
    row: pd.Series,
    series: pd.DataFrame,
    frame_idx: int,
    extraction_cfg: BraidedExtractionConfig,
) -> np.ndarray:
    overlay = _blend_body_mask(
        frame_bgr,
        geom.body_tube_mask,
        extraction_cfg.roi_xyxy,
        color_bgr=(100, 220, 130),
        alpha=0.25,
    )
    x0, y0, x1, y1 = extraction_cfg.roi_xyxy
    roi_offset_xy = np.array([x0, y0], dtype=float)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (80, 180, 255), 2)

    contour = np.round(geom.contour_xy).astype(np.int32).reshape(-1, 1, 2)
    body_contour = np.round(geom.body_contour_xy).astype(np.int32).reshape(-1, 1, 2)
    centerline = np.round(geom.sampled_centerline_xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [contour], isClosed=True, color=(120, 120, 120), thickness=1, lineType=cv2.LINE_AA)
    cv2.polylines(overlay, [body_contour], isClosed=True, color=(40, 180, 80), thickness=2, lineType=cv2.LINE_AA)
    cv2.polylines(overlay, [centerline], isClosed=False, color=(0, 220, 255), thickness=2, lineType=cv2.LINE_AA)

    for segment in geom.sampled_width_segments_xy:
        if np.all(np.isfinite(segment)):
            cv2.line(
                overlay,
                np.round(segment[0]).astype(int),
                np.round(segment[1]).astype(int),
                (190, 190, 255),
                1,
                cv2.LINE_AA,
            )

    max_segment = _max_width_segment_global(geom, roi_offset_xy)
    if max_segment is not None:
        cv2.line(
            overlay,
            np.round(max_segment[0]).astype(int),
            np.round(max_segment[1]).astype(int),
            (220, 60, 220),
            4,
            cv2.LINE_AA,
        )

    cv2.line(
        overlay,
        np.round(geom.anchor_xy).astype(int),
        np.round(geom.tip_xy).astype(int),
        (0, 140, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.circle(overlay, np.round(geom.anchor_xy).astype(int), 6, (0, 200, 0), -1)
    cv2.circle(overlay, np.round(geom.tip_xy).astype(int), 6, (0, 0, 255), -1)

    canvas = np.full((overlay.shape[0], overlay.shape[1] + SIDEBAR_WIDTH, 3), 246, dtype=np.uint8)
    canvas[:, : overlay.shape[1]] = overlay
    cv2.line(
        canvas,
        (overlay.shape[1], 0),
        (overlay.shape[1], overlay.shape[0] - 1),
        (180, 180, 180),
        2,
        cv2.LINE_AA,
    )

    legend_lines = [
        "braided demo2 debug view",
        f"frame={frame_idx:03d}  time={_format_value(float(row.get('time_sec', np.nan)), suffix='s', precision=2)}"
        f"  temp={_format_value(float(row.get('temperature_c', np.nan)), suffix='C', precision=2)}",
        "A: orange axis line + green/red endpoints",
        "B: magenta max-width cross section",
        "C: green filled body region  (numeric C = projected area integral)",
        f"A length_axis={_format_value(float(row.get('length_axis_px', np.nan)), suffix=' px')}"
        f"  rec={_format_value(float(row.get('length_axis_recovery', np.nan)), precision=3)}",
        f"B diameter_max={_format_value(float(row.get('diameter_max_px', np.nan)), suffix=' px')}"
        f"  rec={_format_value(float(row.get('diameter_max_recovery', np.nan)), precision=3)}",
        f"C area_proj={_format_value(float(row.get('area_proj_px2', np.nan)), suffix=' px2', precision=0)}"
        f"  rec={_format_value(float(row.get('area_proj_recovery', np.nan)), precision=3)}",
        f"qc endpoint_jump={_format_value(float(row.get('endpoint_jump_px', np.nan)), suffix=' px')}"
        f"  leak={_format_value(float(row.get('body_mask_attachment_leak_fraction', np.nan)), precision=3)}"
        f"  branch={_format_value(float(row.get('branch_component_count_after_pruning', np.nan)), precision=0)}",
    ]
    panel_x = overlay.shape[1] + 18
    _draw_text_block(canvas, legend_lines, origin_xy=(panel_x, 36), width=SIDEBAR_WIDTH - 36)
    _draw_metric_plot(
        canvas,
        series,
        frame_idx,
        origin_xy=(panel_x, 278),
        size_xy=(SIDEBAR_WIDTH - 36, 132),
    )
    return canvas


def main() -> None:
    args = _parse_args()
    config = yaml.safe_load((ROOT / "configs/minimal.yaml").read_text(encoding="utf-8"))
    extraction_cfg = _make_extraction_config(config["analysis"][args.config_key])

    video_path = args.video.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temperature_csv = args.temperature_csv.resolve() if args.temperature_csv else None
    quicklook_csv = args.quicklook_csv.resolve() if args.quicklook_csv else None
    series = _load_or_compute_series(video_path, extraction_cfg, temperature_csv, quicklook_csv)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width + SIDEBAR_WIDTH, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"failed to open video writer: {output_path}")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame_idx >= len(series):
            break
        row = series.iloc[frame_idx]
        try:
            geom = extract_braided_geometry(frame, extraction_cfg)
            debug_frame = _make_debug_frame(frame, geom, row, series, frame_idx, extraction_cfg)
        except RuntimeError as exc:
            debug_frame = frame.copy()
            msg = f"frame {frame_idx:03d}: extraction failed: {exc}"
            cv2.putText(debug_frame, msg, (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        writer.write(debug_frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"wrote braided A/B/C debug video to {output_path}")


if __name__ == "__main__":
    main()
