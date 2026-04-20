from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import pandas as pd

from .extract import ExtractionConfig, extract_geometry
from .extract_braided import BraidedExtractionConfig, extract_braided_geometry
from .pipeline import AnalysisResult

ObjectType = Literal["wire_like", "braided_like"]

SIDEBAR_WIDTH = 420
PANEL_BG = (246, 246, 246)
TEXT_COLOR = (35, 35, 35)
MUTED_TEXT = (95, 95, 95)
ERROR_COLOR = (0, 0, 255)
ROUTE_COLORS: dict[str, tuple[int, int, int]] = {
    "A": (0, 140, 255),
    "B": (220, 60, 220),
    "C": (60, 180, 80),
}


def render_process_debug_video(
    video_path: str | Path,
    output_path: str | Path,
    *,
    extraction: ExtractionConfig | BraidedExtractionConfig,
    result: AnalysisResult,
    object_type: str | None = None,
    output_fps: float | None = None,
) -> Path:
    object_kind = _normalize_object_type(object_type, extraction, result.series)
    series = _prepare_series(result.series)
    if series.empty:
        raise ValueError("analysis result has no frames to render")

    video_path = Path(video_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(source_fps) or source_fps <= 0.0:
        source_fps = 20.0
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rendered_fps = float(output_fps) if output_fps is not None else _default_output_fps(series["frame"], source_fps)
    if not np.isfinite(rendered_fps) or rendered_fps <= 0.0:
        cap.release()
        raise ValueError("output_fps must be positive")

    writer = _open_browser_compatible_writer(
        output_path=output_path,
        fps=rendered_fps,
        frame_size=(source_width + SIDEBAR_WIDTH, source_height),
    )
    if writer is None:
        cap.release()
        raise RuntimeError(f"failed to open video writer: {output_path}")

    next_frame_pos: int | None = None
    try:
        for frame_idx, row in enumerate(series.itertuples(index=False)):
            target_frame = int(row.frame)
            frame_bgr, next_frame_pos = _read_frame_at(cap, target_frame, next_frame_pos)
            try:
                if object_kind == "braided_like":
                    assert isinstance(extraction, BraidedExtractionConfig)
                    geom = extract_braided_geometry(frame_bgr, extraction)
                    canvas = _render_braided_debug_frame(
                        frame_bgr=frame_bgr,
                        geom=geom,
                        row=row,
                        series=series,
                        frame_idx=frame_idx,
                        extraction=extraction,
                    )
                else:
                    assert isinstance(extraction, ExtractionConfig)
                    geom = extract_geometry(frame_bgr, extraction)
                    canvas = _render_wire_debug_frame(
                        frame_bgr=frame_bgr,
                        geom=geom,
                        row=row,
                        series=series,
                        frame_idx=frame_idx,
                        extraction=extraction,
                    )
            except Exception as exc:  # noqa: BLE001
                canvas = _render_error_canvas(
                    frame_bgr=frame_bgr,
                    title=f"{object_kind.replace('_', '-')} analysis process",
                    message=f"frame {target_frame}: process render failed: {exc}",
                )
            writer.write(canvas)
    finally:
        cap.release()
        writer.release()

    return output_path


def _open_browser_compatible_writer(
    *,
    output_path: Path,
    fps: float,
    frame_size: tuple[int, int],
) -> cv2.VideoWriter | None:
    # Prefer H.264/avc1 so the exported MP4 can play inside browser <video>.
    for codec in ("avc1", "H264", "mp4v"):
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            frame_size,
        )
        if writer.isOpened():
            return writer
        writer.release()
    return None


def _normalize_object_type(
    object_type: str | None,
    extraction: ExtractionConfig | BraidedExtractionConfig,
    series: pd.DataFrame,
) -> ObjectType:
    normalized = str(object_type or "").strip().lower().replace("-", "_")
    if normalized in {"wire", "wire_like", "demo"}:
        return "wire_like"
    if normalized in {"braided", "braided_like", "braided_demo"}:
        return "braided_like"
    if isinstance(extraction, BraidedExtractionConfig):
        return "braided_like"
    if isinstance(extraction, ExtractionConfig):
        return "wire_like"
    if "length_axis_px" in series.columns or "area_proj_px2" in series.columns:
        return "braided_like"
    return "wire_like"


def _prepare_series(series: pd.DataFrame) -> pd.DataFrame:
    prepared = series.copy()
    if "frame" not in prepared.columns:
        raise ValueError("analysis series must contain a frame column")
    prepared["frame"] = pd.to_numeric(prepared["frame"], errors="coerce")
    prepared = prepared[np.isfinite(prepared["frame"])]
    prepared["frame"] = prepared["frame"].astype(int)
    prepared = prepared[prepared["frame"] >= 0]
    if prepared.empty:
        return prepared.reset_index(drop=True)
    prepared = prepared.sort_values("frame").drop_duplicates(subset=["frame"], keep="first")
    return prepared.reset_index(drop=True)


def _default_output_fps(frame_series: pd.Series, source_fps: float) -> float:
    frames = frame_series.to_numpy(dtype=float)
    if len(frames) < 2:
        return source_fps
    diffs = np.diff(frames)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return source_fps
    stride = max(1.0, float(np.median(diffs)))
    return max(source_fps / stride, 1.0)


def _read_frame_at(
    cap: cv2.VideoCapture,
    target_frame: int,
    next_frame_pos: int | None,
) -> tuple[np.ndarray, int]:
    if next_frame_pos is None or target_frame < next_frame_pos or (target_frame - next_frame_pos) > 32:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        next_frame_pos = target_frame
    while next_frame_pos < target_frame:
        ok, _ = cap.read()
        if not ok:
            raise RuntimeError(f"failed to skip to frame {target_frame}")
        next_frame_pos += 1
    ok, frame_bgr = cap.read()
    if not ok:
        raise RuntimeError(f"failed to read frame {target_frame}")
    return frame_bgr, next_frame_pos + 1


def _render_wire_debug_frame(
    *,
    frame_bgr: np.ndarray,
    geom,
    row: Any,
    series: pd.DataFrame,
    frame_idx: int,
    extraction: ExtractionConfig,
) -> np.ndarray:
    overlay = frame_bgr.copy()
    x0, y0, x1, y1 = extraction.roi_xyxy
    cv2.rectangle(overlay, (x0, y0), (x1, y1), ROUTE_COLORS["A"], 2)
    contour = np.round(geom.contour_xy).astype(np.int32).reshape(-1, 1, 2)
    centerline = np.round(geom.sampled_centerline_xy).astype(np.int32).reshape(-1, 1, 2)
    fit_samples = np.round(geom.fit_samples_xy).astype(np.int32).reshape(-1, 1, 2)
    fit_curve = np.round(geom.fitted_curve_xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [contour], isClosed=True, color=(120, 120, 120), thickness=1, lineType=cv2.LINE_AA)
    cv2.polylines(overlay, [centerline], isClosed=False, color=ROUTE_COLORS["B"], thickness=2, lineType=cv2.LINE_AA)
    cv2.polylines(overlay, [fit_samples], isClosed=False, color=(0, 220, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.polylines(overlay, [fit_curve], isClosed=False, color=(255, 180, 0), thickness=2, lineType=cv2.LINE_AA)

    cv2.line(
        overlay,
        np.round(geom.route_a_anchor_xy).astype(int),
        np.round(geom.route_a_tip_xy).astype(int),
        ROUTE_COLORS["A"],
        2,
        cv2.LINE_AA,
    )
    cv2.circle(overlay, np.round(geom.route_a_anchor_xy).astype(int), 5, ROUTE_COLORS["A"], -1)
    cv2.circle(overlay, np.round(geom.route_a_tip_xy).astype(int), 6, ROUTE_COLORS["A"], -1)

    cv2.circle(overlay, np.round(geom.anchor_xy).astype(int), 5, (0, 220, 0), -1)
    cv2.circle(overlay, np.round(geom.tip_xy).astype(int), 6, (0, 0, 255), -1)

    if _has_row_values(row, "route_c_anchor_x", "route_c_anchor_y", "route_c_tip_x", "route_c_tip_y"):
        route_c_anchor = np.array([float(row.route_c_anchor_x), float(row.route_c_anchor_y)], dtype=float)
        route_c_tip = np.array([float(row.route_c_tip_x), float(row.route_c_tip_y)], dtype=float)
        cv2.line(
            overlay,
            np.round(route_c_anchor).astype(int),
            np.round(route_c_tip).astype(int),
            ROUTE_COLORS["C"],
            2,
            cv2.LINE_AA,
        )
        cv2.circle(overlay, np.round(route_c_anchor).astype(int), 5, ROUTE_COLORS["C"], -1)
        cv2.circle(overlay, np.round(route_c_tip).astype(int), 6, ROUTE_COLORS["C"], -1)

    canvas = _make_canvas(overlay)
    panel_x = overlay.shape[1] + 18
    lines = [
        "wire-like analysis process",
        _frame_line(row),
        f"A endpoint/chord  x={_fmt(_row_value(row, 'x_route_a_px'), suffix=' px')}"
        f"  rec={_fmt(_row_value(row, 'x_route_a_recovery'), precision=3)}",
        f"B shape-fit  x_fit={_fmt(_row_value(row, 'x_fit_px'), suffix=' px')}"
        f"  kappa={_fmt(_row_value(row, 'kappa_fit_px_inv'), precision=5)}",
        f"C temporal  x={_fmt(_row_value(row, 'x_route_c_px'), suffix=' px')}"
        f"  kappa={_fmt(_row_value(row, 'kappa_route_c_px_inv'), precision=5)}",
        f"quality={_fmt(_row_value(row, 'quality'), precision=3)}"
        f"  model={_row_text(row, 'model_name') or '-'}",
        "A: magenta chord  B: cyan/orange fit  C: green temporal chord",
    ]
    _draw_text_block(canvas, lines, origin_xy=(panel_x, 36), width=SIDEBAR_WIDTH - 36)
    _draw_trend_plot(
        canvas=canvas,
        series=series,
        frame_idx=frame_idx,
        origin_xy=(panel_x, 248),
        size_xy=(SIDEBAR_WIDTH - 36, 150),
        title="A/B/C recovery trend",
        metric_specs=[
            ("x_route_a_recovery", ROUTE_COLORS["A"], "A"),
            ("kappa_fit_recovery", ROUTE_COLORS["B"], "B"),
            ("kappa_route_c_recovery", ROUTE_COLORS["C"], "C"),
        ],
    )
    return canvas


def _render_braided_debug_frame(
    *,
    frame_bgr: np.ndarray,
    geom,
    row: Any,
    series: pd.DataFrame,
    frame_idx: int,
    extraction: BraidedExtractionConfig,
) -> np.ndarray:
    overlay = _blend_mask(frame_bgr, geom.body_tube_mask, extraction.roi_xyxy, color_bgr=(100, 220, 130), alpha=0.24)
    x0, y0, x1, y1 = extraction.roi_xyxy
    cv2.rectangle(overlay, (x0, y0), (x1, y1), ROUTE_COLORS["A"], 2)

    contour = np.round(geom.contour_xy).astype(np.int32).reshape(-1, 1, 2)
    body_contour = np.round(geom.body_contour_xy).astype(np.int32).reshape(-1, 1, 2)
    centerline = np.round(geom.sampled_centerline_xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [contour], isClosed=True, color=(120, 120, 120), thickness=1, lineType=cv2.LINE_AA)
    cv2.polylines(overlay, [body_contour], isClosed=True, color=ROUTE_COLORS["C"], thickness=2, lineType=cv2.LINE_AA)
    cv2.polylines(overlay, [centerline], isClosed=False, color=(0, 220, 255), thickness=2, lineType=cv2.LINE_AA)

    for segment in getattr(geom, "sampled_width_segments_xy", []):
        if np.all(np.isfinite(segment)):
            cv2.line(
                overlay,
                np.round(segment[0]).astype(int),
                np.round(segment[1]).astype(int),
                (190, 190, 255),
                1,
                cv2.LINE_AA,
            )

    max_segment = _max_segment(getattr(geom, "sampled_width_segments_xy", np.empty((0, 2, 2))))
    if max_segment is not None:
        cv2.line(
            overlay,
            np.round(max_segment[0]).astype(int),
            np.round(max_segment[1]).astype(int),
            ROUTE_COLORS["B"],
            4,
            cv2.LINE_AA,
        )

    cv2.line(
        overlay,
        np.round(geom.anchor_xy).astype(int),
        np.round(geom.tip_xy).astype(int),
        ROUTE_COLORS["A"],
        3,
        cv2.LINE_AA,
    )
    cv2.circle(overlay, np.round(geom.anchor_xy).astype(int), 6, (0, 200, 0), -1)
    cv2.circle(overlay, np.round(geom.tip_xy).astype(int), 6, (0, 0, 255), -1)

    canvas = _make_canvas(overlay)
    panel_x = overlay.shape[1] + 18
    lines = [
        "braided analysis process",
        _frame_line(row),
        f"A axis-length={_fmt(_row_value(row, 'length_axis_px'), suffix=' px')}"
        f"  rec={_fmt(_row_value(row, 'length_axis_recovery'), precision=3)}",
        f"B max-width={_fmt(_row_value(row, 'diameter_max_px'), suffix=' px')}"
        f"  rec={_fmt(_row_value(row, 'diameter_max_recovery'), precision=3)}",
        f"C area={_fmt(_row_value(row, 'area_proj_px2'), suffix=' px2', precision=0)}"
        f"  rec={_fmt(_row_value(row, 'area_proj_recovery'), precision=3)}",
        f"qc jump={_fmt(_row_value(row, 'endpoint_jump_px'), suffix=' px')}"
        f"  leak={_fmt(_row_value(row, 'body_mask_attachment_leak_fraction'), precision=3)}",
        f"centerline disagreement={_fmt(_row_value(row, 'centerline_disagreement'), precision=3)}"
        f"  quality={_fmt(_row_value(row, 'quality'), precision=3)}",
    ]
    _draw_text_block(canvas, lines, origin_xy=(panel_x, 36), width=SIDEBAR_WIDTH - 36)
    _draw_trend_plot(
        canvas=canvas,
        series=series,
        frame_idx=frame_idx,
        origin_xy=(panel_x, 248),
        size_xy=(SIDEBAR_WIDTH - 36, 150),
        title="A/B/C recovery trend",
        metric_specs=[
            ("length_axis_recovery", ROUTE_COLORS["A"], "A"),
            ("diameter_max_recovery", ROUTE_COLORS["B"], "B"),
            ("area_proj_recovery", ROUTE_COLORS["C"], "C"),
        ],
    )
    return canvas


def _render_error_canvas(*, frame_bgr: np.ndarray, title: str, message: str) -> np.ndarray:
    canvas = _make_canvas(frame_bgr)
    panel_x = frame_bgr.shape[1] + 18
    _draw_text_block(canvas, [title, message], origin_xy=(panel_x, 36), width=SIDEBAR_WIDTH - 36, text_color=ERROR_COLOR)
    cv2.putText(canvas, message, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72, ERROR_COLOR, 2, cv2.LINE_AA)
    return canvas


def _make_canvas(frame_bgr: np.ndarray) -> np.ndarray:
    canvas = np.full((frame_bgr.shape[0], frame_bgr.shape[1] + SIDEBAR_WIDTH, 3), PANEL_BG, dtype=np.uint8)
    canvas[:, : frame_bgr.shape[1]] = frame_bgr
    cv2.line(
        canvas,
        (frame_bgr.shape[1], 0),
        (frame_bgr.shape[1], frame_bgr.shape[0] - 1),
        (180, 180, 180),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _draw_text_block(
    canvas: np.ndarray,
    lines: list[str],
    *,
    origin_xy: tuple[int, int],
    width: int,
    text_color: tuple[int, int, int] = TEXT_COLOR,
) -> None:
    ox, oy = origin_xy
    line_h = 24
    height = line_h * len(lines) + 18
    cv2.rectangle(canvas, (ox - 8, oy - 24), (ox + width, oy - 24 + height), (245, 245, 245), -1)
    cv2.rectangle(canvas, (ox - 8, oy - 24), (ox + width, oy - 24 + height), (120, 120, 120), 1)
    for idx, line in enumerate(lines):
        color = MUTED_TEXT if idx == 0 else text_color
        cv2.putText(
            canvas,
            line,
            (ox, oy + idx * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            color,
            1,
            cv2.LINE_AA,
        )


def _draw_trend_plot(
    *,
    canvas: np.ndarray,
    series: pd.DataFrame,
    frame_idx: int,
    origin_xy: tuple[int, int],
    size_xy: tuple[int, int],
    title: str,
    metric_specs: list[tuple[str, tuple[int, int, int], str]],
) -> None:
    ox, oy = origin_xy
    width, height = size_xy
    x1 = ox + width
    y1 = oy + height
    cv2.rectangle(canvas, (ox, oy), (x1, y1), (248, 248, 248), -1)
    cv2.rectangle(canvas, (ox, oy), (x1, y1), (120, 120, 120), 1)
    cv2.putText(canvas, title, (ox + 10, oy + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)

    plot_x0 = ox + 12
    plot_y0 = oy + 32
    plot_w = width - 24
    plot_h = height - 44
    cv2.rectangle(canvas, (plot_x0, plot_y0), (plot_x0 + plot_w, plot_y0 + plot_h), (255, 255, 255), -1)
    cv2.rectangle(canvas, (plot_x0, plot_y0), (plot_x0 + plot_w, plot_y0 + plot_h), (210, 210, 210), 1)

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


def _blend_mask(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    *,
    color_bgr: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    overlay = frame_bgr.copy()
    x0, y0, x1, y1 = roi_xyxy
    roi = overlay[y0:y1, x0:x1]
    color = np.asarray(color_bgr, dtype=np.uint8)
    support = mask > 0
    if np.any(support):
        roi[support] = np.round((1.0 - alpha) * roi[support] + alpha * color[None, :]).astype(np.uint8)
    return overlay


def _max_segment(segments_xy: np.ndarray) -> np.ndarray | None:
    if segments_xy is None or len(segments_xy) == 0:
        return None
    lengths = []
    for segment in segments_xy:
        if not np.all(np.isfinite(segment)):
            lengths.append(-np.inf)
            continue
        lengths.append(float(np.linalg.norm(segment[1] - segment[0])))
    if not lengths or not np.isfinite(max(lengths)):
        return None
    return np.asarray(segments_xy[int(np.argmax(lengths))], dtype=float)


def _frame_line(row: Any) -> str:
    return (
        f"frame={int(_row_value(row, 'frame') or 0):04d}"
        f"  time={_fmt(_row_value(row, 'time_sec'), suffix='s', precision=2)}"
        f"  temp={_fmt(_row_value(row, 'temperature_c'), suffix='C', precision=2)}"
    )


def _row_value(row: Any, field: str) -> float | None:
    value = getattr(row, field, None)
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _row_text(row: Any, field: str) -> str | None:
    value = getattr(row, field, None)
    if value in {None, ""}:
        return None
    return str(value)


def _fmt(value: float | None, *, suffix: str = "", precision: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:.{precision}f}{suffix}"


def _has_row_values(row: Any, *fields: str) -> bool:
    return all(_row_value(row, field) is not None for field in fields)
