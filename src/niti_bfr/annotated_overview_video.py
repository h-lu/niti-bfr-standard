from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import pandas as pd

from .extract import ExtractionConfig, extract_geometry
from .extract_braided import BraidedExtractionConfig, extract_braided_geometry
from .export_contract import route_results_by_alias
from .pipeline import AnalysisResult

ObjectType = Literal["wire_like", "braided_like"]

SIDEBAR_WIDTH = 430
PANEL_BG = (246, 246, 246)
PANEL_BORDER = (140, 140, 140)
TEXT_COLOR = (35, 35, 35)
MUTED_TEXT = (95, 95, 95)
ERROR_COLOR = (0, 0, 255)
ROUTE_COLORS: dict[str, tuple[int, int, int]] = {
    "A": (0, 140, 255),
    "B": (220, 60, 220),
    "C": (60, 180, 80),
}


@dataclass(frozen=True)
class _RouteSpec:
    alias: str
    metric_key: str
    display_label: str
    value_cols: tuple[str, ...]
    recovery_cols: tuple[str, ...]
    color_bgr: tuple[int, int, int]


def render_annotated_overview_video(
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

    route_specs = _route_specs_for_result(object_kind, series, result.route_results)

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
    total_rows = len(series)
    try:
        for analyzed_idx, row in enumerate(series.itertuples(index=False), start=1):
            frame_number = int(row.frame)
            frame_bgr, next_frame_pos = _read_frame_at(cap, frame_number, next_frame_pos)
            overlay = _make_overlay_frame(
                frame_bgr=frame_bgr,
                row=row,
                analyzed_idx=analyzed_idx,
                analyzed_total=total_rows,
                object_type=object_kind,
                extraction=extraction,
            )
            canvas = _compose_canvas(
                overlay=overlay,
                row=row,
                analyzed_idx=analyzed_idx,
                analyzed_total=total_rows,
                object_type=object_kind,
                result=result,
                route_specs=route_specs,
                series=series,
                source_fps=source_fps,
                output_fps=rendered_fps,
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


def _route_specs_for_result(
    object_type: ObjectType,
    series: pd.DataFrame,
    route_results: list[dict[str, Any]] | None,
) -> list[_RouteSpec]:
    defaults = _default_route_specs(object_type)
    entries_by_alias = route_results_by_alias(route_results)
    specs: list[_RouteSpec] = []
    for default in defaults:
        entry = entries_by_alias.get(default.alias, {})
        metric_key = str(entry.get("metric_key") or default.metric_key)
        display_label = str(entry.get("display_label") or default.display_label)
        value_cols = _candidate_columns(
            entry.get("value_series_col"),
            *default.value_cols,
        )
        if object_type == "braided_like" and default.alias == "A":
            value_cols = _candidate_columns(*value_cols, "length_axis_px")
        if object_type == "braided_like" and default.alias == "C":
            value_cols = _candidate_columns(*value_cols, "area_proj_px2")
        recovery_cols = _candidate_columns(
            entry.get("recovery_series_col"),
            *default.recovery_cols,
        )
        specs.append(
            _RouteSpec(
                alias=default.alias,
                metric_key=metric_key,
                display_label=display_label,
                value_cols=tuple(col for col in value_cols if col in series.columns),
                recovery_cols=tuple(col for col in recovery_cols if col in series.columns),
                color_bgr=default.color_bgr,
            )
        )
    return specs


def _default_route_specs(object_type: ObjectType) -> list[_RouteSpec]:
    if object_type == "braided_like":
        return [
            _RouteSpec("A", "length_axis", "A:length_axis", ("length_axis_formal_px", "length_axis_px"), ("length_axis_recovery",), ROUTE_COLORS["A"]),
            _RouteSpec("B", "diameter_max", "B:diameter_max", ("diameter_max_px",), ("diameter_max_recovery",), ROUTE_COLORS["B"]),
            _RouteSpec("C", "area_proj", "C:area_proj", ("area_proj_formal_px2", "area_proj_px2"), ("area_proj_recovery",), ROUTE_COLORS["C"]),
        ]
    return [
        _RouteSpec("A", "x_route_a", "A:x_route_a", ("x_route_a_px",), ("x_route_a_recovery",), ROUTE_COLORS["A"]),
        _RouteSpec("B", "kappa_fit", "B:kappa_fit", ("kappa_fit_px_inv",), ("kappa_fit_recovery",), ROUTE_COLORS["B"]),
        _RouteSpec("C", "kappa_route_c", "C:kappa_route_c", ("kappa_route_c_px_inv",), ("kappa_route_c_recovery",), ROUTE_COLORS["C"]),
    ]


def _candidate_columns(*values: Any) -> tuple[str, ...]:
    seen: set[str] = set()
    cols: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        if not value or value in seen:
            continue
        seen.add(value)
        cols.append(value)
    return tuple(cols)


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


def _make_overlay_frame(
    *,
    frame_bgr: np.ndarray,
    row: Any,
    analyzed_idx: int,
    analyzed_total: int,
    object_type: ObjectType,
    extraction: ExtractionConfig | BraidedExtractionConfig,
) -> np.ndarray:
    overlay = frame_bgr.copy()
    header_lines = [
        f"{object_type.replace('_', '-')} annotated overview",
        f"frame {int(row.frame):04d} | analyzed {analyzed_idx}/{analyzed_total} | t={_format_float(_row_value(row, 'time_sec'), precision=2, suffix='s')}",
    ]
    if object_type == "wire_like":
        return _make_wire_overlay(overlay, row, header_lines, extraction)
    return _make_braided_overlay(overlay, row, header_lines, extraction)


def _make_wire_overlay(
    overlay: np.ndarray,
    row: Any,
    header_lines: list[str],
    extraction: ExtractionConfig | BraidedExtractionConfig,
) -> np.ndarray:
    assert isinstance(extraction, ExtractionConfig)
    source_frame = overlay.copy()
    x0, y0, x1, y1 = extraction.roi_xyxy
    try:
        geom = extract_geometry(source_frame, extraction)
        contour = _points_as_polyline(geom.contour_xy)
        centerline = _points_as_polyline(geom.sampled_centerline_xy)
        fit_samples = _points_as_polyline(geom.fit_samples_xy)
        fitted_curve = _points_as_polyline(geom.fitted_curve_xy)
        if contour is not None:
            cv2.polylines(overlay, [contour], True, (130, 130, 130), 1, cv2.LINE_AA)
        if centerline is not None:
            cv2.polylines(overlay, [centerline], False, (255, 220, 120), 2, cv2.LINE_AA)
        if fit_samples is not None:
            cv2.polylines(overlay, [fit_samples], False, (255, 255, 0), 2, cv2.LINE_AA)
        if fitted_curve is not None:
            cv2.polylines(overlay, [fitted_curve], False, ROUTE_COLORS["B"], 2, cv2.LINE_AA)
        _draw_segment(overlay, geom.route_a_anchor_xy, geom.route_a_tip_xy, ROUTE_COLORS["A"], thickness=3)
        _draw_point(overlay, geom.route_a_anchor_xy, ROUTE_COLORS["A"], radius=5)
        _draw_point(overlay, geom.route_a_tip_xy, ROUTE_COLORS["A"], radius=6)
        _draw_point(overlay, geom.anchor_xy, (60, 200, 60), radius=5)
        _draw_point(overlay, geom.tip_xy, ROUTE_COLORS["B"], radius=6)
        if _has_row_values(row, "route_c_anchor_x", "route_c_anchor_y", "route_c_tip_x", "route_c_tip_y"):
            route_c_anchor = np.array([float(row.route_c_anchor_x), float(row.route_c_anchor_y)], dtype=float)
            route_c_tip = np.array([float(row.route_c_tip_x), float(row.route_c_tip_y)], dtype=float)
            _draw_segment(overlay, route_c_anchor, route_c_tip, ROUTE_COLORS["C"], thickness=3)
            _draw_point(overlay, route_c_anchor, ROUTE_COLORS["C"], radius=4)
            _draw_point(overlay, route_c_tip, ROUTE_COLORS["C"], radius=5)
    except RuntimeError as exc:
        header_lines.append(f"extraction warning: {exc}")
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (80, 180, 255), 2)
    _draw_header_badge(overlay, header_lines)
    _draw_legend_box(
        overlay,
        [
            ("A", "endpoint / chord", ROUTE_COLORS["A"]),
            ("B", "shape fit / centerline", ROUTE_COLORS["B"]),
            ("C", "temporal chord", ROUTE_COLORS["C"]),
        ],
    )
    return overlay


def _make_braided_overlay(
    overlay: np.ndarray,
    row: Any,
    header_lines: list[str],
    extraction: ExtractionConfig | BraidedExtractionConfig,
) -> np.ndarray:
    assert isinstance(extraction, BraidedExtractionConfig)
    source_frame = overlay.copy()
    x0, y0, x1, y1 = extraction.roi_xyxy
    try:
        geom = extract_braided_geometry(source_frame, extraction)
        blended = _blend_mask_on_roi(overlay, geom.body_tube_mask, extraction.roi_xyxy, ROUTE_COLORS["C"], alpha=0.26)
        overlay[:, :] = blended
        contour = _points_as_polyline(geom.contour_xy)
        body_contour = _points_as_polyline(geom.body_contour_xy)
        centerline = _points_as_polyline(geom.sampled_centerline_xy)
        if contour is not None:
            cv2.polylines(overlay, [contour], True, (130, 130, 130), 1, cv2.LINE_AA)
        if body_contour is not None:
            cv2.polylines(overlay, [body_contour], True, ROUTE_COLORS["C"], 2, cv2.LINE_AA)
        if centerline is not None:
            cv2.polylines(overlay, [centerline], False, (255, 235, 120), 2, cv2.LINE_AA)
        _draw_segment(overlay, geom.anchor_xy, geom.tip_xy, ROUTE_COLORS["A"], thickness=3)
        _draw_point(overlay, geom.anchor_xy, (60, 200, 60), radius=5)
        _draw_point(overlay, geom.tip_xy, (0, 0, 255), radius=5)
        for segment in np.asarray(geom.sampled_width_segments_xy):
            if segment.shape == (2, 2) and np.all(np.isfinite(segment)):
                cv2.line(
                    overlay,
                    tuple(np.round(segment[0]).astype(int)),
                    tuple(np.round(segment[1]).astype(int)),
                    (235, 215, 255),
                    1,
                    cv2.LINE_AA,
                )
        max_width_segment = _max_width_segment(geom.sampled_width_segments_xy)
        if max_width_segment is not None:
            cv2.line(
                overlay,
                tuple(np.round(max_width_segment[0]).astype(int)),
                tuple(np.round(max_width_segment[1]).astype(int)),
                ROUTE_COLORS["B"],
                4,
                cv2.LINE_AA,
            )
    except RuntimeError as exc:
        header_lines.append(f"extraction warning: {exc}")
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (80, 180, 255), 2)
    _draw_header_badge(overlay, header_lines)
    _draw_legend_box(
        overlay,
        [
            ("A", "axis / functional length", ROUTE_COLORS["A"]),
            ("B", "max-width cross section", ROUTE_COLORS["B"]),
            ("C", "body mask / projected area", ROUTE_COLORS["C"]),
        ],
    )
    return overlay


def _blend_mask_on_roi(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    color_bgr: tuple[int, int, int],
    *,
    alpha: float,
) -> np.ndarray:
    blended = frame_bgr.copy()
    x0, y0, x1, y1 = roi_xyxy
    roi = blended[y0:y1, x0:x1]
    support = mask > 0
    if np.any(support):
        color = np.asarray(color_bgr, dtype=np.float32)
        roi[support] = np.round((1.0 - alpha) * roi[support].astype(np.float32) + alpha * color[None, :]).astype(np.uint8)
    return blended


def _compose_canvas(
    *,
    overlay: np.ndarray,
    row: Any,
    analyzed_idx: int,
    analyzed_total: int,
    object_type: ObjectType,
    result: AnalysisResult,
    route_specs: list[_RouteSpec],
    series: pd.DataFrame,
    source_fps: float,
    output_fps: float,
) -> np.ndarray:
    height, width = overlay.shape[:2]
    canvas = np.full((height, width + SIDEBAR_WIDTH, 3), PANEL_BG[0], dtype=np.uint8)
    canvas[:, :width] = overlay
    cv2.line(canvas, (width, 0), (width, height - 1), (180, 180, 180), 2, cv2.LINE_AA)

    panel_x = width + 18
    summary_lines = _summary_lines(
        row=row,
        analyzed_idx=analyzed_idx,
        analyzed_total=analyzed_total,
        object_type=object_type,
        result=result,
        source_fps=source_fps,
        output_fps=output_fps,
    )
    _draw_text_panel(canvas, summary_lines, origin_xy=(panel_x, 34), width=SIDEBAR_WIDTH - 36)
    _draw_route_status_panel(
        canvas,
        row=row,
        route_specs=route_specs,
        route_results=result.route_results,
        origin_xy=(panel_x, 178),
        width=SIDEBAR_WIDTH - 36,
    )
    _draw_trend_plot(
        canvas,
        series=series,
        frame_number=int(row.frame),
        route_specs=route_specs,
        origin_xy=(panel_x, 344),
        size_xy=(SIDEBAR_WIDTH - 36, 160),
    )
    return canvas


def _summary_lines(
    *,
    row: Any,
    analyzed_idx: int,
    analyzed_total: int,
    object_type: ObjectType,
    result: AnalysisResult,
    source_fps: float,
    output_fps: float,
) -> list[str]:
    current_time = _row_value(row, "time_sec")
    if not np.isfinite(current_time):
        current_time = int(row.frame) / max(source_fps, 1e-6)
    current_temp = _row_value(row, "temperature_c")
    quality = _row_value(row, "quality")

    route_entries = route_results_by_alias(result.route_results)
    primary_alias = _selected_alias(route_entries, "selected_as_primary")
    formal_alias = _selected_alias(route_entries, "selected_as_formal")
    recommended_alias = formal_alias or primary_alias

    af_parts: list[str] = []
    if result.af95_c is not None and np.isfinite(result.af95_c):
        af_parts.append(f"Af95={result.af95_c:.2f}C")
    if result.aftan_c is not None and np.isfinite(result.aftan_c):
        af_parts.append(f"Aftan={result.aftan_c:.2f}C")
    if not af_parts and result.provisional_af95_c is not None and np.isfinite(result.provisional_af95_c):
        af_parts.append(f"prov Af95={result.provisional_af95_c:.2f}C")
    if not af_parts and result.provisional_aftan_c is not None and np.isfinite(result.provisional_aftan_c):
        af_parts.append(f"prov Aftan={result.provisional_aftan_c:.2f}C")

    return [
        f"{object_type.replace('_', '-')} overview video",
        f"frame={int(row.frame):04d}  analyzed={analyzed_idx}/{analyzed_total}  time={current_time:.2f}s",
        f"mode={result.mode}  reportability={result.reportability_status}  quality={_format_float(quality, precision=2)}",
        f"temp={_format_float(current_temp, precision=2, suffix='C')}  output_fps={output_fps:.2f}  source_fps={source_fps:.2f}",
        f"primary={primary_alias or 'n/a'}  formal={formal_alias or 'n/a'}  recommended={recommended_alias or 'n/a'}",
        "  ".join(af_parts) if af_parts else "Af summary unavailable",
    ]


def _selected_alias(route_entries: dict[str, dict[str, Any]], flag: str) -> str | None:
    for alias in ("A", "B", "C"):
        entry = route_entries.get(alias) or {}
        if bool(entry.get(flag)):
            return alias
    return None


def _draw_route_status_panel(
    canvas: np.ndarray,
    *,
    row: Any,
    route_specs: list[_RouteSpec],
    route_results: list[dict[str, Any]] | None,
    origin_xy: tuple[int, int],
    width: int,
) -> None:
    entries = route_results_by_alias(route_results)
    lines: list[tuple[str, tuple[int, int, int], str]] = []
    for spec in route_specs:
        entry = entries.get(spec.alias, {})
        value = _first_row_value(row, spec.value_cols)
        recovery = _first_row_value(row, spec.recovery_cols)
        status = str(entry.get("reportability_status") or "route_result")
        role = _route_role(entry)
        metric_label = spec.display_label.split(":", 1)[-1]
        metric_text = f"{metric_label}={_format_metric_value(spec.metric_key, value)}"
        recovery_text = f" rec={_format_float(recovery, precision=3)}" if np.isfinite(recovery) else ""
        status_text = f"  {role or status}"
        lines.append((spec.alias, spec.color_bgr, f"{metric_text}{recovery_text}{status_text}"))
    _draw_color_lines_panel(canvas, lines, origin_xy=origin_xy, width=width, title="Current A/B/C metrics")


def _route_role(entry: dict[str, Any]) -> str:
    if bool(entry.get("selected_as_formal")):
        return "formal"
    if bool(entry.get("selected_as_primary")):
        return "primary"
    if bool(entry.get("formal_candidate")):
        return "candidate"
    return str(entry.get("reportability_status") or "route")


def _draw_trend_plot(
    canvas: np.ndarray,
    *,
    series: pd.DataFrame,
    frame_number: int,
    route_specs: list[_RouteSpec],
    origin_xy: tuple[int, int],
    size_xy: tuple[int, int],
) -> None:
    ox, oy = origin_xy
    width, height = size_xy
    x1 = ox + width
    y1 = oy + height
    cv2.rectangle(canvas, (ox, oy), (x1, y1), (248, 248, 248), -1)
    cv2.rectangle(canvas, (ox, oy), (x1, y1), PANEL_BORDER, 1)
    cv2.putText(canvas, "A/B/C trend", (ox + 10, oy + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)

    plot_x0 = ox + 12
    plot_y0 = oy + 32
    plot_w = width - 24
    plot_h = height - 44
    cv2.rectangle(canvas, (plot_x0, plot_y0), (plot_x0 + plot_w, plot_y0 + plot_h), (255, 255, 255), -1)
    cv2.rectangle(canvas, (plot_x0, plot_y0), (plot_x0 + plot_w, plot_y0 + plot_h), (210, 210, 210), 1)

    for frac in (0.0, 0.5, 1.0):
        py = int(round(float(plot_y0 + plot_h - frac * plot_h)))
        cv2.line(canvas, (plot_x0, py), (plot_x0 + plot_w, py), (236, 236, 236), 1, cv2.LINE_AA)

    if len(series) == 1:
        x_values = np.array([plot_x0 + plot_w // 2], dtype=float)
    else:
        x_values = np.linspace(plot_x0, plot_x0 + plot_w, len(series), endpoint=True)

    current_idx = int(np.flatnonzero(series["frame"].to_numpy(dtype=int) == frame_number)[0])
    for spec in route_specs:
        y_values = _plot_values_for_route(series, spec)
        if y_values is None:
            continue
        points: list[list[int]] = []
        for idx, value in enumerate(y_values):
            if not np.isfinite(value):
                continue
            px = int(round(float(x_values[idx])))
            py = int(round(float(plot_y0 + plot_h - np.clip(value, 0.0, 1.0) * plot_h)))
            points.append([px, py])
        if len(points) >= 2:
            cv2.polylines(
                canvas,
                [np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)],
                False,
                spec.color_bgr,
                2,
                cv2.LINE_AA,
            )
        if np.isfinite(y_values[current_idx]):
            px = int(round(float(x_values[current_idx])))
            py = int(round(float(plot_y0 + plot_h - np.clip(y_values[current_idx], 0.0, 1.0) * plot_h)))
            cv2.circle(canvas, (px, py), 4, spec.color_bgr, -1)
            cv2.putText(canvas, spec.alias, (px + 6, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, spec.color_bgr, 1, cv2.LINE_AA)

    current_px = int(round(float(x_values[current_idx])))
    cv2.line(canvas, (current_px, plot_y0), (current_px, plot_y0 + plot_h), (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(canvas, "0", (plot_x0 - 10, plot_y0 + plot_h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, MUTED_TEXT, 1, cv2.LINE_AA)
    cv2.putText(canvas, "1", (plot_x0 - 10, plot_y0 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, MUTED_TEXT, 1, cv2.LINE_AA)


def _plot_values_for_route(series: pd.DataFrame, spec: _RouteSpec) -> np.ndarray | None:
    for col in spec.recovery_cols:
        values = pd.to_numeric(series[col], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(values).any():
            return np.clip(values, 0.0, 1.0)
    for col in spec.value_cols:
        values = pd.to_numeric(series[col], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) < 2:
            continue
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if hi - lo <= 1e-9:
            normalized = np.full_like(values, 0.5, dtype=float)
            normalized[~np.isfinite(values)] = np.nan
            return normalized
        return (values - lo) / (hi - lo)
    return None


def _draw_text_panel(
    canvas: np.ndarray,
    lines: list[str],
    *,
    origin_xy: tuple[int, int],
    width: int,
) -> None:
    ox, oy = origin_xy
    line_h = 24
    height = line_h * len(lines) + 18
    cv2.rectangle(canvas, (ox - 8, oy - 24), (ox + width, oy - 24 + height), (245, 245, 245), -1)
    cv2.rectangle(canvas, (ox - 8, oy - 24), (ox + width, oy - 24 + height), PANEL_BORDER, 1)
    for idx, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (ox, oy + idx * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )


def _draw_color_lines_panel(
    canvas: np.ndarray,
    lines: list[tuple[str, tuple[int, int, int], str]],
    *,
    origin_xy: tuple[int, int],
    width: int,
    title: str,
) -> None:
    ox, oy = origin_xy
    line_h = 28
    height = line_h * len(lines) + 44
    cv2.rectangle(canvas, (ox - 8, oy - 24), (ox + width, oy - 24 + height), (245, 245, 245), -1)
    cv2.rectangle(canvas, (ox - 8, oy - 24), (ox + width, oy - 24 + height), PANEL_BORDER, 1)
    cv2.putText(canvas, title, (ox, oy), cv2.FONT_HERSHEY_SIMPLEX, 0.56, TEXT_COLOR, 1, cv2.LINE_AA)
    for idx, (alias, color, text) in enumerate(lines, start=1):
        y = oy + idx * line_h
        cv2.rectangle(canvas, (ox, y - 13), (ox + 12, y - 1), color, -1)
        cv2.putText(canvas, f"{alias}", (ox + 18, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.54, color, 1, cv2.LINE_AA)
        cv2.putText(canvas, text, (ox + 42, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)


def _draw_header_badge(frame_bgr: np.ndarray, lines: list[str]) -> None:
    _draw_box_lines(
        frame_bgr,
        lines,
        origin_xy=(18, 34),
        width=min(640, frame_bgr.shape[1] - 36),
        font_scale=0.56,
    )


def _draw_legend_box(frame_bgr: np.ndarray, entries: list[tuple[str, str, tuple[int, int, int]]]) -> None:
    ox = 18
    oy = frame_bgr.shape[0] - 116
    width = 320
    line_h = 24
    height = line_h * len(entries) + 18
    cv2.rectangle(frame_bgr, (ox - 8, oy - 24), (ox + width, oy - 24 + height), (245, 245, 245), -1)
    cv2.rectangle(frame_bgr, (ox - 8, oy - 24), (ox + width, oy - 24 + height), PANEL_BORDER, 1)
    for idx, (alias, label, color) in enumerate(entries):
        y = oy + idx * line_h
        cv2.rectangle(frame_bgr, (ox, y - 12), (ox + 12, y), color, -1)
        cv2.putText(frame_bgr, f"{alias}", (ox + 18, y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.54, color, 1, cv2.LINE_AA)
        cv2.putText(frame_bgr, label, (ox + 42, y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.52, TEXT_COLOR, 1, cv2.LINE_AA)


def _draw_box_lines(
    canvas: np.ndarray,
    lines: list[str],
    *,
    origin_xy: tuple[int, int],
    width: int,
    font_scale: float,
) -> None:
    ox, oy = origin_xy
    line_h = max(int(round(26 * font_scale / 0.56)), 22)
    height = line_h * len(lines) + 18
    cv2.rectangle(canvas, (ox - 8, oy - 24), (ox + width, oy - 24 + height), (245, 245, 245), -1)
    cv2.rectangle(canvas, (ox - 8, oy - 24), (ox + width, oy - 24 + height), PANEL_BORDER, 1)
    for idx, line in enumerate(lines):
        color = ERROR_COLOR if idx == len(lines) - 1 and "warning:" in line else TEXT_COLOR
        cv2.putText(canvas, line, (ox, oy + idx * line_h), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)


def _draw_segment(
    frame_bgr: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    color_bgr: tuple[int, int, int],
    *,
    thickness: int,
) -> None:
    if _point_finite(p0) and _point_finite(p1):
        cv2.line(frame_bgr, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), color_bgr, thickness, cv2.LINE_AA)


def _draw_point(frame_bgr: np.ndarray, point_xy: np.ndarray, color_bgr: tuple[int, int, int], *, radius: int) -> None:
    if _point_finite(point_xy):
        cv2.circle(frame_bgr, tuple(np.round(point_xy).astype(int)), radius, color_bgr, -1)


def _points_as_polyline(points_xy: np.ndarray) -> np.ndarray | None:
    points = np.asarray(points_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        return None
    valid = np.all(np.isfinite(points), axis=1)
    if np.count_nonzero(valid) < 2:
        return None
    return np.round(points[valid]).astype(np.int32).reshape(-1, 1, 2)


def _max_width_segment(segments_xy: np.ndarray) -> np.ndarray | None:
    segments = np.asarray(segments_xy, dtype=float)
    if segments.ndim != 3 or segments.shape[1:] != (2, 2):
        return None
    valid_lengths = np.linalg.norm(segments[:, 1, :] - segments[:, 0, :], axis=1)
    valid = np.isfinite(valid_lengths)
    if not np.any(valid):
        return None
    return segments[int(np.nanargmax(np.where(valid, valid_lengths, np.nan)))]


def _row_value(row: Any, name: str) -> float:
    value = getattr(row, name, np.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _first_row_value(row: Any, columns: tuple[str, ...]) -> float:
    for col in columns:
        value = _row_value(row, col)
        if np.isfinite(value):
            return value
    return np.nan


def _has_row_values(row: Any, *columns: str) -> bool:
    return all(np.isfinite(_row_value(row, col)) for col in columns)


def _point_finite(point_xy: np.ndarray) -> bool:
    point = np.asarray(point_xy, dtype=float)
    return point.shape == (2,) and np.all(np.isfinite(point))


def _format_float(value: float, *, precision: int = 2, suffix: str = "") -> str:
    if not np.isfinite(value):
        return f"n/a{suffix}"
    return f"{value:.{precision}f}{suffix}"


def _format_metric_value(metric_key: str, value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    if "area" in metric_key:
        return f"{value:.0f}px2"
    if "kappa" in metric_key:
        return f"{value:.5f}"
    return f"{value:.1f}px"
