from __future__ import annotations

import csv
from collections import deque
from pathlib import Path
import sys

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from niti_bfr.synth_braided import truth_metrics_from_body_mask


SOURCE = Path(
    "/Users/wangxq/Library/Containers/com.tencent.xinWeChat/Data/Documents/"
    "xwechat_files/wxid_kofevra4tax121_f9ff/temp/RWTemp/2026-04/"
    "f9fe7e136520c635994b1205793d49c7.png"
)
OUT_DIR = Path(__file__).resolve().parent

VIDEO_PATH = OUT_DIR / "niti_heating_20s_with_temp.mp4"
PREVIEW_PATH = OUT_DIR / "niti_heating_20s_preview.png"
CSV_PATH = OUT_DIR / "niti_heating_20s_temperature.csv"
ANALYSIS_CSV_PATH = OUT_DIR / "temperature_for_analysis.csv"
LAST_FRAME_PATH = OUT_DIR / "niti_heating_20s_last_frame.png"

FPS = 20
DURATION_SECONDS = 20
TOTAL_FRAMES = FPS * DURATION_SECONDS

THRESHOLD = 210
PADDING = 36

# Assumed but physically plausible heating window for a thermally activated NiTi
# recovery test. This can be retuned later if the user has measured As/Af.
T_LOW = 25.0
T_HIGH = 70.0
AS_TEMP = 42.0
AF_TEMP = 60.0
MAX_LOCAL_SHORTEN = 0.36
TAPER_THRESHOLD_RATIO = 0.25
COMPACTION_THRESHOLD_RATIO = 0.75


def smoothstep01(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def largest_foreground_component(mask: np.ndarray) -> tuple[int, int, int, int, np.ndarray]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best_pixels: list[tuple[int, int]] = []
    best_bbox = (0, 0, width - 1, height - 1)

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue

            queue = deque([(y, x)])
            visited[y, x] = True
            pixels: list[tuple[int, int]] = []
            touches_border = False

            while queue:
                cy, cx = queue.popleft()
                pixels.append((cy, cx))
                if cy in (0, height - 1) or cx in (0, width - 1):
                    touches_border = True

                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))

            if touches_border or len(pixels) <= len(best_pixels):
                continue

            ys = [py for py, _ in pixels]
            xs = [px for _, px in pixels]
            best_pixels = pixels
            best_bbox = (min(xs), min(ys), max(xs), max(ys))

    component = np.zeros_like(mask, dtype=np.uint8)
    for py, px in best_pixels:
        component[py, px] = 255

    x0, y0, x1, y1 = best_bbox
    return x0, y0, x1, y1, component


def bilinear_sample(arr: np.ndarray, sample_x: np.ndarray, sample_y: np.ndarray, fill: float) -> np.ndarray:
    height, width = arr.shape

    sample_x = np.asarray(sample_x, dtype=np.float32)
    sample_y = np.asarray(sample_y, dtype=np.float32)

    finite = np.isfinite(sample_x) & np.isfinite(sample_y)
    safe_x = np.where(finite, sample_x, 0.0)
    safe_y = np.where(finite, sample_y, 0.0)

    x0 = np.floor(safe_x).astype(np.int32)
    y0 = np.floor(safe_y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    dx = safe_x - x0
    dy = safe_y - y0

    valid = finite & (safe_x >= 0) & (safe_x <= width - 1) & (safe_y >= 0) & (safe_y <= height - 1)

    x0c = np.clip(x0, 0, width - 1)
    x1c = np.clip(x1, 0, width - 1)
    y0c = np.clip(y0, 0, height - 1)
    y1c = np.clip(y1, 0, height - 1)

    top_left = arr[y0c, x0c]
    top_right = arr[y0c, x1c]
    bottom_left = arr[y1c, x0c]
    bottom_right = arr[y1c, x1c]

    top = top_left * (1.0 - dx) + top_right * dx
    bottom = bottom_left * (1.0 - dx) + bottom_right * dx
    sampled = top * (1.0 - dy) + bottom * dy
    return np.where(valid, sampled, fill)


def build_overlay(frame_rgb: np.ndarray, temperature_c: float, fraction: float) -> np.ndarray:
    image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(image)

    box_x, box_y = 16, image.height - 70
    box_w, box_h = 180, 46
    draw.rounded_rectangle((box_x, box_y, box_x + box_w, box_y + box_h), radius=8, fill=(255, 255, 255))
    draw.rectangle((box_x + 10, box_y + 28, box_x + 150, box_y + 36), fill=(228, 228, 228))

    fill_w = int(140 * np.clip((temperature_c - T_LOW) / (T_HIGH - T_LOW), 0.0, 1.0))
    fill_color = (
        int(90 + 145 * np.clip((temperature_c - T_LOW) / (T_HIGH - T_LOW), 0.0, 1.0)),
        int(160 - 90 * np.clip((temperature_c - T_LOW) / (T_HIGH - T_LOW), 0.0, 1.0)),
        60,
    )
    draw.rectangle((box_x + 10, box_y + 28, box_x + 10 + fill_w, box_y + 36), fill=fill_color)

    draw.text((box_x + 10, box_y + 8), f"T = {temperature_c:4.1f} C", fill=(30, 30, 30))
    draw.text((box_x + 112, box_y + 8), f"xA = {fraction:0.2f}", fill=(30, 30, 30))
    return np.array(image)


def projected_mask_metrics(mask: np.ndarray) -> tuple[float, float]:
    binary = np.asarray(mask > 0, dtype=np.uint8)
    pixel_area = float(np.count_nonzero(binary))
    if pixel_area <= 0.0:
        return 0.0, 0.0
    contours, _ = cv2.findContours(binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_area = float(max(cv2.contourArea(contour) for contour in contours)) if contours else pixel_area
    return pixel_area, contour_area


def main() -> None:
    source_image = Image.open(SOURCE).convert("L")
    source = np.array(source_image).astype(np.float32)

    foreground = source < THRESHOLD
    x0, y0, x1, y1, component = largest_foreground_component(foreground)

    x0 = max(0, x0 - PADDING)
    y0 = max(0, y0 - PADDING)
    x1 = min(source.shape[1] - 1, x1 + PADDING)
    y1 = min(source.shape[0] - 1, y1 + PADDING)

    crop = source[y0 : y1 + 1, x0 : x1 + 1]
    component_crop = component[y0 : y1 + 1, x0 : x1 + 1]
    component_crop_float = (component_crop > 0).astype(np.float32)
    component_column_area_px = component_crop_float.sum(axis=0).astype(np.float32)
    component_area_ref_px2 = float(component_column_area_px.sum())

    alpha_img = Image.fromarray(component_crop).filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.GaussianBlur(2.0))
    alpha = np.array(alpha_img).astype(np.float32) / 255.0

    cleanup_mask = np.array(
        Image.fromarray(component).filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(1.7))
    ).astype(np.float32) / 255.0
    bg_value = float(np.quantile(source[source > THRESHOLD], 0.3))
    base_clean = source * (1.0 - cleanup_mask) + bg_value * cleanup_mask

    crop_h, crop_w = crop.shape
    grid_x, grid_y = np.meshgrid(np.arange(crop_w, dtype=np.float32), np.arange(crop_h, dtype=np.float32))
    cy = (crop_h - 1) / 2.0
    x_positions = np.arange(crop_w, dtype=np.float32)
    x_norm = np.clip(x_positions / max(crop_w - 1, 1), 0.0, 1.0)

    axial_envelope = 0.22 + 0.78 * (np.clip(np.sin(np.pi * x_norm), 0.0, 1.0) ** 1.25)
    onset_offsets = 2.3 * np.sin(2.0 * np.pi * x_norm + 0.55) + 1.1 * np.cos(5.0 * np.pi * x_norm - 0.8)
    onset_offsets = np.clip(onset_offsets, -3.4, 3.4)
    width_mod = 0.92 + 0.18 * (0.5 + 0.5 * np.sin(3.0 * np.pi * x_norm - 0.35))

    weights = component_crop.sum(axis=0).astype(np.float32)
    if np.all(weights == 0):
        weights = np.ones_like(x_positions, dtype=np.float32)
    weights /= weights.sum()

    csv_rows: list[list[float]] = []
    preview_frames: list[np.ndarray] = []
    last_frame_rgb = None
    area_proj_ref_true_px2 = None

    writer = None
    try:
        writer = imageio.get_writer(
            VIDEO_PATH,
            fps=FPS,
            codec="libx264",
            format="FFMPEG",
            pixelformat="yuv420p",
            macro_block_size=None,
        )
    except ImportError:
        writer = None

    try:
        for frame_idx in range(TOTAL_FRAMES):
            time_s = frame_idx / FPS
            progress = frame_idx / (TOTAL_FRAMES - 1)
            temperature_c = T_LOW + (T_HIGH - T_LOW) * progress

            if frame_idx == 0:
                global_fraction = 0.0
                frame = source.copy()
                warped_component = component_crop_float >= 0.5
                area_proj_model_true_px2 = component_area_ref_px2
            else:
                reduced_temp = (temperature_c - (AS_TEMP + onset_offsets)) / ((AF_TEMP - AS_TEMP) * width_mod)
                local_fraction = smoothstep01(reduced_temp)
                global_fraction = float(np.sum(local_fraction * weights))

                local_scale_x = 1.0 - MAX_LOCAL_SHORTEN * axial_envelope * local_fraction
                local_scale_x = np.clip(local_scale_x, 0.62, 1.0)

                src_x = x_positions
                out_x = np.zeros_like(src_x)
                out_x[1:] = np.cumsum(0.5 * (local_scale_x[:-1] + local_scale_x[1:]))
                out_x += (crop_w - 1 - out_x[-1]) / 2.0

                sample_x = np.interp(grid_x[0], out_x, src_x, left=np.nan, right=np.nan)
                sample_x_2d = np.broadcast_to(sample_x, (crop_h, crop_w))

                sample_fraction = np.interp(sample_x, src_x, local_fraction, left=0.0, right=0.0)
                sample_envelope = np.interp(sample_x, src_x, axial_envelope, left=axial_envelope[0], right=axial_envelope[-1])
                sample_scale_x = np.interp(sample_x, src_x, local_scale_x, left=1.0, right=1.0)

                local_scale_y = 1.0 + (1.0 / np.sqrt(sample_scale_x) - 1.0) * (0.25 + 0.75 * sample_envelope)
                sample_y = cy + (grid_y - cy) / local_scale_y[None, :]
                local_scale_y_source = 1.0 + (1.0 / np.sqrt(local_scale_x) - 1.0) * (0.25 + 0.75 * axial_envelope)
                area_proj_model_true_px2 = float(np.sum(component_column_area_px * local_scale_x * local_scale_y_source))

                warped_crop = bilinear_sample(crop, sample_x_2d, sample_y, bg_value)
                warped_alpha = bilinear_sample(alpha, sample_x_2d, sample_y, 0.0)
                warped_alpha = np.clip(warped_alpha, 0.0, 1.0)
                warped_component = bilinear_sample(component_crop_float, sample_x_2d, sample_y, 0.0) >= 0.5

                frame = base_clean.copy()
                region = frame[y0 : y1 + 1, x0 : x1 + 1]
                frame[y0 : y1 + 1, x0 : x1 + 1] = region * (1.0 - warped_alpha) + warped_crop * warped_alpha

            warped_body_mask = np.asarray(warped_component, dtype=np.uint8) * 255
            truth_metrics = truth_metrics_from_body_mask(
                warped_body_mask,
                taper_threshold_ratio=TAPER_THRESHOLD_RATIO,
                compaction_threshold_ratio=COMPACTION_THRESHOLD_RATIO,
                width_sampling_step_px=4.0,
                centerline_smooth_window=7,
            )
            if area_proj_ref_true_px2 is None:
                area_proj_ref_true_px2 = float(truth_metrics["area_proj_true_px2"])
            area_proj_true_norm = float(truth_metrics["area_proj_true_px2"] / max(area_proj_ref_true_px2, 1e-9))

            frame_component = np.zeros_like(component, dtype=np.uint8)
            frame_component[y0 : y1 + 1, x0 : x1 + 1] = np.asarray(warped_component, dtype=np.uint8)
            area_proj_raster_true_px2, area_proj_contour_true_px2 = projected_mask_metrics(frame_component)

            frame_rgb = np.repeat(np.clip(frame, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
            frame_rgb = build_overlay(frame_rgb, temperature_c, global_fraction)

            if frame_rgb.shape[1] % 2 != 0:
                frame_rgb = np.pad(frame_rgb, ((0, 0), (0, 1), (0, 0)), mode="edge")
            if frame_rgb.shape[0] % 2 != 0:
                frame_rgb = np.pad(frame_rgb, ((0, 1), (0, 0), (0, 0)), mode="edge")

            if writer is not None:
                writer.append_data(frame_rgb)

            csv_rows.append(
                [
                    frame_idx,
                    round(time_s, 4),
                    round(temperature_c, 4),
                    round(global_fraction, 6),
                    round(float(truth_metrics["length_env_true_px"]), 4),
                    round(float(truth_metrics["length_axis_true_px"]), 4),
                    round(float(truth_metrics["length_axis_alt_true_px"]), 4),
                    round(float(truth_metrics["diameter_true_px"]), 4),
                    round(float(truth_metrics["diameter_max_feret_true_px"]), 4),
                    round(float(truth_metrics["area_proj_true_px2"]), 4),
                    round(area_proj_true_norm, 6),
                    round(area_proj_model_true_px2, 4),
                    round(area_proj_raster_true_px2, 4),
                    round(float(truth_metrics["area_proj_contour_true_px2"]), 4),
                ]
            )

            if frame_idx in (0, TOTAL_FRAMES // 2, TOTAL_FRAMES - 1):
                preview_frames.append(frame_rgb)
            if frame_idx == TOTAL_FRAMES - 1:
                last_frame_rgb = frame_rgb
    finally:
        if writer is not None:
            writer.close()

    with CSV_PATH.open("w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(
            [
                "frame",
                "time_s",
                "temperature_C",
                "austenite_fraction",
                "length_env_true_px",
                "length_axis_true_px",
                "length_axis_alt_true_px",
                "diameter_true_px",
                "diameter_max_feret_true_px",
                "area_proj_true_px2",
                "area_proj_true_norm",
                "area_proj_model_true_px2",
                "area_proj_raster_true_px2",
                "area_proj_contour_true_px2",
            ]
        )
        writer_csv.writerows(csv_rows)

    with ANALYSIS_CSV_PATH.open("w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(
            [
                "frame",
                "time_sec",
                "temperature_c",
                "austenite_fraction",
                "length_env_true_px",
                "length_axis_true_px",
                "length_axis_alt_true_px",
                "diameter_true_px",
                "diameter_max_feret_true_px",
                "area_proj_true_px2",
                "area_proj_true_norm",
                "area_proj_model_true_px2",
                "area_proj_raster_true_px2",
                "area_proj_contour_true_px2",
            ]
        )
        writer_csv.writerows(csv_rows)

    if preview_frames:
        separator = np.full((preview_frames[0].shape[0], 14, 3), 255, dtype=np.uint8)
        montage = preview_frames[0]
        for extra in preview_frames[1:]:
            montage = np.concatenate([montage, separator, extra], axis=1)
        Image.fromarray(montage).save(PREVIEW_PATH)

    if last_frame_rgb is not None:
        Image.fromarray(last_frame_rgb).save(LAST_FRAME_PATH)

    af95_temp = next((row[2] for row in csv_rows if row[3] >= 0.95), csv_rows[-1][2])
    if writer is not None:
        print(f"Saved video: {VIDEO_PATH}")
    else:
        print("Skipped video export: imageio FFMPEG plugin not available")
    print(f"Saved preview: {PREVIEW_PATH}")
    print(f"Saved csv: {CSV_PATH}")
    print(f"Saved analysis csv: {ANALYSIS_CSV_PATH}")
    print(f"Saved last frame: {LAST_FRAME_PATH}")
    print(f"Model As/Af: {AS_TEMP:.1f} C / {AF_TEMP:.1f} C")
    print(f"Model Af95: {af95_temp:.2f} C")


if __name__ == "__main__":
    main()
