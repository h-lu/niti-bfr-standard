from __future__ import annotations

from dataclasses import dataclass
import heapq

import cv2
import numpy as np
from skimage.morphology import skeletonize


@dataclass
class ExtractionConfig:
    roi_xyxy: tuple[int, int, int, int] = (220, 0, 760, 560)
    blur_ksize: int = 5
    threshold_dark: int = 100
    open_kernel: int = 3
    close_kernel: int = 5
    min_component_area: int = 80
    anchor_top_band_px: int = 24
    fit_bin_px: float = 3.0
    fit_margin_prefer_quadratic: float = 0.05
    anchor_prior_xy: tuple[float, float] | None = None
    anchor_prior_weight: float = 0.0
    route_a_tip_cluster_radius_px: float = 6.0
    fit_path_fraction: float = 0.72


@dataclass
class ExtractionResult:
    route_a_anchor_xy: np.ndarray
    route_a_tip_xy: np.ndarray
    x_route_a_px: float
    anchor_xy: np.ndarray
    tip_xy: np.ndarray
    x_px: float
    x_fit_px: float
    kappa_fit_px_inv: float
    mask: np.ndarray
    contour_xy: np.ndarray
    sampled_centerline_xy: np.ndarray
    fit_samples_xy: np.ndarray
    fitted_curve_xy: np.ndarray
    quality: float
    model_name: str
    quadratic_rmse_px: float
    circle_rmse_px: float


def _largest_component(mask: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    best_label = 0
    best_area = 0
    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > best_area:
            best_label = label
            best_area = area
    return (labels == best_label).astype(np.uint8) * 255


def _component_contour(component: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("needle contour not found")
    contour = max(contours, key=cv2.contourArea)
    return contour[:, 0, :].astype(float)


def _estimate_tangent(contour_xy: np.ndarray, config: ExtractionConfig) -> np.ndarray:
    top_band = contour_xy[:, 1] <= contour_xy[:, 1].min() + config.anchor_top_band_px
    top_pts = contour_xy[top_band] if np.any(top_band) else contour_xy
    centered = top_pts - top_pts.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    tangent = vh[0]
    if tangent[1] < 0:
        tangent = -tangent
    return tangent / max(np.linalg.norm(tangent), 1e-9)


def _anchor_reference(contour_xy: np.ndarray, config: ExtractionConfig, roi_offset_xy: np.ndarray) -> np.ndarray:
    top_band = contour_xy[:, 1] <= contour_xy[:, 1].min() + config.anchor_top_band_px
    top_pts = contour_xy[top_band] if np.any(top_band) else contour_xy
    anchor_ref = top_pts.mean(axis=0)
    if config.anchor_prior_xy is not None and config.anchor_prior_weight > 0.0:
        prior_local = np.asarray(config.anchor_prior_xy, dtype=float) - roi_offset_xy
        w = float(np.clip(config.anchor_prior_weight, 0.0, 1.0))
        anchor_ref = (1.0 - w) * anchor_ref + w * prior_local
    return anchor_ref


def _neighbor_offsets() -> list[tuple[int, int, float]]:
    offsets: list[tuple[int, int, float]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            offsets.append((dr, dc, float(np.hypot(dr, dc))))
    return offsets


NEIGHBOR_OFFSETS = _neighbor_offsets()


def _extract_skeleton_path_local(
    component: np.ndarray,
    anchor_reference_local: np.ndarray,
) -> np.ndarray:
    skeleton = skeletonize(component > 0)
    if np.count_nonzero(skeleton) < 2:
        raise RuntimeError("needle skeleton too short")
    rows, cols = np.where(skeleton)
    pixels_rc = np.column_stack([rows, cols]).astype(int)
    pixels_xy = np.column_stack([cols, rows]).astype(float)
    if len(pixels_rc) < 2:
        raise RuntimeError("needle skeleton too short")

    coord_to_index = {(int(r), int(c)): idx for idx, (r, c) in enumerate(pixels_rc)}
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(len(pixels_rc))]
    degrees = np.zeros(len(pixels_rc), dtype=int)
    for idx, (r, c) in enumerate(pixels_rc):
        for dr, dc, weight in NEIGHBOR_OFFSETS:
            key = (int(r + dr), int(c + dc))
            j = coord_to_index.get(key)
            if j is None:
                continue
            adjacency[idx].append((j, weight))
        degrees[idx] = len(adjacency[idx])

    endpoint_indices = np.where(degrees <= 1)[0]
    if len(endpoint_indices) == 0:
        endpoint_indices = np.arange(len(pixels_rc))

    anchor_idx = int(np.argmin(np.linalg.norm(pixels_xy - anchor_reference_local[None, :], axis=1)))
    dist = np.full(len(pixels_rc), np.inf, dtype=float)
    parent = np.full(len(pixels_rc), -1, dtype=int)
    dist[anchor_idx] = 0.0
    heap: list[tuple[float, int]] = [(0.0, anchor_idx)]
    while heap:
        d, idx = heapq.heappop(heap)
        if d > dist[idx] + 1e-9:
            continue
        for j, weight in adjacency[idx]:
            cand = d + weight
            if cand + 1e-9 < dist[j]:
                dist[j] = cand
                parent[j] = idx
                heapq.heappush(heap, (cand, j))

    finite_endpoints = endpoint_indices[np.isfinite(dist[endpoint_indices])]
    if len(finite_endpoints) == 0:
        raise RuntimeError("needle skeleton endpoints unreachable")
    tip_idx = int(finite_endpoints[np.argmax(dist[finite_endpoints])])

    path_indices = [tip_idx]
    while path_indices[-1] != anchor_idx:
        parent_idx = int(parent[path_indices[-1]])
        if parent_idx < 0:
            raise RuntimeError("failed to reconstruct skeleton path")
        path_indices.append(parent_idx)
    path_indices.reverse()
    path_local = pixels_xy[np.asarray(path_indices, dtype=int)]
    return path_local


def _refine_endpoint_from_contour(contour_local: np.ndarray, seed_local: np.ndarray, radius_px: float) -> np.ndarray:
    distances = np.linalg.norm(contour_local - seed_local[None, :], axis=1)
    nearby = contour_local[distances <= radius_px]
    if len(nearby) == 0:
        nearest_idx = int(np.argmin(distances))
        nearby = contour_local[[nearest_idx]]
    return nearby.mean(axis=0)


def _extract_route_a_endpoints(
    contour_local: np.ndarray,
    config: ExtractionConfig,
    roi_offset_xy: np.ndarray,
    skeleton_path_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    anchor_local = _refine_endpoint_from_contour(
        contour_local,
        _anchor_reference(contour_local, config, roi_offset_xy),
        config.route_a_tip_cluster_radius_px,
    )
    tip_local = _refine_endpoint_from_contour(
        contour_local,
        skeleton_path_local[-1],
        config.route_a_tip_cluster_radius_px,
    )
    anchor_global = anchor_local + roi_offset_xy
    tip_global = tip_local + roi_offset_xy
    x_route_a = float(np.linalg.norm(tip_global - anchor_global))
    return anchor_global, tip_global, x_route_a


def _sample_centerline_from_path(
    skeleton_path_global: np.ndarray,
    anchor_xy: np.ndarray,
    tangent: np.ndarray,
    config: ExtractionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    normal = np.array([-tangent[1], tangent[0]], dtype=float)
    rel = skeleton_path_global - anchor_xy[None, :]
    local_u = rel @ normal
    local_v = rel @ tangent
    local_path = np.column_stack([local_u, local_v])

    deltas = np.diff(skeleton_path_global, axis=0)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(deltas, axis=1))])
    total_arc = float(arc[-1])
    if total_arc < 10.0:
        raise RuntimeError("needle extent too short for fitting")
    fit_arc_max = max(total_arc * config.fit_path_fraction, 30.0)
    fit_arc_max = min(fit_arc_max, total_arc)
    n_samples = max(int(np.ceil(fit_arc_max / max(config.fit_bin_px, 1e-6))) + 1, 10)
    target_arc = np.linspace(0.0, fit_arc_max, n_samples)
    sampled_u = np.interp(target_arc, arc, local_path[:, 0])
    sampled_v = np.interp(target_arc, arc, local_path[:, 1])
    centerline_local = np.column_stack([sampled_u, sampled_v])
    keep = centerline_local[:, 1] >= centerline_local[:, 1].min() - 1.0
    centerline_local = centerline_local[keep]
    if len(centerline_local) < 6:
        raise RuntimeError("not enough centerline samples for fitting")
    order = np.argsort(centerline_local[:, 1])
    centerline_local = centerline_local[order]
    if len(centerline_local) >= 8:
        coeffs = np.polyfit(centerline_local[:, 1], centerline_local[:, 0], deg=2)
        residual = centerline_local[:, 0] - np.polyval(coeffs, centerline_local[:, 1])
        keep = np.abs(residual) <= 2.5
        if np.count_nonzero(keep) >= 6:
            centerline_local = centerline_local[keep]
    centerline_local = centerline_local - centerline_local[0]
    centers_global = _local_to_global(centerline_local, anchor_xy, tangent)
    return centerline_local, centers_global


def _fit_quadratic(centerline_local: np.ndarray) -> tuple[np.ndarray, float, float]:
    v = centerline_local[:, 1]
    u = centerline_local[:, 0]
    coeffs = np.polyfit(v, u, deg=2)
    u_pred = np.polyval(coeffs, v)
    rmse = float(np.sqrt(np.mean((u_pred - u) ** 2)))
    v_grid = np.linspace(float(v.min()), float(v.max()), 120)
    u_grid = np.polyval(coeffs, v_grid)
    curve = np.column_stack([u_grid, v_grid])
    du = 2.0 * coeffs[0] * v_grid + coeffs[1]
    ddu = np.full_like(v_grid, 2.0 * coeffs[0])
    curvature = np.abs(ddu) / np.maximum((1.0 + du**2) ** 1.5, 1e-9)
    kappa = float(np.mean(curvature))
    return curve, rmse, kappa


def _fit_circle(centerline_local: np.ndarray) -> tuple[np.ndarray, float, float]:
    pts = centerline_local
    a = np.column_stack([2.0 * pts[:, 0], 2.0 * pts[:, 1], np.ones(len(pts))])
    b = pts[:, 0] ** 2 + pts[:, 1] ** 2
    params, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    uc, vc, c0 = params
    radius_sq = c0 + uc**2 + vc**2
    if radius_sq <= 1e-9:
        raise RuntimeError("invalid circle fit")
    radius = float(np.sqrt(radius_sq))
    residual = np.sqrt((pts[:, 0] - uc) ** 2 + (pts[:, 1] - vc) ** 2) - radius
    rmse = float(np.sqrt(np.mean(residual**2)))

    theta = np.unwrap(np.arctan2(pts[:, 1] - vc, pts[:, 0] - uc))
    theta_grid = np.linspace(float(theta[0]), float(theta[-1]), 120)
    curve = np.column_stack([uc + radius * np.cos(theta_grid), vc + radius * np.sin(theta_grid)])
    kappa = 1.0 / radius
    return curve, rmse, float(kappa)


def _local_to_global(points_local: np.ndarray, anchor_xy: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    normal = np.array([-tangent[1], tangent[0]], dtype=float)
    return anchor_xy[None, :] + points_local[:, [0]] * normal[None, :] + points_local[:, [1]] * tangent[None, :]


def extract_geometry(frame_bgr: np.ndarray, config: ExtractionConfig) -> ExtractionResult:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    x0, y0, x1, y1 = config.roi_xyxy
    crop = gray[y0:y1, x0:x1]
    blur = cv2.GaussianBlur(crop, (config.blur_ksize, config.blur_ksize), 0)
    mask = (blur < config.threshold_dark).astype(np.uint8) * 255
    open_kernel = np.ones((config.open_kernel, config.open_kernel), np.uint8)
    close_kernel = np.ones((config.close_kernel, config.close_kernel), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    component = _largest_component(mask)
    if int(np.count_nonzero(component)) < config.min_component_area:
        raise RuntimeError("needle component not found")

    contour_local = _component_contour(component)
    contour_global = contour_local + np.array([x0, y0], dtype=float)
    tangent = _estimate_tangent(contour_local, config)
    roi_offset_xy = np.array([x0, y0], dtype=float)
    anchor_reference_local = _anchor_reference(contour_local, config, roi_offset_xy)
    skeleton_path_local = _extract_skeleton_path_local(component, anchor_reference_local)
    skeleton_path_global = skeleton_path_local + roi_offset_xy[None, :]
    route_a_anchor_global, route_a_tip_global, x_route_a = _extract_route_a_endpoints(
        contour_local,
        config,
        roi_offset_xy,
        skeleton_path_local,
    )
    anchor_global = skeleton_path_global[0]
    tip_global = skeleton_path_global[-1]
    centerline_local, centerline_global = _sample_centerline_from_path(
        skeleton_path_global,
        anchor_global,
        tangent,
        config,
    )
    fit_anchor_global = centerline_global[0]

    centerline_local = centerline_local - centerline_local[0]
    fit_samples_global = _local_to_global(centerline_local, fit_anchor_global, tangent)

    quad_curve_local, quad_rmse, quad_kappa = _fit_quadratic(centerline_local)
    circle_curve_local, circle_rmse, circle_kappa = _fit_circle(centerline_local)

    choose_quadratic = quad_rmse <= circle_rmse * (1.0 + config.fit_margin_prefer_quadratic)
    if choose_quadratic:
        fitted_curve_local = quad_curve_local
        model_name = "quadratic"
        kappa_fit = quad_kappa
        selected_rmse = quad_rmse
    else:
        fitted_curve_local = circle_curve_local
        model_name = "circle"
        kappa_fit = circle_kappa
        selected_rmse = circle_rmse

    fitted_curve_global = _local_to_global(fitted_curve_local, fit_anchor_global, tangent)
    x_fit = float(np.linalg.norm(tip_global - anchor_global))
    v_span = max(float(centerline_local[-1, 1] - centerline_local[0, 1]), 1.0)
    coverage = min(len(centerline_local) * config.fit_bin_px / v_span, 1.0)
    quality = float(np.exp(-selected_rmse / 4.0) * coverage)

    return ExtractionResult(
        route_a_anchor_xy=route_a_anchor_global,
        route_a_tip_xy=route_a_tip_global,
        x_route_a_px=x_route_a,
        anchor_xy=anchor_global,
        tip_xy=tip_global,
        x_px=x_fit,
        x_fit_px=x_fit,
        kappa_fit_px_inv=float(kappa_fit),
        mask=component,
        contour_xy=contour_global,
        sampled_centerline_xy=skeleton_path_global,
        fit_samples_xy=fit_samples_global,
        fitted_curve_xy=fitted_curve_global,
        quality=quality,
        model_name=model_name,
        quadratic_rmse_px=float(quad_rmse),
        circle_rmse_px=float(circle_rmse),
    )


def extract_endpoints(frame_bgr: np.ndarray, config: ExtractionConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = extract_geometry(frame_bgr, config)
    return result.anchor_xy, result.tip_xy, result.mask


def extract_x_metric(frame_bgr: np.ndarray, config: ExtractionConfig) -> float:
    return extract_geometry(frame_bgr, config).x_fit_px
