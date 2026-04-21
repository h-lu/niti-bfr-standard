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
    component_bridge_kernel: int = 11
    min_component_area: int = 80
    anchor_top_band_px: int = 24
    fit_bin_px: float = 3.0
    fit_margin_prefer_quadratic: float = 0.05
    anchor_prior_xy: tuple[float, float] | None = None
    anchor_prior_weight: float = 0.0
    route_a_tip_cluster_radius_px: float = 6.0
    route_b_endpoint_extension_scale: float = 0.0
    route_b_cap_inset_scale: float = 0.08
    fit_path_fraction: float = 0.72
    fit_path_fraction_min: float = 0.42
    fit_curvature_threshold_ratio: float = 0.28


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


def _component_anchor_seed_local(
    mask: np.ndarray,
    config: ExtractionConfig,
    roi_offset_xy: np.ndarray | None,
) -> np.ndarray:
    rows, cols = np.where(mask > 0)
    if len(rows) == 0:
        raise RuntimeError("needle component not found")
    points = np.column_stack([cols, rows]).astype(float)
    top_band = points[:, 1] <= points[:, 1].min() + config.anchor_top_band_px
    top_points = points[top_band] if np.any(top_band) else points
    seed = top_points.mean(axis=0)
    if config.anchor_prior_xy is not None and roi_offset_xy is not None and config.anchor_prior_weight > 0.0:
        local_prior = np.asarray(config.anchor_prior_xy, dtype=float) - roi_offset_xy
        weight = float(np.clip(config.anchor_prior_weight, 0.0, 1.0))
        seed = (1.0 - weight) * seed + weight * local_prior
    return seed


def _merge_distal_fragments(
    mask: np.ndarray,
    config: ExtractionConfig,
    roi_offset_xy: np.ndarray | None,
) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num <= 2:
        return _largest_component(mask)

    seed = _component_anchor_seed_local(mask, config, roi_offset_xy)
    component_points: dict[int, np.ndarray] = {}
    component_centroids: dict[int, np.ndarray] = {}
    component_distances: dict[int, np.ndarray] = {}
    best_label = 0
    best_distance = float("inf")
    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(1, int(config.min_component_area)):
            continue
        rows, cols = np.where(labels == label)
        if len(rows) == 0:
            continue
        points = np.column_stack([cols, rows]).astype(float)
        component_points[label] = points
        component_centroids[label] = points.mean(axis=0)
        distances = np.linalg.norm(points - seed[None, :], axis=1)
        component_distances[label] = distances
        label_distance = float(np.min(distances))
        if label_distance < best_distance:
            best_distance = label_distance
            best_label = label

    if best_label == 0:
        return _largest_component(mask)

    main_points = component_points[best_label]
    farthest_point = main_points[int(np.argmax(component_distances[best_label]))]
    distal_direction = farthest_point - seed
    distal_norm = float(np.linalg.norm(distal_direction))
    if distal_norm < 1e-6:
        return (labels == best_label).astype(np.uint8) * 255
    distal_direction /= distal_norm
    normal = np.array([-distal_direction[1], distal_direction[0]], dtype=float)
    main_along = (main_points - seed[None, :]) @ distal_direction
    main_along_max = float(np.max(main_along))

    combined = np.zeros_like(mask)
    combined[labels == best_label] = 255
    join_distance_px = max(24.0, 2.0 * float(max(config.close_kernel, config.component_bridge_kernel)))
    lateral_limit_px = max(24.0, 0.08 * max(main_along_max, 1.0))
    distal_area_floor = max(24, int(config.min_component_area * 0.2))
    bridge_thickness = max(1, int(np.ceil(max(config.close_kernel, 1) / 2.0)))

    for label, points in component_points.items():
        if label == best_label:
            continue
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < distal_area_floor:
            continue
        centroid = component_centroids[label]
        rel_centroid = centroid - seed
        along = float(rel_centroid @ distal_direction)
        lateral = float(abs(rel_centroid @ normal))
        if along <= main_along_max:
            continue
        if lateral > lateral_limit_px:
            continue
        deltas = main_points[:, None, :] - points[None, :, :]
        distances = np.linalg.norm(deltas, axis=2)
        best_pair = np.unravel_index(int(np.argmin(distances)), distances.shape)
        min_distance = float(distances[best_pair])
        if min_distance > join_distance_px:
            continue
        combined[labels == label] = 255
        main_bridge_point = tuple(np.round(main_points[best_pair[0]]).astype(int))
        fragment_bridge_point = tuple(np.round(points[best_pair[1]]).astype(int))
        cv2.line(combined, main_bridge_point, fragment_bridge_point, 255, thickness=bridge_thickness, lineType=cv2.LINE_8)

    return combined if int(np.count_nonzero(combined)) else _largest_component(mask)


def _component_mask(
    mask: np.ndarray,
    config: ExtractionConfig,
    roi_offset_xy: np.ndarray | None = None,
) -> np.ndarray:
    component_mask = mask
    # A wire can split into two islands for a frame or two when the distal straight segment
    # drops slightly below the dark-threshold response. Use a stronger close only for component
    # selection so we reconnect small gaps without changing the downstream contour definition.
    component_bridge_kernel = max(int(config.close_kernel), int(config.component_bridge_kernel), 1)
    if component_bridge_kernel % 2 == 0:
        component_bridge_kernel += 1
    if component_bridge_kernel > max(int(config.close_kernel), 1):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (component_bridge_kernel, component_bridge_kernel))
        component_mask = cv2.morphologyEx(component_mask, cv2.MORPH_CLOSE, kernel)
    return _merge_distal_fragments(component_mask, config, roi_offset_xy)


def _component_contour(component: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("needle contour not found")
    contour = max(contours, key=cv2.contourArea)
    return contour[:, 0, :].astype(float)


def _convex_hull_xy(points_xy: np.ndarray) -> np.ndarray:
    hull = cv2.convexHull(np.round(points_xy).astype(np.float32))
    return hull[:, 0, :].astype(float)


def _estimate_tangent(contour_xy: np.ndarray, config: ExtractionConfig) -> np.ndarray:
    top_band = contour_xy[:, 1] <= contour_xy[:, 1].min() + config.anchor_top_band_px
    top_pts = contour_xy[top_band] if np.any(top_band) else contour_xy
    centered = top_pts - top_pts.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    tangent = vh[0]
    if tangent[1] < 0:
        tangent = -tangent
    return tangent / max(np.linalg.norm(tangent), 1e-9)


def _estimate_path_tangent(path_xy: np.ndarray) -> np.ndarray:
    if len(path_xy) < 2:
        raise RuntimeError("path too short to estimate tangent")
    n = min(max(len(path_xy) // 6, 6), len(path_xy))
    pts = path_xy[:n]
    centered = pts - pts.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    tangent = vh[0]
    if tangent[1] < 0:
        tangent = -tangent
    return tangent / max(np.linalg.norm(tangent), 1e-9)


def _terminal_direction(path_xy: np.ndarray, head: bool, window: int = 8) -> np.ndarray:
    if len(path_xy) < 2:
        raise RuntimeError("path too short to estimate terminal direction")
    n = min(max(window, 2), len(path_xy))
    pts = path_xy[:n] if head else path_xy[-n:]
    start = pts[0]
    end = pts[-1]
    direction = end - start
    if np.linalg.norm(direction) < 1e-9 and len(pts) >= 2:
        direction = pts[-1] - pts[0]
    if not head:
        return _unit_direction(direction)
    return _unit_direction(direction)


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


def _contour_apex_along_direction(
    contour_local: np.ndarray,
    seed_local: np.ndarray,
    direction_local: np.ndarray,
    radius_px: float,
) -> np.ndarray:
    direction = _unit_direction(direction_local)
    normal = np.array([-direction[1], direction[0]], dtype=float)
    rel = contour_local - seed_local[None, :]
    along = rel @ direction
    lateral = np.abs(rel @ normal)
    lateral_limit = max(3.0 * radius_px, 8.0)
    candidates = contour_local[lateral <= lateral_limit]
    candidate_along = along[lateral <= lateral_limit]
    if len(candidates) == 0:
        candidates = contour_local
        candidate_along = along
    best = candidates[int(np.argmax(candidate_along))]
    return best


def _contour_cap_center_along_direction(
    contour_local: np.ndarray,
    seed_local: np.ndarray,
    direction_local: np.ndarray,
    radius_px: float,
) -> np.ndarray:
    direction = _unit_direction(direction_local)
    normal = np.array([-direction[1], direction[0]], dtype=float)
    rel = contour_local - seed_local[None, :]
    along = rel @ direction
    lateral = np.abs(rel @ normal)
    lateral_limit = max(3.0 * radius_px, 8.0)
    candidates = contour_local[lateral <= lateral_limit]
    candidate_along = along[lateral <= lateral_limit]
    if len(candidates) == 0:
        candidates = contour_local
        candidate_along = along
    target = float(np.max(candidate_along))
    band = max(0.75 * radius_px, 1.5)
    cap = candidates[candidate_along >= target - band]
    if len(cap) == 0:
        cap = candidates[[int(np.argmax(candidate_along))]]
    return cap.mean(axis=0)


def _route_a_distal_apex(contour_local: np.ndarray, anchor_local: np.ndarray, tip_direction: np.ndarray) -> np.ndarray:
    direction = _unit_direction(tip_direction)
    rel = contour_local - anchor_local[None, :]
    along = rel @ direction
    hull = _convex_hull_xy(contour_local)
    hull_rel = hull - anchor_local[None, :]
    hull_along = hull_rel @ direction
    return hull[int(np.argmax(hull_along))]


def _unit_direction(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    return vec / max(float(np.linalg.norm(vec)), 1e-9)


def _point_radius(distance_map: np.ndarray, point_local: np.ndarray) -> float:
    x = int(np.clip(round(float(point_local[0])), 0, distance_map.shape[1] - 1))
    y = int(np.clip(round(float(point_local[1])), 0, distance_map.shape[0] - 1))
    return float(distance_map[y, x])


def _extend_endpoint(point_local: np.ndarray, direction_local: np.ndarray, radius_px: float, sign: float) -> np.ndarray:
    return point_local + sign * radius_px * _unit_direction(direction_local)


def _extract_route_a_endpoints(
    contour_local: np.ndarray,
    config: ExtractionConfig,
    roi_offset_xy: np.ndarray,
    skeleton_path_local: np.ndarray,
    distance_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    anchor_dir = _terminal_direction(skeleton_path_local, head=True)
    tip_dir = _terminal_direction(skeleton_path_local, head=False)
    anchor_radius = _point_radius(distance_map, skeleton_path_local[0])
    tip_radius = _point_radius(distance_map, skeleton_path_local[-1])
    anchor_local = _contour_apex_along_direction(
        contour_local,
        skeleton_path_local[0],
        -anchor_dir,
        anchor_radius,
    )
    tip_local = _route_a_distal_apex(contour_local, anchor_local, tip_dir)
    anchor_global = anchor_local + roi_offset_xy
    tip_global = tip_local + roi_offset_xy
    x_route_a = float(np.linalg.norm(tip_global - anchor_global))
    return anchor_global, tip_global, x_route_a


def _extract_route_b_endpoints(
    contour_local: np.ndarray,
    skeleton_path_local: np.ndarray,
    roi_offset_xy: np.ndarray,
    distance_map: np.ndarray,
    config: ExtractionConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    anchor_dir = _terminal_direction(skeleton_path_local, head=True)
    tip_dir = _terminal_direction(skeleton_path_local, head=False)
    anchor_radius = _point_radius(distance_map, skeleton_path_local[0])
    tip_radius = _point_radius(distance_map, skeleton_path_local[-1])
    scale = float(np.clip(config.route_b_endpoint_extension_scale, 0.0, 1.5))
    inset_scale = float(np.clip(config.route_b_cap_inset_scale, 0.0, 0.5))
    anchor_cap = _contour_cap_center_along_direction(
        contour_local,
        skeleton_path_local[0],
        -anchor_dir,
        anchor_radius,
    )
    tip_cap = _contour_cap_center_along_direction(
        contour_local,
        skeleton_path_local[-1],
        tip_dir,
        tip_radius,
    )
    if scale > 0.0:
        anchor_extend = _extend_endpoint(skeleton_path_local[0], anchor_dir, anchor_radius * scale, sign=-1.0)
        tip_extend = _extend_endpoint(skeleton_path_local[-1], tip_dir, tip_radius * scale, sign=1.0)
        anchor_local = 0.5 * (anchor_cap + anchor_extend)
        tip_local = 0.5 * (tip_cap + tip_extend)
    else:
        anchor_local = anchor_cap
        tip_local = tip_cap
    if inset_scale > 0.0:
        anchor_local = anchor_local + anchor_dir * anchor_radius * inset_scale
        tip_local = tip_local - tip_dir * tip_radius * inset_scale
    anchor_global = anchor_local + roi_offset_xy
    tip_global = tip_local + roi_offset_xy
    x_fit = float(np.linalg.norm(tip_global - anchor_global))
    return anchor_global, tip_global, x_fit


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
    full_n = max(int(np.ceil(total_arc / max(config.fit_bin_px, 1e-6))) + 1, 16)
    full_arc = np.linspace(0.0, total_arc, full_n)
    full_u = np.interp(full_arc, arc, local_path[:, 0])
    full_v = np.interp(full_arc, arc, local_path[:, 1])
    full_local = np.column_stack([full_u, full_v])
    if len(full_local) >= 5:
        du = np.gradient(full_local[:, 0], full_arc)
        dv = np.gradient(full_local[:, 1], full_arc)
        ddu = np.gradient(du, full_arc)
        ddv = np.gradient(dv, full_arc)
        curvature = np.abs(du * ddv - dv * ddu) / np.maximum((du**2 + dv**2) ** 1.5, 1e-9)
        if len(curvature) >= 7:
            kernel = np.ones(7, dtype=float) / 7.0
            curvature = np.convolve(curvature, kernel, mode="same")
        max_curvature = float(np.max(curvature))
        if max_curvature > 1e-9:
            threshold = config.fit_curvature_threshold_ratio * max_curvature
            active = np.flatnonzero(curvature >= threshold)
            if len(active):
                adaptive_arc = float(full_arc[min(int(active[-1] + 2), len(full_arc) - 1)])
            else:
                adaptive_arc = total_arc * config.fit_path_fraction
        else:
            adaptive_arc = total_arc * config.fit_path_fraction
    else:
        adaptive_arc = total_arc * config.fit_path_fraction

    fit_arc_min = total_arc * config.fit_path_fraction_min
    fit_arc_max = float(np.clip(adaptive_arc, fit_arc_min, total_arc * config.fit_path_fraction))
    fit_arc_max = min(max(fit_arc_max, 30.0), total_arc)
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
    roi_offset_xy = np.array([x0, y0], dtype=float)
    blur = cv2.GaussianBlur(crop, (config.blur_ksize, config.blur_ksize), 0)
    mask = (blur < config.threshold_dark).astype(np.uint8) * 255
    open_kernel = np.ones((config.open_kernel, config.open_kernel), np.uint8)
    close_kernel = np.ones((config.close_kernel, config.close_kernel), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    component = _component_mask(mask, config, roi_offset_xy)
    if int(np.count_nonzero(component)) < config.min_component_area:
        raise RuntimeError("needle component not found")

    contour_local = _component_contour(component)
    contour_global = contour_local + np.array([x0, y0], dtype=float)
    anchor_reference_local = _anchor_reference(contour_local, config, roi_offset_xy)
    skeleton_path_local = _extract_skeleton_path_local(component, anchor_reference_local)
    tangent = _estimate_path_tangent(skeleton_path_local)
    skeleton_path_global = skeleton_path_local + roi_offset_xy[None, :]
    distance_map = cv2.distanceTransform((component > 0).astype(np.uint8), cv2.DIST_L2, 5)
    route_a_anchor_global, route_a_tip_global, x_route_a = _extract_route_a_endpoints(
        contour_local,
        config,
        roi_offset_xy,
        skeleton_path_local,
        distance_map,
    )
    anchor_global, tip_global, x_fit = _extract_route_b_endpoints(
        contour_local,
        skeleton_path_local,
        roi_offset_xy,
        distance_map,
        config,
    )
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
