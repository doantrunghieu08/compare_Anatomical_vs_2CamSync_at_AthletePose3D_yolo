"""Robust uncalibrated two-camera pose reconstruction."""

from __future__ import annotations

import numpy as np

from .geometry import CLEAR_JOINTS, H36M_BONES, triangulate_dlt


def normalize_2d_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    centroid = np.mean(points, axis=0)
    dist = np.mean(np.linalg.norm(points - centroid, axis=1))
    scale = np.sqrt(2.0) / max(float(dist), 1e-6)
    transform = np.array([[scale, 0, -scale * centroid[0]], [0, scale, -scale * centroid[1]], [0, 0, 1]])
    homogeneous = np.column_stack([points, np.ones(len(points))])
    return (transform @ homogeneous.T).T[:, :2], transform


def _fundamental_fit(points1, points2, weights):
    norm1, transform1 = normalize_2d_points(points1)
    norm2, transform2 = normalize_2d_points(points2)
    x1, y1 = norm1.T
    x2, y2 = norm2.T
    design = np.column_stack([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, np.ones(len(x1))])
    _, _, vh = np.linalg.svd(design * np.sqrt(weights)[:, None], full_matrices=True)
    raw = vh[-1].reshape(3, 3)
    u, singular, vt = np.linalg.svd(raw)
    singular[-1] = 0.0
    return transform2.T @ (u @ np.diag(singular) @ vt) @ transform1


def estimate_fundamental_matrix(
    pts1: np.ndarray, pts2: np.ndarray,
    conf1: np.ndarray | None = None, conf2: np.ndarray | None = None,
) -> np.ndarray:
    pts1, pts2 = np.asarray(pts1, float), np.asarray(pts2, float)
    confidence = np.ones(len(pts1)) if conf1 is None or conf2 is None else np.minimum(conf1, conf2)
    valid = np.isfinite(pts1).all(1) & np.isfinite(pts2).all(1) & (confidence >= 0.05)
    if valid.sum() < 8:
        raise ValueError("At least eight valid 2D correspondences are required")
    a, b, base = pts1[valid], pts2[valid], np.clip(confidence[valid], 0.01, 1.0)
    weights = base.copy()
    for _ in range(5):
        fundamental = _fundamental_fit(a, b, weights)
        residual = sampson_epipolar_distance(a, b, fundamental)
        scale = max(1.0, 1.4826 * np.median(np.abs(residual - np.median(residual))))
        weights = base * np.minimum(1.0, 1.5 * scale / np.maximum(residual, 1e-9))
    return fundamental


def sampson_epipolar_distance(pts1: np.ndarray, pts2: np.ndarray, fundamental: np.ndarray) -> np.ndarray:
    p1_h = np.column_stack([pts1, np.ones(len(pts1))])
    p2_h = np.column_stack([pts2, np.ones(len(pts2))])
    f_p1, ft_p2 = (fundamental @ p1_h.T).T, (fundamental.T @ p2_h.T).T
    numerator = np.sum(p2_h * f_p1, axis=1) ** 2
    denominator = f_p1[:, 0]**2 + f_p1[:, 1]**2 + ft_p2[:, 0]**2 + ft_p2[:, 1]**2
    return np.sqrt(numerator / np.maximum(denominator, 1e-8))


def approximate_intrinsics(width: float = 1920.0, height: float = 1088.0, f_scale: float = 1.2) -> np.ndarray:
    focal = float(f_scale * max(width, height))
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=float)


def _essential_candidates(essential: np.ndarray, k1: np.ndarray, k2: np.ndarray):
    u, _, vt = np.linalg.svd(essential)
    if np.linalg.det(u) < 0:
        u[:, -1] *= -1
    if np.linalg.det(vt) < 0:
        vt[-1] *= -1
    w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    p1 = k1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    return [(r, sign * u[:, 2], p1, k2 @ np.hstack([r, (sign * u[:, 2]).reshape(3, 1)]))
            for r in (u @ w @ vt, u @ w.T @ vt) for sign in (1.0, -1.0)]


def recover_relative_cameras(pts1, pts2, fundamental, k1, k2):
    essential = k2.T @ fundamental @ k1
    u, singular, vt = np.linalg.svd(essential)
    if np.linalg.det(u) < 0:
        u[:, -1] *= -1
    if np.linalg.det(vt) < 0:
        vt[-1] *= -1
    mean_singular = 0.5 * (singular[0] + singular[1])
    essential = u @ np.diag([mean_singular, mean_singular, 0.0]) @ vt
    candidates = _essential_candidates(essential, k1, k2)
    valid_counts = []
    for rotation, translation, p1, p2 in candidates:
        points_3d = triangulate_dlt(p1, p2, pts1, pts2)
        points_c2 = (rotation @ points_3d.T + translation[:, None]).T
        valid_counts.append(np.count_nonzero((points_3d[:, 2] > 0) & (points_c2[:, 2] > 0)))
    if max(valid_counts, default=0) < 6:
        raise ValueError("Camera recovery failed cheirality validation")
    return candidates[int(np.argmax(valid_counts))]


def solve_metric_scale(points_3d, bone_lengths, conf1=None, conf2=None, min_confidence=0.25):
    poses = np.asarray(points_3d, dtype=float)
    poses = poses[None] if poses.ndim == 2 else poses
    confidence = None if conf1 is None or conf2 is None else np.minimum(conf1, conf2)
    confidence = None if confidence is None else (confidence[None] if confidence.ndim == 1 else confidence)
    ratios = []
    for frame_index, pose in enumerate(poses):
        for (a, b), target in bone_lengths.items():
            if target <= 0 or max(a, b) >= len(pose):
                continue
            if confidence is not None and min(confidence[frame_index, a], confidence[frame_index, b]) <= min_confidence:
                continue
            length = np.linalg.norm(pose[a] - pose[b])
            if np.isfinite(length) and length > 1e-6:
                ratios.append(target / length)
    if len(ratios) < 4:
        raise ValueError("Not enough reliable bones to estimate metric scale")
    ratios = np.asarray(ratios)
    median = np.median(ratios)
    return float(np.median(ratios[np.abs(ratios - median) <= max(0.5 * median, 1e-6)]))


def bone_instability(poses: np.ndarray, bones=H36M_BONES) -> float:
    """Relative variation of bone lengths over time (smaller is more rigid/consistent)."""
    poses = np.asarray(poses)
    if poses.ndim == 2 or len(poses) < 2:
        return 0.0
    lengths = np.stack([np.linalg.norm(poses[:, a] - poses[:, b], axis=-1) for a, b in bones], axis=-1)
    means = lengths.mean(axis=0)
    stds = lengths.std(axis=0)
    valid = means > 1e-6
    return float(np.mean(stds[valid] / means[valid])) if valid.any() else float("inf")


def uncalibrated_triangulation(
    pts1, pts2, conf1, conf2, bone_lengths,
    width=1920.0, height=1088.0,
    f_scale: float | str = 1.2,
    auto_z_up: bool = False,
):
    pts1, pts2 = np.asarray(pts1, float), np.asarray(pts2, float)
    single = pts1.ndim == 2
    pts1, pts2 = (pts1[None], pts2[None]) if single else (pts1, pts2)
    c1 = np.ones(pts1.shape[:2]) if conf1 is None else np.asarray(conf1, float).reshape(pts1.shape[:2])
    c2 = np.ones(pts2.shape[:2]) if conf2 is None else np.asarray(conf2, float).reshape(pts2.shape[:2])
    valid = np.isfinite(pts1).all(-1) & np.isfinite(pts2).all(-1) & (np.minimum(c1, c2) >= 0.05)
    flat1, flat2 = pts1.reshape(-1, 2), pts2.reshape(-1, 2)
    flat_c1, flat_c2, flat_valid = c1.ravel(), c2.ravel(), valid.ravel()
    # ponytail: only use direct COCO joints (CLEAR_JOINTS) for F estimation to avoid synthetic joint bias
    if pts1.shape[1] == 17:
        clear_pts1 = pts1[:, CLEAR_JOINTS, :].reshape(-1, 2)
        clear_pts2 = pts2[:, CLEAR_JOINTS, :].reshape(-1, 2)
        clear_c1 = c1[:, CLEAR_JOINTS].ravel()
        clear_c2 = c2[:, CLEAR_JOINTS].ravel()
        clear_valid = np.isfinite(clear_pts1).all(-1) & np.isfinite(clear_pts2).all(-1) & (np.minimum(clear_c1, clear_c2) >= 0.05)
        if clear_valid.sum() >= 8:
            fundamental = estimate_fundamental_matrix(clear_pts1[clear_valid], clear_pts2[clear_valid], clear_c1[clear_valid], clear_c2[clear_valid])
        else:
            fundamental = estimate_fundamental_matrix(flat1[flat_valid], flat2[flat_valid], flat_c1[flat_valid], flat_c2[flat_valid])
    else:
        fundamental = estimate_fundamental_matrix(flat1[flat_valid], flat2[flat_valid], flat_c1[flat_valid], flat_c2[flat_valid])

    # ponytail: if f_scale is auto, scan candidates for minimum bone instability
    if isinstance(f_scale, str) and f_scale.lower() == "auto":
        best_scale, best_score = 1.2, float("inf")
        for k in np.linspace(0.8, 1.8, 11):
            k1_c = approximate_intrinsics(width, height, f_scale=float(k))
            k2_c = approximate_intrinsics(width, height, f_scale=float(k))
            try:
                _, _, p1_c, p2_c = recover_relative_cameras(flat1[flat_valid], flat2[flat_valid], fundamental, k1_c, k2_c)
                pts_u = triangulate_dlt(p1_c, p2_c, flat1, flat2).reshape(pts1.shape[:2] + (3,))
                score = bone_instability(pts_u)
                if score < best_score:
                    best_score, best_scale = score, float(k)
            except Exception:
                continue
        f_scale = best_scale

    k1 = approximate_intrinsics(width, height, f_scale=float(f_scale))
    k2 = approximate_intrinsics(width, height, f_scale=float(f_scale))
    _, _, p1, p2_unit = recover_relative_cameras(flat1[flat_valid], flat2[flat_valid], fundamental, k1, k2)
    points_unit = triangulate_dlt(p1, p2_unit, flat1, flat2).reshape(pts1.shape[:2] + (3,))
    scale = solve_metric_scale(points_unit, bone_lengths, c1, c2)
    points_metric = points_unit * scale
    points_metric[~valid] = 0.0
    points_metric = np.nan_to_num(points_metric, nan=0.0, posinf=0.0, neginf=0.0)
    rotation, translation = _camera_pose_from_projection(p2_unit, k2)
    p2_metric = k2 @ np.hstack([rotation, (translation * scale).reshape(3, 1)])
    if auto_z_up:
        from .geometry import to_z_up
        points_metric = to_z_up(points_metric)
    return (points_metric[0] if single else points_metric), p1, p2_metric



def _camera_pose_from_projection(p2, intrinsics):
    extrinsics = np.linalg.inv(intrinsics) @ p2
    return extrinsics[:, :3], extrinsics[:, 3]


def uncalibrated_sync_score(
    pts1, pts2, conf1, conf2, bone_lengths, min_confidence=0.25, min_valid=8,
):
    pts1, pts2 = np.asarray(pts1), np.asarray(pts2)
    c1, c2 = np.asarray(conf1), np.asarray(conf2)
    if pts1.ndim == 2:
        pts1, pts2, c1, c2 = pts1[None], pts2[None], c1[None], c2[None]
    confidence = np.minimum(c1, c2)
    valid = np.isfinite(pts1).all(-1) & np.isfinite(pts2).all(-1) & (confidence > min_confidence)
    if np.any(valid.sum(axis=1) < min_valid):
        return np.inf
    try:
        points_3d, p1, p2 = uncalibrated_triangulation(pts1, pts2, conf1, conf2, bone_lengths)
        flat_valid = valid.ravel()
        if pts1.shape[1] == 17:
            clear_pts1 = pts1[:, CLEAR_JOINTS, :].reshape(-1, 2)
            clear_pts2 = pts2[:, CLEAR_JOINTS, :].reshape(-1, 2)
            clear_c1 = c1[:, CLEAR_JOINTS].ravel()
            clear_c2 = c2[:, CLEAR_JOINTS].ravel()
            clear_valid = np.isfinite(clear_pts1).all(-1) & np.isfinite(clear_pts2).all(-1) & (np.minimum(clear_c1, clear_c2) >= 0.05)
            if clear_valid.sum() >= 8:
                fundamental = estimate_fundamental_matrix(clear_pts1[clear_valid], clear_pts2[clear_valid], clear_c1[clear_valid], clear_c2[clear_valid])
            else:
                fundamental = estimate_fundamental_matrix(pts1.reshape(-1, 2)[flat_valid], pts2.reshape(-1, 2)[flat_valid], c1.ravel()[flat_valid], c2.ravel()[flat_valid])
        else:
            fundamental = estimate_fundamental_matrix(pts1.reshape(-1, 2)[flat_valid], pts2.reshape(-1, 2)[flat_valid], c1.ravel()[flat_valid], c2.ravel()[flat_valid])
        epi = np.median(sampson_epipolar_distance(pts1.reshape(-1, 2)[flat_valid],
                                                  pts2.reshape(-1, 2)[flat_valid], fundamental))
        bone_errors = [abs(np.linalg.norm(points_3d[t, a] - points_3d[t, b]) - target) / target
                       for t in range(len(points_3d)) for (a, b), target in bone_lengths.items()
                       if target > 0 and valid[t, a] and valid[t, b]]
        if not bone_errors:
            return np.inf
        return float(epi / 2.0 + 50.0 * np.median(bone_errors) +
                     20.0 * (1.0 - np.mean(confidence[valid])))
    except (ValueError, np.linalg.LinAlgError):
        return np.inf
