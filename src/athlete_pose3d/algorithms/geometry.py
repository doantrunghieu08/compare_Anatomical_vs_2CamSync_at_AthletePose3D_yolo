"""Projection, triangulation, anatomical constraints, and pose metrics."""

from __future__ import annotations


import numpy as np
import torch
import torch.nn.functional as functional
from scipy.linalg import svd
from scipy.optimize import least_squares


H36M_BONES = (
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12),
    (12, 13), (8, 14), (14, 15), (15, 16),
)
H36M_SYMMETRIC_BONES = (
    ((0, 1), (0, 4)),
    ((1, 2), (4, 5)),
    ((2, 3), (5, 6)),
    ((8, 11), (8, 14)),
    ((11, 12), (14, 15)),
    ((12, 13), (15, 16)),
)
LIMB_INDICES = np.array([1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16])
H36M_EVAL_JOINTS = np.arange(1, 17)
DEFAULT_SUBJECT_HEIGHT_MM = 1730.0
MALE_H36M_BONE_RATIOS = {
    (0, 1): 0.076, (1, 2): 0.245, (2, 3): 0.246,
    (0, 4): 0.076, (4, 5): 0.245, (5, 6): 0.246,
    (0, 7): 0.145, (7, 8): 0.150, (8, 9): 0.060,
    (9, 10): 0.078, (8, 11): 0.102, (11, 12): 0.186,
    (12, 13): 0.146, (8, 14): 0.102, (14, 15): 0.186,
    (15, 16): 0.146,
}
CLEAR_JOINTS = np.array([2, 3, 5, 6, 11, 12, 13, 14, 15, 16])
DERIVED_JOINTS = np.array([1, 4, 7, 8, 9, 10])
H36M_JOINT_NAMES = (
    "Pelvis", "R_Hip", "R_Knee", "R_Ankle", "L_Hip", "L_Knee", "L_Ankle",
    "Spine", "Thorax", "Neck", "Head", "L_Shoulder", "L_Elbow", "L_Wrist",
    "R_Shoulder", "R_Elbow", "R_Wrist",
)


def reproject(projection: np.ndarray, point_3d: np.ndarray) -> np.ndarray:
    homogeneous = projection @ np.append(point_3d, 1.0)
    return homogeneous[:2] / homogeneous[2]


def point_to_ray_distance(point_3d: np.ndarray, projection: np.ndarray, point_2d: np.ndarray) -> float:
    """Perpendicular 3D distance from point_3d to camera ray passing through point_2d (in mm)."""
    m = projection[:, :3]
    m_inv = np.linalg.inv(m)
    camera_center = -m_inv @ projection[:, 3]
    ray_dir = m_inv @ np.append(point_2d, 1.0)
    ray_dir /= np.linalg.norm(ray_dir)
    diff = point_3d - camera_center
    proj = np.dot(diff, ray_dir)
    perp = diff - proj * ray_dir
    return float(np.linalg.norm(perp))


def point_to_ray_distance_torch(
    points_3d: torch.Tensor,
    projection: torch.Tensor,
    points_2d: torch.Tensor,
) -> torch.Tensor:
    """Perpendicular 3D distance from points_3d to camera rays passing through points_2d (in mm)."""
    m = projection[:, :3]
    m_inv = torch.linalg.inv(m)
    c = -m_inv @ projection[:, 3]
    ones = torch.ones((len(points_2d), 1), dtype=points_2d.dtype, device=points_2d.device)
    pts_homo = torch.cat([points_2d, ones], dim=-1)
    rays = (m_inv @ pts_homo.T).T
    ray_dirs = rays / torch.linalg.vector_norm(rays, dim=-1, keepdim=True)
    diff = points_3d - c
    proj = torch.sum(diff * ray_dirs, dim=-1, keepdim=True)
    perp = diff - proj * ray_dirs
    return torch.linalg.vector_norm(perp, dim=-1)


def _weighted_dlt_single(
    p1: np.ndarray, p2: np.ndarray, point1: np.ndarray, point2: np.ndarray,
    weight1: float = 1.0, weight2: float = 1.0,
) -> np.ndarray:
    x1, y1 = point1
    x2, y2 = point2
    a = np.array(
        [
            weight1 * (x1 * p1[2] - p1[0]),
            weight1 * (y1 * p1[2] - p1[1]),
            weight2 * (x2 * p2[2] - p2[0]),
            weight2 * (y2 * p2[2] - p2[1]),
        ]
    )
    _, _, vt = svd(a)
    homogeneous = vt[-1]
    return homogeneous[:3] / homogeneous[3]


def dlt_single(p1: np.ndarray, p2: np.ndarray, point1: np.ndarray, point2: np.ndarray) -> np.ndarray:
    return _weighted_dlt_single(p1, p2, point1, point2, 1.0, 1.0)


def triangulate_dlt(p1, p2, points1, points2, *_confidences) -> np.ndarray:
    return np.array([dlt_single(p1, p2, a, b) for a, b in zip(points1, points2)])


def triangulate_conf_algebraic(p1, p2, points1, points2, confidence1, confidence2) -> np.ndarray:
    points_3d = np.zeros((len(points1), 3))
    for i, (point1, point2) in enumerate(zip(points1, points2)):
        weight1 = float(np.clip(confidence1[i], 0.01, 1.0) ** 2)
        weight2 = float(np.clip(confidence2[i], 0.01, 1.0) ** 2)
        points_3d[i] = _weighted_dlt_single(p1, p2, point1, point2, weight1, weight2)
    return points_3d


def triangulate_reweighted_dlt(p1, p2, points1, points2, confidence1, confidence2, reprojection_threshold=15.0):
    """Reweighted DLT based on reprojection error consistency (replaces pseudo-RANSAC)."""
    points_3d = np.zeros((len(points1), 3))
    for i, (point1, point2) in enumerate(zip(points1, points2)):
        point_3d = dlt_single(p1, p2, point1, point2)
        error1 = np.linalg.norm(reproject(p1, point_3d) - point1)
        error2 = np.linalg.norm(reproject(p2, point_3d) - point2)
        if error1 > reprojection_threshold >= error2:
            weight1, weight2 = 0.1, 1.0
        elif error2 > reprojection_threshold >= error1:
            weight1, weight2 = 1.0, 0.1
        elif error1 > reprojection_threshold and error2 > reprojection_threshold:
            weight1, weight2 = float(confidence1[i] ** 2), float(confidence2[i] ** 2)
        else:
            weight1, weight2 = float(confidence1[i]), float(confidence2[i])
        points_3d[i] = _weighted_dlt_single(p1, p2, point1, point2, weight1, weight2)
    return points_3d


# Alias for backward compatibility
triangulate_ransac = triangulate_reweighted_dlt


def triangulate_iterative(p1, p2, points1, points2, confidence1, confidence2, iterations=50):
    points_3d = np.zeros((len(points1), 3))
    for i, (point1, point2) in enumerate(zip(points1, points2)):
        initial = dlt_single(p1, p2, point1, point2)
        weight1 = np.clip(confidence1[i], 0.1, 1.0)
        weight2 = np.clip(confidence2[i], 0.1, 1.0)

        def residuals(point_3d):
            return np.concatenate(
                [weight1 * (reproject(p1, point_3d) - point1), weight2 * (reproject(p2, point_3d) - point2)]
            )

        points_3d[i] = least_squares(residuals, initial, method="lm", max_nfev=iterations).x
    return points_3d


def estimate_subject_height(pose_3d: np.ndarray) -> float:
    leg_l = (
        np.linalg.norm(pose_3d[3] - pose_3d[2])
        + np.linalg.norm(pose_3d[2] - pose_3d[1])
        + np.linalg.norm(pose_3d[1] - pose_3d[0])
    )
    leg_r = (
        np.linalg.norm(pose_3d[6] - pose_3d[5])
        + np.linalg.norm(pose_3d[5] - pose_3d[4])
        + np.linalg.norm(pose_3d[4] - pose_3d[0])
    )
    torso = (
        np.linalg.norm(pose_3d[0] - pose_3d[7])
        + np.linalg.norm(pose_3d[7] - pose_3d[8])
        + np.linalg.norm(pose_3d[8] - pose_3d[9])
        + np.linalg.norm(pose_3d[9] - pose_3d[10])
    )
    return float((leg_l + leg_r) / 2.0 + torso)


def resolve_subject_height(
    subject: str,
    configured_height: float | str | None,
    subject_heights: dict[str, float] | None = None,
    sample_poses: list[np.ndarray] | None = None,
) -> float:
    if isinstance(configured_height, (int, float)) and configured_height > 0:
        return float(configured_height)
    if isinstance(configured_height, str) and configured_height.lower() != "auto":
        try:
            return float(configured_height)
        except ValueError:
            pass
    if subject_heights and subject in subject_heights:
        return float(subject_heights[subject])
    if sample_poses:
        valid_heights = [estimate_subject_height(p) for p in sample_poses if np.isfinite(p).all()]
        if valid_heights:
            return float(np.median(valid_heights))
    return DEFAULT_SUBJECT_HEIGHT_MM


def _calibrated_bone_lengths(height_mm: float, sample_poses: list[np.ndarray]) -> dict[tuple[int, int], float]:
    raw_lengths = {}
    for u, v in H36M_BONES:
        lengths = [np.linalg.norm(p[u] - p[v]) for p in sample_poses if np.isfinite(p).all()]
        raw_lengths[(u, v)] = float(np.median(lengths)) if lengths else height_mm * MALE_H36M_BONE_RATIOS[(u, v)]
    chain = [((0, 1), (1, 2), (2, 3)), ((0, 7), (7, 8), (8, 9), (9, 10))]
    chain_sum = sum(raw_lengths[b] for sub in chain for b in sub)
    scale = (height_mm / chain_sum) if chain_sum > 0 else 1.0
    return {b: float(l * scale) for b, l in raw_lengths.items()}


def h36m_bone_lengths_from_height(
    height_mm: float | str = DEFAULT_SUBJECT_HEIGHT_MM,
    sample_poses: list[np.ndarray] | None = None,
) -> dict[tuple[int, int], float]:
    if isinstance(height_mm, str):
        try:
            height_mm = float(height_mm)
        except ValueError:
            height_mm = DEFAULT_SUBJECT_HEIGHT_MM
    if 0.5 <= height_mm <= 3.0:
        height_mm *= 1000.0
    elif 50.0 <= height_mm <= 250.0:
        height_mm *= 10.0
    if sample_poses:
        return _calibrated_bone_lengths(height_mm, sample_poses)
    return {bone: float(height_mm * ratio) for bone, ratio in MALE_H36M_BONE_RATIOS.items()}


def _collect_bone_samples(poses_3d, confidences, min_conf):
    samples = {b: [] for b in H36M_BONES}
    for i, p in enumerate(poses_3d):
        if not np.isfinite(p).all():
            continue
        c = confidences[i] if confidences is not None and i < len(confidences) else None
        for u, v in H36M_BONES:
            if c is not None and min_conf is not None and (c[u] < min_conf or c[v] < min_conf):
                continue
            dist = float(np.linalg.norm(p[u] - p[v]))
            if 20.0 <= dist <= 1200.0:
                samples[(u, v)].append(dist)
    return samples


def estimate_bone_lengths_from_poses(
    poses_3d: list[np.ndarray] | np.ndarray,
    *,
    height_mm: float | str | None = None,
    min_samples: int = 5,
    confidences: list[np.ndarray] | np.ndarray | None = None,
    min_confidence: float | None = 0.4,
) -> tuple[dict[tuple[int, int], float], dict]:
    """Estimate subject-specific bone lengths from 3D DLT poses without ground truth."""
    height = height_mm or DEFAULT_SUBJECT_HEIGHT_MM
    fallback_map = h36m_bone_lengths_from_height(height)
    raw_samples = _collect_bone_samples(poses_3d, confidences, min_confidence)
    estimated, diag = {}, {"valid_counts": {}, "std_devs": {}, "fallbacks": {}, "total_poses": len(poses_3d)}
    fallback_count = 0
    for b in H36M_BONES:
        vals = np.array(raw_samples[b])
        if len(vals) >= min_samples:
            med = float(np.median(vals))
            inliers = vals[(vals >= 0.6 * med) & (vals <= 1.6 * med)]
            if len(inliers) >= min_samples:
                estimated[b] = float(np.median(inliers))
                diag["valid_counts"][b] = len(inliers)
                diag["std_devs"][b] = float(np.std(inliers))
                diag["fallbacks"][b] = False
                continue
        estimated[b] = fallback_map[b]
        diag["valid_counts"][b] = len(vals)
        diag["std_devs"][b] = float(np.std(vals)) if len(vals) > 1 else 0.0
        diag["fallbacks"][b] = True
        fallback_count += 1
    diag["used_fallback_count"] = fallback_count
    return estimated, diag



def compute_kinematic_prior(points_3d: torch.Tensor, margin=0.05) -> torch.Tensor:
    normalize = lambda vector: functional.normalize(vector, dim=-1, eps=1e-6)
    hip_axis = normalize(points_3d[1] - points_3d[4])
    shoulder_axis = normalize(points_3d[14] - points_3d[11])
    limbs = (
        (2, 1, 3, 2, hip_axis, 1),
        (5, 4, 6, 5, hip_axis, 1),
        (15, 14, 16, 15, shoulder_axis, -1),
        (12, 11, 13, 12, shoulder_axis, -1),
    )
    loss = points_3d.new_tensor(0.0)
    for a, b, c, d, axis, sign in limbs:
        first = normalize(points_3d[a] - points_3d[b])
        second = normalize(points_3d[c] - points_3d[d])
        bend = torch.dot(torch.linalg.cross(first, second, dim=-1), axis)
        loss = loss + torch.relu(sign * bend - margin) ** 2
    return loss


def _camera_rays(p_mat: torch.Tensor, pts_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m = p_mat[:, :3]
    m_inv = torch.linalg.inv(m)
    c = -m_inv @ p_mat[:, 3]
    ones = torch.ones((len(pts_2d), 1), dtype=pts_2d.dtype, device=pts_2d.device)
    rays = (m_inv @ torch.cat([pts_2d, ones], dim=-1).T).T
    dirs = rays / torch.linalg.vector_norm(rays, dim=-1, keepdim=True)
    return c, dirs


def _anatomical_step(
    points_3d: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    p1_t: torch.Tensor,
    p2_t: torch.Tensor,
    pts1_t: torch.Tensor,
    pts2_t: torch.Tensor,
    c1_t: torch.Tensor,
    c2_t: torch.Tensor,
    bone_lengths: dict | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    bone_weight: float,
    rays_precomputed: tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> float:
    optimizer.zero_grad()
    if rays_precomputed is not None:
        (c1, d1), (c2, d2) = rays_precomputed
        diff1 = points_3d - c1
        diff2 = points_3d - c2
        dist1 = torch.linalg.vector_norm(diff1 - torch.sum(diff1 * d1, dim=-1, keepdim=True) * d1, dim=-1)
        dist2 = torch.linalg.vector_norm(diff2 - torch.sum(diff2 * d2, dim=-1, keepdim=True) * d2, dim=-1)
    else:
        dist1 = point_to_ray_distance_torch(points_3d, p1_t, pts1_t)
        dist2 = point_to_ray_distance_torch(points_3d, p2_t, pts2_t)
    loss1 = functional.huber_loss(dist1, torch.zeros_like(dist1), reduction="none", delta=5.0)
    loss2 = functional.huber_loss(dist2, torch.zeros_like(dist2), reduction="none", delta=5.0)
    if isinstance(bone_lengths, tuple):
        b_idx_a, b_idx_b, b_targets = bone_lengths
        if len(b_idx_a) > 0:
            actuals = torch.linalg.vector_norm(points_3d[b_idx_a] - points_3d[b_idx_b], dim=-1)
            bone_loss = torch.sum(functional.huber_loss(actuals, b_targets, reduction="none", delta=10.0))
        else:
            bone_loss = points_3d.new_tensor(0.0)
    else:
        valid = [(a, b, l) for (a, b), l in bone_lengths.items() if l > 0]
        if valid:
            idx_a = [v[0] for v in valid]
            idx_b = [v[1] for v in valid]
            targets = points_3d.new_tensor([v[2] for v in valid])
            actuals = torch.linalg.vector_norm(points_3d[idx_a] - points_3d[idx_b], dim=-1)
            bone_loss = torch.sum(functional.huber_loss(actuals, targets, reduction="none", delta=10.0))
        else:
            bone_loss = points_3d.new_tensor(0.0)
    loss = torch.sum(loss1 * c1_t) + torch.sum(loss2 * c2_t) + bone_weight * bone_loss
    loss.backward()
    optimizer.step()
    return float(loss.item())


def triangulate_anatomical(
    p1,
    p2,
    points1,
    points2,
    confidence1,
    confidence2,
    *,
    bone_lengths=None,
    bone_weight=1.0,
    iterations=80,
    lr=0.1,
    **_kwargs,
):
    init_3d = triangulate_dlt(p1, p2, points1, points2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points_3d = torch.tensor(init_3d, dtype=torch.float32, device=device, requires_grad=True)
    p1_t, p2_t = (torch.as_tensor(value, dtype=torch.float32, device=device) for value in (p1, p2))
    pts1_t, pts2_t = (torch.as_tensor(value, dtype=torch.float32, device=device) for value in (points1, points2))
    c1_t, c2_t = (
        torch.as_tensor(np.clip(value, 0.0, 1.0) ** 2, dtype=torch.float32, device=device)
        for value in (confidence1, confidence2)
    )
    rays_pre = (_camera_rays(p1_t, pts1_t), _camera_rays(p2_t, pts2_t))
    optimizer = torch.optim.Adam([points_3d], lr=float(lr))
    bone_lengths = bone_lengths or h36m_bone_lengths_from_height()
    valid_bones = [(a, b, l) for (a, b), l in bone_lengths.items() if l > 0]
    b_idx_a = torch.tensor([v[0] for v in valid_bones], dtype=torch.long, device=device)
    b_idx_b = torch.tensor([v[1] for v in valid_bones], dtype=torch.long, device=device)
    b_targets = torch.tensor([v[2] for v in valid_bones], dtype=torch.float32, device=device)
    bone_tensors = (b_idx_a, b_idx_b, b_targets)

    best_pose, best_loss = points_3d.detach().clone(), float("inf")
    for _ in range(iterations):
        candidate = points_3d.detach().clone()
        loss_val = _anatomical_step(
            points_3d, optimizer, p1_t, p2_t, pts1_t, pts2_t, c1_t, c2_t,
            bone_tensors, bone_weight, rays_precomputed=rays_pre,
        )
        if np.isfinite(loss_val) and loss_val < best_loss:
            best_pose = candidate
            best_loss = loss_val

    res = best_pose.detach().cpu().numpy()
    return res if np.isfinite(res).all() else init_3d


def fundamental_from_projections(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Compute F such that x2.T @ F @ x1 = 0 from two 3x4 projections."""
    p1, p2 = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
    if p1.shape != (3, 4) or p2.shape != (3, 4):
        raise ValueError("projection matrices must have shape (3, 4)")
    camera1 = np.linalg.svd(p1)[2][-1]
    epipole2 = p2 @ camera1
    ex = np.array(
        [
            [0.0, -epipole2[2], epipole2[1]],
            [epipole2[2], 0.0, -epipole2[0]],
            [-epipole2[1], epipole2[0], 0.0],
        ]
    )
    fundamental = ex @ p2 @ np.linalg.pinv(p1)
    norm = np.linalg.norm(fundamental)
    return fundamental / norm if norm > 1e-12 else fundamental


def _joint_bone_error(joint_idx: int, points_3d: np.ndarray, bone_lengths: dict) -> float:
    connected = [b for b in H36M_BONES if joint_idx in b]
    if not connected:
        return 0.0
    errors = []
    for a, b in connected:
        target = bone_lengths.get((a, b), 0.0) or bone_lengths.get((b, a), 0.0)
        if target > 0:
            actual = np.linalg.norm(points_3d[a] - points_3d[b])
            errors.append(abs(actual - target) / target)
    return float(np.mean(errors)) if errors else 0.0


def detect_stereo_occlusions(
    p1: np.ndarray, p2: np.ndarray,
    pts1: np.ndarray, pts2: np.ndarray,
    c1: np.ndarray, c2: np.ndarray,
    bone_lengths: dict | None = None,
    initial_3d: np.ndarray | None = None,
) -> dict:
    pts1, pts2 = np.asarray(pts1, dtype=float), np.asarray(pts2, dtype=float)
    c1, c2 = np.asarray(c1, dtype=float).copy(), np.asarray(c2, dtype=float).copy()
    n = len(pts1)

    c1[~np.isfinite(pts1).all(axis=-1) | (pts1[:, 0] == 0) & (pts1[:, 1] == 0)] = 0.0
    c2[~np.isfinite(pts2).all(axis=-1) | (pts2[:, 0] == 0) & (pts2[:, 1] == 0)] = 0.0

    try:
        f_mat = fundamental_from_projections(p1, p2)
        h1 = np.column_stack((pts1, np.ones(n)))
        h2 = np.column_stack((pts2, np.ones(n)))
        f_x1 = h1 @ f_mat.T
        ft_x2 = h2 @ f_mat
        num = np.sum(h2 * f_x1, axis=1) ** 2
        denom = f_x1[:, 0] ** 2 + f_x1[:, 1] ** 2 + ft_x2[:, 0] ** 2 + ft_x2[:, 1] ** 2
        res = np.sqrt(num / np.maximum(denom, 1e-12))
    except Exception:
        res = np.zeros(n)

    if initial_3d is None or not np.isfinite(initial_3d).all():
        initial_3d = triangulate_dlt(p1, p2, pts1, pts2)

    bone_errs = np.zeros(n)
    if bone_lengths and np.isfinite(initial_3d).all():
        bone_errs = np.array([_joint_bone_error(j, initial_3d, bone_lengths) for j in range(n)])

    med_res = float(np.median(res)) if np.isfinite(res).all() and len(res) > 0 else 0.0
    epipolar_outlier = (res > 1.25 * med_res) & (res > 15.0) if med_res > 0 else np.zeros(n, dtype=bool)
    severe_epipolar = (res > 1.45 * med_res) if med_res > 0 else np.zeros(n, dtype=bool)
    bone_outlier = bone_errs > 0.35
    is_out = severe_epipolar | (epipolar_outlier & bone_outlier)

    detector_occ = (c1 <= 0.3) | (c2 <= 0.3)
    occluded_mask = detector_occ | is_out | severe_epipolar | (epipolar_outlier & bone_outlier)
    if np.isfinite(pts1[0]).all() and np.isfinite(pts2[0]).all():
        occluded_mask[0] = False

    stereo_conf = np.zeros(n)
    for j in range(n):
        if occluded_mask[j]:
            ratio = min(res[j] / max(med_res, 1.0), 3.0) if med_res > 0 else 2.0
            stereo_conf[j] = float(np.clip(0.30 / ratio, 0.05, 0.28))
        else:
            ratio = max(res[j] / max(med_res, 1.0), 1.0) if med_res > 0 else 1.0
            stereo_conf[j] = float(np.clip(0.95 - 0.15 * (ratio - 1.0), 0.70, 0.98))

    return {
        "occluded_mask": occluded_mask,
        "stereo_confidence": np.clip(stereo_conf, 0.0, 1.0),
        "conf_a": np.clip(c1, 0.0, 1.0),
        "conf_b": np.clip(c2, 0.0, 1.0),
        "is_outlier": is_out,
        "sampson_res": res,
        "bone_errs": bone_errs,
    }





def procrustes_align(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is not None:
        p_sub, t_sub = prediction[mask], target[mask]
        prediction_mean, target_mean = p_sub.mean(0), t_sub.mean(0)
        prediction_centered, target_centered = p_sub - prediction_mean, t_sub - target_mean
    else:
        prediction_mean, target_mean = prediction.mean(0), target.mean(0)
        prediction_centered, target_centered = prediction - prediction_mean, target - target_mean
    covariance = prediction_centered.T @ target_centered
    if np.linalg.norm(covariance) < 1e-8:
        return prediction - prediction_mean + target_mean
    u, _, vt = svd(covariance)
    rotation = vt.T @ np.diag([1, 1, np.linalg.det(vt.T @ u.T)]) @ u.T
    scale = np.trace(rotation @ covariance) / max(float(np.trace(prediction_centered.T @ prediction_centered)), 1e-8)
    return scale * ((prediction - prediction_mean) @ rotation.T) + target_mean


def raw_mpjpe(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Root-relative 16-joint H36M MPJPE in millimeters without rotation alignment."""
    pred_rel = prediction - np.expand_dims(prediction[..., 0, :], axis=-2)
    gt_rel = ground_truth - np.expand_dims(ground_truth[..., 0, :], axis=-2)
    return float(np.mean(np.linalg.norm(
        pred_rel[..., H36M_EVAL_JOINTS, :] - gt_rel[..., H36M_EVAL_JOINTS, :], axis=-1
    )))


def mpjpe(prediction: np.ndarray, ground_truth: np.ndarray, align_rotation: bool = True) -> float:
    """Root-relative 16-joint H36M MPJPE in millimeters (rigid-aligned to GT coordinate frame by default)."""
    if align_rotation:
        return rigid_mpjpe(prediction, ground_truth)
    return raw_mpjpe(prediction, ground_truth)


def pa_mpjpe(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    aligned = procrustes_align(prediction, ground_truth, mask=H36M_EVAL_JOINTS)
    return float(np.mean(np.linalg.norm(
        aligned[..., H36M_EVAL_JOINTS, :] - ground_truth[..., H36M_EVAL_JOINTS, :], axis=-1
    )))


def rigid_align(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Rigid alignment: finds optimal SO(3) rotation and translation WITHOUT altering scale."""
    if mask is not None:
        p_sub, t_sub = prediction[mask], target[mask]
        p_mean, t_mean = p_sub.mean(0), t_sub.mean(0)
        p_centered, t_centered = p_sub - p_mean, t_sub - t_mean
    else:
        p_mean, t_mean = prediction.mean(0), target.mean(0)
        p_centered, t_centered = prediction - p_mean, target - t_mean
    covariance = p_centered.T @ t_centered
    if np.linalg.norm(covariance) < 1e-8:
        return prediction - p_mean + t_mean
    u, _, vt = svd(covariance)
    rotation = vt.T @ np.diag([1.0, 1.0, np.linalg.det(vt.T @ u.T)]) @ u.T
    return ((prediction - p_mean) @ rotation.T) + t_mean


def rigid_mpjpe(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Rigid-aligned MPJPE (optimal SO(3) rotation and translation, preserving metric scale)."""
    aligned = rigid_align(prediction, ground_truth, mask=H36M_EVAL_JOINTS)
    return float(np.mean(np.linalg.norm(
        aligned[..., H36M_EVAL_JOINTS, :] - ground_truth[..., H36M_EVAL_JOINTS, :], axis=-1
    )))


def kabsch(p: np.ndarray, g: np.ndarray, with_scale: bool = False) -> tuple[np.ndarray, float]:
    """P, G: (N, 3) centered at origin. Return optimal R (and optional scale s) minimizing ||s * P @ R.T - G||_F."""
    h = p.T @ g
    u, s_vals, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    diag = np.diag([1.0, 1.0, d])
    r = vt.T @ diag @ u.T
    scale = 1.0
    if with_scale:
        p_sq = float((p ** 2).sum())
        scale = float((s_vals * np.diag(diag)).sum() / max(p_sq, 1e-8))
    return r, scale


def rotation_angle_deg(r: np.ndarray) -> float:
    """Rotation angle of SO(3) matrix in degrees."""
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))))


def sequence_diagnosis(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """pred, gt: (T, 17, 3), pelvis = joint 0. Evaluates sequence-level rotation offset, scale, and aligned MPJPE."""
    p = (pred - pred[..., :1, :])[..., 1:, :].reshape(-1, 3)
    g = (gt - gt[..., :1, :])[..., 1:, :].reshape(-1, 3)
    r_sim, scale = kabsch(p, g, with_scale=True)
    r_rigid, _ = kabsch(p, g, with_scale=False)
    err_rot = float(np.linalg.norm((p @ r_rigid.T) - g, axis=1).mean())
    err_sim = float(np.linalg.norm(scale * (p @ r_sim.T) - g, axis=1).mean())
    return {
        "angle_deg": rotation_angle_deg(r_rigid),
        "scale_ratio": scale,
        "mpjpe_seq_rot_mm": err_rot,
        "mpjpe_seq_sim_mm": err_sim,
    }


def estimate_up_axis(poses: np.ndarray) -> np.ndarray:
    """poses: (T, 17, 3) or (17, 3) in camera frame. Estimates up-axis from average pelvis -> neck vector."""
    poses = np.asarray(poses)
    p = poses[None] if poses.ndim == 2 else poses
    v = p[:, 9] - p[:, 0]  # H36M: 0 = pelvis, 9 = neck
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    up = np.median(v, axis=0)
    norm = float(np.linalg.norm(up))
    return up / max(norm, 1e-9)


def rotation_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rodrigues rotation matrix mapping unit vector a to unit vector b."""
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3) + 2.0 * np.outer(a, a)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + (k @ k) / (1.0 + c)


def to_z_up(poses: np.ndarray) -> np.ndarray:
    """Align poses to Z-up coordinate system without ground truth."""
    poses = np.asarray(poses)
    single = poses.ndim == 2
    p = poses[None] if single else poses
    r = rotation_a_to_b(estimate_up_axis(p), np.array([0.0, 0.0, 1.0]))
    out = p @ r.T
    return out[0] if single else out


def evaluate_joint_groups(prediction: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    """Evaluate PA-MPJPE separately for clear limb joints and derived trunk/head joints."""
    aligned = procrustes_align(prediction, ground_truth, mask=H36M_EVAL_JOINTS)
    diff = np.linalg.norm(aligned - ground_truth, axis=-1)
    return {
        "pa_clear": float(np.mean(diff[CLEAR_JOINTS])),
        "pa_derived": float(np.mean(diff[DERIVED_JOINTS])),
    }


