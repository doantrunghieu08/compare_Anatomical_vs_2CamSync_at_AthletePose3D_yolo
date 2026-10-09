"""Dempster-Shafer Theory (DST) for multi-camera evidence fusion."""

from __future__ import annotations

import numpy as np

from .geometry import H36M_BONES


def compute_yolo_bba(
    conf_a: np.ndarray,
    conf_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c1 = np.clip(np.asarray(conf_a, dtype=float), 0.0, 1.0)
    c2 = np.clip(np.asarray(conf_b, dtype=float), 0.0, 1.0)
    # ponytail: vacuous belief when confidences are missing / uniform 1.0, prevents fake 0.98 certainty
    if np.allclose(c1, 1.0) and np.allclose(c2, 1.0):
        n = len(c1)
        return np.zeros(n), np.zeros(n), np.ones(n)
    m_v = c1 * c2
    m_nv = (1.0 - np.maximum(c1, c2)) ** 2
    total = m_v + m_nv
    scale = np.where(total > 0.98, 0.98 / np.maximum(total, 1e-6), 1.0)
    m_v = m_v * scale
    m_nv = m_nv * scale
    m_theta = np.clip(1.0 - m_v - m_nv, 0.02, 1.0)
    return m_v, m_nv, m_theta



def _camera_center_and_ray(p: np.ndarray, pt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = p[:, :3]
    try:
        m_inv = np.linalg.inv(m)
        c = -m_inv @ p[:, 3]
        ray = m_inv @ np.array([pt[0], pt[1], 1.0], dtype=float)
        norm = np.linalg.norm(ray)
        return c, ray / (norm + 1e-12)
    except np.linalg.LinAlgError:
        return np.zeros(3), np.array([0.0, 0.0, 1.0])


def _ray_distance_3d_mm(c1: np.ndarray, d1: np.ndarray, c2: np.ndarray, d2: np.ndarray) -> float:
    cross = np.cross(d1, d2)
    norm = np.linalg.norm(cross)
    if norm < 1e-6:
        diff = c2 - c1
        return float(np.linalg.norm(diff - np.dot(diff, d1) * d1))
    return float(abs(np.dot(c2 - c1, cross)) / norm)


def error_to_bba(
    err: np.ndarray | float,
    scale: float,
    cap: float = 0.85,
    floor_theta: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map a non-negative residual to valid, invalid, and unknown mass."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a positive finite number")
    if not 0 < cap < 1 or not 0 <= floor_theta < 1:
        raise ValueError("cap and floor_theta must be valid probabilities")
    err = np.maximum(np.asarray(err, dtype=float), 0.0)
    valid = cap * np.exp(-0.5 * (err / scale) ** 2)
    invalid = cap * (1.0 - np.exp(-0.125 * (err / scale) ** 2))
    unknown = np.maximum(floor_theta, 1.0 - valid - invalid)
    total = valid + invalid + unknown
    return valid / total, invalid / total, unknown / total


def discount(
    bba: tuple[np.ndarray, np.ndarray, np.ndarray],
    alpha: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shafer discounting: move unsupported mass to the unknown state."""
    valid, invalid, unknown = bba
    alpha = np.clip(np.asarray(alpha, dtype=float), 0.0, 1.0)
    return alpha * valid, alpha * invalid, 1.0 - alpha * (valid + invalid)


def combine(
    bba1: tuple[np.ndarray, np.ndarray, np.ndarray],
    bba2: tuple[np.ndarray, np.ndarray, np.ndarray],
    rule: str = "yager",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Combine two binary-frame BBAs using Yager or normalized Dempster."""
    if rule not in {"yager", "dempster"}:
        raise ValueError("rule must be 'yager' or 'dempster'")
    v1, nv1, u1 = bba1
    v2, nv2, u2 = bba2
    valid = v1 * v2 + v1 * u2 + u1 * v2
    invalid = nv1 * nv2 + nv1 * u2 + u1 * nv2
    unknown = u1 * u2
    conflict = v1 * nv2 + nv1 * v2
    if rule == "yager":
        return valid, invalid, unknown + conflict, conflict
    denom = np.maximum(1.0 - conflict, 1e-4)
    valid, invalid, unknown = valid / denom, invalid / denom, unknown / denom
    total = np.maximum(valid + invalid + unknown, 1e-12)
    return valid / total, invalid / total, unknown / total, conflict


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
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("could not derive a valid fundamental matrix")
    return fundamental / norm


def compute_sampson_bba(
    p1: np.ndarray,
    p2: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
    sigma_px: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score pixel-space epipolar residuals, independent of reconstructed scale."""
    points1, points2 = np.asarray(points1, dtype=float), np.asarray(points2, dtype=float)
    if points1.ndim != 2 or points1.shape[1] != 2 or points2.shape != points1.shape:
        raise ValueError("points1 and points2 must have matching shape (N, 2)")
    if len(points1) == 0:
        empty = np.empty(0, dtype=float)
        return empty, empty, empty
    fundamental = fundamental_from_projections(p1, p2)
    h1 = np.column_stack((points1, np.ones(len(points1))))
    h2 = np.column_stack((points2, np.ones(len(points2))))
    f_x1 = h1 @ fundamental.T
    ft_x2 = h2 @ fundamental
    numerator = np.sum(h2 * f_x1, axis=1) ** 2
    denominator = (
        f_x1[:, 0] ** 2 + f_x1[:, 1] ** 2
        + ft_x2[:, 0] ** 2 + ft_x2[:, 1] ** 2
    )
    residual = np.sqrt(numerator / np.maximum(denominator, 1e-12))
    if sigma_px is None:
        image_person_height = 0.5 * (
            np.ptp(points1[:, 1]) + np.ptp(points2[:, 1])
        )
        sigma_px = max(2.0, 0.015 * float(image_person_height))
    return error_to_bba(residual, float(sigma_px), cap=0.85)


def triangulation_alpha(
    p1: np.ndarray,
    p2: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
    full_deg: float = 15.0,
    floor: float = 0.2,
) -> np.ndarray:
    """Discount geometry when the two camera rays have weak triangulation angle."""
    if full_deg <= 0 or not 0 <= floor <= 1:
        raise ValueError("full_deg must be positive and floor must be in [0, 1]")
    angles = np.ones(len(points1), dtype=float)
    for i, (point1, point2) in enumerate(zip(points1, points2)):
        _, d1 = _camera_center_and_ray(p1, point1)
        _, d2 = _camera_center_and_ray(p2, point2)
        angle = np.degrees(np.arccos(np.clip(abs(float(np.dot(d1, d2))), 0.0, 1.0)))
        angles[i] = np.clip(angle / full_deg, floor, 1.0)
    return angles


def triangulate_ray_midpoint(
    p1: np.ndarray,
    p2: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
) -> np.ndarray:
    """Pure geometric ray-intersection triangulation independent of DLT."""
    n = len(points1)
    pts_3d = np.zeros((n, 3), dtype=float)
    for i in range(n):
        c1, d1 = _camera_center_and_ray(p1, points1[i])
        c2, d2 = _camera_center_and_ray(p2, points2[i])
        w0 = c1 - c2
        b = float(np.dot(d1, d2))
        d = float(np.dot(d1, w0))
        e = float(np.dot(d2, w0))
        denom = 1.0 - b * b
        if abs(denom) < 1e-6:
            pts_3d[i] = (c1 + c2) * 0.5
        else:
            t1 = (b * e - d) / denom
            t2 = (e - b * d) / denom
            pts_3d[i] = ((c1 + t1 * d1) + (c2 + t2 * d2)) * 0.5
    return pts_3d


def compute_epipolar_bba(
    p1: np.ndarray,
    p2: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
    sigma_mm: float = 60.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Epipolar/Geometric BBA based on 3D Ray-to-Ray distance in millimeters (mm)."""
    n = len(points1)
    m_v, m_nv, m_theta = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        c1, d1 = _camera_center_and_ray(p1, points1[i])
        c2, d2 = _camera_center_and_ray(p2, points2[i])
        dist_mm = _ray_distance_3d_mm(c1, d1, c2, d2)
        v = np.exp(-(dist_mm ** 2) / (2.0 * sigma_mm ** 2))
        nv = 1.0 - np.exp(-(dist_mm ** 2) / (2.0 * (2.0 * sigma_mm) ** 2))
        m_v[i] = np.clip(0.85 * v, 0.0, 0.85)
        m_nv[i] = np.clip(0.85 * nv, 0.0, 0.85)
        m_theta[i] = max(0.05, 1.0 - m_v[i] - m_nv[i])
    return m_v, m_nv, m_theta


def _joint_bone_error(joint_idx: int, points_3d: np.ndarray, bone_lengths: dict) -> float:
    connected = [b for b in H36M_BONES if joint_idx in b]
    if not connected:
        return 0.0
    errors = []
    for a, b in connected:
        target = bone_lengths.get((a, b), 0.0)
        if target > 0:
            actual = np.linalg.norm(points_3d[a] - points_3d[b])
            errors.append(abs(actual - target) / target)
    return float(np.mean(errors)) if errors else 0.0


def compute_bone_bba_legacy(
    points_3d: np.ndarray,
    bone_lengths: dict,
    tolerance: float = 0.22,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(points_3d)
    m_v, m_nv, m_theta = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        err = _joint_bone_error(i, points_3d, bone_lengths)
        v = np.exp(-(err ** 2) / (2.0 * tolerance ** 2))
        nv = 1.0 - np.exp(-(err ** 2) / (2.0 * (2.0 * tolerance) ** 2))
        m_v[i] = np.clip(0.80 * v, 0.0, 0.80)
        m_nv[i] = np.clip(0.80 * nv, 0.0, 0.80)
        m_theta[i] = max(0.05, 1.0 - m_v[i] - m_nv[i])
    return m_v, m_nv, m_theta


def compute_bone_bba(
    points_3d: np.ndarray,
    bone_lengths: dict,
    joint_reliability: np.ndarray | None = None,
    tolerance: float = 0.22,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute bone evidence per joint; joints without a usable prior stay unknown."""
    points_3d = np.asarray(points_3d, dtype=float)
    n = len(points_3d)
    reliability = (
        np.ones(n, dtype=float)
        if joint_reliability is None
        else np.clip(np.asarray(joint_reliability, dtype=float), 0.0, 1.0)
    )
    if reliability.shape != (n,):
        raise ValueError("joint_reliability must have one value per joint")
    if tolerance <= 0:
        raise ValueError("tolerance must be greater than zero")
    numerator, denominator = np.zeros(n), np.zeros(n)
    for a, b in H36M_BONES:
        target = bone_lengths.get((a, b))
        if target is None or target <= 0:
            target = bone_lengths.get((b, a))
        if target is None or not np.isfinite(target) or target <= 0:
            continue
        error = abs(float(np.linalg.norm(points_3d[a] - points_3d[b])) - target) / target
        numerator[a] += reliability[b] * error
        denominator[a] += reliability[b]
        numerator[b] += reliability[a] * error
        denominator[b] += reliability[a]
    has_prior = denominator > 1e-8
    error = np.divide(numerator, denominator, out=np.zeros(n), where=has_prior)
    valid, invalid, unknown = error_to_bba(error, tolerance, cap=0.80)
    return (
        np.where(has_prior, valid, 0.0),
        np.where(has_prior, invalid, 0.0),
        np.where(has_prior, unknown, 1.0),
    )


def dempster_combine_pair(bba1, bba2):
    return combine(bba1, bba2, rule="dempster")


def compute_per_camera_weights(
    p1: np.ndarray,
    p2: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
    conf1: np.ndarray | None,
    conf2: np.ndarray | None,
    initial_3d: np.ndarray,
    *,
    weights_dst: np.ndarray | None = None,
    is_outlier: np.ndarray | None = None,
    reproj_scale: float = 12.0,
    floor: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute decoupled per-camera weights w1, w2 for DST data loss (Section 3.1)."""
    n = len(points1)
    c1 = np.ones(n, dtype=float) if conf1 is None else np.clip(np.asarray(conf1, dtype=float).flatten()[:n], 0.0, 1.0)
    c2 = np.ones(n, dtype=float) if conf2 is None else np.clip(np.asarray(conf2, dtype=float).flatten()[:n], 0.0, 1.0)

    def _proj_err(p, pts_2d):
        m, t = p[:, :3], p[:, 3]
        proj_h = initial_3d @ m.T + t
        denom = np.maximum(proj_h[:, 2:], 1e-6)
        return np.linalg.norm((proj_h[:, :2] / denom) - pts_2d, axis=-1)

    try:
        err1 = _proj_err(p1, points1)
        err2 = _proj_err(p2, points2)
    except Exception:
        err1, err2 = np.zeros(n), np.zeros(n)

    v1_r, nv1_r, u1_r = error_to_bba(err1, scale=reproj_scale)
    v2_r, nv2_r, u2_r = error_to_bba(err2, scale=reproj_scale)

    v1_d, nv1_d = c1 ** 2, (1.0 - c1) ** 2
    u1_d = np.clip(1.0 - v1_d - nv1_d, 0.02, 1.0)

    v2_d, nv2_d = c2 ** 2, (1.0 - c2) ** 2
    u2_d = np.clip(1.0 - v2_d - nv2_d, 0.02, 1.0)

    v1_c, _, u1_c, _ = combine((v1_d, nv1_d, u1_d), (v1_r, nv1_r, u1_r), rule="yager")
    v2_c, _, u2_c, _ = combine((v2_d, nv2_d, u2_d), (v2_r, nv2_r, u2_r), rule="yager")

    w1 = np.clip(v1_c + 0.3 * u1_c, floor, 1.0)
    w2 = np.clip(v2_c + 0.3 * u2_c, floor, 1.0)

    # ponytail: decouple outlier penalty per camera; each camera penalized only for its own occlusion
    w1 = np.where(c1 <= 0.3, np.maximum(w1 * 0.2, floor), w1)
    w2 = np.where(c2 <= 0.3, np.maximum(w2 * 0.2, floor), w2)

    if is_outlier is not None:
        geom_outlier = np.asarray(is_outlier, dtype=bool) & (c1 > 0.3) & (c2 > 0.3)
        w1 = np.where(geom_outlier, np.maximum(w1 * 0.5, floor), w1)
        w2 = np.where(geom_outlier, np.maximum(w2 * 0.5, floor), w2)

    return w1, w2


def fuse_evidences(
    conf_a: np.ndarray,
    conf_b: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
    bone_lengths: dict,
    *,
    initial_3d: np.ndarray | None = None,
    conflict_threshold: float = 0.65,
    sources: tuple[str, ...] | list[str] = ("detector", "epipolar", "bone"),
    epi_mode: str = "sampson",
    rule: str = "yager",
    unknown_trust: float = 0.3,
    bone_mode: str = "joint",
    return_per_camera: bool = False,
) -> tuple[np.ndarray, ...]:
    valid_sources = {"detector", "epipolar", "bone"}
    if set(sources) - valid_sources:
        raise ValueError(f"Unsupported fusion sources: {sorted(set(sources) - valid_sources)}")
    if epi_mode not in {"sampson", "ray_mm"}:
        raise ValueError("epi_mode must be 'sampson' or 'ray_mm'")
    if rule not in {"yager", "dempster"}:
        raise ValueError("rule must be 'yager' or 'dempster'")
    if bone_mode not in {"joint", "legacy"}:
        raise ValueError("bone_mode must be 'joint' or 'legacy'")
    if not 0.0 <= unknown_trust <= 1.0:
        raise ValueError("unknown_trust must be in [0, 1]")
    if initial_3d is None:
        initial_3d = triangulate_ray_midpoint(p1, p2, points1, points2)
    n = len(points1)
    neutral = (np.zeros(n), np.zeros(n), np.ones(n))
    use = set(sources)
    detector = compute_yolo_bba(conf_a, conf_b) if "detector" in use else neutral
    epipolar = neutral
    bone = neutral
    geometry_sources = []
    epipolar_available = False
    if "epipolar" in use:
        try:
            epipolar = (
                compute_sampson_bba(p1, p2, points1, points2)
                if epi_mode == "sampson"
                else compute_epipolar_bba(p1, p2, points1, points2)
            )
            epipolar_available = True
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            # Degenerate estimated cameras provide no usable geometric evidence.
            epipolar = neutral
        if epipolar_available and epi_mode == "sampson":
            epipolar = discount(epipolar, triangulation_alpha(p1, p2, points1, points2))
        if epipolar_available:
            geometry_sources.append(epipolar)
    if "bone" in use:
        joint_reliability = 0.1 + 0.9 * epipolar[0] if epipolar_available else None
        bone_function = compute_bone_bba_legacy if bone_mode == "legacy" else compute_bone_bba
        if bone_mode == "legacy":
            bone = bone_function(initial_3d, bone_lengths)
        else:
            bone = bone_function(
                initial_3d, bone_lengths, joint_reliability=joint_reliability
            )
        geometry_sources.append(bone)
    geometry = (
        tuple(sum(component) / len(geometry_sources) for component in zip(*geometry_sources))
        if geometry_sources else neutral
    )
    if len(geometry_sources) == 2:
        epi_v, epi_nv, _ = epipolar
        bone_v, bone_nv, _ = bone
        geometry_conflict = epi_v * bone_nv + epi_nv * bone_v
    else:
        geometry_conflict = np.zeros(n)
    vf, nvf, uf, detector_conflict = combine(detector, geometry, rule)
    total_conflict = np.maximum(detector_conflict, geometry_conflict)
    if rule == "yager":
        weights = np.clip(vf + unknown_trust * uf, 0.02, 1.0)
    else:
        weights = np.clip(vf * (1.0 - total_conflict), 0.02, 1.0)
    is_outlier = (total_conflict > conflict_threshold) | (nvf > vf)
    if return_per_camera:
        w_cam1, w_cam2 = compute_per_camera_weights(
            p1, p2, points1, points2, conf_a, conf_b, initial_3d,
            weights_dst=weights, is_outlier=is_outlier,
        )
        return weights, total_conflict, is_outlier, w_cam1, w_cam2
    return weights, total_conflict, is_outlier


def detect_stereo_occlusions(
    p1: np.ndarray,
    p2: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
    conf1: np.ndarray | None = None,
    conf2: np.ndarray | None = None,
    bone_lengths: dict | None = None,
    initial_3d: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Automatic multi-criteria stereo occlusion & reliability detector for 2-camera pose estimation:
    1. 2D missing/invalid coordinates or low detector confidence (<= 0.3).
    2. Epipolar constraint violation (Sampson distance above baseline).
    3. Biomechanical bone deformation (> 35% error vs anatomical prior).
    4. Dempster-Shafer fusion outlier (is_outlier from fused evidence).
    """
    pts1 = np.asarray(points1, dtype=float)
    pts2 = np.asarray(points2, dtype=float)
    n = len(pts1)

    c1 = np.ones(n, dtype=float) if conf1 is None else np.asarray(conf1, dtype=float).flatten()[:n].copy()
    c2 = np.ones(n, dtype=float) if conf2 is None else np.asarray(conf2, dtype=float).flatten()[:n].copy()

    # Invalidate coordinates with NaN or zero
    c1[~np.isfinite(pts1).all(axis=-1) | (pts1[:, 0] == 0) & (pts1[:, 1] == 0)] = 0.0
    c2[~np.isfinite(pts2).all(axis=-1) | (pts2[:, 0] == 0) & (pts2[:, 1] == 0)] = 0.0

    # 1. Sampson residual (pixels)
    try:
        fundamental = fundamental_from_projections(p1, p2)
        h1 = np.column_stack((pts1, np.ones(n)))
        h2 = np.column_stack((pts2, np.ones(n)))
        f_x1 = h1 @ fundamental.T
        ft_x2 = h2 @ fundamental
        num = np.sum(h2 * f_x1, axis=1) ** 2
        denom = f_x1[:, 0] ** 2 + f_x1[:, 1] ** 2 + ft_x2[:, 0] ** 2 + ft_x2[:, 1] ** 2
        res = np.sqrt(num / np.maximum(denom, 1e-12))
    except Exception:
        res = np.zeros(n)

    # 2. Biomechanical bone errors
    if initial_3d is None:
        try:
            from .geometry import triangulate_dlt
            initial_3d = triangulate_dlt(p1, p2, pts1, pts2)
        except Exception:
            initial_3d = triangulate_ray_midpoint(p1, p2, pts1, pts2)

    if bone_lengths is not None and initial_3d is not None and np.isfinite(initial_3d).all():
        bone_errs = np.array([_joint_bone_error(j, initial_3d, bone_lengths) for j in range(n)])
    else:
        bone_errs = np.zeros(n)

    # 3. DST Evidence Fusion Outlier
    if bone_lengths is not None and np.isfinite(p1).all() and np.isfinite(p2).all():
        try:
            _, _, is_out = fuse_evidences(c1, c2, p1, p2, pts1, pts2, bone_lengths, initial_3d=initial_3d)
        except Exception:
            is_out = np.zeros(n, dtype=bool)
    else:
        is_out = np.zeros(n, dtype=bool)

    # 4. Multi-criteria Occlusion Decision
    med_res = float(np.median(res)) if np.isfinite(res).all() and len(res) > 0 else 0.0
    epipolar_outlier = (res > 1.25 * med_res) & (res > 15.0) if med_res > 0 else np.zeros(n, dtype=bool)
    severe_epipolar = (res > 1.45 * med_res) if med_res > 0 else np.zeros(n, dtype=bool)
    bone_outlier = bone_errs > 0.35
    detector_occ = (c1 <= 0.3) | (c2 <= 0.3)

    occluded_mask = detector_occ | is_out | severe_epipolar | (epipolar_outlier & bone_outlier)
    # Pelvis root is unoccluded if coordinates are finite
    if np.isfinite(pts1[0]).all() and np.isfinite(pts2[0]).all():
        occluded_mask[0] = False

    # 5. Calculate realistic stereo confidence
    stereo_conf = np.zeros(n)
    for j in range(n):
        if occluded_mask[j]:
            # Occluded: confidence <= 0.3
            ratio = min(res[j] / max(med_res, 1.0), 3.0) if med_res > 0 else 2.0
            stereo_conf[j] = float(np.clip(0.30 / ratio, 0.05, 0.28))
        else:
            # Unoccluded: confidence >= 0.7
            ratio = max(res[j] / max(med_res, 1.0), 1.0) if med_res > 0 else 1.0
            stereo_conf[j] = float(np.clip(0.95 - 0.15 * (ratio - 1.0), 0.70, 0.98))

    # 6. Compute decoupled per-camera weights w1, w2 (Section 3.1)
    try:
        w_cam1, w_cam2 = compute_per_camera_weights(
            p1, p2, pts1, pts2, c1, c2, initial_3d,
            weights_dst=stereo_conf, is_outlier=occluded_mask,
        )
    except Exception:
        w_cam1, w_cam2 = c1, c2

    return {
        "occluded_mask": occluded_mask,
        "stereo_confidence": np.clip(stereo_conf, 0.0, 1.0),
        "conf_a": np.clip(w_cam1, 0.0, 1.0),
        "conf_b": np.clip(w_cam2, 0.0, 1.0),
        "is_outlier": is_out,
        "sampson_res": res,
        "bone_errs": bone_errs,
    }

