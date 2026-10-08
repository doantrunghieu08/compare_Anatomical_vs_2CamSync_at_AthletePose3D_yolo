"""Physics and biomechanical constraints for 3D human pose refinement."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional

from .geometry import (
    H36M_BONES,
    H36M_SYMMETRIC_BONES,
    compute_kinematic_prior,
    h36m_bone_lengths_from_height,
    point_to_ray_distance_torch,
)

try:
    import nimblephysics as nimble  # type: ignore
    HAS_NIMBLE = True
except ImportError:
    HAS_NIMBLE = False

KINEMATIC_PARENT = {1: 0, 2: 1, 3: 2, 4: 0, 5: 4, 6: 5, 11: 8, 12: 11, 13: 12, 14: 8, 15: 14, 16: 15}
EVAL_ORDER = [1, 4, 14, 11, 2, 5, 15, 12, 3, 6, 16, 13]


def is_nimble_available() -> bool:
    return HAS_NIMBLE


def compute_ground_contact_loss(
    points_3d: torch.Tensor,
    ground_z: float = 0.0,
    margin: float = 5.0,
) -> torch.Tensor:
    left_ankle_z = points_3d[..., 6, 2]
    right_ankle_z = points_3d[..., 3, 2]
    penalty_left = functional.relu(ground_z - left_ankle_z - margin) ** 2
    penalty_right = functional.relu(ground_z - right_ankle_z - margin) ** 2
    return torch.mean(penalty_left + penalty_right)


def compute_bone_rigidity_loss(
    points_3d: torch.Tensor,
    bone_lengths: dict[tuple[int, int], float] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    bone_weights: dict[tuple[int, int], float] | None = None,
    delta: float = 10.0,
) -> torch.Tensor:
    if isinstance(bone_lengths, tuple):
        b_idx_a, b_idx_b, target_lens, weights = bone_lengths
        if len(b_idx_a) == 0:
            return points_3d.new_tensor(0.0)
        actual_lens = torch.linalg.vector_norm(points_3d[..., b_idx_a, :] - points_3d[..., b_idx_b, :], dim=-1)
        loss = functional.huber_loss(actual_lens, target_lens, reduction="none", delta=delta)
        return torch.sum(loss * weights)
    valid_bones = [(a, b, target_len) for (a, b), target_len in bone_lengths.items() if target_len > 0]
    if not valid_bones:
        return points_3d.new_tensor(0.0)
    b_idx_a = [b[0] for b in valid_bones]
    b_idx_b = [b[1] for b in valid_bones]
    target_lens = points_3d.new_tensor([b[2] for b in valid_bones])
    weights = points_3d.new_tensor([bone_weights.get((b[0], b[1]), 1.0) if bone_weights else 1.0 for b in valid_bones])
    actual_lens = torch.linalg.vector_norm(points_3d[..., b_idx_a, :] - points_3d[..., b_idx_b, :], dim=-1)
    loss = functional.huber_loss(actual_lens, target_lens, reduction="none", delta=delta)
    return torch.sum(loss * weights)


def compute_symmetry_loss(
    points_3d: torch.Tensor,
    delta: float = 10.0,
    sym_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    if sym_tensors is not None:
        l1, l2, r1, r2 = sym_tensors
    else:
        l1 = [pair[0][0] for pair in H36M_SYMMETRIC_BONES]
        l2 = [pair[0][1] for pair in H36M_SYMMETRIC_BONES]
        r1 = [pair[1][0] for pair in H36M_SYMMETRIC_BONES]
        r2 = [pair[1][1] for pair in H36M_SYMMETRIC_BONES]
    len_left = torch.linalg.vector_norm(points_3d[..., l1, :] - points_3d[..., l2, :], dim=-1)
    len_right = torch.linalg.vector_norm(points_3d[..., r1, :] - points_3d[..., r2, :], dim=-1)
    return torch.sum(functional.huber_loss(len_left, len_right, reduction="none", delta=delta))


def _clamp_outlier_kinematics(
    init_3d: np.ndarray,
    bone_lengths: dict[tuple[int, int], float],
    is_outlier: np.ndarray,
    threshold_mm: float = 35.0,
) -> np.ndarray:
    clamped = init_3d.copy()
    for j in EVAL_ORDER:
        p = KINEMATIC_PARENT.get(j)
        if p is not None and is_outlier[j]:
            bone = (p, j) if (p, j) in bone_lengths else (j, p)
            t_len = bone_lengths.get(bone, 0.0)
            if t_len > 0:
                vec = clamped[j] - clamped[p]
                cur_l = float(np.linalg.norm(vec))
                if cur_l > 1e-6 and abs(cur_l - t_len) > threshold_mm:
                    clamped[j] = clamped[p] + vec * (t_len / cur_l)
    return clamped


def _optimize_biomechanics_step(
    points_3d: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    p1: torch.Tensor,
    p2: torch.Tensor,
    pts1: torch.Tensor,
    pts2: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bone_lengths: dict | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ground_z: float | None,
    bone_weight: float = 1.0,
    anchor_3d: torch.Tensor | None = None,
    anchor_weights: torch.Tensor | None = None,
    bone_weights: dict | None = None,
    sym_weight: float = 0.2,
    anchor_weight: float = 0.1,
    data_weight: float = 0.2,
    delta_ray_mm: float = 5.0,
    sym_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> float:
    optimizer.zero_grad()
    dist1_mm = point_to_ray_distance_torch(points_3d, p1, pts1)
    dist2_mm = point_to_ray_distance_torch(points_3d, p2, pts2)
    zero_ref = torch.zeros_like(dist1_mm)
    ray_loss1 = functional.huber_loss(dist1_mm, zero_ref, reduction="none", delta=delta_ray_mm)
    ray_loss2 = functional.huber_loss(dist2_mm, zero_ref, reduction="none", delta=delta_ray_mm)
    weighted_data = torch.sum(ray_loss1 * w1) + torch.sum(ray_loss2 * w2)
    bone_loss = compute_bone_rigidity_loss(points_3d, bone_lengths, bone_weights)
    sym_loss = compute_symmetry_loss(points_3d, sym_tensors=sym_tensors)
    loss = data_weight * weighted_data + bone_weight * bone_loss + sym_weight * sym_loss
    if anchor_3d is not None and anchor_weights is not None:
        anchor_loss = functional.huber_loss(points_3d, anchor_3d, reduction="none", delta=10.0).sum(-1)
        loss = loss + anchor_weight * torch.sum(anchor_loss * anchor_weights)
    if ground_z is not None:
        loss = loss + 0.5 * compute_ground_contact_loss(points_3d, ground_z=ground_z)
    loss.backward()
    optimizer.step()
    return float(loss.item())


_NIMBLE_MODEL = None

H36M_TO_NIMBLE = {
    0: "ground_pelvis", 1: "hip_r", 2: "walker_knee_r", 3: "ankle_r",
    4: "hip_l", 5: "walker_knee_l", 6: "ankle_l", 8: "back",
    11: "acromial_l", 12: "elbow_l", 13: "radius_hand_l",
    14: "acromial_r", 15: "elbow_r", 16: "radius_hand_r",
}


def _get_nimble_skeleton():
    global _NIMBLE_MODEL
    if _NIMBLE_MODEL is None and HAS_NIMBLE:
        _NIMBLE_MODEL = nimble.models.RajagopalHumanBodyModel()
        pelvis = _NIMBLE_MODEL.skeleton.getJoint("ground_pelvis")
        pelvis.setPositionLowerLimits(np.array([-np.pi, -np.pi, -np.pi, -100.0, -100.0, -100.0]))
        pelvis.setPositionUpperLimits(np.array([np.pi, np.pi, np.pi, 100.0, 100.0, 100.0]))
    return _NIMBLE_MODEL.skeleton if _NIMBLE_MODEL is not None else None


def fit_skeleton_nimble(
    pose_3d: np.ndarray,
    weights: np.ndarray | None = None,
    min_weight: float = 0.4,
    max_steps: int = 40,
) -> np.ndarray:
    skeleton = _get_nimble_skeleton()
    if skeleton is None:
        return pose_3d
    indices = [i for i, name in H36M_TO_NIMBLE.items() if weights is None or weights[i] >= min_weight]
    if len(indices) < 6:
        indices = list(H36M_TO_NIMBLE.keys())
    target_joints = [skeleton.getJoint(H36M_TO_NIMBLE[i]) for i in indices]
    pelvis = pose_3d[0:1]
    centered_m = (pose_3d - pelvis) / 1000.0
    targets_m = np.ascontiguousarray(centered_m[indices].flatten().reshape(-1, 1), dtype=np.float64)
    try:
        skeleton.fitJointsToWorldPositions(target_joints, targets_m, scaleBodies=False, maxStepCount=max_steps)
        jm = skeleton.getJointWorldPositionsMap()
        refined = pose_3d.copy()
        for idx, name in H36M_TO_NIMBLE.items():
            refined[idx] = (jm[name].flatten() * 1000.0) + pelvis.flatten()
        return refined
    except Exception:
        return pose_3d


def _prepare_optimization_tensors(
    confidence1, confidence2, weights_dst, is_outlier, bone_lengths, init_3d, device,
    bone_reliability_modulation: bool = False,
):
    pose = torch.tensor(init_3d, dtype=torch.float32, device=device, requires_grad=True)
    anchor_t = torch.as_tensor(init_3d, dtype=torch.float32, device=device)
    c1, c2 = np.clip(confidence1, 0.0, 1.0), np.clip(confidence2, 0.0, 1.0)
    w_dst = np.sqrt(c1 * c2) if weights_dst is None else weights_dst
    out_mask = np.zeros(len(c1), dtype=bool) if is_outlier is None else is_outlier
    reliability = np.clip(w_dst * np.where(out_mask, 0.5, 1.0), 0.02, 1.0)
    reliability = np.where((c1 > 0) & (c2 > 0), reliability, 0.0)
    reproj_dst = reliability
    w1_t = torch.as_tensor(reproj_dst, dtype=torch.float32, device=device)
    w2_t = torch.as_tensor(reproj_dst, dtype=torch.float32, device=device)
    anchor_w = torch.as_tensor(reliability, dtype=torch.float32, device=device)
    if bone_reliability_modulation:
        bone_w = {b: max(1.0 - reliability[b[0]], 1.0 - reliability[b[1]]) for b in bone_lengths}
    else:
        bone_w = {b: 1.0 for b in bone_lengths}
    valid_bones = [(a, b, t_len) for (a, b), t_len in bone_lengths.items() if t_len > 0]
    if valid_bones:
        b_idx_a = torch.tensor([b[0] for b in valid_bones], dtype=torch.long, device=device)
        b_idx_b = torch.tensor([b[1] for b in valid_bones], dtype=torch.long, device=device)
        b_targets = torch.tensor([b[2] for b in valid_bones], dtype=torch.float32, device=device)
        b_weights = torch.tensor([bone_w.get((b[0], b[1]), 1.0) for b in valid_bones], dtype=torch.float32, device=device)
        bone_tensors = (b_idx_a, b_idx_b, b_targets, b_weights)
    else:
        bone_tensors = (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
        )
    sym_tensors = (
        torch.tensor([pair[0][0] for pair in H36M_SYMMETRIC_BONES], dtype=torch.long, device=device),
        torch.tensor([pair[0][1] for pair in H36M_SYMMETRIC_BONES], dtype=torch.long, device=device),
        torch.tensor([pair[1][0] for pair in H36M_SYMMETRIC_BONES], dtype=torch.long, device=device),
        torch.tensor([pair[1][1] for pair in H36M_SYMMETRIC_BONES], dtype=torch.long, device=device),
    )
    return pose, anchor_t, w1_t, w2_t, anchor_w, bone_tensors, sym_tensors


def triangulate_physics_refine(
    p1: np.ndarray, p2: np.ndarray, points1: np.ndarray, points2: np.ndarray,
    confidence1: np.ndarray, confidence2: np.ndarray, *,
    bone_lengths: dict | None = None, bone_weight: float = 1.0,
    weights_dst: np.ndarray | None = None, is_outlier: np.ndarray | None = None,
    iterations: int = 60, ground_z: float | None = None, lr: float = 1.0,
    use_nimble_ik: bool = False, data_weight: float = 0.2, anchor_weight: float = 0.1,
    sym_weight: float = 0.2, bone_reliability_modulation: bool = False,
    delta_ray_mm: float = 5.0,
    init_3d: np.ndarray | None = None,
) -> np.ndarray:
    if init_3d is None:
        if weights_dst is not None:
            from .evidence_fusion import triangulate_ray_midpoint
            init_3d = triangulate_ray_midpoint(p1, p2, points1, points2)
        else:
            from .geometry import triangulate_dlt
            init_3d = triangulate_dlt(p1, p2, points1, points2)
    if use_nimble_ik and HAS_NIMBLE:
        return fit_skeleton_nimble(init_3d, weights=weights_dst, max_steps=min(iterations, 50))
    bone_lengths = bone_lengths or h36m_bone_lengths_from_height()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pose, anchor_t, w1_t, w2_t, anchor_w, bone_tensors, sym_tensors = _prepare_optimization_tensors(
        confidence1, confidence2, weights_dst, is_outlier, bone_lengths, init_3d, device,
        bone_reliability_modulation=bone_reliability_modulation,
    )
    p1_t, p2_t = torch.as_tensor(p1, dtype=torch.float32, device=device), torch.as_tensor(p2, dtype=torch.float32, device=device)
    pts1_t, pts2_t = torch.as_tensor(points1, dtype=torch.float32, device=device), torch.as_tensor(points2, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([pose], lr=lr)
    best_pose, best_loss = pose.detach().clone(), float("inf")
    for _ in range(iterations):
        loss = _optimize_biomechanics_step(
            pose, optimizer, p1_t, p2_t, pts1_t, pts2_t, w1_t, w2_t, bone_tensors,
            ground_z, bone_weight=bone_weight, anchor_3d=anchor_t, anchor_weights=anchor_w,
            sym_weight=sym_weight, anchor_weight=anchor_weight,
            data_weight=data_weight, delta_ray_mm=delta_ray_mm, sym_tensors=sym_tensors,
        )
        if np.isfinite(loss) and loss < best_loss:
            best_pose = pose.detach().clone()
            best_loss = loss
    result = best_pose.cpu().numpy()
    return result if np.isfinite(result).all() else init_3d


def triangulate_dst_physics(
    p1: np.ndarray, p2: np.ndarray, points1: np.ndarray, points2: np.ndarray,
    confidence1: np.ndarray, confidence2: np.ndarray, *,
    bone_lengths: dict | None = None, bone_weight: float = 1.0,
    iterations: int = 60, ground_z: float | None = None, lr: float = 1.0,
    use_nimble_ik: bool = False, data_weight: float = 0.2, anchor_weight: float = 0.1,
    sym_weight: float = 0.2, bone_reliability_modulation: bool = False,
    delta_ray_mm: float = 5.0,
    fusion_sources: tuple[str, ...] | list[str] = ("detector", "epipolar", "bone"),
    fusion_epi_mode: str = "sampson", fusion_rule: str = "yager",
    fusion_unknown_trust: float = 0.3, fusion_bone_mode: str = "joint",
) -> np.ndarray:
    from .evidence_fusion import fuse_evidences
    from .geometry import triangulate_dlt
    bone_lengths = bone_lengths or h36m_bone_lengths_from_height()
    weights_dst, _, is_outlier = fuse_evidences(
        confidence1, confidence2, p1, p2, points1, points2, bone_lengths,
        sources=fusion_sources, epi_mode=fusion_epi_mode, rule=fusion_rule,
        unknown_trust=fusion_unknown_trust, bone_mode=fusion_bone_mode,
    )
    init_3d = triangulate_dlt(p1, p2, points1, points2)
    return triangulate_physics_refine(
        p1, p2, points1, points2, confidence1, confidence2,
        bone_lengths=bone_lengths, bone_weight=bone_weight,
        weights_dst=weights_dst, is_outlier=is_outlier,
        iterations=iterations, ground_z=ground_z, lr=lr,
        use_nimble_ik=use_nimble_ik, data_weight=data_weight,
        anchor_weight=anchor_weight, sym_weight=sym_weight,
        bone_reliability_modulation=bone_reliability_modulation,
        delta_ray_mm=delta_ray_mm,
        init_3d=init_3d,
    )

