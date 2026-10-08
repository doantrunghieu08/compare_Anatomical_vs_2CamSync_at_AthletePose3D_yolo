from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from ..settings import REFINED_METHOD, TWOCAM_METHOD
from .geometry import (
    H36M_BONES, H36M_SYMMETRIC_BONES,
    mpjpe, pa_mpjpe,
)


def _sequence_bone_targets(sequence, bone_lengths=None):
    targets = {}
    for a, b in H36M_BONES:
        lengths = np.asarray(
            [np.linalg.norm(pose[a] - pose[b]) for pose in sequence if np.isfinite(pose).all()]
        )
        lengths = lengths[np.isfinite(lengths) & (lengths > 1e-6)]
        prior = (bone_lengths or {}).get((a, b), 0)
        if not len(lengths) and prior <= 0:
            continue
        median = float(np.median(lengths)) if len(lengths) else float(prior)
        targets[(a, b)] = median
    return targets


def _torch_ray_distances(poses, projections, points):
    matrices = projections[:, :, :3]
    inverses = torch.linalg.inv(matrices)
    centers = -torch.einsum("tij,tj->ti", inverses, projections[:, :, 3])
    homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
    rays = torch.einsum("tij,tkj->tki", inverses, homogeneous)
    rays = torch.nn.functional.normalize(rays, dim=-1)
    offsets = poses - centers[:, None, :]
    along = torch.sum(offsets * rays, dim=-1, keepdim=True)
    return torch.linalg.vector_norm(offsets - along * rays, dim=-1)


def _robust_cost(residuals, scale=35.0):
    scaled = residuals / scale
    return torch.sum(2.0 * scale**2 * (torch.sqrt(1.0 + scaled**2) - 1.0))


def _optimize_sequence_torch(sequence, items, targets, weights, max_iterations):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = torch.as_tensor(sequence, dtype=torch.float32, device=device)
    poses = torch.nn.Parameter(base.clone())
    optimizer = torch.optim.LBFGS(
        [poses], lr=1.0, max_iter=max_iterations, tolerance_grad=1e-5,
        tolerance_change=1e-7, line_search_fn="strong_wolfe",
    )
    bones = torch.tensor([pair for pair in H36M_BONES], dtype=torch.long, device=device)
    bone_targets = torch.tensor([targets.get(tuple(pair), 0.0) for pair in H36M_BONES], device=device)
    symmetric = torch.tensor(
        [[l1, l2, r1, r2] for (l1, l2), (r1, r2) in H36M_SYMMETRIC_BONES],
        dtype=torch.long, device=device,
    )

    camera_data = []
    for projection_key, points_key, confidence_key in (
            ("P1", "kps2d_a_h36m", "conf_a_h36m"),
            ("P2", "kps2d_b_h36m", "conf_b_h36m"),
    ):
        projections, points, confidences = [], [], []
        for _, result in items:
            projection = result.get(projection_key)
            keypoints = result.get(points_key)
            if projection is None or keypoints is None:
                projections.append(np.column_stack((np.eye(3), np.zeros(3))))
                points.append(np.zeros((sequence.shape[1], 2)))
                confidences.append(np.zeros(sequence.shape[1]))
                continue
            confidence = result.get(confidence_key)
            confidence = np.ones(len(keypoints)) if confidence is None else np.asarray(confidence)
            valid = (confidence > 0) & np.isfinite(keypoints).all(axis=-1)
            projections.append(projection)
            points.append(np.where(valid[:, None], keypoints, 0.0))
            confidences.append(np.sqrt(np.clip(confidence, 0.0, 1.0)) * valid)
        camera_data.append((
            torch.as_tensor(np.asarray(projections), dtype=torch.float32, device=device),
            torch.as_tensor(np.asarray(points), dtype=torch.float32, device=device),
            torch.as_tensor(np.asarray(confidences), dtype=torch.float32, device=device),
        ))

    frame_nums = np.array([res["frame"] for _, res in items])
    is_contiguous = len(frame_nums) >= 3 and bool(np.all(np.diff(frame_nums) == 1))

    def closure():
        optimizer.zero_grad()
        root = weights["root"] * (poses[:, 0] - base[:, 0])
        relative = weights["data"] * ((poses[:, 1:] - poses[:, :1]) - (base[:, 1:] - base[:, :1]))
        lengths = torch.linalg.vector_norm(poses[:, bones[:, 0]] - poses[:, bones[:, 1]], dim=-1)
        bone_residual = weights["bone"] * (lengths - bone_targets)
        sym_l = torch.linalg.vector_norm(poses[:, symmetric[:, 0]] - poses[:, symmetric[:, 1]], dim=-1)
        sym_r = torch.linalg.vector_norm(poses[:, symmetric[:, 2]] - poses[:, symmetric[:, 3]], dim=-1)
        residuals = [root.reshape(-1), relative.reshape(-1), bone_residual.reshape(-1),
                     (weights["symmetry"] * (sym_l - sym_r)).reshape(-1)]
        for projections, points, confidence in camera_data:
            distance = _torch_ray_distances(poses, projections, points)
            residuals.append((weights["reprojection"] * confidence * distance).reshape(-1))
        if is_contiguous:
            centered = poses - poses[:, :1]
            smooth = centered[:-2] - 2.0 * centered[1:-1] + centered[2:]
            residuals.append((weights["smoothness"] * smooth).reshape(-1))
        loss = _robust_cost(torch.cat(residuals))
        loss.backward()
        return loss

    optimizer.step(closure)
    return poses.detach().cpu().numpy()


def optimize_sequence(
    items, source_method=TWOCAM_METHOD, *, bone_lengths=None, data_weight=0.40,
    root_weight=1.0, bone_weight=0.70, reprojection_weight=0.42,
    smoothness_weight=0.060, symmetry_weight=0.12, max_evaluations=90, max_drift=220.0,
):
    sequence = np.array([result["all_methods"][source_method]["recon_3d"] for _, result in items], dtype=float)
    if not len(sequence):
        return sequence
    targets = _sequence_bone_targets(sequence, bone_lengths)
    weights = {
        "data": data_weight, "root": root_weight, "bone": bone_weight,
        "reprojection": reprojection_weight, "smoothness": smoothness_weight, "symmetry": symmetry_weight,
    }
    optimized = _optimize_sequence_torch(sequence, items, targets, weights, max_evaluations)
    if not np.isfinite(optimized).all() or np.median(np.linalg.norm(optimized - sequence, axis=2)) > max_drift:
        return sequence
    return optimized


def refine_results(
    results,
    source_method=TWOCAM_METHOD,
    target_method=REFINED_METHOD,
    chunk_size=100,
    **optimization_options,
):
    grouped = defaultdict(list)
    for index, result in enumerate(results):
        if source_method in result["all_methods"]:
            key = (result["subject"], result["motion"], result["cam_a"], result["cam_b"])
            grouped[key].append((index, result))

    for items in grouped.values():
        items.sort(key=lambda item: item[1]["frame"])
        refined = []
        for start in range(0, len(items), chunk_size):
            refined.extend(
                optimize_sequence(items[start : start + chunk_size], source_method, **optimization_options)
            )
        for (index, result), prediction in zip(items, refined):
            metrics = {
                "recon_3d": prediction,
                "mpjpe": mpjpe(prediction, result["gt_3d"]),
                "pa_mpjpe": pa_mpjpe(prediction, result["gt_3d"]),
            }
            results[index]["all_methods"][target_method] = metrics
            results[index].update(metrics, best_method=target_method)
    return results
