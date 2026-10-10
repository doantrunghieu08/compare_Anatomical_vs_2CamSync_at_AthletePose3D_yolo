#!/usr/bin/env python3
"""Evaluate candidate constraints to see which ones measurably reduce MPJPE and PA-MPJPE."""

import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

import numpy as np
import torch
import torch.nn.functional as functional

from athlete_pose3d.algorithms.geometry import (
    H36M_BONES, H36M_SYMMETRIC_BONES,
    compute_kinematic_prior,
    h36m_bone_lengths_from_height, mpjpe, pa_mpjpe,
    triangulate_dlt, _camera_rays,
)
from athlete_pose3d.algorithms.physics_refine import compute_symmetry_loss
from athlete_pose3d.io.data import PoseRepository, load_json
from athlete_pose3d.settings import load_settings
from athlete_pose3d.main import _eligible_pairs
from athlete_pose3d.pipeline import _frame_inputs, select_best_pair, _fps_scaled_sync_params, _add_dynamic_sync, _sync_score_options

# Define Derived vs Clear joints in H36M
DERIVED_JOINTS = {0, 1, 4, 7, 8, 9, 10}  # Pelvis, Hips, Spine, Thorax, Neck, Head
CLEAR_JOINTS = {2, 3, 5, 6, 11, 12, 13, 14, 15, 16}  # Knees, Ankles, Shoulders, Elbows, Wrists


def custom_anatomical_step(
    points_3d: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    p1_t: torch.Tensor, p2_t: torch.Tensor,
    pts1_t: torch.Tensor, pts2_t: torch.Tensor,
    c1_t: torch.Tensor, c2_t: torch.Tensor,
    bone_tensors: tuple,
    rays_pre: tuple,
    # Ablation toggles:
    derived_ray_weight: float = 1.0,
    sym_weight: float = 0.0,
    kin_weight: float = 0.0,
    spine_weight: float = 0.0,
    sym_tensors: tuple = None,
) -> float:
    optimizer.zero_grad()
    (c1, d1), (c2, d2) = rays_pre
    diff1 = points_3d - c1
    diff2 = points_3d - c2
    dist1 = torch.linalg.vector_norm(diff1 - torch.sum(diff1 * d1, dim=-1, keepdim=True) * d1, dim=-1)
    dist2 = torch.linalg.vector_norm(diff2 - torch.sum(diff2 * d2, dim=-1, keepdim=True) * d2, dim=-1)
    
    loss1 = functional.huber_loss(dist1, torch.zeros_like(dist1), reduction="none", delta=5.0)
    loss2 = functional.huber_loss(dist2, torch.zeros_like(dist2), reduction="none", delta=5.0)
    
    # Weight down derived joints if specified
    if derived_ray_weight != 1.0:
        w_mask = points_3d.new_ones(17)
        for j in DERIVED_JOINTS:
            w_mask[j] = derived_ray_weight
        loss1 = loss1 * w_mask
        loss2 = loss2 * w_mask

    b_idx_a, b_idx_b, b_targets = bone_tensors
    actuals = torch.linalg.vector_norm(points_3d[b_idx_a] - points_3d[b_idx_b], dim=-1)
    bone_loss = torch.sum(functional.huber_loss(actuals, b_targets, reduction="none", delta=10.0))

    total_loss = torch.sum(loss1 * c1_t) + torch.sum(loss2 * c2_t) + 1.0 * bone_loss

    # 1. Symmetry constraint
    if sym_weight > 0.0:
        total_loss = total_loss + sym_weight * compute_symmetry_loss(points_3d, sym_tensors=sym_tensors)

    # 2. Kinematic prior (knee/elbow hyperextension)
    if kin_weight > 0.0:
        total_loss = total_loss + kin_weight * compute_kinematic_prior(points_3d)

    # 3. Spine column alignment
    if spine_weight > 0.0:
        v_s1 = functional.normalize(points_3d[7] - points_3d[0], dim=-1, eps=1e-6)
        v_s2 = functional.normalize(points_3d[8] - points_3d[7], dim=-1, eps=1e-6)
        v_s3 = functional.normalize(points_3d[9] - points_3d[8], dim=-1, eps=1e-6)
        spine_loss = functional.relu(0.85 - torch.dot(v_s1, v_s2)) ** 2 + functional.relu(0.85 - torch.dot(v_s2, v_s3)) ** 2
        total_loss = total_loss + spine_weight * spine_loss

    total_loss.backward()
    optimizer.step()
    return float(total_loss.item())


def run_eval(triangulate_fn, data_list):
    mpjpes, pa_mpjpes = [], []
    for (p1, p2, l_kps, r_kps, l_conf, r_conf, gt) in data_list:
        pred = triangulate_fn(p1, p2, l_kps, r_kps, l_conf, r_conf)
        mpjpes.append(mpjpe(pred, gt))
        pa_mpjpes.append(pa_mpjpe(pred, gt))
    return np.mean(mpjpes), np.mean(pa_mpjpes)


def main():
    inputs, output, cfg = load_settings(repo_root / "configs/default.yml")
    repo = PoseRepository(inputs.pose2d_root, inputs.pose_2d_suffix, inputs.key_frames_suffix, inputs.pose_cache_limit)
    pairs = _eligible_pairs(inputs, cfg)
    s1_pairs = [p for p in pairs if p["subject"] == "S1"]
    pair_info = s1_pairs[0]
    sync_p = _fps_scaled_sync_params(pair_info, cfg)
    score_opts = _sync_score_options(cfg, "S1")
    best_pair = select_best_pair(
        pair_info, repo, max_candidates=cfg.max_pair_candidates,
        max_offset=sync_p["max_offset"], sync_samples=cfg.sync_samples,
        coarse_step=sync_p["coarse_step"], minimum_valid_ratio=cfg.minimum_pair_valid_ratio,
        validation_radius=sync_p["local_radius"], score_options=score_opts,
    )
    cam_a, cam_b = best_pair["cam_a"], best_pair["cam_b"]
    v_a, v_b = cam_a["video_path"], cam_b["video_path"]
    gt_path = Path(cam_a["ground_truth_path"])
    raw_gt = np.load(gt_path, allow_pickle=False)
    frame_count = min(best_pair["n_a"], best_pair["n_b"], len(raw_gt))
    metadata_path = Path(cam_a["metadata_path"])
    marker_names = load_json(metadata_path)["keypoint_name"]
    joint_markers = cfg.subject_gt_joint_markers.get("S1")
    p1, p2 = best_pair["P1"], best_pair["P2"]
    bone_lengths = h36m_bone_lengths_from_height(1591.0)

    start_f = max(20, frame_count // 3)
    num_f = 15
    contiguous_frames = list(range(start_f, min(start_f + num_f, frame_count)))
    context = {
        "pair_info": pair_info, "best_pair": best_pair, "camera_a": cam_a, "camera_b": cam_b,
        "video_a": v_a, "video_b": v_b, "raw_gt": raw_gt, "frames": contiguous_frames,
        "marker_names": marker_names, "joint_markers": joint_markers, "p1": p1, "p2": p2,
        "subject_height_mm": 1591.0, "bone_lengths": bone_lengths,
        "score_options": score_opts,
        "fps": sync_p["fps"], "sync_local_radius": sync_p["local_radius"],
        "dynamic_local_radius": sync_p["dynamic_radius"],
        "dynamic_transition_weight": sync_p["transition_weight"],
    }
    _add_dynamic_sync(context, repo, cfg)

    # Collect data for 60 frames
    data_list = []
    for f in contiguous_frames:
        inp = _frame_inputs(context, f, repo, cfg)
        if inp is not None:
            left, right, gt = inp[0], inp[1], inp[-1]
            data_list.append((p1, p2, left["kps_h36m"], right["kps_h36m"], left["conf_h36m"], right["conf_h36m"], gt))

    print(f"Loaded {len(data_list)} frames.")

    # Symmetry tensor precomputation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    l1 = torch.tensor([pair[0][0] for pair in H36M_SYMMETRIC_BONES], dtype=torch.long, device=device)
    l2 = torch.tensor([pair[0][1] for pair in H36M_SYMMETRIC_BONES], dtype=torch.long, device=device)
    r1 = torch.tensor([pair[1][0] for pair in H36M_SYMMETRIC_BONES], dtype=torch.long, device=device)
    r2 = torch.tensor([pair[1][1] for pair in H36M_SYMMETRIC_BONES], dtype=torch.long, device=device)
    sym_tensors = (l1, l2, r1, r2)

    valid_bones = [(a, b, l) for (a, b), l in bone_lengths.items() if l > 0]
    b_idx_a = torch.tensor([v[0] for v in valid_bones], dtype=torch.long, device=device)
    b_idx_b = torch.tensor([v[1] for v in valid_bones], dtype=torch.long, device=device)
    b_targets = torch.tensor([v[2] for v in valid_bones], dtype=torch.float32, device=device)
    bone_tensors = (b_idx_a, b_idx_b, b_targets)

    # Factory for solver
    def make_solver(derived_w=1.0, sym_w=0.0, kin_w=0.0, spine_w=0.0, iters=80):
        def solve(p1_np, p2_np, pts1_np, pts2_np, c1_np, c2_np):
            init_3d = triangulate_dlt(p1_np, p2_np, pts1_np, pts2_np)
            points_3d = torch.tensor(init_3d, dtype=torch.float32, device=device, requires_grad=True)
            p1_t, p2_t = (torch.as_tensor(v, dtype=torch.float32, device=device) for v in (p1_np, p2_np))
            pts1_t, pts2_t = (torch.as_tensor(v, dtype=torch.float32, device=device) for v in (pts1_np, pts2_np))
            c1_t, c2_t = (torch.as_tensor(np.clip(v, 0.0, 1.0) ** 2, dtype=torch.float32, device=device) for v in (c1_np, c2_np))
            rays_pre = (_camera_rays(p1_t, pts1_t), _camera_rays(p2_t, pts2_t))
            optimizer = torch.optim.Adam([points_3d], lr=0.1)

            best_pose, best_loss = points_3d.detach().clone(), float("inf")
            for _ in range(iters):
                candidate = points_3d.detach().clone()
                loss_val = custom_anatomical_step(
                    points_3d, optimizer, p1_t, p2_t, pts1_t, pts2_t, c1_t, c2_t,
                    bone_tensors, rays_pre,
                    derived_ray_weight=derived_w, sym_weight=sym_w,
                    kin_weight=kin_w, spine_weight=spine_w,
                    sym_tensors=sym_tensors,
                )
                if np.isfinite(loss_val) and loss_val < best_loss:
                    best_pose = candidate
                    best_loss = loss_val
            res = best_pose.detach().cpu().numpy()
            return res if np.isfinite(res).all() else init_3d
        return solve

    experiments = [
        ("0. Baseline Anatomical", 1.0, 0.0, 0.0, 0.0),
        ("1. + Symmetry Loss (sym=0.2)", 1.0, 0.2, 0.0, 0.0),
        ("2. + Kinematic Hyperextension (kin=0.15)", 1.0, 0.0, 0.15, 0.0),
        ("3. + Spine Column Alignment (spine=0.15)", 1.0, 0.0, 0.0, 0.15),
        ("4. + Derived Ray Downweight (derived=0.4)", 0.4, 0.0, 0.0, 0.0),
        ("5. + Derived Ray Downweight (derived=0.2)", 0.2, 0.0, 0.0, 0.0),
        ("6. + Sym(0.2) + Kin(0.15) + Spine(0.15)", 1.0, 0.2, 0.15, 0.15),
        ("7. Combined Best (Derived=0.3 + Sym=0.2 + Spine=0.15 + Kin=0.1)", 0.3, 0.2, 0.1, 0.15),
    ]

    base_m, base_p = None, None
    print("\n" + "=" * 80)
    print(f"{'Experiment':<45} | {'MPJPE (mm)':<10} | {'PA-MPJPE (mm)':<12} | {'Delta MPJPE'}")
    print("=" * 80)

    for name, d_w, s_w, k_w, sp_w in experiments:
        t0 = time.time()
        solver = make_solver(derived_w=d_w, sym_w=s_w, kin_w=k_w, spine_w=sp_w)
        m, p = run_eval(solver, data_list)
        el = time.time() - t0
        if base_m is None:
            base_m, base_p = m, p
            print(f"{name:<45} | {m:>8.2f} mm | {p:>10.2f} mm |   baseline", flush=True)
        else:
            del_m = (base_m - m) / base_m * 100.0
            del_p = (base_p - p) / base_p * 100.0
            print(f"{name:<45} | {m:>8.2f} mm | {p:>10.2f} mm |  {del_m:>+6.2f}% (PA: {del_p:>+5.2f}%) [{el:.1f}s]", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
