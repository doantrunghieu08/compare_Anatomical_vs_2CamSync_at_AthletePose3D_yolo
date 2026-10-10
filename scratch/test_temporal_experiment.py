#!/usr/bin/env python3
"""Empirical test: Does temporal smoothness & velocity constraint improve MPJPE on contiguous frames?"""

import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

import numpy as np
import torch

from athlete_pose3d.algorithms.geometry import (
    H36M_BONES, H36M_SYMMETRIC_BONES,
    h36m_bone_lengths_from_height, mpjpe, pa_mpjpe,
    triangulate_anatomical,
)
from athlete_pose3d.algorithms.refinement import (
    _sequence_bone_targets, _torch_ray_distances, _robust_cost,
)
from athlete_pose3d.io.data import PoseRepository, load_json
from athlete_pose3d.settings import BenchmarkConfig
from athlete_pose3d.pipeline import _frame_inputs, select_best_pair, _fps_scaled_sync_params, _add_dynamic_sync


def test_contiguous_window():
    from athlete_pose3d.settings import load_settings
    inputs, output, cfg = load_settings(repo_root / "configs/default.yml")
    repo = PoseRepository(inputs.pose2d_root, inputs.pose_2d_suffix, inputs.key_frames_suffix, inputs.pose_cache_limit)
    
    from athlete_pose3d.main import _eligible_pairs
    pairs = _eligible_pairs(inputs, cfg)
    s1_pairs = [p for p in pairs if p["subject"] == "S1"]
    if not s1_pairs:
        print("No S1 pairs found.")
        return

    pair_info = s1_pairs[0]
    sync_p = _fps_scaled_sync_params(pair_info, cfg)
    from athlete_pose3d.pipeline import _sync_score_options
    score_opts = _sync_score_options(cfg, "S1")
    best_pair = select_best_pair(
        pair_info, repo, max_candidates=cfg.max_pair_candidates,
        max_offset=sync_p["max_offset"], sync_samples=cfg.sync_samples,
        coarse_step=sync_p["coarse_step"], minimum_valid_ratio=cfg.minimum_pair_valid_ratio,
        validation_radius=sync_p["local_radius"], score_options=score_opts,
    )
    if not best_pair:
        print("No best pair found.")
        return

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

    # Take 60 contiguous frames in the middle
    start_f = max(20, frame_count // 3)
    num_f = 60
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
    print(f"Testing on {len(contiguous_frames)} contiguous frames: [{contiguous_frames[0]}..{contiguous_frames[-1]}]")

    items = []
    anatomical_poses = []
    gts = []

    for f in contiguous_frames:
        inp = _frame_inputs(context, f, repo, cfg)
        if inp is None:
            continue
        left, right, gt = inp[0], inp[1], inp[-1]
        
        # Anatomical Triangulation
        anat_pt = triangulate_anatomical(
            p1, p2, left["kps_h36m"], right["kps_h36m"], left["conf_h36m"], right["conf_h36m"],
            bone_lengths=bone_lengths, bone_weight=1.0, iterations=80,
        )
        anatomical_poses.append(anat_pt)
        gts.append(gt)

        result = {
            "frame": f, "P1": p1, "P2": p2,
            "kps2d_a_h36m": left["kps_h36m"], "conf_a_h36m": left["conf_h36m"],
            "kps2d_b_h36m": right["kps_h36m"], "conf_b_h36m": right["conf_h36m"],
            "all_methods": {"Anatomical": {"recon_3d": anat_pt}},
            "gt_3d": gt,
        }
        items.append((len(items), result))

    anatomical_poses = np.array(anatomical_poses)
    gts = np.array(gts)
    
    anat_errs = [mpjpe(p, g) for p, g in zip(anatomical_poses, gts)]
    anat_pa_errs = [pa_mpjpe(p, g) for p, g in zip(anatomical_poses, gts)]
    print(f"\n[Baseline Anatomical] MPJPE: {np.mean(anat_errs):.2f} mm | PA-MPJPE: {np.mean(anat_pa_errs):.2f} mm")

    # Sequence Refine Function with Velocity support
    def run_seq_refine(smoothness_w=0.06, velocity_w=0.0, max_iter=60):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base = torch.as_tensor(anatomical_poses, dtype=torch.float32, device=device)
        poses = torch.nn.Parameter(base.clone())
        optimizer = torch.optim.LBFGS(
            [poses], lr=1.0, max_iter=max_iter, tolerance_grad=1e-5,
            tolerance_change=1e-7, line_search_fn="strong_wolfe",
        )
        bones = torch.tensor([pair for pair in H36M_BONES], dtype=torch.long, device=device)
        targets = _sequence_bone_targets(anatomical_poses, bone_lengths)
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
            projs, pts, confs = [], [], []
            for _, res in items:
                projs.append(res[projection_key])
                pts.append(res[points_key])
                conf = np.sqrt(np.clip(res[confidence_key], 0.0, 1.0))
                confs.append(conf)
            camera_data.append((
                torch.as_tensor(np.asarray(projs), dtype=torch.float32, device=device),
                torch.as_tensor(np.asarray(pts), dtype=torch.float32, device=device),
                torch.as_tensor(np.asarray(confs), dtype=torch.float32, device=device),
            ))

        def closure():
            optimizer.zero_grad()
            root = 0.8 * (poses[:, 0] - base[:, 0])
            relative = 0.34 * ((poses[:, 1:] - poses[:, :1]) - (base[:, 1:] - base[:, :1]))
            lengths = torch.linalg.vector_norm(poses[:, bones[:, 0]] - poses[:, bones[:, 1]], dim=-1)
            bone_res = 0.7 * (lengths - bone_targets)
            sym_l = torch.linalg.vector_norm(poses[:, symmetric[:, 0]] - poses[:, symmetric[:, 1]], dim=-1)
            sym_r = torch.linalg.vector_norm(poses[:, symmetric[:, 2]] - poses[:, symmetric[:, 3]], dim=-1)
            residuals = [root.reshape(-1), relative.reshape(-1), bone_res.reshape(-1), (0.12 * (sym_l - sym_r)).reshape(-1)]
            for projections, points, confidence in camera_data:
                dist = _torch_ray_distances(poses, projections, points)
                residuals.append((0.46 * confidence * dist).reshape(-1))
            
            centered = poses - poses[:, :1]
            if smoothness_w > 0:
                # Acceleration: x_{t-1} - 2x_t + x_{t+1}
                smooth = centered[:-2] - 2.0 * centered[1:-1] + centered[2:]
                residuals.append((smoothness_w * smooth).reshape(-1))
            if velocity_w > 0:
                # Velocity: x_t - x_{t-1}
                vel = centered[1:] - centered[:-1]
                residuals.append((velocity_w * vel).reshape(-1))
            
            loss = _robust_cost(torch.cat(residuals))
            loss.backward()
            return loss

        optimizer.step(closure)
        out = poses.detach().cpu().numpy()
        ref_errs = [mpjpe(p, g) for p, g in zip(out, gts)]
        ref_pa_errs = [pa_mpjpe(p, g) for p, g in zip(out, gts)]
        return np.mean(ref_errs), np.mean(ref_pa_errs)

    # Test variants
    variants = [
        ("No temporal (smooth=0, vel=0)", 0.0, 0.0),
        ("Smoothness only (smooth=0.06, vel=0)", 0.06, 0.0),
        ("Higher Smoothness (smooth=0.20, vel=0)", 0.20, 0.0),
        ("Smoothness + Velocity (smooth=0.06, vel=0.03)", 0.06, 0.03),
        ("Smoothness + Velocity (smooth=0.10, vel=0.05)", 0.10, 0.05),
    ]

    for label, sw, vw in variants:
        t0 = time.time()
        m, p = run_seq_refine(smoothness_w=sw, velocity_w=vw)
        elapsed = time.time() - t0
        delta_m = (np.mean(anat_errs) - m) / np.mean(anat_errs) * 100.0
        delta_p = (np.mean(anat_pa_errs) - p) / np.mean(anat_pa_errs) * 100.0
        print(f"[{label}] MPJPE: {m:.2f} mm ({delta_m:+.2f}%) | PA: {p:.2f} mm ({delta_p:+.2f}%) | {elapsed:.2f}s")


if __name__ == "__main__":
    test_contiguous_window()
