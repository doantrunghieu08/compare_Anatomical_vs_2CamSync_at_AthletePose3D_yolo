"""Run comprehensive ablation study comparing 3D pose triangulation methods with per-joint error tracking.

Evaluates:
1. DLT Baseline
2. Anatomical
3. DST-Anatomical
4. Physics (No DST)
5. DST+Physics Default (bone_w=1.0, data_w=0.2, anchor_w=0.1, mod=True)
6. DST+Physics Tuned (bone_w=0.5, data_w=1.0, anchor_w=0.3, mod=False)
7. DST+Physics Tuned + SequenceRefine (full production pipeline)

Outputs:
- outputs/ablation_per_frame_joint_errors.csv: Every frame with MPJPE, PA-MPJPE, and 17 individual joint errors.
- outputs/ablation_per_joint_summary.csv: Average error for every single joint across all methods.
- outputs/ablation_methods_summary.csv: Method-level performance, win rates, and relative improvements.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from tqdm import tqdm

from athlete_pose3d.algorithms.geometry import (
    H36M_EVAL_JOINTS,
    mpjpe,
    pa_mpjpe,
    procrustes_align,
    triangulate_anatomical,
    triangulate_dlt,
    triangulate_dst_anatomical,
)
from athlete_pose3d.algorithms.physics_refine import (
    triangulate_dst_physics,
    triangulate_physics_refine,
)
from athlete_pose3d.algorithms.refinement import optimize_sequence
from athlete_pose3d.io.data import (
    PoseRepository,
    find_multicam_pairs,
    load_valid_videos,
    scan_local_dataset,
)
from athlete_pose3d.pipeline import (
    _add_dynamic_sync,
    _fps_scaled_sync_params,
    _frame_inputs,
    _load_motion_context,
    _sync_score_options,
    select_best_pair,
)
from athlete_pose3d.settings import load_settings

JOINT_NAMES = [
    "Pelvis", "R_Hip", "R_Knee", "R_Ankle",
    "L_Hip", "L_Knee", "L_Ankle", "Spine",
    "Thorax", "Neck", "Head", "L_Shoulder",
    "L_Elbow", "L_Wrist", "R_Shoulder", "R_Elbow", "R_Wrist"
]


def evaluate_frame_joints(pred_3d: np.ndarray, gt_3d: np.ndarray):
    """Compute root-relative Euclidean error for all 17 joints in mm."""
    pred_rel = pred_3d - pred_3d[0:1, :]
    gt_rel = gt_3d - gt_3d[0:1, :]
    raw_joint_errors = np.linalg.norm(pred_rel - gt_rel, axis=-1)  # (17,)

    # Procrustes aligned per-joint errors
    pred_pa = procrustes_align(pred_3d, gt_3d, mask=H36M_EVAL_JOINTS)
    pred_pa_rel = pred_pa - pred_pa[0:1, :]
    pa_joint_errors = np.linalg.norm(pred_pa_rel - gt_rel, axis=-1)  # (17,)

    raw_mpjpe = float(np.mean(raw_joint_errors[H36M_EVAL_JOINTS]))
    pa_mpjpe_val = float(np.mean(pa_joint_errors[H36M_EVAL_JOINTS]))
    return raw_mpjpe, pa_mpjpe_val, raw_joint_errors, pa_joint_errors


def run_ablation(
    config_path: str = "configs/default.yml",
    subject: str = "S1",
    max_motions: int = 5,
    output_dir: str = "outputs",
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    inputs, output, config = load_settings(config_path)
    repository = PoseRepository(
        inputs.pose2d_root, inputs.pose_2d_suffix, inputs.key_frames_suffix, inputs.pose_cache_limit
    )
    dataset = scan_local_dataset(
        inputs.test_set_dir, inputs.video_extension, inputs.metadata_extension, inputs.ground_truth_extension
    )
    all_pairs = find_multicam_pairs(dataset)
    selected_pairs = [p for p in all_pairs if p["subject"] == subject][:max_motions]

    if not selected_pairs:
        raise RuntimeError(f"No pairs found for subject {subject}")

    print("=" * 80)
    print(f"STARTING ABLATION STUDY: Subject={subject}, Motions={len(selected_pairs)}")
    print("=" * 80)

    valid_videos = load_valid_videos(inputs.manifest, inputs.minimum_manifest_confidence)

    # Storage for per-frame records
    all_frame_rows = []

    for pair_info in tqdm(selected_pairs, desc=f"Ablation on {subject}"):
        score_options = _sync_score_options(config, pair_info["subject"])
        sync_p = _fps_scaled_sync_params(pair_info, config)

        best_pair = select_best_pair(
            pair_info, repository, valid_videos=valid_videos,
            max_candidates=config.max_pair_candidates, max_offset=sync_p["max_offset"],
            sync_samples=config.sync_samples, coarse_step=sync_p["coarse_step"],
            minimum_valid_ratio=config.minimum_pair_valid_ratio,
            validation_radius=sync_p["local_radius"], score_options=score_options,
        )
        if best_pair is None:
            continue

        context = _load_motion_context(pair_info, best_pair, repository, config)
        if context is None:
            continue

        context.update(
            score_options=score_options, fps=sync_p["fps"],
            sync_local_radius=sync_p["local_radius"],
            dynamic_local_radius=sync_p["dynamic_radius"],
            dynamic_transition_weight=sync_p["transition_weight"],
        )
        _add_dynamic_sync(context, repository, config)

        bone_lens = context["bone_lengths"]
        p1, p2 = context["p1"], context["p2"]

        # Define 6 frame-level ablation variants
        methods = {
            "1_DLT_Baseline": lambda k1, k2, c1, c2: triangulate_dlt(p1, p2, k1, k2),
            "2_Anatomical": lambda k1, k2, c1, c2: triangulate_anatomical(
                p1, p2, k1, k2, c1, c2, bone_lengths=bone_lens, bone_weight=1.0, iterations=80
            ),
            "3_DST_Anatomical": lambda k1, k2, c1, c2: triangulate_dst_anatomical(
                p1, p2, k1, k2, c1, c2, bone_lengths=bone_lens, bone_weight=1.0, iterations=80, conflict_threshold=0.65
            ),
            "4_Physics_No_DST": lambda k1, k2, c1, c2: triangulate_physics_refine(
                p1, p2, k1, k2, c1, c2, bone_lengths=bone_lens, iterations=80, lr=1.0,
                data_weight=1.0, bone_weight=0.5, anchor_weight=0.3, sym_weight=0.2,
                bone_reliability_modulation=False, delta_ray_mm=5.0
            ),
            "5_DST_Physics_Default": lambda k1, k2, c1, c2: triangulate_dst_physics(
                p1, p2, k1, k2, c1, c2, bone_lengths=bone_lens, iterations=80, lr=1.0,
                data_weight=0.2, bone_weight=1.0, anchor_weight=0.1, sym_weight=0.2,
                bone_reliability_modulation=True, delta_ray_mm=5.0
            ),
            "6_DST_Physics_Tuned": lambda k1, k2, c1, c2: triangulate_dst_physics(
                p1, p2, k1, k2, c1, c2, bone_lengths=bone_lens, iterations=80, lr=1.0,
                data_weight=1.0, bone_weight=0.5, anchor_weight=0.3, sym_weight=0.2,
                bone_reliability_modulation=False, delta_ray_mm=5.0
            ),
        }

        motion_frames_data = []

        # Step 1: Collect inputs & evaluate frame-level methods
        for frame in context["frames"]:
            inp = _frame_inputs(context, frame, repository, config)
            if inp is None:
                continue
            left, right, dyn_off, loc_off, s_score, d_score, gt = inp
            k1, k2 = left["kps_h36m"], right["kps_h36m"]
            c1, c2 = left["conf_h36m"], right["conf_h36m"]

            frame_res = {
                "motion": pair_info["motion"],
                "subject": pair_info["subject"],
                "cam_a": context["camera_a"]["cam_id"],
                "cam_b": context["camera_b"]["cam_id"],
                "frame": frame,
                "gt_3d": gt,
                "P1": p1, "P2": p2,
                "kps2d_a_h36m": k1, "kps2d_b_h36m": k2,
                "conf_a_h36m": c1, "conf_b_h36m": c2,
                "preds": {},
            }

            for m_name, fn in methods.items():
                pred = fn(k1, k2, c1, c2)
                frame_res["preds"][m_name] = pred

            motion_frames_data.append(frame_res)

        if not motion_frames_data:
            continue

        # Step 2: Sequence Refinement on Variant 6 (DST_Physics_Tuned)
        refine_items = []
        for fr in motion_frames_data:
            mock_res = {
                "frame": fr["frame"],
                "all_methods": {"2cam": {"recon_3d": fr["preds"]["6_DST_Physics_Tuned"]}},
                "P1": fr["P1"], "P2": fr["P2"],
                "kps2d_a_h36m": fr["kps2d_a_h36m"], "kps2d_b_h36m": fr["kps2d_b_h36m"],
                "conf_a_h36m": fr["conf_a_h36m"], "conf_b_h36m": fr["conf_b_h36m"],
            }
            refine_items.append((fr["motion"], mock_res))

        opt_seq = optimize_sequence(
            refine_items, source_method="2cam", bone_lengths=bone_lens,
            data_weight=config.sequence_refinement["data_weight"],
            root_weight=config.sequence_refinement["root_weight"],
            bone_weight=config.sequence_refinement["bone_weight"],
            reprojection_weight=config.sequence_refinement["reprojection_weight"],
            smoothness_weight=config.sequence_refinement["smoothness_weight"],
            symmetry_weight=config.sequence_refinement["symmetry_weight"],
            max_evaluations=config.sequence_refinement["max_evaluations"],
        )

        for i, fr in enumerate(motion_frames_data):
            fr["preds"]["7_DST_Physics_SeqRefine"] = opt_seq[i]

        # Step 3: Record per-joint metrics for every method on every frame
        for fr in motion_frames_data:
            gt = fr["gt_3d"]
            dlt_pred = fr["preds"]["1_DLT_Baseline"]
            dlt_mpjpe, dlt_pa, _, _ = evaluate_frame_joints(dlt_pred, gt)

            for m_name, pred in fr["preds"].items():
                raw_m, pa_m, j_raw, j_pa = evaluate_frame_joints(pred, gt)
                delta_m_pct = (dlt_mpjpe - raw_m) / max(dlt_mpjpe, 1e-6) * 100.0
                delta_pa_pct = (dlt_pa - pa_m) / max(dlt_pa, 1e-6) * 100.0

                row = {
                    "Motion": fr["motion"],
                    "Subject": fr["subject"],
                    "Cam_A": fr["cam_a"],
                    "Cam_B": fr["cam_b"],
                    "Frame": fr["frame"],
                    "Method": m_name,
                    "MPJPE_mm": round(raw_m, 2),
                    "PA_MPJPE_mm": round(pa_m, 2),
                    "Delta_MPJPE_vs_DLT_pct": round(delta_m_pct, 2),
                    "Delta_PA_vs_DLT_pct": round(delta_pa_pct, 2),
                }

                # Add 17 individual joint errors
                for j_idx, j_name in enumerate(JOINT_NAMES):
                    row[f"Joint_{j_idx}_{j_name}_mm"] = round(float(j_raw[j_idx]), 2)
                    row[f"PA_Joint_{j_idx}_{j_name}_mm"] = round(float(j_pa[j_idx]), 2)

                all_frame_rows.append(row)

    df_frames = pd.DataFrame(all_frame_rows)
    per_frame_csv = out_path / "ablation_per_frame_joint_errors.csv"
    df_frames.to_csv(per_frame_csv, index=False)
    print(f"\nSaved Per-Frame Joint Errors ({len(df_frames)} rows) -> {per_frame_csv}")

    # =========================================================================
    # SUMMARY 1: Methods Overall Summary
    # =========================================================================
    all_methods_list = sorted(list(df_frames["Method"].unique()))
    method_summary = []

    dlt_df = df_frames[df_frames["Method"] == "1_DLT_Baseline"].set_index(["Motion", "Frame"])

    for m in all_methods_list:
        sub_df = df_frames[df_frames["Method"] == m].set_index(["Motion", "Frame"])
        m_raw = sub_df["MPJPE_mm"].values
        m_pa = sub_df["PA_MPJPE_mm"].values
        dlt_raw = dlt_df.loc[sub_df.index, "MPJPE_mm"].values
        dlt_pa = dlt_df.loc[sub_df.index, "PA_MPJPE_mm"].values

        delta_raw_pct = (dlt_raw - m_raw) / np.maximum(dlt_raw, 1e-6) * 100.0
        delta_pa_pct = (dlt_pa - m_pa) / np.maximum(dlt_pa, 1e-6) * 100.0

        method_summary.append({
            "Method": m,
            "Frames": len(m_raw),
            "Mean_MPJPE_mm": round(float(np.mean(m_raw)), 2),
            "Median_MPJPE_mm": round(float(np.median(m_raw)), 2),
            "Mean_PA_MPJPE_mm": round(float(np.mean(m_pa)), 2),
            "Median_PA_MPJPE_mm": round(float(np.median(m_pa)), 2),
            "Delta_MPJPE_vs_DLT_pct": round(float(np.mean(delta_raw_pct)), 2),
            "Delta_PA_vs_DLT_pct": round(float(np.mean(delta_pa_pct)), 2),
            "Win_Rate_MPJPE_pct": round(float(np.mean(delta_raw_pct > 0) * 100.0), 1),
            "Win_Rate_PA_pct": round(float(np.mean(delta_pa_pct > 0) * 100.0), 1),
        })

    df_summary = pd.DataFrame(method_summary)
    summary_csv = out_path / "ablation_methods_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"Saved Methods Summary -> {summary_csv}")

    # =========================================================================
    # SUMMARY 2: Per-Joint Breakdown across Methods
    # =========================================================================
    joint_records = []
    dlt_means = {}

    for j_idx, j_name in enumerate(JOINT_NAMES):
        col = f"Joint_{j_idx}_{j_name}_mm"
        dlt_means[col] = float(df_frames[df_frames["Method"] == "1_DLT_Baseline"][col].mean())

    for j_idx, j_name in enumerate(JOINT_NAMES):
        col = f"Joint_{j_idx}_{j_name}_mm"
        rec = {
            "Joint_Index": j_idx,
            "Joint_Name": j_name,
        }
        for m in all_methods_list:
            m_val = float(df_frames[df_frames["Method"] == m][col].mean())
            rec[f"{m}_mm"] = round(m_val, 2)
            if m != "1_DLT_Baseline":
                d_val = (dlt_means[col] - m_val) / max(dlt_means[col], 1e-6) * 100.0
                rec[f"{m}_delta_pct"] = round(d_val, 2)
        joint_records.append(rec)

    # Average row across evaluated joints 1..16
    avg_rec = {
        "Joint_Index": "Avg (1-16)",
        "Joint_Name": "Overall MPJPE",
    }
    for m in all_methods_list:
        m_val = float(df_frames[df_frames["Method"] == m]["MPJPE_mm"].mean())
        avg_rec[f"{m}_mm"] = round(m_val, 2)
        if m != "1_DLT_Baseline":
            dlt_v = float(df_frames[df_frames["Method"] == "1_DLT_Baseline"]["MPJPE_mm"].mean())
            avg_rec[f"{m}_delta_pct"] = round((dlt_v - m_val) / dlt_v * 100.0, 2)
    joint_records.append(avg_rec)

    df_joints = pd.DataFrame(joint_records)
    joints_csv = out_path / "ablation_per_joint_summary.csv"
    df_joints.to_csv(joints_csv, index=False)
    print(f"Saved Per-Joint Summary -> {joints_csv}")

    print("\n" + "=" * 80)
    print("ABLATION STUDY COMPLETED!")
    print("=" * 80)
    print(df_summary[["Method", "Mean_MPJPE_mm", "Mean_PA_MPJPE_mm", "Delta_MPJPE_vs_DLT_pct", "Delta_PA_vs_DLT_pct", "Win_Rate_PA_pct"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation test with per-joint error tracking")
    parser.add_argument("--config", default="configs/default.yml", help="Path to config")
    parser.add_argument("--subject", default="S1", help="Subject to evaluate")
    parser.add_argument("--max-motions", type=int, default=5, help="Maximum motions to evaluate")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    args = parser.parse_args()

    run_ablation(args.config, args.subject, args.max_motions, args.output_dir)
