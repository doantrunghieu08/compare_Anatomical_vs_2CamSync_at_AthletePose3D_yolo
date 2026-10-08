"""
Full S1 DST + Physics ablation and validation experiment
- Strict Dev Set (6 motions, 173 frames) vs Val Set (57 motions, 1578 frames) split.
- Hypotheses 1, 2, 3 verification.
- Paired statistics, win-rates, motion-level bootstrap 95% CIs.
- Complete artifact generation with zero GT leakage.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json
import hashlib
import time
import pickle
import platform
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional

from athlete_pose3d.algorithms.geometry import (
    triangulate_dlt, mpjpe, limb_mpjpe, pa_mpjpe, rigid_mpjpe,
    H36M_BONES, H36M_SYMMETRIC_BONES, H36M_EVAL_JOINTS, LIMB_INDICES,
    h36m_bone_lengths_from_height
)
from athlete_pose3d.algorithms.evidence_fusion import fuse_evidences


DEV_MOTIONS = ["Axel_1", "Axel_3", "Axel_7", "Axel_9", "Comb_1", "Comb_2"]
OUTPUT_DIR = Path("outputs/ablation/2026-10-07_dst_physics_ablation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class FastMotionBatch:
    """Pre-computed tensors for a single motion sequence."""
    def __init__(self, m_data: dict, device: torch.device):
        self.motion = m_data["motion"]
        self.frames = m_data["frames"]
        self.B = len(self.frames)
        self.p1 = m_data["p1"]
        self.p2 = m_data["p2"]
        self.bone_lens = m_data["bone_lengths"]
        self.device = device

        self.k1 = np.stack([f["k1"] for f in self.frames])
        self.k2 = np.stack([f["k2"] for f in self.frames])
        self.c1 = np.stack([f["c1"] for f in self.frames])
        self.c2 = np.stack([f["c2"] for f in self.frames])
        self.gt = np.stack([f["gt"] for f in self.frames])

        # DLT Baseline (mm)
        self.dlt = np.stack([triangulate_dlt(self.p1, self.p2, self.k1[i], self.k2[i]) for i in range(self.B)])

        # GT bones for diagnostic upper-bound only (mm)
        self.gt_bones = []
        for i in range(self.B):
            gb = {b: float(np.linalg.norm(self.gt[i, b[0]] - self.gt[i, b[1]])) for b in H36M_BONES}
            self.gt_bones.append(gb)

        # Evidence fusion
        w_list, out_list = [], []
        for i in range(self.B):
            w, _, is_o = fuse_evidences(self.c1[i], self.c2[i], self.p1, self.p2, self.k1[i], self.k2[i], self.bone_lens)
            w_list.append(w)
            out_list.append(is_o)
        self.w_dst = np.stack(w_list) # (B, 17)
        self.out_mask = np.stack(out_list) # (B, 17)

        # GPU Tensors
        self.k1_t = torch.tensor(self.k1, dtype=torch.float32, device=device)
        self.k2_t = torch.tensor(self.k2, dtype=torch.float32, device=device)
        self.p1_t = torch.tensor(self.p1, dtype=torch.float32, device=device)
        self.p2_t = torch.tensor(self.p2, dtype=torch.float32, device=device)

        m1_inv = torch.linalg.inv(self.p1_t[:, :3])
        self.c1_cam = -m1_inv @ self.p1_t[:, 3]
        m2_inv = torch.linalg.inv(self.p2_t[:, :3])
        self.c2_cam = -m2_inv @ self.p2_t[:, 3]

        ones = torch.ones((self.B, 17, 1), dtype=torch.float32, device=device)
        rays1 = (m1_inv @ torch.cat([self.k1_t, ones], dim=-1).transpose(-1, -2)).transpose(-1, -2)
        self.ray_dirs1 = rays1 / torch.linalg.vector_norm(rays1, dim=-1, keepdim=True)
        rays2 = (m2_inv @ torch.cat([self.k2_t, ones], dim=-1).transpose(-1, -2)).transpose(-1, -2)
        self.ray_dirs2 = rays2 / torch.linalg.vector_norm(rays2, dim=-1, keepdim=True)

        self.bone_pairs = list(self.bone_lens.keys())
        self.b_idx_a = torch.tensor([a for (a, b) in self.bone_pairs], dtype=torch.long, device=device)
        self.b_idx_b = torch.tensor([b for (a, b) in self.bone_pairs], dtype=torch.long, device=device)
        self.b_targets_default = torch.tensor([self.bone_lens[b] for b in self.bone_pairs], dtype=torch.float32, device=device)

        self.sym_l1 = torch.tensor([l1 for (l1, l2), (r1, r2) in H36M_SYMMETRIC_BONES], dtype=torch.long, device=device)
        self.sym_l2 = torch.tensor([l2 for (l1, l2), (r1, r2) in H36M_SYMMETRIC_BONES], dtype=torch.long, device=device)
        self.sym_r1 = torch.tensor([r1 for (l1, l2), (r1, r2) in H36M_SYMMETRIC_BONES], dtype=torch.long, device=device)
        self.sym_r2 = torch.tensor([r2 for (l1, l2), (r1, r2) in H36M_SYMMETRIC_BONES], dtype=torch.long, device=device)
        self.zero_ref = torch.zeros((self.B, 17), dtype=torch.float32, device=device)

    def optimize(
        self,
        use_dst: bool = True,
        use_outlier_penalty: bool = True,
        bone_weight: float = 1.0,
        data_weight: float = 0.2,
        anchor_weight: float = 0.1,
        sym_weight: float = 0.2,
        bone_reliability_modulation: bool = True,
        custom_bone_lens_mm: torch.Tensor | None = None,
        use_oracle_gt_bones: bool = False,
        anatomical_loss: bool = False,
        iterations: int = 60,
        lr: float = 1.0,
    ) -> np.ndarray:
        B = self.B
        device = self.device

        # Reliability & weights
        c1_c = np.clip(self.c1, 0.0, 1.0)
        c2_c = np.clip(self.c2, 0.0, 1.0)
        if use_dst:
            w_base = self.w_dst
        else:
            w_base = np.sqrt(c1_c * c2_c)

        if use_outlier_penalty:
            rel = np.clip(w_base * np.where(self.out_mask, 0.5, 1.0), 0.02, 1.0)
        else:
            rel = np.clip(w_base, 0.02, 1.0)
        rel = np.where((c1_c > 0) & (c2_c > 0), rel, 0.0)

        w1_t = torch.tensor(rel, dtype=torch.float32, device=device)
        w2_t = torch.tensor(rel, dtype=torch.float32, device=device)
        anchor_w_t = torch.tensor(rel, dtype=torch.float32, device=device)

        if bone_reliability_modulation:
            bone_w_arr = np.stack([
                [max(1.0 - rel[i, a], 1.0 - rel[i, b]) for (a, b) in self.bone_pairs]
                for i in range(B)
            ])
            bone_w_t = torch.tensor(bone_w_arr, dtype=torch.float32, device=device)
        else:
            bone_w_t = torch.ones((B, len(self.bone_pairs)), dtype=torch.float32, device=device)

        # Bone targets
        if use_oracle_gt_bones:
            b_targets_t = torch.tensor(np.stack([
                [self.gt_bones[i][b] for b in self.bone_pairs] for i in range(B)
            ]), dtype=torch.float32, device=device)
        elif custom_bone_lens_mm is not None:
            b_targets_t = custom_bone_lens_mm.unsqueeze(0).expand(B, -1)
        else:
            b_targets_t = self.b_targets_default.unsqueeze(0).expand(B, -1)

        pose = torch.tensor(self.dlt, dtype=torch.float32, device=device, requires_grad=True)
        anchor_t = torch.tensor(self.dlt, dtype=torch.float32, device=device)
        optimizer = torch.optim.Adam([pose], lr=lr)

        for _ in range(iterations):
            optimizer.zero_grad()

            # Ray loss 1
            diff1 = pose - self.c1_cam
            proj1 = torch.sum(diff1 * self.ray_dirs1, dim=-1, keepdim=True)
            dist1 = torch.linalg.vector_norm(diff1 - proj1 * self.ray_dirs1, dim=-1)
            ray_l1 = functional.huber_loss(dist1, self.zero_ref, reduction="none", delta=5.0)

            # Ray loss 2
            diff2 = pose - self.c2_cam
            proj2 = torch.sum(diff2 * self.ray_dirs2, dim=-1, keepdim=True)
            dist2 = torch.linalg.vector_norm(diff2 - proj2 * self.ray_dirs2, dim=-1)
            ray_l2 = functional.huber_loss(dist2, self.zero_ref, reduction="none", delta=5.0)

            if anatomical_loss:
                data_l = torch.sum((dist1 ** 2) * w1_t + (dist2 ** 2) * w2_t)
            else:
                data_l = torch.sum(ray_l1 * w1_t + ray_l2 * w2_t)

            # Bone loss
            actual_bone_len = torch.linalg.vector_norm(pose[:, self.b_idx_a] - pose[:, self.b_idx_b], dim=-1)
            if anatomical_loss:
                bone_l = torch.sum(((actual_bone_len - b_targets_t) ** 2) * bone_w_t)
            else:
                bone_huber = functional.huber_loss(actual_bone_len, b_targets_t, reduction="none", delta=10.0)
                bone_l = torch.sum(bone_huber * bone_w_t)

            # Symmetry loss
            if sym_weight > 0 and not anatomical_loss:
                len_l = torch.linalg.vector_norm(pose[:, self.sym_l1] - pose[:, self.sym_l2], dim=-1)
                len_r = torch.linalg.vector_norm(pose[:, self.sym_r1] - pose[:, self.sym_r2], dim=-1)
                sym_l = torch.sum(functional.huber_loss(len_l, len_r, reduction="none", delta=10.0))
            else:
                sym_l = torch.tensor(0.0, device=device)

            # Anchor loss
            if anchor_weight > 0 and not anatomical_loss:
                anc_huber = functional.huber_loss(pose, anchor_t, reduction="none", delta=10.0).sum(-1)
                anc_l = torch.sum(anc_huber * anchor_w_t)
            else:
                anc_l = torch.tensor(0.0, device=device)

            total_l = data_weight * data_l + bone_weight * bone_l + sym_weight * sym_l + anchor_weight * anc_l
            total_l.backward()
            optimizer.step()

        return pose.detach().cpu().numpy()


def compute_sequence_bootstrap_ci(
    motion_names: list[str],
    deltas: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42
) -> tuple[float, float]:
    """Compute 95% bootstrap confidence interval resampled at MOTION level."""
    unique_motions = sorted(list(set(motion_names)))
    motion_indices = {m: np.where(np.array(motion_names) == m)[0] for m in unique_motions}

    rng = np.random.default_rng(seed)
    boot_means = []
    k = len(unique_motions)
    for _ in range(n_boot):
        sampled_m = rng.choice(unique_motions, size=k, replace=True)
        sampled_idx = np.concatenate([motion_indices[m] for m in sampled_m])
        boot_means.append(np.mean(deltas[sampled_idx]))

    ci_low = float(np.percentile(boot_means, 100 * (alpha / 2)))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return ci_low, ci_high


def evaluate_predictions(all_preds: np.ndarray, all_gts: np.ndarray, all_dlts: np.ndarray, motion_names: list[str]) -> dict:
    N = len(all_preds)

    # Compute per-frame metrics
    raw_m = np.array([mpjpe(all_preds[i], all_gts[i]) for i in range(N)])
    rigid_m = np.array([rigid_mpjpe(all_preds[i], all_gts[i]) for i in range(N)])
    pa_m = np.array([pa_mpjpe(all_preds[i], all_gts[i]) for i in range(N)])
    limb_m = np.array([limb_mpjpe(all_preds[i], all_gts[i]) for i in range(N)])

    raw_dlt = np.array([mpjpe(all_dlts[i], all_gts[i]) for i in range(N)])
    rigid_dlt = np.array([rigid_mpjpe(all_dlts[i], all_gts[i]) for i in range(N)])
    pa_dlt = np.array([pa_mpjpe(all_dlts[i], all_gts[i]) for i in range(N)])

    delta_raw = (raw_dlt - raw_m) / np.maximum(raw_dlt, 1e-8) * 100.0
    delta_rigid = (rigid_dlt - rigid_m) / np.maximum(rigid_dlt, 1e-8) * 100.0
    delta_pa = (pa_dlt - pa_m) / np.maximum(pa_dlt, 1e-8) * 100.0

    ci_raw_low, ci_raw_high = compute_sequence_bootstrap_ci(motion_names, delta_raw)
    ci_rigid_low, ci_rigid_high = compute_sequence_bootstrap_ci(motion_names, delta_rigid)
    ci_pa_low, ci_pa_high = compute_sequence_bootstrap_ci(motion_names, delta_pa)

    return {
        "raw_mean": float(np.mean(raw_m)),
        "raw_median": float(np.median(raw_m)),
        "raw_p90": float(np.percentile(raw_m, 90)),
        "rigid_mean": float(np.mean(rigid_m)),
        "rigid_median": float(np.median(rigid_m)),
        "rigid_p90": float(np.percentile(rigid_m, 90)),
        "pa_mean": float(np.mean(pa_m)),
        "pa_median": float(np.median(pa_m)),
        "pa_p90": float(np.percentile(pa_m, 90)),
        "limb_mean": float(np.mean(limb_m)),
        "delta_raw_mean": float(np.mean(delta_raw)),
        "ci_raw_95": f"[{ci_raw_low:+.2f}%, {ci_raw_high:+.2f}%]",
        "delta_rigid_mean": float(np.mean(delta_rigid)),
        "ci_rigid_95": f"[{ci_rigid_low:+.2f}%, {ci_rigid_high:+.2f}%]",
        "delta_pa_mean": float(np.mean(delta_pa)),
        "ci_pa_95": f"[{ci_pa_low:+.2f}%, {ci_pa_high:+.2f}%]",
        "win_count_raw": int(np.sum(delta_raw > 0)),
        "win_rate_raw": float(np.mean(delta_raw > 0) * 100),
        "win_count_pa": int(np.sum(delta_pa > 0)),
        "win_rate_pa": float(np.mean(delta_pa > 0) * 100),
        "raw_m": raw_m,
        "rigid_m": rigid_m,
        "pa_m": pa_m,
        "delta_raw": delta_raw,
        "delta_pa": delta_pa,
    }


def main():
    print("=" * 80)
    print("STARTING FULL S1 DST + PHYSICS ABLATION")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cache_path = Path("scratch/s1_all_motions_cache.pkl")
    with open(cache_path, "rb") as f:
        cache_bytes = f.read()
    cache_hash = hashlib.sha256(cache_bytes).hexdigest()
    all_data = pickle.loads(cache_bytes)

    meta = {
        "date": "2026-10-07",
        "platform": platform.platform(),
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "cache_path": str(cache_path),
        "cache_sha256": cache_hash,
        "dev_motions": DEV_MOTIONS,
        "val_motions": [m for m in all_data.keys() if m not in DEV_MOTIONS],
        "total_motions": len(all_data),
    }
    with open(OUTPUT_DIR / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Logged metadata to {OUTPUT_DIR / 'run_metadata.json'}")

    dev_batches = [FastMotionBatch(all_data[m], device) for m in DEV_MOTIONS]
    val_batches = [FastMotionBatch(all_data[m], device) for m in all_data if m not in DEV_MOTIONS]

    dev_gt = np.concatenate([b.gt for b in dev_batches], axis=0)
    dev_dlt = np.concatenate([b.dlt for b in dev_batches], axis=0)
    dev_m_names = [b.motion for b in dev_batches for _ in range(b.B)]

    val_gt = np.concatenate([b.gt for b in val_batches], axis=0)
    val_dlt = np.concatenate([b.dlt for b in val_batches], axis=0)
    val_m_names = [b.motion for b in val_batches for _ in range(b.B)]

    print(f"Dev Set: {len(dev_batches)} motions, {len(dev_gt)} frames")
    print(f"Val Set: {len(val_batches)} motions, {len(val_gt)} frames")

    # Estimate Subject Bone Prior on DEV SET ONLY (ZERO GT LEAKAGE)
    all_dev_bones = []
    for b in dev_batches:
        for i in range(b.B):
            if min(b.c1[i].mean(), b.c2[i].mean()) > 0.65:
                pts = b.dlt[i]
                blens = [float(np.linalg.norm(pts[u] - pts[v])) for u, v in H36M_BONES]
                all_dev_bones.append(blens)
    dev_median_bones_mm = np.median(all_dev_bones, axis=0)
    dev_median_bones_t = torch.tensor(dev_median_bones_mm, dtype=torch.float32, device=device)
    print(f"Estimated Subject Median Bone Lengths from {len(all_dev_bones)} Dev frames (ZERO GT):")
    print(np.round(dev_median_bones_mm, 1))

    # -------------------------------------------------------------
    # PHASE 1: DEV SET ABLATIONS (Testing Hypotheses 1, 2, 3)
    # -------------------------------------------------------------
    dev_variants = [
        # 5.1 Baselines & Controls
        ("1.1 Baseline DLT", lambda b: b.dlt),
        ("1.2 Physics-Refine (no DST)", lambda b: b.optimize(use_dst=False, bone_reliability_modulation=False)),
        ("1.3 DST-Anatomical", lambda b: b.optimize(use_dst=True, anatomical_loss=True, anchor_weight=0.0, sym_weight=0.0)),
        ("1.4 DST+Physics Default", lambda b: b.optimize(use_dst=True, bone_weight=1.0, data_weight=0.2, anchor_weight=0.1, sym_weight=0.2, bone_reliability_modulation=True)),

        # 5.2 Hypothesis 1: Trọng số Ray Loss
        ("2.1 Ray w=0.2 (Default)", lambda b: b.optimize(data_weight=0.2)),
        ("2.2 Ray w=0.5 (Moderate)", lambda b: b.optimize(data_weight=0.5)),
        ("2.3 Ray w=1.0 (High)", lambda b: b.optimize(data_weight=1.0)),

        # 5.3 Hypothesis 2: Trọng số Xương & Reliability Modulation
        ("3.1 Mod=True, Bone w=0.5", lambda b: b.optimize(bone_weight=0.5, bone_reliability_modulation=True)),
        ("3.2 Mod=True, Bone w=2.0", lambda b: b.optimize(bone_weight=2.0, bone_reliability_modulation=True)),
        ("3.3 Mod=False, Bone w=0.5", lambda b: b.optimize(bone_weight=0.5, bone_reliability_modulation=False)),
        ("3.4 Mod=False, Bone w=1.0", lambda b: b.optimize(bone_weight=1.0, bone_reliability_modulation=False)),
        ("3.5 Mod=False, Bone w=2.0", lambda b: b.optimize(bone_weight=2.0, bone_reliability_modulation=False)),

        # 5.4 Hypothesis 2: Anchor DLT
        ("4.1 Anchor w=0.0 (No Anchor)", lambda b: b.optimize(anchor_weight=0.0)),
        ("4.2 Anchor w=0.3 (Strong Anchor)", lambda b: b.optimize(anchor_weight=0.3)),
        ("4.3 Anchor w=0.5 (Dominant Anchor)", lambda b: b.optimize(anchor_weight=0.5)),

        # 5.5 Hypothesis 3: Bone Prior Thích Nghi vs H36M vs ORACLE_GT
        ("5.1 Subject Bone Prior (Default Weights)", lambda b: b.optimize(custom_bone_lens_mm=dev_median_bones_t)),
        ("5.2 Subject Bone Prior + Tuned (rw=1.0, aw=0.3, bw=0.5, mod=False)", lambda b: b.optimize(custom_bone_lens_mm=dev_median_bones_t, data_weight=1.0, anchor_weight=0.3, bone_weight=0.5, bone_reliability_modulation=False)),
        ("5.3 ORACLE_GT (Upper Bound Diagnostic)", lambda b: b.optimize(use_oracle_gt_bones=True)),
    ]

    print("\n" + "=" * 95)
    print("RUNNING DEV SET ABLATIONS (173 FRAMES, 6 MOTIONS)")
    print("=" * 95)

    dev_summary_records = []
    dev_preds_dict = {}

    for name, run_fn in dev_variants:
        t0 = time.time()
        preds = np.concatenate([run_fn(b) for b in dev_batches], axis=0)
        dt = time.time() - t0
        dev_preds_dict[name] = preds

        res = evaluate_predictions(preds, dev_gt, dev_dlt, dev_m_names)
        print(f"{name:<55} | Raw: {res['raw_mean']:6.2f} mm ({res['delta_raw_mean']:+5.2f}%) | Rigid: {res['rigid_mean']:5.2f} mm ({res['delta_rigid_mean']:+5.2f}%) | PA: {res['pa_mean']:5.2f} mm ({res['delta_pa_mean']:+5.2f}%) | Win: {res['win_rate_raw']:4.1f}% ({dt:.1f}s)")

        dev_summary_records.append({
            "Variant": name,
            "Raw_Mean_mm": round(res["raw_mean"], 2),
            "Raw_Median_mm": round(res["raw_median"], 2),
            "Raw_P90_mm": round(res["raw_p90"], 2),
            "Delta_Raw_vs_DLT_pct": round(res["delta_raw_mean"], 2),
            "CI95_Delta_Raw": res["ci_raw_95"],
            "Rigid_Mean_mm": round(res["rigid_mean"], 2),
            "Delta_Rigid_vs_DLT_pct": round(res["delta_rigid_mean"], 2),
            "CI95_Delta_Rigid": res["ci_rigid_95"],
            "PA_Mean_mm": round(res["pa_mean"], 2),
            "Delta_PA_vs_DLT_pct": round(res["delta_pa_mean"], 2),
            "CI95_Delta_PA": res["ci_pa_95"],
            "Win_Rate_Raw_pct": round(res["win_rate_raw"], 1),
            "Win_Rate_PA_pct": round(res["win_rate_pa"], 1),
            "Runtime_sec": round(dt, 2),
        })

    df_dev = pd.DataFrame(dev_summary_records)
    df_dev.to_csv(OUTPUT_DIR / "summary_dev.csv", index=False)
    print(f"\nSaved Dev Summary to {OUTPUT_DIR / 'summary_dev.csv'}")

    # -------------------------------------------------------------
    # PHASE 2: VALIDATION SET INDEPENDENT CONFIRMATION (1,578 frames)
    # -------------------------------------------------------------
    print("\n" + "=" * 95)
    print("RUNNING VALIDATION SET INDEPENDENT CONFIRMATION (1,578 FRAMES, 57 MOTIONS)")
    print("=" * 95)

    val_variants = [
        ("1.0 Baseline DLT", lambda b: b.dlt),
        ("1.1 Default DST+Physics (Production)", lambda b: b.optimize(use_dst=True, bone_weight=1.0, data_weight=0.2, anchor_weight=0.1, sym_weight=0.2, bone_reliability_modulation=True)),
        ("2.0 Tuned Ray & Anchor (H36M Bone)", lambda b: b.optimize(data_weight=1.0, anchor_weight=0.3, bone_weight=0.5, bone_reliability_modulation=False)),
        ("3.0 Subject Bone Prior + Tuned (NO GT)", lambda b: b.optimize(custom_bone_lens_mm=dev_median_bones_t, data_weight=1.0, anchor_weight=0.3, bone_weight=0.5, bone_reliability_modulation=False)),
        ("4.0 ORACLE_GT (Upper Bound Diagnostic)", lambda b: b.optimize(use_oracle_gt_bones=True)),
    ]

    val_summary_records = []
    val_preds_dict = {}

    for name, run_fn in val_variants:
        t0 = time.time()
        preds = np.concatenate([run_fn(b) for b in val_batches], axis=0)
        dt = time.time() - t0
        val_preds_dict[name] = preds

        res = evaluate_predictions(preds, val_gt, val_dlt, val_m_names)
        print(f"{name:<45} | Raw: {res['raw_mean']:6.2f} mm ({res['delta_raw_mean']:+5.2f}%) | Rigid: {res['rigid_mean']:5.2f} mm ({res['delta_rigid_mean']:+5.2f}%) | PA: {res['pa_mean']:5.2f} mm ({res['delta_pa_mean']:+5.2f}%) | Win: {res['win_rate_raw']:4.1f}% ({dt:.1f}s)")

        val_summary_records.append({
            "Variant": name,
            "Raw_Mean_mm": round(res["raw_mean"], 2),
            "Raw_Median_mm": round(res["raw_median"], 2),
            "Raw_P90_mm": round(res["raw_p90"], 2),
            "Delta_Raw_vs_DLT_pct": round(res["delta_raw_mean"], 2),
            "CI95_Delta_Raw": res["ci_raw_95"],
            "Rigid_Mean_mm": round(res["rigid_mean"], 2),
            "Delta_Rigid_vs_DLT_pct": round(res["delta_rigid_mean"], 2),
            "CI95_Delta_Rigid": res["ci_rigid_95"],
            "PA_Mean_mm": round(res["pa_mean"], 2),
            "Delta_PA_vs_DLT_pct": round(res["delta_pa_mean"], 2),
            "CI95_Delta_PA": res["ci_pa_95"],
            "Win_Rate_Raw_pct": round(res["win_rate_raw"], 1),
            "Win_Rate_PA_pct": round(res["win_rate_pa"], 1),
            "Runtime_sec": round(dt, 2),
        })

    df_val = pd.DataFrame(val_summary_records)
    df_val.to_csv(OUTPUT_DIR / "summary_val.csv", index=False)
    print(f"\nSaved Val Summary to {OUTPUT_DIR / 'summary_val.csv'}")

    # -------------------------------------------------------------
    # PHASE 3: PER-JOINT & PER-MOTION BREAKDOWN ON VALIDATION SET
    # -------------------------------------------------------------
    joint_names = [
        "0.Pelvis", "1.R_Hip", "2.R_Knee", "3.R_Ankle",
        "4.L_Hip", "5.L_Knee", "6.L_Ankle", "7.Spine",
        "8.Thorax", "9.Neck", "10.Head", "11.L_Shoulder",
        "12.L_Elbow", "13.L_Wrist", "14.R_Shoulder", "15.R_Elbow", "16.R_Wrist"
    ]
    p_dlt = val_preds_dict["1.0 Baseline DLT"]
    p_def = val_preds_dict["1.1 Default DST+Physics (Production)"]
    p_opt = val_preds_dict["3.0 Subject Bone Prior + Tuned (NO GT)"]

    dlt_rel = p_dlt - p_dlt[:, 0:1, :]
    def_rel = p_def - p_def[:, 0:1, :]
    opt_rel = p_opt - p_opt[:, 0:1, :]
    gt_rel = val_gt - val_gt[:, 0:1, :]

    dlt_j_err = np.mean(np.linalg.norm(dlt_rel - gt_rel, axis=-1), axis=0)
    def_j_err = np.mean(np.linalg.norm(def_rel - gt_rel, axis=-1), axis=0)
    opt_j_err = np.mean(np.linalg.norm(opt_rel - gt_rel, axis=-1), axis=0)

    joint_records = []
    for j in range(17):
        dlt_v = float(dlt_j_err[j])
        def_v = float(def_j_err[j])
        opt_v = float(opt_j_err[j])
        joint_records.append({
            "Joint_Index": j,
            "Joint_Name": joint_names[j],
            "DLT_Raw_mm": round(dlt_v, 2),
            "Default_DST_Raw_mm": round(def_v, 2),
            "Proposed_Opt_Raw_mm": round(opt_v, 2),
            "Delta_Proposed_vs_DLT_pct": round((dlt_v - opt_v) / max(dlt_v, 1e-8) * 100.0, 2),
            "Delta_Proposed_vs_Default_pct": round((def_v - opt_v) / max(def_v, 1e-8) * 100.0, 2),
        })
    df_joints = pd.DataFrame(joint_records)
    df_joints.to_csv(OUTPUT_DIR / "per_joint_breakdown.csv", index=False)
    print(f"Saved Joint Breakdown to {OUTPUT_DIR / 'per_joint_breakdown.csv'}")

    # Per-motion breakdown
    motion_records = []
    unique_val_m = sorted(list(set(val_m_names)))
    for m in unique_val_m:
        idx = np.where(np.array(val_m_names) == m)[0]
        m_raw_dlt = float(np.mean([mpjpe(p_dlt[i], val_gt[i]) for i in idx]))
        m_raw_def = float(np.mean([mpjpe(p_def[i], val_gt[i]) for i in idx]))
        m_raw_opt = float(np.mean([mpjpe(p_opt[i], val_gt[i]) for i in idx]))
        motion_records.append({
            "Motion": m,
            "Num_Frames": len(idx),
            "DLT_Raw_mm": round(m_raw_dlt, 2),
            "Default_Raw_mm": round(m_raw_def, 2),
            "Proposed_Opt_Raw_mm": round(m_opt_v, 2),
            "Delta_Opt_vs_DLT_pct": round((m_raw_dlt - m_raw_opt) / max(m_raw_dlt, 1e-8) * 100.0, 2),
        })
    df_motions = pd.DataFrame(motion_records)
    df_motions.to_csv(OUTPUT_DIR / "per_motion_val.csv", index=False)
    print(f"Saved Motion Breakdown to {OUTPUT_DIR / 'per_motion_val.csv'}")

    print("\n" + "=" * 80)
    print("ALL ABLATION EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
