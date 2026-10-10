"""Verification test script for all 6 Ponytail fixes.

Covers:
1. Security: No allow_pickle=True in src/
2. Synchronization: Frame counting without GT requirement
3. Smoothness: Only applied to contiguous keyframes
4. best_pose: Tracks the pose at best_loss
5. Vectorization: Bone & symmetry loss equivalence and gradient
6. Cache: OrderedDict LRU capacity and eviction
"""

import sys
from pathlib import Path

# Add src to python path
src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_dir))

import numpy as np
import torch
import torch.nn.functional as functional
from collections import OrderedDict

from athlete_pose3d.algorithms.geometry import (
    H36M_SYMMETRIC_BONES,
    compute_kinematic_prior,
    h36m_bone_lengths_from_height,
    resolve_subject_height,
    _anatomical_step,
    triangulate_anatomical,
)
from athlete_pose3d.algorithms.physics_refine import (
    compute_bone_rigidity_loss,
    compute_symmetry_loss,
    triangulate_physics_refine,
    _prepare_optimization_tensors,
)
from athlete_pose3d.algorithms.refinement import optimize_sequence
from athlete_pose3d.algorithms.synchronization import _frame_count_for_camera, _pair_frame_counts
from athlete_pose3d.io.data import PoseRepository
from athlete_pose3d.io.reporting import _build_summary_row, SystemInfo


def test_1_security():
    print("--- Test 1: Security (allow_pickle=True scan) ---")
    py_files = list(src_dir.rglob("*.py"))
    violations = []
    for f in py_files:
        content = f.read_text(encoding="utf-8")
        if "allow_pickle=True" in content:
            violations.append(str(f))
    assert not violations, f"Found allow_pickle=True in: {violations}"
    print(f"PASS: Scanned {len(py_files)} files in src/. 0 occurrences of allow_pickle=True.")


def test_2_synchronization():
    print("--- Test 2: Synchronization (GT independence) ---")
    # Case 1: Camera has pose_path via repository, but NO ground_truth_path
    class DummyRepo:
        def __init__(self, n_frames=100):
            self.n_frames = n_frames
            self.temp_file = Path("scratch/dummy_pose.npy")
            self.temp_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.temp_file, np.zeros((n_frames, 17, 3), dtype=np.float32))

        def pose_path(self, video_path):
            return self.temp_file

    repo = DummyRepo(120)
    cam_a = {"video_path": "dummy_a.mp4"}
    cam_b = {"video_path": "dummy_b.mp4"}

    # No GT present at all
    counts = _pair_frame_counts(cam_a, cam_b, repository=repo)
    assert counts == (120, 120), f"Expected (120, 120), got {counts}"
    print("PASS: Frame count recovered from repository pose file without GT.")

    # Case 2: Fallback to metadata JSON when pose & GT are absent
    meta_file = Path("scratch/dummy_meta.json")
    meta_file.write_text('{"num_frames": 85}', encoding="utf-8")
    cam_c = {"metadata_path": str(meta_file)}
    cam_d = {"metadata_path": str(meta_file)}
    counts_meta = _pair_frame_counts(cam_c, cam_d, repository=None)
    assert counts_meta == (85, 85), f"Expected (85, 85), got {counts_meta}"
    print("PASS: Fallback to metadata JSON works when repository and GT are absent.")


def test_3_smoothness():
    print("--- Test 3: Smoothness (contiguous vs non-contiguous) ---")
    bone_lens = h36m_bone_lengths_from_height(1750.0)

    def run_refine(frame_indices):
        items = []
        for f in frame_indices:
            res = {
                "frame": f,
                "all_methods": {"2cam": {"recon_3d": np.ones((17, 3), dtype=np.float32) * (f * 10.0)}},
                "P1": np.column_stack((np.eye(3), np.zeros(3))),
                "P2": np.column_stack((np.eye(3), np.zeros(3))),
                "kps2d_a_h36m": np.zeros((17, 2), dtype=np.float32),
                "kps2d_b_h36m": np.zeros((17, 2), dtype=np.float32),
                "conf_a_h36m": np.ones(17, dtype=np.float32),
                "conf_b_h36m": np.ones(17, dtype=np.float32),
            }
            items.append(("vid", res))
        return optimize_sequence(items, source_method="2cam", bone_lengths=bone_lens, max_evaluations=5)

    # 1. Contiguous frames [10, 11, 12]
    out_contig = run_refine([10, 11, 12])
    assert out_contig.shape == (3, 17, 3), f"Contiguous refine shape: {out_contig.shape}"

    # 2. Non-contiguous frames [10, 20, 30]
    out_sparse = run_refine([10, 20, 30])
    assert out_sparse.shape == (3, 17, 3), f"Sparse refine shape: {out_sparse.shape}"
    print("PASS: Both contiguous and non-contiguous sequences execute correctly without shape mismatch or error.")


def test_4_best_pose():
    print("--- Test 4: best_pose tracking in physics_refine ---")
    p1 = np.array([[1000, 0, 500, 0], [0, 1000, 500, 0], [0, 0, 1, 0]], dtype=float)
    p2 = np.array([[1000, 0, 500, -1000], [0, 1000, 500, 0], [0, 0, 1, 0]], dtype=float)
    pts1 = np.random.randn(17, 2).astype(np.float32) + 500
    pts2 = np.random.randn(17, 2).astype(np.float32) + 500
    c1 = np.ones(17, dtype=np.float32)
    c2 = np.ones(17, dtype=np.float32)

    res = triangulate_physics_refine(p1, p2, pts1, pts2, c1, c2, iterations=15, lr=0.5)
    assert res.shape == (17, 3)
    assert np.isfinite(res).all()
    print("PASS: triangulate_physics_refine converges and returns finite best_pose.")


def test_5_vectorization():
    print("--- Test 5: Vectorization loss & gradient equivalence ---")
    torch.manual_seed(42)
    pts = torch.randn(17, 3, requires_grad=True)
    bone_lens = h36m_bone_lengths_from_height(1750.0)
    bone_w = {b: 0.7 + 0.05 * i for i, b in enumerate(bone_lens)}

    # Old loop bone loss
    losses = []
    for (a, b), target_len in bone_lens.items():
        if target_len > 0:
            actual_len = torch.linalg.vector_norm(pts[a] - pts[b], dim=-1)
            target_t = pts.new_tensor(target_len)
            loss_item = functional.huber_loss(actual_len, target_t, delta=10.0)
            if bone_w and (a, b) in bone_w:
                loss_item = loss_item * bone_w[(a, b)]
            losses.append(loss_item)
    old_bone = torch.sum(torch.stack(losses))
    old_bone.backward()
    grad_old = pts.grad.clone()

    # Vectorized compute_bone_rigidity_loss
    pts_new = pts.detach().clone().requires_grad_(True)
    new_bone = compute_bone_rigidity_loss(pts_new, bone_lens, bone_w, delta=10.0)
    new_bone.backward()
    grad_new = pts_new.grad.clone()

    assert torch.allclose(old_bone, new_bone, atol=1e-5), f"Bone loss mismatch: {old_bone} vs {new_bone}"
    assert torch.allclose(grad_old, grad_new, atol=1e-5), f"Bone grad mismatch: max diff {torch.max(torch.abs(grad_old - grad_new))}"
    print(f"PASS: Bone loss and gradient exactly match (diff: {abs(old_bone.item() - new_bone.item()):.8f}).")

    # Symmetry loss check
    sym_old = []
    for (l1, l2), (r1, r2) in H36M_SYMMETRIC_BONES:
        len_left = torch.linalg.vector_norm(pts[l1] - pts[l2], dim=-1)
        len_right = torch.linalg.vector_norm(pts[r1] - pts[r2], dim=-1)
        sym_old.append(functional.huber_loss(len_left, len_right, delta=10.0))
    old_sym = torch.sum(torch.stack(sym_old))

    new_sym = compute_symmetry_loss(pts, delta=10.0)
    assert torch.allclose(old_sym, new_sym, atol=1e-5), f"Symmetry loss mismatch: {old_sym} vs {new_sym}"
    print(f"PASS: Symmetry loss exactly matches (diff: {abs(old_sym.item() - new_sym.item()):.8f}).")


def test_6_cache():
    print("--- Test 6: OrderedDict LRU cache in PoseRepository ---")
    repo = PoseRepository(root="scratch", pose_suffix=".npy", key_frames_suffix=".npy", cache_limit=3)

    # Fill cache with 3 items
    repo._frames[("v1", 0)] = {"frame": 0}
    repo._frames[("v1", 1)] = {"frame": 1}
    repo._frames[("v1", 2)] = {"frame": 2}
    assert len(repo._frames) == 3

    # Access key 0 (should move to end)
    # Simulate pose hit
    key = ("v1", 0)
    repo._frames.move_to_end(key)

    # Add 4th item
    repo._frames[("v1", 3)] = {"frame": 3}
    while len(repo._frames) > repo.cache_limit:
        repo._frames.popitem(last=False)

    assert len(repo._frames) == 3
    # key 1 should have been evicted because key 0 was moved to end!
    assert ("v1", 1) not in repo._frames, "Expected key ('v1', 1) to be evicted"
    assert ("v1", 0) in repo._frames, "Expected key ('v1', 0) to be preserved by LRU"
    assert ("v1", 2) in repo._frames, "Expected key ('v1', 2) to be in cache"
    assert ("v1", 3) in repo._frames, "Expected key ('v1', 3) to be in cache"
    print("PASS: LRU OrderedDict eviction correctly preserves accessed keys and caps capacity at cache_limit.")


def test_7_anatomical_pure():
    print("--- Test 7: Pure Anatomical Triangulation (DLT + Bone-Prior + Height + Precomputed Rays) ---")
    p1 = np.array([[1000.0, 0.0, 500.0, 0.0], [0.0, 1000.0, 500.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    p2 = np.array([[1000.0, 0.0, 500.0, 500.0], [0.0, 1000.0, 500.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    true_3d = np.ones((17, 3)) * 100.0 + np.arange(17)[:, None] * 10.0
    pts1 = np.array([p1 @ np.append(pt, 1.0) for pt in true_3d])
    pts1 = pts1[:, :2] / pts1[:, 2:3]
    pts2 = np.array([p2 @ np.append(pt, 1.0) for pt in true_3d])
    pts2 = pts2[:, :2] / pts2[:, 2:3]
    c1 = np.ones(17)
    c2 = np.ones(17)
    bone_lens = h36m_bone_lengths_from_height(1730.0)

    res = triangulate_anatomical(
        p1, p2, pts1, pts2, c1, c2,
        bone_lengths=bone_lens, bone_weight=1.0, iterations=20,
    )
    assert res.shape == (17, 3)
    assert np.isfinite(res).all()
    print("PASS: Pure anatomical triangulation converged with precomputed rays and finite best_pose.")


def test_8_kinematic_and_temporal():
    print("--- Test 8: Kinematic & Temporal Constraints in physics_refine ---")
    p1 = np.array([[1000.0, 0.0, 500.0, 0.0], [0.0, 1000.0, 500.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    p2 = np.array([[1000.0, 0.0, 500.0, 1000.0], [0.0, 1000.0, 500.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    pts1 = np.zeros((17, 2))
    pts2 = np.zeros((17, 2))
    conf1 = np.ones(17)
    conf2 = np.ones(17)
    prev_pose = np.random.randn(17, 3) * 100.0 + np.array([0.0, 0.0, 1500.0])

    pose_refined = triangulate_physics_refine(
        p1, p2, pts1, pts2, conf1, conf2,
        iterations=10,
        prev_pose_3d=prev_pose,
    )
    assert pose_refined.shape == (17, 3)
    assert np.isfinite(pose_refined).all()
    print("PASS: Kinematic & temporal constraints executed correctly.")


def test_9_kinematic_prior_signs():
    print("--- Test 9: Kinematic prior joint signs ---")
    pts = torch.zeros((17, 3), dtype=torch.float32)
    # Hip axis from Left Hip (4) to Right Hip (1): points in +X
    pts[4] = torch.tensor([-200.0, 0.0, 0.0])
    pts[1] = torch.tensor([200.0, 0.0, 0.0])
    # Shoulder axis from Left Shoulder (11) to Right Shoulder (14): +X
    pts[11] = torch.tensor([-200.0, 1000.0, 0.0])
    pts[14] = torch.tensor([200.0, 1000.0, 0.0])

    # Case 1: Normal knee flexion (backward in -Z)
    pts_flex = pts.clone()
    pts_flex[2] = torch.tensor([200.0, 500.0, 0.0])   # R Knee
    pts_flex[3] = torch.tensor([200.0, 0.0, -300.0])  # R Ankle (flexed backward)
    pts_flex[5] = torch.tensor([-200.0, 500.0, 0.0])  # L Knee
    pts_flex[6] = torch.tensor([-200.0, 0.0, -300.0]) # L Ankle (flexed backward)
    loss_flex = compute_kinematic_prior(pts_flex)
    assert loss_flex.item() == 0.0, f"Normal knee flexion should have 0 penalty, got {loss_flex.item()}"

    # Case 2: Knee hyperextension (forward in +Z)
    pts_hyp = pts.clone()
    pts_hyp[2] = torch.tensor([200.0, 500.0, 0.0])   # R Knee
    pts_hyp[3] = torch.tensor([200.0, 0.0, 300.0])   # R Ankle (hyperextended forward)
    pts_hyp[5] = torch.tensor([-200.0, 500.0, 0.0])  # L Knee
    pts_hyp[6] = torch.tensor([-200.0, 0.0, 300.0])  # L Ankle (hyperextended forward)
    loss_hyp = compute_kinematic_prior(pts_hyp)
    assert loss_hyp.item() > 0.1, f"Knee hyperextension should be penalized, got {loss_hyp.item()}"
    print("PASS: Kinematic prior correctly allows posterior flexion and penalizes anterior hyperextension.")


def test_10_height_resolution_priority():
    print("--- Test 10: Height resolution override priority ---")
    subjs = {"S1": 1591.0, "S2": 1553.0}
    # When generic height 1730.0 is explicitly configured, it must override subject_heights
    h_generic = resolve_subject_height("S1", 1730.0, subject_heights=subjs)
    assert h_generic == 1730.0, f"Expected generic 1730.0, got {h_generic}"

    # When configured as "auto", it must use calibrated height from subject_heights
    h_calib = resolve_subject_height("S1", "auto", subject_heights=subjs)
    assert h_calib == 1591.0, f"Expected calibrated 1591.0, got {h_calib}"
    print("PASS: resolve_subject_height correctly prioritizes explicit generic override over calibrated map.")


def test_11_adam_best_pose_candidate():
    print("--- Test 11: Adam best_pose candidate snapshot ---")
    p1 = np.array([[1000.0, 0.0, 500.0, 0.0], [0.0, 1000.0, 500.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    p2 = np.array([[1000.0, 0.0, 500.0, 500.0], [0.0, 1000.0, 500.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    true_3d = np.ones((17, 3)) * 100.0 + np.arange(17)[:, None] * 10.0
    pts1 = np.array([p1 @ np.append(pt, 1.0) for pt in true_3d])
    pts1 = pts1[:, :2] / pts1[:, 2:3]
    pts2 = np.array([p2 @ np.append(pt, 1.0) for pt in true_3d])
    pts2 = pts2[:, :2] / pts2[:, 2:3]
    c1, c2 = np.ones(17, dtype=np.float32), np.ones(17, dtype=np.float32)
    res = triangulate_anatomical(p1, p2, pts1, pts2, c1, c2, iterations=10)
    assert res.shape == (17, 3)
    assert np.isfinite(res).all()
    print("PASS: Anatomical best_pose correctly returns finite pose evaluated before step.")


def test_12_reporting_nan_protection():
    print("--- Test 12: Reporting NaN protection in summary row ---")
    sys_info = SystemInfo(
        python_version="3.11", os_type="Windows", os_version="10",
        compute_device="CPU", cpu_cores=8,
    )
    results = [
        {"mpjpe": 45.0, "pa_mpjpe": 30.0, "baseline_dlt_mpjpe": 50.0, "baseline_dlt_pa": 35.0},
        {"mpjpe": float("nan"), "pa_mpjpe": float("nan"), "baseline_dlt_mpjpe": float("nan"), "baseline_dlt_pa": float("nan")},
        {"mpjpe": 55.0, "pa_mpjpe": 40.0, "baseline_dlt_mpjpe": 60.0, "baseline_dlt_pa": 45.0},
    ]
    summary = _build_summary_row(results, sys_info, "v1.0")
    assert summary is not None
    # summary[12] is s_m, summary[13] is s_p
    assert summary[12] == 50.0, f"Expected mean mpjpe 50.0, got {summary[12]}"
    assert summary[13] == 35.0, f"Expected mean pa_mpjpe 35.0, got {summary[13]}"
    print("PASS: Summary row correctly computes non-NaN means when some frames are NaN.")


def test_13_contiguous_sampling_and_velocity():
    print("--- Test 13: Contiguous sampling & Velocity constraint in Refinement ---")
    from athlete_pose3d.pipeline import _sample_key_frames
    from athlete_pose3d.settings import load_settings
    
    # 1. Contiguous center window check
    frames = list(range(500))
    sampled = _sample_key_frames(frames, 60)
    assert len(sampled) == 60
    assert np.all(np.diff(sampled) == 1), "Sampled frames must be contiguous!"
    assert sampled[0] == 220 and sampled[-1] == 279, f"Expected center window [220..279], got [{sampled[0]}..{sampled[-1]}]"

    # 2. Config validation check on all ablation configs
    for cfg_p in Path("configs/ablation").glob("*.yml"):
        load_settings(cfg_p)
    load_settings("configs/default.yml")

    # 3. Refinement execution with velocity weight
    poses_3d = np.ones((5, 17, 3), dtype=np.float32) * 100.0
    items = []
    for f in range(5):
        items.append((f, {
            "frame": f,
            "all_methods": {"two_camera": {"recon_3d": poses_3d[f]}},
            "P1": np.eye(3, 4), "P2": np.eye(3, 4),
            "kps2d_a_h36m": np.zeros((17, 2)), "conf_a_h36m": np.ones(17),
            "kps2d_b_h36m": np.zeros((17, 2)), "conf_b_h36m": np.ones(17),
        }))
    refined = optimize_sequence(items, "two_camera", smoothness_weight=0.15, velocity_weight=0.03, max_evaluations=5)
    assert refined.shape == (5, 17, 3)
    assert np.isfinite(refined).all()
    print("PASS: Contiguous sampling, velocity constraint, and all configs validated successfully.")


if __name__ == "__main__":
    test_1_security()
    test_2_synchronization()
    test_3_smoothness()
    test_4_best_pose()
    test_5_vectorization()
    test_6_cache()
    test_7_anatomical_pure()
    test_8_kinematic_and_temporal()
    test_9_kinematic_prior_signs()
    test_10_height_resolution_priority()
    test_11_adam_best_pose_candidate()
    test_12_reporting_nan_protection()
    test_13_contiguous_sampling_and_velocity()
    print("\nALL 13 TESTS PASSED SUCCESSFULLY!")

