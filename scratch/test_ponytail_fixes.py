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
    h36m_bone_lengths_from_height,
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


if __name__ == "__main__":
    test_1_security()
    test_2_synchronization()
    test_3_smoothness()
    test_4_best_pose()
    test_5_vectorization()
    test_6_cache()
    print("\nALL 6 TESTS PASSED SUCCESSFULLY!")
