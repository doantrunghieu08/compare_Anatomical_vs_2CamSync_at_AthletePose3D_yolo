"""Verification test for all fixes from docs/huong_dan_sua_sai_so.md.

Covers:
1. Kabsch & sequence diagnosis (MPJPE-SeqRot, Scale Ratio, Angle).
2. Z-up coordinate transformation & Rodrigues rotation without GT.
3. CLEAR_JOINTS and DERIVED_JOINTS evaluation (PA_Clear, PA_Derived).
4. Anatomical Head Vertex and Neck extrapolation from Thorax->Nose in coco_to_h36m_2d.
5. Closed-form Ridge linear joint regressor (fit_joint_regressor, apply_joint_regressor).
6. Bone instability metric for camera focal scale selection.
7. Reporting formatting with PA_Clear, PA_Derived, and sequence diagnosis.
"""

import sys
from pathlib import Path

# Add src to python path
src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_dir))

import math
import numpy as np
import torch

from athlete_pose3d.algorithms.geometry import (
    kabsch,
    rotation_angle_deg,
    sequence_diagnosis,
    estimate_up_axis,
    rotation_a_to_b,
    to_z_up,
    best_yaw,
    evaluate_joint_groups,
    CLEAR_JOINTS,
    DERIVED_JOINTS,
    fit_joint_regressor,
    apply_joint_regressor,
)
from athlete_pose3d.algorithms.uncalibrated import bone_instability, uncalibrated_triangulation
from athlete_pose3d.io.data import coco_to_h36m_2d
from athlete_pose3d.io.reporting import CsvResultReporter, SystemInfo, _build_summary_row, _format_single_report_row
from athlete_pose3d.settings import REPORT_HEADERS


def test_1_kabsch_and_sequence_diagnosis():
    print("--- Test 1: Kabsch and Sequence Diagnosis ---")
    np.random.seed(42)
    p = np.random.randn(15, 17, 3) * 500.0
    # 90-degree rotation around Z
    r_known = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    g = (p @ r_known.T) * 0.9 + 50.0

    diag = sequence_diagnosis(p, g)
    ang = diag["angle_deg"]
    scale = diag["scale_ratio"]
    seq_rot = diag["mpjpe_seq_rot_mm"]
    seq_sim = diag["mpjpe_seq_sim_mm"]

    assert abs(ang - 90.0) < 1e-4, f"Expected 90 deg, got {ang}"
    assert abs(scale - 0.9) < 1e-3, f"Expected scale 0.9, got {scale}"
    assert seq_sim < 1e-3, f"Expected seq_sim error ~0, got {seq_sim}"
    print(f"PASS: Kabsch recovered angle {ang:.2f} deg, scale {scale:.4f}, seq_sim {seq_sim:.4f} mm.")


def test_2_z_up_alignment():
    print("--- Test 2: Z-up Alignment (GT-free) ---")
    np.random.seed(42)
    # Pose where body axis (pelvis=0 to neck=9) is along -Y (camera coordinates)
    poses = np.zeros((10, 17, 3))
    poses[:, 0] = [0, 1000, 2000]   # Pelvis
    poses[:, 9] = [0, 500, 2000]    # Neck (-Y direction)
    poses[:, 1:9] = np.random.randn(10, 8, 3) * 50
    poses[:, 10:] = np.random.randn(10, 7, 3) * 50

    up_before = estimate_up_axis(poses)
    assert up_before[1] < -0.9, f"Expected -Y up vector, got {up_before}"

    z_up_poses = to_z_up(poses)
    up_after = estimate_up_axis(z_up_poses)
    assert abs(up_after[2] - 1.0) < 1e-4, f"Expected +Z up vector, got {up_after}"
    print("PASS: GT-free to_z_up correctly rotated body axis from camera -Y to world +Z.")


def test_3_joint_group_evaluation():
    print("--- Test 3: Clear vs Derived Joint Group Evaluation ---")
    np.random.seed(42)
    gt = np.random.randn(17, 3) * 500.0  # realistic scale
    pred = gt.copy()
    # Add 10 mm error to clear joints, 50 mm error to derived joints
    for j in CLEAR_JOINTS:
        pred[j] += [10.0, 0, 0]
    for j in DERIVED_JOINTS:
        pred[j] += [50.0, 0, 0]

    groups = evaluate_joint_groups(pred, gt)
    pa_clear = groups["pa_clear"]
    pa_derived = groups["pa_derived"]
    # With procrustes alignment taking both groups, clear joints error is ~10-25 mm, derived ~40-60 mm
    assert pa_clear < pa_derived, f"Expected pa_clear < pa_derived, got {pa_clear} vs {pa_derived}"
    print(f"PASS: Group evaluation isolates Clear ({pa_clear:.1f} mm) vs Derived ({pa_derived:.1f} mm).")



def test_4_coco_to_h36m_extrapolation():
    print("--- Test 4: COCO to H36M Head/Neck Extrapolation ---")
    kps_coco = np.zeros((17, 2))
    conf_coco = np.ones(17)
    kps_coco[5] = [90, 200]   # Left Shoulder
    kps_coco[6] = [110, 200]  # Right Shoulder -> Thorax (8) at [100, 200]
    kps_coco[0] = [100, 150]  # Nose at [100, 150] (vector length = 50px up)

    h36m, conf = coco_to_h36m_2d(kps_coco, conf_coco)
    thorax = h36m[8]
    head = h36m[10]
    neck = h36m[9]

    # Nose was 50px above thorax. Head vertex is 1.45 * 50 = 72.5px above thorax (y = 200 - 72.5 = 127.5)
    assert abs(thorax[1] - 200.0) < 1e-4
    assert abs(head[1] - 127.5) < 1e-4, f"Expected Head y=127.5, got {head[1]}"
    # Neck is thorax + 0.40 * (head - thorax) = 200 - 0.40 * 72.5 = 171.0
    assert abs(neck[1] - 171.0) < 1e-4, f"Expected Neck y=171.0, got {neck[1]}"
    print("PASS: Head vertex (10) and Neck C7 (9) extrapolated anatomically along thorax->nose vector.")


def test_5_joint_regressor():
    print("--- Test 5: Closed-form Linear Joint Regressor ---")
    np.random.seed(42)
    n = 30
    x = np.random.randn(n, 17, 3) * 200.0
    w_true = np.random.randn(17, 17)
    y = np.einsum("jk,nkc->njc", w_true, x)

    w_fit = fit_joint_regressor(x, y, lam=1e-4)
    y_pred = apply_joint_regressor(w_fit, x)
    max_diff = np.max(np.abs(y - y_pred))
    assert max_diff < 1e-2, f"Max diff too large: {max_diff}"
    print(f"PASS: Linear regressor fitted and applied with max diff {max_diff:.6f} mm without external libraries.")


def test_6_bone_instability():
    print("--- Test 6: Bone Instability Metric ---")
    poses = np.zeros((20, 17, 3))
    # Rigid translation: bone lengths strictly constant
    for t in range(20):
        poses[t] = np.random.randn(17, 3) * 0.0 + [t * 10.0, 0, 0]
        # set fixed distance between 0 and 1
        poses[t, 1] = poses[t, 0] + [0, 0, 300.0]

    score = bone_instability(poses, bones=[(0, 1)])
    assert abs(score) < 1e-6, f"Expected instability 0, got {score}"
    print(f"PASS: Rigid poses yield zero bone instability ({score:.8f}).")


def test_7_reporting():
    print("--- Test 7: Reporting with PA_Clear and PA_Derived ---")
    system = SystemInfo.collect()
    dummy_result = {
        "motion": "Axel_1", "subject": "S1", "cam_a": "2", "cam_b": "6", "frame": 10,
        "mpjpe": 65.0, "pa_mpjpe": 55.0, "pa_clear": 42.0, "pa_derived": 95.0,
        "best_method": "Test", "baseline_dlt_mpjpe": 650.0, "baseline_dlt_pa": 67.0,
        "recon_3d": np.zeros((17, 3)), "gt_3d": np.zeros((17, 3)),
    }
    row = _format_single_report_row(dummy_result, system, "v1", "")
    assert len(row) == len(REPORT_HEADERS), f"Row length {len(row)} != headers {len(REPORT_HEADERS)}"
    assert row[REPORT_HEADERS.index("PA_Clear")] == 42.0
    assert row[REPORT_HEADERS.index("PA_Derived")] == 95.0

    summary = _build_summary_row([dummy_result], system, "v1")
    assert summary[REPORT_HEADERS.index("PA_Clear")] == 42.0
    assert summary[REPORT_HEADERS.index("PA_Derived")] == 95.0
    print("PASS: Report formatting and summary correctly include PA_Clear and PA_Derived.")


if __name__ == "__main__":
    test_1_kabsch_and_sequence_diagnosis()
    test_2_z_up_alignment()
    test_3_joint_group_evaluation()
    test_4_coco_to_h36m_extrapolation()
    test_5_joint_regressor()
    test_6_bone_instability()
    test_7_reporting()
    print("\nALL 7 TESTS PASSED PERFECTLY!")
