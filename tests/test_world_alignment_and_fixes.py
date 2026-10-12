import tempfile
from pathlib import Path
import numpy as np
import pytest

from athlete_pose3d.algorithms.geometry import align_sequence_to_world, mpjpe
from athlete_pose3d.algorithms.synchronization import _record_evaluated_pairs_csv
from athlete_pose3d.settings import REPORT_HEADERS
from athlete_pose3d.io.reporting import _format_single_report_row, _build_summary_row, SystemInfo


def test_align_sequence_to_world():
    N, J = 8, 17
    np.random.seed(42)
    gt = np.random.randn(N, J, 3) * 500.0
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    preds = (gt @ R.T) + 250.0

    aligned, r_mat = align_sequence_to_world(preds, gt)
    err = mpjpe(aligned, gt, align_rotation=False)
    assert err < 1e-4
    # Pelvis (joint 0) must align exactly to gt pelvis across all frames
    assert np.allclose(aligned[:, 0, :], gt[:, 0, :], atol=1e-5)
    # Rotation matrix should be orthonormal
    assert np.allclose(r_mat @ r_mat.T, np.eye(3), atol=1e-5)


def test_reporting_headers_length_consistency():
    system = SystemInfo.collect()
    dummy_res = {
        "subject": "S1", "motion": "jump", "cam_a": "c1", "cam_b": "c2", "frame": 0,
        "frames": 5, "baseline_dlt_raw_mpjpe": 500, "baseline_dlt_rigid_mpjpe": 80,
        "baseline_dlt_mpjpe": 70, "baseline_dlt_pa": 60,
        "raw_mpjpe": 500, "rigid_mpjpe": 75, "mpjpe": 65, "pa_mpjpe": 55,
        "baseline_dlt_seq_rot_mpjpe": 68.0, "seq_rot_mpjpe": 64.0,
        "pa_clear": 50, "pa_derived": 60,
        "best_method": "Weighted DLT",
        "recon_3d": np.zeros((17, 3)), "gt_3d": np.zeros((17, 3)),
        "all_methods": {"Weighted DLT": {"pa_error": 55, "mpjpe": 65}},
        "conf_a": np.ones(17), "conf_b": np.ones(17),
    }
    row = _format_single_report_row(dummy_res, system, "v1", "")
    assert len(row) == len(REPORT_HEADERS), f"Row length {len(row)} != headers {len(REPORT_HEADERS)}"

    sum_row = _build_summary_row([dummy_res], system, "v1")
    assert len(sum_row) == len(REPORT_HEADERS), f"Summary row length {len(sum_row)} != headers {len(REPORT_HEADERS)}"


def test_record_evaluated_pairs_deduplication():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "all_camera_pairs.csv"
        pair_info = {"subject": "S1", "motion": "jump"}
        best = {"cam_a": {"cam_id": "c1"}, "cam_b": {"cam_id": "c2"}}
        c1 = {"cam_a": {"cam_id": "c1"}, "cam_b": {"cam_id": "c2"}, "score": 10.0, "sync_score": 1.0}
        c2 = {"cam_a": {"cam_id": "c1"}, "cam_b": {"cam_id": "c3"}, "score": 20.0, "sync_score": 2.0}

        # First recording
        _record_evaluated_pairs_csv(pair_info, [c1, c2], best, csv_path)
        with open(csv_path, "r", encoding="utf-8") as f:
            lines1 = [line.strip() for line in f if line.strip()]
        assert len(lines1) == 3  # Header + 2 rows

        # Second recording for same (S1, jump) should update/replace, NOT duplicate
        _record_evaluated_pairs_csv(pair_info, [c1, c2], best, csv_path)
        with open(csv_path, "r", encoding="utf-8") as f:
            lines2 = [line.strip() for line in f if line.strip()]
        assert len(lines2) == 3, f"Expected 3 lines, got {len(lines2)}"

        # Recording for another motion (S1, run) should append
        pair_info2 = {"subject": "S1", "motion": "run"}
        _record_evaluated_pairs_csv(pair_info2, [c1], best, csv_path)
        with open(csv_path, "r", encoding="utf-8") as f:
            lines3 = [line.strip() for line in f if line.strip()]
        assert len(lines3) == 4
