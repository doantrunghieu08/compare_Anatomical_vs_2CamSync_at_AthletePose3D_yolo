import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from athlete_pose3d.io.reporting import (
    CsvResultReporter,
    SystemInfo,
    _build_per_joint_analysis,
    _build_summary_row,
    _format_single_report_row,
)
from athlete_pose3d.settings import REPORT_HEADERS


def test_per_joint_analysis():
    print("--- Test: Per-Joint Reliability and Occlusion Reporting ---")
    system = SystemInfo.collect()

    # Tạo 2 dummy frames với các khớp rõ và khớp bị che khuất
    # Frame 1: khớp 1..10 rõ (conf=0.9), khớp 11..16 bị che ở cam B (conf=0.2)
    conf_a_1 = np.full(17, 0.9)
    conf_b_1 = np.full(17, 0.9)
    conf_b_1[11:] = 0.2  # occluded on cam B

    recon_1 = np.zeros((17, 3))
    recon_1[1:11] = np.ones((10, 3)) * 50.0   # 50 mm error
    recon_1[11:] = np.ones((6, 3)) * 120.0    # 120 mm error
    gt_1 = np.zeros((17, 3))

    # Frame 2: tất cả rõ (conf=0.85)
    conf_a_2 = np.full(17, 0.85)
    conf_b_2 = np.full(17, 0.85)
    recon_2 = np.zeros((17, 3))
    recon_2[1:] = np.ones((16, 3)) * 40.0
    gt_2 = np.zeros((17, 3))

    results = [
        {
            "motion": "Axel_1", "subject": "S1", "cam_a": "2", "cam_b": "6", "frame": 1,
            "mpjpe": 70.0, "pa_mpjpe": 60.0, "pa_clear": 50.0, "pa_derived": 80.0,
            "best_method": "DST+Physics", "baseline_dlt_mpjpe": 500.0, "baseline_dlt_pa": 75.0,
            "recon_3d": recon_1, "gt_3d": gt_1,
            "conf_a_h36m": conf_a_1, "conf_b_h36m": conf_b_1,
            "all_methods": {"DLT (baseline)": {"recon_3d": recon_1 * 2, "mpjpe": 140.0, "pa_mpjpe": 120.0}},
        },
        {
            "motion": "Axel_1", "subject": "S1", "cam_a": "2", "cam_b": "6", "frame": 2,
            "mpjpe": 40.0, "pa_mpjpe": 35.0, "pa_clear": 35.0, "pa_derived": 45.0,
            "best_method": "DST+Physics", "baseline_dlt_mpjpe": 450.0, "baseline_dlt_pa": 65.0,
            "recon_3d": recon_2, "gt_3d": gt_2,
            "conf_a_h36m": conf_a_2, "conf_b_h36m": conf_b_2,
            "all_methods": {"DLT (baseline)": {"recon_3d": recon_2 * 2, "mpjpe": 80.0, "pa_mpjpe": 70.0}},
        },
    ]

    # Kiểm tra format dòng
    row1 = _format_single_report_row(results[0], system, "v1", "")
    assert len(row1) == len(REPORT_HEADERS)
    print("Row 1 Num Occluded Joints:", row1[REPORT_HEADERS.index("Num_Occluded_Joints")])
    assert row1[REPORT_HEADERS.index("Num_Occluded_Joints")] == 6

    # Kiểm tra summary row
    summary = _build_summary_row(results, system, "v1")
    assert len(summary) == len(REPORT_HEADERS)
    print("Summary Mean Conf:", summary[REPORT_HEADERS.index("Mean_Confidence")])
    print("Summary Unoccluded MPJPE:", summary[REPORT_HEADERS.index("MPJPE_Unoccluded")])
    print("Summary Occluded MPJPE:", summary[REPORT_HEADERS.index("MPJPE_Occluded")])

    # Kiểm tra per-joint analysis
    df = _build_per_joint_analysis(results)
    assert len(df) == 17
    print("\nPer-Joint Analysis DataFrame Sample:")
    print(df[["Joint_ID", "Joint_Name", "Group", "Conf_Stereo", "Occlusion_Rate_pct", "Method_MPJPE_mm", "MPJPE_Unoccluded_mm", "MPJPE_Occluded_mm"]])

    # Test file saving qua CsvResultReporter
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_csv = Path(tmp_dir) / "test_results.csv"
        reporter = CsvResultReporter(out_csv, version="v1")
        reporter.append(results)
        reporter.finish()

        assert out_csv.exists()
        per_joint_csv = Path(tmp_dir) / "test_results_per_joint_summary.csv"
        assert per_joint_csv.exists(), "Per-joint CSV was not created!"
        df_saved = pd.read_csv(per_joint_csv)
        assert len(df_saved) == 17
        print(f"\nPASS: Per-joint summary successfully saved to {per_joint_csv} ({len(df_saved)} rows)")

    print("\nALL PER-JOINT TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_per_joint_analysis()
