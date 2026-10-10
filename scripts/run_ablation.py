#!/usr/bin/env python3
"""Run full ablation study across all 5 configurations without DST."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add src to Python path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

import numpy as np
import pandas as pd
from athlete_pose3d.main import run_benchmark


DEFAULT_CONFIGS = [
    "configs/ablation/A1_dlt_generic.yml",
    "configs/ablation/A2_dlt_calibrated.yml",
    "configs/ablation/B1_anat_generic.yml",
    "configs/ablation/B2_anat_calibrated.yml",
    "configs/ablation/C1_refine_accel.yml",
    "configs/ablation/C2_refine_accel_vel.yml",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run full ablation suite.")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS, help="List of YAML config paths.")
    parser.add_argument("--max-pairs", type=int, default=None, help="Override maximum camera pairs per subject.")
    parser.add_argument("-S1", "--S1", dest="s1", action="store_true", help="Run only S1.")
    parser.add_argument("-S2", "--S2", dest="s2", action="store_true", help="Run only S2.")
    parser.add_argument("-S3", "--S3", dest="s3", action="store_true", help="Run only S3.")
    parser.add_argument("--subjects", nargs="+", default=None, help="Explicit list of subjects.")
    parser.add_argument("--summary-csv", type=Path, default=Path("outputs/ablation_summary.csv"), help="Output summary CSV.")
    return parser.parse_args()


def extract_subjects(args) -> list[str] | None:
    subs = list(args.subjects or [])
    if getattr(args, "s1", False) and "S1" not in subs:
        subs.append("S1")
    if getattr(args, "s2", False) and "S2" not in subs:
        subs.append("S2")
    if getattr(args, "s3", False) and "S3" not in subs:
        subs.append("S3")
    return subs if subs else None


def bootstrap_ci(deltas, n=5000, seed=0) -> tuple[float, list[float]]:
    x = np.asarray(deltas, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0, [0.0, 0.0]
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), np.percentile(means, [2.5, 97.5]).round(2).tolist()


def summarize_results_csv(csv_path: Path, config_name: str, elapsed_sec: float) -> list[dict]:
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    # Filter out summary / footer rows
    df = df[~df["Time"].astype(str).str.startswith(("Summary", "End"), na=False)].copy()
    if df.empty:
        return []

    for col in (
        "Baseline_DLT_Raw_MPJPE", "Baseline_DLT_Rigid_MPJPE", "Baseline_DLT_MPJPE", "Baseline_DLT_PA",
        "Selected_Method_Raw_MPJPE", "Selected_Method_Rigid_MPJPE", "Selected_Method_MPJPE", "Selected_Method_PA"
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif col == "Selected_Method_Rigid_MPJPE" and "Selected_Method_MPJPE" in df.columns:
            df[col] = pd.to_numeric(df["Selected_Method_MPJPE"], errors="coerce")
        elif col == "Baseline_DLT_Rigid_MPJPE" and "Baseline_DLT_MPJPE" in df.columns:
            df[col] = pd.to_numeric(df["Baseline_DLT_MPJPE"], errors="coerce")

    # Read sequence metrics if available
    seq_csv = csv_path.parent / f"{csv_path.stem}_sequence_metrics.csv"
    seq_rot_mean = float("nan")
    if seq_csv.exists():
        try:
            seq_df = pd.read_csv(seq_csv)
            if "Method_SeqRot_MPJPE" in seq_df.columns:
                seq_rot_mean = float(pd.to_numeric(seq_df["Method_SeqRot_MPJPE"], errors="coerce").mean())
        except Exception:
            pass

    # Per-motion bootstrap delta vs baseline
    per_motion_d = []
    if "Selected_Method_MPJPE" in df.columns and "Baseline_DLT_MPJPE" in df.columns:
        valid_df = df.dropna(subset=["Selected_Method_MPJPE", "Baseline_DLT_MPJPE"])
        if not valid_df.empty and "Subject" in valid_df.columns and "Motion" in valid_df.columns:
            diff_series = valid_df.assign(d=valid_df["Baseline_DLT_MPJPE"] - valid_df["Selected_Method_MPJPE"])
            per_motion_d = diff_series.groupby(["Subject", "Motion"])["d"].mean().values

    mean_gain, ci_95 = bootstrap_ci(per_motion_d)

    records = []
    raw_m = df["Selected_Method_Raw_MPJPE"].mean() if "Selected_Method_Raw_MPJPE" in df.columns else float("nan")
    rigid_m = df["Selected_Method_Rigid_MPJPE"].mean() if "Selected_Method_Rigid_MPJPE" in df.columns else df["Selected_Method_MPJPE"].mean()
    pa_m = df["Selected_Method_PA"].mean()
    dlt_rigid = df["Baseline_DLT_Rigid_MPJPE"].mean() if "Baseline_DLT_Rigid_MPJPE" in df.columns else df["Baseline_DLT_MPJPE"].mean()
    dlt_pa = df["Baseline_DLT_PA"].mean()

    records.append({
        "Config": config_name,
        "Subject": "ALL",
        "Frames": len(df),
        "Method_Raw": round(float(raw_m), 2) if np.isfinite(raw_m) else "N/A",
        "Method_Rigid": round(float(rigid_m), 2),
        "Method_SeqRot": round(float(seq_rot_mean), 2) if np.isfinite(seq_rot_mean) else "N/A",
        "Method_PA": round(float(pa_m), 2),
        "DLT_Rigid": round(float(dlt_rigid), 2),
        "DLT_PA": round(float(dlt_pa), 2),
        "Gain_vs_DLT": f"{mean_gain:+.2f} mm [95% CI: {ci_95[0]}, {ci_95[1]}]" if len(per_motion_d) else "N/A",
        "Elapsed_s": round(elapsed_sec, 1),
    })

    # Per-subject breakdown
    for subj, sub_df in df.groupby("Subject"):
        s_raw = sub_df["Selected_Method_Raw_MPJPE"].mean() if "Selected_Method_Raw_MPJPE" in sub_df.columns else float("nan")
        s_rigid = sub_df["Selected_Method_Rigid_MPJPE"].mean() if "Selected_Method_Rigid_MPJPE" in sub_df.columns else sub_df["Selected_Method_MPJPE"].mean()
        s_pa = sub_df["Selected_Method_PA"].mean()
        s_dlt_rigid = sub_df["Baseline_DLT_Rigid_MPJPE"].mean() if "Baseline_DLT_Rigid_MPJPE" in sub_df.columns else sub_df["Baseline_DLT_MPJPE"].mean()
        s_dlt_pa = sub_df["Baseline_DLT_PA"].mean()

        records.append({
            "Config": config_name,
            "Subject": str(subj),
            "Frames": len(sub_df),
            "Method_Raw": round(float(s_raw), 2) if np.isfinite(s_raw) else "N/A",
            "Method_Rigid": round(float(s_rigid), 2),
            "Method_SeqRot": "-",
            "Method_PA": round(float(s_pa), 2),
            "DLT_Rigid": round(float(s_dlt_rigid), 2),
            "DLT_PA": round(float(s_dlt_pa), 2),
            "Gain_vs_DLT": "-",
            "Elapsed_s": round(elapsed_sec, 1),
        })

    return records


def main():
    args = parse_args()
    subjects = extract_subjects(args)
    summary_records = []

    print("=" * 85)
    print("STARTING FULL ABLATION STUDY (WITHOUT DST)")
    print(f"Configs to evaluate: {len(args.configs)}")
    if subjects:
        print(f"Subjects: {', '.join(subjects)}")
    print("=" * 85)

    for cfg_str in args.configs:
        cfg_path = Path(cfg_str)
        if not cfg_path.exists():
            print(f"[SKIP] Config not found: {cfg_path}")
            continue

        cfg_name = cfg_path.stem
        csv_out = Path(f"outputs/ablation/{cfg_name}.csv")
        csv_out.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n>>> Running ablation step: {cfg_name} ({cfg_path.name})")
        t0 = time.time()
        try:
            run_benchmark(
                config_path=cfg_path,
                output_path=csv_out,
                max_pairs=args.max_pairs,
                overwrite=True,
                included_subjects=subjects,
            )
            elapsed = time.time() - t0
            recs = summarize_results_csv(csv_out, cfg_name, elapsed)
            summary_records.extend(recs)
            # Print immediate feedback for this config's ALL row
            all_rec = next((r for r in recs if r["Subject"] == "ALL"), None)
            if all_rec:
                print(
                    f"[{cfg_name}] Frames={all_rec['Frames']} | "
                    f"Raw: {all_rec['Method_Raw']} | "
                    f"Rigid: {all_rec['DLT_Rigid']} -> {all_rec['Method_Rigid']} | "
                    f"SeqRot: {all_rec['Method_SeqRot']} | "
                    f"PA: {all_rec['DLT_PA']} -> {all_rec['Method_PA']} | "
                    f"Gain: {all_rec['Gain_vs_DLT']} | Time={elapsed:.1f}s"
                )
        except Exception as e:
            print(f"[ERROR] Failed to run {cfg_name}: {e}")

    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(args.summary_csv, index=False)
        print("\n" + "=" * 85)
        print("ABLATION STUDY COMPLETED — SUMMARY TABLE")
        print("=" * 85)
        # Display only ALL rows for clear comparison
        display_df = summary_df[summary_df["Subject"] == "ALL"]
        print(display_df.to_string(index=False))
        print(f"\nFull breakdown (including per-subject) saved to: {args.summary_csv}")
    else:
        print("\nNo results generated.")


if __name__ == "__main__":
    main()
