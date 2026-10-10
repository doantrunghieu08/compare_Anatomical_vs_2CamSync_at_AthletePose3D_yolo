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
    "configs/ablation/01_dlt.yml",
    "configs/ablation/02_anatomical_generic_height.yml",
    "configs/ablation/03_anatomical_calibrated_height.yml",
    "configs/ablation/04_sequence_refine.yml",
    "configs/ablation/05_full_pipeline.yml",
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


def summarize_results_csv(csv_path: Path, config_name: str, elapsed_sec: float) -> list[dict]:
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    # Filter out any summary lines if present
    df = df[~df["Time"].astype(str).str.startswith(("Summary", "End"), na=False)].copy()
    if df.empty:
        return []

    for col in ("Baseline_DLT_MPJPE", "Baseline_DLT_PA", "Selected_Method_MPJPE", "Selected_Method_PA"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    records = []
    # Overall summary row
    dlt_m, dlt_p = df["Baseline_DLT_MPJPE"].mean(), df["Baseline_DLT_PA"].mean()
    met_m, met_p = df["Selected_Method_MPJPE"].mean(), df["Selected_Method_PA"].mean()
    delta_m = (dlt_m - met_m) / dlt_m * 100.0 if dlt_m > 0 else 0.0
    delta_p = (dlt_p - met_p) / dlt_p * 100.0 if dlt_p > 0 else 0.0

    records.append({
        "Config": config_name,
        "Subject": "ALL",
        "Frames": len(df),
        "DLT_MPJPE": round(float(dlt_m), 2),
        "Method_MPJPE": round(float(met_m), 2),
        "Delta_MPJPE_%": round(float(delta_m), 2),
        "DLT_PA": round(float(dlt_p), 2),
        "Method_PA": round(float(met_p), 2),
        "Delta_PA_%": round(float(delta_p), 2),
        "Elapsed_s": round(elapsed_sec, 1),
    })

    # Per-subject breakdown
    for subj, sub_df in df.groupby("Subject"):
        s_dlt_m, s_dlt_p = sub_df["Baseline_DLT_MPJPE"].mean(), sub_df["Baseline_DLT_PA"].mean()
        s_met_m, s_met_p = sub_df["Selected_Method_MPJPE"].mean(), sub_df["Selected_Method_PA"].mean()
        s_del_m = (s_dlt_m - s_met_m) / s_dlt_m * 100.0 if s_dlt_m > 0 else 0.0
        s_del_p = (s_dlt_p - s_met_p) / s_dlt_p * 100.0 if s_dlt_p > 0 else 0.0
        records.append({
            "Config": config_name,
            "Subject": str(subj),
            "Frames": len(sub_df),
            "DLT_MPJPE": round(float(s_dlt_m), 2),
            "Method_MPJPE": round(float(s_met_m), 2),
            "Delta_MPJPE_%": round(float(s_del_m), 2),
            "DLT_PA": round(float(s_dlt_p), 2),
            "Method_PA": round(float(s_met_p), 2),
            "Delta_PA_%": round(float(s_del_p), 2),
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
                    f"MPJPE: {all_rec['DLT_MPJPE']} -> {all_rec['Method_MPJPE']} ({all_rec['Delta_MPJPE_%']:+.2f}%) | "
                    f"PA: {all_rec['DLT_PA']} -> {all_rec['Method_PA']} ({all_rec['Delta_PA_%']:+.2f}%) | "
                    f"Time={elapsed:.1f}s"
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
