"""Run 7 DST Evidence Fusion ablation configurations in parallel across subjects S1, S2, S3."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel runner for 7 DST ablation configurations.")
    parser.add_argument("--configs-dir", type=str, default="configs/ablation", help="Directory containing ablation YAMLs.")
    parser.add_argument("--output-dir", type=str, default="outputs/ablation", help="Base directory for output results.")
    parser.add_argument("--subjects", nargs="+", default=["S1", "S2", "S3"], help="Subjects to run (e.g. S1 S2 S3).")
    parser.add_argument("--workers", type=int, default=3, help="Number of parallel worker processes (default: 3).")
    parser.add_argument("--max-pairs", type=int, default=None, help="Limit max pairs per motion (for testing).")
    parser.add_argument("--overwrite", action="store_true", default=True, help="Overwrite existing output CSVs.")
    return parser.parse_args()


def run_single_task(cfg_path: Path, subject: str, out_file: Path, log_file: Path, max_pairs: int | None, overwrite: bool):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    src_dir = Path("src").resolve()
    env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{env.get('PYTHONPATH', '')}"

    cmd = [
        sys.executable, "-m", "athlete_pose3d.main",
        "--config", str(cfg_path.resolve()),
        f"-{subject}",
        "--output", str(out_file.resolve()),
    ]
    if overwrite:
        cmd.append("--overwrite")
    if max_pairs is not None:
        cmd.extend(["--max-pairs", str(max_pairs)])

    start_time = time.time()
    cfg_name = cfg_path.stem
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [STARTED] {cfg_name} | Subject {subject}")

    with open(log_file, "w", encoding="utf-8") as f_log:
        res = subprocess.run(cmd, env=env, stdout=f_log, stderr=subprocess.STDOUT, text=True)

    elapsed = time.time() - start_time
    if res.returncode == 0:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [SUCCESS] {cfg_name} | Subject {subject} in {elapsed:.1f}s")
        return {"config": cfg_name, "subject": subject, "status": "ok", "elapsed_s": elapsed, "out_file": out_file}
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [FAILED]  {cfg_name} | Subject {subject} (exit code {res.returncode}, see {log_file})")
        return {"config": cfg_name, "subject": subject, "status": "failed", "elapsed_s": elapsed, "out_file": out_file}


def generate_summary(results: list[dict], output_dir: Path):
    summary_rows = []
    for r in results:
        if r["status"] != "ok" or not r["out_file"].exists():
            continue
        try:
            df = pd.read_csv(r["out_file"])
            # Remove summary rows if present
            df_frames = df[pd.to_numeric(df["Frame"], errors="coerce").notna()].copy()
            if df_frames.empty:
                continue
            dlt_m = pd.to_numeric(df_frames["Baseline_DLT_MPJPE"], errors="coerce").mean()
            sel_m = pd.to_numeric(df_frames["Selected_Method_MPJPE"], errors="coerce").mean()
            dlt_p = pd.to_numeric(df_frames["Baseline_DLT_PA"], errors="coerce").mean()
            sel_p = pd.to_numeric(df_frames["Selected_Method_PA"], errors="coerce").mean()
            delta_m = ((dlt_m - sel_m) / dlt_m * 100.0) if dlt_m > 0 else 0.0
            delta_p = ((dlt_p - sel_p) / dlt_p * 100.0) if dlt_p > 0 else 0.0

            summary_rows.append({
                "Config": r["config"],
                "Subject": r["subject"],
                "Frames": len(df_frames),
                "DLT_MPJPE": round(dlt_m, 2),
                "Method_MPJPE": round(sel_m, 2),
                "Delta_MPJPE_%": round(delta_m, 2),
                "DLT_PA": round(dlt_p, 2),
                "Method_PA": round(sel_p, 2),
                "Delta_PA_%": round(delta_p, 2),
                "Elapsed_s": round(r["elapsed_s"], 1),
            })
        except Exception as e:
            print(f"Failed to parse {r['out_file']}: {e}")

    if not summary_rows:
        return

    summary_df = pd.DataFrame(summary_rows).sort_values(by=["Config", "Subject"])
    summary_csv = output_dir / "ablation_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print(f"\nSummary table saved to: {summary_csv}")


def main():
    args = parse_args()
    configs_dir = Path(args.configs_dir)
    output_dir = Path(args.output_dir)

    cfg_files = sorted(configs_dir.glob("*.yml"))
    if not cfg_files:
        print(f"Error: No YAML config files found in {configs_dir.resolve()}")
        sys.exit(1)

    print(f"Found {len(cfg_files)} configs in {configs_dir}:")
    for f in cfg_files:
        print(f"  - {f.name}")
    print(f"Subjects: {args.subjects}")
    print(f"Parallel workers: {args.workers}")
    print(f"Output directory: {output_dir.resolve()}\n")

    # Build task list: config x subject
    tasks = []
    for cfg in cfg_files:
        cfg_name = cfg.stem
        for subj in args.subjects:
            out_file = output_dir / cfg_name / f"{subj}.csv"
            log_file = output_dir / cfg_name / f"{subj}.log"
            tasks.append((cfg, subj, out_file, log_file))

    print(f"Queued {len(tasks)} tasks across {args.workers} workers.\n")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_single_task, cfg, subj, out_f, log_f, args.max_pairs, args.overwrite): (cfg.stem, subj)
            for cfg, subj, out_f, log_f in tasks
        }
        for future in as_completed(futures):
            res = future.result()
            results.append(res)

    generate_summary(results, output_dir)


if __name__ == "__main__":
    main()
