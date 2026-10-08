from __future__ import annotations

import getpass
import math
import os
import platform
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..settings import REPORT_HEADERS


@dataclass(frozen=True)
class SystemInfo:
    python_version: str
    os_type: str
    os_version: str
    compute_device: str
    cpu_cores: int | None
    user: str

    @classmethod
    def collect(cls):
        device = (
            f"GPU: {torch.cuda.get_device_name(0)}"
            if torch.cuda.is_available()
            else f"CPU: {platform.processor()}"
        )
        return cls(
            platform.python_version(), platform.system(), platform.release(),
            device, os.cpu_count(), getpass.getuser(),
        )


def _format_single_report_row(result, system: SystemInfo, version: str, notes: str):
    all_m = result.get("all_methods", {})
    dlt_m = all_m.get("DLT (baseline)") or all_m.get("DLT (raw baseline)") or {}
    b_mpjpe = float(result.get("baseline_dlt_mpjpe") if result.get("baseline_dlt_mpjpe") is not None else dlt_m.get("mpjpe", float("nan")))
    b_pa = float(result.get("baseline_dlt_pa") if result.get("baseline_dlt_pa") is not None else dlt_m.get("pa_mpjpe", float("nan")))
    s_mpjpe, s_pa = float(result["mpjpe"]), float(result["pa_mpjpe"])

    d_mpjpe = (
        round((b_mpjpe - s_mpjpe) / b_mpjpe * 100.0, 2)
        if not math.isnan(b_mpjpe) and abs(b_mpjpe) > 1e-9
        else float("nan")
    )
    d_pa = (
        round((b_pa - s_pa) / b_pa * 100.0, 2)
        if not math.isnan(b_pa) and abs(b_pa) > 1e-9
        else float("nan")
    )

    pa_c = round(float(result["pa_clear"]), 2) if result.get("pa_clear") is not None and not math.isnan(float(result["pa_clear"])) else float("nan")
    pa_d = round(float(result["pa_derived"]), 2) if result.get("pa_derived") is not None and not math.isnan(float(result["pa_derived"])) else float("nan")

    return [
        result.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        str(result["motion"]), str(result["subject"]),
        str(result["cam_a"]), str(result["cam_b"]), int(result["frame"]),
        round(b_mpjpe, 2), round(b_pa, 2), round(s_mpjpe, 2), round(s_pa, 2),
        d_mpjpe, d_pa,
        pa_c, pa_d,
        str(result["best_method"]),
        int(result.get("global_sync_delta", 0)),
        int(result.get("dynamic_sync_delta", 0)),
        system.python_version, system.os_type, system.os_version,
        system.compute_device, str(system.cpu_cores), system.user,
        version, notes,
    ]


def build_report_rows(results, system: SystemInfo, version: str, notes: str):
    return [_format_single_report_row(r, system, version, notes) for r in results]


def _build_summary_row(results, system: SystemInfo, version: str):
    if not results:
        return None
    n = len(results)
    b_mpjpe_vals = [float(r["baseline_dlt_mpjpe"]) for r in results
                    if not math.isnan(float(r.get("baseline_dlt_mpjpe", float("nan"))))]
    b_pa_vals = [float(r["baseline_dlt_pa"]) for r in results
                 if not math.isnan(float(r.get("baseline_dlt_pa", float("nan"))))]
    b_m = sum(b_mpjpe_vals) / len(b_mpjpe_vals) if b_mpjpe_vals else float("nan")
    b_p = sum(b_pa_vals) / len(b_pa_vals) if b_pa_vals else float("nan")
    s_m = sum(float(r["mpjpe"]) for r in results) / n
    s_p = sum(float(r["pa_mpjpe"]) for r in results) / n
    n_dlt = len(b_pa_vals)
    b_m_str = round(b_m, 2) if not math.isnan(b_m) else "N/A"
    b_p_str = round(b_p, 2) if not math.isnan(b_p) else "N/A"
    d_m_str = (
        round((b_m - s_m) / b_m * 100.0, 2)
        if not math.isnan(b_m) and abs(b_m) > 1e-9
        else "N/A"
    )
    d_p_str = (
        round((b_p - s_p) / b_p * 100.0, 2)
        if not math.isnan(b_p) and abs(b_p) > 1e-9
        else "N/A"
    )
    pa_c_vals = [float(r["pa_clear"]) for r in results if r.get("pa_clear") is not None and not math.isnan(float(r["pa_clear"]))]
    pa_d_vals = [float(r["pa_derived"]) for r in results if r.get("pa_derived") is not None and not math.isnan(float(r["pa_derived"]))]
    pa_c_mean = round(sum(pa_c_vals) / len(pa_c_vals), 2) if pa_c_vals else "N/A"
    pa_d_mean = round(sum(pa_d_vals) / len(pa_d_vals), 2) if pa_d_vals else "N/A"

    return [
        "Summary_Mean", "ALL", "ALL", "-", "-", n,
        b_m_str, b_p_str, round(s_m, 2), round(s_p, 2),
        d_m_str, d_p_str,
        pa_c_mean, pa_d_mean,
        "AVERAGE", 0, 0,
        system.python_version, system.os_type, system.os_version,
        system.compute_device, str(system.cpu_cores), system.user,
        version, f"Mean across all evaluated frames; DLT valid: {n_dlt}/{n}",
    ]


class CsvResultReporter:
    """Append benchmark rows to a local CSV."""

    def __init__(self, output_csv: str | Path, version="local", notes=""):
        self.output_csv = Path(output_csv)
        self.version = version
        self.notes = notes
        self.system = SystemInfo.collect()
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self._all_results = []

    def append(self, results):
        self._all_results.extend(results)
        rows = build_report_rows(results, self.system, self.version, self.notes)
        if rows:
            self._append(rows)
            print(f"Saved {len(rows)} rows to {self.output_csv}")

    def finish(self):
        summary = _build_summary_row(self._all_results, self.system, self.version)
        if summary:
            self._append([summary])
            self._print_summary(summary)
        # ponytail: compute sequence-level rotation & scale diagnosis per camera pair
        from collections import defaultdict
        from ..algorithms.geometry import sequence_diagnosis

        pair_groups = defaultdict(list)
        for r in self._all_results:
            if (
                "recon_3d" in r
                and r.get("gt_3d") is not None
                and np.isfinite(r["recon_3d"]).all()
                and np.isfinite(r["gt_3d"]).all()
            ):
                key = (r.get("subject", ""), r.get("motion", ""), r.get("cam_a", ""), r.get("cam_b", ""))
                pair_groups[key].append((r["recon_3d"], r["gt_3d"]))

        if pair_groups:
            pair_diags = []
            for key, pairs in pair_groups.items():
                if len(pairs) >= 1:
                    preds, gts = zip(*pairs)
                    diag = sequence_diagnosis(np.stack(preds), np.stack(gts))
                    pair_diags.append((len(pairs), diag))

            total_frames = sum(cnt for cnt, _ in pair_diags)
            if total_frames > 0:
                mean_angle = sum(cnt * d["angle_deg"] for cnt, d in pair_diags) / total_frames
                mean_scale = sum(cnt * d["scale_ratio"] for cnt, d in pair_diags) / total_frames
                mean_rot = sum(cnt * d["mpjpe_seq_rot_mm"] for cnt, d in pair_diags) / total_frames
                mean_sim = sum(cnt * d["mpjpe_seq_sim_mm"] for cnt, d in pair_diags) / total_frames
                print(f"Sequence Diagnosis (Per Camera Pair, N = {len(pair_diags)} pairs, {total_frames} frames):")
                print(f"  Mean Rotation Angle: {mean_angle:.1f} deg (Rotation offset between Camera 1 and Mocap GT)")
                print(f"  Mean Scale Ratio:    {mean_scale:.3f} (Metric scale ratio vs GT)")
                print(f"  MPJPE-SeqRot:        {mean_rot:.2f} mm (MPJPE after aligning Camera 1 -> GT coordinate frame)")
                print(f"  MPJPE-SeqSim:        {mean_sim:.2f} mm (MPJPE after rigid rotation + scale alignment)")
        self._append([["End"] * len(REPORT_HEADERS)])

    def _print_summary(self, s):
        def _fmt(v, sign=False):
            if isinstance(v, str):
                return v
            return f"{v:+.2f}" if sign else f"{v:.2f}"

        print("\n" + "=" * 52)
        print(f"BENCHMARK SUMMARY (N = {s[5]} frames)")
        print(f"Baseline DLT MPJPE:    {_fmt(s[6])} mm | PA: {_fmt(s[7])} mm")
        print(f"Selected Method MPJPE: {_fmt(s[8])} mm | PA: {_fmt(s[9])} mm")
        print(f"Delta (Base - New):    {_fmt(s[10], sign=True)} % | PA: {_fmt(s[11], sign=True)} %")
        if s[12] != "N/A" and s[13] != "N/A":
            print(f"Joint Groups:          PA-Clear: {s[12]} mm | PA-Derived: {s[13]} mm")
        print("=" * 52 + "\n")


    def _append(self, rows):
        exists = self.output_csv.exists()
        pd.DataFrame(rows, columns=REPORT_HEADERS).to_csv(
            self.output_csv,
            mode="a" if exists else "w",
            header=not exists,
            index=False,
        )
