from __future__ import annotations

import getpass
import math
import os
import platform
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..algorithms.geometry import (
    CLEAR_JOINTS,
    DERIVED_JOINTS,
    H36M_EVAL_JOINTS,
    H36M_JOINT_NAMES,
    procrustes_align,
    rigid_align,
    sequence_diagnosis,
)
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


def _format_belief_array(arr: np.ndarray | list | None) -> str:
    """Format keypoint beliefs as a stringified list of floats in [0, 1] rounded to 3 decimals."""
    if arr is None:
        return "[]"
    vals = [round(float(np.clip(v, 0.0, 1.0)), 3) for v in np.asarray(arr, dtype=float).flatten()[:17]]
    return f"[{', '.join(f'{v:.3f}' for v in vals)}]"


def _extract_confidence_and_occlusion(result):
    """Extract per-joint stereo confidence and occlusion status for H36M joints.
    Uses multi-criteria stereo geometry & DST detection if available, falling back
    to 2D confidence thresholding (<= 0.3)."""
    if result.get("occluded_mask") is not None:
        occluded_mask = np.asarray(result["occluded_mask"], dtype=bool).flatten()[:17]
        ca = np.clip(np.nan_to_num(np.asarray(result.get("belief_master", result.get("conf_a_h36m", np.ones(17))), dtype=float).flatten()[:17], nan=0.0), 0.0, 1.0)
        cb = np.clip(np.nan_to_num(np.asarray(result.get("belief_slave", result.get("conf_b_h36m", np.ones(17))), dtype=float).flatten()[:17], nan=0.0), 0.0, 1.0)
        stereo_conf = np.clip(np.nan_to_num(np.asarray(result.get("belief_fusion", result.get("stereo_conf_h36m", 0.5 * (ca + cb))), dtype=float).flatten()[:17], nan=0.0), 0.0, 1.0)
        return ca, cb, stereo_conf, occluded_mask

    # If P1, P2, kps2d are available in result, run detect_stereo_occlusions dynamically
    p1, p2 = result.get("P1"), result.get("P2")
    kps_a = result.get("kps2d_a_h36m") if result.get("kps2d_a_h36m") is not None else result.get("kps2d_a")
    kps_b = result.get("kps2d_b_h36m") if result.get("kps2d_b_h36m") is not None else result.get("kps2d_b")
    if p1 is not None and p2 is not None and kps_a is not None and kps_b is not None:
        try:
            from ..algorithms.evidence_fusion import detect_stereo_occlusions
            occ_info = detect_stereo_occlusions(
                p1, p2, kps_a, kps_b,
                result.get("belief_master", result.get("conf_a_h36m")),
                result.get("belief_slave", result.get("conf_b_h36m")),
                initial_3d=result.get("recon_3d"),
            )
            return occ_info["conf_a"], occ_info["conf_b"], occ_info["stereo_confidence"], occ_info["occluded_mask"]
        except Exception:
            pass

    ca = result.get("belief_master", result.get("conf_a_h36m"))
    if ca is None:
        ca = result.get("conf_a")
    cb = result.get("belief_slave", result.get("conf_b_h36m"))
    if cb is None:
        cb = result.get("conf_b")

    if ca is not None and cb is not None:
        ca = np.asarray(ca, dtype=float).flatten()
        cb = np.asarray(cb, dtype=float).flatten()
        if len(ca) < 17:
            pad = np.ones(17 - len(ca))
            ca = np.concatenate([ca, pad])
        if len(cb) < 17:
            pad = np.ones(17 - len(cb))
            cb = np.concatenate([cb, pad])
        ca = np.clip(np.nan_to_num(ca[:17], nan=0.0), 0.0, 1.0)
        cb = np.clip(np.nan_to_num(cb[:17], nan=0.0), 0.0, 1.0)
    else:
        ca = np.ones(17, dtype=float)
        cb = np.ones(17, dtype=float)

    stereo_conf = np.clip(0.5 * (ca + cb), 0.0, 1.0)
    occluded_mask = (ca <= 0.3) | (cb <= 0.3)
    return ca, cb, stereo_conf, occluded_mask


def _compute_frame_joint_errors(result):
    """Compute per-joint MPJPE (root-relative) and occlusion metrics for a single frame result."""
    recon = result.get("recon_3d")
    gt = result.get("gt_3d")
    ca, cb, stereo_conf, occluded_mask = _extract_confidence_and_occlusion(result)

    eval_conf = stereo_conf[H36M_EVAL_JOINTS]
    mean_conf = round(float(np.mean(eval_conf)), 3)
    num_occl = int(np.sum(occluded_mask[H36M_EVAL_JOINTS]))

    if recon is not None and gt is not None and np.isfinite(recon).all() and np.isfinite(gt).all():
        recon = np.asarray(recon, dtype=float)
        gt = np.asarray(gt, dtype=float)
        aligned_rigid = rigid_align(recon, gt, mask=H36M_EVAL_JOINTS)
        joint_errs = np.linalg.norm(aligned_rigid - gt, axis=-1)

        eval_errs = joint_errs[H36M_EVAL_JOINTS]
        eval_occl = occluded_mask[H36M_EVAL_JOINTS]

        unoccl_errs = eval_errs[~eval_occl]
        occl_errs = eval_errs[eval_occl]

        mpjpe_unoccl = round(float(np.mean(unoccl_errs)), 2) if len(unoccl_errs) > 0 else float("nan")
        mpjpe_occl = round(float(np.mean(occl_errs)), 2) if len(occl_errs) > 0 else float("nan")
    else:
        joint_errs = np.full(17, float("nan"))
        mpjpe_unoccl = float("nan")
        mpjpe_occl = float("nan")

    return {
        "mean_conf": mean_conf,
        "mpjpe_unoccl": mpjpe_unoccl,
        "mpjpe_occl": mpjpe_occl,
        "num_occl": num_occl,
        "ca": ca,
        "cb": cb,
        "stereo_conf": stereo_conf,
        "occluded_mask": occluded_mask,
        "joint_errs": joint_errs,
    }


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

    metrics = _compute_frame_joint_errors(result)

    return [
        result.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        str(result["motion"]), str(result["subject"]),
        str(result["cam_a"]), str(result["cam_b"]), int(result["frame"]),
        round(b_mpjpe, 2), round(b_pa, 2), round(s_mpjpe, 2), round(s_pa, 2),
        d_mpjpe, d_pa,
        pa_c, pa_d,
        metrics["mean_conf"],
        metrics["mpjpe_unoccl"] if not math.isnan(metrics["mpjpe_unoccl"]) else "N/A",
        metrics["mpjpe_occl"] if not math.isnan(metrics["mpjpe_occl"]) else "N/A",
        metrics["num_occl"],
        _format_belief_array(metrics["ca"]),
        _format_belief_array(metrics["cb"]),
        _format_belief_array(metrics["stereo_conf"]),
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

    all_metrics = [_compute_frame_joint_errors(r) for r in results]
    conf_vals = [m["mean_conf"] for m in all_metrics if not math.isnan(m["mean_conf"])]
    unoccl_vals = [m["mpjpe_unoccl"] for m in all_metrics if not math.isnan(m["mpjpe_unoccl"])]
    occl_vals = [m["mpjpe_occl"] for m in all_metrics if not math.isnan(m["mpjpe_occl"])]
    num_occl_vals = [m["num_occl"] for m in all_metrics]

    conf_mean = round(sum(conf_vals) / len(conf_vals), 3) if conf_vals else "N/A"
    unoccl_mean = round(sum(unoccl_vals) / len(unoccl_vals), 2) if unoccl_vals else "N/A"
    occl_mean = round(sum(occl_vals) / len(occl_vals), 2) if occl_vals else "N/A"
    num_occl_mean = round(sum(num_occl_vals) / len(num_occl_vals), 1) if num_occl_vals else 0

    return [
        "Summary_Mean", "ALL", "ALL", "-", "-", n,
        b_m_str, b_p_str, round(s_m, 2), round(s_p, 2),
        d_m_str, d_p_str,
        pa_c_mean, pa_d_mean,
        conf_mean, unoccl_mean, occl_mean, num_occl_mean,
        "-", "-", "-",
        "AVERAGE", 0, 0,
        system.python_version, system.os_type, system.os_version,
        system.compute_device, str(system.cpu_cores), system.user,
        version, f"Mean across all evaluated frames; DLT valid: {n_dlt}/{n}",
    ]


def _build_per_joint_analysis(results: list[dict]) -> pd.DataFrame:
    """Build detailed per-joint reliability, occlusion, and 3D error statistics across all frames."""
    n_joints = len(H36M_JOINT_NAMES)

    # Per-joint accumulators
    conf_a_acc = [[] for _ in range(n_joints)]
    conf_b_acc = [[] for _ in range(n_joints)]
    conf_stereo_acc = [[] for _ in range(n_joints)]
    occluded_acc = [[] for _ in range(n_joints)]

    method_err_acc = [[] for _ in range(n_joints)]
    method_pa_err_acc = [[] for _ in range(n_joints)]
    dlt_err_acc = [[] for _ in range(n_joints)]

    for r in results:
        ca, cb, stereo_conf, occl_mask = _extract_confidence_and_occlusion(r)
        for j in range(n_joints):
            conf_a_acc[j].append(float(ca[j]))
            conf_b_acc[j].append(float(cb[j]))
            conf_stereo_acc[j].append(float(stereo_conf[j]))
            occluded_acc[j].append(bool(occl_mask[j]))

        recon = r.get("recon_3d")
        gt = r.get("gt_3d")
        if recon is not None and gt is not None and np.isfinite(recon).all() and np.isfinite(gt).all():
            recon = np.asarray(recon, dtype=float)
            gt = np.asarray(gt, dtype=float)
            aligned_rigid = rigid_align(recon, gt, mask=H36M_EVAL_JOINTS)
            errs = np.linalg.norm(aligned_rigid - gt, axis=-1)

            aligned = procrustes_align(recon, gt, mask=H36M_EVAL_JOINTS)
            pa_errs = np.linalg.norm(aligned - gt, axis=-1)

            for j in range(n_joints):
                method_err_acc[j].append(float(errs[j]))
                method_pa_err_acc[j].append(float(pa_errs[j]))

        # DLT baseline errors
        all_m = r.get("all_methods", {})
        dlt_m = all_m.get("DLT (baseline)") or all_m.get("DLT (raw baseline)")
        if dlt_m and "recon_3d" in dlt_m and dlt_m["recon_3d"] is not None:
            dlt_recon = np.asarray(dlt_m["recon_3d"], dtype=float)
            if np.isfinite(dlt_recon).all() and gt is not None and np.isfinite(gt).all():
                dlt_aligned = rigid_align(dlt_recon, gt, mask=H36M_EVAL_JOINTS)
                d_errs = np.linalg.norm(dlt_aligned - gt, axis=-1)
                for j in range(n_joints):
                    dlt_err_acc[j].append(float(d_errs[j]))

    rows = []
    for j in range(n_joints):
        name = H36M_JOINT_NAMES[j]
        if j == 0:
            category = "Root"
        elif j in CLEAR_JOINTS:
            category = "Clear"
        else:
            category = "Derived"

        ca_m = np.mean(conf_a_acc[j]) if conf_a_acc[j] else float("nan")
        cb_m = np.mean(conf_b_acc[j]) if conf_b_acc[j] else float("nan")
        cs_m = np.mean(conf_stereo_acc[j]) if conf_stereo_acc[j] else float("nan")
        occl_rate = np.mean(occluded_acc[j]) * 100.0 if occluded_acc[j] else 0.0

        m_err = method_err_acc[j]
        m_pa_err = method_pa_err_acc[j]
        d_err = dlt_err_acc[j]
        occ = occluded_acc[j]

        m_mean = np.mean(m_err) if m_err else float("nan")
        m_pa_mean = np.mean(m_pa_err) if m_pa_err else float("nan")
        d_mean = np.mean(d_err) if d_err else float("nan")

        unoccl_errs = [e for e, o in zip(m_err, occ) if not o]
        occl_errs = [e for e, o in zip(m_err, occ) if o]

        unoccl_mean = np.mean(unoccl_errs) if unoccl_errs else float("nan")
        occl_mean = np.mean(occl_errs) if occl_errs else float("nan")

        delta_pct = (
            round((d_mean - m_mean) / d_mean * 100.0, 2)
            if not math.isnan(d_mean) and abs(d_mean) > 1e-9 and not math.isnan(m_mean)
            else float("nan")
        )

        rows.append({
            "Joint_ID": j,
            "Joint_Name": name,
            "Group": category,
            "Conf_CamA": round(ca_m, 3),
            "Conf_CamB": round(cb_m, 3),
            "Conf_Stereo": round(cs_m, 3),
            "Occlusion_Rate_pct": round(occl_rate, 1),
            "DLT_MPJPE_mm": round(d_mean, 2) if not math.isnan(d_mean) else "N/A",
            "Method_MPJPE_mm": round(m_mean, 2) if not math.isnan(m_mean) else "N/A",
            "Method_PA_MPJPE_mm": round(m_pa_mean, 2) if not math.isnan(m_pa_mean) else "N/A",
            "MPJPE_Unoccluded_mm": round(unoccl_mean, 2) if not math.isnan(unoccl_mean) else "N/A",
            "MPJPE_Occluded_mm": round(occl_mean, 2) if not math.isnan(occl_mean) else "N/A",
            "Delta_vs_DLT_pct": delta_pct if not math.isnan(delta_pct) else "N/A",
        })

    df = pd.DataFrame(rows)
    return df


class CsvResultReporter:
    """Append benchmark rows to a local CSV with per-joint and occlusion summary."""

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
        df_per_joint = None
        if self._all_results:
            df_per_joint = _build_per_joint_analysis(self._all_results)
            per_joint_path = self.output_csv.parent / f"{self.output_csv.stem}_per_joint_summary.csv"
            df_per_joint.to_csv(per_joint_path, index=False)
            print(f"Saved per-joint detailed summary to {per_joint_path}")

        if summary:
            self._append([summary])
            self._print_summary(summary, df_per_joint)

        # Sequence-level rotation & scale diagnosis per camera pair
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

    def _print_summary(self, s, df_per_joint: pd.DataFrame | None = None):
        def _fmt(v, sign=False):
            if isinstance(v, str):
                return v
            return f"{v:+.2f}" if sign else f"{v:.2f}"

        print("\n" + "=" * 60)
        print(f"BENCHMARK SUMMARY (N = {s[5]} frames)")
        print(f"Baseline DLT MPJPE:    {_fmt(s[6])} mm | PA: {_fmt(s[7])} mm")
        print(f"Selected Method MPJPE: {_fmt(s[8])} mm | PA: {_fmt(s[9])} mm")
        print(f"Delta (Base - New):    {_fmt(s[10], sign=True)} % | PA: {_fmt(s[11], sign=True)} %")
        if s[12] != "N/A" and s[13] != "N/A":
            print(f"Joint Groups:          PA-Clear: {s[12]} mm | PA-Derived: {s[13]} mm")
        print(f"Joint Reliability:     Mean Conf: {s[14]} | Occluded Joints/Frame: {s[17]}")
        print(f"Occlusion Breakdown:   Unoccluded MPJPE: {s[15]} mm | Occluded MPJPE: {s[16]} mm")
        print("=" * 60)

        if df_per_joint is not None and not df_per_joint.empty:
            print("\n" + "-" * 98)
            print(f"{'ID':>2} | {'Joint Name':<12} | {'Group':<7} | {'Conf':>5} | {'Occl%':>6} | {'DLT(mm)':>8} | {'Method(mm)':>10} | {'PA(mm)':>8} | {'Unoccl(mm)':>10} | {'Occl(mm)':>9}")
            print("-" * 98)
            for _, r in df_per_joint.iterrows():
                jid = r["Joint_ID"]
                name = r["Joint_Name"]
                grp = r["Group"]
                conf = f"{r['Conf_Stereo']:.3f}" if isinstance(r['Conf_Stereo'], (int, float)) else str(r['Conf_Stereo'])
                occl_pct = f"{r['Occlusion_Rate_pct']:.1f}%" if isinstance(r['Occlusion_Rate_pct'], (int, float)) else str(r['Occlusion_Rate_pct'])
                dlt = f"{r['DLT_MPJPE_mm']}"
                meth = f"{r['Method_MPJPE_mm']}"
                pa = f"{r['Method_PA_MPJPE_mm']}"
                unoccl = f"{r['MPJPE_Unoccluded_mm']}"
                occl = f"{r['MPJPE_Occluded_mm']}"
                print(f"{jid:>2} | {name:<12} | {grp:<7} | {conf:>5} | {occl_pct:>6} | {dlt:>8} | {meth:>10} | {pa:>8} | {unoccl:>10} | {occl:>9}")
            print("-" * 98 + "\n")

    def _append(self, rows):
        exists = self.output_csv.exists()
        pd.DataFrame(rows, columns=REPORT_HEADERS).to_csv(
            self.output_csv,
            mode="a" if exists else "w",
            header=not exists,
            index=False,
        )

