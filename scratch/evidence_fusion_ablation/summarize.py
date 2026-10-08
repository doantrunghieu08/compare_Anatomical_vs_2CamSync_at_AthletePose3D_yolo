from pathlib import Path
import pandas as pd
root = Path.cwd()
out = root / "outputs" / "evidence_fusion_ablation"
def load(name):
    data = pd.read_csv(out / (name + ".csv"))
    data = data[data["Subject"].isin(["S1", "S2", "S3"])].copy()
    data["Frame"] = pd.to_numeric(data["Frame"], errors="coerce")
    for col in ["Baseline_DLT_MPJPE", "Baseline_DLT_PA", "Selected_Method_MPJPE", "Selected_Method_PA"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data[data["Frame"].notna()]
legacy = pd.concat([load("A_legacy"), load("A_legacy_S2"), load("A_legacy_S3")], ignore_index=True)
candidate = pd.concat([load("D_candidate_S1"), load("D_candidate_S2"), load("D_candidate_S3")], ignore_index=True)
candidate_s1_motions = set(candidate.loc[candidate.Subject == "S1", "Motion"])
legacy = legacy[(legacy.Subject != "S1") | legacy.Motion.isin(candidate_s1_motions)]
keys = ["Subject", "Motion", "Cam_A", "Cam_B", "Frame"]
left = legacy[keys + ["Selected_Method_MPJPE", "Selected_Method_PA"]].rename(columns={
    "Selected_Method_MPJPE": "Legacy_MPJPE",
    "Selected_Method_PA": "Legacy_PA",
})
right = candidate[keys + [
    "Baseline_DLT_MPJPE", "Baseline_DLT_PA", "Selected_Method_MPJPE", "Selected_Method_PA"
]].rename(columns={
    "Selected_Method_MPJPE": "Candidate_MPJPE",
    "Selected_Method_PA": "Candidate_PA",
})
matched = left.merge(right, on=keys, how="inner", validate="one_to_one")
matched["Candidate_vs_DLT_pct_frame"] = 100 * (matched.Candidate_MPJPE / matched.Baseline_DLT_MPJPE - 1)
matched["Candidate_vs_Legacy_pct_frame"] = 100 * (matched.Candidate_MPJPE / matched.Legacy_MPJPE - 1)
rows = []
groups = list(matched.groupby("Subject"))
groups.append(("ALL", matched))
for subject, group in groups:
    dlt, old, new = (group[name].mean() for name in [
        "Baseline_DLT_MPJPE", "Legacy_MPJPE", "Candidate_MPJPE"
    ])
    rows.append({
        "Subject": subject,
        "Frames": len(group),
        "Motions": group.Motion.nunique(),
        "DLT_MPJPE_mm": dlt,
        "Legacy_MPJPE_mm": old,
        "Candidate_MPJPE_mm": new,
        "Candidate_vs_DLT_mean_pct": 100 * (new / dlt - 1),
        "Candidate_win_rate_vs_DLT_pct": 100 * (group.Candidate_MPJPE < group.Baseline_DLT_MPJPE).mean(),
        "Candidate_vs_Legacy_mean_pct": 100 * (new / old - 1),
        "Candidate_win_rate_vs_Legacy_pct": 100 * (group.Candidate_MPJPE < group.Legacy_MPJPE).mean(),
        "DLT_PA_mm": group.Baseline_DLT_PA.mean(),
        "Legacy_PA_mm": group.Legacy_PA.mean(),
        "Candidate_PA_mm": group.Candidate_PA.mean(),
        "Candidate_vs_DLT_PA_mean_pct": 100 * (group.Candidate_PA.mean() / group.Baseline_DLT_PA.mean() - 1),
    })
summary = pd.DataFrame(rows)
summary.to_csv(out / "subject_comparison.csv", index=False, float_format="%.4f")
matched.to_csv(out / "matched_frames.csv", index=False, float_format="%.4f")
print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
print("matched frame rows:", len(matched))
