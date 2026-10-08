import pandas as pd
import numpy as np
from pathlib import Path

csv_path = Path('outputs/results.csv')
if not csv_path.exists():
    print("results.csv not found!")
    exit(1)

df = pd.read_csv(csv_path)
print("Unique subjects in results.csv:", df['Subject'].unique().tolist())

# Also check other csv files in outputs/
for f in Path('outputs').glob('*.csv'):
    print("Found CSV in outputs:", f.name)

for subj in ['S1', 'S2', 'S3']:
    sub_df = df[(df['Subject'] == subj) & (~df['Time'].str.startswith('Summary', na=False)) & (~df['Time'].str.startswith('End', na=False))].copy()
    if len(sub_df) == 0:
        print(f"\n=== {subj}: NO DATA in results.csv ===")
        continue
    sub_df['Selected_Method_PA'] = pd.to_numeric(sub_df['Selected_Method_PA'], errors='coerce')
    sub_df['Baseline_DLT_PA'] = pd.to_numeric(sub_df['Baseline_DLT_PA'], errors='coerce')
    sub_df['Selected_Method_MPJPE'] = pd.to_numeric(sub_df['Selected_Method_MPJPE'], errors='coerce')
    sub_df['Baseline_DLT_MPJPE'] = pd.to_numeric(sub_df['Baseline_DLT_MPJPE'], errors='coerce')
    
    mean_pa = sub_df['Selected_Method_PA'].mean()
    mean_dlt_pa = sub_df['Baseline_DLT_PA'].mean()
    delta_pa = ((mean_pa - mean_dlt_pa) / mean_dlt_pa) * 100
    
    mean_mp = sub_df['Selected_Method_MPJPE'].mean()
    mean_dlt_mp = sub_df['Baseline_DLT_MPJPE'].mean()
    delta_mp = ((mean_mp - mean_dlt_mp) / mean_dlt_mp) * 100
    
    better_pa = (sub_df['Selected_Method_PA'] < sub_df['Baseline_DLT_PA']).sum()
    better_mp = (sub_df['Selected_Method_MPJPE'] < sub_df['Baseline_DLT_MPJPE']).sum()
    n = len(sub_df)
    n_motions = sub_df['Motion'].nunique()
    
    print(f"\n=== {subj} ({n} frames, {n_motions} motions) ===")
    print(f"  PA-MPJPE: Selected = {mean_pa:.2f} mm | DLT Baseline = {mean_dlt_pa:.2f} mm | Diff = {delta_pa:+.2f}% | Better frames = {better_pa}/{n} ({better_pa/n*100:.1f}%)")
    print(f"  MPJPE:    Selected = {mean_mp:.2f} mm | DLT Baseline = {mean_dlt_mp:.2f} mm | Diff = {delta_mp:+.2f}% | Better frames = {better_mp}/{n} ({better_mp/n*100:.1f}%)")
