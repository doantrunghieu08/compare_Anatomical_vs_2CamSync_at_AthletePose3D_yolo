import pandas as pd
import numpy as np

df = pd.read_csv('outputs/results.csv')
data = df[~df['Time'].str.startswith('Summary', na=False) & ~df['Time'].str.startswith('End', na=False)].copy()
for c in ['Selected_Method_PA', 'Baseline_DLT_PA', 'PA_Clear', 'PA_Derived']:
    data[c] = pd.to_numeric(data[c], errors='coerce')

print("=== S2 Results Summary ===")
print(f"method PA:  {data['Selected_Method_PA'].mean():.1f} mm")
print(f"DLT PA:     {data['Baseline_DLT_PA'].mean():.1f} mm")
print(f"PA_Clear:   {data['PA_Clear'].mean():.1f} mm")
print(f"PA_Derived: {data['PA_Derived'].mean():.1f} mm")

data['improved'] = data['Selected_Method_PA'] < data['Baseline_DLT_PA']
print(f"\nImproved frames: {data['improved'].sum()} / {len(data)} = {data['improved'].mean()*100:.0f}%")

# Check motions where clear joints are worse than derived
mg = data.groupby('Motion')[['PA_Clear', 'PA_Derived', 'Baseline_DLT_PA', 'Selected_Method_PA']].mean().round(1)
mg['clear_worse_than_derived'] = mg['PA_Clear'] > mg['PA_Derived']
print("\nPer-motion breakdown:")
print(mg.to_string())
print()

# The real question: is this S2 worse than S1?
# S1 ablation baseline was 67.7 mm PA-MPJPE
# S2 is 95.4 mm -- but these are DIFFERENT subjects + motions
print("Note: S1 Axel jump PA was ~63-68 mm (from ablation)")
print("      S2 Running PA is 95.4 mm -- Running_40/44/48/51 > 120mm are probably camera pair issues")

# Spot-check the worst motions
print("\nWorst motions (PA > 100mm):")
worst = mg[mg['Selected_Method_PA'] > 100]
print(worst[['PA_Clear', 'PA_Derived', 'Selected_Method_PA', 'Baseline_DLT_PA']].to_string())
