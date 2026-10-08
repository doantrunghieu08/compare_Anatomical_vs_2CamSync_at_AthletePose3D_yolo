"""A/B test: old vs new coco_to_h36m_2d on actual S2 data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np

# Load a coco pose file for S2
coco_file = Path('inputs/data/S2/Running_37_cam_1_coco.npy')
arr = np.load(str(coco_file))
print(f"COCO array shape: {arr.shape}")  # expect (N, 17, 3) -> (kp_x, kp_y, conf)

if arr.ndim == 3:
    sample = arr[0]  # First frame, shape (17, 3)
    kps = sample[:, :2]  # (17, 2) x,y
    conf = sample[:, 2]  # (17,) confidence
    
    nose = kps[0]
    l_shoulder = kps[5]
    r_shoulder = kps[6]
    thorax_est = 0.5 * (l_shoulder + r_shoulder)
    v = nose - thorax_est
    v_len = np.linalg.norm(v)
    
    print(f"\nFrame 0 geometry (pixels):")
    print(f"  Nose (COCO[0]):     {nose}")
    print(f"  L_Shoulder (5):     {l_shoulder}")
    print(f"  R_Shoulder (6):     {r_shoulder}")
    print(f"  Thorax estimate:    {thorax_est}")
    print(f"  Vector Thorax->Nose: {v}, len={v_len:.1f}")
    
    # Old: head2d = nose
    head_old = nose
    # New: head2d = thorax + 1.45 * v
    if v_len > 1e-4:
        head_new = thorax_est + 1.45 * v
        neck_new = thorax_est + 0.40 * (head_new - thorax_est)
    else:
        head_new = nose
        neck_new = 0.5 * (thorax_est + head_new)
    
    print(f"\n  OLD head2d:  {head_old}  (just nose)")
    print(f"  NEW head2d:  {head_new}  (extrapolated)")
    print(f"  NEW neck2d:  {neck_new}")
    print(f"  Extension beyond nose: {np.linalg.norm(head_new - nose):.1f} px")
    print(f"  -- NEW head is {np.linalg.norm(head_new - thorax_est):.1f} px from thorax")
    print(f"  -- OLD head was {v_len:.1f} px from thorax")
    print(f"  -- Ratio actual: {np.linalg.norm(head_new - thorax_est) / v_len:.2f}x")
    
    # Check across all frames
    noses = arr[:, 0, :2]
    l_shs = arr[:, 5, :2]
    r_shs = arr[:, 6, :2]
    thoraxes = 0.5 * (l_shs + r_shs)
    vs = noses - thoraxes
    v_lens = np.linalg.norm(vs, axis=1)
    heads_new = thoraxes + 1.45 * vs
    
    print(f"\nAcross {len(arr)} frames:")
    print(f"  Mean thorax->nose dist: {v_lens.mean():.1f} px")
    print(f"  Mean thorax->new_head:  {np.linalg.norm(heads_new - thoraxes, axis=1).mean():.1f} px")
    print(f"  Nose conf mean: {arr[:, 0, 2].mean():.2f}")

elif arr.ndim == 2:
    print("2D array (likely already H36M format), shape:", arr.shape)
