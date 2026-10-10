import numpy as np
import pytest
from athlete_pose3d.algorithms.geometry import raw_mpjpe, rigid_mpjpe, pa_mpjpe


def test_metric_ordering():
    rng = np.random.default_rng(42)
    # Generate ground truth 17 joints in 3D (mm scale)
    gt = rng.uniform(-500.0, 500.0, size=(17, 3))
    
    # Introduce arbitrary rotation, translation, scale error, and noise
    theta = np.radians(25.0)
    rot_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    trans = np.array([120.0, -80.0, 45.0])
    noise = rng.normal(0, 5.0, size=(17, 3))
    scale = 1.15
    
    pred = scale * (gt @ rot_z.T) + trans + noise
    
    raw = raw_mpjpe(pred, gt)
    rigid = rigid_mpjpe(pred, gt)
    pa = pa_mpjpe(pred, gt)
    
    assert raw >= rigid, f"Raw ({raw:.2f}) must be >= Rigid ({rigid:.2f})"
    assert rigid >= pa, f"Rigid ({rigid:.2f}) must be >= PA ({pa:.2f})"


def test_bootstrap_ci():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from run_ablation import bootstrap_ci
    
    deltas = np.array([2.0, 2.5, 3.0, 1.5, 2.2, 2.8])
    mean, ci = bootstrap_ci(deltas, n=1000, seed=42)
    assert np.isclose(mean, deltas.mean(), atol=1e-3)
    assert ci[0] <= mean <= ci[1]

