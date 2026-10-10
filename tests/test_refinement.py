import numpy as np
import pytest
from athlete_pose3d.algorithms.refinement import optimize_sequence


def test_temporal_terms_gap_masking():
    # Sequence of 6 frames with a gap: [0, 1, 2, 5, 6, 7]
    frame_ids = [0, 1, 2, 5, 6, 7]
    rng = np.random.default_rng(123)
    
    poses_3d = rng.uniform(-400, 400, size=(len(frame_ids), 17, 3))
    items = []
    for i, f in enumerate(frame_ids):
        items.append((i, {
            "frame": f,
            "all_methods": {"two_camera": {"recon_3d": poses_3d[i]}},
            "P1": np.eye(3, 4),
            "P2": np.eye(3, 4),
            "kps2d_a_h36m": rng.uniform(100, 500, size=(17, 2)),
            "conf_a_h36m": np.ones(17),
            "kps2d_b_h36m": rng.uniform(100, 500, size=(17, 2)),
            "conf_b_h36m": np.ones(17),
        }))
        
    refined, applied = optimize_sequence(
        items, "two_camera",
        smoothness_weight=0.15,
        velocity_weight=0.03,
        max_evaluations=10,
        return_status=True,
    )
    assert refined.shape == (6, 17, 3)
    assert np.isfinite(refined).all()
    assert applied is True
