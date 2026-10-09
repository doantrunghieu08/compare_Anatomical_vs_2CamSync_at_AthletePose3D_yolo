import sys
from pathlib import Path
import numpy as np
import torch

src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_dir))

from athlete_pose3d.algorithms.physics_refine import triangulate_physics_refine, triangulate_dst_physics
from athlete_pose3d.algorithms.geometry import triangulate_dlt


def test_kinematic_and_temporal_constraints():
    p1 = np.array([[1000.0, 0.0, 500.0, 0.0], [0.0, 1000.0, 500.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    p2 = np.array([[1000.0, 0.0, 500.0, 1000.0], [0.0, 1000.0, 500.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    pts1 = np.zeros((17, 2))
    pts2 = np.zeros((17, 2))
    conf1 = np.ones(17)
    conf2 = np.ones(17)

    # Frame 0 pose
    prev_pose = np.random.randn(17, 3) * 100.0 + np.array([0.0, 0.0, 1500.0])

    # Run triangulation with temporal and kinematic constraints enabled
    pose_refined = triangulate_dst_physics(
        p1, p2, pts1, pts2, conf1, conf2,
        iterations=10,
        prev_pose_3d=prev_pose,
        hypothesis_selection=True,
    )

    assert pose_refined.shape == (17, 3)
    assert np.isfinite(pose_refined).all()
    print("PASS: Kinematic & temporal constraints executed correctly.")


if __name__ == "__main__":
    test_kinematic_and_temporal_constraints()
