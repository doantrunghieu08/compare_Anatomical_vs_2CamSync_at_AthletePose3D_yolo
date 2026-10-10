import numpy as np
import pytest
from athlete_pose3d.algorithms.geometry import (
    triangulate_dlt,
    triangulate_anatomical,
    triangulate_reweighted_dlt,
    reproject,
)


def test_dlt_exact_reconstruction():
    # Construct two synthetic cameras
    k = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
    r1, t1 = np.eye(3), np.array([0.0, 0.0, 0.0])
    p1 = k @ np.column_stack((r1, t1))
    
    r2, t2 = np.eye(3), np.array([500.0, 0.0, 0.0])
    p2 = k @ np.column_stack((r2, t2))
    
    # 3D points in front of cameras (Z = 2000 mm)
    pts_3d = np.array([
        [0.0, 0.0, 2000.0],
        [100.0, -150.0, 2200.0],
        [-200.0, 300.0, 1800.0],
    ])
    
    pts1_2d = np.array([reproject(p1, pt) for pt in pts_3d])
    pts2_2d = np.array([reproject(p2, pt) for pt in pts_3d])
    
    recon = triangulate_dlt(p1, p2, pts1_2d, pts2_2d)
    diff = np.max(np.abs(recon - pts_3d))
    assert diff < 1e-4, f"DLT reconstruction error too high: {diff}"


def test_anatomical_custom_lr():
    p1 = np.column_stack((np.eye(3), np.zeros(3)))
    p2 = np.column_stack((np.eye(3), np.array([500.0, 0.0, 0.0])))
    pts1 = np.ones((17, 2), dtype=np.float32) * 200.0
    pts2 = np.ones((17, 2), dtype=np.float32) * 205.0
    c1 = np.ones(17, dtype=np.float32)
    c2 = np.ones(17, dtype=np.float32)
    
    # Check that lr=0.5 runs without error and returns finite 17x3 array
    res = triangulate_anatomical(p1, p2, pts1, pts2, c1, c2, iterations=5, lr=0.5)
    assert res.shape == (17, 3)
    assert np.isfinite(res).all()


def test_weighted_dlt_and_reweighted():
    from athlete_pose3d.algorithms.geometry import (
        _weighted_dlt_single,
        triangulate_conf_algebraic,
        dlt_single,
    )
    k = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
    p1 = k @ np.column_stack((np.eye(3), np.zeros(3)))
    p2 = k @ np.column_stack((np.eye(3), np.array([500.0, 0.0, 0.0])))
    
    pt3d = np.array([50.0, 50.0, 2000.0])
    pt1_2d = reproject(p1, pt3d)
    pt2_2d = reproject(p2, pt3d)
    
    # 1. dlt_single
    single_res = dlt_single(p1, p2, pt1_2d, pt2_2d)
    assert np.allclose(single_res, pt3d, atol=1e-3)
    
    # 2. _weighted_dlt_single
    w_res = _weighted_dlt_single(p1, p2, pt1_2d, pt2_2d, 1.0, 0.8)
    assert np.allclose(w_res, pt3d, atol=1e-3)
    
    # 3. triangulate_conf_algebraic
    pts1 = np.tile(pt1_2d, (17, 1))
    pts2 = np.tile(pt2_2d, (17, 1))
    c1, c2 = np.ones(17), np.ones(17)
    algebraic_res = triangulate_conf_algebraic(p1, p2, pts1, pts2, c1, c2)
    assert algebraic_res.shape == (17, 3)
    assert np.allclose(algebraic_res[0], pt3d, atol=1e-3)
    
    # 4. triangulate_reweighted_dlt
    reweighted_res = triangulate_reweighted_dlt(p1, p2, pts1, pts2, c1, c2)
    assert reweighted_res.shape == (17, 3)
    assert np.allclose(reweighted_res[0], pt3d, atol=1e-3)

