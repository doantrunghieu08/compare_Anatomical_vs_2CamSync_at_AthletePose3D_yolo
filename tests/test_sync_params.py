from dataclasses import replace
import pytest
from athlete_pose3d.settings import load_settings
from athlete_pose3d.pipeline import _fps_scaled_sync_params


def test_local_radius_zero_is_honored():
    _, _, cfg = load_settings("configs/default.yml")
    cfg_zero = replace(cfg, sync_local_radius=0)
    params = _fps_scaled_sync_params({"subject": "S1", "cameras": []}, cfg_zero)
    assert params["local_radius"] == 0, f"Expected local_radius 0, got {params['local_radius']}"


def test_local_radius_positive():
    _, _, cfg = load_settings("configs/default.yml")
    cfg_two = replace(cfg, sync_local_radius=2)
    params = _fps_scaled_sync_params({"subject": "S1", "cameras": []}, cfg_two)
    assert params["local_radius"] >= 2
