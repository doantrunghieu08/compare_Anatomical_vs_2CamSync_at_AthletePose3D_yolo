#!/usr/bin/env python3
"""Generate clean 2x2 ablation configurations based on configs/default.yml."""

from __future__ import annotations

import copy
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs" / "ablation"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yml"
base = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def create_variant(name: str, method: str, height: float | str, refine: bool, smooth: float = 0.15, vel: float = 0.0):
    cfg = copy.deepcopy(base)
    cfg["output"]["results_csv"] = f"outputs/ablation/{name}.csv"
    cfg["output"]["overwrite"] = True
    cfg["pipeline"]["subject_height_mm"] = height
    cfg["pipeline"]["sequence_refinement"].update(
        enabled=refine,
        smoothness_weight=smooth,
        velocity_weight=vel,
    )
    if method == "dlt":
        cfg["method"] = {"name": "dlt", "options": {}}
    else:
        cfg["method"] = {
            "name": "anatomical",
            "options": {"iterations": 80, "bone_weight": 1.0, "lr": 0.1},
        }

    out_file = CONFIG_DIR / f"{name}.yml"
    out_file.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Generated {out_file.relative_to(REPO_ROOT)}")


def main():
    print("Generating 2x2 Clean Ablation Matrix configs...")
    create_variant("A1_dlt_generic", "dlt", 1730.0, False)
    create_variant("A2_dlt_calibrated", "dlt", "auto", False)
    create_variant("B1_anat_generic", "anatomical", 1730.0, False)
    create_variant("B2_anat_calibrated", "anatomical", "auto", False)
    create_variant("C1_refine_accel", "anatomical", "auto", True, smooth=0.15, vel=0.0)
    create_variant("C2_refine_accel_vel", "anatomical", "auto", True, smooth=0.15, vel=0.03)
    print("Done!")


if __name__ == "__main__":
    main()
