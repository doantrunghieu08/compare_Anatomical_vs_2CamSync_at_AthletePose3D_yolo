from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


def _dir_has_test_videos(path: Path) -> bool:
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if child.is_dir() and (list(child.glob("*_cam_*")) or child.name in ("S1", "S2", "S3")):
            return True
        if child.is_file() and "_cam_" in child.name:
            return True
    return False


@dataclass(frozen=True)
class InputSettings:
    dataset_root: Path
    pose2d_root: Path
    test_set_subdir: Path
    manifest_file: str
    video_extension: str
    metadata_extension: str
    ground_truth_extension: str
    pose_2d_suffix: str
    key_frames_suffix: str
    pose_cache_limit: int
    minimum_manifest_confidence: float

    @property
    def test_set_dir(self) -> Path:
        primary = self.dataset_root / self.test_set_subdir
        if _dir_has_test_videos(primary):
            return primary
        for candidate in (
            self.dataset_root / "data" / "data" / "test_set",
            self.dataset_root / "data",
            self.dataset_root,
        ):
            if _dir_has_test_videos(candidate):
                return candidate
        return primary

    @property
    def manifest(self) -> Path:
        primary = self.pose2d_root / self.manifest_file
        if primary.is_file():
            return primary
        fallback = self.pose2d_root.parent / self.manifest_file
        return fallback if fallback.is_file() else primary


@dataclass(frozen=True)
class OutputSettings:
    results_csv: Path
    overwrite: bool


@dataclass(frozen=True)
class BenchmarkConfig:
    max_pairs: int
    chunk_size: int
    frames_per_pair: int
    sync_max_offset: int
    sync_samples: int
    sync_coarse_step: int
    sync_local_radius: int
    dynamic_local_radius: int
    max_pair_candidates: int
    max_dynamic_sync_score: float
    sync_minimum_joint_confidence: float
    sync_minimum_valid_joints: int
    minimum_pair_valid_ratio: float
    dynamic_transition_weight: float
    minimum_joint_confidence: float
    minimum_valid_joints: int
    subject_height_mm: float | str
    subject_heights: dict[str, float]
    subject_gt_joint_markers: dict[str, dict[int, list[str]]]
    excluded_subjects: tuple[str, ...]
    excluded_motions: frozenset[str]
    triangulation_method: str
    method_options: dict[str, float | int | bool | str | list[str] | None]
    sequence_refinement: dict[str, float | int | bool]
    bone_prior: str
    bone_prior_min_samples: int
    bone_prior_min_confidence: float
    reference_fps: float = 60.0
    fps_mode: str | float = "auto"
    f_scale: str | float = "auto"
    subject_fps: dict[str, str | float] = field(default_factory=dict)
    included_subjects: tuple[str, ...] = ()


TWOCAM_METHOD = "TwoCam DynamicSync"
REFINED_METHOD = "TwoCam DynamicSync + SMPLifyStyle"

METHOD_LABELS = {
    "dlt": "DLT (baseline)",
    "confidence_algebraic": "Conf-Algebraic",
    "ransac_dlt": "RANSAC-DLT",
    "iterative_refine": "Iterative Refine",
    "anatomical": "Anatomical (SOTA)",
    "dst_anatomical": "DST-Anatomical",
    "physics_refine": "Physics-Refine",
    "dst_physics": "DST+Physics",
}

REPORT_HEADERS = [
    "Time", "Motion", "Subject", "Cam_A", "Cam_B", "Frame",
    "Baseline_DLT_MPJPE", "Baseline_DLT_PA", "Selected_Method_MPJPE",
    "Selected_Method_PA", "Delta_MPJPE_pct", "Delta_PA_pct",
    "PA_Clear", "PA_Derived",
    "Mean_Confidence", "MPJPE_Unoccluded", "MPJPE_Occluded", "Num_Occluded_Joints",
    "Belief_Master", "Belief_Slave", "Belief_Fusion",
    "Best_Method",
    "Global_Sync_Delta", "Dynamic_Sync_Delta", "Python_Version",
    "OS_Type", "OS_Version", "Compute_Device", "CPU cores",
    "User_Info", "Code version", "Notes",
]



def method_label(method: str) -> str:
    try:
        return METHOD_LABELS[method]
    except KeyError as error:
        choices = ", ".join(METHOD_LABELS)
        raise ValueError(f"Unknown triangulation method '{method}'. Choose one of: {choices}") from error


def _section(raw, name, schema):
    values = dict(raw.get(name, {}))
    required = {item.name for item in fields(schema)}
    missing, unknown = required - values.keys(), values.keys() - required
    if missing or unknown:
        raise ValueError(f"Invalid {name}: missing={sorted(missing)}, unknown={sorted(unknown)}")
    return values


def _project_root(config_path):
    resolved = config_path.resolve()
    for parent in resolved.parents:
        if parent.name == "configs":
            return parent.parent
    return resolved.parent


def _load_input(config_path, raw):
    values = _section(raw, "input", InputSettings)
    root = _project_root(config_path)
    for name in ("dataset_root", "pose2d_root"):
        path = Path(values[name]).expanduser()
        values[name] = path if path.is_absolute() else root / path
    values["test_set_subdir"] = Path(values["test_set_subdir"])
    return InputSettings(**values)


def _load_output(config_path, raw):
    values = _section(raw, "output", OutputSettings)
    path = Path(values["results_csv"]).expanduser()
    values["results_csv"] = path if path.is_absolute() else _project_root(config_path) / path
    if values["results_csv"].suffix.lower() != ".csv":
        raise ValueError("output.results_csv must end with .csv")
    return OutputSettings(**values)


def _load_benchmark(raw):
    values = dict(raw.get("pipeline", {}))
    method, refinement = raw.get("method", {}), values.pop("sequence_refinement", {})
    values.setdefault("subject_heights", {})
    values.setdefault("subject_gt_joint_markers", {})
    values.setdefault("subject_fps", {})
    values.setdefault("reference_fps", 60.0)
    values.setdefault("fps_mode", "auto")
    values.setdefault("f_scale", "auto")
    values.setdefault("included_subjects", ())
    values.update(
        triangulation_method=method.get("name"),
        method_options=dict(method.get("options", {})),
        sequence_refinement=dict(refinement),
    )
    valid_priors = ("h36m", "sequence_median")
    if values["bone_prior"] not in valid_priors:
        raise ValueError(f"Invalid bone_prior '{values['bone_prior']}'. Supported: {', '.join(valid_priors)}")
    required = {item.name for item in fields(BenchmarkConfig)}

    missing = {name for name in required if name not in values or values[name] is None}
    unknown = values.keys() - required
    if missing or unknown:
        raise ValueError(f"Invalid pipeline: missing={sorted(missing)}, unknown={sorted(unknown)}")
    method_label(values["triangulation_method"])
    values["excluded_subjects"] = tuple(values["excluded_subjects"])
    values["included_subjects"] = tuple(values["included_subjects"])
    values["excluded_motions"] = frozenset(values["excluded_motions"])
    values["subject_gt_joint_markers"] = {
        subject: {int(joint): list(markers) for joint, markers in mapping.items()}
        for subject, mapping in values["subject_gt_joint_markers"].items()
    }
    expected_joints = set(range(17))
    for subject, mapping in values["subject_gt_joint_markers"].items():
        if set(mapping) != expected_joints or any(not names for names in mapping.values()):
            raise ValueError(f"pipeline.subject_gt_joint_markers.{subject} must define marker names for joints 0–16")
    return BenchmarkConfig(**values)


def _validate(inputs, config):
    if config.max_pairs <= 0 or config.chunk_size <= 0 or inputs.pose_cache_limit <= 0:
        raise ValueError("max_pairs, chunk_size, and pose_cache_limit must be greater than zero")
    confidences = (config.minimum_joint_confidence, config.sync_minimum_joint_confidence)
    if config.frames_per_pair < 0 or any(not 0 <= value <= 1 for value in confidences):
        raise ValueError("frames_per_pair or confidence thresholds are invalid")
    h = config.subject_height_mm
    if isinstance(h, (int, float)) and h <= 0:
        raise ValueError("subject_height_mm must be greater than zero")
    if (
        isinstance(config.bone_prior_min_samples, bool)
        or not isinstance(config.bone_prior_min_samples, int)
        or config.bone_prior_min_samples <= 0
    ):
        raise ValueError("bone_prior_min_samples must be greater than zero")
    if (
        isinstance(config.bone_prior_min_confidence, bool)
        or not isinstance(config.bone_prior_min_confidence, (int, float))
        or not math.isfinite(config.bone_prior_min_confidence)
        or not 0 <= config.bone_prior_min_confidence <= 1
    ):
        raise ValueError("bone_prior_min_confidence must be between zero and one")
    required_refinement = {
        "enabled", "data_weight", "root_weight", "bone_weight", "reprojection_weight",
        "smoothness_weight", "symmetry_weight", "max_evaluations", "max_drift",
    }
    if required_refinement != config.sequence_refinement.keys():
        raise ValueError("pipeline.sequence_refinement has missing or unknown options")


def load_settings(config_path: str | Path) -> tuple[InputSettings, OutputSettings, BenchmarkConfig]:
    path = Path(config_path)
    with path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    inputs, output, config = _load_input(path, raw), _load_output(path, raw), _load_benchmark(raw)
    _validate(inputs, config)
    return inputs, output, config
