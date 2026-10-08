"""Dataset discovery and cached access to pose data."""

from __future__ import annotations

import json
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def load_json(path: str | Path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def load_valid_videos(manifest_path: str | Path, minimum_confidence: float) -> set[str] | None:
    path = Path(manifest_path)
    if not path.exists():
        print(f"Manifest not found: {path}. Manifest filtering is disabled.")
        return None
    frame = pd.read_csv(path)
    video_column = "video" if "video" in frame.columns else "video_name"
    valid = frame.loc[frame["mean_conf"] >= minimum_confidence, video_column]
    result = {Path(str(name)).with_suffix("").as_posix() for name in valid}
    result.update(Path(str(name)).stem for name in valid)
    print(f"Manifest accepted {len(valid)}/{len(frame)} videos with mean_conf >= {minimum_confidence}")
    return result


def _scan_motion_videos(videos, video_extension, metadata_extension, ground_truth_extension):
    motions = defaultdict(list)
    for video in sorted(videos):
        parts = video.stem.rsplit("_cam_", 1)
        if len(parts) != 2:
            continue
        motions[parts[0]].append(
            {
                "cam_id": parts[1],
                "base": video.stem,
                "path": str(video.with_suffix("")),
                "video_path": str(video),
                "metadata_path": str(video.with_suffix(metadata_extension)),
                "ground_truth_path": str(video.with_suffix(ground_truth_extension)),
            }
        )
    return dict(motions)


def scan_local_dataset(
    physical_base_path: str | Path,
    video_extension: str,
    metadata_extension: str,
    ground_truth_extension: str,
) -> dict:
    """Build the same dataset index directly from a local test-set directory."""
    root = Path(physical_base_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Local test-set directory does not exist: {root}")

    subjects = {}
    for subject_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        motions = _scan_motion_videos(
            subject_dir.glob(f"*_cam_*{video_extension}"),
            video_extension, metadata_extension, ground_truth_extension,
        )
        if motions:
            subjects[subject_dir.name] = motions

    if not subjects:
        motions = _scan_motion_videos(
            root.glob(f"*_cam_*{video_extension}"),
            video_extension, metadata_extension, ground_truth_extension,
        )
        if motions:
            subject_name = root.name if root.name not in ("data", "inputs") else "S3"
            subjects[subject_name] = motions

    return {"test_set": subjects}


def find_multicam_pairs(dataset_info: dict) -> list[dict]:
    return [
        {"split": split, "subject": subject, "motion": motion, "cameras": cameras, "n_cams": len(cameras)}
        for split, subjects in dataset_info.items()
        for subject, motions in subjects.items()
        for motion, cameras in motions.items()
        if len(cameras) >= 2
    ]


def extract_gt_3d(
    raw_data: np.ndarray,
    marker_names: list[str],
    frame_index: int,
    joint_markers: dict[int, list[str]],
) -> np.ndarray | None:
    if frame_index >= len(raw_data):
        return None
    frame = raw_data[frame_index]
    marker_indices = {name: index for index, name in enumerate(marker_names)}
    pose = np.zeros((17, 3))
    missing = []
    for joint, candidates in joint_markers.items():
        indices = [marker_indices[name] for name in candidates if name in marker_indices]
        if indices:
            pose[joint] = np.mean(frame[indices], axis=0)
        else:
            missing.append(joint)
    if missing:
        raise ValueError(f"GT markers missing for H36M joints {missing}")
    return pose


COCO_TO_H36M_JOINTS = (
    (1, 12), (2, 14), (3, 16),
    (4, 11), (5, 13), (6, 15),
    (11, 5), (12, 7), (13, 9),
    (14, 6), (15, 8), (16, 10),
)


def coco_to_h36m_2d(kps_coco: np.ndarray, conf_coco: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h36m = np.zeros((17, 2), dtype=float)
    h36m_conf = np.zeros(17, dtype=float)
    for h_idx, c_idx in COCO_TO_H36M_JOINTS:
        h36m[h_idx] = kps_coco[c_idx]
        h36m_conf[h_idx] = conf_coco[c_idx]
    h36m[0] = 0.5 * (h36m[1] + h36m[4])
    h36m_conf[0] = min(h36m_conf[1], h36m_conf[4])
    h36m[8] = 0.5 * (h36m[11] + h36m[14])
    h36m_conf[8] = min(h36m_conf[11], h36m_conf[14])
    h36m[7] = 0.5 * (h36m[0] + h36m[8])
    h36m_conf[7] = min(h36m_conf[0], h36m_conf[8])
    # ponytail: extrapolate head vertex (10) and neck (9) from thorax (8) -> nose vector
    v = kps_coco[0] - h36m[8]
    if np.linalg.norm(v) > 1e-4:
        h36m[10] = h36m[8] + 1.45 * v
        h36m[9] = h36m[8] + 0.40 * (h36m[10] - h36m[8])
    else:
        h36m[10] = kps_coco[0]
        h36m[9] = 0.5 * (h36m[8] + h36m[10])
    h36m_conf[10] = conf_coco[0]
    h36m_conf[9] = min(h36m_conf[8], h36m_conf[10])
    return h36m, h36m_conf



class PoseRepository:
    """Find and lazily cache precomputed H36M poses and key-frame indices."""

    def __init__(self, root: str | Path, pose_suffix: str, key_frames_suffix: str, cache_limit: int):
        self.root = Path(root)
        self.pose_suffix = pose_suffix
        self.key_frames_suffix = key_frames_suffix
        self.cache_limit = cache_limit
        self._arrays: dict[str, np.ndarray] = {}
        self._frames: OrderedDict[tuple[str, int], dict | None] = OrderedDict()

    @staticmethod
    def _video_parts(video_path: str | Path) -> tuple[str, str | None, str]:
        path = Path(video_path)
        subject = path.parent.name
        split = None
        parts = path.parts
        for candidate in ("train_set", "valid_set", "test_set"):
            if candidate in parts:
                index = parts.index(candidate)
                split = candidate
                if index + 1 < len(parts):
                    subject = parts[index + 1]
                break
        return subject, split, path.stem

    def _candidates(self, video_path: str | Path, filename: str) -> list[Path]:
        subject, split, _ = self._video_parts(video_path)
        candidates = [
            self.root / subject / filename,
            self.root / "data" / subject / filename,
            self.root / filename,
            self.root / "data" / filename,
        ]
        if split:
            candidates.insert(1, self.root / split / subject / filename)
            candidates.insert(2, self.root / "data" / split / subject / filename)
        return candidates

    @staticmethod
    def _first_existing(candidates: list[Path]) -> Path:
        return next((path for path in candidates if path.exists()), candidates[0])

    def pose_path(self, video_path: str | Path) -> Path:
        _, _, stem = self._video_parts(video_path)
        return self._first_existing(self._candidates(video_path, f"{stem}{self.pose_suffix}"))

    def key_frame_path(self, video_path: str | Path) -> Path:
        _, _, stem = self._video_parts(video_path)
        return self._first_existing(self._candidates(video_path, f"{stem}{self.key_frames_suffix}"))

    def has_pose(self, video_path: str | Path) -> bool:
        return self.pose_path(video_path).exists()

    def pose(self, video_path: str | Path, frame_index: int) -> dict | None:
        key = (str(video_path), int(frame_index))
        if key in self._frames:
            self._frames.move_to_end(key)
            return self._frames[key]

        pose_path = self.pose_path(video_path)
        is_coco = "_coco" in pose_path.name
        array = self._arrays.get(str(pose_path))
        if array is None:
            if not pose_path.exists():
                result = None
            else:
                try:
                    array = np.load(pose_path, mmap_mode="r")
                    self._arrays[str(pose_path)] = array
                    result = self._pose_from_array(array, frame_index, is_coco=is_coco)
                except (OSError, ValueError) as error:
                    print(f"Failed to read Pose2D {pose_path}: {error}")
                    result = None
        else:
            result = self._pose_from_array(array, frame_index, is_coco=is_coco)

        self._frames[key] = result
        while len(self._frames) > self.cache_limit:
            self._frames.popitem(last=False)
        return result

    @classmethod
    def _pose_from_array(cls, array: np.ndarray, frame_index: int, is_coco: bool = False) -> dict | None:
        if frame_index < 0 or frame_index >= len(array):
            return None
        pose = np.asarray(array[frame_index], dtype=np.float32)
        if pose.ndim != 2 or pose.shape[0] < 17:
            return None
        kps = pose[:17, :2].copy()
        if pose.shape[1] >= 3:
            conf = np.clip(pose[:17, 2], 0.0, 1.0)
        else:
            conf = np.where(np.isfinite(kps).all(axis=-1), 1.0, 0.0).astype(np.float32)
        kps = np.nan_to_num(kps, nan=0.0)
        if is_coco:
            h36m_kps, h36m_conf = coco_to_h36m_2d(kps, conf)
            return {
                "frame": None, "kps_coco": kps, "conf_coco": conf,
                "kps_h36m": h36m_kps, "conf_h36m": h36m_conf,
            }
        return {
            "frame": None, "kps_coco": kps.copy(), "conf_coco": conf.copy(),
            "kps_h36m": kps, "conf_h36m": conf,
        }

    def key_frames(self, video_path: str | Path, frame_count: int) -> list[int]:
        path = self.key_frame_path(video_path)
        if not path.exists():
            return list(range(frame_count))
        try:
            values = np.asarray(np.load(path, allow_pickle=False)).reshape(-1)
            if not np.issubdtype(values.dtype, np.number):
                return list(range(frame_count))
            values = values[np.isfinite(values)].astype(np.int64)
            valid = np.unique(values[(values >= 0) & (values < frame_count)]).tolist()
            return valid if valid else list(range(frame_count))
        except (OSError, ValueError) as error:
            print(f"Failed to read key-frame file {path}: {error}")
            return list(range(frame_count))
