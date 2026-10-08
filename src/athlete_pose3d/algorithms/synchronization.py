from __future__ import annotations

import csv
import os
from itertools import permutations
from pathlib import Path

import numpy as np

from ..io.data import PoseRepository, load_json
from .geometry import (
    H36M_BONES,
    H36M_SYMMETRIC_BONES,
    h36m_bone_lengths_from_height,
    point_to_ray_distance,
    triangulate_conf_algebraic,
)
from .uncalibrated import uncalibrated_sync_score, uncalibrated_triangulation


def robust_median(values, default=np.inf) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    return float(np.median(finite)) if len(finite) else float(default)


def _valid_joint_mask(points1, points2, confidence1, confidence2, minimum_confidence):
    return (np.isfinite(points1).all(axis=1) & np.isfinite(points2).all(axis=1)
            & (confidence1 > minimum_confidence) & (confidence2 > minimum_confidence))


def _reprojection_errors(points_3d, valid, p1, p2, points1, points2):
    return [
        error for index, point in enumerate(points_3d) if valid[index]
        for error in (point_to_ray_distance(point, p1, points1[index]),
                      point_to_ray_distance(point, p2, points2[index]))
    ]


def _bone_errors(points_3d, valid, bone_lengths):
    return [
        abs(np.linalg.norm(points_3d[a] - points_3d[b]) - bone_lengths[(a, b)]) / bone_lengths[(a, b)]
        for a, b in H36M_BONES if valid[a] and valid[b] and bone_lengths.get((a, b), 0) > 0
    ]


def _symmetry_errors(points_3d, valid):
    errors = []
    for (left1, left2), (right1, right2) in H36M_SYMMETRIC_BONES:
        if all(valid[joint] for joint in (left1, left2, right1, right2)):
            left = np.linalg.norm(points_3d[left1] - points_3d[left2])
            right = np.linalg.norm(points_3d[right1] - points_3d[right2])
            errors.append(abs(left - right) / (0.5 * (left + right) + 1e-6))
    return errors


def plausibility_score(
    p1, p2, points1, points2, confidence1, confidence2,
    bone_lengths, minimum_confidence, minimum_valid_joints,
) -> float:
    valid = _valid_joint_mask(points1, points2, confidence1, confidence2, minimum_confidence)
    if valid.sum() < minimum_valid_joints:
        return np.inf
    try:
        points_3d = triangulate_conf_algebraic(p1, p2, points1, points2, confidence1, confidence2)
    except (ValueError, np.linalg.LinAlgError):
        return np.inf
    if not np.isfinite(points_3d).all():
        return np.inf

    reprojections = _reprojection_errors(points_3d, valid, p1, p2, points1, points2)
    bone_errors = _bone_errors(points_3d, valid, bone_lengths)
    symmetry_errors = _symmetry_errors(points_3d, valid)
    mean_confidence = float(np.mean(np.minimum(confidence1[valid], confidence2[valid])))
    return (
        robust_median(reprojections, 120.0) / 60.0
        + 320.0 * robust_median(bone_errors, 4.0)
        + 120.0 * robust_median(symmetry_errors, 2.0)
        + 60.0 * (1.0 - mean_confidence)
    )


def estimate_pair_time_offset(
    repository: PoseRepository,
    video_a,
    video_b,
    p1,
    p2,
    frame_count_a,
    frame_count_b,
    score_options,
    max_offset=45,
    sample_count=7,
    coarse_step=5,
):
    lower = min(max_offset + 2, max(0, frame_count_a // 4))
    upper = max(lower + 1, min(frame_count_a - lower - 1, frame_count_b - 1))
    frames = np.linspace(lower, upper, min(sample_count, max(1, upper - lower + 1)), dtype=int)

    def score(delta):
        scores = []
        for frame in frames:
            other_frame = int(frame + delta)
            if not 0 <= other_frame < frame_count_b:
                continue
            left = repository.pose(video_a, int(frame))
            right = repository.pose(video_b, other_frame)
            if left and right:
                scores.append(
                    plausibility_score(
                        p1, p2, left["kps_h36m"], right["kps_h36m"],
                        left["conf_h36m"], right["conf_h36m"], **score_options,
                    )
                )
        return robust_median(scores)

    coarse_scores = {offset: score(offset) for offset in range(-max_offset, max_offset + 1, coarse_step)}
    best_coarse = min(coarse_scores, key=coarse_scores.get)
    refine_range = range(max(-max_offset, best_coarse - coarse_step), min(max_offset, best_coarse + coarse_step) + 1)
    refined_scores = {offset: score(offset) for offset in refine_range}
    best_offset = min(refined_scores, key=refined_scores.get)
    return int(best_offset), float(refined_scores[best_offset])


def _validation_frames(frame_count_a, frame_count_b, offset, samples, radius):
    margin = min(abs(int(offset)) + radius + 2, max(0, frame_count_a // 4))
    lower = margin
    upper = max(lower + 1, min(frame_count_a - margin - 1, frame_count_b - 1))
    count = min(samples, max(1, upper - lower + 1))
    return np.linspace(lower, upper, count, dtype=int)


def _best_validation_match(
    repository, video_b, p1, p2, left, frame, frame_count_b, offset, radius, score_options,
):
    best_score, best_confidence = np.inf, 0.0
    for delta in range(int(offset) - radius, int(offset) + radius + 1):
        other_frame = int(frame) + delta
        right = repository.pose(video_b, other_frame) if 0 <= other_frame < frame_count_b else None
        if right is None:
            continue
        score = plausibility_score(
            p1, p2, left["kps_h36m"], right["kps_h36m"],
            left["conf_h36m"], right["conf_h36m"], **score_options,
        )
        if np.isfinite(score) and score < best_score:
            threshold = score_options["minimum_confidence"]
            valid = (left["conf_h36m"] > threshold) & (right["conf_h36m"] > threshold)
            confidence = np.minimum(left["conf_h36m"][valid], right["conf_h36m"][valid])
            best_score = float(score)
            best_confidence = float(np.mean(confidence)) if valid.any() else 0.0
    return best_score, best_confidence


def validate_candidate(
    repository, video_a, video_b, p1, p2, frame_count_a, frame_count_b,
    offset, score_options, samples=5, radius=2,
):
    frames = _validation_frames(frame_count_a, frame_count_b, offset, samples, radius)
    best_scores, best_confidences = [], []
    for frame in frames:
        left = repository.pose(video_a, int(frame))
        if left is None:
            continue
        score, confidence = _best_validation_match(
            repository, video_b, p1, p2, left, frame, frame_count_b, offset, radius, score_options,
        )
        if np.isfinite(score):
            best_scores.append(score)
            best_confidences.append(confidence)
    if not best_scores:
        return {"validation_score": np.inf, "validation_iqr": np.inf, "valid_ratio": 0.0, "mean_conf": 0.0}
    scores = np.asarray(best_scores)
    return {
        "validation_score": float(np.median(scores)),
        "validation_iqr": float(np.percentile(scores, 75) - np.percentile(scores, 25)) if len(scores) >= 2 else 0.0,
        "valid_ratio": len(best_scores) / max(1, len(frames)),
        "mean_conf": float(np.mean(best_confidences)),
    }


def _candidate_pairs(pair_info, repository):
    cameras = sorted(
        (camera for camera in pair_info["cameras"] if repository.has_pose(camera["video_path"])),
        key=lambda camera: int(camera["cam_id"]) if camera.get("cam_id", "").isdigit() else camera.get("cam_id", ""),
    )
    return list(permutations(cameras, 2))


def _frame_count_for_camera(camera, repository: PoseRepository | None = None) -> int | None:
    if repository is not None:
        try:
            pose_path = repository.pose_path(camera.get("video_path", ""))
            if pose_path.exists():
                arr = np.load(pose_path, mmap_mode="r", allow_pickle=False)
                return len(arr)
        except (OSError, ValueError):
            pass

    gt_path = camera.get("ground_truth_path")
    if gt_path and os.path.exists(gt_path):
        try:
            arr = np.load(gt_path, mmap_mode="r", allow_pickle=False)
            return len(arr)
        except (OSError, ValueError):
            pass

    meta_path = camera.get("metadata_path")
    if meta_path and os.path.exists(meta_path):
        try:
            meta = load_json(meta_path)
            for k in ("num_frames", "total_frames", "frame_count", "n_frames"):
                if k in meta:
                    return int(meta[k])
        except (OSError, ValueError, TypeError):
            pass

    return None


def _pair_frame_counts(camera_a, camera_b, repository: PoseRepository | None = None):
    count_a = _frame_count_for_camera(camera_a, repository)
    count_b = _frame_count_for_camera(camera_b, repository)
    if count_a is None or count_b is None:
        return None
    counts = (count_a, count_b)
    return counts if min(counts) >= 2 else None


def _total_pair_score(validation, offset):
    return (validation["validation_score"] + 0.35 * validation["validation_iqr"]
            + 70.0 * (1.0 - validation["valid_ratio"])
            + 38.0 * max(0.0, 0.82 - validation["mean_conf"])
            + 0.08 * abs(offset))


def _recover_uncalibrated_p1_p2(repository, video_a, video_b, frames, best_offset, frame_count_b, bone_lengths):
    points_a, points_b, confidence_a, confidence_b = [], [], [], []
    for frame in frames:
        other = int(frame + best_offset)
        left = repository.pose(video_a, int(frame))
        right = repository.pose(video_b, other) if 0 <= other < frame_count_b else None
        if left and right:
            valid = (left["conf_h36m"] > 0.3) & (right["conf_h36m"] > 0.3)
            if valid.sum() >= 8:
                points_a.append(left["kps_h36m"])
                points_b.append(right["kps_h36m"])
                confidence_a.append(left["conf_h36m"])
                confidence_b.append(right["conf_h36m"])
    if not points_a:
        return None, None
    try:
        _, p1, p2 = uncalibrated_triangulation(
            np.asarray(points_a), np.asarray(points_b), np.asarray(confidence_a), np.asarray(confidence_b), bone_lengths
        )
        return p1, p2
    except (ValueError, np.linalg.LinAlgError):
        return None, None


def estimate_pair_time_offset_uncalibrated(
    repository: PoseRepository, video_a, video_b, frame_count_a, frame_count_b,
    bone_lengths, max_offset=45, sample_count=7, coarse_step=5, min_confidence=0.25, min_valid=8,
):
    lower = min(max_offset + 2, max(0, frame_count_a // 4))
    upper = max(lower + 1, min(frame_count_a - lower - 1, frame_count_b - 1))
    frames = np.linspace(lower, upper, min(sample_count, max(1, upper - lower + 1)), dtype=int)

    def score(delta):
        points_a, points_b, confidence_a, confidence_b = [], [], [], []
        for frame in frames:
            other = int(frame + delta)
            if not 0 <= other < frame_count_b:
                continue
            left, right = repository.pose(video_a, int(frame)), repository.pose(video_b, other)
            if left and right:
                valid = (left["conf_h36m"] > min_confidence) & (right["conf_h36m"] > min_confidence)
                if valid.sum() >= min_valid:
                    points_a.append(left["kps_h36m"])
                    points_b.append(right["kps_h36m"])
                    confidence_a.append(left["conf_h36m"])
                    confidence_b.append(right["conf_h36m"])
        if not points_a:
            return np.inf
        return uncalibrated_sync_score(
            np.asarray(points_a), np.asarray(points_b), np.asarray(confidence_a), np.asarray(confidence_b),
            bone_lengths, min_confidence=min_confidence, min_valid=min_valid,
        )

    coarse_scores = {offset: score(offset) for offset in range(-max_offset, max_offset + 1, coarse_step)}
    best_coarse = min(coarse_scores, key=coarse_scores.get)
    refine_range = range(max(-max_offset, best_coarse - coarse_step), min(max_offset, best_coarse + coarse_step) + 1)
    refined_scores = {offset: score(offset) for offset in refine_range}
    best_offset, best_score = min(refined_scores.items(), key=lambda kv: kv[1])
    if not np.isfinite(best_score):
        return int(best_offset), np.inf, None, None
    p1, p2 = _recover_uncalibrated_p1_p2(
        repository, video_a, video_b, frames, best_offset, frame_count_b, bone_lengths
    )
    return int(best_offset), float(best_score), p1, p2


def _pair_offset_and_p1_p2(
    repository, video_a, video_b, frame_count_a, frame_count_b,
    max_offset, sync_samples, coarse_step, score_options,
):
    bone_lens = score_options.get("bone_lengths") or h36m_bone_lengths_from_height()
    return estimate_pair_time_offset_uncalibrated(
        repository, video_a, video_b, frame_count_a, frame_count_b, bone_lens,
        max_offset=max_offset, sample_count=sync_samples, coarse_step=coarse_step,
        min_confidence=score_options.get("minimum_confidence", 0.25),
        min_valid=score_options.get("minimum_valid_joints", 8),
    )


def _evaluate_pair(
    pair, repository, valid_videos, max_offset, sync_samples, coarse_step,
    minimum_valid_ratio, validation_radius, score_options,
):
    camera_a, camera_b = pair
    names = [f"{Path(camera['path']).parent.name}/{Path(camera['path']).name}" for camera in (camera_a, camera_b)]
    if valid_videos is not None and any(
        name not in valid_videos and Path(camera["path"]).name not in valid_videos
        for name, camera in zip(names, (camera_a, camera_b))
    ):
        return None
    counts = _pair_frame_counts(camera_a, camera_b, repository=repository)
    if counts is None:
        return None
    frame_count_a, frame_count_b = counts
    video_a, video_b = camera_a["video_path"], camera_b["video_path"]
    offset, sync_score, p1, p2 = _pair_offset_and_p1_p2(
        repository, video_a, video_b, frame_count_a, frame_count_b,
        max_offset, sync_samples, coarse_step, score_options,
    )
    if not np.isfinite(sync_score) or p1 is None or p2 is None:
        return None
    validation = validate_candidate(
        repository, video_a, video_b, p1, p2, frame_count_a, frame_count_b, offset,
        score_options, samples=max(4, sync_samples), radius=validation_radius,
    )
    if not np.isfinite(validation["validation_score"]) or validation["valid_ratio"] < minimum_valid_ratio:
        return None
    return {
        "cam_a": camera_a, "cam_b": camera_b, "P1": p1, "P2": p2,
        "n_a": frame_count_a, "n_b": frame_count_b, "delta": offset,
        "sync_score": sync_score, "score": float(_total_pair_score(validation, offset)),
        **validation,
    }


def _record_evaluated_pairs_csv(pair_info, evaluated_all, best, csv_path):
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    sorted_pairs = sorted(evaluated_all, key=lambda c: c.get("score", float("inf")))
    best_ids = (best["cam_a"]["cam_id"], best["cam_b"]["cam_id"]) if best else (None, None)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Subject", "Motion", "Rank", "Cam_A", "Cam_B", "Is_Best",
                "Pair_Score", "Sync_Delta", "Sync_Score", "Valid_Ratio",
                "Validation_Score", "Validation_IQR", "Mean_Confidence",
            ])
        for rank, c in enumerate(sorted_pairs, start=1):
            cam_a_id = c["cam_a"].get("cam_id")
            cam_b_id = c["cam_b"].get("cam_id")
            is_best = (cam_a_id, cam_b_id) == best_ids
            score_val = c.get("score", float("inf"))
            sync_val = c.get("sync_score", float("inf"))
            val_score = c.get("validation_score", float("inf"))
            val_iqr = c.get("validation_iqr", float("inf"))
            writer.writerow([
                pair_info.get("subject"),
                pair_info.get("motion"),
                rank,
                cam_a_id,
                cam_b_id,
                is_best,
                f"{score_val:.3f}" if np.isfinite(score_val) else "inf",
                c.get("delta") if c.get("delta") is not None else "",
                f"{sync_val:.3f}" if np.isfinite(sync_val) else "inf",
                f"{c.get('valid_ratio', 0.0):.3f}",
                f"{val_score:.3f}" if np.isfinite(val_score) else "inf",
                f"{val_iqr:.3f}" if np.isfinite(val_iqr) else "inf",
                f"{c.get('mean_conf', 0.0):.3f}",
            ])


def select_best_pair(
    pair_info, repository: PoseRepository, score_options, valid_videos=None, max_candidates=None,
    max_offset=45, sync_samples=5, coarse_step=7, minimum_valid_ratio=0.5,
    validation_radius=2, record_all_csv: str | Path | None = "outputs/all_camera_pairs.csv",
):
    best = None
    entries = _candidate_pairs(pair_info, repository)
    if max_candidates is not None and max_candidates > 0:
        entries = entries[:max_candidates]
    evaluated_all = []
    for entry in entries:
        candidate = _evaluate_pair(
            entry, repository, valid_videos, max_offset, sync_samples, coarse_step,
            minimum_valid_ratio, validation_radius, score_options,
        )
        if candidate is not None:
            evaluated_all.append(candidate)
            if best is None or candidate["score"] < best["score"]:
                best = candidate
        else:
            evaluated_all.append({
                "cam_a": entry[0], "cam_b": entry[1], "score": float("inf"),
                "delta": None, "sync_score": float("inf"), "valid_ratio": 0.0,
                "validation_score": float("inf"), "validation_iqr": float("inf"),
                "mean_conf": 0.0,
            })
    if record_all_csv:
        _record_evaluated_pairs_csv(pair_info, evaluated_all, best, record_all_csv)
    return best


def _dynamic_cost_rows(
    repository, video_a, video_b, p1, p2, frames, offsets, frame_count_b, score_options,
):
    costs = []
    for frame in frames:
        left = repository.pose(video_a, int(frame))
        row = []
        for offset in offsets:
            other_frame = int(frame) + offset
            right = repository.pose(video_b, other_frame) if 0 <= other_frame < frame_count_b else None
            score = (
                plausibility_score(
                    p1, p2, left["kps_h36m"], right["kps_h36m"],
                    left["conf_h36m"], right["conf_h36m"], **score_options,
                )
                if left and right else np.inf
            )
            row.append(score)
        costs.append(np.asarray(row))
    return costs


def _fill_invalid_costs(cost_matrix, offsets, base_offset, transition_weight):
    finite = cost_matrix[np.isfinite(cost_matrix)]
    fallback = float(np.median(finite)) if len(finite) else 1e6
    offset_array = np.asarray(offsets)
    for row in cost_matrix:
        if not np.isfinite(row).any():
            row[:] = fallback + transition_weight * np.abs(offset_array - int(base_offset))
        else:
            row[~np.isfinite(row)] = np.max(row[np.isfinite(row)]) + 100.0
    return offset_array


def _optimal_offset_path(cost_matrix, offset_array, transition_weight):
    dynamic = np.zeros_like(cost_matrix)
    backtrack = np.zeros(cost_matrix.shape, dtype=int)
    dynamic[0] = cost_matrix[0]
    for time in range(1, cost_matrix.shape[0]):
        offset_change = np.abs(offset_array[:, None] - offset_array[None, :])
        transition = dynamic[time - 1][:, None] + transition_weight * offset_change
        backtrack[time] = np.argmin(transition, axis=0)
        dynamic[time] = cost_matrix[time] + np.min(transition, axis=0)
    index = int(np.argmin(dynamic[-1]))
    path = [index]
    for time in range(cost_matrix.shape[0] - 1, 0, -1):
        index = int(backtrack[time, index])
        path.append(index)
    path.reverse()
    return path


def estimate_dynamic_offsets(
    repository, video_a, video_b, p1, p2, frames, base_offset,
    frame_count_b, score_options, radius=8, transition_weight=7.5,
):
    offsets = list(range(int(base_offset) - radius, int(base_offset) + radius + 1))
    if not frames or not offsets:
        return {}, {}
    costs = _dynamic_cost_rows(
        repository, video_a, video_b, p1, p2, frames, offsets, frame_count_b, score_options,
    )
    cost_matrix = np.vstack(costs)
    if not np.isfinite(cost_matrix).any():
        return {frame: int(base_offset) for frame in frames}, {frame: np.inf for frame in frames}
    offset_array = _fill_invalid_costs(cost_matrix, offsets, base_offset, transition_weight)
    path = _optimal_offset_path(cost_matrix, offset_array, transition_weight)
    return (
        {frame: int(offsets[index]) for frame, index in zip(frames, path)},
        {frame: float(costs[time][path[time]]) for time, frame in enumerate(frames)},
    )


def select_synced_pose(
    repository, video_a, video_b, p1, p2, frame, base_offset, frame_count_b,
    score_options, radius=1,
):
    left = repository.pose(video_a, int(frame))
    if left is None:
        return None, None, None, np.inf
    best, best_score, best_offset = None, np.inf, int(base_offset)
    for offset in range(int(base_offset) - radius, int(base_offset) + radius + 1):
        other_frame = int(frame) + offset
        right = repository.pose(video_b, other_frame) if 0 <= other_frame < frame_count_b else None
        if right:
            score = plausibility_score(
                p1, p2, left["kps_h36m"], right["kps_h36m"],
                left["conf_h36m"], right["conf_h36m"], **score_options,
            )
            if score < best_score:
                best, best_score, best_offset = right, score, offset
    return left, best, best_offset, best_score
