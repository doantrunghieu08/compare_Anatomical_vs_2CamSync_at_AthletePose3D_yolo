from __future__ import annotations

from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .algorithms.geometry import (
    H36M_BONES,
    H36M_SYMMETRIC_BONES,
    estimate_bone_lengths_from_poses,
    h36m_bone_lengths_from_height,
    mpjpe,
    pa_mpjpe,
    evaluate_joint_groups,
    resolve_subject_height,
    triangulate_anatomical,
    triangulate_conf_algebraic,
    triangulate_dlt,
    triangulate_dst_anatomical,
    triangulate_iterative,
    triangulate_ransac,
)

from .algorithms.evidence_fusion import detect_stereo_occlusions
from .algorithms.physics_refine import triangulate_dst_physics, triangulate_physics_refine
from .algorithms.refinement import refine_results
from .algorithms.synchronization import estimate_dynamic_offsets, select_best_pair, select_synced_pose
from .algorithms.uncalibrated import uncalibrated_triangulation
from .io.data import PoseRepository, extract_gt_3d, load_json
from .settings import BenchmarkConfig, method_label



def _sample_key_frames(frames: list[int], limit: int) -> list[int]:
    if limit <= 0 or len(frames) <= limit:
        return frames
    return [frames[index] for index in np.linspace(0, len(frames) - 1, limit, dtype=int)]


def _sync_score_options(config, subject=None):
    height = resolve_subject_height(subject or "", config.subject_height_mm, config.subject_heights)
    return {
        "bone_lengths": h36m_bone_lengths_from_height(height),
        "minimum_confidence": config.sync_minimum_joint_confidence,
        "minimum_valid_joints": config.sync_minimum_valid_joints,
    }


def _make_triangulation_function(method, options, config, bone_lens=None):
    if bone_lens is None:
        bone_lens = h36m_bone_lengths_from_height(config.subject_height_mm)
    opts = dict(options)
    if method == "uncalibrated_metric":
        return lambda p1, p2, k1, k2, c1, c2: uncalibrated_triangulation(k1, k2, c1, c2, bone_lens)[0]
    if method == "dst_physics":
        bone_w = opts.pop("bone_weight", 1.0)
        return partial(triangulate_dst_physics, bone_lengths=bone_lens, bone_weight=bone_w, **opts)
    if method == "dst_anatomical":
        bone_w = opts.pop("bone_weight", 1.0)
        return partial(triangulate_dst_anatomical, bone_lengths=bone_lens, bone_weight=bone_w, **opts)
    if method == "physics_refine":
        bone_w = opts.pop("bone_weight", 1.0)
        return partial(triangulate_physics_refine, bone_lengths=bone_lens, bone_weight=bone_w, **opts)
    if method == "anatomical":
        return partial(triangulate_anatomical, bone_lengths=bone_lens, bone_weight=opts.pop("bone_weight"), **opts)
    simple = {
        "dlt": triangulate_dlt,
        "confidence_algebraic": triangulate_conf_algebraic,
        "ransac_dlt": partial(triangulate_ransac, **opts),
        "iterative_refine": partial(triangulate_iterative, **opts),
    }
    return simple[method]


def _validate_method_options(method: str, options: dict) -> None:
    for k in ("bone_weight", "data_weight", "anchor_weight", "sym_weight", "lr", "ground_z", "delta_ray_mm"):
        if k in options:
            val = options[k]
            if k == "ground_z" and val is None:
                continue
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not np.isfinite(val) or val < 0:
                raise ValueError(f"Option '{k}' for {method} must be a non-negative finite number, got {val}")
            if k == "delta_ray_mm" and val == 0:
                raise ValueError("Option 'delta_ray_mm' must be greater than zero")
    if "iterations" in options:
        val = options["iterations"]
        if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
            raise ValueError(f"Option 'iterations' for {method} must be a positive integer, got {val}")
    if "bone_reliability_modulation" in options:
        val = options["bone_reliability_modulation"]
        if not isinstance(val, bool):
            raise ValueError(f"Option 'bone_reliability_modulation' must be a boolean, got {type(val).__name__}")
    if "use_nimble_ik" in options and not isinstance(options["use_nimble_ik"], bool):
        raise ValueError("Option 'use_nimble_ik' must be a boolean")
    if "fusion_sources" in options:
        sources = options["fusion_sources"]
        if (
            not isinstance(sources, (list, tuple))
            or any(not isinstance(source, str) for source in sources)
            or set(sources) - {"detector", "epipolar", "bone"}
        ):
            raise ValueError("Option 'fusion_sources' must contain detector, epipolar, and/or bone")
    epi_mode = options.get("fusion_epi_mode", "sampson")
    if not isinstance(epi_mode, str) or epi_mode not in {"sampson", "ray_mm"}:
        raise ValueError("Option 'fusion_epi_mode' must be 'sampson' or 'ray_mm'")
    fusion_rule = options.get("fusion_rule", "yager")
    if not isinstance(fusion_rule, str) or fusion_rule not in {"yager", "dempster"}:
        raise ValueError("Option 'fusion_rule' must be 'yager' or 'dempster'")
    bone_mode = options.get("fusion_bone_mode", "joint")
    if not isinstance(bone_mode, str) or bone_mode not in {"joint", "legacy"}:
        raise ValueError("Option 'fusion_bone_mode' must be 'joint' or 'legacy'")
    if "fusion_unknown_trust" in options:
        val = options["fusion_unknown_trust"]
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not 0 <= val <= 1:
            raise ValueError("Option 'fusion_unknown_trust' must be between zero and one")


def _selected_triangulation(config: BenchmarkConfig):
    method = config.triangulation_method
    options = dict(config.method_options)
    allowed_options = {
        "dlt": set(),
        "confidence_algebraic": set(),
        "ransac_dlt": {"reprojection_threshold"},
        "iterative_refine": {"iterations"},
        "anatomical": {"iterations", "bone_weight"},
        "dst_anatomical": {"iterations", "bone_weight", "conflict_threshold"},
        "physics_refine": {
            "iterations", "bone_weight", "ground_z", "lr", "use_nimble_ik",
            "data_weight", "anchor_weight", "sym_weight", "bone_reliability_modulation", "delta_ray_mm",
        },
        "dst_physics": {
            "iterations", "bone_weight", "ground_z", "lr", "use_nimble_ik",
            "data_weight", "anchor_weight", "sym_weight", "bone_reliability_modulation", "delta_ray_mm",
            "fusion_sources", "fusion_epi_mode", "fusion_rule", "fusion_unknown_trust", "fusion_bone_mode",
        },
        "uncalibrated_metric": {"iterations", "bone_weight"},
    }
    label = method_label(method)
    unknown = options.keys() - allowed_options[method]
    if unknown:
        raise ValueError(f"Unsupported options for {method}: {', '.join(sorted(unknown))}")
    _validate_method_options(method, options)
    return label, _make_triangulation_function(method, options, config)



def _resolve_context_height(pair_info, config):
    height = resolve_subject_height(pair_info["subject"], config.subject_height_mm, config.subject_heights)
    return height, h36m_bone_lengths_from_height(height)


def _estimate_uncalib_p1_p2(video_a, video_b, frames, delta, repository, bone_lengths, f_scale: float | str = "auto"):
    points_a, points_b, confidence_a, confidence_b = [], [], [], []
    sample_ids = np.linspace(0, len(frames) - 1, min(20, len(frames)), dtype=int) if frames else []
    for sample_index in sample_ids:
        f = frames[int(sample_index)]
        la = repository.pose(video_a, int(f))
        lb = repository.pose(video_b, int(f + delta))
        if la is not None and lb is not None:
            c1, c2 = la["conf_h36m"], lb["conf_h36m"]
            if (c1 > 0.3).sum() >= 8 and (c2 > 0.3).sum() >= 8:
                points_a.append(la["kps_h36m"])
                points_b.append(lb["kps_h36m"])
                confidence_a.append(c1)
                confidence_b.append(c2)
    if not points_a:
        return None, None
    try:
        _, p1, p2 = uncalibrated_triangulation(
            np.asarray(points_a), np.asarray(points_b), np.asarray(confidence_a), np.asarray(confidence_b),
            bone_lengths, f_scale=f_scale,
        )
        return p1, p2
    except (ValueError, np.linalg.LinAlgError):
        return None, None


def _harmonize_symmetric_bones(bone_lengths: dict[tuple[int, int], float]) -> dict[tuple[int, int], float]:
    harmonized = dict(bone_lengths)
    for (l1, l2), (r1, r2) in H36M_SYMMETRIC_BONES:
        bone_l = (l1, l2) if (l1, l2) in harmonized else (l2, l1)
        bone_r = (r1, r2) if (r1, r2) in harmonized else (r2, r1)
        if bone_l in harmonized and bone_r in harmonized:
            avg_len = 0.5 * (harmonized[bone_l] + harmonized[bone_r])
            harmonized[bone_l] = avg_len
            harmonized[bone_r] = avg_len
    return harmonized


def _fit_context_bone_prior(context, repository, config):
    if config.bone_prior == "h36m":
        return context["bone_lengths"]
    p1, p2, delta = context["p1"], context["p2"], context["best_pair"]["delta"]
    v_a, v_b = context["video_a"], context["video_b"]
    fit_poses, fit_confs = [], []
    for f in context["frames"]:
        la, lb = repository.pose(v_a, int(f)), repository.pose(v_b, int(f + delta))
        if la and lb:
            c_mean = 0.5 * (la["conf_h36m"] + lb["conf_h36m"])
            if float(np.mean(c_mean)) >= config.bone_prior_min_confidence:
                dlt_pt = triangulate_dlt(p1, p2, la["kps_h36m"], lb["kps_h36m"])
                if np.isfinite(dlt_pt).all():
                    fit_poses.append(dlt_pt)
                    fit_confs.append(c_mean)
    if fit_poses:
        prior, _ = estimate_bone_lengths_from_poses(
            fit_poses, height_mm=context["subject_height_mm"], confidences=fit_confs,
            min_samples=config.bone_prior_min_samples,
            min_confidence=config.bone_prior_min_confidence,
        )
        return _harmonize_symmetric_bones(prior)
    return context["bone_lengths"]


def _load_motion_context(pair_info, best_pair, repository, config):
    camera_a, camera_b = best_pair["cam_a"], best_pair["cam_b"]
    video_a, video_b = camera_a["video_path"], camera_b["video_path"]
    gt_path, metadata_path = Path(camera_a["ground_truth_path"]), Path(camera_a["metadata_path"])
    required = [Path(video_a), Path(video_b), metadata_path, Path(camera_b["metadata_path"]), gt_path]
    if not all(path.exists() for path in required):
        print(f"SKIP [/{pair_info['subject']}/{pair_info['motion']}]: required mp4/json/npy file is missing.")
        return None
    raw_gt = np.load(gt_path, allow_pickle=False)
    frame_count = min(best_pair["n_a"], best_pair["n_b"], len(raw_gt))
    frames = _sample_key_frames(repository.key_frames(video_a, frame_count), config.frames_per_pair)
    if not frames:
        print(f"SKIP [/{pair_info['subject']}/{pair_info['motion']}]: no valid key frame.")
        return None
    p1, p2 = best_pair["P1"], best_pair["P2"]
    marker_names = load_json(metadata_path)["keypoint_name"]
    joint_markers = config.subject_gt_joint_markers.get(pair_info["subject"])
    if joint_markers is None:
        raise ValueError(f"No GT joint marker mapping configured for subject {pair_info['subject']}")
    height, bone_lengths = _resolve_context_height(pair_info, config)
    if bone_lengths is not None:
        bone_lengths = _harmonize_symmetric_bones(bone_lengths)
    f_scale = getattr(config, "f_scale", "auto")
    u1, u2 = _estimate_uncalib_p1_p2(video_a, video_b, frames, best_pair["delta"], repository, bone_lengths, f_scale=f_scale)
    if u1 is not None and u2 is not None:
        p1, p2 = u1, u2
    ctx = {
        "pair_info": pair_info, "best_pair": best_pair, "camera_a": camera_a, "camera_b": camera_b,
        "video_a": video_a, "video_b": video_b, "raw_gt": raw_gt, "frames": frames,
        "marker_names": marker_names, "joint_markers": joint_markers, "p1": p1, "p2": p2,
        "subject_height_mm": height, "bone_lengths": bone_lengths,
    }
    ctx["bone_lengths"] = _fit_context_bone_prior(ctx, repository, config)
    return ctx



def _resolve_pair_fps(pair_info, config) -> float:
    if isinstance(config.fps_mode, (int, float)) and config.fps_mode > 0:
        return float(config.fps_mode)
    subj_fps = config.subject_fps.get(pair_info.get("subject", ""))
    if isinstance(subj_fps, (int, float)) and subj_fps > 0:
        return float(subj_fps)
    for cam in pair_info.get("cameras", []):
        meta_p = cam.get("metadata_path")
        if meta_p and Path(meta_p).is_file():
            try:
                val = float(load_json(meta_p).get("fps", 0))
                if val > 0:
                    return val
            except (ValueError, TypeError, OSError):
                pass
    return float(getattr(config, "reference_fps", 60.0))


def _fps_scaled_sync_params(pair_info, config):
    fps = _resolve_pair_fps(pair_info, config)
    scale = fps / max(1.0, float(getattr(config, "reference_fps", 60.0)))
    return {
        "fps": fps,
        "max_offset": int(round(config.sync_max_offset * scale)),
        "coarse_step": max(1, int(round(config.sync_coarse_step * scale))),
        "local_radius": max(1, int(round(config.sync_local_radius * scale))),
        "dynamic_radius": max(1, int(round(config.dynamic_local_radius * scale))),
        "transition_weight": config.dynamic_transition_weight * (1.0 / scale),
    }


def _add_dynamic_sync(context, repository, config):
    best_pair = context["best_pair"]
    dyn_r = context.get("dynamic_local_radius", config.dynamic_local_radius)
    trans_w = context.get("dynamic_transition_weight", config.dynamic_transition_weight)
    offsets, scores = estimate_dynamic_offsets(
        repository, context["video_a"], context["video_b"], context["p1"], context["p2"],
        context["frames"], best_pair["delta"], best_pair["n_b"], context["score_options"],
        radius=dyn_r, transition_weight=trans_w,
    )
    context.update(dynamic_offsets=offsets, dynamic_scores=scores)
    pair_info, camera_a, camera_b = context["pair_info"], context["camera_a"], context["camera_b"]
    fps = context.get("fps")
    fps_info = f", {fps:.1f} FPS" if fps else ""
    print(
        f"TwoCam {pair_info['motion']} ({pair_info['subject']}{fps_info}) "
        f"cam{camera_a['cam_id']}->cam{camera_b['cam_id']}: "
        f"global_delta={best_pair['delta']:+d}, score={best_pair['score']:.2f}"
    )


def _frame_inputs(context, frame, repository, config):
    dynamic_score = float(context["dynamic_scores"].get(frame, np.inf))
    if not np.isfinite(dynamic_score) or dynamic_score > config.max_dynamic_sync_score:
        return None
    best_pair = context["best_pair"]
    dynamic_offset = int(context["dynamic_offsets"].get(frame, best_pair["delta"]))
    local_r = context.get("sync_local_radius", config.sync_local_radius)
    left, right, local_offset, sync_score = select_synced_pose(
        repository, context["video_a"], context["video_b"], context["p1"], context["p2"],
        frame, dynamic_offset, best_pair["n_b"], context["score_options"], radius=local_r,
    )
    if left is None or right is None or not np.isfinite(sync_score):
        return None
    if sync_score > config.max_dynamic_sync_score:
        return None
    valid = (left["conf_h36m"] > config.minimum_joint_confidence) & (
        right["conf_h36m"] > config.minimum_joint_confidence
    )
    if valid.sum() < config.minimum_valid_joints:
        return None
    ground_truth = extract_gt_3d(
        context["raw_gt"], context["marker_names"], frame, context["joint_markers"]
    )
    if ground_truth is None:
        return None
    return left, right, dynamic_offset, local_offset, sync_score, dynamic_score, ground_truth


def _evaluate_methods(methods, context, left, right, ground_truth):
    results = {}
    for name, triangulate in methods.items():
        reconstruction = triangulate(
            context["p1"], context["p2"], left["kps_h36m"], right["kps_h36m"],
            left["conf_h36m"], right["conf_h36m"],
        )
        groups = evaluate_joint_groups(reconstruction, ground_truth)
        results[name] = {
            "recon_3d": reconstruction,
            "mpjpe": mpjpe(reconstruction, ground_truth),
            "pa_mpjpe": pa_mpjpe(reconstruction, ground_truth),
            "pa_clear": groups["pa_clear"],
            "pa_derived": groups["pa_derived"],
        }
    return results


def _dlt_synced_baseline(context, left, right, ground_truth):
    """Pure DLT triangulation on the DTW/dynamically-synchronized pose pair."""
    recon = triangulate_dlt(context["p1"], context["p2"], left["kps_h36m"], right["kps_h36m"])
    groups = evaluate_joint_groups(recon, ground_truth)
    return {
        "recon_3d": recon,
        "mpjpe": mpjpe(recon, ground_truth),
        "pa_mpjpe": pa_mpjpe(recon, ground_truth),
        "pa_clear": groups["pa_clear"],
        "pa_derived": groups["pa_derived"],
    }


def _build_frame_result(context, frame, inputs, method_results, selected_label, dlt_baseline):
    left, right, dynamic_offset, local_offset, sync_score, dynamic_score, ground_truth = inputs
    primary, pair, best = method_results[selected_label], context["pair_info"], context["best_pair"]
    dlt_mpjpe = dlt_baseline["mpjpe"] if dlt_baseline else float("nan")
    dlt_pa = dlt_baseline["pa_mpjpe"] if dlt_baseline else float("nan")

    occ_info = detect_stereo_occlusions(
        context["p1"], context["p2"], left["kps_h36m"], right["kps_h36m"],
        left["conf_h36m"], right["conf_h36m"],
        bone_lengths=context.get("bone_lengths"),
        initial_3d=primary.get("recon_3d"),
    )

    result = {
        "motion": pair["motion"], "subject": pair["subject"],
        "cam_a": context["camera_a"]["cam_id"], "cam_b": context["camera_b"]["cam_id"],
        "frame": frame, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "mpjpe": primary["mpjpe"], "pa_mpjpe": primary["pa_mpjpe"],
        "pa_clear": primary.get("pa_clear", float("nan")),
        "pa_derived": primary.get("pa_derived", float("nan")),
        "gt_3d": ground_truth, "recon_3d": primary["recon_3d"],
        "kps2d_a": left["kps_coco"], "kps2d_b": right["kps_coco"],
        "conf_a": left["conf_coco"], "conf_b": right["conf_coco"],
        "frame_a": frame, "frame_b": frame + local_offset, "P1": context["p1"], "P2": context["p2"],
    }
    result.update({
        "kps2d_a_h36m": left["kps_h36m"], "kps2d_b_h36m": right["kps_h36m"],
        "conf_a_h36m": occ_info["conf_a"], "conf_b_h36m": occ_info["conf_b"],
        "stereo_conf_h36m": occ_info["stereo_confidence"],
        "occluded_mask": occ_info["occluded_mask"],
        "is_outlier": occ_info["is_outlier"],
        "global_sync_delta": best["delta"], "dynamic_sync_delta": dynamic_offset,
        "local_sync_delta": local_offset, "sync_pair_score": sync_score,
        "dynamic_sync_score": dynamic_score, "best_method": selected_label,
        "all_methods": {
            **method_results,
            "DLT (baseline)": dlt_baseline or {},
        },
        "baseline_dlt_mpjpe": dlt_mpjpe,
        "baseline_dlt_pa": dlt_pa,
        "baseline_dlt_pa_clear": dlt_baseline.get("pa_clear", float("nan")) if dlt_baseline else float("nan"),
        "baseline_dlt_pa_derived": dlt_baseline.get("pa_derived", float("nan")) if dlt_baseline else float("nan"),
    })
    return result



def _process_motion(pair_info, repository, config, valid_videos, selected_label):
    score_options = _sync_score_options(config, pair_info["subject"])
    sync_p = _fps_scaled_sync_params(pair_info, config)
    best_pair = select_best_pair(
        pair_info, repository, valid_videos=valid_videos,
        max_candidates=config.max_pair_candidates, max_offset=sync_p["max_offset"],
        sync_samples=config.sync_samples, coarse_step=sync_p["coarse_step"],
        minimum_valid_ratio=config.minimum_pair_valid_ratio,
        validation_radius=sync_p["local_radius"], score_options=score_options,
    )
    if best_pair is None:
        print(f"SKIP [/{pair_info['subject']}/{pair_info['motion']}]: no valid camera pair.")
        return []
    context = _load_motion_context(pair_info, best_pair, repository, config)
    if context is None:
        return []
    context.update(
        score_options=score_options, fps=sync_p["fps"],
        sync_local_radius=sync_p["local_radius"],
        dynamic_local_radius=sync_p["dynamic_radius"],
        dynamic_transition_weight=sync_p["transition_weight"],
    )
    _add_dynamic_sync(context, repository, config)
    triangulate_fn = _make_triangulation_function(
        config.triangulation_method, config.method_options, config, context["bone_lengths"],
    )
    results = []
    for frame in context["frames"]:
        inputs = _frame_inputs(context, frame, repository, config)
        if inputs is None:
            continue
        left, right, ground_truth = inputs[0], inputs[1], inputs[-1]
        dlt_baseline = _dlt_synced_baseline(context, left, right, ground_truth)
        method_results = _evaluate_methods({selected_label: triangulate_fn}, context, left, right, ground_truth)
        results.append(_build_frame_result(context, frame, inputs, method_results, selected_label, dlt_baseline))
    return results


def run_pose_pipeline(multicam_pairs, repository, config, valid_videos=None):
    selected_label, _ = _selected_triangulation(config)
    results, processed = [], set()
    pairs = tqdm(multicam_pairs[: config.max_pairs], desc="Processing 2-camera motions")
    for pair_info in pairs:
        motion_key = f"/{pair_info['subject']}/{pair_info['motion']}"
        subj_included = not config.included_subjects or pair_info["subject"] in config.included_subjects
        excluded = not subj_included or pair_info["subject"] in config.excluded_subjects or motion_key in config.excluded_motions
        if excluded or motion_key in processed:
            continue
        processed.add(motion_key)
        results.extend(_process_motion(
            pair_info, repository, config, valid_videos, selected_label,
        ))
    return results


def run_benchmark_chunk(multicam_pairs, repository, config, valid_videos=None):
    results = run_pose_pipeline(multicam_pairs, repository, config, valid_videos)
    refinement = dict(config.sequence_refinement)
    if not results or not refinement.pop("enabled"):
        return results
    source_method = method_label(config.triangulation_method)
    bone_lengths = h36m_bone_lengths_from_height(config.subject_height_mm)
    return refine_results(
        results,
        source_method=source_method,
        target_method=f"{source_method} + SequenceRefine",
        bone_lengths=bone_lengths,
        **refinement,
    )
