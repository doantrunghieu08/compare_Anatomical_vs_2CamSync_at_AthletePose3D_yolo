from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .io.data import PoseRepository, find_multicam_pairs, load_json, load_valid_videos, scan_local_dataset
from .io.reporting import CsvResultReporter
from .pipeline import run_benchmark_chunk
from .settings import load_settings, method_label


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yml"


def _apply_overrides(
    inputs, output, config, input_overrides, output_path, max_pairs, overwrite,
    method_name=None, subject_height=None, uncalibrated=None,
    included_subjects=None, fps_override=None,
):
    if input_overrides:
        inputs = replace(inputs, **input_overrides)
    if output_path is not None:
        output = replace(output, results_csv=output_path)
    if overwrite is not None:
        output = replace(output, overwrite=overwrite)
    if max_pairs is not None:
        if max_pairs <= 0:
            raise ValueError("max_pairs must be greater than zero")
        config = replace(config, max_pairs=max_pairs)
    if method_name is not None:
        options = config.method_options if method_name == config.triangulation_method else {}
        config = replace(config, triangulation_method=method_name, method_options=options)
    if uncalibrated is not None:
        config = replace(config, uncalibrated=uncalibrated)
    if subject_height is not None:
        try:
            h_val = float(subject_height)
        except ValueError:
            h_val = str(subject_height)
        config = replace(config, subject_height_mm=h_val)
    if included_subjects:
        config = replace(config, included_subjects=tuple(included_subjects))
    if fps_override is not None:
        try:
            fps_val = float(fps_override)
        except ValueError:
            fps_val = str(fps_override)
        config = replace(config, fps_mode=fps_val)
    return inputs, output, config


def _eligible_pairs(inputs, config):
    dataset = scan_local_dataset(
        inputs.test_set_dir,
        inputs.video_extension,
        inputs.metadata_extension,
        inputs.ground_truth_extension,
    )
    pairs = sorted(find_multicam_pairs(dataset), key=lambda item: (item["subject"], item["motion"]))
    return [
        pair
        for pair in pairs
        if (not config.included_subjects or pair["subject"] in config.included_subjects)
        and pair["subject"] not in config.excluded_subjects
        and f"/{pair['subject']}/{pair['motion']}" not in config.excluded_motions
    ][: config.max_pairs]


def _run_chunks(pairs, repository, config, valid_videos, reporter):
    results = []
    for start in range(0, len(pairs), config.chunk_size):
        chunk = pairs[start : start + config.chunk_size]
        chunk_config = replace(config, max_pairs=len(chunk))
        chunk_results = run_benchmark_chunk(chunk, repository, chunk_config, valid_videos)
        reporter.append(chunk_results)
        results.extend(chunk_results)
    reporter.finish()
    return results


def run_benchmark(
    config_path: str | Path = DEFAULT_CONFIG, *, input_overrides=None,
    output_path=None, max_pairs=None, overwrite=None, method_name=None,
    subject_height=None, uncalibrated=None, included_subjects=None,
    fps_override=None,
):
    inputs, output, config = load_settings(config_path)
    inputs, output, config = _apply_overrides(
        inputs, output, config, input_overrides, output_path, max_pairs, overwrite,
        method_name, subject_height, uncalibrated, included_subjects, fps_override,
    )
    if output.results_csv.exists() and not output.overwrite:
        raise FileExistsError(f"Output already exists: {output.results_csv}. Pass --overwrite to replace it.")
    pairs = _eligible_pairs(inputs, config)
    if not pairs:
        target = f" matching {config.included_subjects}" if config.included_subjects else ""
        raise RuntimeError(f"No eligible multi-camera motions found in {inputs.test_set_dir}{target}")
    label = method_label(config.triangulation_method)
    if output.results_csv.exists():
        output.results_csv.unlink()
    reporter = CsvResultReporter(
        output.results_csv,
        notes=f"config={Path(config_path).name}; method={label}; options={config.method_options}",
    )
    print(f"Method: {label}")
    if config.included_subjects:
        print(f"Subjects: {', '.join(config.included_subjects)}")
    print(f"Motions: {len(pairs)} | chunk_size: {config.chunk_size} | output: {output.results_csv}")
    repo = PoseRepository(inputs.pose2d_root, inputs.pose_2d_suffix, inputs.key_frames_suffix, inputs.pose_cache_limit)
    valid_vids = load_valid_videos(inputs.manifest, inputs.minimum_manifest_confidence)
    return _run_chunks(pairs, repo, config, valid_vids, reporter)


def build_parser():
    parser = argparse.ArgumentParser(description="Run the AthletePose3D two-camera benchmark locally.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config (default: configs/default.yml).")
    parser.add_argument("--dataset-root", type=Path, help="Override input.dataset_root from YAML.")
    parser.add_argument("--pose2d-root", type=Path, help="Override input.pose2d_root from YAML.")
    parser.add_argument("--output", type=Path, help="Override output.results_csv from YAML.")
    parser.add_argument("--max-pairs", type=int, help="Override pipeline.max_pairs from YAML.")
    parser.add_argument("--method", type=str, help="Override method.name from YAML.")
    parser.add_argument("--subject-height", type=str, help="Subject height in mm (e.g. 1732.0) or 'auto'.")
    parser.add_argument("--uncalibrated", action="store_true", default=None, help="Run without camera parameters after pair selection.")
    parser.add_argument("--overwrite", action="store_true", default=None, help="Replace an existing output CSV.")
    parser.add_argument("-S1", "--S1", dest="s1", action="store_true", help="Run only subject S1.")
    parser.add_argument("-S2", "--S2", dest="s2", action="store_true", help="Run only subject S2.")
    parser.add_argument("-S3", "--S3", dest="s3", action="store_true", help="Run only subject S3.")
    parser.add_argument("--subject", "--subjects", dest="subjects", nargs="+", type=str, help="Run specific subject(s).")
    parser.add_argument("--fps", type=str, help="Override fps mode: 'auto' or numeric value (e.g. 60, 120).")
    return parser


def _extract_subjects(args) -> list[str]:
    subjects = list(args.subjects or [])
    if getattr(args, "s1", False) and "S1" not in subjects:
        subjects.append("S1")
    if getattr(args, "s2", False) and "S2" not in subjects:
        subjects.append("S2")
    if getattr(args, "s3", False) and "S3" not in subjects:
        subjects.append("S3")
    return subjects


def main(argv=None):
    args = build_parser().parse_args(argv)
    input_overrides = {
        name: value.resolve()
        for name, value in {
            "dataset_root": args.dataset_root,
            "pose2d_root": args.pose2d_root,
        }.items()
        if value is not None
    }
    subjects = _extract_subjects(args)
    results = run_benchmark(
        args.config,
        input_overrides=input_overrides,
        output_path=args.output.resolve() if args.output else None,
        max_pairs=args.max_pairs,
        overwrite=args.overwrite,
        method_name=args.method,
        subject_height=args.subject_height,
        uncalibrated=args.uncalibrated,
        included_subjects=subjects if subjects else None,
        fps_override=args.fps,
    )
    print(f"Completed: {len(results)} valid frames")


if __name__ == "__main__":
    main()
