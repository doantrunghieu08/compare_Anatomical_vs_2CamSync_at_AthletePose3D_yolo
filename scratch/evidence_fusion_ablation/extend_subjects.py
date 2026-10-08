import os, sys, time, traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
root = Path.cwd()
os.environ["PYTHONPATH"] = "src"
sys.path.insert(0, os.environ["PYTHONPATH"])
from athlete_pose3d.main import run_benchmark
out = root / "outputs" / "evidence_fusion_ablation"
while not (out / "D_candidate_S3.status").exists():
    time.sleep(2)
cases = [
    ("A_legacy_S2", "S2"),
    ("D_candidate_S2", "S2"),
    ("A_legacy_S3", "S3"),
    ("D_candidate_S3", "S3"),
]
for name, subject in cases:
    status = out / (name + "_extended.status")
    try:
        with (out / (name + "_extended.log")).open("w", encoding="utf-8") as log:
            with redirect_stdout(log), redirect_stderr(log):
                results = run_benchmark(
                    "scratch/evidence_fusion_ablation/" + name + ".yml",
                    included_subjects=(subject,), max_pairs=5, overwrite=True,
                )
        status.write_text("ok frames=" + str(len(results)), encoding="utf-8")
    except BaseException:
        with (out / (name + "_extended.log")).open("a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
        status.write_text("error", encoding="utf-8")
        raise
