import os, sys, time, traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
root = Path.cwd()
os.environ["PYTHONPATH"] = "src"
sys.path.insert(0, os.environ["PYTHONPATH"])
sys.path.insert(0, "experiments")
from run_ablation_with_joints import run_ablation
out = root / "outputs" / "evidence_fusion_ablation"
while not (out / "D_final_S3.status").exists():
    time.sleep(2)
for subject in ["S2", "S3"]:
    name = "physics_ablation_" + subject
    status = out / (name + ".status")
    try:
        with (out / (name + ".log")).open("w", encoding="utf-8") as log:
            with redirect_stdout(log), redirect_stderr(log):
                run_ablation(
                    config_path="configs/default.yml",
                    subject=subject,
                    max_motions=5,
                    output_dir=str(out / name),
                )
        status.write_text("ok", encoding="utf-8")
    except BaseException:
        with (out / (name + ".log")).open("a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
        status.write_text("error", encoding="utf-8")
        raise
