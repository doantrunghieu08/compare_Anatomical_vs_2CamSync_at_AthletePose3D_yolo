import os, sys, time, traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
root = Path.cwd()
os.environ["PYTHONPATH"] = "src"
sys.path.insert(0, os.environ["PYTHONPATH"])
sys.path.insert(0, "experiments")
from run_ablation_with_joints import run_ablation
out = root / "outputs" / "evidence_fusion_ablation"
while not (out / "dst_anat_final_S3.status").exists():
    time.sleep(2)
status = out / "h36m_ablation_S3.status"
try:
    with (out / "h36m_ablation_S3.log").open("w", encoding="utf-8") as log:
        with redirect_stdout(log), redirect_stderr(log):
            run_ablation(
                config_path="scratch/evidence_fusion_ablation/h36m_S3.yml",
                subject="S3", max_motions=5,
                output_dir=str(out / "h36m_ablation_S3"),
            )
    status.write_text("ok", encoding="utf-8")
except BaseException:
    with (out / "h36m_ablation_S3.log").open("a", encoding="utf-8") as log:
        traceback.print_exc(file=log)
    status.write_text("error", encoding="utf-8")
    raise
