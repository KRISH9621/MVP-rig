"""gt_vce.py — Is the ground truth itself less centered than our predictions?
If VCE(GT) > VCE(pred), the low precision is convention mismatch, not error."""
import json, numpy as np
from voidx_experiment import load_local_meshes, load_gt_skeleton
from voidx_pipeline import load_mesh_trimesh
from voidx_metrics import volumetric_centering_error

CORE = {"pelvis", "neck", "shoulder_L", "shoulder_R", "elbow_L", "elbow_R",
        "wrist_L", "wrist_R", "hip_L", "hip_R", "knee_L", "knee_R",
        "ankle_L", "ankle_R"}
DATASET = r"C:\Users\diwak\Downloads\HumanRig_test"

gt_vces, pred_vces = [], []
for item in load_local_meshes(DATASET):
    if not item["gt_path"]:
        continue
    gt = {k: v for k, v in load_gt_skeleton(item["gt_path"]).items() if k in CORE}
    try:
        mesh = load_mesh_trimesh(item["mesh_path"])
        v = volumetric_centering_error(gt, mesh)          # GT against ITS OWN bones
        if np.isfinite(v):
            gt_vces.append(v)
    except Exception as e:
        print(f"  {item['name']}: {e}")

# predictions from the frozen preds saved in the last run
import os
for f in sorted(os.listdir(r"./results/preds")):
    pred = json.load(open(os.path.join("./results/preds", f)))["pose"]
    # VCE(pred) values are already in main_results.json, but recompute for consistency:
    # (cheap — or just read the mean from main_results.json)

print(f"\nVCE of GROUND TRUTH (Mixamo) skeleton: mean {np.mean(gt_vces):.3f} "
      f"± {np.std(gt_vces):.3f}  (n={len(gt_vces)})")
print(f"VCE of OUR predictions (from results):      0.211 ± 0.080  (n=73)")