# gt_displacement_audit.py — settle it with numbers, not screenshots.
import json, numpy as np
from voidx_experiment import load_local_meshes, load_gt_skeleton
from voidx_pipeline import load_mesh_trimesh
from voidx_metrics import joints_to_array, bbox_diagonal

CORE = {"pelvis", "neck", "shoulder_L", "shoulder_R", "elbow_L", "elbow_R",
        "wrist_L", "wrist_R", "hip_L", "hip_R", "knee_L", "knee_R",
        "ankle_L", "ankle_R"}
DATASET = r"C:\Users\diwak\Downloads\HumanRig_test"

# A constant displacement has ONE fingerprint: the offset from the
# volumetric center is the SAME VECTOR on every joint of every model.
# We estimate that vector as: (reference centroid) - (mesh bbox center),
# per model, then check whether it's consistent across models after
# normalization. Random anatomical placement does NOT correlate across
# models; a frame bug DOES.

offsets = []
surf_bias = {a: [] for a in range(3)}   # signed surface-distance bias per axis
for item in load_local_meshes(DATASET):
    if not item["gt_path"]:
        continue
    gt = {k: v for k, v in load_gt_skeleton(item["gt_path"]).items() if k in CORE}
    if not gt:
        continue
    mesh = load_mesh_trimesh(item["mesh_path"])
    G = joints_to_array(gt)
    bmin, bmax = mesh.bounds
    diag = bbox_diagonal(mesh)

    # --- Test 1: centroid offset, normalized by diag ---
    mesh_c = (bmin + bmax) / 2
    off = (G.mean(axis=0) - mesh_c) / diag
    offsets.append(off)

    # --- Test 2: is the reference consistently pushed toward ONE surface?
    # For each joint, signed distance from the mesh's Z-center (front-back).
    # If bones hug the back on EVERY model, mean signed offset << 0.
    # (Z is your front-back axis after the trimesh load; verify below.)
    surf_bias[0].extend(((G[:, 0] - mesh_c[0]) / diag).tolist())
    surf_bias[1].extend(((G[:, 1] - mesh_c[1]) / diag).tolist())
    surf_bias[2].extend(((G[:, 2] - mesh_c[2]) / diag).tolist())

offsets = np.array(offsets)
print(f"n_models = {len(offsets)}")
print(f"\nCentroid offset (GT - mesh center), % of diag:")
print(f"  X: mean {offsets[:,0].mean()*100:+.2f}%  std {offsets[:,0].std()*100:.2f}%")
print(f"  Y: mean {offsets[:,1].mean()*100:+.2f}%  std {offsets[:,1].std()*100:.2f}%")
print(f"  Z: mean {offsets[:,2].mean()*100:+.2f}%  std {offsets[:,2].std()*100:.2f}%")

print(f"\nPer-joint signed offset from mesh center (all joints, all models):")
for i, ax in enumerate("XYZ"):
    b = np.array(surf_bias[i])
    print(f"  {ax}: mean {b.mean()*100:+.2f}%  std {b.std()*100:.2f}%  "
          f"fraction |offset|>10%: {(np.abs(b)>0.10).mean()*100:.0f}%")

print(f"""
HOW TO READ:
- A CONSTANT DISPLACEMENT BUG shows: one axis with mean offset > +5% or
  < -5% AND std < 2% (the same push on every model). That's a frame bug
  -> we fix the extractor, recompute metrics on frozen preds (~5 min).
- ANATOMICAL CONVENTION shows: Z (front-back) mean maybe -1% to -3%
  (spine at back) but X ~0, and LOTS of variance (std >= mean).
- Y-up/Z-up ROTATION shows: huge offsets (>20%) on two axes, or a
  mean offset comparable to the skeleton's own extent.
""")