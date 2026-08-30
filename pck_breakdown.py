"""pck_breakdown.py — per-joint PCK@5%: do misses concentrate where the
reference pivot floats outside the mesh?"""
import os, json
import numpy as np
from scipy.optimize import linear_sum_assignment
from voidx_experiment import load_local_meshes, load_gt_skeleton
from voidx_pipeline import load_mesh_trimesh
from voidx_metrics import joints_to_array, bbox_diagonal

CORE = {"pelvis", "neck", "shoulder_L", "shoulder_R", "elbow_L", "elbow_R",
        "wrist_L", "wrist_R", "hip_L", "hip_R", "knee_L", "knee_R",
        "ankle_L", "ankle_R"}
JUNCTION_JOINTS = {"shoulder_L", "shoulder_R", "hip_L", "hip_R",
                   "pelvis", "neck"}
DATASET, PREDS, TAU = r"C:\Users\diwak\Downloads\HumanRig_test", r"./results/preds", 0.05

def main():
    dataset = {it["name"]: it for it in load_local_meshes(DATASET)}
    from trimesh.proximity import ProximityQuery
    stats = {}   # joint -> [hits, n, sum_err, sum_surf]
    for f in sorted(os.listdir(PREDS)):
        name = f[:-5]; item = dataset[name]
        pred = json.load(open(os.path.join(PREDS, f)))["pose"]
        gt = {k: v for k, v in load_gt_skeleton(item["gt_path"]).items() if k in CORE}
        if not gt or not pred:
            continue
        mesh = load_mesh_trimesh(item["mesh_path"])
        P, G = joints_to_array(pred), joints_to_array(gt)
        cost = np.linalg.norm(P[:, None, :] - G[None, :, :], axis=2)
        r, c = linear_sum_assignment(cost)
        diag = bbox_diagonal(mesh)
        _, surf, _ = ProximityQuery(mesh).on_surface(G)
        surf = np.asarray(surf, dtype=float)
        gnames = list(gt.keys())
        for k in range(len(c)):
            jn = gnames[c[k]]
            s = stats.setdefault(jn, [0, 0, 0.0, 0.0])
            s[1] += 1
            s[2] += 100 * cost[r[k], c[k]] / diag          # error % diag
            s[3] += 100 * surf[c[k]] / diag                # GT pivot surf dist % diag
            if cost[r[k], c[k]] <= TAU * diag:
                s[0] += 1
    print(f"{'joint':<12} {'PCK@5%':>7} {'n':>4} {'err%diag':>9} {'GTsurf%diag':>11}")
    for j in sorted(stats):
        h, n, se, ss = stats[j]
        print(f"{j:<12} {100*h/n:7.1f} {n:4d} {se/n:9.2f} {ss/n:11.2f}")

if __name__ == "__main__":
    main()