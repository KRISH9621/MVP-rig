"""calibrate_tolerance.py — choose the correspondence tolerance ONCE, from data."""
import os, json
import numpy as np
from scipy.optimize import linear_sum_assignment
from voidx_experiment import load_local_meshes, load_gt_skeleton
from voidx_pipeline import load_mesh_trimesh
from voidx_metrics import joints_to_array, bbox_diagonal
from voidx_pipeline import load_mesh_trimesh
from voidx_experiment import load_local_meshes

wt = [load_mesh_trimesh(it["mesh_path"]).is_watertight
      for it in load_local_meshes(r"C:\Users\diwak\Downloads\HumanRig_test")]
print(f"{sum(wt)}/{len(wt)} meshes watertight")
RESULTS, DATASET = r"./results", r"C:\Users\diwak\Downloads\HumanRig_test"
CORE_JOINTS = {"pelvis", "neck", "shoulder_L", "shoulder_R", "elbow_L",
               "elbow_R", "wrist_L", "wrist_R", "hip_L", "hip_R",
               "knee_L", "knee_R", "ankle_L", "ankle_R"}

CRITERIA = [
    ("surface-dist x1.0 (current)", lambda s, D: s),
    ("surface-dist x2.0",           lambda s, D: 2.0 * s),
    ("PCK 2.5% diag",               lambda s, D: 0.025 * D * np.ones_like(s)),
    ("PCK 5%   diag",               lambda s, D: 0.050 * D * np.ones_like(s)),
    ("PCK 7.5% diag",               lambda s, D: 0.075 * D * np.ones_like(s)),
    ("PCK 10%  diag",               lambda s, D: 0.100 * D * np.ones_like(s)),
]

def main():
    dataset = {it["name"]: it for it in load_local_meshes(DATASET)}
    preds_dir = os.path.join(RESULTS, "preds")
    names = sorted(f[:-5] for f in os.listdir(preds_dir) if f.endswith(".json"))
    print(f"{len(names)} frozen predictions found.")
    matched = {c[0]: 0 for c in CRITERIA}
    tot_p = tot_g = outside = gt_total = 0
    surf_pcts, dump = [], []

    for i, name in enumerate(names):
        item = dataset[name]
        pred = json.load(open(os.path.join(preds_dir, name + ".json")))["pose"]
        gt = {k: v for k, v in load_gt_skeleton(item["gt_path"]).items()
              if k in CORE_JOINTS}
        if not gt or not pred:
            continue
        mesh = load_mesh_trimesh(item["mesh_path"])
        P, G = joints_to_array(pred), joints_to_array(gt)
        cost = np.linalg.norm(P[:, None, :] - G[None, :, :], axis=2)
        r, c = linear_sum_assignment(cost)
        d_pg, diag = cost[r, c], bbox_diagonal(mesh)

        from trimesh.proximity import ProximityQuery
        _, surf, _ = ProximityQuery(mesh).on_surface(G)
        surf = np.asarray(surf, dtype=float)
        try:
            outside += int((~mesh.contains(G)).sum()); gt_total += len(G)
        except Exception:
            pass
        surf_pcts.extend((100 * surf / diag).tolist())

        for cname, fn in CRITERIA:
            thr = fn(surf, diag)
            matched[cname] += int(np.sum(d_pg <= thr[c]))
        tot_p += len(P); tot_g += len(G)

        if i == 0:  # per-joint dump for the first character
            gnames = list(gt.keys())
            dump = [(gnames[c[k]], 100*d_pg[k]/diag, 100*surf[c[k]]/diag)
                    for k in range(len(c))]

    print(f"\nGT joints OUTSIDE mesh: {outside}/{gt_total}")
    print(f"GT surface distance: mean {np.mean(surf_pcts):.2f}% diag, "
          f"median {np.median(surf_pcts):.2f}% diag")
    print("\n%-30s %10s %8s %8s" % ("Criterion", "Precision", "Recall", "IoU"))
    for cname, _ in CRITERIA:
        m = matched[cname]
        iou = m / (tot_p + tot_g - m) * 100 if (tot_p + tot_g - m) else 0
        print("%-30s %10.1f %8.1f %8.1f" %
              (cname, m / tot_p * 100, m / tot_g * 100, iou))

    print("\nPer-joint detail (first character):  joint | err%diag | GTsurf%diag")
    for jn, e, s in dump:
        print(f"  {jn:<14} {e:7.2f}   {s:7.2f}")

if __name__ == "__main__":
    main()