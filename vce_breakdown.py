"""vce_breakdown.py — which joints drive the 0.21? Reference vs ours."""
import os, json, math
import numpy as np
from voidx_experiment import load_local_meshes, load_gt_skeleton
from voidx_pipeline import load_mesh_trimesh
from voidx_metrics import bones_to_segments, _point_to_segment_dist

CORE = {"pelvis", "neck", "shoulder_L", "shoulder_R", "elbow_L", "elbow_R",
        "wrist_L", "wrist_R", "hip_L", "hip_R", "knee_L", "knee_R",
        "ankle_L", "ankle_R"}
DATASET, PREDS = r"C:\Users\diwak\Downloads\HumanRig_test", r"./results/preds"

def per_joint_vce(skel, mesh):
    bones = bones_to_segments(skel)
    out = {}
    for name, pos in skel.items():
        p = np.array([pos["x"], pos["y"], pos["z"]])
        best, best_d = None, np.inf
        for a, b in bones:
            d = _point_to_segment_dist(p, a, b)
            if d < best_d: best_d, best = d, (a, b)
        a, b = best; ab = b - a
        t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
        proj = a + t * ab
        bd = ab / max(np.linalg.norm(ab), 1e-12)
        ref = np.array([1., 0, 0]) if abs(bd[0]) < 0.9 else np.array([0, 1., 0])
        u = np.cross(bd, ref); u /= max(np.linalg.norm(u), 1e-12)
        v = np.cross(bd, u)
        hits = {}
        for k in range(8):
            ang = 2 * math.pi * k / 8
            d = math.cos(ang) * u + math.sin(ang) * v
            try:
                locs, _, _ = mesh.ray.intersects_location(
                    ray_origins=np.array([proj]), ray_directions=np.array([d]))
                if len(locs) > 0: hits[k] = np.asarray(locs[0])
            except Exception: pass
        mids, pairs = [], []
        for k in range(4):
            if k in hits and k + 4 in hits:
                mids.append(0.5 * (hits[k] + hits[k + 4]))
                pairs.append(float(np.linalg.norm(hits[k] - hits[k + 4])))
        if len(pairs) < 2 or np.median(pairs) < 1e-6: continue
        center = np.mean(mids, axis=0)
        ev = p - center
        radial = ev - np.dot(ev, bd) * bd
        out[name] = float(np.linalg.norm(radial) / np.median(pairs))
    return out

def main():
    dataset = {it["name"]: it for it in load_local_meshes(DATASET)}
    gt_j, our_j, gt_m, our_m = {}, {}, [], []
    for f in sorted(os.listdir(PREDS)):
        name = f[:-5]
        item = dataset[name]
        gt = {k: v for k, v in load_gt_skeleton(item["gt_path"]).items() if k in CORE}
        pred = json.load(open(os.path.join(PREDS, f)))["pose"]
        mesh = load_mesh_trimesh(item["mesh_path"])
        vg, vo = per_joint_vce(gt, mesh), per_joint_vce(pred, mesh)
        for j, e in vg.items(): gt_j.setdefault(j, []).append(e)
        for j, e in vo.items(): our_j.setdefault(j, []).append(e)
        if vg and vo:
            gt_m.append(np.mean(list(vg.values())))
            our_m.append(np.mean(list(vo.values())))
    print(f"{'joint':<12} {'GT':>7} {'ours':>7} {'n':>4}")
    for j in sorted(gt_j):
        print(f"{j:<12} {np.mean(gt_j[j]):7.3f} "
              f"{np.mean(our_j.get(j, [float('nan')])):7.3f} {len(gt_j[j]):4d}")
    from scipy.stats import ttest_rel
    t, p = ttest_rel(gt_m, our_m)
    print(f"\noverall: GT {np.mean(gt_m):.3f} | ours {np.mean(our_m):.3f}")
    print(f"paired t-test (n={len(gt_m)}): t={t:.3f}, p={p:.3f}")

if __name__ == "__main__":
    main()