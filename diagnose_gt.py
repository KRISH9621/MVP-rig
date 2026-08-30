r"""
diagnose_gt.py — Is bone_3d.json in the wrong coordinate frame (fixable)
or genuinely misplaced (needs re-rigging)?

Usage:
    python diagnose_gt.py --dataset_dir "C:\Users\diwak\Downloads\HumanRig_test"
"""
import argparse, json
import numpy as np
import trimesh
from pathlib import Path
from collections import Counter

TRANSFORMS = {
    "identity":   np.array([[1,0,0],[0,1,0],[0,0,1]]),
    "zup_to_yup": np.array([[1,0,0],[0,0,1],[0,-1,0]]),  # (x,y,z)->(x,z,-y)  Blender->OBJ
    "yup_to_zup": np.array([[1,0,0],[0,0,-1],[0,1,0]]),  # (x,y,z)->(x,-z,y)  inverse
    "rot_x180":   np.array([[1,0,0],[0,-1,0],[0,0,-1]]),
    "rot_y180":   np.array([[-1,0,0],[0,1,0],[0,0,-1]]),
    "rot_z180":   np.array([[-1,0,0],[0,-1,0],[0,0,1]]),
    "mirror_z": np.array([[1,0,0],[0,1,0],[0,0,-1]]),  # the Y/Z-swap bug fix
}

def variants(P, mesh):
    bmin, bmax = mesh.bounds
    mcenter = (bmin + bmax) / 2
    msize = bmax - bmin
    out = {}
    for tname, M in TRANSFORMS.items():
        Q = P @ M.T
        out[tname] = Q
        jcenter = (Q.min(0) + Q.max(0)) / 2
        out[f"{tname}+recenter"] = Q + (mcenter - jcenter)
        jsize = Q.max(0) - Q.min(0) + 1e-9
        out[f"{tname}+recenter+rescale"] = (Q - jcenter) * (msize / jsize) + mcenter
    return out

def inside_score(Q, mesh):
    bmin, bmax = mesh.bounds
    return float(np.all((Q >= bmin) & (Q <= bmax), axis=1).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    args = ap.parse_args()
    root = Path(args.dataset_dir)

    report, verdicts = [], Counter()
    chars = [d for d in sorted(root.iterdir()) if d.is_dir()]
    for d in chars:
        mesh_files = [f for f in d.iterdir() if f.suffix.lower() in (".obj", ".glb", ".ply")]
        gt_file = d / "bone_3d.json"
        if not mesh_files or not gt_file.exists():
            continue
        try:
            mesh = trimesh.load(str(mesh_files[0]), force="mesh")
            gt = json.load(open(gt_file))
            P = np.array([[c[0] if isinstance(c, list) else c["x"],
                           c[1] if isinstance(c, list) else c["y"],
                           c[2] if isinstance(c, list) else c["z"]]
                          for c in gt.values()], dtype=float)
        except Exception as e:
            print(f"  {d.name}: LOAD ERROR {e}")
            continue

        best_name, best_score, best_Q = "identity", -1, P
        for vname, Q in variants(P, mesh).items():
            s = inside_score(Q, mesh)
            if s > best_score:
                best_name, best_score, best_Q = vname, s, Q

        # Stronger check for the best variant: mean distance to mesh surface
        try:
            _, dist, _ = trimesh.proximity.closest_point(mesh, best_Q)
            surf_dist = float(np.mean(dist) / max(mesh.extents))
        except Exception:
            surf_dist = -1

        if inside_score(P, mesh) >= 0.95:
            verdict = "OK (already correct)"
        elif best_score >= 0.95:
            verdict = f"FIXABLE by {best_name}"
        else:
            verdict = "NEEDS RE-RIG"
        verdicts[verdict.split(" by ")[0]] += 1
        report.append((d.name, verdict, best_score, surf_dist))
        print(f"  {d.name:12s} {verdict:35s} score={best_score:.2f} surf={surf_dist:.3f}")

    print("\n" + "=" * 60)
    print("SUMMARY:", dict(verdicts))
    print("=" * 60)
    fixable = Counter(v.split(" by ")[1] for n, v, s, d in report if v.startswith("FIXABLE"))
    if fixable:
        print("Winning transforms:", dict(fixable))
        print("\n>>> Run fix_gt.py with the most common transform above.")
    json.dump([{"char": n, "verdict": v, "score": s, "surf_dist": d}
               for n, v, s, d in report], open("gt_diagnosis.json", "w"), indent=2)
    print("Full report: gt_diagnosis.json")

if __name__ == "__main__":
    main()