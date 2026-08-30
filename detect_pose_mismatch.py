r"""
detect_pose_mismatch.py — Flag characters whose GT skeleton is T-posed
but whose mesh is NOT (the Mixamo "T-pose download" mismatch).

Logic:
  1. GT is T-posed if wrists sit at shoulder height in the GT skeleton.
  2. Mesh is NOT T-posed if the GT's wrist-to-wrist span exceeds the
     mesh's actual horizontal width by >20%.

Usage:
    python detect_pose_mismatch.py --dataset_dir "C:\Users\diwak\Downloads\HumanRig_test"
"""
import argparse, json
import numpy as np
import trimesh
from pathlib import Path

def pos(skel, name):
    c = skel.get(name)
    if c is None:
        return None
    if isinstance(c, dict):
        return np.array([c["x"], c["y"], c["z"]], dtype=float)
    return np.array(c[:3], dtype=float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    args = ap.parse_args()
    root = Path(args.dataset_dir)

    affected, suspect, gt_not_t, errors, ok = [], [], [], [], []

    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        gt_file = d / "bone_3d.json"
        mesh_files = [f for f in d.iterdir() if f.suffix.lower() in (".obj", ".glb", ".ply")]
        if not gt_file.exists() or not mesh_files:
            continue
        try:
            gt = json.load(open(gt_file))
            mesh = trimesh.load(str(mesh_files[0]), force="mesh")

            sl, sr = pos(gt, "shoulder_L"), pos(gt, "shoulder_R")
            wl, wr = pos(gt, "wrist_L"), pos(gt, "wrist_R")
            if any(p is None for p in (sl, sr, wl, wr)):
                errors.append((d.name, "GT missing arm joints"))
                continue

            scale = float(max(mesh.extents))  # roughly character height

            # --- Check 1: is the GT skeleton itself T-posed? ---
            gt_tpose = (abs(wl[1] - sl[1]) < 0.12 * scale and
                        abs(wr[1] - sr[1]) < 0.12 * scale)

            # --- Check 2: does the GT arm span fit inside the mesh? ---
            gt_span = max(wl[0], wr[0]) - min(wl[0], wr[0])
            ext = sorted(mesh.extents)          # [smallest, middle, largest]
            mesh_w = ext[1]                     # widest HORIZONTAL extent
                                                # (largest = vertical Y)
            ratio = gt_span / max(mesh_w, 1e-9)

            if not gt_tpose:
                gt_not_t.append((d.name, ratio))
                tag = "GT_NOT_TPOSE (verify visually)"
            elif ratio > 1.20:
                affected.append((d.name, ratio))
                tag = "AFFECTED  <<< needs re-rig"
            elif ratio > 1.08:
                suspect.append((d.name, ratio))
                tag = "SUSPECT   (check visually)"
            else:
                ok.append(d.name)
                tag = "ok"
            print(f"  {d.name:12s} span/meshW={ratio:5.2f}  {tag}")
        except Exception as e:
            errors.append((d.name, str(e)))

    print("\n" + "=" * 60)
    print(f"OK:        {len(ok)}")
    print(f"AFFECTED:  {len(affected)}  {[n for n, r in affected]}")
    print(f"SUSPECT:   {len(suspect)}  {[n for n, r in suspect]}")
    print(f"GT_NOT_T:  {len(gt_not_t)}  {[n for n, r in gt_not_t]}")
    print(f"ERRORS:    {len(errors)}")
    print("=" * 60)
    json.dump({"affected": affected, "suspect": suspect,
               "gt_not_tpose": gt_not_t, "errors": errors},
              open("pose_mismatch_report.json", "w"), indent=2)
    print("Report saved: pose_mismatch_report.json")

if __name__ == "__main__":
    main()