r"""
validate_dataset.py — Pre-publication gate for the MVP-Rig dataset.
Run this BEFORE zipping. If anything fails, fix it, don't ignore it.

    python validate_dataset.py "C:\Users\diwak\Desktop\MVP-Rig-Dataset-v1.0"
"""
import argparse, json
import numpy as np
import trimesh
from pathlib import Path

MIXAMO_MAP = {
    "Hips": "pelvis", "Neck": "neck",
    "LeftShoulder": "shoulder_L", "RightShoulder": "shoulder_R",
    "LeftForeArm": "elbow_L", "RightForeArm": "elbow_R",
    "LeftHand": "wrist_L", "RightHand": "wrist_R",
    "LeftUpLeg": "hip_L", "RightUpLeg": "hip_R",
    "LeftLeg": "knee_L", "RightLeg": "knee_R",
    "LeftFoot": "ankle_L", "RightFoot": "ankle_R",
}
CORE = {"pelvis","neck","shoulder_L","shoulder_R","elbow_L","elbow_R",
        "wrist_L","wrist_R","hip_L","hip_R","knee_L","knee_R","ankle_L","ankle_R"}

def check_dataset(root: Path):
    chars = sorted([d for d in root.iterdir() if d.is_dir()])
    print(f"Found {len(chars)} character folders\n")
    report, verts, faces, tpose = [], [], [], 0

    for d in chars:
        issues = []
        mesh_file = d / "mesh.obj"
        gt_file = d / "bone_3d.json"

        # -- Check 1: files exist --
        if not mesh_file.exists():
            issues.append("MISSING mesh.obj")
        if not gt_file.exists():
            issues.append("MISSING bone_3d.json")
        if issues:
            report.append((d.name, issues)); continue

        # -- Check 2: GT parses as JSON, has core joints after mapping --
        try:
            gt_raw = json.load(open(gt_file))
        except json.JSONDecodeError:
            issues.append("bone_3d.json is INVALID JSON"); report.append((d.name, issues)); continue
        mapped = {MIXAMO_MAP.get(k, k): v for k, v in gt_raw.items()}
        core = {k: v for k, v in mapped.items() if k in CORE}
        if len(core) < 14:
            issues.append(f"only {len(core)}/14 core joints mapped")

        # -- Check 3: mesh loads + GT joints inside mesh bounds --
        try:
            mesh = trimesh.load(str(mesh_file), force="mesh")
            verts.append(len(mesh.vertices)); faces.append(len(mesh.faces))
        except Exception as e:
            issues.append(f"mesh.obj failed to load: {e}"); report.append((d.name, issues)); continue

        if core:
            P = np.array([[c[0] if isinstance(c, list) else c["x"],
                           c[1] if isinstance(c, list) else c["y"],
                           c[2] if isinstance(c, list) else c["z"]]
                          for c in core.values()], dtype=float)
            bmin, bmax = mesh.bounds
            margin = 0.05 * float(np.linalg.norm(bmax - bmin))
            n_out = int(np.any((P < bmin - margin) | (P > bmax + margin), axis=1).sum())
            if n_out > 0:
                issues.append(f"{n_out} GT joints outside mesh bounds (coordinate-frame bug?)")
            # crude pose check: wrists at shoulder height = T-pose
            try:
                sl, wl = mapped.get("shoulder_L"), mapped.get("wrist_L")
                if sl and wl:
                    y_sl = sl[1] if isinstance(sl, list) else sl["y"]
                    y_wl = wl[1] if isinstance(wl, list) else wl["y"]
                    if abs(y_sl - y_wl) < 0.12 * max(mesh.extents): tpose += 1
            except Exception: pass

        report.append((d.name, issues))

    # ---- Report ----
    n_bad = sum(1 for _, iss in report if iss)
    print("=" * 60)
    for name, iss in report:
        print(f"  {name:12s} {'OK' if not iss else '; '.join(iss)}")
    print("=" * 60)
    print(f"RESULT: {len(chars) - n_bad}/{len(chars)} characters PASS")
    if n_bad:
        print(">>> FIX THE FAILURES BEFORE PUBLISHING. Do not ignore. <<<")

    if verts:
        print(f"\n--- Statistics (paste into DATASET.md) ---")
        print(f"Characters: {len(chars)} ({tpose} T-posed, {len(chars)-tpose} A/other)")
        print(f"Vertices: min {min(verts)}, max {max(verts)}, mean {np.mean(verts):.0f}")
        print(f"Faces:    min {min(faces)}, max {max(faces)}, mean {np.mean(faces):.0f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir")
    args = ap.parse_args()
    check_dataset(Path(args.dataset_dir))