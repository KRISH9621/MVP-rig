"""
check_dataset.py — full validation of the HumanRig local dataset.
Usage: python check_dataset.py "D:\HumanRig_Test_Set"

Checks per character:
  1. Mesh loads (with GLB node transforms applied)
  2. bone_3d.json exists, parses, schema is recognized
  3. Joint names map onto the 14 core joints
  4. GT joints fall inside the mesh bounding box (frame check)
"""
import sys, json
from pathlib import Path
import numpy as np
import trimesh

CORE_JOINTS = {
    "pelvis", "neck", "shoulder_L", "shoulder_R", "elbow_L", "elbow_R",
    "wrist_L", "wrist_R", "hip_L", "hip_R", "knee_L", "knee_R",
    "ankle_L", "ankle_R",
}
MIXAMO_MAP = {
    "Hips": "pelvis", "Neck": "neck",
    "LeftShoulder": "shoulder_L", "RightShoulder": "shoulder_R",
    "LeftForeArm": "elbow_L", "RightForeArm": "elbow_R",
    "LeftHand": "wrist_L", "RightHand": "wrist_R",
    "LeftUpLeg": "hip_L", "RightUpLeg": "hip_R",
    "LeftLeg": "knee_L", "RightLeg": "knee_R",
    "LeftFoot": "ankle_L", "RightFoot": "ankle_R",
}

def parse_gt(path):
    with open(path, 'r') as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        for key in ("joints", "bones", "skeleton", "pose"):
            if key in raw and isinstance(raw[key], (dict, list)):
                raw = raw[key]; break
    if isinstance(raw, list):
        out = {}
        for rec in raw:
            if isinstance(rec, dict):
                n = rec.get("name") or rec.get("joint")
                p = rec.get("position") or rec.get("pos")
                if n and isinstance(p, (list, tuple)) and len(p) == 3:
                    out[n] = p
        return out
    if isinstance(raw, dict):
        return {k: v for k, v in raw.items()
                if isinstance(v, (list, tuple)) and len(v) == 3}
    return {}

def load_mesh(path):
    scene = trimesh.load(path)
    if isinstance(scene, trimesh.Scene):
        geoms = []
        for node in scene.graph.nodes_geometry:
            T, gname = scene.graph.get(node)
            g = scene.geometry[gname].copy()
            g.apply_transform(T)
            geoms.append(g)
        return trimesh.util.concatenate(geoms)
    return scene

def main(root):
    root = Path(root)
    if not root.is_dir():
        print(f"Path does not exist or is not a directory: {root}"); return

    seen, problems = set(), 0
    meshes = sorted(p for p in root.rglob("*")
                    if p.suffix.lower() == ".obj")   # .obj only
    print(f"Found {len(meshes)} .obj meshes.\n")

    for p in meshes:
        name = str(p.parent.relative_to(root))
        if name in seen:
            print(f"[DUP ] {name}: duplicate folder name!"); problems += 1
        seen.add(name)

        issues = []
        try:
            mesh = load_mesh(str(p))
        except Exception as e:
            print(f"[FAIL] {name}: mesh won't load: {e}"); problems += 1
            continue

        jp = p.parent / "bone_3d.json"
        if not jp.exists():
            print(f"[WARN] {name}: no bone_3d.json next to mesh")
            problems += 1; continue

        try:
            gt = parse_gt(jp)
        except Exception as e:
            print(f"[FAIL] {name}: JSON parse error: {e}"); problems += 1
            continue

        if not gt:
            print(f"[FAIL] {name}: JSON schema NOT recognized — "
                  f"load_gt_skeleton will return EMPTY!"); problems += 1
            continue

        # Map names, check core coverage
        mapped = {}
        for jn, coords in gt.items():
            clean = jn.split(":")[-1].strip()
            internal = next((v for k, v in MIXAMO_MAP.items()
                             if k.lower() == clean.lower()), clean)
            mapped[internal] = coords
        core = {k: v for k, v in mapped.items() if k in CORE_JOINTS}
        if len(core) < 12:
            issues.append(f"only {len(core)}/14 core joints matched "
                          f"(unmapped names: "
                          f"{[n for n in mapped if n not in CORE_JOINTS][:5]})")

        # Frame check
        if core:
            arr = np.array(list(core.values()), dtype=float)
            bmin, bmax = mesh.bounds
            m = 0.1 * np.linalg.norm(bmax - bmin)
            out = ((arr < bmin - m) | (arr > bmax + m)).any(axis=1).sum()
            if out > 0:
                issues.append(f"{out}/{len(arr)} joints OUTSIDE mesh bbox "
                              f"(coordinate frame mismatch!)")

        if issues:
            print(f"[ISSUE] {name}: " + "; ".join(issues)); problems += 1

    print(f"\n{'='*60}")
    print(f"Checked {len(meshes)} characters — {problems} with problems.")
    if problems == 0:
        print("Dataset is CLEAN. Safe to run experiments.")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else r"D:\HumanRig_Test_Set")