"""check_gt_status.py — Are the bone_3d.json files fixed, unfixed, or unknown?"""
import argparse, json
from pathlib import Path

def negate(obj):
    out = {}
    for name, c in obj.items():
        if isinstance(c, list) and len(c) == 3:
            out[name] = [c[0], c[1], -c[2]]
        elif isinstance(c, dict) and "z" in c:
            out[name] = {"x": c["x"], "y": c["y"], "z": -c["z"]}
        else:
            out[name] = c
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    args = ap.parse_args()
    root = Path(args.dataset_dir)
    backup_dir = root / "bone_3d_backup_mirrored"

    stats = {"FIXED": 0, "NOT_FIXED": 0, "NO_BACKUP": 0, "UNKNOWN": 0}
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name == backup_dir.name:
            continue
        gt_file = d / "bone_3d.json"
        if not gt_file.exists():
            continue
        current = json.load(open(gt_file))
        backup_file = backup_dir / f"{d.name}_bone_3d.json"
        if not backup_file.exists():
            stats["NO_BACKUP"] += 1
            print(f"  {d.name}: NO BACKUP (original fix never ran here)")
            continue
        original = json.load(open(backup_file))
        if current == original:
            stats["NOT_FIXED"] += 1
        elif current == negate(original):
            stats["FIXED"] += 1
        else:
            stats["UNKNOWN"] += 1
            print(f"  {d.name}: UNKNOWN state (manual edits?)")
    print(f"\nSUMMARY: {stats}")
    print("NOT_FIXED = fix never applied OR applied twice (canceling out)")
    print("FIXED     = correctly mirrored, ready to use")

if __name__ == "__main__":
    main()