# A Mathematical Approach for Auto-Rigging 3D Models Using 2D MediaPipe

[Example.webm](https://github.com/user-attachments/assets/e7130336-2682-4382-9a6c-953cdf2357d4
)






![Auto-rigging Banner](banner.png)

[![arXiv](https://img.shields.io/badge/arXiv-Pending-b31b1b.svg)](https://arxiv.org/)
# VoidX — Training-Free Volumetric Auto-Rigging

![teaser](docs/teaser.png)

**VoidX rigs a T-posed humanoid mesh with a 14-joint skeleton in ~3–5 seconds — no training, no datasets, no learned weights.**

It renders the mesh from 8 orthographic viewpoints, detects 2D pose landmarks with MediaPipe Holistic, lifts them to 3D with closed-form trigonometry, then anchors every joint to the *volumetric center* of its limb using raycasting.

```
mesh.obj ──▶ 8 orthographic renders ──▶ MediaPipe 2D landmarks
          ──▶ Phase 1: multi-view 3D lifting (closed-form trig)
          ──▶ Phase 2: raycast volumetric centering
          ──▶ 14-joint skeleton (JSON)
```

---

## Results at a glance

Evaluated on a 73-character humanoid pilot set (see [Dataset](#dataset-format) and [Limitations](#known-limitations)):

| Metric | Value |
|---|---|
| CD-J2J (joint Chamfer, % of bbox diagonal) | **4.19 ± 1.38** |
| CD-J2B (joint–bone Chamfer) | 3.62 ± 1.39 |
| CD-B2B (bone–bone Chamfer) | 3.18 ± 1.24 |
| PCK@τ (τ = 2.5 / 5 / 7.5 / 10% diag) | 26.6 / **70.4** / 90.8 / 94.5 % |
| VCE — ours vs. reference | 0.211 vs 0.217 (paired t, p = 0.60) |
| Runtime per character | ~3–5 s (CPU, session-dependent) |

**Per-joint behavior is bimodal:** core & lower-body joints (hips, pelvis, knees, neck) reach 89–97% PCK@5%, while extremities (shoulders, elbows, wrists, ankles) reach 36–60%. See [Known limitations](#known-limitations).

Monte-Carlo verification of the estimation-theory analysis (CRLB for multi-view depth) matches theory at ratio ≈ 1.00 for every angle configuration — see `results/theorem_verification.json` after running.

---

## Installation

Tested on **Windows 11, Python 3.11**. Should work on Linux/macOS with a display (see [Troubleshooting](#troubleshooting)).

```bash
git clone https://github.com/<you>/voidx-autorigging.git
cd voidx-autorigging
pip install numpy scipy trimesh open3d mediapipe opencv-python plotly
```

> MediaPipe and Open3D are **required** — the pipeline fails fast with install instructions if either is missing. There is intentionally no synthetic fallback.

---

## Dataset format

```
<dataset_root>/
├── char1/
│   ├── model.obj          # T-posed humanoid mesh
│   └── bone_3d.json       # reference skeleton (optional but recommended)
├── char2/
│   ├── model.obj
│   └── bone_3d.json
└── ...
```

- **Mesh:** `.obj` (`.glb` also handled). One mesh per subfolder.
- **Reference skeleton** `bone_3d.json`: flat JSON dict `{"JointName": [x, y, z], ...}` in the **same coordinate frame as the mesh**. Mixamo joint names (e.g. `LeftShoulder`, `Hips`) are mapped automatically; internal names (`shoulder_L`, `pelvis`) pass through.
- Characters must be **T-posed humanoids** — the 2D detector is a human pose model. Non-humans will produce empty detections.

**Validate before running** (catches schema, naming, and coordinate-frame problems):

```bash
python test_path.py path/to/dataset
```

You want `Dataset is CLEAN. Safe to run experiments.`

---

## Usage

### 1. Rig a single mesh

```bash
python voidx_pipeline.py path/to/char1/model.obj
# → VOIDX_ULTIMATE_DNA.json  (14 joints + metadata)
```

### 2. Full evaluation suite

```bash
python voidx_experiment.py --mesh_dir path/to/dataset --output_dir ./results
```

Runs, in order:
1. **Theorem verification** — Monte-Carlo check of the multi-view estimation bounds (~2 min)
2. **Main experiment** — pipeline + all metrics on every character
3. **Ablations** — 5 ablation studies × 13 configurations across 10 meshes (~15–25 min)
4. **Runtime pass** — dedicated per-phase timing on 20 meshes

Flags: `--skip_theorems`, `--skip_ablations`. Expect **~10 minutes total** for ~70 characters on a typical laptop.

### 3. Outputs

| File | Contents |
|---|---|
| `results/main_results.csv` / `.json` | Per-character metrics + aggregate |
| `results/preds/*.json` | **Frozen per-character predictions** (see below) |
| `results/ablations.json` | Ablation tables (per-mesh + aggregated mean±std) |
| `results/runtime.json` | Per-phase timings |
| `results/theorem_verification.json` | CRLB verification tables |

### 4. Frozen predictions — analyze without re-running

The pipeline output for every character is saved once to `results/preds/`. **All analysis scripts run on these frozen predictions — never on the pipeline.** Changing a metric, tolerance, or study never requires re-rendering:

```bash
python calibrate_tolerance.py   # PCK τ-curve + reference-pivot statistics
python pck_breakdown.py         # per-joint PCK@5% vs. reference placement
python gt_vce.py                # is the reference itself centered?
python vce_breakdown.py         # per-joint centering: reference vs. ours
```

Each finishes in minutes. If you modify the **pipeline** (not the metrics), delete `results/` and re-run the full suite — never mix predictions from two pipeline versions.

### 5. Visualization

Interactive 3D QA viewer (Plotly HTML):

```bash
python view_rig.py results/preds/char2.json        # one character
python view_rig.py --batch results/preds           # everything
python view_rig.py --worst 3 --opacity 0.35 --out ./figures_pretty
```

**Legend:** pale blue = mesh · **red dots / green bones** = VoidX prediction · **blue diamonds** = reference skeleton · cyan × = computed cross-section centers (limb joints only — undefined at junctions) · orange dashed = prediction→reference error. Hover any joint for its centering error, reference distance, and PCK@5% pass/fail.

---

## Metrics

- **CD-J2J / CD-J2B / CD-B2B** — symmetric Chamfer distances (joint↔joint, joint↔bone, bone↔bone), normalized by bbox diagonal, following the definitions of Xu et al., *RigNet* (SIGGRAPH 2020). Restricted to the 14 core joints present in both prediction and reference — **not comparable** to full-skeleton protocols.
- **PCK@τ** — a predicted joint is *correct* if within τ × bbox diagonal of its Hungarian-matched reference joint. We use absolute tolerance because reference pivots follow animator placement conventions: in our set their median distance to the mesh surface is **1.34% of diagonal** and 59% are measured outside the mesh, so reference-derived tolerances would measure placement convention rather than prediction error.
- **VCE (Volumetric Centering Error)** — radial distance of a joint from the center of its local cross-section (8-ray cast, opposite-pair midpoints), normalized by local diameter. Defined for **limb joints only**; junction geometry (shoulder armpit/deltoid merge) breaks the cross-section model.

---

## Known limitations

We state these explicitly rather than let you discover them:

1. **Reference rigs are auto-generated, not artist ground truth.** `bone_3d.json` files come from an industry auto-rigger (Mixamo-style template fitting). Agreement metrics measure *convention agreement*, not truth. The VCE parity analysis (ours 0.211 vs. reference 0.217, p = 0.60) quantifies this.
2. **No external baselines** (Pinocchio, RigNet) are included. Comparison hooks exist but were not run for this pilot.
3. **Humanoid-only.** The 2D detector is a human pose model; non-human characters yield no skeleton.
4. **Bimodal per-joint performance.** Extremities (36–60% PCK@5%) lag core/lower-body (89–97%), due to (a) animator convention at skin-level pivots and (b) 2D detection limits on thin limbs in T-pose renders.
5. **Junction centering is open.** Shoulders are the worst region for *both* our skeleton and the reference; cross-section-based centering is not defined there.
6. **Runtime varies ±2× across sessions** on identical hardware; the dedicated runtime pass is the number to report, with hardware stated.
7. **73 characters is a pilot study**, not a benchmark.

---

## Troubleshooting

| Symptom | Meaning / fix |
|---|---|
| `ImportError: MediaPipe is required...` | `pip install mediapipe` — intentional fail-fast, no fallback |
| `XNNPACK runtime` errors after many runs | Fixed in current code (MediaPipe sessions are closed in `destroy()`). Update if you have an old checkout. |
| `Warning: no front image captured` during ablations | **Expected** — the N=1 configuration scans only 90°, so no frontal render exists. |
| `GT skeleton EMPTY for ...` warning | The `bone_3d.json` schema wasn't recognized — check it's a flat `{name: [x,y,z]}` dict. |
| Crashes or blank renders on a headless server | The renderer needs a display context. Use a virtual display (`xvfb-run` on Linux) or run on a desktop session. |
| Two pipeline processes writing to one folder | Don't — temp render files (`temp_cortex_*.png`) live in the working directory and will race. |
| Numbers differ slightly on another machine | Expected: MediaPipe/rendering are not bit-reproducible across environments. Report environment + versions. |

---

## Repository layout

```
voidx_pipeline.py      # Phase 1 + Phase 2 pipeline (the method)
voidx_experiment.py    # Full evaluation suite (main / ablations / runtime)
voidx_metrics.py       # CD-*, PCK, VCE, ED metrics
voidx_ablation.py      # 5 ablation studies
voidx_theorems.py      # Estimation-theory analysis + Monte-Carlo verification
test_path.py           # Dataset validator
view_rig.py            # Interactive 3D QA / figure viewer
calibrate_tolerance.py # PCK τ-curve + reference-pivot statistics
pck_breakdown.py       # Per-joint PCK analysis
gt_vce.py              # Reference-skeleton centering analysis
vce_breakdown.py       # Per-joint VCE: reference vs. prediction
```

---

## ✒️Citation 
     @software{Krishnanand2026autorigging,
      title={A Mathematical Approach for Auto-Rigging 3D Models Using 2D MediaPipe}, 
      author={Krishnanand},
      year={2026}
    }
```

## 📜 License 

### This project is licensed under the [MIT License](LICENSE).
---

---

## Before you push — 5-minute checklist

1. **Take the hero screenshot.** Open one `rig_qa_*.html` (a good character), screenshot it, save as `docs/teaser.png`. If you can, use the one showing the floating blue reference joints vs. your centered green skeleton — it's the single most intriguing image this project produces and it makes the README's metric rationale visual instantly.
2. **Fill the placeholders:** your GitHub username, author names, arXiv link when it exists.
3. **Add a `requirements.txt`** with the seven packages and their versions (`pip freeze | findstr "numpy scipy trimesh open3d mediapipe opencv plotly"` on your machine gives exact pins — pin them, it's the reproducibility-friendly move).
4. **Decide on the dataset.** If licensing allows including it, add a `data/` note; if not, add one line in the Dataset section saying "dataset available on request" — do **not** upload meshes you don't have rights to redistribute.
5. **Delete `viewer.py`** (the old hardcoded-path one) — `view_rig.py` fully replaces it, and dead scripts confuse GitHub visitors.

One last thing worth saying: notice that the Limitations section isn't modesty — it's the section that makes the repo *credible*. Anyone who runs your code will discover every one of those seven points anyway; the README stating them first is the difference between "the authors know their system" and "the authors hid the flaws." You spent three weeks learning that lesson in your metrics — the README is where it becomes public. 🎯
---

# 📬 Contact
For questions, collaboration, or feedback, please reach out:
Email: nanddyasty5@gmail.com

GitHub: [@KRISH9621](https://github.com/KRISH9621/MVP-rig)