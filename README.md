# Effect of Image Manipulation on CNNs and Vision Transformers

This repository measures how different models are affected by faces under the **Thatcher illusion** versus a **Non‑Semantic Local Manipulation (NSLM)** control using a **single, unified dataset layout**.  
For each model we compute, per layer/block:

- **Percentage Ratio (PR)**: ((DU/DI − 1) × 100)  
  where **DU** = L2 distance between *upright_normal* and *effect upright* features. 
  and **DI** = L2 distance between *inverted_normal* and *effect inverted* features.

  This helps in understanding how much stronger the model's feature distance is for upright manipulations **DU** compared to inverted manipulations **DI**. The one liner to remember.`A large positive PR means the model distinguishes upright manipulations more than inverted ones.`
- **Consistency (CD)**: percentage of subjects where **DU > DI**.

Across all subject, how consistently upright manipulations produce larger differences than inverted ones. The one liner here is, `Higher CD means reliable effect sensitivity.`

Each run produces:

- A flat **`saliency/`** folder with **six** overlays:  
  `normal_upright.png`, `normal_inverted.png`, `thatcher_upright.png`, `thatcher_inverted.png`, `NSLM_upright.png`, `NSLM_inverted.png`
- Two plots:
  - `PercentRatio_Consistency_comparison_BOTHNORMALIZED.png` (For easier comparison between different models)
  - `PercentRatio_Consistency_comparison_RAW.png` (For in-depth detail into each particular model and how it's affected by a certain manipulation)
- A compact CSV: `metrics_layerwise.csv` (layer/block‑wise PR & CD)

All four models share the **same CLI** and **the same outputs**. You can run any one model or all sequentially.

---

## Table of Contents

- [Folder Structure](#folder-structure)
- [Environment & Dependencies](#environment--dependencies)
- [Models & Data](#models--data)
- [Quick Start](#quick-start)
- [Command‑line Options](#command-line-options)
- [Outputs](#outputs)
- [How It Works](#how-it-works)
- [Extending: Add Your Own Model](#extending-add-your-own-model)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## Folder Structure

> A single `data/` root that shares normal images and stores effect variants separately.

```
data/
├─ upright_normal/             # shared normal (upright) across all effects
├─ inverted_normal/            # shared normal (inverted) across all effects
├─ thatcher/
│  ├─ upright/                 # thatcherized version of upright_normal
│  └─ inverted/                # thatcherized version of inverted_normal
└─ NSLM/
   ├─ upright/                 # NSLM local manipulation of upright_normal
   └─ inverted/                # NSLM local manipulation of inverted_normal
```

**Index alignment is critical:** `upright_normal`, `inverted_normal`, `thatcher/upright`, `thatcher/inverted`, `NSLM/upright`, `NSLM/inverted` must all contain images for the **same subjects in the same sorted order**. The code assumes `sorted(glob("*"))[i]` corresponds to subject *i* across all these folders.

Parent folder layout structure:

```
image_manipulation/
├─ code/
│  ├─ main.py               # unified CLI (run vgg16, vggface, vit, vitface, or run-all)
│  ├─ vgg_face_dag.py       # model sources (if you want to inspect)
│  ├─ vit_face.py
│  └─ __pycache__/
├─ data/                    # (see unified layout above)
├─ models/
│  ├─ vgg16_mcn.pth
│  ├─ vgg_face_dag.pth
│  └─ facevit_pretrained_8.pth
└─ results/
   ├─ vgg16_analysis/
   ├─ vggface_analysis/
   ├─ vit_analysis/
   └─ vitface_analysis/
└─ requirements.txt
```

---

## Environment & Dependencies

- **Python:** 3.9+ (3.10/3.11 OK)
- **PyTorch:** 2.0+ (CPU works; CUDA strongly recommended)
- **CUDA:** optional, but recommended for speed

Install with `pip` (use a virtual env / conda if you like):

```bash
# Pick the correct torch build (CUDA/CPU) for your machine:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install the packages for the environment using the requirements file
pip install -r /path/to/requirements.txt
```

Saliency packages used by the code:

```bash
# For ViT / ViT-Face saliency (pytorch-grad-cam)
pip install pytorch-grad-cam
```

> **Note about CNN saliency (VGG-16/VGG-Face):** the code imports `from gradcam import GradCAM` and `from gradcam.utils import visualize_cam`.  
> If your environment doesn’t already provide this `gradcam` module, either:
> - add your local `gradcam` implementation to `PYTHONPATH`, **or**
> - adapt those imports to use `pytorch-grad-cam` (like the ViT path) if you prefer.  
> The rest of the pipeline is unaffected.

---

## Generating Thatcherized and NSLM images

**This reposity was used for thatcherization of images**

https://github.com/Erfaniaa/thatcher-effect-dataset-generator

> Clone the reposity in a custom environment to avoid package mismatch. The [non-semantic.py](code/non_semantic.py) file will only run in this custom environment made for this repository as it uses some of the tools provided in that repo such as `facial_landmark_detection.py`. Place the [non-semantic.py](code/non_semantic.py) file in the newly folder `thatcher-effect-dataset-generator` based on the instructions of that repo, and it should run to generate the NSLM images.

## Models & Data

**Checkpoints (place in `models/`):**

- `vgg16_mcn.pth` — VGG-16 with your custom weights. If it contains an `'average'` entry, that mean image is used for centering; otherwise zeros are used. 
Download link: https://www.vlfeat.org/matconvnet/pretrained/ (this was used as the study previously being looked at used this as well. (https://github.com/visionlabiisc/CNN-perception/tree/master/exp01_thatcher_effect))
- `vgg_face_dag.pth` — VGG-Face DAG weights. The model’s `meta['mean']` is used for centering.
Download link: https://www.kaggle.com/datasets/twanghcmut/vgg-face-dag
- `facevit_pretrained_8.pth` — ViT-Face checkpoint.
Download link : https://drive.usercontent.google.com/download?id=1OZRU430CjABSJtXU0oHZHlxgzXn6Gaqu&export=download&authuser=0&confirm=t&uuid=201c7322-a4f6-41c5-9a3a-ecc157587356&at=AN8xHopRgiNA5ZDQM9uRFtUDvVSN:1753792175357

**Dataset** — see [Folder Structure](#folder-structure).  
All six folders must be **index-aligned** across subjects.

---

## Quick Start

From `image_manipulation/code/`:

```bash
# VGG-16 only
python main.py --model vgg16 --data-root ../data

# VGG-Face only
python main.py --model vggface --data-root ../data

# ViT (timm) only
python main.py --model vit --data-root ../data

# ViT-Face only
python main.py --model vitface --data-root ../data

# Run all models sequentially
python main.py --model run-all --data-root ../data
```

Faster on GPU + mixed precision (AMP), and skip saliency overlays:

```bash
python main.py --model vit --data-root ../data --device cuda --amp --skip-cam
```

Use a different timm architecture:

```bash
python main.py --model vit --data-root ../data --vit-arch vit_base_patch16_224
```

Override model/output paths (examples):

```bash
python main.py --model vgg16 --data-root ../data   --vgg16-model ../models/vgg16_mcn.pth   --out-vgg16 ../results/vgg16_analysis
```

---

## Command‑line Options

```text
--model {vgg16,vggface,vit,vitface,run-all}   Which models to run (required)

# Paths
--data-root          ../data                   # unified dataset root (REQUIRED; new layout only)
--vgg16-model        ../models/vgg16_mcn.pth
--vggface-model      ../models/vgg_face_dag.pth
--vitface-model      ../models/facevit_pretrained_8.pth

# Output directories (each model writes to its own folder)
--out-vgg16          ../results/vgg16_analysis
--out-vggface        ../results/vggface_analysis
--out-vit            ../results/vit_analysis
--out-vitface        ../results/vitface_analysis

# Runtime
--img-size           224          # for VGG* models
--vitface-img-size   112          # for ViT-Face
--vit-arch           vit_base_patch8_224   # timm name for ViT
--device             cuda|cpu
--amp                              # enable autocast mixed precision
--skip-cam                         # do not write saliency overlays
--cam-per-class      1            # how many images per class for saliency (suffixes if >1)
```
---

## Outputs

Each model writes to its own analysis folder under `results/`:

> The original results of the project are provided in the [results](results) folder. It can be used for comparison to check for correct functioning of code. 
```
results/<model>_analysis/
├─ saliency/
│  ├─ normal_upright.png
│  ├─ normal_inverted.png
│  ├─ thatcher_upright.png
│  ├─ thatcher_inverted.png
│  ├─ NSLM_upright.png
│  └─ NSLM_inverted.png
├─ metrics_layerwise.csv
├─ PercentRatio_Consistency_comparison_BOTHNORMALIZED.png
└─ PercentRatio_Consistency_comparison_RAW.png
```

- **BOTHNORMALIZED.png**: PR & CD min–max normalized per dataset to reveal trends.  
- **RAW.png**: PR (left axis) and CD (right axis) in original units.  
- **CSV** columns: `dataset, layer, PR_mean, CD`.

---

## How It Works

1. **Preprocessing & Centering**
   - **VGG-16**: subtracts a mean image from the checkpoint (`'average'` if present, else zeros).
   - **VGG-Face**: uses `model.meta['mean']` for centering.
   - **ViT (timm)**: uses `model.default_cfg['mean/std']` with a resize to its input size.
   - **ViT-Face**: normalized to ImageNet mean/std, default input size **112×112** (configurable via `--vitface-img-size`).

2. **Feature Extraction**
   - **VGG-16**: flattened activations from all conv & FC layers.
   - **VGG-Face**: flattened activations via lightweight forward hooks (mirrors the original repo style).
   - **ViT (timm)**: post‑`norm1` token outputs from every transformer block (drop CLS, flatten patches).
   - **ViT-Face**: original forward with hooks capturing **CLS token** at each transformer layer (matches your working code).

3. **Metrics** (per layer/block)
   - For **Thatcher**: compute DU (upright_normal vs. thatcher/upright) and DI (inverted_normal vs. thatcher/inverted).
   - For **NSLM**: same as above, substituting thatcher folders with NSLM folders.
   - Compute **PR** and **CD** across all subjects; report layerwise means for PR and CD.

4. **Saliency**
   - **VGG-16 / VGG-Face**: Grad‑CAM overlays on the last conv layer (flat `saliency/` folder).
   - **ViT**: `pytorch-grad-cam` with a reshape transform on the final block.
   - **ViT-Face**: attention roll‑up overlays (mean over heads, dynamic aggregation across layers). The heatmaps for ViT-face can be improved, the existing method doesn't 
   - Exactly **six** overlays are produced when `--cam-per-class 1`. Larger values add numeric suffixes (`_2`, `_3`, …).

---

## Extending: Add Your Own Model

1. Implement `load_<name>()` → returns the model (and any preprocessor/mean/target layer if needed).
2. Implement `feats_<name>(model, x)` → return a list of 1D tensors (one per layer/block).
3. Implement `compute_dataset_metrics_<name>()` following one of the existing templates:
   - CNN: `compute_dataset_metrics_cnn_from_lists(...)`
   - ViT: `compute_dataset_metrics_vit_from_lists(...)`
   - ViT-Face: `compute_dataset_metrics_vitface_from_lists(...)`
4. Implement `<name>_saliency_one(...)`. The existing saliency methods can be modified, depending on the type of model being used. Is explained in detail in the report. For transformers we adopt attention-based rollout/gradient method to obtain pixel-space maps. This work cautions that attention is not necessarily explanation in a causal sense. We use these maps primarily for comparative, model-to-model sanity checks rather than definitive mechanistic claims.
5. Add a `run_<name>_pipeline(args, folders_by_ds)` and a CLI branch in `main.py`.

Because plotting and CSV export are already standardized, your new model will automatically produce the same artifacts.

---

## Troubleshooting

- **`RuntimeError: mismatch counts [...] under new layout`**  
  One or more folders are missing or have different numbers of images. Ensure all relevant folders are index‑aligned (same subjects, same sorted order).

- **CNN saliency import error (`gradcam` not found)**  
  The VGG pipelines import `gradcam`/`visualize_cam`. If that module isn’t in your environment, either:
  - add your local `gradcam` package/module (as used in your previous projects), or
  - switch the imports to `pytorch-grad-cam` (like the ViT path).

- **CUDA out of memory (OOM)**  
  Try `--device cpu`, or disable AMP, or close other GPU processes. The code processes one image at a time, but feature tensors can still be large.

- **ViT reshape errors** (`Cannot reshape N patches into a square grid`)  
  Ensure your ViT input resolution matches the architecture’s patch configuration. Stick with the default `--vit-arch` unless you know you need a different one.

- **Saliency count not six**  
  Check `--cam-per-class`. `1` ensures exactly six files; larger values append suffixes.

---
