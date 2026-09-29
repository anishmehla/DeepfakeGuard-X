# DeepfakeGuard-X: Team Workflow & Architecture Manual

> **Welcome to the team!** This document outlines our project structure, execution steps, git workflow, and coding standards. Read through this before opening a branch.

---

## 1. Directory Structure

```text
DeepfakeGuard-X/
├── configs/                     # Hyperparameter configurations (YAML)
│   └── base_config.yaml         # Learning rates, batch size, frame counts, seeds
│
├── data/                        # Datasets (NEVER commit raw or cropped images)
│   ├── splits/                  # Tracked: CSVs (metadata, video IDs, labels, source IDs)
│   ├── raw/                     # Untracked: Raw MP4 video files (FF++, Celeb-DF)
│   └── crops/                   # Untracked: Extracted 256x256 face images
│
├── src/                         # All project logic and runnable modules
│   ├── __init__.py
│   ├── data.py                  # Step 1: Disjoint splits + Face extraction
│   ├── dataset.py               # Step 2: PyTorch Dataset, DataLoader, transforms
│   ├── model.py                 # Step 3: timm backbones + classification head
│   ├── train.py                 # Step 4: Training loop with AMP and W&B logging
│   ├── evaluate.py              # Step 5: Metrics (AUC/EER/pAUC) + Video pooling
│   ├── stress.py                # Step 6: JPEG/blur/resize degradation sweeps
│   └── calibrate.py             # Step 7: Temperature scaling for uncertainty
│
├── checkpoints/                 # Untracked: Saved model weights (.pt files)
├── reports/                     # Tracked: Output metrics (JSON) and final plots (PNG)
│   └── figures/
│
├── .gitignore                   # Strict ML exclusion rules
├── requirements.txt             # Pinned package dependencies
└── README.md                    # Project overview
```

---

## 2. Directory & File Breakdown

| File / Folder | Role & Purpose | What It Contains / How to Run |
| :--- | :--- | :--- |
| `configs/base_config.yaml` | Experiment control | Central settings: epochs, learning rate, batch size, frames per video. |
| `data/splits/` | Split records | CSVs with columns: `video_id, video_path, source_id, manipulation_type, label`. Only metadata lives here. |
| `data/raw/` | Raw data | Stored on local disk or external storage. Untracked by git. |
| `data/crops/` | Extracted faces | Cropped 256×256 facial PNGs organized by `video_id`. Untracked by git. |
| `src/data.py` | Data preparation | **Runnable:** Generates disjoint splits and extracts 16 face crops/video with a 20% margin. |
| `src/dataset.py` | PyTorch loading | Defines `DeepfakeDataset` and Albumentations augmentations. |
| `src/model.py` | Network backbone | Wraps pretrained architectures (e.g., EfficientNet-B4) with a binary logit output. |
| `src/train.py` | Training runner | **Runnable:** Executes the training loop on FF++ using BCE loss and AdamW. |
| `src/evaluate.py` | Core evaluation | **Runnable:** Computes frame/video-level AUC, pAUC (FPR <= 0.05), EER, and AP. |
| `src/stress.py` | Robustness sweep | **Runnable:** Sweeps JPEG compression (Q = 90 to 30), blur, and downscaling on Celeb-DF. |
| `src/calibrate.py` | Uncertainty head | **Runnable:** Tunes temperature parameter T > 0 to minimize calibration error (ECE). |

---

## 3. Environment Setup (One-Time Setup)

Every team member should use Python 3.10 with an isolated environment.

### Option A: Using Conda (Recommended)

```bash
conda create -n deepfake-guard python=3.10 -y
conda activate deepfake-guard

# Install PyTorch with CUDA support (adjust for your hardware if needed)
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# Install project packages
pip install -r requirements.txt
```

### Option B: Using Python `venv`

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Git & Branching Workflow (Step-by-Step)

We follow a strict **feature-branch + pull request (PR)** routine. Nobody commits directly to `main`.

```text
       feat/data-splits
         ┌────────────┐
         │ Commit 1,2 │
         └─────┬──────┘
               │ Open PR (Closes #X) & Request Review
               ▼
main ──────────────────────────────▶ (Squash & Merged by teammate)
```

### Step 1: Assign a Task on the Board
1. Open the GitHub **Project Board**.
2. Pick an open card from the **Backlog**.
3. Assign yourself and drag it to **In Progress**.
4. Note the issue number (e.g., `#2`).

### Step 2: Update Your Local `main`
Before branching, always pull the freshest code:

```bash
git checkout main
git pull origin main
```

### Step 3: Create Your Feature Branch
Use descriptive prefixes (`feat/`, `fix/`, `eval/`):

```bash
git checkout -b feat/<short-topic>
# Examples:
# git checkout -b feat/disjoint-splits
# git checkout -b feat/timm-backbone
# git checkout -b feat/metrics-eer
```

### Step 4: Write & Test Code
* Work inside the designated file in `src/`.
* Test your changes locally to ensure the code executes without errors.

### Step 5: Check Git Hygiene (Crucial)
Before staging files, verify you aren't accidentally tracking images, videos, or model checkpoints:

```bash
git status
```

* **Alert:** If you see any `.mp4`, `.png`, `.jpg`, `.pt`, `.pth`, or `wandb/` paths listed under untracked files, **DO NOT ADD THEM**. Ensure your `.gitignore` is active.
* Stage only Python files, configs, and documentation:

```bash
git add src/ configs/ requirements.txt
git commit -m "feat: implement 20% margin bounding box face cropping"
```

### Step 6: Push & Open a Pull Request
1. Push your branch to GitHub:

```bash
git push -u origin feat/<short-topic>
```

2. Open your browser and go to your GitHub repository.
3. Click the green **Compare & pull request** button.
4. In the PR description, link the issue to automate board movement:

```text
Closes #<issue_number>
```

*(e.g., `Closes #2`. This automatically moves the card to **Done** when merged.)*

5. Request a review from at least one teammate.

### Step 7: Review, Squash & Merge
1. The reviewer pulls the branch or inspects the diff on GitHub.
2. If the code runs and meets requirements, the reviewer clicks **Approve**.
3. Select **Squash and merge**, then delete the remote branch.

---

## 5. End-to-End Pipeline Execution Guide

Run all modules as Python packages from the repository root using the `-m` flag.

### Step 1: Prepare Splits & Crop Faces
* **File:** `src/data.py`
* **What it does:**
  * Maps manipulated videos back to parent `source_id`.
  * Splits identities: 80% train, 20% validation.
  * Asserts zero overlap: `assert len(set(train_ids) & set(val_ids)) == 0`.
  * Samples 16 frames uniformly per clip.
  * Expands detected face bounding boxes by **20% outer margin** to capture jawline and blending seams.
  * Resizes crops to 256×256 PNGs.
* **Run command:**

```bash
python -m src.data
```

### Step 2: Train In-Domain Baseline (FF++)
* **File:** `src/train.py`
* **What it does:**
  * Reads training crops via `src/dataset.py`.
  * Initializes pretrained `efficientnet_b4` via `src/model.py`.
  * Trains using `BCEWithLogitsLoss` and AdamW (`lr = 1e-4`) with Cosine Annealing.
  * Uses Automatic Mixed Precision (`torch.cuda.amp.autocast`).
  * Logs metrics to Weights & Biases (`wandb`).
  * Saves best checkpoint to `checkpoints/baseline_best.pt`.
* **Run command:**

```bash
python -m src.train --config configs/base_config.yaml
```

### Step 3: Run Zero-Shot Cross-Generator Evaluation
* **File:** `src/evaluate.py`
* **What it does:**
  * Evaluates `checkpoints/baseline_best.pt` directly on unseen Celeb-DF crops without target fine-tuning.
  * Computes frame-level and video-level (Mean and Top-3 pooling) metrics:
    * ROC-AUC
    * Partial AUC (FPR <= 0.05)
    * Equal Error Rate (EER)
    * Average Precision (AP)
  * Logs the cross-generator drop: `Delta_AUC = AUC_FF++ - AUC_Celeb-DF`.
* **Run command:**

```bash
python -m src.evaluate --checkpoint checkpoints/baseline_best.pt --split data/splits/celebdf_test_split.csv
```

### Step 4: Run Social Media Stress Sweeps
* **File:** `src/stress.py`
* **What it does:**
  * Runs inference across Celeb-DF under controlled post-processing degradations:
    * JPEG Compression: Q in `[90, 70, 50, 30]`
    * Gaussian Blur: sigma in `[1, 3, 5]`
    * Spatial Rescaling: Downscale by 0.5x and 0.25x, then upscale back.
  * Exports performance degradation records to `reports/stress_results.json`.
* **Run command:**

```bash
python -m src.stress --checkpoint checkpoints/baseline_best.pt
```

### Step 5: Train Robust Model & Calibrate Uncertainty
* **Files:** `src/train.py` & `src/calibrate.py`
* **What it does:**
  * Retrains the backbone with Albumentations compression/blur augmentations active during training batches.
  * Learns post-hoc temperature parameter `T > 0` on validation logits (`z / T`) to reduce Expected Calibration Error (ECE).
  * Re-runs Step 4 on `checkpoints/robust_best.pt` to prove robustness recovery under compression.
* **Run command:**

```bash
python -m src.train --config configs/robust_config.yaml
python -m src.calibrate --checkpoint checkpoints/robust_best.pt
```

---

## 6. Daily Communication Standard

Post a 60-second update in your team group chat before every work session:

```text
[Daily Standup - Your Name]
1. Yesterday: Completed Issue #2 (Disjoint split generator tested).
2. Today:     Working on Issue #3 (Face cropping pipeline).
3. Blockers:  None / (e.g., need disk space for raw Celeb-DF files).
```
