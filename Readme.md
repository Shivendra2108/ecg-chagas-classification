# Chagas Disease Detection from ECG — Baseline Pipeline

Baseline 1D-CNN for detecting Chagas disease from 12-lead ECG signals, built for the [PhysioNet Challenge 2025](https://physionet.org/) using the CODE-15% dataset.

## Overview

Chagas disease is often asymptomatic in early stages but can cause serious cardiac damage over time. This project explores whether a lightweight 1D-CNN trained on raw 12-lead ECG signals can flag Chagas-positive cases, as a step toward low-cost, ECG-based screening — with an eventual target of running on edge/FPGA hardware for deployment in resource-constrained settings.

This repo currently contains a **clean, leak-free baseline pipeline** — the goal at this stage is a working, bug-free foundation before scaling up data and model complexity.

## Dataset

- **Source:** CODE-15% (subset of the CODE dataset), converted to WFDB format
- **Labels:** `code15_chagas_labels.csv` — ground-truth Chagas labels per `exam_id`
- **Signal format:** 12-lead ECG, variable length, fixed to 2000 timesteps (5s @ 400Hz) via truncation/zero-padding
- **Current scale:** 300 samples (pipeline validation stage — scaling to 1000+ next)

## Pipeline

1. **Load** ECG signals from WFDB `.dat`/`.hea` files
2. **Label** each exam from the CSV (not from ECG header comments — see [Design Decisions](#design-decisions))
3. **Split** train/test (80/20, stratified) — done *before* normalization to prevent leakage
4. **Normalize** using train-set statistics only, applied identically to test set
5. **Train** a 3-block 1D-CNN with `BCEWithLogitsLoss`
6. **Evaluate** on accuracy, AUC-ROC, confusion matrix, and per-class precision/recall

## Model

`AdaptiveAvgPool1d` means the model isn't tied to a fixed input length. Sigmoid is applied only at inference time — `BCEWithLogitsLoss` handles it internally during training for numerical stability.

## Results (300 samples)

| Metric | Value |
|---|---|
| Test Accuracy | `<fill in>` |
| AUC-ROC | `<fill in>` |

Baseline expectation at this sample size is AUC ~0.65–0.72; 0.80+ will need class-imbalance handling and more data (see Roadmap).

## Design decisions (bugs this baseline deliberately avoids)

- **Labels from CSV, not ECG header comments** — header-based label extraction was tried earlier and produced degenerate all-one labels
- **Split before normalization** — prevents test-set statistics leaking into training
- **`BCEWithLogitsLoss` with no `Sigmoid` in the model** — avoids the double-sigmoid bug that silently corrupts gradients
- **`torch.sigmoid()` applied only at prediction time**, never during training

## Setup

```bash
pip install torch numpy pandas scikit-learn matplotlib wfdb
```

Update the paths at the top of the notebook to point to your local copy of the processed WFDB records and the labels CSV:
```python
DATA_PATH   = "../processed_data"
LABELS_PATH = "../data/code15_chagas_labels.csv"
```

## Roadmap

| Phase | What | Why it matters |
|---|---|---|
| 2 | Bandpass filter (0.5–40Hz), resample to common Hz | CODE-15% is 400Hz, PTB-XL is 500Hz — need consistent input |
| 2 | `pos_weight` for class imbalance | Highest-impact fix for AUC at current scale |
| 2 | Scale to 1000+ samples via `DataLoader` | Better generalization estimates |
| 3 | Cross-dataset eval (train CODE-15%, test SaMi-Trop) | Tests real-world robustness — the core research gap |
| 3 | GradCAM on 1D signals | Explainability — publishable and hireable |
| 4 | INT8 quantization | Required before FPGA deployment |
| 4 | FPGA deployment via HLS4ML or FINN | Edge-deployable inference — the differentiated contribution |
| 5 | HuggingFace Spaces demo | Lets reviewers/recruiters interact without running code |

## Repository structure
.
baseline2.ipynb        # Clean baseline pipeline (this README describes this notebook)
data/                  # Labels CSV (not tracked — see .gitignore)
processed_data/        # WFDB signal files (not tracked)
models/                # Saved model weights

