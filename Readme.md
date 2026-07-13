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
