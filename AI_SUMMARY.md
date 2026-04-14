# AI Project Summary — Plant Leaf Segmentation & Anomaly Detection

## Project overview

This repository contains code, data and notebooks to segment plant leaves and extract masks/contours for downstream analysis (disease/anomaly detection). The primary dataset is derived from PlantVillage and additional raw folders under `data/raw`. The main deliverable is a reproducible preprocessing + segmentation pipeline that exports masks, edge maps and per-image summaries.

## Quick repo layout

- `data/raw/` — raw images organized by class/folder (PlantVillage subsets included)
- `data/processed/` — processed outputs, histograms, preprocessing_report.json
- `notebooks/segmentation.ipynb` — interactive pipeline and batch runner (now supports `SAMPLES_PER_FOLDER`)
- `src/preprocessing.py` — preprocessing helpers and transforms
- `results/segmentation/` — generated masks, visual outputs and JSON summaries
- `models/` — placeholder for trained models (if added)
- `requirements.txt` — Python dependencies

## Primary objective

- Produce reliable leaf segmentation masks and auxiliary outputs (Sobel/Canny, HSV mask, K-Means) to support visual inspection, annotation and later model training.

## Pipeline (high level)

1. Load image → resize to `RESIZE_TO`
2. Convert to grayscale / HSV
3. Edge & texture detection: Sobel, Canny
4. Segmentation heuristics: Otsu & adaptive thresholding
5. Color-based mask: HSV range + morphological cleanup
6. Color clustering: K-Means to identify leaf cluster
7. Postprocess masks, save RGB/PNG outputs and per-image JSON summary
8. Batch mode: process up to `SAMPLES_PER_FOLDER` images per class folder and write `one_per_folder_summary.json` (or aggregated summary files)

## How to run (quick)

1. Activate your virtual environment (example PowerShell):

   & "c:\Users\louay\OneDrive\Desktop\computer vision\venv\Scripts\Activate.ps1"

2. Install dependencies:

   pip install -r requirements.txt

3. Open `notebooks/segmentation.ipynb` in Jupyter and run cells, or run headless:

   jupyter nbconvert --to notebook --execute notebooks/segmentation.ipynb --inplace --ExecutePreprocessor.timeout=600

4. To test exactly 3 images per folder, set `SAMPLES_PER_FOLDER = 3` at the top of the notebook (the repo already includes this change).

## Outputs produced

- `<folder>_<image>_<idx>_mask_otsu.png`
- `<folder>_<image>_<idx>_mask_hsv.png`
- `<folder>_<image>_<idx>_mask_kmeans.png`
- `<folder>_<image>_<idx>_sobel.png`
- `<folder>_<image>_<idx>_canny.png`
- `<folder>_<image>_<idx>_leaf_*.png` (various mask-applied RGB images)
- `one_per_folder_summary.json` and per-image `*_segmentation_summary.json`

All outputs are written to `results/segmentation/` by default.

## Tips & configuration

- Change `RESIZE_TO`, `SAMPLES_PER_FOLDER` and `RANDOM_SEED` at the top of the notebook for different experiments.
- Adjust HSV bounds (`lower_green`, `upper_green`) for different crops/backgrounds.
- For large batch runs prefer converting the notebook to a script/CLI (see project plan).

## Suggested next steps

- Convert the notebook into a reproducible CLI script for batch processing.
- Add automated unit tests and a tiny example dataset (3 images per class) for quick checks.
- Create a small labelled validation set and compute IoU / F1 to quantify segmentation quality.
- Experiment with a lightweight U-Net model if heuristic masks are insufficient.

---
If you want, I can run the notebook locally (headless) to produce a sample run and show the resulting files.
