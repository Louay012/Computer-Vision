# Project Plan — Plant Leaf Segmentation & Anomaly Detection

## Objective

Deliver a reproducible pipeline to segment plant leaves, extract masks and edge features, and prepare data for downstream disease/anomaly detection and model training.

## High-level milestones

1. Documentation & quick reproducibility (0.5–1 day)
   - Create AI summary and project plan (done).
   - Add a tiny example dataset (3 images per class) for smoke tests.

2. Reproducible environment & data sanity (1–2 days)
   - Ensure `requirements.txt` installs cleanly in a virtualenv.
   - Add a lightweight `scripts/check_data.py` to verify folder structure and sample counts.
   - Produce a preprocessing report (`data/processed/preprocessing_report.json`).

3. Baseline segmentation & batch runner (1–2 days)
   - Convert `notebooks/segmentation.ipynb` to a CLI script `scripts/segmentation_batch.py` (arguments: `--input`, `--output`, `--samples-per-folder`, `--resize`).
   - Add logging and error handling; keep outputs in `results/segmentation/`.
   - Include a dry-run option that only lists chosen sample files.

4. Evaluation & reporting (2–4 days)
   - Create a small labelled validation set (mask annotations) for quantitative metrics.
   - Implement IoU / Dice / Precision-Recall evaluation and a summary report (CSV + JSON).
   - Add visualization HTML or notebook summarizing failures.

5. Improve segmentation (2–14 days, iterative)
   - Tune heuristic pipeline (morphology, HSV ranges, clustering parameters).
   - If heuristics plateau, train a lightweight U-Net on processed masks.
   - Automate experiments and record results.

6. Anomaly detection & classification (2–10 days)
   - Extract features from masked leaf regions (texture, color histograms, CNN embeddings).
   - Train a classifier to distinguish healthy vs disease classes and prototype an anomaly detector.

7. Packaging & deployment (1–3 days)
   - Create a simple inference script `scripts/infer.py` and a small Flask/FastAPI demo for local testing.
   - Containerize with Docker for reproducible demos.

## Immediate next steps (this week)

- Run `notebooks/segmentation.ipynb` with `SAMPLES_PER_FOLDER = 3` (quick smoke test across folders).
- Inspect `results/segmentation/one_per_folder_summary.json` to confirm expected outputs.
- Create `scripts/segmentation_batch.py` (simple wrapper around current notebook code) and add CLI flags.

## Risks & dependencies

- Labelling effort: quantitative evaluation requires mask annotations.
- Compute: training deep models needs GPU for reasonable turn-around.
- Data variability: different crops/backgrounds may require per-dataset HSV tuning.

## Success criteria

- Reproducible batch run producing masks for N folders within expected runtime.
- Measured IoU/Dice above threshold on validation set (project-specific target).
- Clean CLI and documentation so a new contributor can run a full pipeline locally.

## Deliverables

- `AI_SUMMARY.md`, `PROJECT_PLAN.md` (this doc)
- `scripts/segmentation_batch.py` (CLI)
- `results/segmentation/` example outputs
- Evaluation reports (CSV/JSON) and a short demo for inference
