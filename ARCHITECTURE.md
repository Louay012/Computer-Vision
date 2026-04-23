# Project architecture, structure and data flow

This document describes the overall architecture, file layout, main components, and runtime data flow for the project.

**Goals**
- Provide a concise map of how frontend, backend and model artifacts interact.
- Point to the key source files for maintenance and extension.
- Explain the ML vs DL inference paths and model I/O semantics.

---

## High-level architecture

```mermaid
graph LR
  User[User / Browser] --> Frontend[Frontend (React + Vite)]
  Frontend -->|multipart POST| Backend[Backend (FastAPI)]
  Backend --> Preproc[src/preprocessing.py]
  Backend --> Features[src/features.py]
  Backend --> MLModel[ML artifacts (joblib) in models/]
  Backend --> DLModel[DL artifacts (torch state_dict) in models/]
  Backend --> ModelIO[src/model_io.py]
  MLModel --> LabelEnc[label encoder (joblib)]
  DLModel --> TorchState[torch state_dict (.pt/.pth)]
  Backend --> Response[JSON result]
```

---

## Repository layout (important files)

- **Root**: project entry, setup, requirements and run scripts
  - [requirements.txt](requirements.txt)
  - [setup.bat](setup.bat)
- **backend/**: FastAPI application
  - [backend/app/main.py](backend/app/main.py) — FastAPI app, `ModelManager`, endpoints
- **frontend/**: React + Vite UI
  - [frontend/src/App.jsx](frontend/src/App.jsx) — upload UI, preview, model selector
  - [frontend/src/App.css](frontend/src/App.css)
- **src/**: shared Python helpers used by notebooks and backend
  - [src/features.py](src/features.py) — `extract_features(rgb, mask=None)` used by ML pipeline
  - [src/preprocessing.py](src/preprocessing.py) — `generate_auto_mask_from_bgr()` (Otsu + morphology)
  - [src/model_io.py](src/model_io.py) — atomic save/load helpers for ML/DL artifacts
  - [src/__init__.py](src/__init__.py)
- **models/**: saved model files (joblib / torch state_dict)
- **data/**: raw / processed image datasets and splits
- **results/**: aggregated output, including canonical features CSV
  - [results/combined_run/features/features.csv](results/combined_run/features/features.csv)
- **scripts/**: helper CLI / tests
  - [scripts/e2e_predict_test.py](scripts/e2e_predict_test.py) — test client for `/api/predict`
  - [scripts/check_health.py](scripts/check_health.py)
- **notebooks/**: training and preprocessing notebooks

---

## Backend: endpoints and responsibilities

- `GET /api/health` — basic health check (returns `{"status":"ok"}`).
- `GET /api/ml_models` — list available ML models and metadata (joblib files).
- `POST /api/predict` — main inference endpoint.

POST /api/predict request fields (multipart/form-data):
- `file` — image file (jpeg/png)
- `model_type` — `ml` or `dl` (string)
- `model_name` — optional; ML model filename (e.g., `RandomForest`) to select specific joblib

Behavior inside `POST /api/predict` (high level):
1. Read upload into memory and decode to an RGB numpy array (PIL / OpenCV).
2. Run preprocessing: if no mask provided, call [src/preprocessing.py](src/preprocessing.py)::`generate_auto_mask_from_bgr()` to produce a foreground mask.
3. Branch based on `model_type`:
   - ML path:
     - Call [src/features.py](src/features.py)::`extract_features(rgb, mask)` to produce the 12-dimensional feature vector (see Features mapping).
     - Load the requested joblib model + label encoder via [src/model_io.py](src/model_io.py).
     - Run `model.predict_proba()` or `model.predict()` to get label + probability.
   - DL path:
     - Apply deterministic transforms (resize / center / normalize — no auto-weight downloads).
     - Load PyTorch model state (loader tolerates several state formats and strips `module.` prefixes); instantiate model with `weights=None`.
     - Run forward pass on GPU/CPU and softmax to obtain predicted class and probability.
4. Return JSON: `{ "model_type": "ml"|"dl", "prediction": "Label", "probability": 0.123, ... }`.

Key backend implementation files: [backend/app/main.py](backend/app/main.py), [src/model_io.py](src/model_io.py).

---

## Feature extraction (ML) details

- The ML models rely on a fixed 12-length numeric feature vector matching the canonical CSV at:
  [results/combined_run/features/features.csv](results/combined_run/features/features.csv)

- Feature order (exact):
  1. `h_mean`
  2. `s_mean`
  3. `v_mean`
  4. `h_std`
  5. `s_std`
  6. `v_std`
  7. `contrast`
  8. `energy`
  9. `homogeneity`
  10. `area`
  11. `perimeter`
  12. `circularity`

- Where these are produced by [src/features.py](src/features.py)::`extract_features(rgb, mask=None)`;
  `mask` is optional and created by [src/preprocessing.py](src/preprocessing.py) when absent.

---

## Model I/O and saving semantics

- `src/model_io.py` implements atomic save helpers for both joblib and torch state dicts to prevent partial writes:
  - Save to a temporary file then `os.replace()` to move into place.
  - When re-training with the same logical model name, older artifacts with the same name are removed so only the latest remains.
- Models are stored in the `models/` folder using clear naming conventions (e.g., `RandomForest.joblib`, `label_encoder.joblib`, `resnet50_Plant.pth`).

---

## Frontend behavior

- User actions: drag/drop or choose an image, select `ml` or `dl`, optionally pick an ML model name, click predict.
- UI shows image preview, spinner during request, and result card with label + probability.
- Key files: [frontend/src/App.jsx](frontend/src/App.jsx) and [frontend/src/App.css](frontend/src/App.css).

---

## Data flow (step-by-step)

1. User uploads an image in the browser.
2. Frontend sends multipart `POST /api/predict` with `file` and `model_type`.
3. Backend decodes image → calls preprocessing → obtains `rgb` + `mask`.
4. ML path: `rgb+mask` → `extract_features()` → numeric vector → joblib model → label & prob.
   DL path: `rgb` → deterministic transforms → tensor → torch model forward → label & prob.
5. Backend returns JSON response; frontend displays it.

---

## Quick developer run commands

From project root (PowerShell example):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& ".\venv\Scripts\Activate.ps1"
& ".\venv\Scripts\python.exe" -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Start frontend (separate terminal):

```bash
cd frontend
npm install
npm run dev
```

E2E test for ML (venv active):

```bash
python scripts/e2e_predict_test.py ml RandomForest
```

---

## Troubleshooting & notes

- If the server fails to start with import errors, ensure `src/__init__.py` exists and the working directory is the project root.
- Avoid runtime downloads (torchvision weights) inside server code — models are loaded with `weights=None` to keep startup deterministic.
- When editing `src` while `uvicorn --reload` is running expect short reload cycles; check server logs for detailed exceptions.

---

## Next steps (suggested)

- Run the ML E2E test and confirm matching labels with the saved joblib model.
- Add a lightweight OpenAPI/docs page or a small README in `frontend/` with usage screenshots.

---

File created: [ARCHITECTURE.md](ARCHITECTURE.md)
