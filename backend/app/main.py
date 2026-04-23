from pathlib import Path
from typing import Optional, List
import io
import sys
import warnings

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import joblib
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


APP_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = APP_ROOT / "results" / "combined_run" / "models"

# Lazy import for feature extraction helper. We import on demand inside the ML
# prediction branch to avoid crashing the server at import time if the module
# is missing during development.
extract_features = None

def _load_extract_features():
    global extract_features
    if extract_features is not None:
        return extract_features
    try:
        from src.features import extract_features as _ef
        extract_features = _ef
        return extract_features
    except Exception:
        # Ensure both the repository root and the `src` folder are on sys.path.
        sys.path.insert(0, str(APP_ROOT))
        sys.path.insert(0, str(APP_ROOT / 'src'))
        try:
            from src.features import extract_features as _ef
            extract_features = _ef
            return extract_features
        except Exception:
            # Fallback: try loading the module directly from the file location.
            import importlib.util
            feat_path = APP_ROOT / 'src' / 'features.py'
            if feat_path.exists():
                spec = importlib.util.spec_from_file_location('src.features', str(feat_path))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                extract_features = getattr(module, 'extract_features')
                return extract_features
            raise


app = FastAPI(title="Plant Disease API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelManager:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self._ml_cache = {}
        self.label_encoder = None
        self._dl_model = None
        self._preprocess = None

    def list_ml_models(self) -> List[str]:
        if not self.model_dir.exists():
            return []
        return [p.stem for p in self.model_dir.glob('*.joblib') if p.name != 'label_encoder.joblib']

    def load_label_encoder(self):
        p = self.model_dir / 'label_encoder.joblib'
        if p.exists():
            self.label_encoder = joblib.load(p)
        return self.label_encoder

    def load_ml_model(self, name: str):
        if name in self._ml_cache:
            return self._ml_cache[name]
        p = self.model_dir / f"{name}.joblib"
        if not p.exists():
            raise FileNotFoundError(p)
        m = joblib.load(p)
        self._ml_cache[name] = m
        return m

    def _build_preprocess(self):
        # Build a lightweight, deterministic preprocessing pipeline without
        # relying on torchvision weight helpers (which may attempt to download
        # pretrained weights and block the server).
        if self._preprocess is None:
            self._preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        return self._preprocess

    def load_dl_model(self):
        if self._dl_model is not None:
            return self._dl_model
        p = self.model_dir / 'model_resnet18.pth'
        if not p.exists():
            raise FileNotFoundError(p)
        print(f"[ModelManager] Loading DL model from {p}")
        # Try to load only tensor weights where supported to avoid executing
        # arbitrary pickle code. If the installed torch doesn't support
        # `weights_only` or if the file requires full unpickling, fall back
        # to the original behaviour while logging a warning.
        try:
            try:
                state = torch.load(p, map_location='cpu', weights_only=True)
            except TypeError:
                # Older torch versions don't accept weights_only
                state = torch.load(p, map_location='cpu')
        except Exception as e:
            warnings.warn(f"weights_only load failed ({e}); falling back to full torch.load() which may execute arbitrary code from the file. Only load trusted model files.")
            state = torch.load(p, map_location='cpu')
        print(f"[ModelManager] Raw state type: {type(state)}")
        # Normalize different save formats: support saved state_dict, wrapped dicts,
        # or full model objects.
        if not isinstance(state, dict):
            try:
                state = state.state_dict()
                print('[ModelManager] Extracted state_dict() from saved model object')
            except Exception:
                raise RuntimeError('Unrecognized model file format (not a state_dict)')
        # Some saved state_dicts include a top-level 'state_dict' key
        if 'state_dict' in state and isinstance(state['state_dict'], dict):
            state = state['state_dict']
        # Strip potential DataParallel 'module.' prefixes
        new_state = {}
        for k, v in state.items():
            nk = k
            if isinstance(k, str) and k.startswith('module.'):
                nk = k[len('module.'):]
            new_state[nk] = v
        state = new_state
        # ensure label encoder loaded if present
        if self.label_encoder is None:
            self.load_label_encoder()
        if self.label_encoder is not None:
            num_classes = len(self.label_encoder.classes_)
        else:
            # infer from state dict
            if 'fc.weight' in state:
                num_classes = int(state['fc.weight'].shape[0])
            else:
                raise RuntimeError('Cannot determine number of classes for DL model')
        # Instantiate ResNet without attempting to download pretrained weights.
        try:
            model = resnet18(weights=None)
        except TypeError:
            # older torchvision versions use the `pretrained` kwarg
            model = resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(state)
        model.eval()
        self._dl_model = model
        return model


manager = ModelManager(MODEL_DIR)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/ml_models")
def ml_models():
    return {"models": manager.list_ml_models()}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), model_type: str = Form("dl"), model_name: Optional[str] = Form(None)):
    content = await file.read()
    if model_type.lower() == 'ml':
        # classical ML pipeline: extract features and predict
        model_name = model_name or 'RandomForest'
        try:
            model = manager.load_ml_model(model_name)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"ML model '{model_name}' not found")
        img = Image.open(io.BytesIO(content)).convert('RGB')
        arr = np.array(img)
        try:
            ef = _load_extract_features()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load feature extractor: {e}")
        vec, details = ef(arr)
        try:
            pred = model.predict(vec.reshape(1, -1))[0]
        except Exception as e:
            # model might be a pipeline expecting a list-like input
            raise HTTPException(status_code=500, detail=str(e))
        if manager.label_encoder is None:
            manager.load_label_encoder()
        if manager.label_encoder is not None:
            label = manager.label_encoder.inverse_transform([pred])[0]
        else:
            label = str(pred)
        return {"model_type": "ml", "model_name": model_name, "prediction": label}

    elif model_type.lower() == 'dl':
        try:
            model = manager.load_dl_model()
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="DL model not found")
        img = Image.open(io.BytesIO(content)).convert('RGB')
        preprocess = manager._preprocess or manager._build_preprocess()
        inp = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            out = model(inp)
            probs = F.softmax(out, dim=1)
            top_prob, top_idx = torch.max(probs, dim=1)
            idx = int(top_idx.item())
            prob = float(top_prob.item())
        if manager.label_encoder is None:
            manager.load_label_encoder()
        if manager.label_encoder is not None:
            label = manager.label_encoder.inverse_transform([idx])[0]
        else:
            label = str(idx)
        return {"model_type": "dl", "prediction": label, "probability": prob}

    else:
        raise HTTPException(status_code=400, detail="model_type must be 'ml' or 'dl'")
