from pathlib import Path
import shutil
import joblib
import torch
from typing import Union


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _cleanup_previous(model_dir: Path, pattern: str):
    """Remove files in model_dir that match pattern (glob pattern).

    Example: pattern='RandomForest*' will remove old RandomForest.joblib files.
    """
    for p in model_dir.glob(pattern):
        try:
            if p.is_file():
                p.unlink()
        except Exception:
            # best-effort: ignore failures
            pass


def save_joblib_model(model, model_dir: Union[str, Path], name: str):
    """Save a scikit-learn (joblib) model to model_dir with name (no extension).

    This will delete previous files with the same name pattern before saving the new model.
    Returns the Path to the saved file.
    """
    model_dir = Path(model_dir)
    _ensure_dir(model_dir)
    # delete previous versions (e.g., RandomForest.joblib, RandomForest_*.joblib)
    _cleanup_previous(model_dir, f"{name}*.joblib")
    dest = model_dir / f"{name}.joblib"
    joblib.dump(model, dest)
    return dest


def save_label_encoder(le, model_dir: Union[str, Path]):
    model_dir = Path(model_dir)
    _ensure_dir(model_dir)
    _cleanup_previous(model_dir, "label_encoder*.joblib")
    dest = model_dir / "label_encoder.joblib"
    joblib.dump(le, dest)
    return dest


def save_torch_state_dict(state_dict, model_dir: Union[str, Path], name: str = "model"):
    """Save a PyTorch state_dict atomically and remove previous versions matching the name.

    `state_dict` may be either a state dict or a model object (nn.Module). If it's a model,
    we'll use model.state_dict().
    """
    model_dir = Path(model_dir)
    _ensure_dir(model_dir)
    _cleanup_previous(model_dir, f"{name}*.pth")
    dest = model_dir / f"{name}.pth"
    # write to a temp file then move to ensure atomicity
    tmp = dest.with_suffix('.tmp')
    if hasattr(state_dict, 'state_dict'):
        sd = state_dict.state_dict()
    else:
        sd = state_dict
    torch.save(sd, tmp)
    try:
        shutil.move(str(tmp), str(dest))
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
    return dest
