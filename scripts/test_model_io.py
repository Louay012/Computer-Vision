from pathlib import Path
import sys
# ensure project root is on sys.path so `src` package is importable when running this script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model_io import save_joblib_model, save_torch_state_dict, save_label_encoder

models_dir = Path('results/combined_run/models')
models_dir.mkdir(parents=True, exist_ok=True)

dummy = {'x': 123}

print('Saving TEST_MODEL (1)')
p1 = save_joblib_model(dummy, models_dir, 'TEST_MODEL')
print('Saved to', p1)

print('Saving TEST_MODEL (2)')
p2 = save_joblib_model(dummy, models_dir, 'TEST_MODEL')
print('Saved to', p2)

print('Listing matches:')
for p in sorted(models_dir.glob('TEST_MODEL*.joblib')):
    print(' -', p)
