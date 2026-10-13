"""
STEP 4 — USE THE TRAINED PIPELINE ON A NEW RECORDING

    .venv/bin/python predict.py data/raw/<session folder>

Saves predictions.csv in the session folder (heart rate for every window).
"""

import sys
from pathlib import Path

import joblib
import numpy as np

from common import CLASS_NAMES, MODEL_PATH, apply_pipeline, load_session
from features import feature_columns, session_windows

folder = Path(sys.argv[1])
models = joblib.load(MODEL_PATH)
windows = session_windows(load_session(folder))
result = windows.join(apply_pipeline(models, windows[feature_columns(windows)]))
result[["time", "pred_class", "hr_pred", "fft_hr", "label_hr"]].to_csv(folder / "predictions.csv", index=False)

for c, n in result["pred_class"].value_counts().sort_index().items():
    print(f"{CLASS_NAMES[c]:<12} {n} windows")
print(f"Heart rate: median {result.hr_pred.median():.0f} BPM")

ref = result.dropna(subset=["label_hr"])
if len(ref):
    err = lambda col: np.sqrt(np.mean((ref[col] - ref["label_hr"]) ** 2))
    print(f"RMSE vs chest strap: pipeline {err('hr_pred'):.2f} BPM, FFT only {err('fft_hr'):.2f} BPM")
print(f"Saved -> {folder / 'predictions.csv'}")
