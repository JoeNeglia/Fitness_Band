"""
Shared settings, file loading, filters and the final pipeline, used by every other script.

Session folder layout (one folder per recording, under data/raw/):
    band.csv       raw band data, 100 rows per second
                   host_time,t_ms,red,ir,ax,ay,az,gx,gy,gz
    reference.csv  true heart rate from a chest strap: time,hr
    meta.json      the labelled activity segments (start/end times)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

# =============================================================================
#  PATHS
# =============================================================================
ML_DIR = Path(__file__).resolve().parent
RAW_DIR = ML_DIR / "data" / "raw"
WINDOWS_CSV = ML_DIR / "data" / "windows.csv"
MODEL_PATH = ML_DIR / "models" / "hr_pipeline.joblib"
REPORT_DIR = ML_DIR / "reports"

# =============================================================================
#  SETTINGS
# =============================================================================
FS = 100              # band samples per second
WINDOW_S = 6          # paper: 6-second windows
STEP_S = 1            # a new window every second
HR_BAND_HZ = (0.6, 3.0)   # paper: heart rate lives in 0.6-3 Hz (36-180 BPM)

# Paper Table 1. Class 1 = rest, 2 = hand/finger motion, 3 = walking
PROTOCOL = [
    ("rest", 5), ("read", 3), ("rest", 2), ("walk_slow", 5), ("rest", 2),
    ("write", 3), ("rest", 2), ("walk_fast", 5), ("rest", 2),
]
ACTIVITY_CLASS = {"rest": 1, "read": 2, "write": 2, "walk_slow": 3, "walk_fast": 3}
CLASS_NAMES = {1: "rest", 2: "hand motion", 3: "walking"}


# =============================================================================
#  LOADING
# =============================================================================
def list_sessions():
    return sorted(p for p in RAW_DIR.iterdir() if (p / "band.csv").exists())


def load_session(folder):
    folder = Path(folder)
    band = pd.read_csv(folder / "band.csv")
    ref = folder / "reference.csv"
    meta = folder / "meta.json"
    return {
        "id": folder.name,
        "band": band,
        "reference": pd.read_csv(ref) if ref.exists() else None,
        "segments": json.loads(meta.read_text())["segments"] if meta.exists() else [],
    }


# =============================================================================
#  FILTERS (paper section 2.3)
#  - PPG: band-pass 0.6-3 Hz keeps only the heart beat range
#  - IMU: high-pass removes each axis' baseline (gravity, gyro offset),
#         then the 3 axes are combined into one magnitude
# =============================================================================
_PPG_FILTER = signal.butter(2, HR_BAND_HZ, btype="bandpass", fs=FS, output="sos")
_IMU_FILTER = signal.butter(2, 0.3, btype="highpass", fs=FS, output="sos")


def _filter(sos, x):
    x = np.asarray(x, dtype=float)
    y, _ = signal.sosfilt(sos, x, zi=signal.sosfilt_zi(sos) * x[0])   # start at the first value
    return y


def preprocess(band):
    """Raw band columns -> the 4 signals the features are computed from."""
    return pd.DataFrame({
        "time": band["host_time"].to_numpy(dtype=float),
        "ppg_raw": band["ir"].to_numpy(dtype=float),
        "ppg": _filter(_PPG_FILTER, -band["ir"]),      # flipped so heart beats are peaks
        "acc": np.sqrt(sum(_filter(_IMU_FILTER, band[c]) ** 2 for c in ("ax", "ay", "az"))),
        "gyro": np.sqrt(sum(_filter(_IMU_FILTER, band[c]) ** 2 for c in ("gx", "gy", "gz"))),
    })


def fft_heart_rate(ppg_window):
    """Paper's rest-class method: the strongest frequency in the 0.6-3 Hz band, in BPM."""
    x = (ppg_window - ppg_window.mean()) * np.hanning(len(ppg_window))
    power = np.abs(np.fft.rfft(x, n=8192)) ** 2          # zero-padded for finer resolution
    freqs = np.fft.rfftfreq(8192, d=1 / FS)
    band = (freqs >= HR_BAND_HZ[0]) & (freqs <= HR_BAND_HZ[1])
    return freqs[band][np.argmax(power[band])] * 60


# =============================================================================
#  THE FINAL PIPELINE (paper Figure 10)
#  `models` is the dict saved by train.py.
# =============================================================================
def apply_pipeline(models, windows):
    """Returns predicted activity class and heart rate for each window."""
    pred_class = models["classifier"].predict(windows[models["classifier_features"]])
    hr = windows["fft_hr"].to_numpy(dtype=float).copy()      # rest: FFT heart rate
    for cls, reg in models["regressors"].items():             # motion: regression model
        rows = pred_class == cls
        if rows.any():
            hr[rows] = reg.predict(windows.loc[rows, models["regressor_features"][cls]])
    return pd.DataFrame({"pred_class": pred_class, "hr_pred": hr}, index=windows.index)
