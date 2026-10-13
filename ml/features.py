"""
STEP 2 — TURN RECORDINGS INTO A TABLE OF FEATURES

    .venv/bin/python features.py

Cuts every session into 6-second windows (a new one every second) and
computes, for each window:
  - the paper's statistics for 3 signals: PPG, accelerometer, gyroscope
  - fft_hr:   heart rate from the FFT of the PPG (used directly at rest)
  - rest_hr:  the wearer's recent resting heart rate (see below)
  - labels:   activity class (from meta.json) and true heart rate (strap)

Output: data/windows.csv, one row per window.
"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from common import FS, STEP_S, WINDOW_S, WINDOWS_CSV, fft_heart_rate, list_sessions, load_session, preprocess

SIGNALS = ["ppg", "acc", "gyro"]

# A window counts as "still" when the wrist barely moves. Check these against
# the acc__std / gyro__std columns of your real rest windows.
STILL_MAX_ACC_STD = 0.01     # g
STILL_MAX_GYRO_STD = 3.0     # degrees/second


# =============================================================================
#  PAPER FEATURES (section 2.3) FOR ONE SIGNAL
# =============================================================================
def signal_features(x):
    p25, p75 = np.percentile(x, [25, 75])
    pos = np.abs(x) + 1e-9                          # geometric/harmonic means need positive numbers
    counts, edges = np.histogram(x, bins=20)        # "mode" of a continuous signal = fullest bin
    i = np.argmax(counts)
    spectrum = np.abs(np.fft.rfft(x - x.mean())) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1 / FS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skew, kurt = stats.skew(x), stats.kurtosis(x)
    return {
        "mean": x.mean(),
        "median": np.median(x),
        "std": x.std(),
        "mad": np.mean(np.abs(x - x.mean())),
        "p25": p25,
        "p75": p75,
        "iqr": p75 - p25,
        "skew": skew,
        "kurtosis": kurt,
        "mode": (edges[i] + edges[i + 1]) / 2,
        "trim_mean": stats.trim_mean(x, 0.1),
        "geo_mean": np.exp(np.mean(np.log(pos))),
        "harm_mean": len(x) / np.sum(1 / pos),
        "var": x.var(),
        "deriv1": np.mean(np.abs(np.diff(x))) * FS,
        "deriv2": np.mean(np.abs(np.diff(x, 2))) * FS ** 2,
        "fund_freq": freqs[1:][np.argmax(spectrum[1:])],   # skip the 0 Hz bin
        "energy": spectrum.sum(),
    }


# =============================================================================
#  ONE SESSION -> WINDOWS
# =============================================================================
def label_class(segments, t0, t1):
    """Activity class if the window lies fully inside one activity, else empty."""
    for seg in segments:
        if seg["start"] <= t0 and t1 <= seg["end"]:
            return seg["class"]
    return None


def label_hr(reference, t0, t1):
    """Average strap heart rate during the window."""
    if reference is None:
        return None
    hr = reference.loc[(reference["time"] >= t0) & (reference["time"] < t1), "hr"]
    return hr.mean() if len(hr) >= 3 else None


def session_windows(session):
    sig = preprocess(session["band"])
    size, step = WINDOW_S * FS, STEP_S * FS
    rows = []
    for start in range(10 * FS, len(sig) - size + 1, step):    # skip 10 s while filters settle
        win = sig.iloc[start:start + size]
        t0, t1 = win["time"].iloc[0], win["time"].iloc[-1]
        row = {
            "session": session["id"],
            "time": (t0 + t1) / 2,
            "label_class": label_class(session["segments"], t0, t1),
            "label_hr": label_hr(session["reference"], t0, t1),
        }
        # PPG strength depends on skin and strap tightness, so divide by the light level
        signals = {
            "ppg": win["ppg"].to_numpy() / win["ppg_raw"].mean() * 1000,
            "acc": win["acc"].to_numpy(),
            "gyro": win["gyro"].to_numpy(),
        }
        for name in SIGNALS:
            for stat, value in signal_features(signals[name]).items():
                row[f"{name}__{stat}"] = value
        row["fft_hr"] = fft_heart_rate(win["ppg"].to_numpy())
        rows.append(row)

    df = pd.DataFrame(rows)
    df["rest_hr"] = recent_rest_hr(df)
    return df.fillna({c: 0.0 for c in feature_columns(df)})


# =============================================================================
#  REST_HR — the one feature not from the paper
#
#  The paper's heart rate models relied on body temperature, which this band
#  doesn't measure. Without it, accelerometer and gyroscope only say HOW MUCH
#  you're moving, not what YOUR heart rate is. So each window also gets the
#  wearer's recent resting heart rate: the median FFT heart rate of the last
#  10 still windows before it. The model learns "rest HR + rise for this
#  activity". It only uses earlier windows, so it works live too.
# =============================================================================
def recent_rest_hr(df):
    still = (df["acc__std"] < STILL_MAX_ACC_STD) & (df["gyro__std"] < STILL_MAX_GYRO_STD)
    history, result = [], []
    for fft_hr, is_still in zip(df["fft_hr"], still):
        result.append(np.median(history[-10:]) if history else fft_hr)
        if is_still:
            history.append(fft_hr)
    return result


def feature_columns(df):
    return [c for c in df.columns if c not in ("session", "time", "label_class", "label_hr")]


if __name__ == "__main__":
    frames = []
    for folder in list_sessions():
        df = session_windows(load_session(folder))
        print(f"  {folder.name}: {len(df)} windows")
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    WINDOWS_CSV.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(WINDOWS_CSV, index=False)
    print(f"Saved {len(data)} windows x {len(feature_columns(data))} features -> {WINDOWS_CSV}")
