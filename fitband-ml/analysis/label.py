"""
label.py — turn raw recordings into a table of labelled 6-second windows.

    python analysis/label.py

Reads every data/raw/session*/ folder, pairs each band_<activity>_<stamp>.csv
with its hrm_<activity>_<stamp>.csv, cuts each block into non-overlapping
6-second windows and computes for every window:

  features     mean, std, skewness, kurtosis and 0.5-3 Hz FFT energy of
               ppg, ax, ay, az, gx, gy, gz and the accel magnitude
  hr_truth     mean chest-strap heart rate in the window (the label)
  hr_truth_rr  heart rate from the strap's beat-to-beat (RR) intervals
  fft_bpm      baseline: dominant frequency of the band-pass filtered PPG
  device_bpm   the band's own on-board heart rate (median in the window)

Writes data/processed/windows.csv and prints a summary.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats

# =============================================================================
#  SETTINGS
# =============================================================================
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUT_CSV = PROJECT_DIR / "data" / "processed" / "windows.csv"

WINDOW_S = 6.0                  # window length in seconds (non-overlapping)
MIN_STRAP_READINGS = 3          # drop windows with fewer strap readings than this
MIN_BAND_FRACTION = 0.8         # drop windows with less than 80% of the expected band rows

ENERGY_BAND_HZ = (0.5, 3.0)     # band used for the "energy" feature
HR_BAND_HZ = (0.67, 3.5)        # heart rate search range: 40-210 BPM
RR_RANGE_MS = (300, 2000)       # RR intervals outside this are treated as glitches

# Band CSV column for each signal we compute features on (acc_mag is added below)
SIGNAL_COLUMNS = {
    "ppg": "ppg",
    "ax": "ax_mg", "ay": "ay_mg", "az": "az_mg",
    "gx": "gx_dps", "gy": "gy_dps", "gz": "gz_dps",
    "acc_mag": "acc_mag",
}
FEATURE_NAMES = ["mean", "std", "skew", "kurt", "energy"]


# =============================================================================
#  STEP 1: FIND THE FILES
#  Each block is one activity recording: a band file plus its matching hrm file.
# =============================================================================
def activity_from_filename(filename):
    """band_walking_20261112_101500.csv -> 'walking'"""
    stem = Path(filename).stem                      # band_walking_20261112_101500
    return stem[len("band_"):].rsplit("_", 2)[0]    # drop the date and time at the end


def find_blocks():
    blocks = []
    for session_dir in sorted(RAW_DIR.glob("session*")):
        if not session_dir.is_dir():
            continue
        for band_path in sorted(session_dir.glob("band_*.csv")):
            hrm_path = band_path.with_name("hrm_" + band_path.name[len("band_"):])
            if not hrm_path.exists():
                print(f"WARNING: no matching hrm file for {band_path.name}, skipping")
                continue
            blocks.append({
                "session": session_dir.name,
                "activity": activity_from_filename(band_path.name),
                "band_path": band_path,
                "hrm_path": hrm_path,
            })
    return blocks


# =============================================================================
#  STEP 2: LOAD ONE BLOCK
# =============================================================================
def load_band(path):
    band = pd.read_csv(path)
    band["acc_mag"] = np.sqrt(band["ax_mg"] ** 2 + band["ay_mg"] ** 2 + band["az_mg"] ** 2)
    return band


def load_hrm(path):
    # rr_ms is read as text because one row can hold several values like "812;798"
    return pd.read_csv(path, dtype={"rr_ms": str})


def sampling_rate(band):
    """Band rows per second, measured from the board's own clock (the ms column)."""
    return 1000.0 / np.median(np.diff(band["ms"]))


# =============================================================================
#  STEP 3: BEAT TIMES FROM RR INTERVALS
#  Each strap packet lists the RR intervals of the beats that ended just
#  before it arrived, oldest first. The last beat happened at about the
#  packet's pc_time, so we walk backwards: each earlier beat happened one
#  RR interval before the next one.
# =============================================================================
def rr_beats(hrm):
    times, rr_values = [], []
    for pc_time, rr_text in zip(hrm["pc_time"], hrm["rr_ms"]):
        if not isinstance(rr_text, str) or rr_text.strip() == "":
            continue
        beat_time = pc_time
        for rr in reversed([float(v) for v in rr_text.split(";")]):
            times.append(beat_time)            # this interval ended at beat_time
            rr_values.append(rr)
            beat_time -= rr / 1000             # the previous beat was rr ms earlier
    beats = pd.DataFrame({"time": times, "rr_ms": rr_values})
    return beats[beats["rr_ms"].between(*RR_RANGE_MS)]


# =============================================================================
#  STEP 4: SIGNAL HELPERS
# =============================================================================
def resample_window(rows, fs):
    """
    Bluetooth delivers rows unevenly, so we use the board's ms column to
    place each sample at its true time, then interpolate onto an even grid
    (WINDOW_S * fs samples). Filters and the FFT assume even spacing.
    """
    t = (rows["ms"].to_numpy() - rows["ms"].iloc[0]) / 1000.0
    grid = np.arange(int(round(WINDOW_S * fs))) / fs
    return {name: np.interp(grid, t, rows[col].to_numpy(dtype=float))
            for name, col in SIGNAL_COLUMNS.items()}


def band_energy(x, fs, band):
    """Signal power between band[0] and band[1] Hz (from the FFT)."""
    spectrum = np.abs(np.fft.rfft(x - x.mean())) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1 / fs)
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    return spectrum[in_band].sum() / len(x)


def bandpass(x, fs, band):
    """Keep only frequencies inside `band` (Butterworth, applied forwards and backwards)."""
    high = min(band[1], 0.45 * fs)     # the top of the band must stay below fs/2
    sos = signal.butter(2, [band[0], high], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def dominant_bpm(x, fs, band):
    """Strongest frequency inside `band`, converted to beats per minute."""
    x = (x - x.mean()) * np.hanning(len(x))
    spectrum = np.abs(np.fft.rfft(x, n=4096)) ** 2      # zero-padding gives finer steps
    freqs = np.fft.rfftfreq(4096, d=1 / fs)
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    return freqs[in_band][np.argmax(spectrum[in_band])] * 60


def signal_features(x, fs):
    with np.errstate(all="ignore"):     # skew/kurtosis of a flat signal is undefined
        values = {
            "mean": x.mean(),
            "std": x.std(),
            "skew": stats.skew(x),
            "kurt": stats.kurtosis(x),
            "energy": band_energy(x, fs, ENERGY_BAND_HZ),
        }
    return {k: (0.0 if not np.isfinite(v) else float(v)) for k, v in values.items()}


# =============================================================================
#  STEP 5: ONE BLOCK -> WINDOWS
# =============================================================================
def process_block(block, drops):
    band = load_band(block["band_path"])
    hrm = load_hrm(block["hrm_path"])
    if len(band) < 2:
        print(f"WARNING: {block['band_path'].name} is empty, skipping")
        return []

    fs = sampling_rate(band)
    beats = rr_beats(hrm)
    start = band["pc_time"].iloc[0]
    n_windows = int((band["pc_time"].iloc[-1] - start) // WINDOW_S)   # only full windows
    min_band_rows = MIN_BAND_FRACTION * WINDOW_S * fs
    print(f"  {block['session']}/{block['activity']}: {n_windows * WINDOW_S:.0f} s, "
          f"{fs:.1f} Hz, {n_windows} windows")

    rows = []
    for w in range(n_windows):
        t0 = start + w * WINDOW_S
        t1 = t0 + WINDOW_S
        band_rows = band[(band["pc_time"] >= t0) & (band["pc_time"] < t1)]
        strap_hr = hrm.loc[(hrm["pc_time"] >= t0) & (hrm["pc_time"] < t1), "hr"]

        # ---- Skip windows without enough data ----
        if len(strap_hr) < MIN_STRAP_READINGS:
            drops["too few strap readings"] += 1
            continue
        if len(band_rows) < min_band_rows:
            drops["too few band rows"] += 1
            continue

        # ---- Features from evenly resampled band signals ----
        signals = resample_window(band_rows, fs)
        row = {"session": block["session"], "activity": block["activity"], "window": w}
        for name, x in signals.items():
            for feat, value in signal_features(x, fs).items():
                row[f"{name}_{feat}"] = value

        # ---- Labels and baselines ----
        row["hr_truth"] = strap_hr.mean()
        rr = beats.loc[(beats["time"] >= t0) & (beats["time"] < t1), "rr_ms"]
        row["hr_truth_rr"] = 60000 / rr.mean() if len(rr) else np.nan
        row["fft_bpm"] = dominant_bpm(bandpass(signals["ppg"], fs, HR_BAND_HZ), fs, HR_BAND_HZ)
        device = band_rows.loc[band_rows["bpm"] > 0, "bpm"]          # 0 means "no reading"
        row["device_bpm"] = device.median() if len(device) else np.nan
        rows.append(row)
    return rows


# =============================================================================
#  STEP 6: SUMMARY
# =============================================================================
def rmse(pred, truth):
    ok = pred.notna() & truth.notna()
    if not ok.any():
        return np.nan, 0
    return float(np.sqrt(np.mean((pred[ok] - truth[ok]) ** 2))), int(ok.sum())


def print_summary(df, drops):
    total_dropped = sum(drops.values())
    print(f"\nKept {len(df)} windows, dropped {total_dropped}"
          + "".join(f"\n  {reason}: {n}" for reason, n in drops.items()))

    print("\nWindows per session and activity:")
    print(pd.crosstab(df["session"], df["activity"], margins=True, margins_name="total"))

    both = df.dropna(subset=["hr_truth_rr"])
    if len(both):
        diff = (both["hr_truth"] - both["hr_truth_rr"]).abs().mean()
        print(f"\nStrap check: mean |hr_truth - hr_truth_rr| = {diff:.2f} BPM ({len(both)} windows)")
        print("  (a large value means the strap's own heart rate is smoothed or lagging)")
    else:
        print("\nStrap check: no RR intervals recorded, can't compare.")

    print("\nRMSE against hr_truth (BPM)     fft_bpm        device_bpm")
    groups = [("all", df)] + list(df.groupby("activity"))
    for name, part in groups:
        fft_err, fft_n = rmse(part["fft_bpm"], part["hr_truth"])
        dev_err, dev_n = rmse(part["device_bpm"], part["hr_truth"])
        print(f"  {name:<28} {fft_err:6.2f} (n={fft_n:<4}) {dev_err:6.2f} (n={dev_n})")


# =============================================================================
#  MAIN
# =============================================================================
def main():
    blocks = find_blocks()
    if not blocks:
        sys.exit(f"No band/hrm file pairs found in {RAW_DIR}/session*/")

    print(f"Found {len(blocks)} blocks")
    drops = {"too few strap readings": 0, "too few band rows": 0}
    rows = []
    for block in blocks:
        rows += process_block(block, drops)
    if not rows:
        sys.exit("Every window was dropped. Check that the band and hrm files overlap in time.")

    df = pd.DataFrame(rows)
    feature_cols = [f"{s}_{f}" for s in SIGNAL_COLUMNS for f in FEATURE_NAMES]
    df = df[["session", "activity", "window"] + feature_cols +
            ["hr_truth", "hr_truth_rr", "fft_bpm", "device_bpm"]]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"\nSaved {len(df)} windows x {len(feature_cols)} features -> {OUT_CSV}")
    print_summary(df, drops)


if __name__ == "__main__":
    main()
