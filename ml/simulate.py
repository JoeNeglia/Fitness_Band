"""
TEST DATA — FAKE SESSIONS FOR TRYING THE PIPELINE WITHOUT HARDWARE

    .venv/bin/python simulate.py                 # 8 subjects, full 30-minute protocol
    .venv/bin/python simulate.py --sessions 4 --scale 0.3

Writes sessions to data/raw/sim_*/ in exactly the format record.py
produces (band.csv, reference.csv, meta.json), so every later step can be
tested end to end.

The fake data imitates the effects in the paper: a pulse wave at a
changing heart rate, arm-swing artifacts while walking (visible in the
IMU, "Mmec") and finger/tendon artifacts while writing and reading
(mostly NOT visible in the IMU, "Mvascular").

These numbers are invented, so accuracy on simulated data says
nothing about accuracy on real data. Delete data/raw/sim_* before
training on real recordings.
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy import signal

from common import ACTIVITY_CLASS, FS, PROTOCOL, RAW_DIR

# Heart rate rise above resting for each activity (BPM)
HR_RISE = {"rest": 0, "read": 4, "write": 7, "walk_slow": 22, "walk_fast": 40}
STEP_CADENCE_HZ = {"walk_slow": 1.6, "walk_fast": 2.0}


def band_noise(rng, n, lo, hi):
    """Random noise limited to lo-hi Hz, scaled to standard deviation 1."""
    sos = signal.butter(2, [lo, hi], btype="bandpass", fs=FS, output="sos")
    x = signal.sosfiltfilt(sos, rng.standard_normal(n + 2 * FS))[FS:-FS]
    return x / (x.std() + 1e-12)


def lowpass(x, tau_s, start=None):
    """First-order lag: x follows its input with time constant tau_s, starting at `start`
    (default: the first input value; use 0 for noise)."""
    a = 1 / (tau_s * FS)
    y0 = x[0] if start is None else start
    y, _ = signal.lfilter([a], [1, a - 1], x, zi=[y0 * (1 - a)])
    return y


def pulse_shape(phase):
    """One heart beat as a function of phase 0-1: systolic peak + smaller dicrotic wave."""
    p = phase % 1.0
    return np.exp(-((p - 0.15) / 0.07) ** 2) + 0.4 * np.exp(-((p - 0.45) / 0.10) ** 2)


def simulate_session(idx, scale, rng):
    rest_hr = rng.normal(66, 8)
    dc_ir, dc_red = rng.uniform(60e3, 150e3), rng.uniform(40e3, 100e3)
    pulse_amp = rng.uniform(0.004, 0.010)          # pulse is ~0.5-1% of the light level
    t0 = 1.76e9 + idx * 86400                      # one fake day per session

    # ---- Timeline: protocol activities with short unlabelled gaps between ----
    segments, activity = [], []
    t = 0.0
    for name, minutes in PROTOCOL:
        gap = rng.uniform(5, 15)                   # time taken to press Enter
        activity += ["gap"] * int(gap * FS)
        t += gap
        dur = minutes * 60 * scale
        segments.append({"activity": name, "class": ACTIVITY_CLASS[name],
                         "start": t0 + t, "end": t0 + t + dur})
        activity += [name] * int(dur * FS)
        t += dur
    activity = np.array(activity)
    n = len(activity)
    time_s = np.arange(n) / FS

    # ---- True heart rate: lags behind activity changes, plus slow wander ----
    target = rest_hr + np.array([HR_RISE.get(a, 0) for a in activity], dtype=float)
    hr = lowpass(target, 25) + 3 * lowpass(rng.standard_normal(n), 20, start=0) * np.sqrt(40 * FS)
    hr += 1.5 * np.sin(2 * np.pi * 0.25 * time_s)  # breathing makes HR rise and fall slightly
    phase = np.cumsum(hr / 60 / FS)

    # ---- Motion (IMU) and motion artifacts ----
    acc = np.zeros((n, 3))
    gyro = np.zeros((n, 3))
    mmec = np.zeros(n)        # artifact from the sensor moving on the skin (follows the IMU)
    mvasc = np.zeros(n)       # artifact from finger/tendon movement (mostly invisible to the IMU)
    for seg_name in set(activity):
        m = activity == seg_name
        k = m.sum()
        if seg_name in STEP_CADENCE_HZ:
            f = STEP_CADENCE_HZ[seg_name] * rng.uniform(0.92, 1.08)
            tt = time_s[m]
            strength = 0.25 if seg_name == "walk_slow" else 0.45
            swing = np.sin(2 * np.pi * f / 2 * tt)                    # arm swings once per 2 steps
            impact = np.maximum(0, np.sin(2 * np.pi * f * tt)) ** 4   # heel strike every step
            acc[m, 0] = strength * swing + 0.05 * band_noise(rng, k, 1, 10)
            acc[m, 2] = strength * 0.8 * impact + 0.05 * band_noise(rng, k, 1, 10)
            gyro[m, 1] = (80 if seg_name == "walk_slow" else 140) * np.cos(2 * np.pi * f / 2 * tt)
            mmec[m] = rng.uniform(1.5, 3.0) * (0.6 * swing + impact - impact.mean())
            mvasc[m] = 0.5 * band_noise(rng, k, 0.6, 3)
        elif seg_name == "write":
            acc[m] = 0.02 * np.column_stack([band_noise(rng, k, 1, 8) for _ in range(3)])
            gyro[m] = 12 * np.column_stack([band_noise(rng, k, 1, 6) for _ in range(3)])
            mmec[m] = 0.3 * band_noise(rng, k, 0.6, 3)
            mvasc[m] = rng.uniform(1.0, 2.0) * band_noise(rng, k, 0.7, 2.5)
        elif seg_name == "read":
            # A page turn every ~5 s: a 1-second wrist flick
            flick = np.zeros(k)
            for start in np.arange(rng.uniform(0, 5), k / FS - 1, 5):
                i0 = int(start * FS)
                flick[i0:i0 + FS] = np.sin(np.linspace(0, np.pi, FS))
            acc[m, 1] = 0.15 * flick + 0.01 * band_noise(rng, k, 1, 8)
            gyro[m, 0] = 90 * flick
            mmec[m] = 2.0 * flick
            mvasc[m] = rng.uniform(0.6, 1.2) * band_noise(rng, k, 0.7, 2.5)
        # rest + gaps: almost still
    acc += 0.004 * rng.standard_normal((n, 3))
    acc[:, 2] += 1.0                                               # gravity
    gyro += 0.4 * rng.standard_normal((n, 3)) + rng.normal(0, 1.5, 3)   # noise + gyro bias

    # ---- PPG: light level drops a little with each beat ----
    breathing = 0.002 * np.sin(2 * np.pi * 0.22 * time_s)
    drift = 0.01 * lowpass(rng.standard_normal(n), 60, start=0) * np.sqrt(120 * FS)
    beat = pulse_shape(phase)
    artifact = (mmec + mvasc) * pulse_amp            # artifacts sized relative to the pulse
    ir = dc_ir * (1 + drift + breathing - pulse_amp * beat + artifact) + 30 * rng.standard_normal(n)
    red = dc_red * (1 + drift + breathing - 0.6 * pulse_amp * beat + 1.3 * artifact) + 30 * rng.standard_normal(n)

    band = pd.DataFrame({
        "host_time": t0 + time_s, "t_ms": (time_s * 1000).astype(int),
        "red": red.astype(int), "ir": ir.astype(int),
        "ax": acc[:, 0], "ay": acc[:, 1], "az": acc[:, 2],
        "gx": gyro[:, 0], "gy": gyro[:, 1], "gz": gyro[:, 2],
    })
    # Chest strap: one reading per second, ±1 BPM
    sec = np.arange(0, n, FS)
    reference = pd.DataFrame({"time": t0 + time_s[sec],
                              "hr": np.round(hr[sec] + rng.normal(0, 1, len(sec))).astype(int)})

    session_id = f"sim_{idx:02d}"
    meta = {"segments": segments}
    return session_id, band, reference, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", type=int, default=8, help="number of fake subjects")
    ap.add_argument("--scale", type=float, default=1.0, help="scale activity durations")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    for i in range(args.sessions):
        sid, band, reference, meta = simulate_session(i, args.scale, rng)
        out = RAW_DIR / sid
        out.mkdir(parents=True, exist_ok=True)
        band.to_csv(out / "band.csv", index=False, float_format="%.4f")
        reference.to_csv(out / "reference.csv", index=False)
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"  {sid}: {len(band) / FS / 60:.1f} min")
    print(f"Simulated {args.sessions} sessions -> {RAW_DIR}")


if __name__ == "__main__":
    main()
