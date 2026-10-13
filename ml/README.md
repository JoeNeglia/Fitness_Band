# FitBand ML heart rate pipeline

Makes heart rate more accurate while the wrist is moving. A simplified version of
Vicente-Samper et al., *An ML-Based Approach to Reconstruct Heart Rate from PPG in Presence of
Motion Artifacts*, Biosensors 2023 ([doi:10.3390/bios13070718](https://doi.org/10.3390/bios13070718)).

## How it works (paper Figure 10)

```
6-second window of PPG + accelerometer + gyroscope
        │
        ▼
   features (paper's statistics for each signal)
        │
        ▼
 activity classifier ──► rest        → heart rate from the FFT of the PPG
                     ──► hand motion → heart rate from model B.1
                     ──► walking     → heart rate from model B.2
```

At rest the PPG is clean, so the FFT works. While moving, the PPG is mostly motion noise, so a
trained model estimates heart rate instead.

## Files

| File | What it does |
|---|---|
| `../Arduino/data_logger/data_logger.ino` | Band firmware: streams raw PPG + IMU over USB, 100×/s |
| `record.py` | **Step 1.** Guides you through the activities, saves band + chest strap data |
| `features.py` | **Step 2.** Cuts recordings into 6-s windows and computes features |
| `train.py` | **Step 3.** Trains the classifier and the two heart rate models, tests them |
| `predict.py` | **Step 4.** Runs the trained pipeline on a new recording |
| `common.py` | Settings, filters and the final pipeline, shared by the scripts above |
| `simulate.py` | Fake recordings, to test the code without hardware |

## Setup

```bash
cd ml
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Test it now (no hardware)

```bash
.venv/bin/python simulate.py          # 8 fake recordings -> data/raw/sim_*
.venv/bin/python features.py          # -> data/windows.csv
.venv/bin/python train.py             # -> models/, reports/
.venv/bin/python predict.py data/raw/sim_05
```

The fake data is much cleaner than a real wrist, so these scores only prove the code runs.
**Delete `data/raw/sim_*` before training on real recordings.**

## When the band is rebuilt

1. Upload `Arduino/data_logger/data_logger.ino`. The Serial Monitor should show lines like
   `1230,52011,88340,0.0123,-0.0210,0.9981,1.23,-0.45,0.30`.
2. Get a **Bluetooth chest strap** (Polar H10 is best; Garmin, Wahoo, Coospo also work). It
   gives the true heart rate the models learn from. The paper used an ECG chest strap too.
3. Short test run: `.venv/bin/python record.py joe --quick`
4. Record **at least 5–6 full sessions** (~30 min each), ideally on different days or with
   different people: `.venv/bin/python record.py joe`. The band is on a USB cable, so carry the
   laptop when walking.
5. Check `STILL_MAX_ACC_STD` / `STILL_MAX_GYRO_STD` in `features.py` against the
   `acc__std` / `gyro__std` values of your rest windows in `data/windows.csv`.
6. Run `features.py`, then `train.py`, and read `reports/report.txt` and the plots.

## Activities (paper Table 1)

| Activity | Minutes | Class |
|---|---|---|
| Rest | 5 | rest |
| Read (turn a page every ~5 s) | 3 | hand motion |
| Rest | 2 | rest |
| Walk slow | 5 | walking |
| Rest | 2 | rest |
| Write / draw | 3 | hand motion |
| Rest | 2 | rest |
| Walk fast | 5 | walking |
| Rest | 2 | rest |

## Differences from the paper

| Paper | Here | Why |
|---|---|---|
| Green + IR PPG, body temperature, accel, gyro | IR PPG, accel, gyro | This band's sensors |
| Their regression models used only temperature features | Adds one feature: `rest_hr`, the wearer's recent resting heart rate (FFT heart rate of the last still windows) | Without temperature, the IMU only says how much you're moving, not what *your* heart rate is. The model learns "resting HR + rise for this activity" |
| Random 70/30 split of windows | 70/30 split of whole recordings | Overlapping windows from one recording are near-copies, so a random split makes scores look better than they really are |
| ReliefF feature selection | Mutual information | Simpler, built into scikit-learn |
| Classifier: AdaBoost, 492 trees | Bagged trees and 4 other types, best one kept | Simpler |
| Gaussian process model for B.1 | Linear / tree / SVM / bagged trees, best one kept | Gaussian processes get very slow with many windows |
| PPG 20 Hz, IMU 10 Hz | Both 100 Hz | Better timing, no cost over USB |
