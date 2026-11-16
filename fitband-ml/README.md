# fitband-ml

Estimating heart rate from a wrist PPG sensor while the wearer moves, using IMU motion data and
machine learning. CU Boulder machine learning class project.

**Approach** (based on Vicente-Samper et al., 2023):
1. Classify the activity of each 6-second window: resting, typing, walking or running.
2. Estimate heart rate: FFT of the band-pass filtered PPG for resting windows, a regression
   model for moving windows.

**Baseline:** the FFT applied to every window. **Metric:** RMSE in BPM against a chest strap.
**Split:** by session, never by random window.

## Hardware

| Part | Role |
|---|---|
| Seeed XIAO nRF52840 | Microcontroller with Bluetooth LE |
| MAX30102 | PPG (heart rate) sensor, I2C address 0x57 |
| LSM6DS3 | Accelerometer + gyroscope, I2C address 0x6B |
| Garmin HRM 200 | Chest strap, ground-truth heart rate |

Wiring: both sensors on SDA → D4, SCL → D5, 3V3, GND.

## Project layout

```
firmware/fit_band_ble/   band sketch: streams CSV lines over Bluetooth (Nordic UART)
collect/collect.py       records the band + chest strap together on one computer
analysis/label.py        cuts recordings into labelled 6-s windows -> data/processed/windows.csv
analysis/                the modelling pipeline (see "Analysis pipeline" below)
data/raw/sessionN/       the recordings (sessions 1-5, committed)
results/                 tables, figures and numbers produced by the pipeline
docs/                    proposal, data collection guide, design_choices.md, reference paper
DECLARED_CHANGES.md      every deviation from the locked design
AI_LOG.md                bugs and wrong assumptions hit while building, and how they were caught
```

## One-time setup

1. Upload `firmware/fit_band_ble/fit_band_ble.ino` in the Arduino IDE
   (board: *Seeed nRF52 Boards → Seeed XIAO nRF52840*). The Serial Monitor (115200) should
   show CSV lines.
2. Install Python packages:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **macOS:** give Terminal (or VS Code, if you run it there) Bluetooth permission:
   *System Settings → Privacy & Security → Bluetooth*. Without it, no devices are found.

## 60-second test run

Wear the band and the strap (wet the strap's electrodes), then:

```bash
python collect/collect.py test 60
```

Every 5 s it prints the row counts. Expect ~10 band rows per second and ~2 strap rows per
second. Two files appear in `data/raw/`. Delete them when you're done.

## Full session protocol

Do each block as its own recording, with about **2 minutes of rest between blocks**:

```bash
python collect/collect.py resting 300    # sit still, hand resting
python collect/collect.py typing 300     # type on a keyboard
python collect/collect.py walking 300    # normal pace
python collect/collect.py running 300    # steady pace
```

After each session, move that session's 8 files into a new folder:

```bash
mkdir data/raw/session1 && mv data/raw/*.csv data/raw/session1/
```

Record **at least 3 sessions on different days**. Then build the window table:

```bash
python analysis/label.py
```

## Analysis pipeline

Run these three commands, in this order, from the `fitband-ml` folder:

```bash
python analysis/label.py      # raw recordings -> data/processed/windows.csv
python analysis/run_all.py    # all development results (sessions 1-4 only)
python analysis/final.py      # the one-time test on session 5
```

- `run_all.py` writes tables to `results/tables/`, figures to `results/figures/`
  and every number worth quoting to `results/numbers_for_report.md`. It never
  loads session 5. It takes about 1.5 minutes.
- **The headline number is the session-5 result from `final.py`**
  (`results/final/final_summary.md`). It trains on sessions 1-4 and tests on
  session 5 once, then writes `results/FINAL_RUN_DONE` and refuses to run again
  unless given `--force` (which is logged in `DECLARED_CHANGES.md`).

| Script | What it does |
|---|---|
| `config.py` | seed (42), feature lists, paths, tuning grids, frozen choices |
| `data.py` | loads windows.csv; removes session 5 unless called from final.py |
| `models.py` | shared pipelines, leave-one-session-out, tuning loop, k-means pace feature |
| `stage1_activity.py` | activity classifier: majority, one-feature rule, depth-3 tree, random forest |
| `stage2_bpm.py` | heart rate: baselines, ridge (PCA) and gradient boosting, 3 ablations |
| `unsupervised.py` | PCA with vs without, scree plot, loadings; k-means vs activity and pace |
| `side_experiment.py` | train sessions 1-3, test session 4 (varied pace) |
| `failure_analysis.py` | slice RMSE with block-bootstrap intervals; within-block proxy check |
| `final.py` | train 1-4, test 5, once |
| `run_all.py` | runs everything except final.py, plus sanity and leakage checks |

Sessions 1-3 kept a constant pace. In sessions 4-5, walking and running changed
pace every 100 s (session 4: low → high; session 5: medium → hard → easy).

## CSV columns

**`band_<activity>_<YYYYMMDD_HHMMSS>.csv`**, about 10 rows per second:

| Column | Meaning |
|---|---|
| `pc_time` | Computer clock when the row arrived (seconds since 1970). Used to line up with the strap |
| `ms` | Band clock (ms since power-on). Used to space samples evenly |
| `ppg` | Raw PPG light level, averaged since the previous row |
| `pulse` | Band's own filtered pulse wave |
| `ax_mg`, `ay_mg`, `az_mg` | Acceleration (milli-g) |
| `gx_dps`, `gy_dps`, `gz_dps` | Rotation speed (degrees/second) |
| `motion` | Band's motion level: 0 still, 1 moderate, 2 heavy |
| `bpm` | Band's own heart rate estimate (0 = no reading) |
| `status` | 0 no skin contact, 1 searching, 2 ok, 3 too much motion |
| `steps` | Steps since the band was powered on |

**`hrm_<activity>_<YYYYMMDD_HHMMSS>.csv`**, about 2 rows per second:

| Column | Meaning |
|---|---|
| `pc_time` | Computer clock when the packet arrived (seconds since 1970) |
| `hr` | Strap heart rate (BPM) |
| `rr_ms` | Beat-to-beat intervals in ms, oldest first, separated by `;`. Can be empty. The HRM 200 often repeats the previous packet's values; `label.py` skips repeats |

**`data/processed/windows.csv`**, one row per 6-second window: `session`, `activity`, `window`,
40 features (`<signal>_<mean|std|skew|kurt|energy>` for ppg, ax, ay, az, gx, gy, gz, acc_mag),
`hr_truth` (mean strap hr), `hr_truth_rr` (from RR intervals), `fft_bpm` (baseline),
`device_bpm` (band's own estimate), `pace_step` (0/1/2, the 100-s pace step;
walking/running in sessions 4-5 only, otherwise blank) and `transition`
(True if the window starts within 30 s after a pace change).
