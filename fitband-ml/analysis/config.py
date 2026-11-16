"""
config.py — every setting of the modelling pipeline in one place.

All scripts import from here, so a number (the random seed, a feature list,
a tuning grid) only ever has to be changed in one spot. The settings marked
FROZEN were decided on development data (sessions 1-4) and must not be
changed after final.py has been run.
"""

from pathlib import Path

import numpy as np

# =============================================================================
#  REPRODUCIBILITY
# =============================================================================
# Every random step (forests, boosting, k-means, bootstrap) uses this seed,
# so running the scripts twice gives exactly the same numbers.
SEED = 42

# =============================================================================
#  PATHS
# =============================================================================
PROJECT_DIR = Path(__file__).resolve().parent.parent
WINDOWS_CSV = PROJECT_DIR / "data" / "processed" / "windows.csv"
RESULTS_DIR = PROJECT_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
FINAL_DIR = RESULTS_DIR / "final"
FINAL_DONE_FLAG = RESULTS_DIR / "FINAL_RUN_DONE"
NUMBERS_FILE = RESULTS_DIR / "numbers_for_report.md"
DECLARED_CHANGES_FILE = PROJECT_DIR / "DECLARED_CHANGES.md"

# =============================================================================
#  DATA SPLIT
# =============================================================================
# Split by session, never by random window: windows next to each other are
# almost copies, so a random split would put near-identical windows in both
# training and test and make every model look better than it is.
DEV_SESSIONS = ["session1", "session2", "session3", "session4"]
FINAL_SESSION = "session5"          # the final test, used once by final.py
CONSTANT_PACE_SESSIONS = ["session1", "session2", "session3"]
VARIED_PACE_SESSION = "session4"    # side experiment: train 1-3, test 4

ACTIVITIES = ["resting", "typing", "walking", "running"]
MOVING_ACTIVITIES = ["walking", "running"]

# =============================================================================
#  FEATURES
# =============================================================================
STATS = ["mean", "std", "skew", "kurt", "energy"]
IMU_SIGNALS = ["ax", "ay", "az", "gx", "gy", "gz", "acc_mag"]
IMU_FEATURES = [f"{signal}_{stat}" for signal in IMU_SIGNALS for stat in STATS]   # 35
PPG_FEATURES = [f"ppg_{stat}" for stat in STATS]                                 # 5
ACTIVITY_ONE_HOT = [f"pred_{activity}" for activity in ACTIVITIES]                # 4

# K-means only looks at how much the wrist moves: the spread (std) and the
# 0.5-3 Hz energy of each IMU signal.
KMEANS_FEATURES = [f"{signal}_{stat}" for signal in IMU_SIGNALS for stat in ["std", "energy"]]

TARGET = "hr_truth"

# =============================================================================
#  TUNING GRIDS (kept tiny on purpose: with only 3-4 training sessions, a big
#  grid would mostly learn the noise of those few sessions)
# =============================================================================
PCA_COMPONENTS_GRID = [5, 8, 12]
RIDGE_ALPHAS = np.logspace(-2, 4, 13)       # RidgeCV picks one of these
GBM_MAX_DEPTH_GRID = [2, 3]
GBM_N_ESTIMATORS_GRID = [100, 200]
GBM_LEARNING_RATE = 0.05

STAGE1_TREE_DEPTH = 3                       # the readable activity classifier
STAGE1_RULE_DEPTH = 2                       # the one-feature rule on acc_mag_std
STAGE1_FOREST_TREES = 200

# =============================================================================
#  FROZEN CHOICES (decided on sessions 1-4, see results/numbers_for_report.md)
# =============================================================================
STAGE1_MODEL = "random_forest"      # activity classifier fed to Stage 2
HEADLINE_MODEL = "gbm_fft_rest"     # heart-rate model named in the report: GBM, but the
                                    # FFT for windows Stage 1 predicts as resting
REST_ACTIVITY = "resting"           # activity where the FFT replaces the model (see stage2_bpm.py)
PACE_K = 2                          # k-means clusters per moving activity (variant c)

# =============================================================================
#  FAILURE ANALYSIS
# =============================================================================
HIGH_HR_BPM = 100                   # slice: heart rate >= 100 vs < 100
LABEL_NOISE_BPM = 5                 # |hr_truth - hr_truth_rr| above this = noisy label
MIN_SLICE_WINDOWS = 30              # smallest slice allowed to be named "worst"
BOOTSTRAP_REPS = 2000
