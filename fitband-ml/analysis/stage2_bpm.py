"""
stage2_bpm.py — Stage 2: heart rate (BPM) for each 6-second window.

    python analysis/stage2_bpm.py

Target: hr_truth (chest strap). Leave-one-session-out on sessions 1-4.

Baselines reported beside every result:
  fft                   FFT of the band-passed PPG, on every window
  activity_mean         average training heart rate of the PREDICTED activity
  activity_mean_oracle  same, but with the TRUE activity (not available in
                        real use; shows what perfect Stage 1 would give)
  device                the band's own firmware estimate (sometimes missing)

Two models, same splits and metric:
  ridge  StandardScaler -> PCA -> Ridge (readable)
  gbm    GradientBoostingRegressor (flexible)
Inputs: one-hot predicted activity + 35 IMU features + 5 PPG features.

Ablations (clearly separate from the primary result):
  a) with_fft      fft_bpm added as an input
  b) residual      predict the difference from the predicted activity's mean
  c) pace_cluster  k-means pace level added as an input
"""

import pandas as pd

import config
import data
import models
import plots

MODEL_NAMES = ["ridge", "gbm"]
ABLATIONS = {
    "with_fft": "a) + fft_bpm as an input",
    "residual": "b) predict difference from activity mean",
    "pace_cluster": "c) + k-means pace level as an input",
}

# Rows of every results table: (label, prediction column)
METHODS = [
    ("Baseline: FFT on every window", "fft"),
    ("Baseline: activity mean (predicted activity)", "activity_mean"),
    ("Baseline: activity mean (true activity, ORACLE)", "activity_mean_oracle"),
    ("Baseline: band's device_bpm", "device"),
    ("Model: scaler -> PCA -> ridge (readable)", "ridge"),
    ("Model: gradient boosting (flexible)", "gbm"),
]

# Columns copied from the windows into every predictions table
KEEP_COLUMNS = ["session", "activity", "window", config.TARGET, "hr_truth_rr",
                "pace_step", "transition", "acc_mag_std", "ppg_mean"]


# =============================================================================
#  INPUTS FOR ONE SPLIT
# =============================================================================
def build_inputs(train, test, train_activity, test_activity, variant):
    """Feature tables for Stage 2. Nothing here looks at hr_truth."""
    sensor_columns = config.IMU_FEATURES + config.PPG_FEATURES
    X_train = pd.concat([models.one_hot_activity(train_activity), train[sensor_columns]], axis=1)
    X_test = pd.concat([models.one_hot_activity(test_activity), test[sensor_columns]], axis=1)

    if variant == "with_fft":
        X_train["fft_bpm"] = train["fft_bpm"]
        X_test["fft_bpm"] = test["fft_bpm"]
    if variant == "pace_cluster":
        X_train["pace_level"], X_test["pace_level"] = models.pace_cluster_feature(
            train, train_activity, test, test_activity)
    return X_train, X_test


# =============================================================================
#  ONE SPLIT: train on `train`, predict `test` with every baseline and model
# =============================================================================
def predict_split(train, test, variant="primary", model_names=MODEL_NAMES):
    """
    Returns (predictions table for the test windows, chosen settings).
    Used for every fold here, by the side experiment and by final.py.
    """
    # Stage 1 output: out-of-fold for training windows, normal for test windows.
    train_activity, test_activity = models.predicted_activity_for_split(train, test)

    # Average heart rate of each activity, learned from the training sessions only.
    activity_means = train.groupby("activity")[config.TARGET].mean()

    predictions = test[KEEP_COLUMNS].copy()
    predictions["predicted_activity"] = test_activity
    predictions["fft"] = test["fft_bpm"]
    predictions["device"] = test["device_bpm"]
    predictions["activity_mean"] = test_activity.map(activity_means)
    predictions["activity_mean_oracle"] = test["activity"].map(activity_means)

    X_train, X_test = build_inputs(train, test, train_activity, test_activity, variant)
    y_train = train[config.TARGET]
    if variant == "residual":
        # Learn only what the activity mean misses (e.g. effort within walking).
        y_train = y_train - train_activity.map(activity_means)

    chosen_settings = {}
    for name in model_names:
        model, settings = models.tune_and_fit(name, X_train, y_train, train["session"])
        predicted = model.predict(X_test)
        if variant == "residual":
            predicted = predicted + predictions["activity_mean"].to_numpy()
        predictions[name] = predicted
        chosen_settings[name] = settings
    return predictions, chosen_settings


def leave_one_session_out(windows, variant="primary", model_names=MODEL_NAMES):
    """Run predict_split once per held-out session and stack the results."""
    all_predictions, settings_rows = [], []
    for held_out in sorted(windows["session"].unique()):
        train = windows[windows["session"] != held_out]
        test = windows[windows["session"] == held_out]
        predictions, settings = predict_split(train, test, variant, model_names)
        all_predictions.append(predictions)
        for name, chosen in settings.items():
            settings_rows.append({"held_out": held_out, "variant": variant, "model": name, **chosen})
    return pd.concat(all_predictions), pd.DataFrame(settings_rows)


# =============================================================================
#  RESULTS TABLE
# =============================================================================
def results_table(predictions, methods=METHODS):
    """RMSE overall and per activity for every method; coverage for device_bpm."""
    rows = []
    for label, column in methods:
        row = {"method": label,
               "rmse_all": models.rmse(predictions[column], predictions[config.TARGET]),
               "coverage_pct": 100 * predictions[column].notna().mean()}
        for activity in config.ACTIVITIES:
            part = predictions[predictions["activity"] == activity]
            row[f"rmse_{activity}"] = models.rmse(part[column], part[config.TARGET])
        rows.append(row)
    return pd.DataFrame(rows)


def per_session_table(predictions, columns=("fft", "activity_mean", "ridge", "gbm")):
    table = predictions.groupby("session").apply(
        lambda part: pd.Series({c: models.rmse(part[c], part[config.TARGET]) for c in columns}))
    return table


# =============================================================================
#  MAIN
# =============================================================================
def main():
    development = data.load_windows()
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Primary result ----
    predictions, settings = leave_one_session_out(development)
    table = results_table(predictions)
    table.to_csv(config.TABLES_DIR / "stage2_results.csv", index=False, float_format="%.2f")
    predictions.to_csv(config.TABLES_DIR / "stage2_oof_predictions.csv", index=False, float_format="%.3f")
    sessions = per_session_table(predictions)
    sessions.to_csv(config.TABLES_DIR / "stage2_rmse_per_session.csv", float_format="%.2f")

    # ---- Ablations ----
    ablation_rows = [{"variant": "primary (no change)", "ridge": models.rmse(predictions["ridge"], predictions[config.TARGET]),
                      "gbm": models.rmse(predictions["gbm"], predictions[config.TARGET])}]
    all_settings = [settings]
    for variant, label in ABLATIONS.items():
        variant_predictions, variant_settings = leave_one_session_out(development, variant)
        all_settings.append(variant_settings)
        ablation_rows.append({"variant": label,
                              "ridge": models.rmse(variant_predictions["ridge"], variant_predictions[config.TARGET]),
                              "gbm": models.rmse(variant_predictions["gbm"], variant_predictions[config.TARGET])})
    ablations = pd.DataFrame(ablation_rows)
    ablations.to_csv(config.TABLES_DIR / "stage2_ablations.csv", index=False, float_format="%.2f")
    pd.concat(all_settings).to_csv(config.TABLES_DIR / "stage2_chosen_settings.csv", index=False)

    headline = config.HEADLINE_MODEL
    plots.predicted_vs_true_figure(predictions, headline,
                                   f"{headline}: predicted vs true (sessions 1-4, held-out)",
                                   config.FIGURES_DIR / "stage2_predicted_vs_true.png")

    print("Stage 2: heart rate, leave-one-session-out on sessions 1-4 (RMSE, BPM)")
    print(table.round(2).to_string(index=False))
    print(f"\nRMSE per held-out session:\n{sessions.round(2)}")
    print(f"\nAblations (overall RMSE):\n{ablations.round(2).to_string(index=False)}")
    print(f"\nChosen settings per fold:\n{settings.to_string(index=False)}")

    numbers = []
    for row in table.itertuples():
        text = f"{row.rmse_all:.2f} BPM"
        if row.coverage_pct < 100:
            text += f" (coverage {row.coverage_pct:.0f}% of windows)"
        numbers.append((f"Stage 2 LOSO RMSE, {row.method}", text))
        numbers.append((f"Stage 2 LOSO RMSE by activity, {row.method}",
                        ", ".join(f"{a} {getattr(row, 'rmse_' + a):.2f}" for a in config.ACTIVITIES)))
    for row in ablations.itertuples():
        numbers.append((f"Stage 2 ablation RMSE, {row.variant}", f"ridge {row.ridge:.2f}, gbm {row.gbm:.2f}"))
    numbers += ablation_findings(ablations)

    # A fact from the table above, not a new model: the FFT beats both models at rest.
    resting = table.set_index("method")["rmse_resting"]
    numbers.append(("Observation: resting RMSE, FFT vs models",
                    f"FFT {resting.iloc[0]:.2f}, ridge {resting.iloc[4]:.2f}, gbm {resting.iloc[5]:.2f} BPM"))
    return numbers


def ablation_findings(ablations, meaningful_bpm=0.5):
    """
    One plain sentence per ablation. A change smaller than 0.5 BPM is called
    "no real improvement": the RMSE moves by several BPM from one held-out
    session to the next, so a smaller change is indistinguishable from noise.
    """
    primary = ablations.iloc[0]
    findings = []
    for row in ablations.iloc[1:].itertuples():
        change = {name: getattr(row, name) - primary[name] for name in MODEL_NAMES}
        if all(c > -meaningful_bpm for c in change.values()):
            text = (f"Checked '{row.variant}': it changed RMSE by {change['ridge']:+.2f} (ridge) and "
                    f"{change['gbm']:+.2f} (gbm) BPM, less than {meaningful_bpm} BPM, so it did not "
                    "meaningfully help and is not used in the primary model.")
        else:
            text = (f"Checked '{row.variant}': it changed RMSE by {change['ridge']:+.2f} (ridge) and "
                    f"{change['gbm']:+.2f} (gbm) BPM. Reported as an ablation only; the primary model is unchanged.")
        findings.append(("Finding", text))
    return findings


if __name__ == "__main__":
    main()
