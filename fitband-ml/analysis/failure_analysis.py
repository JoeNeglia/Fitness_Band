"""
failure_analysis.py — where does the headline model go wrong?

    python analysis/failure_analysis.py      (needs stage2_bpm.py to have run)

Works on the held-out (out-of-fold) predictions of sessions 1-4.

Slices, decided BEFORE looking at results ("pre-specified"):
  heart rate >= 100 vs < 100, session, pace step, quartile of acc_mag_std
  (how much the wrist moves), quartile of ppg_mean (sensor contact), and
  the label-noise flag (strap hr and RR-based hr differ by > 5 BPM).
Activity is shown for context only: Stage 2 uses the PREDICTED activity as
an input, so it is not counted when naming the worst slice. Any slice added
after looking at results is marked "exploratory".

Each slice gets: windows, 5-minute blocks, RMSE and a 95% interval from a
BLOCK bootstrap. Neighbouring windows are near-copies, so resampling single
windows would pretend we have far more independent data than we do; we
resample whole (session, activity) blocks instead.

The proxy check asks: inside one block, does the prediction move up and
down with the true heart rate, or is the model only getting the activity
right?
"""

import sys

import numpy as np
import pandas as pd

import config
import models
import plots


# =============================================================================
#  SLICES
# =============================================================================
def define_slices(predictions):
    """Returns a list of (variable, level, kind, boolean mask)."""
    slices = []
    truth = predictions[config.TARGET]

    slices.append(("heart rate", f">= {config.HIGH_HR_BPM}", "pre-specified", truth >= config.HIGH_HR_BPM))
    slices.append(("heart rate", f"< {config.HIGH_HR_BPM}", "pre-specified", truth < config.HIGH_HR_BPM))

    for session in sorted(predictions["session"].unique()):
        slices.append(("session", session, "pre-specified", predictions["session"] == session))

    for step in sorted(predictions["pace_step"].dropna().unique()):
        slices.append(("pace step", int(step), "pre-specified", predictions["pace_step"] == step))

    for column, name in [("acc_mag_std", "acc_mag_std quartile"), ("ppg_mean", "ppg_mean quartile")]:
        quartile = pd.qcut(predictions[column], 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
        for level in quartile.cat.categories:
            slices.append((name, level, "pre-specified", quartile == level))

    noise = (predictions[config.TARGET] - predictions["hr_truth_rr"]).abs() > config.LABEL_NOISE_BPM
    slices.append(("label noise", f"|hr - hr_rr| > {config.LABEL_NOISE_BPM}", "pre-specified", noise))
    slices.append(("label noise", "clean", "pre-specified", ~noise))

    for activity in config.ACTIVITIES:
        slices.append(("activity (context)", activity, "context", predictions["activity"] == activity))

    # Added after seeing that heart rate lags behind pace changes: exploratory.
    slices.append(("transition (<30 s after pace change)", "yes", "exploratory", predictions["transition"]))
    return slices


# =============================================================================
#  BLOCK BOOTSTRAP
# =============================================================================
def block_bootstrap_interval(predicted, truth, blocks, rng):
    """95% interval of the RMSE, resampling whole blocks with replacement."""
    errors = (np.asarray(predicted, dtype=float) - np.asarray(truth, dtype=float)) ** 2
    squared_by_block = [errors[blocks == block] for block in np.unique(blocks)]
    if len(squared_by_block) < 2:
        return np.nan, np.nan             # one block: nothing to resample
    rmse_samples = np.empty(config.BOOTSTRAP_REPS)
    for rep in range(config.BOOTSTRAP_REPS):
        picked = rng.integers(0, len(squared_by_block), len(squared_by_block))
        rmse_samples[rep] = np.sqrt(np.concatenate([squared_by_block[i] for i in picked]).mean())
    return np.percentile(rmse_samples, [2.5, 97.5])


def slice_table(predictions, model_column):
    rng = np.random.default_rng(config.SEED)
    blocks = (predictions["session"] + "/" + predictions["activity"]).to_numpy()
    rows = []
    for variable, level, kind, mask in define_slices(predictions):
        mask = mask.to_numpy(dtype=bool)
        if mask.sum() == 0:
            continue
        part_pred = predictions[model_column].to_numpy()[mask]
        part_truth = predictions[config.TARGET].to_numpy()[mask]
        low, high = block_bootstrap_interval(part_pred, part_truth, blocks[mask], rng)
        rows.append({"variable": variable, "level": level, "kind": kind,
                     "windows": int(mask.sum()), "blocks": len(np.unique(blocks[mask])),
                     "rmse": models.rmse(part_pred, part_truth), "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def worst_slice(table):
    candidates = table[(table["kind"] == "pre-specified") & (table["windows"] >= config.MIN_SLICE_WINDOWS)]
    return candidates.loc[candidates["rmse"].idxmax()]


# =============================================================================
#  PROXY CHECK: does the prediction follow heart rate INSIDE a block?
# =============================================================================
def within_block_correlation(predictions, column):
    blocks = predictions["session"] + "/" + predictions["activity"]
    predicted = predictions[column] - predictions.groupby(blocks)[column].transform("mean")
    truth = predictions[config.TARGET] - predictions.groupby(blocks)[config.TARGET].transform("mean")
    if predicted.std() < 1e-9:
        return 0.0                         # a constant inside every block cannot track anything
    return float(np.corrcoef(predicted, truth)[0, 1])


def proxy_columns():
    """The headline model first, then the plain models and the two baselines."""
    columns = [config.HEADLINE_MODEL, "gbm", "ridge", "fft", "activity_mean"]
    return list(dict.fromkeys(columns))       # drop a repeat if the headline is a plain model


def proxy_table(predictions, columns):
    return pd.DataFrame([{"method": c, "within_block_correlation": within_block_correlation(predictions, c)}
                         for c in columns])


# =============================================================================
#  MAIN
# =============================================================================
def main():
    path = config.TABLES_DIR / "stage2_oof_predictions.csv"
    if not path.exists():
        sys.exit("Run python analysis/stage2_bpm.py (or run_all.py) first.")
    predictions = pd.read_csv(path)
    predictions["transition"] = predictions["transition"].astype(bool)
    headline = config.HEADLINE_MODEL

    table = slice_table(predictions, headline)
    worst = worst_slice(table)
    proxy = proxy_table(predictions, proxy_columns())

    table.to_csv(config.TABLES_DIR / "failure_slices.csv", index=False, float_format="%.2f")
    proxy.to_csv(config.TABLES_DIR / "proxy_check.csv", index=False, float_format="%.3f")
    plots.slice_rmse_figure(table, f"{headline}: RMSE by slice (sessions 1-4, held-out)",
                            config.FIGURES_DIR / "slice_rmse.png")

    print(f"Failure analysis of '{headline}' on held-out windows of sessions 1-4")
    print(table.round(2).to_string(index=False))
    print(f"\nWorst pre-specified slice (>= {config.MIN_SLICE_WINDOWS} windows): "
          f"{worst['variable']} = {worst['level']}, RMSE {worst['rmse']:.2f}")
    print(f"\nProxy check (correlation of block-centred prediction and truth):\n{proxy.round(3).to_string(index=False)}")

    numbers = [(f"Worst pre-specified slice, {headline}",
                f"{worst['variable']} = {worst['level']}: RMSE {worst['rmse']:.2f} "
                f"[{worst['ci_low']:.2f}, {worst['ci_high']:.2f}], {worst['windows']} windows, {worst['blocks']} blocks")]
    for row in table.itertuples():
        numbers.append((f"Slice RMSE ({row.kind}), {row.variable} = {row.level}",
                        f"{row.rmse:.2f} [{row.ci_low:.2f}, {row.ci_high:.2f}], n={row.windows}, blocks={row.blocks}"))
    for row in proxy.itertuples():
        numbers.append((f"Within-block correlation, {row.method}", f"{row.within_block_correlation:.3f}"))
    return numbers


if __name__ == "__main__":
    main()
