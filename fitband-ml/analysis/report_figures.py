"""
report_figures.py — figures and error rates for the written report.

    python analysis/report_figures.py      (after final.py has run)

Reads only the SAVED session-5 predictions (results/final/final_predictions.csv).
Nothing is fitted or re-run, so this does not touch the one-time test again.

Writes:
  results/figures/final_timeseries.png    truth vs predictions through the
                                          session-5 walking and running blocks
  results/figures/final_predicted_vs_true.png
  results/final/final_error_rates.csv     share of windows within +-5 / +-10 BPM,
                                          mean bias, and share under/over by > 10 BPM,
                                          for still (resting, typing) and moving windows
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import plots

METHODS = {config.HEADLINE_MODEL: "Headline (GBM + FFT at rest)", "ridge_fft_rest": "Ridge + FFT at rest",
           "activity_mean": "Activity mean (baseline)", "fft": "FFT on every window (baseline)"}


# =============================================================================
#  ERROR RATES THAT MATTER FOR THE DECISION
#  Underestimating heart rate during exercise is the costly error (the wearer
#  thinks they are working less hard than they are), so it is counted apart.
# =============================================================================
def error_rates(predictions):
    group = np.where(predictions["activity"].isin(config.MOVING_ACTIVITIES), "moving", "still")
    rows = []
    for name, part in list(predictions.groupby(group)) + [("all", predictions)]:
        for column, label in METHODS.items():
            error = part[column] - part[config.TARGET]
            rows.append({"windows": name, "method": label, "n": len(part),
                         "rmse": np.sqrt((error ** 2).mean()),
                         "pct_within_5": 100 * (error.abs() <= 5).mean(),
                         "pct_within_10": 100 * (error.abs() <= 10).mean(),
                         "mean_bias": error.mean(),
                         "pct_under_by_10": 100 * (error < -10).mean(),
                         "pct_over_by_10": 100 * (error > 10).mean()})
    return pd.DataFrame(rows)


# =============================================================================
#  FIGURE: heart rate through the walking and running blocks
# =============================================================================
def timeseries_figure(predictions, path):
    styles = {config.TARGET: ("Chest strap (truth)", "black", "-"),
              config.HEADLINE_MODEL: ("Headline (GBM + FFT at rest)", "#d62728", "-"),
              "activity_mean": ("Activity mean (baseline)", "#1f77b4", "--"),
              "fft": ("FFT (baseline)", "#999999", ":")}
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    for axis, activity in zip(axes, config.MOVING_ACTIVITIES):
        block = predictions[predictions["activity"] == activity].sort_values("window")
        seconds = block["window"] * 6
        for column, (label, color, line) in styles.items():
            axis.plot(seconds, block[column], color=color, linestyle=line, linewidth=1.6, label=label)
        for change in (100, 200):                       # the pace changes
            axis.axvline(change, color="gray", linewidth=0.8, alpha=0.5)
        axis.set_title(f"Session 5, {activity} (pace: medium, hard, easy)")
        axis.set_xlabel("Time in block (s)")
    axes[0].set_ylabel("Heart rate (BPM)")
    axes[1].legend(fontsize=8, loc="lower right")
    plots.save(figure, path)


# =============================================================================
#  MAIN
# =============================================================================
def main():
    path = config.FINAL_DIR / "final_predictions.csv"
    if not path.exists():
        sys.exit("Run python analysis/final.py first.")
    predictions = pd.read_csv(path)

    rates = error_rates(predictions)
    rates.to_csv(config.FINAL_DIR / "final_error_rates.csv", index=False, float_format="%.1f")
    timeseries_figure(predictions, config.FIGURES_DIR / "final_timeseries.png")
    plots.predicted_vs_true_figure(predictions, config.HEADLINE_MODEL, "Session 5: headline vs chest strap",
                                   config.FIGURES_DIR / "final_predicted_vs_true.png")
    print(rates.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
