"""
side_experiment.py — what happens when the pace varies?

    python analysis/side_experiment.py

Train on sessions 1-3 (constant pace) and test on session 4 (walking and
running changed pace every 100 s). Same baselines, models and tuning as
Stage 2; only the split is different. The test session is never used for
fitting or tuning.
"""

import pandas as pd

import config
import data
import models
import stage2_bpm


def rmse_per_pace_step(predictions, columns=("fft", "activity_mean", "ridge", "gbm")):
    """RMSE inside each 100-s pace step of the walking and running blocks."""
    moving = predictions[predictions["activity"].isin(config.MOVING_ACTIVITIES)]
    rows = []
    for (activity, step), part in moving.groupby(["activity", "pace_step"]):
        row = {"activity": activity, "pace_step": int(step), "windows": len(part),
               "mean_hr_truth": part[config.TARGET].mean()}
        for column in columns:
            row[column] = models.rmse(part[column], part[config.TARGET])
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    development = data.load_windows()
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    train = development[development["session"].isin(config.CONSTANT_PACE_SESSIONS)]
    test = development[development["session"] == config.VARIED_PACE_SESSION]
    predictions, settings = stage2_bpm.predict_split(train, test)

    table = stage2_bpm.results_table(predictions)
    steps = rmse_per_pace_step(predictions)
    table.to_csv(config.TABLES_DIR / "side_experiment_results.csv", index=False, float_format="%.2f")
    steps.to_csv(config.TABLES_DIR / "side_experiment_pace_steps.csv", index=False, float_format="%.2f")

    # How far beyond the training range does session 4 go?
    train_max = train.groupby("activity")[config.TARGET].max()
    test_max = test.groupby("activity")[config.TARGET].max()

    print("Side experiment: train sessions 1-3, test session 4 (RMSE, BPM)")
    print(table.round(2).to_string(index=False))
    print(f"\nPer pace step (walking/running):\n{steps.round(2).to_string(index=False)}")
    print(f"\nChosen settings: {settings}")
    print(f"\nHighest heart rate, train vs test:\n{pd.DataFrame({'train': train_max, 'test': test_max}).round(0)}")

    numbers = [(f"Side experiment (train 1-3, test 4) RMSE, {row.method}",
                f"{row.rmse_all:.2f} BPM" + (f" (coverage {row.coverage_pct:.0f}%)" if row.coverage_pct < 100 else ""))
               for row in table.itertuples()]
    for row in steps.itertuples():
        numbers.append((f"Side experiment, {row.activity} pace step {row.pace_step} (true HR {row.mean_hr_truth:.0f})",
                        f"FFT {row.fft:.2f}, activity mean {row.activity_mean:.2f}, ridge {row.ridge:.2f}, gbm {row.gbm:.2f}"))
    for activity in config.MOVING_ACTIVITIES:
        numbers.append((f"Highest {activity} HR, sessions 1-3 vs session 4",
                        f"{train_max[activity]:.0f} vs {test_max[activity]:.0f} BPM"))
    return numbers


if __name__ == "__main__":
    main()
