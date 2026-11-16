"""
final.py — the one-time test on session 5.

    python analysis/final.py            (refuses if it has already run)
    python analysis/final.py --force    (runs again; the rerun is logged in
                                         DECLARED_CHANGES.md)

Trains on sessions 1-4 with the settings frozen during development
(config.py: STAGE1_MODEL, HEADLINE_MODEL, grids, seed) and tests on session 5
exactly once. The tuning PROCEDURE is frozen: hyperparameters are chosen by
leave-one-session-out inside sessions 1-4, exactly as in every development
fold. Session 5 is never used to fit or choose anything.

Writes results/final/ and the marker file results/FINAL_RUN_DONE.
This file is the only script allowed to load session 5.
"""

import sys
from datetime import datetime

import pandas as pd
from sklearn.metrics import f1_score

import config
import data
import failure_analysis
import models
import stage1_activity
import stage2_bpm


# =============================================================================
#  RUN-ONCE GUARD
# =============================================================================
def check_run_once():
    force = "--force" in sys.argv
    if config.FINAL_DONE_FLAG.exists() and not force:
        sys.exit(f"final.py has already been run ({config.FINAL_DONE_FLAG.read_text().strip()}).\n"
                 "Session 5 is a one-time test. Use --force only if you must, and it will be logged.")
    if config.FINAL_DONE_FLAG.exists() and force:
        with open(config.DECLARED_CHANGES_FILE, "a") as file:
            file.write(f"\n- {datetime.now():%Y-%m-%d %H:%M}: final.py was run AGAIN with --force. "
                       "Explain why here: ...\n")
        print("Forced rerun: logged in DECLARED_CHANGES.md")


# =============================================================================
#  MAIN
# =============================================================================
def main():
    check_run_once()

    windows = data.load_windows(include_final=True)
    train = windows[windows["session"].isin(config.DEV_SESSIONS)]
    test = windows[windows["session"] == config.FINAL_SESSION]
    assert len(test) > 0, "no session 5 windows found"
    config.FINAL_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1 on session 5 ----
    predicted_activity = models.fit_predict_activity(config.STAGE1_MODEL, train, test)
    stage1_f1 = f1_score(test["activity"], predicted_activity, labels=config.ACTIVITIES,
                         average="macro", zero_division=0)
    confusion = stage1_activity.confusion_table(test["activity"], predicted_activity)

    # ---- Stage 2 on session 5 (all baselines + both models) ----
    predictions, settings = stage2_bpm.predict_split(train, test)
    table = stage2_bpm.results_table(predictions)

    # ---- Slices and proxy check for the headline model ----
    headline = config.HEADLINE_MODEL
    slices = failure_analysis.slice_table(predictions, headline)
    proxy = failure_analysis.proxy_table(predictions, failure_analysis.proxy_columns())

    # ---- Save ----
    table.to_csv(config.FINAL_DIR / "final_results.csv", index=False, float_format="%.2f")
    slices.to_csv(config.FINAL_DIR / "final_slices.csv", index=False, float_format="%.2f")
    proxy.to_csv(config.FINAL_DIR / "final_proxy_check.csv", index=False, float_format="%.3f")
    predictions.to_csv(config.FINAL_DIR / "final_predictions.csv", index=False, float_format="%.3f")
    confusion.to_csv(config.FINAL_DIR / "final_stage1_confusion.csv")

    def as_text(frame, index=False):
        """Plain-text table inside a code block (no extra packages needed)."""
        return "```\n" + frame.to_string(index=index) + "\n```"

    report = [
        "# Final test: train sessions 1-4, test session 5 (run once)",
        f"\nRun at {datetime.now():%Y-%m-%d %H:%M}. Headline model: {headline}. "
        f"Stage 1 classifier: {config.STAGE1_MODEL}.",
        f"\nChosen settings (tuned inside sessions 1-4): {settings}",
        f"\n## Stage 1\n\nMacro-F1 on session 5: {stage1_f1:.3f}\n\n{as_text(confusion, index=True)}",
        f"\n## Stage 2 (RMSE, BPM)\n\n{as_text(table.round(2))}",
        f"\n## Slices ({headline})\n\n{as_text(slices.round(2))}",
        "\nSession 5 has only 4 blocks (one per activity), so the block-bootstrap intervals are wide.",
        f"\n## Proxy check (within-block correlation)\n\n{as_text(proxy.round(3))}",
    ]
    (config.FINAL_DIR / "final_summary.md").write_text("\n".join(report) + "\n")

    print("\n".join(report))
    config.FINAL_DONE_FLAG.write_text(f"run at {datetime.now():%Y-%m-%d %H:%M}\n")
    print(f"\nSaved to {config.FINAL_DIR}. Wrote {config.FINAL_DONE_FLAG}.")


if __name__ == "__main__":
    main()
