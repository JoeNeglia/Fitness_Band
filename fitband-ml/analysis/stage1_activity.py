"""
stage1_activity.py — Stage 1: which activity is each 6-second window?

    python analysis/stage1_activity.py

Uses IMU features only (the PPG is reserved for heart rate). Compares four
classifiers with leave-one-session-out on sessions 1-4:

  majority       always guesses the most common activity (the floor)
  acc_rule       a depth-2 tree on acc_mag_std only (one obvious rule)
  tree_depth3    a depth-3 decision tree (the readable model)
  random_forest  200 trees (the flexible model)

Metric: macro-F1, the average of the F1 scores of the four activities, so
every activity counts equally. Saves the out-of-fold predictions; the
classifier named in config.STAGE1_MODEL is the one Stage 2 receives.
"""

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.tree import export_text

import config
import data
import models
import plots

ACTIVITY_MODELS = ["majority", "acc_rule", "tree_depth3", "random_forest"]


# =============================================================================
#  OUT-OF-FOLD PREDICTIONS
# =============================================================================
def out_of_fold_activity(windows, name):
    """Each session is predicted by a classifier trained on the other sessions."""
    predicted = pd.Series(index=windows.index, dtype=object)
    for fit_rows, held_rows in models.leave_one_session_out(windows["session"]):
        predicted.iloc[held_rows] = models.fit_predict_activity(
            name, windows.iloc[fit_rows], windows.iloc[held_rows])
    return predicted


def macro_f1(truth, predicted):
    return f1_score(truth, predicted, labels=config.ACTIVITIES, average="macro", zero_division=0)


def confusion_table(truth, predicted):
    matrix = confusion_matrix(truth, predicted, labels=config.ACTIVITIES)
    return pd.DataFrame(matrix,
                        index=[f"true {a}" for a in config.ACTIVITIES],
                        columns=[f"predicted {a}" for a in config.ACTIVITIES])


# =============================================================================
#  MAIN
# =============================================================================
def main():
    development = data.load_windows()
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    predictions = development[["session", "activity", "window"]].copy()
    summary_rows = []
    for name in ACTIVITY_MODELS:
        predicted = out_of_fold_activity(development, name)
        predictions[f"predicted_{name}"] = predicted

        # Macro-F1 of all held-out windows together, plus the range across
        # the four held-out sessions (shows how much it varies by day).
        per_session = [macro_f1(development.loc[rows, "activity"], predicted[rows])
                       for _, rows in development.groupby("session").groups.items()]
        summary_rows.append({"model": name,
                             "macro_f1": macro_f1(development["activity"], predicted),
                             "worst_session_f1": min(per_session),
                             "best_session_f1": max(per_session)})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(config.TABLES_DIR / "stage1_macro_f1.csv", index=False, float_format="%.3f")
    predictions.to_csv(config.TABLES_DIR / "stage1_oof_predictions.csv", index=False)

    confusion = confusion_table(development["activity"], predictions[f"predicted_{config.STAGE1_MODEL}"])
    confusion.to_csv(config.TABLES_DIR / f"stage1_confusion_{config.STAGE1_MODEL}.csv")
    tree_confusion = confusion_table(development["activity"], predictions["predicted_tree_depth3"])
    tree_confusion.to_csv(config.TABLES_DIR / "stage1_confusion_tree_depth3.csv")
    plots.confusion_matrix_figure(confusion, f"Stage 1 ({config.STAGE1_MODEL}), sessions 1-4 out-of-fold",
                                  config.FIGURES_DIR / "stage1_confusion_matrix.png")

    # The readable tree, trained on all four sessions, written out as rules.
    tree = models.make_activity_model("tree_depth3")
    tree.fit(development[config.IMU_FEATURES], development["activity"])
    rules = export_text(tree.named_steps["model"], feature_names=config.IMU_FEATURES)
    (config.TABLES_DIR / "stage1_tree_depth3_rules.txt").write_text(rules)

    print("Stage 1: activity classifier, leave-one-session-out on sessions 1-4")
    print(summary.round(3).to_string(index=False))
    print(f"\nConfusion matrix ({config.STAGE1_MODEL}):\n{confusion}")

    numbers = [(f"Stage 1 macro-F1, {row.model} (LOSO, sessions 1-4)",
                f"{row.macro_f1:.3f} (sessions range {row.worst_session_f1:.3f}-{row.best_session_f1:.3f})")
               for row in summary.itertuples()]
    numbers.append(("Stage 1 classifier passed to Stage 2", config.STAGE1_MODEL))
    return numbers


if __name__ == "__main__":
    main()
