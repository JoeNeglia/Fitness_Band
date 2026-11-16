# Declared changes

Everything added to, or interpreted in, the locked design (session split,
session 5 touched once, RMSE and macro-F1, FFT baseline). Each entry says
what was done and why.

## Interpretations of the specification

1. **Predicted activity for Stage 2 is nested inside each fold.** For training
   windows, Stage 2 receives out-of-fold predictions made by leave-one-session-out
   *within that fold's training sessions*. Test windows are predicted by a Stage 1
   model fitted on all training sessions. Reusing the single file of Stage 1
   out-of-fold predictions would have used a classifier trained partly on the
   held-out session to make the training inputs. The file
   (`results/tables/stage1_oof_predictions.csv`) is still saved, as requested.
2. **RidgeCV uses session-based folds.** By default RidgeCV chooses alpha by
   leaving out one *window* at a time, which is the random-window split the design
   forbids. It is given leave-one-session-out folds instead. Alpha candidates:
   13 values from 0.01 to 10,000. The tuning loop itself is only the 3 PCA sizes.
   Scaling and PCA are fitted before RidgeCV's internal folds, so the alpha choice
   (and only that) sees the scaling of all training sessions.
3. **The k-means input (ablation c) is a "pace level".** One k-means per moving
   activity is fitted on the training windows of that true activity. Clusters are
   numbered 1..k from the gentlest to the hardest movement (by mean acc_mag_std);
   0 means "not walking or running". Windows are assigned using the *predicted*
   activity, as they would be in real use. k = 2 was chosen because it had the
   highest silhouette for both walking and running.
4. **The activity slice uses the true activity** and is labelled "context". It is
   excluded when naming the worst slice, because Stage 2 receives the predicted
   activity as an input.
5. **pace_step is computed from the window's start time** (window index × 6 s
   after the block's first band row): 0-99 s → 0, 100-199 s → 1, 200 s+ → 2.
   `transition` = window starts 0-29 s after a pace change (at 100 s or 200 s).
   The 100-s steps were confirmed with the band's step counter: in session 4,
   walking cadence rises at 100 s and at 200 s.
6. **final.py re-runs the frozen tuning procedure.** "Settings frozen from
   development" means the grids, the seed, the Stage 1 classifier
   (`STAGE1_MODEL`) and the headline model (`HEADLINE_MODEL`). The hyperparameters
   themselves are re-chosen by leave-one-session-out inside sessions 1-4, the
   same procedure as in every development fold. Session 5 is never used to choose.
7. **Activity-mean baseline**: the mean is computed per *true* activity in the
   training sessions. At test time it is looked up with the predicted activity
   (primary) or the true activity (oracle row). Ablation (b) subtracts the
   predicted-activity mean from the target, using out-of-fold predictions for the
   training windows.

## Additions

8. **Exploratory slice**: windows within 30 s after a pace change (`transition`).
   It is marked "exploratory" in the slice table.
9. **The proxy check also includes the other model** (ridge), not only the
   headline model, the FFT and the activity mean.
10. **"Did not help" threshold**: an ablation that changes RMSE by less than 0.5 BPM
    is reported as "no meaningful improvement". RMSE varies by several BPM between
    held-out sessions, so smaller changes cannot be told apart from noise.
11. **Observation line**: the numbers file notes that the FFT beats both models on
    resting windows. This comes from the existing results table; no extra model was
    built.
12. **Data is committed to git** (`data/raw`, `data/processed`; about 6 MB), as
    requested. It was previously ignored.
13. **Packages added**: scikit-learn and matplotlib (requirements.txt).

## Choices made on development data (sessions 1-4)

14. **Stage 1 classifier passed to Stage 2: random forest** (macro-F1 0.999), not
    the depth-3 tree (1.000). They differ by one window out of 790. The tree
    separates resting from typing using one wrist-orientation threshold (`ay_mean`),
    which would break if the band were worn slightly rotated. The forest combines
    many features.
15. **Headline model: gradient boosting**, the lower leave-one-session-out RMSE
    (7.93 vs 8.30 BPM for ridge).

## Notes on the slice table

16. The label-noise slice (|hr_truth − hr_truth_rr| > 5 BPM) has only 10
    development windows, below the 30-window minimum, so it cannot be named the
    worst slice.
17. In development, the pace-step slices come from session 4 only (2 blocks each).
    Their bootstrap intervals resample just 2 blocks and are rough.

## Forced reruns of final.py

(final.py appends a line here automatically if it is ever run with --force.)
