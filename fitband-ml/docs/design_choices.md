# Design choices

One section per choice: **what** was chosen, **why**, and **what would change
my mind**. Numbers are development results (leave-one-session-out on sessions
1-4) from `results/numbers_for_report.md`.

## Split: by session, leave-one-session-out, session 5 held back

**What.** Sessions 1-4 are development data. Every development number comes from
holding out one whole session, training on the other three, and repeating for
each session. Session 5 is tested once, by `final.py`, after every choice is
frozen.

**Why.** Windows next to each other are almost copies: the same heart rate, the
same band position, the same sweat. A random window split would put near-twins on
both sides and measure memory, not generalisation. The real question is "does it
work on a day it has never seen?", and holding out a session asks exactly that.
Session 5 is touched once so that no choice can be tuned to it, even
unconsciously.

**What would change my mind.** With many subjects I would hold out whole
*people*, since that is the harder and more useful question. With only one or two
sessions, a held-out final test would leave too little to train on, and I would
report leave-one-session-out only.

## Metrics: RMSE (heart rate) and macro-F1 (activity)

**What.** RMSE in BPM for heart rate; macro-F1 for the activity classifier.

**Why.** RMSE is in the same units as heart rate, and it punishes large misses
more than small ones. A 40-BPM miss (the FFT locking onto arm swing) is far worse
than four 10-BPM misses, and RMSE says so. Macro-F1 averages the F1 score of each
activity, so every activity counts equally. Plain accuracy would hide a classifier
that fails on one activity.

**What would change my mind.** If the use case were "alert when heart rate is too
high", I'd use error at high heart rates or a detection metric. If the activity
classes were very unbalanced, I would also show per-class recall.

## Baselines

**What.** (1) FFT on every window. (2) The mean heart rate of the predicted
activity, plus an oracle row using the true activity. (3) The band's own
`device_bpm`, with its coverage.

**Why.** The FFT is the standard signal-processing answer and the baseline named
in the proposal. The activity mean is the "dumb but strong" baseline: if a model
can't beat "walking means about 80 BPM", it hasn't learned heart rate, only
activity. The oracle row shows how much is lost through Stage 1 mistakes (here
almost nothing: 8.21 vs 8.19). `device_bpm` shows what the band does on its own
(75 BPM RMSE, 90% coverage).

**What would change my mind.** If a published method for motion-robust PPG were
available on this hardware, it would be a more demanding baseline than the plain
FFT.

## Stage 1 models: majority, one-feature rule, depth-3 tree, random forest

**What.** A ladder from trivial to flexible, using IMU features only. The random
forest feeds Stage 2.

**Why.** The ladder shows where the gain comes from. Majority (macro-F1 0.16) is
the floor. The rule on acc_mag_std alone (0.67) separates "moving" from "still"
but not resting from typing. The depth-3 tree (1.00) is readable: 3 questions you
can draw. The forest (0.999) is the flexible model. I passed the forest on even
though the tree scored one window higher. The tree tells resting from typing with
a single wrist-orientation threshold (`ay_mean`), which would break if the band
sat slightly rotated on another day. The forest spreads its decision over many
features.

**What would change my mind.** If the forest made clearly more mistakes than the
tree on held-out sessions, or if Stage 1 had to run on the band itself, where the
tree's 3 comparisons are far cheaper.

## Stage 2 models: scaler → PCA → ridge, and gradient boosting

**What.** Exactly two models, with the same inputs (one-hot predicted activity, 35
IMU features, 5 PPG features), splits and metric.

**Why.** Ridge is the readable model: a straight-line formula whose coefficients
are shrunk towards zero, which keeps it stable when there are few sessions.
Gradient boosting is the flexible model: many shallow trees that can capture
curves and interactions, such as "the PPG means something different while
running". Comparing them answers whether flexibility pays off. Here it pays off a
little: 7.93 (GBM) vs 8.30 (ridge). But neither clearly beats the activity mean
(8.21), and both are worse than the FFT at rest (FFT 1.96, GBM 4.67).

**What would change my mind.** If a model failed to beat the activity mean on
session 5 as well, the honest conclusion is that these window features don't carry
usable heart-rate information during motion. I would then move to richer PPG
features, such as spectral peaks after removing the arm-swing frequency.

## FFT at rest (the headline design)

**What.** For windows that Stage 1 predicts as resting, the heart rate is the FFT
estimate. For every other window, it is the model's prediction. The headline is
GBM with this rule (`gbm_fft_rest`). Ridge with the rule and both plain models are
reported beside it.

**Why.** When the wrist is still, the PPG pulse is clean and the FFT finds it
directly: 1.96 BPM error at rest, while the best model managed 4.67. When the
wrist moves, the FFT locks onto the arm swing (46 BPM error while running), so the
model takes over. Stage 1 is reliable enough to switch on: all 198 resting windows
were predicted as resting, and no other window was called resting. The rule has
nothing to fit, it was the plan in the original proposal, and it helped or tied in
every held-out session (overall 7.93 → 7.64). It was added after looking at
development results, which is allowed. It was frozen before session 5 was ever
used, and it is listed in DECLARED_CHANGES.md.

**What would change my mind.** If Stage 1 started calling moving windows "resting"
(e.g. a slow walk with a still arm), the FFT would be used where it fails. I would
then switch on both "predicted resting" and low acc_mag_std, or drop the rule.

## Tuning grids: tiny, chosen inside the training sessions

**What.** Ridge: PCA components in {5, 8, 12}, with alpha chosen by RidgeCV. GBM:
max_depth in {2, 3} × n_estimators in {100, 200}, learning rate 0.05. Settings are
chosen by leave-one-session-out *inside* the training sessions of each fold.

**Why.** With 3 training sessions per fold, each candidate is judged on very
little data. The more candidates you try, the more likely the winner just
happened to fit those sessions' noise. A tiny grid keeps that risk small. Choosing
inside the training sessions keeps the held-out session out of every decision.
RidgeCV is given session-based folds, because its default leaves out single
windows.

**What would change my mind.** If the chosen settings always sat at the edge of a
grid, I would extend the grid in that direction. Ridge picked the middle value
(8 components) in 3 of 4 folds. GBM picked depth 2 with 200 trees, an edge of its
grid, in 3 of 4 folds. That hints that more, shallower trees might help a little.
I did not widen the grid, because it was fixed in advance. `results/tables/stage2_chosen_settings.csv`
lists every fold's choice.

## PCA (inside the readable model)

**What.** Standardise, then keep the first 5-12 principal components before ridge.

**Why.** Many features say the same thing: ax_std, ay_std, acc_mag_std and the
energy features all measure how hard the wrist moves. PC1 alone explains 40% of
the variance and loads on exactly those. Ridge on 44 correlated features was
unstable (11.41 BPM without PCA). With PCA it was 8.30.

**What would change my mind.** If the information needed for heart rate sat in a
low-variance direction (a small PPG feature, say), PCA would throw it away. A
large error drop without PCA would show that.

## K-means (structure check and ablation c)

**What.** K-means on log-transformed, standardised IMU std/energy features,
fitted on training sessions only. k = 4 is compared with the activities. Inside
walking and running, k = 2 and 3 look for pace regimes.

**Why.** It checks whether the data has natural groups the models could use
without labels. k = 4 matched the activities almost perfectly (ARI 0.95), which
is why Stage 1 is easy. Within walking, clusters partly followed session 4's
pace steps (ARI up to 0.62). Within running they did not (ARI 0.01). As an input
(ablation c), the pace level changed RMSE by less than 0.5 BPM, so it is not in
the primary model.

**What would change my mind.** If more sessions with varied pace showed the
clusters reliably following pace on held-out days, the pace level would be worth
adding.

## Failure-analysis slices and the block bootstrap

**What.** Pre-specified slices: heart rate ≥ 100, session, pace step,
acc_mag_std quartile, ppg_mean quartile, and the label-noise flag. Each slice gets
its window count, block count, RMSE, and a 95% interval from resampling whole
(session, activity) blocks. The worst slice is named only among pre-specified
slices with at least 30 windows. Anything added later is marked exploratory.

**Why.** Pre-specifying slices stops me from hunting through dozens of cuts and
reporting the scariest one. Block resampling is honest about the real amount of
independent data: 16 blocks in development, not 790 independent windows. The
proxy check asks whether predictions follow heart rate *within* a block, which
separates "tracks heart rate" from "just recognises the activity".

**What would change my mind.** With many more blocks, I would add finer slices
(e.g. heart rate in 10-BPM bins), and their intervals would be narrow enough to
mean something.
