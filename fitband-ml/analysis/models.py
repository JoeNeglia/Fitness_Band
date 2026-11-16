"""
models.py — the building blocks shared by every modelling script.

  * the activity classifiers (Stage 1) and heart-rate models (Stage 2), all
    as scikit-learn Pipelines
  * leave-one-session-out splitting and the small tuning loop
  * predicted activity for a train/test split (what Stage 2 receives)
  * the k-means "pace cluster" feature used by ablation (c)
  * the RMSE metric

Keeping these in one file guarantees that development (run_all.py), the side
experiment and the final test all build their models in exactly the same way.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

import config


# =============================================================================
#  METRIC
# =============================================================================
def rmse(predicted, truth):
    """Root mean squared error in BPM. Windows without a prediction are skipped."""
    predicted = np.asarray(predicted, dtype=float)
    truth = np.asarray(truth, dtype=float)
    usable = ~np.isnan(predicted) & ~np.isnan(truth)
    if not usable.any():
        return np.nan
    return float(np.sqrt(np.mean((predicted[usable] - truth[usable]) ** 2)))


# =============================================================================
#  LEAVE-ONE-SESSION-OUT
#  Each fold holds out one whole session. This mimics the real question
#  "how well does the model work on a day it has never seen?"
# =============================================================================
def leave_one_session_out(sessions):
    sessions = np.asarray(sessions)
    return list(LeaveOneGroupOut().split(np.zeros(len(sessions)), groups=sessions))


# =============================================================================
#  STAGE 1: ACTIVITY CLASSIFIERS (inputs: IMU features only)
# =============================================================================
def make_activity_model(name):
    if name == "majority":
        # Always predicts the most common activity: the floor any model must beat.
        return Pipeline([("model", DummyClassifier(strategy="most_frequent"))])
    if name == "acc_rule":
        # One feature (how much the wrist moves) and at most 2 questions:
        # shows how far a single obvious rule gets.
        return Pipeline([("model", DecisionTreeClassifier(max_depth=config.STAGE1_RULE_DEPTH,
                                                          random_state=config.SEED))])
    if name == "tree_depth3":
        # A depth-3 tree is at most 3 yes/no questions: small enough to draw
        # and explain, yet able to use several features.
        return Pipeline([("model", DecisionTreeClassifier(max_depth=config.STAGE1_TREE_DEPTH,
                                                          random_state=config.SEED))])
    if name == "random_forest":
        # 200 trees, each on a random resample of the data, vote together.
        # More flexible than one tree and hard to overfit badly.
        return Pipeline([("model", RandomForestClassifier(n_estimators=config.STAGE1_FOREST_TREES,
                                                          random_state=config.SEED, n_jobs=-1))])
    raise ValueError(f"unknown activity model {name}")


def activity_inputs(name):
    """The one-feature rule sees acc_mag_std only; every other classifier sees all IMU features."""
    return ["acc_mag_std"] if name == "acc_rule" else config.IMU_FEATURES


def fit_predict_activity(name, train, test):
    columns = activity_inputs(name)
    model = make_activity_model(name)
    model.fit(train[columns], train["activity"])
    return model.predict(test[columns])


def predicted_activity_for_split(train, test, name=config.STAGE1_MODEL):
    """
    Predicted activity for every training and test window of one split.

    Training windows get OUT-OF-FOLD predictions: each training session is
    predicted by a classifier trained on the other training sessions. That
    way Stage 2 trains on the same kind of classifier mistakes it will meet
    at test time. Test windows are predicted by a classifier trained on all
    training sessions. The test sessions are never used to fit anything here.
    """
    train_predicted = pd.Series(index=train.index, dtype=object)
    for fit_rows, held_rows in leave_one_session_out(train["session"]):
        fit_part, held_part = train.iloc[fit_rows], train.iloc[held_rows]
        train_predicted.iloc[held_rows] = fit_predict_activity(name, fit_part, held_part)
    test_predicted = pd.Series(fit_predict_activity(name, train, test), index=test.index)
    return train_predicted, test_predicted


def one_hot_activity(predicted):
    """4 columns (pred_resting, ...), 1 for the predicted activity and 0 otherwise."""
    return pd.DataFrame({f"pred_{a}": (predicted == a).astype(float) for a in config.ACTIVITIES},
                        index=predicted.index)


# =============================================================================
#  STAGE 2: HEART-RATE MODELS
# =============================================================================
def ridge_alpha_cv(sessions):
    """
    RidgeCV chooses its penalty (alpha) by cross-validation. By default it
    leaves out one WINDOW at a time, which is the random-window split we
    avoid. So we hand it session-based folds instead.
    """
    if len(set(sessions)) < 2:
        return None
    return leave_one_session_out(sessions)


def make_ridge(pca_components, sessions):
    # 1. StandardScaler: puts every feature on the same scale (PCA and the
    #    ridge penalty are both sensitive to units, e.g. mg vs deg/s).
    # 2. PCA: replaces 44 correlated features with a few uncorrelated
    #    directions of largest variation. Many features say the same thing
    #    (ax_std, acc_mag_std, ...), and PCA removes that repetition.
    # 3. Ridge: a straight-line model whose coefficients are shrunk towards 0,
    #    which keeps it stable with few sessions. Each coefficient is readable.
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=pca_components, random_state=config.SEED)),
        ("ridge", RidgeCV(alphas=config.RIDGE_ALPHAS, cv=ridge_alpha_cv(sessions))),
    ])


def make_ridge_without_pca(sessions):
    """Same as make_ridge but with PCA removed, to measure what PCA adds."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("ridge", RidgeCV(alphas=config.RIDGE_ALPHAS, cv=ridge_alpha_cv(sessions))),
    ])


def make_gbm(max_depth, n_estimators):
    # Gradient boosting adds many small trees, each fixing the errors of the
    # ones before. It can capture curved relations and interactions (e.g.
    # "PPG means something different while running"). Shallow trees and a
    # small learning rate keep it from memorising the training sessions.
    return Pipeline([
        ("gbm", GradientBoostingRegressor(max_depth=max_depth, n_estimators=n_estimators,
                                          learning_rate=config.GBM_LEARNING_RATE,
                                          random_state=config.SEED)),
    ])


def model_grid(model_name):
    """The candidate settings for one model, plus a function that builds it."""
    if model_name == "ridge":
        grid = [{"pca_components": k} for k in config.PCA_COMPONENTS_GRID]
        return grid, lambda params, sessions: make_ridge(params["pca_components"], sessions)
    if model_name == "ridge_no_pca":
        return [{}], lambda params, sessions: make_ridge_without_pca(sessions)
    if model_name == "gbm":
        grid = [{"max_depth": d, "n_estimators": n}
                for d in config.GBM_MAX_DEPTH_GRID for n in config.GBM_N_ESTIMATORS_GRID]
        return grid, lambda params, sessions: make_gbm(params["max_depth"], params["n_estimators"])
    raise ValueError(f"unknown heart-rate model {model_name}")


def tune_and_fit(model_name, X, y, sessions):
    """
    Pick the best setting with leave-one-session-out INSIDE the training
    sessions only, then refit that setting on all training sessions.
    The test session is never seen while choosing.
    """
    grid, build = model_grid(model_name)
    # X stays a DataFrame so the fitted model remembers the feature names.
    X = X.reset_index(drop=True)
    y, sessions = np.asarray(y, dtype=float), np.asarray(sessions)

    best_params, best_error = grid[0], np.inf
    if len(grid) > 1:
        for params in grid:
            inner_predictions = np.empty(len(y))
            for fit_rows, held_rows in leave_one_session_out(sessions):
                model = build(params, sessions[fit_rows])
                model.fit(X.iloc[fit_rows], y[fit_rows])
                inner_predictions[held_rows] = model.predict(X.iloc[held_rows])
            error = rmse(inner_predictions, y)
            if error < best_error:
                best_params, best_error = params, error

    final_model = build(best_params, sessions)
    final_model.fit(X, y)
    return final_model, best_params


# =============================================================================
#  K-MEANS PACE CLUSTERS (ablation c)
#  Inside walking and inside running, k-means groups windows by how hard the
#  wrist moves. The clusters are numbered 1..k from gentlest to hardest
#  (0 = not walking/running), so the number means "pace level".
# =============================================================================
def make_kmeans(k):
    # log1p: std and energy are skewed (a few very large values), the log
    # evens them out. Scaling stops one feature dominating the distances.
    return Pipeline([
        ("log", FunctionTransformer(np.log1p)),
        ("scale", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=config.SEED)),
    ])


def pace_cluster_feature(train, train_activity, test, test_activity, k=config.PACE_K):
    """
    Fit one k-means per moving activity on the training windows (true labels),
    then number every window that is PREDICTED to be that activity.
    Returns (train_feature, test_feature) as Series.
    """
    train_feature = pd.Series(0.0, index=train.index)
    test_feature = pd.Series(0.0, index=test.index)
    for activity in config.MOVING_ACTIVITIES:
        fit_rows = train[train["activity"] == activity]
        kmeans = make_kmeans(k).fit(fit_rows[config.KMEANS_FEATURES])

        # Order the clusters by average acc_mag_std so 1 = gentlest movement.
        labels = kmeans.predict(fit_rows[config.KMEANS_FEATURES])
        mean_motion = fit_rows.groupby(labels)["acc_mag_std"].mean().sort_values()
        level_of_cluster = {cluster: level + 1 for level, cluster in enumerate(mean_motion.index)}

        for rows, predicted, feature in [(train, train_activity, train_feature),
                                         (test, test_activity, test_feature)]:
            chosen = predicted == activity
            if chosen.any():
                clusters = kmeans.predict(rows.loc[chosen, config.KMEANS_FEATURES])
                feature[chosen] = [level_of_cluster[c] for c in clusters]
    return train_feature, test_feature
