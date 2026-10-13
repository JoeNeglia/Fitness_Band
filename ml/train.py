"""
STEP 3 — TRAIN THE MODELS (paper sections 2.4-2.6)

    .venv/bin/python train.py

1. Hold out 30% of the SESSIONS for testing. (The paper split random
   windows, but overlapping windows from the same recording are almost
   identical, which makes test scores look better than they are.)
2. Activity classifier: pick the 9 best features (NCA, as in the paper),
   try 5 model types, keep the most accurate.
3. Heart rate models, one for hand motion and one for walking (paper's
   B.1 and B.2): pick the 15 best features, try 4 model types, keep the
   one with the lowest error.
4. Test the whole pipeline on the held-out sessions vs. plain FFT.

Output: models/hr_pipeline.joblib and reports/ (report.txt + one plot per test session)
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from common import CLASS_NAMES, MODEL_PATH, REPORT_DIR, WINDOWS_CSV, apply_pipeline
from features import feature_columns

# =============================================================================
#  SETTINGS
# =============================================================================
TEST_FRACTION = 0.3
N_CLASSIFIER_FEATURES = 9     # paper
N_REGRESSOR_FEATURES = 15     # paper

# Model types compared (paper Tables 2 and 3). Every model gets a StandardScaler
# in front of it, which is the paper's z-score normalisation (eq. 6).
CLASSIFIERS = {
    "decision tree": DecisionTreeClassifier(random_state=0),
    "KNN": KNeighborsClassifier(n_neighbors=10),
    "SVM": SVC(),
    "bagged trees": BaggingClassifier(DecisionTreeClassifier(), n_estimators=50, random_state=0),
    "naive Bayes": GaussianNB(),
}
REGRESSORS = {
    "linear": LinearRegression(),
    "decision tree": DecisionTreeRegressor(min_samples_leaf=5, random_state=0),
    "SVM": SVR(C=10),
    "bagged trees": BaggingRegressor(DecisionTreeRegressor(min_samples_leaf=3), n_estimators=14, random_state=0),
}


def rmse(pred, true):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def compare_models(candidates, X, y, groups, score):
    """Cross-validates each model, grouped by session. Returns {name: score}."""
    cv = GroupKFold(n_splits=min(5, groups.nunique()))
    results = {}
    for name, model in candidates.items():
        pred = cross_val_predict(make_pipeline(StandardScaler(), clone(model)), X, y, cv=cv, groups=groups)
        results[name] = score(pred, y)
    return results


# =============================================================================
#  STEP 2: ACTIVITY CLASSIFIER (paper 2.4)
# =============================================================================
def train_classifier(train, log):
    data = train.dropna(subset=["label_class"])
    X, y = data[feature_columns(data)], data["label_class"].astype(int)

    # NCA feature selection: each feature's weight = how much NCA relies on it.
    # (NCA is slow on many rows, so it looks at a random 2000.)
    sample = data.sample(min(len(data), 2000), random_state=0)
    nca = NeighborhoodComponentsAnalysis(random_state=0, max_iter=50)
    nca.fit(StandardScaler().fit_transform(sample[X.columns]), sample["label_class"].astype(int))
    weights = pd.Series(np.linalg.norm(nca.components_, axis=0), index=X.columns)
    features = list(weights.nlargest(N_CLASSIFIER_FEATURES).index)
    log(f"Features: {', '.join(features)}")

    scores = compare_models(CLASSIFIERS, X[features], y, data["session"], accuracy_score)
    for name, acc in scores.items():
        log(f"  {name:<14} accuracy {acc:.1%}")
    best = max(scores, key=scores.get)
    log(f"  -> using {best}")
    return make_pipeline(StandardScaler(), clone(CLASSIFIERS[best])).fit(X[features], y), features


# =============================================================================
#  STEP 3: HEART RATE MODELS (paper 2.5), one per motion class
# =============================================================================
def train_regressor(data, log):
    X, y = data[feature_columns(data)], data["label_hr"]

    # Feature selection: mutual information between each feature and heart rate
    # (a simpler stand-in for the paper's ReliefF)
    weights = pd.Series(mutual_info_regression(StandardScaler().fit_transform(X), y, random_state=0),
                        index=X.columns)
    features = list(weights.nlargest(N_REGRESSOR_FEATURES).index)
    log(f"Features: {', '.join(features)}")

    scores = compare_models(REGRESSORS, X[features], y, data["session"], rmse)
    for name, err in scores.items():
        log(f"  {name:<14} RMSE {err:5.2f} BPM")
    log(f"  {'FFT only':<14} RMSE {rmse(data['fft_hr'], y):5.2f} BPM  (for comparison)")
    best = min(scores, key=scores.get)
    log(f"  -> using {best}")
    return make_pipeline(StandardScaler(), clone(REGRESSORS[best])).fit(X[features], y), features


# =============================================================================
#  STEP 4: TEST ON HELD-OUT SESSIONS (paper section 3)
# =============================================================================
def evaluate(models, test, log):
    test = test.join(apply_pipeline(models, test[feature_columns(test)]))

    labelled = test.dropna(subset=["label_class"])
    log(f"Activity accuracy: {accuracy_score(labelled['label_class'], labelled['pred_class']):.1%}")
    log("Confusion matrix (rows = true rest/hand/walk, columns = predicted):")
    log(str(confusion_matrix(labelled["label_class"], labelled["pred_class"], labels=[1, 2, 3])))

    with_hr = test.dropna(subset=["label_hr"])
    log("\nHeart rate RMSE (BPM)     pipeline   FFT only")
    log(f"  all windows             {rmse(with_hr.hr_pred, with_hr.label_hr):7.2f}   "
        f"{rmse(with_hr.fft_hr, with_hr.label_hr):7.2f}")
    for c, name in CLASS_NAMES.items():
        part = with_hr[with_hr["label_class"] == c]
        if len(part):
            log(f"  {name:<22}  {rmse(part.hr_pred, part.label_hr):7.2f}   {rmse(part.fft_hr, part.label_hr):7.2f}")
    plot(test)


def plot(test):
    """One plot per test session, like paper Figures 11-13."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {1: "tab:red", 2: "tab:green", 3: "tab:blue"}
    for session, s in test.groupby("session"):
        t = s["time"] - s["time"].min()
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t, s["fft_hr"], color="0.75", lw=0.8, label="FFT only")
        ax.plot(t, s["label_hr"], color="black", lw=1.5, label="chest strap")
        for c, color in colors.items():
            m = s["pred_class"] == c
            ax.scatter(t[m], s.loc[m, "hr_pred"], s=6, color=color, label=f"pipeline: {CLASS_NAMES[c]}")
        ax.set(xlabel="time (s)", ylabel="heart rate (BPM)", title=session, ylim=(30, 200))
        ax.legend(loc="upper left", fontsize=8, ncol=5)
        fig.tight_layout()
        fig.savefig(REPORT_DIR / f"{session}.png", dpi=120)
        plt.close(fig)


# =============================================================================
#  MAIN
# =============================================================================
def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    df = pd.read_csv(WINDOWS_CSV)
    sessions = df["session"].unique()
    if len(sessions) < 3:
        raise SystemExit("Need at least 3 recorded sessions (2 to train, 1 to test).")

    # ---- Step 1: split by session ----
    rng = np.random.default_rng(0)
    test_sessions = rng.choice(sessions, max(1, round(len(sessions) * TEST_FRACTION)), replace=False)
    train, test = df[~df.session.isin(test_sessions)], df[df.session.isin(test_sessions)]
    log(f"Train: {', '.join(sorted(train.session.unique()))}")
    log(f"Test:  {', '.join(sorted(test_sessions))}")

    # ---- Step 2 + 3: train ----
    log("\n=== Activity classifier ===")
    classifier, classifier_features = train_classifier(train, log)

    regressors, regressor_features = {}, {}
    for c in (2, 3):
        log(f"\n=== Heart rate model: {CLASS_NAMES[c]} ===")
        data = train[(train["label_class"] == c)].dropna(subset=["label_hr"])
        regressors[c], regressor_features[c] = train_regressor(data, log)

    models = {"classifier": classifier, "classifier_features": classifier_features,
              "regressors": regressors, "regressor_features": regressor_features}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, MODEL_PATH)

    # ---- Step 4: test ----
    log("\n=== Test sessions ===")
    evaluate(models, test, log)

    (REPORT_DIR / "report.txt").write_text("\n".join(lines) + "\n")
    print(f"\nSaved model -> {MODEL_PATH}\nSaved report + plots -> {REPORT_DIR}")


if __name__ == "__main__":
    main()
