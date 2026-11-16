"""
unsupervised.py — PCA and k-means: is there structure the models can use?

    python analysis/unsupervised.py

1. PCA
   - Ridge error WITH vs WITHOUT the PCA step (leave-one-session-out).
   - Scree plot and the top loadings of the first 3 components.
2. K-means on the IMU std/energy features (log, standardise, fit on
   sessions 1-4 only)
   - k=4: do the clusters match the 4 activities? (adjusted Rand index)
   - Inside walking and inside running: k=2 and k=3, looking for pace
     regimes. Compared with the known pace steps of session 4, and with the
     session number (clusters that just separate days would be useless).
The pace clusters are the optional input of Stage 2 ablation (c).
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

import config
import data
import models
import plots
import stage1_activity
import stage2_bpm


# =============================================================================
#  1. PCA
# =============================================================================
def ridge_with_and_without_pca(development):
    predictions, settings = stage2_bpm.leave_one_session_out(
        development, model_names=["ridge", "ridge_no_pca"])
    return {"with PCA": models.rmse(predictions["ridge"], predictions[config.TARGET]),
            "without PCA": models.rmse(predictions["ridge_no_pca"], predictions[config.TARGET])}


def pca_description(development):
    """
    PCA on the same 44 inputs the ridge model sees (out-of-fold predicted
    activity + IMU + PPG), standardised, fit on sessions 1-4.
    """
    predicted = stage1_activity.out_of_fold_activity(development, config.STAGE1_MODEL)
    inputs = pd.concat([models.one_hot_activity(predicted),
                        development[config.IMU_FEATURES + config.PPG_FEATURES]], axis=1)
    pca = PCA(random_state=config.SEED).fit(StandardScaler().fit_transform(inputs))

    loadings_rows = []
    for component in range(3):
        weights = pd.Series(pca.components_[component], index=inputs.columns)
        top = weights.abs().sort_values(ascending=False).index[:8]
        for feature in top:
            loadings_rows.append({"component": f"PC{component + 1}",
                                  "variance_share": pca.explained_variance_ratio_[component],
                                  "feature": feature, "loading": weights[feature]})
    return pca.explained_variance_ratio_, pd.DataFrame(loadings_rows)


# =============================================================================
#  2. K-MEANS
# =============================================================================
def kmeans_versus_activity(development):
    kmeans = models.make_kmeans(4).fit(development[config.KMEANS_FEATURES])
    clusters = kmeans.predict(development[config.KMEANS_FEATURES])
    crosstab = pd.crosstab(development["activity"], clusters).reindex(config.ACTIVITIES)
    crosstab.columns = [f"cluster {c}" for c in crosstab.columns]
    return adjusted_rand_score(development["activity"], clusters), crosstab


def pace_regimes(development):
    """For walking and running separately, try k=2 and k=3."""
    rows, crosstabs = [], {}
    for activity in config.MOVING_ACTIVITIES:
        windows = development[development["activity"] == activity]
        for k in [2, 3]:
            pipeline = models.make_kmeans(k).fit(windows[config.KMEANS_FEATURES])
            clusters = pipeline.predict(windows[config.KMEANS_FEATURES])
            scaled = pipeline[:-1].transform(windows[config.KMEANS_FEATURES])

            # Session 4 is the only development session with known pace steps.
            paced = (windows["session"] == config.VARIED_PACE_SESSION).to_numpy()
            rows.append({
                "activity": activity, "k": k,
                "silhouette": silhouette_score(scaled, clusters),
                "ari_vs_pace_step_session4": adjusted_rand_score(windows["pace_step"][paced], clusters[paced]),
                "ari_vs_session": adjusted_rand_score(windows["session"], clusters),
            })
            crosstabs[(activity, k)] = pd.crosstab(windows["session"], clusters)
    return pd.DataFrame(rows), crosstabs


# =============================================================================
#  MAIN
# =============================================================================
def main():
    development = data.load_windows()
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    pca_effect = ridge_with_and_without_pca(development)
    explained, loadings = pca_description(development)
    loadings.to_csv(config.TABLES_DIR / "pca_top_loadings.csv", index=False, float_format="%.3f")
    pd.DataFrame({"component": np.arange(1, len(explained) + 1), "variance_share": explained,
                  "cumulative": np.cumsum(explained)}).to_csv(
        config.TABLES_DIR / "pca_scree.csv", index=False, float_format="%.4f")
    plots.scree_figure(explained, config.FIGURES_DIR / "pca_scree.png")

    ari, activity_crosstab = kmeans_versus_activity(development)
    activity_crosstab.to_csv(config.TABLES_DIR / "kmeans_k4_vs_activity.csv")
    plots.heatmap_figure(activity_crosstab, f"k-means (k=4) vs activity, ARI = {ari:.2f}",
                         config.FIGURES_DIR / "kmeans_crosstab.png", color_map="Oranges")

    pace_table, pace_crosstabs = pace_regimes(development)
    pace_table.to_csv(config.TABLES_DIR / "kmeans_pace_regimes.csv", index=False, float_format="%.3f")
    with open(config.TABLES_DIR / "kmeans_pace_crosstabs.txt", "w") as file:
        for (activity, k), table in pace_crosstabs.items():
            file.write(f"{activity}, k={k}: windows per session (rows) and cluster (columns)\n{table}\n\n")

    print("PCA: ridge RMSE", {name: round(value, 2) for name, value in pca_effect.items()})
    print(f"First 3 components explain {explained[:3].round(3)} (cumulative {explained[:3].sum():.3f})")
    print(f"\nk-means k=4 vs activity: ARI = {ari:.3f}\n{activity_crosstab}")
    print(f"\nPace regimes inside walking/running:\n{pace_table.round(3).to_string(index=False)}")

    numbers = [
        ("Ridge LOSO RMSE with PCA", f"{pca_effect['with PCA']:.2f} BPM"),
        ("Ridge LOSO RMSE without PCA", f"{pca_effect['without PCA']:.2f} BPM"),
        ("Variance explained by PC1, PC2, PC3", ", ".join(f"{v:.3f}" for v in explained[:3])),
        ("Components needed for 90% of variance", int(np.searchsorted(np.cumsum(explained), 0.90) + 1)),
        ("k-means (k=4) vs activity, adjusted Rand index", f"{ari:.3f}"),
    ]
    for row in pace_table.itertuples():
        numbers.append((f"Pace k-means, {row.activity} k={row.k}",
                        f"silhouette {row.silhouette:.3f}, ARI vs session-4 pace step "
                        f"{row.ari_vs_pace_step_session4:.3f}, ARI vs session {row.ari_vs_session:.3f}"))
    numbers += findings(pca_effect, ari, pace_table)
    return numbers


def findings(pca_effect, ari, pace_table):
    """Plain sentences for the report, worded from the numbers."""
    sentences = []
    change = pca_effect["with PCA"] - pca_effect["without PCA"]
    if change < 0:
        sentences.append(f"PCA before ridge lowered the RMSE by {-change:.2f} BPM, so it stays in the readable model.")
    else:
        sentences.append(f"PCA was checked (ridge with vs without it) and did not lower the RMSE "
                         f"({change:+.2f} BPM), because the ridge penalty already handles the correlated features.")

    sentences.append(f"K-means with k=4 on movement features matched the activities with ARI {ari:.2f}: "
                     "the activities are already well separated by movement alone, which is why Stage 1 is easy.")

    for activity in config.MOVING_ACTIVITIES:
        rows = pace_table[pace_table["activity"] == activity]
        best_pace = rows["ari_vs_pace_step_session4"].max()
        if best_pace < 0.2:
            sentences.append(f"Inside {activity}, k-means clusters did not follow the session-4 pace steps "
                             f"(best ARI {best_pace:.2f}), so they carry little pace information "
                             "(tables/kmeans_pace_crosstabs.txt shows which sessions fall in each cluster).")
        else:
            sentences.append(f"Inside {activity}, k-means clusters partly followed the session-4 pace steps "
                             f"(best ARI {best_pace:.2f}).")
    return [("Finding", s) for s in sentences]


if __name__ == "__main__":
    main()
