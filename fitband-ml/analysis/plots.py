"""
plots.py — the figures written to results/figures/.

Each function takes finished numbers (a table or predictions) and saves one
PNG. No modelling happens here.
"""

import matplotlib

matplotlib.use("Agg")           # write files only, never open a window
import matplotlib.pyplot as plt
import numpy as np

import config

ACTIVITY_COLORS = {"resting": "#1f77b4", "typing": "#2ca02c", "walking": "#ff7f0e", "running": "#d62728"}


def save(figure, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def heatmap_figure(table, title, path, color_map="Blues"):
    """A table of counts drawn as coloured squares with the number in each."""
    figure, axis = plt.subplots(figsize=(6, 4.5))
    axis.imshow(table.to_numpy(), cmap=color_map)
    axis.set_xticks(range(table.shape[1]), table.columns, rotation=30, ha="right")
    axis.set_yticks(range(table.shape[0]), table.index)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            axis.text(j, i, int(table.iat[i, j]), ha="center", va="center")
    axis.set_title(title)
    save(figure, path)


def confusion_matrix_figure(confusion, title, path):
    heatmap_figure(confusion, title, path)


def predicted_vs_true_figure(predictions, model_column, title, path):
    figure, axis = plt.subplots(figsize=(5.5, 5.5))
    for activity, color in ACTIVITY_COLORS.items():
        rows = predictions[predictions["activity"] == activity]
        axis.scatter(rows[config.TARGET], rows[model_column], s=10, alpha=0.6, color=color, label=activity)
    low, high = 40, 160
    axis.plot([low, high], [low, high], color="gray", linestyle="--", linewidth=1, label="perfect")
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_xlabel("True heart rate, chest strap (BPM)")
    axis.set_ylabel("Predicted heart rate (BPM)")
    axis.set_title(title)
    axis.legend()
    save(figure, path)


def slice_rmse_figure(slices, title, path):
    """Horizontal bars of RMSE per slice with the 95% block-bootstrap interval."""
    labels = [f"{row.variable}: {row.level}" for row in slices.itertuples()]
    rmse = slices["rmse"].to_numpy()
    error_low = rmse - slices["ci_low"].to_numpy()
    error_high = slices["ci_high"].to_numpy() - rmse
    figure, axis = plt.subplots(figsize=(7, 0.32 * len(slices) + 1.2))
    positions = np.arange(len(slices))
    axis.barh(positions, rmse, xerr=[error_low, error_high], color="#7aa6d6", capsize=2)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("RMSE (BPM) with 95% block-bootstrap interval")
    axis.set_title(title)
    save(figure, path)


def scree_figure(explained_ratio, path):
    figure, axis = plt.subplots(figsize=(6, 4))
    components = np.arange(1, len(explained_ratio) + 1)
    axis.bar(components, explained_ratio, color="#7aa6d6", label="each component")
    axis.plot(components, np.cumsum(explained_ratio), color="black", marker="o", markersize=3, label="cumulative")
    axis.set_xlabel("Principal component")
    axis.set_ylabel("Share of variance explained")
    axis.set_title("PCA scree plot (standardised features, sessions 1-4)")
    axis.legend()
    save(figure, path)
