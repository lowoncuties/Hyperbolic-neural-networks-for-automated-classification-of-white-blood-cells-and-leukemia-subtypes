"""Visualization and reporting helpers shared between training scripts."""

from __future__ import annotations

from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np


def save_confusion_matrix_figure(
    cm: np.ndarray, classes: List[str], out_path: str, normalize: bool = False
) -> None:
    """Persist a confusion matrix to disk."""

    cm_plot = cm.astype(float)
    if normalize:
        with np.errstate(all="ignore"):
            row_sums = cm_plot.sum(axis=1, keepdims=True)
            cm_plot = np.divide(cm_plot, np.maximum(row_sums, 1e-12))

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm_plot, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm_plot.shape[1]),
        yticks=np.arange(cm_plot.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix" + (" (Normalized)" if normalize else ""),
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            val = cm_plot[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, txt, ha="center", va="center", color="w")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_classification_report_csv_and_image(
    report_dict: Dict[str, Any], classes: List[str], csv_path: str, img_path: str
) -> None:
    """Save both CSV and image versions of a classification report."""

    import csv

    headers = ["class", "precision", "recall", "f1", "support"]
    rows = []
    for i, cls in enumerate(classes):
        stats = report_dict.get(cls, {})
        rows.append(
            [
                cls,
                f"{float(stats.get('precision', 0.0)):.4f}",
                f"{float(stats.get('recall', 0.0)):.4f}",
                f"{float(stats.get('f1-score', 0.0)):.4f}",
                int(stats.get("support", 0)),
            ]
        )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    table = ax.table(
        cellText=[[v for v in row] for row in rows],
        colLabels=headers,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    ax.set_title("Classification Report")
    fig.tight_layout()
    fig.savefig(img_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "save_confusion_matrix_figure",
    "save_classification_report_csv_and_image",
]
