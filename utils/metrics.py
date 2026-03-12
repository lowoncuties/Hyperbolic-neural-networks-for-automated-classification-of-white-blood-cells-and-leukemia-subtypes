"""Shared metric helpers for white blood cell and leukemia subtype classifiers."""

from __future__ import annotations

from typing import Dict, Any, Iterable

import numpy as np


def compute_sensitivity_specificity_multiclass(cm: np.ndarray):
    """Return aggregate sensitivity/specificity statistics for a confusion matrix."""

    num_classes = cm.shape[0]
    support = cm.sum(axis=1)
    total_samples = support.sum()

    tp = np.zeros(num_classes)
    tn = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    sensitivity = np.zeros(num_classes)
    specificity = np.zeros(num_classes)

    for i in range(num_classes):
        tp[i] = cm[i, i]
        fp[i] = cm[:, i].sum() - tp[i]
        fn[i] = cm[i, :].sum() - tp[i]
        tn[i] = cm.sum() - (tp[i] + fp[i] + fn[i])

        sensitivity[i] = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) != 0 else 0.0
        specificity[i] = tn[i] / (tn[i] + fp[i]) if (tn[i] + fp[i]) != 0 else 0.0

    TP_total = int(np.trace(cm))
    FP_total = int((cm.sum(axis=0) - np.diag(cm)).sum())
    FN_total = int((cm.sum(axis=1) - np.diag(cm)).sum())
    weighted_TN = float(np.sum(tn * support) / total_samples) if total_samples != 0 else 0.0
    weighted_sensitivity = (
        float(np.sum(sensitivity * support) / total_samples) if total_samples != 0 else 0.0
    )
    weighted_specificity = (
        float(np.sum(specificity * support) / total_samples) if total_samples != 0 else 0.0
    )

    return TP_total, FP_total, weighted_TN, FN_total, weighted_sensitivity, weighted_specificity


def f1_macro_from_report(report_dict: Dict[str, Any]) -> float:
    """Extract macro F1 from a sklearn classification_report dict."""

    return float(report_dict.get("macro avg", {}).get("f1-score", 0.0))


def f1_macro_from_preds_labels(
    preds: Iterable[int], labels: Iterable[int], num_classes: int
) -> float:
    """Compute macro-F1 manually from integer predictions and labels."""

    preds = np.array(list(preds), dtype=int)
    labels = np.array(list(labels), dtype=int)
    f1s = []
    for c in range(num_classes):
        tp = np.sum((preds == c) & (labels == c))
        fp = np.sum((preds == c) & (labels != c))
        fn = np.sum((preds != c) & (labels == c))
        denom = 2 * tp + fp + fn
        f1 = 0.0 if denom == 0 else (2 * tp) / denom
        f1s.append(f1)
    return float(sum(f1s) / max(1, len(f1s)))


__all__ = [
    "compute_sensitivity_specificity_multiclass",
    "f1_macro_from_report",
    "f1_macro_from_preds_labels",
]
