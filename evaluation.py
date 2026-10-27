"""Classification metrics for binary and multiclass held-out predictions.

ROC-AUC is emitted only when valid class probabilities/scores are provided;
multiclass AUC uses one-vs-rest. Precision-recall average precision is emitted
only for binary tasks with probability scores. Results use ordinary Python
numbers/lists/dicts so they can be exported as JSON.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             average_precision_score)


def _py(value):
    if isinstance(value, dict):
        return {str(k): _py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_py(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def evaluate_classification(y_true, y_pred, probabilities=None, labels=None, positive_label=None) -> dict:
    """Return metrics and diagnostics; binary metrics use explicit/inferred positive class.

    If ``positive_label`` is omitted, a uniquely recognizable attack/anomaly
    label is preferred; otherwise the second value in ``labels`` is used.
    """
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    if yt.ndim != 1 or yp.ndim != 1 or len(yt) != len(yp) or len(yt) == 0:
        raise ValueError("y_true and y_pred must be non-empty one-dimensional arrays of equal length.")
    classes = list(labels) if labels is not None else list(dict.fromkeys(yt.tolist() + yp.tolist()))
    if not classes:
        raise ValueError("At least one class is required.")
    metrics = {
        "accuracy": accuracy_score(yt, yp),
        "precision_macro": precision_score(yt, yp, labels=classes, average="macro", zero_division=0),
        "recall_macro": recall_score(yt, yp, labels=classes, average="macro", zero_division=0),
        "f1_macro": f1_score(yt, yp, labels=classes, average="macro", zero_division=0),
        "precision_weighted": precision_score(yt, yp, average="weighted", zero_division=0),
        "recall_weighted": recall_score(yt, yp, average="weighted", zero_division=0),
        "f1_weighted": f1_score(yt, yp, average="weighted", zero_division=0),
    }
    if len(classes) == 2:
        attack_tokens = ("attack", "malicious", "intrusion", "anomaly", "botnet", "dos", "ddos", "exploit")
        inferred = [label for label in classes if any(token in str(label).lower() for token in attack_tokens)]
        positive_class = positive_label if positive_label is not None else (inferred[0] if len(inferred) == 1 else classes[1])
        if positive_class not in classes:
            raise ValueError(f"positive_label {positive_class!r} is not one of the observed classes {classes!r}.")
        true_binary = (yt == positive_class).astype(int)
        pred_binary = (yp == positive_class).astype(int)
        tn, fp, fn, tp = confusion_matrix(true_binary, pred_binary, labels=[0, 1]).ravel()
        metrics["positive_class"] = positive_class
        metrics["false_positive_rate"] = float(fp / (fp + tn)) if fp + tn else None
    proba = None if probabilities is None else np.asarray(probabilities, dtype=float)
    if proba is not None and proba.ndim == 1 and len(classes) == 2:
        # A one-dimensional score means confidence for the second class.
        proba = np.column_stack((1.0 - proba, proba))
    if proba is not None and len(classes) == 2 and proba.ndim == 2 and proba.shape == (len(yt), 2):
        try:
            positive_index = classes.index(metrics["positive_class"])
            binary_true = (yt == metrics["positive_class"]).astype(int)
            metrics["roc_auc"] = roc_auc_score(binary_true, proba[:, positive_index])
            metrics["average_precision"] = average_precision_score(binary_true, proba[:, positive_index])
        except (ValueError, TypeError):
            pass
    elif proba is not None and len(classes) > 2 and proba.ndim == 2 and proba.shape == (len(yt), len(classes)):
        try:
            metrics["roc_auc_ovr_macro"] = roc_auc_score(yt, proba, labels=classes,
                                                          multi_class="ovr", average="macro")
        except (ValueError, TypeError):
            pass
    report = classification_report(yt, yp, labels=classes, output_dict=True, zero_division=0)
    matrix = confusion_matrix(yt, yp, labels=classes)
    return {"metrics": _py(metrics), "confusion_matrix": matrix.astype(int).tolist(),
            "labels": _py(classes), "classification_report": _py(report)}


# Conventional shorter alias for app code.
evaluate = evaluate_classification
