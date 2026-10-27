"""Prediction helpers producing confidence and explicitly non-conclusive triage labels.

Risk is not proof. An attack-risk label is computed only when the caller passes
``positive_label`` or the model's class names unambiguously resemble common
attack/anomaly labels; otherwise the output says ``unscored`` rather than
assuming class semantics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ATTACK_WORDS = ("attack", "malicious", "intrusion", "anomaly", "botnet", "dos", "ddos", "exploit")


def _risk_label(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


def predict_records(pipeline, records, *, positive_label=None, risk_thresholds=(0.5, 0.8)) -> pd.DataFrame:
    """Predict rows and return input columns plus prediction/confidence/risk fields.

    ``positive_label`` explicitly identifies the class interpreted as attack.
    If omitted, a positive class is inferred only from recognizable class-name
    tokens; no such assumption is made about feature column names.
    """
    if isinstance(records, pd.Series):
        frame = records.to_frame().T
    elif isinstance(records, dict):
        frame = pd.DataFrame([records])
    elif isinstance(records, pd.DataFrame):
        frame = records.copy()
    else:
        raise TypeError("records must be a DataFrame, Series, or mapping")
    if frame.empty:
        raise ValueError("At least one record is required.")
    preds = np.asarray(pipeline.predict(frame))
    result = frame.reset_index(drop=True).copy()
    result["prediction"] = preds
    proba = None
    classes = list(getattr(pipeline, "classes_", []))
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = np.asarray(pipeline.predict_proba(frame), dtype=float)
        except (ValueError, AttributeError, NotImplementedError):
            proba = None
    if proba is not None and proba.ndim == 2 and proba.shape[0] == len(frame):
        conf = np.max(proba, axis=1)
    else:
        conf = np.full(len(frame), np.nan)
    result["confidence"] = conf
    if positive_label is None and len(classes) == 2:
        candidates = [c for c in classes if any(w in str(c).lower() for w in ATTACK_WORDS)]
        if len(candidates) == 1:
            positive_label = candidates[0]
    if positive_label is not None and proba is not None and positive_label in classes:
        ix = classes.index(positive_label)
        score = proba[:, ix]
        low, high = map(float, risk_thresholds)
        if not (0 <= low <= high <= 1):
            raise ValueError("risk_thresholds must satisfy 0 <= low <= high <= 1")
        result["risk_score"] = score
        result["risk_label"] = ["high" if s >= high else "medium" if s >= low else "low" for s in score]
    else:
        result["risk_score"] = np.nan
        result["risk_label"] = "unscored"
    result["risk_note"] = "Triage estimate only; a model output is not proof of intrusion."
    return result


predict = predict_records
